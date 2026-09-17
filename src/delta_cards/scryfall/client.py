"""Scryfall REST API client for Magic: The Gathering card data, pricing, and Cardmarket mapping."""

import json
import logging
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCRYFALL_BASE_URL = "https://api.scryfall.com"
DEFAULT_USER_AGENT = "delta-cards/1.0 (https://github.com/Javier-Jimenez99/delta-cards)"
MIN_REQUEST_INTERVAL_SECONDS = 0.08  # ~80ms (Scryfall requests 50-100ms)
_last_request_time = 0.0


def _rate_limit() -> None:
    """Enforces Scryfall's recommended rate limiting (50-100ms between requests)."""
    global _last_request_time
    now = time.perf_counter()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _last_request_time = time.perf_counter()


def build_magic_cardmarket_url(id_product: Optional[Any], language: str = "es") -> Optional[str]:
    """Generates direct Cardmarket product URL for Magic from product ID."""
    if not id_product:
        return None
    return f"https://www.cardmarket.com/{language}/Magic/Products?idProduct={id_product}"


def extract_scryfall_cardmarket_prices(card_data: Dict[str, Any], language: str = "es") -> Dict[str, Any]:
    """Extracts Cardmarket product ID, direct URL, and EUR pricing from Scryfall card data.

    Returns:
        Dict with keys: idProduct, url, prices (low, avg, trend, eur, eur_foil),
        name, set, set_name, collector_number, image_url, raw.
    """
    if not card_data or not isinstance(card_data, dict):
        return {
            "idProduct": None,
            "url": None,
            "prices": {"low": None, "avg": None, "trend": None, "eur": None, "eur_foil": None},
            "name": None,
            "set": None,
            "set_name": None,
            "collector_number": None,
            "image_url": None,
            "raw": {},
        }

    cm_id = card_data.get("cardmarket_id")
    purchase_uris = card_data.get("purchase_uris") or {}
    cm_url = purchase_uris.get("cardmarket")
    if not cm_url and cm_id:
        cm_url = build_magic_cardmarket_url(cm_id, language=language)

    prices_dict = card_data.get("prices") or {}
    eur_str = prices_dict.get("eur")
    eur_foil_str = prices_dict.get("eur_foil")

    eur_val: Optional[float] = None
    eur_foil_val: Optional[float] = None

    if eur_str is not None:
        try:
            eur_val = float(eur_str)
        except (ValueError, TypeError):
            eur_val = None

    if eur_foil_str is not None:
        try:
            eur_foil_val = float(eur_foil_str)
        except (ValueError, TypeError):
            eur_foil_val = None

    # Reference market price: primary regular EUR price, fallback to foil EUR price
    ref_price = eur_val if eur_val is not None else eur_foil_val

    # Images
    image_uris = card_data.get("image_uris") or {}
    image_url = (
        image_uris.get("normal")
        or image_uris.get("large")
        or image_uris.get("png")
        or image_uris.get("small")
    )
    if not image_url and isinstance(card_data.get("card_faces"), list) and card_data["card_faces"]:
        first_face = card_data["card_faces"][0]
        if isinstance(first_face, dict):
            face_uris = first_face.get("image_uris") or {}
            image_url = face_uris.get("normal") or face_uris.get("large")

    return {
        "idProduct": str(cm_id) if cm_id is not None else None,
        "url": cm_url,
        "prices": {
            "low": ref_price,
            "avg": ref_price,
            "trend": ref_price,
            "eur": eur_val,
            "eur_foil": eur_foil_val,
        },
        "name": card_data.get("name"),
        "set": card_data.get("set"),
        "set_name": card_data.get("set_name"),
        "collector_number": card_data.get("collector_number"),
        "image_url": image_url,
        "raw": card_data,
    }


class ScryfallClient:
    """Synchronous and asynchronous client for the Scryfall API."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 6.0):
        self.user_agent = user_agent
        self.timeout = timeout

    def _get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Makes an HTTP GET request to Scryfall with rate limiting and error handling."""
        _rate_limit()
        url = f"{SCRYFALL_BASE_URL}/{endpoint.lstrip('/')}"
        if params:
            query_str = urllib.parse.urlencode(params)
            url = f"{url}?{query_str}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
                logger.warning(f"Scryfall returned status {resp.status} for {url}")
                return None
        except urllib.error.HTTPError as err:
            if err.code == 404:
                logger.debug(f"Scryfall 404 Not Found for {url}")
            else:
                logger.warning(f"Scryfall HTTPError {err.code} for {url}: {err}")
            return None
        except Exception as exc:
            logger.debug(f"Scryfall request exception for {url}: {exc}")
            return None

    def get_card_by_id(self, scryfall_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single card by its Scryfall UUID."""
        if not scryfall_id:
            return None
        return self._get(f"cards/{scryfall_id}")

    def get_card_by_set_number(
        self, set_code: str, collector_number: str, lang: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single card by set abbreviation and collector number."""
        if not set_code or not collector_number:
            return None
        clean_set = set_code.lower().strip()
        clean_num = collector_number.lower().strip()
        if lang:
            res = self._get(f"cards/{clean_set}/{clean_num}/{lang}")
            if res:
                return res
        return self._get(f"cards/{clean_set}/{clean_num}")

    def get_card_by_name(
        self, name: str, set_code: Optional[str] = None, exact: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single card by card name using exact or fuzzy search."""
        if not name:
            return None
        clean_name = name.strip()
        params: Dict[str, str] = {}
        if exact:
            params["exact"] = clean_name
        else:
            params["fuzzy"] = clean_name
        if set_code:
            params["set"] = set_code.lower().strip()

        return self._get("cards/named", params=params)

    def search_cards(self, query: str, order: str = "released") -> List[Dict[str, Any]]:
        """Searches Scryfall cards matching the given syntax query."""
        if not query:
            return []
        res = self._get("cards/search", params={"q": query, "order": order})
        if res and isinstance(res.get("data"), list):
            return res["data"]
        return []


_default_client = ScryfallClient()


@lru_cache(maxsize=2048)
def fetch_scryfall_card_pricing(
    card_identifier: str,
    set_code: Optional[str] = None,
    collector_number: Optional[str] = None,
    language: str = "es",
) -> Dict[str, Any]:
    """Fetches Cardmarket pricing and product details from Scryfall for a given card.

    Args:
        card_identifier: Scryfall UUID or card name.
        set_code: Optional set code (e.g. 'mh3', 'ltr').
        collector_number: Optional collector number (e.g. '1', '250').
        language: Language code for URL building.

    Returns:
        Structured dictionary with idProduct, url, prices, name, set, collector_number.
    """
    card_data: Optional[Dict[str, Any]] = None

    # 1. If set_code and collector_number are given, search by set + number
    if set_code and collector_number:
        card_data = _default_client.get_card_by_set_number(
            set_code=set_code, collector_number=collector_number, lang=language
        )

    # 2. Check if card_identifier looks like a UUID (36 chars with dashes)
    if not card_data and len(card_identifier) == 36 and card_identifier.count("-") == 4:
        card_data = _default_client.get_card_by_id(card_identifier)

    # 3. Fallback: Search by card name (fuzzy search)
    if not card_data and card_identifier:
        card_data = _default_client.get_card_by_name(card_identifier, set_code=set_code)

    if not card_data:
        return {}

    return extract_scryfall_cardmarket_prices(card_data, language=language)
