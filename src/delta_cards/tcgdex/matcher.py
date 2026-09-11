"""TCGDex matching engine for assigning Wallapop card offers to TCGDex & Cardmarket IDs."""

import json
import logging
import re
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz
from tcgdexsdk import Query, TCGdex

from .parser import ParsedCardInfo, parse_card_title

logger = logging.getLogger(__name__)

# Use a sane socket timeout for external requests if not already configured
if socket.getdefaulttimeout() is None:
    socket.setdefaulttimeout(10.0)




def build_cardmarket_url(
    id_product: Optional[Any],
    language: str = "es",
) -> Optional[str]:
    """Builds the direct Cardmarket redirect URL from a product ID.

    Args:
        id_product: The Cardmarket product ID (e.g. '483559' or 483559).
        language: Language code for the URL path (default 'es').

    Returns:
        Direct Cardmarket product redirect URL or None if id_product is empty/None.
    """
    if not id_product:
        return None
    return f"https://www.cardmarket.com/{language}/Pokemon/Products?idProduct={id_product}"


@dataclass
class MatchResult:
    """Result of matching an offer against TCGDex."""

    tcgdex_id: Optional[str]
    cardmarket_product_id: Optional[str]
    confidence_score: float
    card_language: str = "es"
    matched_name: Optional[str] = None
    matched_local_id: Optional[str] = None
    matched_set_name: Optional[str] = None
    cardmarket_url: Optional[str] = None
    cardmarket_low_price: Optional[float] = None
    cardmarket_trend_price: Optional[float] = None


class TCGDexMatcher:
    """Matches Pokemon card listings against the TCGDex database using the official Python SDK."""

    def __init__(
        self,
        default_language: str = "es",
        fallback_language: str = "en",
        min_confidence: float = 60.0,
    ):
        """Initializes the matcher.

        Args:
            default_language: Primary language for TCGDex searches (default: "es").
            fallback_language: Fallback language if no results found (default: "en").
            min_confidence: Minimum score (0-100) required to accept a match.
        """
        self.default_language = default_language
        self.fallback_language = fallback_language
        self.min_confidence = min_confidence

        self._sdk_es = TCGdex(default_language)
        self._sdk_en = TCGdex(fallback_language)

        # In-memory caches to prevent redundant network requests
        self._search_cache: Dict[str, List[Any]] = {}
        self._card_details_cache: Dict[str, Dict[str, Any]] = {}
        self._offer_cache: Dict[str, MatchResult] = {}

    def _get_sdk(self, language: str) -> TCGdex:
        return self._sdk_es if language == "es" else self._sdk_en

    def _search_cards(self, query_name: str, language: str = "es") -> List[Any]:
        """Searches TCGDex cards matching a query name in the given language."""
        cache_key = f"{language}:{query_name.lower().strip()}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        sdk = self._get_sdk(language)
        results = []
        try:
            # Query containing the name
            results = sdk.card.listSync(Query().contains("name", query_name))
        except Exception as exc:
            logger.debug(f"TCGDex search failed for '{query_name}' in {language}: {exc}")

        # If empty and contains multiple words, try searching by the primary word (e.g. Pokemon base name)
        if not results and " " in query_name:
            first_word = query_name.split()[0]
            if len(first_word) >= 3:
                try:
                    results = sdk.card.listSync(Query().contains("name", first_word))
                except Exception as exc:
                    logger.debug(f"TCGDex fallback search failed for '{first_word}': {exc}")

        self._search_cache[cache_key] = results or []
        return self._search_cache[cache_key]

    def _fetch_card_raw_details(self, card_id: str, language: str = "es") -> Dict[str, Any]:
        """Fetches raw card details from the TCGDex REST endpoint to access pricing & Cardmarket ID."""
        cache_key = f"{language}:{card_id}"
        if cache_key in self._card_details_cache:
            return self._card_details_cache[cache_key]

        import urllib.request

        sdk = self._get_sdk(language)
        url = f"{sdk.getEndpoint()}/{language}/cards/{card_id}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "@tcgdex/python-sdk@2.3.0"},
        )
        data = {}
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"Failed to fetch raw details for card {card_id}: {exc}")
            # Try English endpoint as fallback
            if language != "en":
                try:
                    url_en = f"{self._sdk_en.getEndpoint()}/en/cards/{card_id}"
                    req_en = urllib.request.Request(
                        url_en,
                        headers={"User-Agent": "@tcgdex/python-sdk@2.3.0"},
                    )
                    with urllib.request.urlopen(req_en, timeout=5) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except Exception:
                    pass

        self._card_details_cache[cache_key] = data
        return data

    def extract_cardmarket_id(self, raw_card_data: Dict[str, Any]) -> Optional[str]:
        """Extracts the Cardmarket Product ID (idProduct) from raw TCGDex card data."""
        if not raw_card_data:
            return None

        # 1. Check pricing.cardmarket.idProduct
        pricing = raw_card_data.get("pricing")
        if isinstance(pricing, dict):
            cm = pricing.get("cardmarket")
            if isinstance(cm, dict) and cm.get("idProduct"):
                return str(cm["idProduct"])

        # 2. Check variants_detailed[].thirdParty.cardmarket
        variants = raw_card_data.get("variants_detailed")
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    tp = variant.get("thirdParty")
                    if isinstance(tp, dict) and tp.get("cardmarket"):
                        return str(tp["cardmarket"])
                    # Check pricing inside variant
                    v_pricing = variant.get("pricing")
                    if isinstance(v_pricing, dict):
                        v_cm = v_pricing.get("cardmarket")
                        if isinstance(v_cm, dict) and v_cm.get("idProduct"):
                            return str(v_cm["idProduct"])

        return None

    def extract_cardmarket_prices(
        self, raw_card_data: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extracts low price and trend price from TCGDex card pricing."""
        if not raw_card_data:
            return None, None

        pricing = raw_card_data.get("pricing")
        if isinstance(pricing, dict):
            cm = pricing.get("cardmarket")
            if isinstance(cm, dict):
                low = cm.get("low")
                trend = cm.get("trend")
                return (
                    float(low) if low is not None else None,
                    float(trend) if trend is not None else None,
                )

        variants = raw_card_data.get("variants_detailed")
        if isinstance(variants, list):
            for v in variants:
                if isinstance(v, dict):
                    v_cm = v.get("pricing", {}).get("cardmarket", {})
                    low = v_cm.get("low")
                    trend = v_cm.get("trend")
                    if low is not None or trend is not None:
                        return (
                            float(low) if low is not None else None,
                            float(trend) if trend is not None else None,
                        )

        return None, None

    def build_cardmarket_url(self, id_product: Optional[Any]) -> Optional[str]:
        """Constructs the Cardmarket redirect URL using the matcher's default language."""
        return build_cardmarket_url(id_product, language=self.default_language)

    def _score_candidate(
        self,
        candidate: Any,
        parsed: ParsedCardInfo,
    ) -> float:
        """Calculates a confidence score (0-100) for a candidate card matching parsed info."""
        score = 0.0

        cand_name = getattr(candidate, "name", "")
        cand_local_id = str(getattr(candidate, "localId", "")).lstrip("0")
        cand_id = getattr(candidate, "id", "")

        # 1. Card Number match (very strong signal: 45 points)
        if parsed.local_id and cand_local_id:
            parsed_num = parsed.local_id.lstrip("0")
            if parsed_num == cand_local_id:
                score += 45.0
            elif parsed.local_id.lower() in cand_id.lower():
                score += 35.0
        elif not parsed.local_id:
            # Neutral if no number present in title
            score += 15.0

        # 2. Name fuzzy similarity (up to 40 points)
        name_sim = fuzz.token_set_ratio(parsed.card_name.lower(), cand_name.lower())
        score += (name_sim / 100.0) * 40.0

        # 3. Modifier matching (VMAX, EX, V, etc. up to 15 points)
        for mod in parsed.modifier_hints:
            if mod.lower() in cand_name.lower() or mod.lower() in cand_id.lower():
                score += 15.0
                break

        return min(100.0, score)

    def match(
        self,
        title: str,
        description: Optional[str] = None,
    ) -> MatchResult:
        """Matches a card listing title & description to a TCGDex card.

        Returns:
            MatchResult containing tcgdex_id, cardmarket_product_id, and confidence score.
        """
        cache_key = f"{title.strip()}|{(description or '').strip()[:100]}"
        if cache_key in self._offer_cache:
            return self._offer_cache[cache_key]

        parsed = parse_card_title(title, description)
        if not parsed.card_name:
            result = MatchResult(
                tcgdex_id=None,
                cardmarket_product_id=None,
                confidence_score=0.0,
            )
            self._offer_cache[cache_key] = result
            return result

        # Search in detected card language first, fallback to alternate language
        primary_lang = parsed.language if parsed.language in ("es", "en") else self.default_language
        secondary_lang = self.fallback_language if primary_lang == self.default_language else self.default_language

        candidates = self._search_cards(parsed.card_name, language=primary_lang)
        used_lang = primary_lang
        if not candidates:
            candidates = self._search_cards(parsed.card_name, language=secondary_lang)
            used_lang = secondary_lang

        if not candidates:
            result = MatchResult(
                tcgdex_id=None,
                cardmarket_product_id=None,
                confidence_score=0.0,
            )
            self._offer_cache[cache_key] = result
            return result

        # Score all candidates
        scored = []
        for cand in candidates:
            score = self._score_candidate(cand, parsed)
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand = scored[0]

        if best_score < self.min_confidence:
            result = MatchResult(
                tcgdex_id=None,
                cardmarket_product_id=None,
                confidence_score=best_score,
            )
            self._offer_cache[cache_key] = result
            return result

        # Retrieve card raw details to get Cardmarket Product ID & pricing
        raw_data = self._fetch_card_raw_details(best_cand.id, language=used_lang)
        cardmarket_id = self.extract_cardmarket_id(raw_data)
        cardmarket_url = self.build_cardmarket_url(cardmarket_id)
        cm_low, cm_trend = self.extract_cardmarket_prices(raw_data)

        matched_set = None
        if hasattr(best_cand, "set") and hasattr(best_cand.set, "name"):
            matched_set = best_cand.set.name
        elif isinstance(raw_data.get("set"), dict):
            matched_set = raw_data["set"].get("name")

        result = MatchResult(
            tcgdex_id=best_cand.id,
            cardmarket_product_id=cardmarket_id,
            confidence_score=best_score,
            card_language=used_lang,
            matched_name=getattr(best_cand, "name", None),
            matched_local_id=str(getattr(best_cand, "localId", "")),
            matched_set_name=matched_set,
            cardmarket_url=cardmarket_url,
            cardmarket_low_price=cm_low,
            cardmarket_trend_price=cm_trend,
        )
        self._offer_cache[cache_key] = result
        return result
