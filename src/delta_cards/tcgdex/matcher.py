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



@dataclass
class MatchResult:
    """Result of matching an offer against TCGDex."""

    tcgdex_id: Optional[str]
    cardmarket_product_id: Optional[str]
    confidence_score: float
    matched_name: Optional[str] = None
    matched_local_id: Optional[str] = None
    matched_set_name: Optional[str] = None


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

        # First search in default language (Spanish), fallback to English if needed
        candidates = self._search_cards(parsed.card_name, language=self.default_language)
        used_lang = self.default_language
        if not candidates:
            candidates = self._search_cards(parsed.card_name, language=self.fallback_language)
            used_lang = self.fallback_language

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

        # Retrieve card raw details to get Cardmarket Product ID
        raw_data = self._fetch_card_raw_details(best_cand.id, language=used_lang)
        cardmarket_id = self.extract_cardmarket_id(raw_data)

        matched_set = None
        if hasattr(best_cand, "set") and hasattr(best_cand.set, "name"):
            matched_set = best_cand.set.name
        elif isinstance(raw_data.get("set"), dict):
            matched_set = raw_data["set"].get("name")

        result = MatchResult(
            tcgdex_id=best_cand.id,
            cardmarket_product_id=cardmarket_id,
            confidence_score=best_score,
            matched_name=getattr(best_cand, "name", None),
            matched_local_id=str(getattr(best_cand, "localId", "")),
            matched_set_name=matched_set,
        )
        self._offer_cache[cache_key] = result
        return result
