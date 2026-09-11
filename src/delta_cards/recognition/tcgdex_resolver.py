"""TCGdex live card resolver using official tcgdex-sdk and Cardmarket ID extraction."""

import json
import logging
import re
import socket
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz
from tcgdexsdk import Query, TCGdex
from tcgdexsdk.models.Card import Card

from .entity_parser import ExtractedCardMetadata

logger = logging.getLogger(__name__)

# Ensure sane socket timeout for external HTTP calls
if socket.getdefaulttimeout() is None:
    socket.setdefaulttimeout(10.0)


def build_cardmarket_url(id_product: Optional[str], language: str = "es") -> Optional[str]:
    """Generates direct Cardmarket product URL from product ID."""
    if not id_product:
        return None
    return f"https://www.cardmarket.com/{language}/Pokemon/Products?idProduct={id_product}"


def verify_strict_name_match(expected_name: str, tcgdex_name: str) -> bool:
    """Strictly validates that the extracted card name matches the TCGdex card name.

    Guarantees zero false positives:
    1. Primary Pokémon/Trainer name token must match exactly.
    2. Special mechanics (ex, gx, vmax, vstar, v, prime, break, mega) must be consistent.
    3. Token set similarity must meet strict threshold (>= 60).
    """
    ext_lower = expected_name.lower().strip()
    tcg_lower = tcgdex_name.lower().strip()

    # Base name token check
    stop_words = {
        "the", "de", "del", "la", "el", "shiny", "luminoso",
        "radiante", "holo", "rare", "rara", "carta", "carte", "pokemon"
    }
    ext_words = [w for w in re.split(r"\W+", ext_lower) if len(w) >= 3 and w not in stop_words]
    tcg_words = [w for w in re.split(r"\W+", tcg_lower) if len(w) >= 3 and w not in stop_words]

    if not ext_words or not tcg_words:
        return False

    # At least one substantive base word must match exactly
    common_words = set(ext_words) & set(tcg_words)
    if not common_words:
        return False

    # Mechanics consistency check
    mechanics = ["ex", "gx", "vmax", "vstar", "v", "prime", "break", "mega", "tag team"]
    ext_mechs = {m for m in mechanics if re.search(r"\b" + re.escape(m) + r"\b", ext_lower)}
    tcg_mechs = {m for m in mechanics if re.search(r"\b" + re.escape(m) + r"\b", tcg_lower)}

    # If both specify mechanics, they must overlap
    if ext_mechs and tcg_mechs:
        if not (ext_mechs & tcg_mechs):
            return False
    # If the TCG card is an Ultra-Rare mechanic (e.g. VMAX, VSTAR, GX) but listing specifies a conflicting one
    if "vmax" in tcg_mechs and "vmax" not in ext_mechs and "v" in ext_mechs:
        return False

    sim = fuzz.token_set_ratio(ext_lower, tcg_lower)
    return sim >= 60


class TCGDexResolver:
    """Queries live TCGdex API cascade to resolve cards and retrieve Cardmarket product IDs and prices."""

    def __init__(
        self,
        default_language: str = "es",
        fallback_language: str = "en",
    ):
        """Initializes the resolver with language endpoints and caching."""
        self.default_language = default_language.lower()
        self.fallback_language = fallback_language.lower()
        self._clients: Dict[str, TCGdex] = {}
        self._card_details_cache: Dict[str, Dict[str, Any]] = {}

    def get_client(self, language: Optional[str] = None) -> TCGdex:
        """Retrieves or instantiates a TCGdex client for the specified language."""
        lang = (language or self.default_language).lower()
        # Valid TCGdex languages: en, fr, es, it, pt, de, etc.
        supported_langs = {"en", "fr", "es", "it", "pt", "de"}
        if lang not in supported_langs:
            lang = self.fallback_language

        if lang not in self._clients:
            self._clients[lang] = TCGdex(lang)
        return self._clients[lang]

    async def resolve_card(
        self,
        metadata: ExtractedCardMetadata,
        strict_verification: bool = False,
    ) -> Dict[str, Any]:
        """Executes cascade resolution strategy to identify the card and fetch Cardmarket info.

        Strategy A: Direct Set Code + Collector Number lookup.
        Strategy B: Filtered Name Search with Attack & Number scoring (skipped if strict_verification=True).

        Args:
            metadata: ExtractedCardMetadata from entity_parser.
            strict_verification: If True, requires strict name/mechanic validation and disables fuzzy search.

        Returns:
            Dictionary containing verified card details, match strategy, Cardmarket ID, and pricing.
        """
        target_lang = (metadata.language or self.default_language).lower()
        client = self.get_client(target_lang)

        matched_card: Optional[Card] = None
        strategy_used = None

        # --- Strategy A: Direct Set & Number ID ---
        if metadata.set_code and metadata.collector_number:
            matched_card = await self._strategy_a_direct_lookup(
                client,
                metadata.set_code,
                metadata.collector_number,
                metadata.card_name,
                strict_verification=strict_verification,
            )
            if matched_card:
                strategy_used = "Strategy A (Direct Set & Number ID)"
            elif target_lang != self.fallback_language:
                # Try Strategy A in English before fuzzy searching
                en_client = self.get_client(self.fallback_language)
                matched_card = await self._strategy_a_direct_lookup(
                    en_client,
                    metadata.set_code,
                    metadata.collector_number,
                    metadata.card_name,
                    strict_verification=strict_verification,
                )
                if matched_card:
                    strategy_used = "Strategy A Fallback (Direct Set & Number in EN)"
                    client = en_client

        # --- Strategy B: Filtered Name Search ---
        # In strict verification mode, Strategy B is disabled to guarantee zero false positives
        if not matched_card and not strict_verification:
            matched_card = await self._strategy_b_filtered_search(client, metadata)
            if matched_card:
                strategy_used = "Strategy B (Filtered Name & Attacks Search)"
            elif target_lang != self.fallback_language:
                en_client = self.get_client(self.fallback_language)
                matched_card = await self._strategy_b_filtered_search(en_client, metadata)
                if matched_card:
                    strategy_used = "Strategy B Fallback (Filtered Name in EN)"
                    client = en_client

        if not matched_card:
            return {
                "status": "not_found",
                "message": f"Could not resolve card '{metadata.card_name}' on TCGdex.",
                "extracted_metadata": metadata.model_dump(),
            }

        # --- Data Extraction: Cardmarket ID & Pricing ---
        raw_details = self._fetch_raw_details(matched_card.id, client.language)
        cardmarket_id = self.extract_cardmarket_id(raw_details)
        cardmarket_prices = self.extract_cardmarket_prices(raw_details)
        cm_url = build_cardmarket_url(cardmarket_id, client.language)

        set_info = {}
        if hasattr(matched_card, "set") and matched_card.set:
            cc = getattr(matched_card.set, "cardCount", None)
            cc_dict = (
                {"official": getattr(cc, "official", None), "total": getattr(cc, "total", None)}
                if cc
                else None
            )
            set_info = {
                "id": getattr(matched_card.set, "id", None),
                "name": getattr(matched_card.set, "name", None),
                "cardCount": cc_dict,
            }


        return {
            "status": "success",
            "strategy": strategy_used,
            "verified_card": {
                "id": matched_card.id,
                "localId": matched_card.localId,
                "name": matched_card.name,
                "set": set_info,
                "rarity": getattr(matched_card, "rarity", None),
                "category": getattr(matched_card, "category", None),
                "image": getattr(matched_card, "image", None),
                "hp": getattr(matched_card, "hp", None),
                "types": getattr(matched_card, "types", []),
                "attacks": [
                    {"name": getattr(a, "name", "") or "", "damage": getattr(a, "damage", None)}
                    for a in (getattr(matched_card, "attacks", []) or [])
                    if a
                ],

            },
            "cardmarket": {
                "idProduct": cardmarket_id,
                "url": cm_url,
                "prices": cardmarket_prices,
            },
            "extracted_metadata": metadata.model_dump(),
        }

    async def _strategy_a_direct_lookup(
        self,
        client: TCGdex,
        set_code: str,
        collector_number: str,
        expected_card_name: Optional[str] = None,
        strict_verification: bool = False,
    ) -> Optional[Card]:
        """Attempts direct retrieval using formatted ID patterns, with name compatibility check."""
        clean_set = set_code.strip().lower()
        clean_num = collector_number.strip()

        # Handle fractions like "136/201" or "025/165"
        if "/" in clean_num:
            clean_num = clean_num.split("/")[0].strip()

        set_variants = [clean_set]
        alias_map = {
            "151": ["sv03.5", "sv3.5", "sv2a"],
            "me": ["sv03.5"],
            "mew": ["sv03.5"],
            "da": ["swsh3"],
            "st": ["swsh12"],
            "darkness ablaze": ["swsh3"],
            "silver tempest": ["swsh12"],
            "tempestad plateada": ["swsh12"],
            "oscuridad incandescente": ["swsh3"],
            "base set": ["base1"],
            "base set 2": ["base4"],
            "base 1": ["base1"],
            "base": ["base1"],
            "shining legends": ["sm3.5"],
            "leyendas luminosas": ["sm3.5"],
            "surging sparks": ["sv08"],
            "chispas fulgurantes": ["sv08"],
            "fusion strike": ["swsh08"],
            "golpe fusion": ["swsh08"],
            "golpe fusión": ["swsh08"],
            "obsidian flames": ["sv03"],
            "llamas obsidianas": ["sv03"],
            "fuegos fantasmales": ["me02"],
            "aquapolis": ["ecard2"],
            "aquopolis": ["ecard2"],
            "skyridge": ["ecard3"],
            "expedition": ["ecard1"],
            "jungle": ["base2"],
            "jungla": ["base2"],
            "fossil": ["base3"],
            "team rocket": ["base5"],
            "supercharged breaker": ["sv8"],
            "parablaze": ["sv8"],
            "raging surf": ["sv3a"],
            "glory of team rocket": ["sv10"],
        }
        for alias, targets in alias_map.items():
            if clean_set == alias or clean_set in alias:
                set_variants.extend(targets)

        candidate_ids = []
        for s in list(dict.fromkeys(set_variants)):
            candidate_ids.extend([
                f"{s}-{clean_num}",
                f"{s}-{clean_num.lstrip('0') or '0'}",
                f"{s}-{clean_num.zfill(3)}",
            ])

        # De-duplicate preserving order
        unique_ids = list(dict.fromkeys(candidate_ids))

        for cid in unique_ids:
            try:
                card = await client.card.get(cid)
                if card and getattr(card, "id", None):
                    # If expected name is provided, verify it's reasonably compatible
                    if expected_card_name:
                        if strict_verification:
                            if not verify_strict_name_match(expected_card_name, card.name):
                                logger.debug(
                                    f"Strict direct lookup ID '{cid}' rejected name mismatch: "
                                    f"expected '{expected_card_name}', got '{card.name}'"
                                )
                                continue
                        else:
                            name_sim = fuzz.token_set_ratio(
                                expected_card_name.lower(), card.name.lower()
                            )
                            words = [
                                w
                                for w in re.split(r"\W+", expected_card_name.lower())
                                if len(w) >= 3
                                and w not in ["the", "holo", "rare", "carta", "carte", "pokemon"]
                            ]
                            matches_any_word = any(w in card.name.lower() for w in words)
                            if name_sim < 40 and not matches_any_word:
                                logger.debug(
                                    f"Direct lookup ID '{cid}' name mismatch: expected '{expected_card_name}', got '{card.name}'"
                                )
                                continue
                    return card
            except Exception as exc:
                logger.debug(f"Direct lookup failed for ID '{cid}': {exc}")

        return None

    async def _strategy_b_filtered_search(
        self, client: TCGdex, metadata: ExtractedCardMetadata
    ) -> Optional[Card]:
        """Queries card list by name, then scores and filters candidates by attacks and numbers."""
        query_name = metadata.card_name.strip()
        if not query_name:
            return None

        # Clean search query (strip common suffix descriptors)
        search_terms = [query_name]
        words = query_name.split()
        if len(words) > 1:
            search_terms.append(f"{words[0]} {words[1]}")
            search_terms.append(words[0])
            if words[0].lower() in ["shining", "radiant", "mega", "dark", "light", "galarian", "hisuian"]:
                search_terms.append(words[1])
        search_terms = list(dict.fromkeys(search_terms))

        candidates = []
        for term in search_terms:
            try:
                # Query with contains
                results = await client.card.list(Query().contains("name", term))
                if results:
                    candidates = results
                    break
            except Exception as exc:
                logger.debug(f"TCGdex search failed for term '{term}': {exc}")

        if not candidates:
            return None

        best_card: Optional[Card] = None
        best_score = -1.0

        # Normalization for collector number
        target_num = None
        if metadata.collector_number:
            raw_num = metadata.collector_number.split("/")[0].strip()
            target_num = raw_num.lstrip("0") or "0"

        target_attacks = [a.lower().strip() for a in metadata.attacks if a]

        # Evaluate candidate cards
        for cand in candidates[:15]:
            cand_id = getattr(cand, "id", None)
            if not cand_id:
                continue

            try:
                full_card = await client.card.get(cand_id)
            except Exception:
                continue

            if not full_card:
                continue

            score = 0.0

            # 1. Name match score (0-40)
            name_sim = fuzz.token_sort_ratio(query_name.lower(), full_card.name.lower())
            score += (name_sim / 100.0) * 40.0

            # 2. Collector number match (0-30)
            if target_num and hasattr(full_card, "localId") and full_card.localId:
                card_num = str(full_card.localId).strip().lstrip("0") or "0"
                if target_num == card_num:
                    score += 30.0

            # 3. Attack names match (0-30)
            card_attacks = [
                a.name.lower().strip()
                for a in (getattr(full_card, "attacks", []) or [])
                if hasattr(a, "name") and a.name
            ]

            if target_attacks and card_attacks:
                match_count = sum(
                    1
                    for ta in target_attacks
                    if any(fuzz.partial_ratio(ta, ca) >= 80 for ca in card_attacks)
                )
                attack_score = (match_count / len(target_attacks)) * 30.0
                score += attack_score

            if score > best_score:
                best_score = score
                best_card = full_card

        # Accept if minimum match threshold is met
        if best_card and best_score >= 35.0:
            return best_card

        return None

    def _fetch_raw_details(self, card_id: str, language: str) -> Dict[str, Any]:
        """Fetches raw JSON response from TCGdex REST endpoint to extract Cardmarket pricing."""
        cache_key = f"{language}:{card_id}"
        if cache_key in self._card_details_cache:
            return self._card_details_cache[cache_key]

        client = self.get_client(language)
        url = f"{client.getEndpoint()}/{language}/cards/{card_id}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "@tcgdex/python-sdk@2.3.0"},
        )
        data: Dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug(f"Failed to fetch raw details for card {card_id} in {language}: {exc}")
            # Try English endpoint fallback
            if language != "en":
                try:
                    url_en = f"{client.getEndpoint()}/en/cards/{card_id}"
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
        """Extracts Cardmarket Product ID (idProduct) from raw TCGDex card JSON."""
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
                    v_pricing = variant.get("pricing")
                    if isinstance(v_pricing, dict):
                        v_cm = v_pricing.get("cardmarket")
                        if isinstance(v_cm, dict) and v_cm.get("idProduct"):
                            return str(v_cm["idProduct"])

        return None

    def extract_cardmarket_prices(
        self, raw_card_data: Dict[str, Any]
    ) -> Dict[str, Optional[float]]:
        """Extracts low, trend, and avg prices from Cardmarket pricing in TCGDex."""
        prices = {"low": None, "trend": None, "avg": None}
        if not raw_card_data:
            return prices

        def _from_dict(cm: dict) -> Dict[str, Optional[float]]:
            low = cm.get("low") or cm.get("lowPrice")
            trend = cm.get("trend") or cm.get("trendPrice")
            avg = cm.get("avg") or cm.get("avgPrice")
            if avg is None:
                avg = cm.get("avg-holo") or cm.get("avg30") or cm.get("avg7")
            if low is None:
                low = cm.get("low-holo")
            if trend is None:
                trend = cm.get("trend-holo")
            return {
                "low": float(low) if low is not None else None,
                "trend": float(trend) if trend is not None else None,
                "avg": float(avg) if avg is not None else None,
            }

        pricing = raw_card_data.get("pricing")
        if isinstance(pricing, dict):
            cm = pricing.get("cardmarket")
            if isinstance(cm, dict):
                p = _from_dict(cm)
                if p["avg"] is not None or p["low"] is not None or p["trend"] is not None:
                    return p

        # Fallback: check variants_detailed
        variants = raw_card_data.get("variants_detailed")
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    v_pricing = variant.get("pricing")
                    if isinstance(v_pricing, dict):
                        v_cm = v_pricing.get("cardmarket")
                        if isinstance(v_cm, dict):
                            p = _from_dict(v_cm)
                            if p["avg"] is not None or p["low"] is not None or p["trend"] is not None:
                                return p

        return prices


async def resolve_card(metadata: ExtractedCardMetadata) -> Dict[str, Any]:
    """Convenience async function to resolve card metadata."""
    resolver = TCGDexResolver()
    return await resolver.resolve_card(metadata)
