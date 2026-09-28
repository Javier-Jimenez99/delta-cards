"""Cardmarket Non-Singles (Sealed Products) Matcher and Assigner.

Matches freeform marketplace listings (Wallapop, Vinted) for sealed products
(Booster Packs, Booster Boxes, Displays, ETBs, Tins, Collection Boxes)
against the official Cardmarket non-singles catalog using FlashText entity extraction
and BM25 ranking.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from flashtext import KeywordProcessor
from rank_bm25 import BM25Okapi

from ..models import OfferListing

logger = logging.getLogger(__name__)

# Condition and language tags excluded from title BM25 ranking
METADATA_TAGS: Set[str] = {
    "Sealed",
    "Spanish",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Italian",
    "Simplified Chinese",
    "Traditional Chinese",
}

# Tags identifying non-sealed listings (single cards, bulk lots, empty boxes, merch)
EXCLUSION_TAGS: Set[str] = {
    "Excluded_NonSealed",
}

# Category constraints: maps format entity to Cardmarket categoryName
POKEMON_FORMAT_TO_CATEGORIES: Dict[str, Set[str]] = {
    "Elite Trainer Box": {"Pokémon Elite Trainer Boxes"},
    "Pokémon Center Elite Trainer Box": {"Pokémon Elite Trainer Boxes"},
    "Booster Box": {"Pokémon Display"},
    "Booster Bundle": {"Pokémon Display", "Pokémon Box Set"},
    "Booster": {"Pokémon Booster", "Pokémon Blisters"},
    "Sleeved Booster": {"Pokémon Booster"},
    "Ultra-Premium Collection": {"Pokémon Box Set"},
    "Premium Collection": {"Pokémon Box Set"},
    "Special Collection": {"Pokémon Box Set"},
    "Tin": {"Pokémon Tins"},
    "Mini Tin": {"Pokémon Tins"},
    "1-Pack Blister": {"Pokémon Blisters"},
    "2-Pack Blister": {"Pokémon Blisters"},
    "3-Pack Blister": {"Pokémon Blisters"},
    "Premium Checklane Blister": {"Pokémon Blisters"},
    "Theme Deck": {"Pokémon Theme Decks"},
    "Battle Deck": {"Pokémon Theme Decks"},
    "League Battle Deck": {"Pokémon Theme Decks"},
    "Build & Battle Box": {"Pokémon Box Set"},
    "Build & Battle Stadium": {"Pokémon Box Set"},
    "Collector Chest": {"Pokémon Box Set", "Pokémon Tins"},
    "Poster Collection": {"Pokémon Box Set"},
    "Binder Collection": {"Pokémon Box Set"},
    "ex Box": {"Pokémon Box Set"},
    "EX Box": {"Pokémon Box Set"},
    "GX Box": {"Pokémon Box Set"},
    "V Box": {"Pokémon Box Set"},
    "VMAX Box": {"Pokémon Box Set"},
    "VSTAR Box": {"Pokémon Box Set"},
    "Poké Ball Tin": {"Pokémon Tins", "Pokémon Box Set"},
    "Stacking Tin": {"Pokémon Tins"},
}

MAGIC_FORMAT_TO_CATEGORIES: Dict[str, Set[str]] = {
    "Booster Box": {"Magic Display"},
    "Collector Booster Box": {"Magic Display"},
    "Play Booster Box": {"Magic Display"},
    "Draft Booster Box": {"Magic Display"},
    "Set Booster Box": {"Magic Display"},
    "Jumpstart Display": {"Magic Display"},
    "Bundle": {"Magic Fatpack"},
    "Gift Bundle": {"Magic Fatpack"},
    "Booster": {"Magic Booster"},
    "Collector Booster": {"Magic Booster"},
    "Play Booster": {"Magic Booster"},
    "Draft Booster": {"Magic Booster"},
    "Set Booster": {"Magic Booster"},
    "Commander Deck": {"Magic Intropack"},
    "Commander Deck Set": {"Magic Theme Deck Display"},
    "Commander Display": {"Magic Theme Deck Display"},
    "Prerelease Pack": {"Magic TournamentPack", "Magic Booster"},
    "Starter Kit": {"Magic Starter Deck", "Magic Intropack"},
    "Secret Lair": {"MtG Set", "Magic Booster"},
    "Draft Night": {"Magic Theme Deck Display"},
    "Scene Box": {"Magic Theme Deck Display", "MtG Set"},
}

# Alias for backwards compatibility
FORMAT_TO_CATEGORIES: Dict[str, Set[str]] = POKEMON_FORMAT_TO_CATEGORIES

# Cardmarket categoryName to normalized product_type identifier
POKEMON_CATEGORY_TO_PRODUCT_TYPE: Dict[str, str] = {
    "Pokémon Booster": "booster_pack",
    "Pokémon Blisters": "booster_pack",
    "Pokémon Display": "booster_box",
    "Pokémon Elite Trainer Boxes": "etb",
    "Pokémon Tins": "tin",
    "Pokémon Box Set": "box_set",
    "Pokémon Theme Decks": "deck",
}

MAGIC_CATEGORY_TO_PRODUCT_TYPE: Dict[str, str] = {
    "Magic Display": "booster_box",
    "Magic Booster": "booster_pack",
    "Magic Fatpack": "bundle",
    "Magic Intropack": "deck",
    "Magic Starter Deck": "deck",
    "Magic TournamentPack": "box_set",
    "Magic Theme Deck Display": "box_set",
    "MtG Set": "box_set",
    "Magic Lot": "bulk",
}

CATEGORY_TO_PRODUCT_TYPE: Dict[str, str] = POKEMON_CATEGORY_TO_PRODUCT_TYPE

MIN_SCORE_THRESHOLD: float = 6.5


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


def resolve_product_type_from_category(category_name: str, game: str = "pokemon") -> str:
    """Deterministically maps Cardmarket category to normalized product_type."""
    if str(game).lower() in ("magic", "mtg"):
        return MAGIC_CATEGORY_TO_PRODUCT_TYPE.get(category_name, "sealed")
    return POKEMON_CATEGORY_TO_PRODUCT_TYPE.get(category_name, "sealed")


class NonSinglesMatcher:
    """Extracts domain entities using FlashText and ranks Cardmarket non-single products with BM25."""

    _instances: Dict[str, "NonSinglesMatcher"] = {}

    def __init__(
        self,
        game: str = "pokemon",
        keywords_path: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
    ):
        self.game = str(game).lower()
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        game_folder = "magic" if self.game in ("magic", "mtg") else "pokemon"
        self.format_to_categories = (
            MAGIC_FORMAT_TO_CATEGORIES if game_folder == "magic" else POKEMON_FORMAT_TO_CATEGORIES
        )

        # Resolve keywords JSON path
        if keywords_path is None:
            candidates = [
                project_root / f"data/catalogs/{game_folder}/keywords.json",
                project_root / f"data/catalogs/{game_folder}/{game_folder}_keywords.json",
                project_root / f"data/catalogs/{game_folder}_keywords.json",
                project_root / f"notebooks/data/non-singles/{game_folder}_keywords.json",
            ]
            keywords_path = next((c for c in candidates if c.exists()), candidates[0])

        # Resolve catalog CSV path
        if catalog_path is None:
            candidates = [
                project_root / f"data/catalogs/{game_folder}/nonsingles.csv",
                project_root / f"data/catalogs/{game_folder}/{game_folder}_nonsingles.csv",
                project_root / f"data/catalogs/{game_folder}_nonsingles.csv",
                project_root / f"notebooks/data/non-singles/{game_folder}_nonsingles.csv",
            ]
            catalog_path = next((c for c in candidates if c.exists()), candidates[0])

        self.keywords_path = Path(keywords_path)
        self.catalog_path = Path(catalog_path)

        # 1. Initialize FlashText Extractor
        self.extractor = KeywordProcessor(case_sensitive=False)
        for char in "áéíóúüñÁÉÍÓÚÜÑ":
            self.extractor.add_non_word_boundary(char)

        if self.keywords_path.exists():
            with open(self.keywords_path, "r", encoding="utf-8") as f:
                self.extractor.add_keywords_from_dict(json.load(f))
            logger.info("NonSinglesMatcher (%s) loaded keywords from %s", self.game, self.keywords_path)
        else:
            logger.warning("Keywords JSON not found at %s", self.keywords_path)

        # 2. Load Catalog & Build BM25
        if self.catalog_path.exists():
            self.df_catalog = pd.read_csv(self.catalog_path)
            self.catalog_names = self.df_catalog["name"].tolist()
            tokenized_catalog = [tokenize(name) for name in self.catalog_names]
            self.bm25 = BM25Okapi(tokenized_catalog)
            logger.info(
                "NonSinglesMatcher (%s) loaded %d products from %s",
                self.game,
                len(self.df_catalog),
                self.catalog_path,
            )
        else:
            logger.warning("Catalog CSV not found at %s", self.catalog_path)
            self.df_catalog = pd.DataFrame()
            self.catalog_names = []
            self.bm25 = None

    @classmethod
    def get_instance(cls, game: str = "pokemon") -> "NonSinglesMatcher":
        """Returns a singleton instance of NonSinglesMatcher for the given game."""
        g = "magic" if str(game).lower() in ("magic", "mtg") else "pokemon"
        if g not in cls._instances:
            cls._instances[g] = cls(game=g)
        return cls._instances[g]

    def search_product(
        self,
        title: str,
        top_k: int = 3,
        min_score: float = MIN_SCORE_THRESHOLD,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Extract canonical entities and rank Cardmarket catalog products.

        Args:
            title: Freeform listing title or text.
            top_k: Maximum number of matches to return.
            min_score: Minimum BM25 score required for candidate acceptance.

        Returns:
            Tuple of (extracted_entities, list_of_matching_dicts).
        """
        if self.bm25 is None or self.df_catalog.empty:
            return [], []

        raw_entities = self.extractor.extract_keywords(title)
        unique_entities = list(dict.fromkeys(raw_entities))

        # 1. Immediately reject listings with explicit non-sealed indicators
        if any(e in EXCLUSION_TAGS for e in unique_entities):
            logger.debug(
                "Listing '%s' rejected due to non-sealed exclusion tags: %s",
                title,
                unique_entities,
            )
            return unique_entities, []

        # Filter out metadata tags (e.g. language/condition) for title matching
        product_entities = [
            e for e in unique_entities if e not in METADATA_TAGS and e not in EXCLUSION_TAGS
        ]
        if not product_entities:
            return unique_entities, []

        # Check if a product format constraint exists (e.g. Booster Box, ETB, Tin, Bundle)
        detected_formats = [e for e in product_entities if e in self.format_to_categories]
        non_format_entities = [e for e in product_entities if e not in self.format_to_categories]

        # 2. Guard 1: If only a generic format was detected without any specific expansion/set/character,
        # skip blind matching to prevent false positive defaults (e.g. 151 Bundle, Celebrations ETB)
        if detected_formats and not non_format_entities:
            logger.debug(
                "Listing '%s' has format %s but no expansion/set entity; skipping match.",
                title,
                detected_formats,
            )
            return unique_entities, []

        # Guard 2: Sealed non-singles assigner strictly requires a recognized format entity.
        # Listings without a format entity (e.g. loose cards, accessories, land packs, or set name alone)
        # must not be blindly assigned to sealed products.
        if not detected_formats:
            logger.debug(
                "Listing '%s' has entities %s but no recognized sealed format; skipping match.",
                title,
                non_format_entities,
            )
            return unique_entities, []

        # Precedence filter for detected formats:
        # If a display/box, bundle, or commander deck format is detected, loose booster packs
        # must not pollute allowed categories (e.g. "Display de 12 Boosters", "Caja 36 sobres", "Bundle 9 sobres")
        box_display_formats = {
            "Booster Box", "Collector Booster Box", "Play Booster Box",
            "Draft Booster Box", "Set Booster Box", "Jumpstart Display",
        }
        pack_formats = {"Booster", "Collector Booster", "Play Booster", "Draft Booster", "Set Booster"}
        bundle_formats = {"Bundle", "Gift Bundle"}
        deck_formats = {"Commander Deck", "Commander Deck Set", "Commander Display"}

        if (
            any(f in detected_formats for f in box_display_formats)
            or any(f in detected_formats for f in bundle_formats)
            or any(f in detected_formats for f in deck_formats)
        ):
            detected_formats = [f for f in detected_formats if f not in pack_formats]

        allowed_categories: Set[str] = set()
        for fmt in detected_formats:
            allowed_categories.update(self.format_to_categories[fmt])

        # Tokenize the query words
        query_tokens = [
            token for entity in product_entities for token in tokenize(entity)
        ]
        if not query_tokens:
            return unique_entities, []

        title_lower = title.lower()
        has_collector_signal = (
            any(t in title_lower for t in ["collector", "coleccionista"])
            or any("collector" in e.lower() for e in unique_entities)
        )
        has_bundle_signal = (
            "bundle" in title_lower
            or bool(re.search(r"\b6\s+(?:sobres|buste|packs|bustos)\b", title_lower))
            or any("bundle" in e.lower() for e in unique_entities)
        )

        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break

            cat = self.df_catalog.iloc[idx]["categoryName"]

            # Enforce category constraint if a format was detected
            if allowed_categories and cat not in allowed_categories:
                continue

            cand_name = self.catalog_names[idx]
            cand_name_lower = cand_name.lower()

            # Disambiguation 1: Collector vs Non-Collector in displays/boosters
            if not has_collector_signal and "collector" in cand_name_lower:
                continue
            if has_collector_signal and "collector" not in cand_name_lower:
                continue

            # Disambiguation 2: Booster Bundle vs 36-pack Booster Box
            if has_bundle_signal and "booster box" in cand_name_lower and "bundle" not in cand_name_lower:
                continue
            if not has_bundle_signal and "bundle" in cand_name_lower and any(b in title_lower for b in ["booster box", "display", "36 sobres", "36 buste"]):
                continue

            # Disambiguation 3: Individual items vs Multi-item Displays / Sets / Complete Sets
            # Guard against single Commander decks, scene boxes, or Secret Lair drops matching
            # full deck sets, scene box sets, or complete factory sets (: Full Set, : Mythic Set, : Rare Set)
            is_multi_item_cand = (
                "deck set" in cand_name_lower
                or "scene box set" in cand_name_lower
                or ": full set" in cand_name_lower
                or cand_name_lower.endswith(" full set")
                or ": mythic set" in cand_name_lower
                or ": rare set" in cand_name_lower
                or ": common set" in cand_name_lower
                or ": uncommon set" in cand_name_lower
                or (cat in ("Magic Theme Deck Display", "MtG Set") and ("display" in cand_name_lower or ("set" in cand_name_lower and not any(k in cand_name_lower for k in ["starter set", "prerelease"]))))
            )
            has_multi_set_signal = any(
                s in title_lower
                for s in [
                    "deck set", "set de mazos", "todos los mazos", "4 mazos", "5 mazos",
                    "pack 4", "pack de 4", "4x", "4 decks", "5 decks", "all 4", "all 5",
                    "coleccion completa", "colección completa", "display de commander",
                    "commander display", "deck display", "full set", "complete set",
                    "set completo", "todas las cartas", "mythic set", "rare set",
                    "scene box set", "set de scene box", "4 scene boxes", "4 scene box"
                ]
            ) or "Commander Deck Set" in detected_formats or "Commander Display" in detected_formats

            if not has_multi_set_signal and is_multi_item_cand:
                continue
            if has_multi_set_signal and not is_multi_item_cand:
                continue

            # Disambiguation 4: Cross-set collision guard (e.g. Shining Fates having no Booster Box matching Shining Darkness)
            # If multi-word set/character entities were extracted, ensure distinguishing tokens match candidate
            if non_format_entities:
                set_mismatch = False
                for nfe in non_format_entities:
                    set_tokens = [
                        t for t in tokenize(nfe)
                        if t not in {
                            "the", "of", "de", "del", "la", "el", "los", "las", "and", "y",
                            "box", "pack", "set", "pokemon", "magic", "tcg"
                        }
                    ]
                    if not set_tokens:
                        continue
                    # If set has edition/sequence number (e.g. MH3 vs MH2, Base Set 2), the number must match
                    if set_tokens[-1].isdigit() and set_tokens[-1] not in cand_name_lower:
                        set_mismatch = True
                        break
                    matched = [tok for tok in set_tokens if tok in cand_name_lower]
                    if len(set_tokens) >= 2 and set_tokens[-1] not in cand_name_lower and len(matched) < 2:
                        set_mismatch = True
                        break
                if set_mismatch:
                    continue

            row = self.df_catalog.iloc[idx]
            trend_val = float(row["trend"]) if pd.notna(row.get("trend")) else None
            avg_val = float(row["avg"]) if pd.notna(row.get("avg")) else None
            low_val = float(row["low"]) if pd.notna(row.get("low")) else None

            # Enforce low <= avg: Cardmarket's low is active asking price while avg is historical sales.
            # If low > avg, the lower price is the true minimum transaction/historical floor,
            # and the higher price is the current market value / ceiling.
            if low_val is not None and avg_val is not None and low_val > avg_val:
                all_prices = [p for p in (low_val, avg_val, trend_val) if p is not None]
                low_val = min(all_prices)
                avg_val = max(all_prices)

            results.append(
                {
                    "idProduct": int(row["idProduct"]),
                    "name": cand_name,
                    "categoryName": cat,
                    "trendPrice": trend_val,
                    "avgPrice": avg_val,
                    "lowPrice": low_val,
                    "score": round(score, 4),
                }
            )
            if len(results) >= top_k:
                break

        return unique_entities, results


def assign_nonsingles_to_offers(
    offers: List[OfferListing],
    matcher: Optional[NonSinglesMatcher] = None,
    game: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> List[OfferListing]:
    """Enriches sealed / non-single OfferListing objects with Cardmarket Product ID and prices.

    Matches listings using FlashText and BM25. If high confidence match is found,
    updates cardmarket_product_id, cardmarket_trend_price, and determines the
    concrete product_type (e.g. booster_pack, booster_box, etb, bundle) based on the Cardmarket category.

    Args:
        offers: List of OfferListing instances to assign.
        matcher: Optional NonSinglesMatcher instance.
        game: Default game ('pokemon' or 'magic') if not specified on individual offers.
        progress_callback: Optional callback receiving (index, total, offer).

    Returns:
        List[OfferListing]: Enriched offers.
    """
    if not offers:
        return []

    total = len(offers)
    for idx, offer in enumerate(offers, 1):
        # Skip if already assigned
        if getattr(offer, "cardmarket_product_id", None):
            if progress_callback:
                progress_callback(idx, total, offer)
            continue

        offer_game = (getattr(offer, "game", None) or game or "pokemon").lower()
        active_matcher = matcher or NonSinglesMatcher.get_instance(game=offer_game)

        entities, matches = active_matcher.search_product(offer.title, top_k=1)
        if matches:
            best = matches[0]
            cm_id = str(best["idProduct"])
            category = best["categoryName"]
            resolved_pt = resolve_product_type_from_category(category, game=offer_game)

            offer.cardmarket_product_id = cm_id
            offer.cardmarket_trend_price = best.get("trendPrice")
            offer.cardmarket_low_price = best.get("lowPrice")
            offer.cardmarket_avg_price = best.get("avgPrice")
            game_url_path = "Magic" if offer_game in ("magic", "mtg") else "Pokemon"
            offer.cardmarket_url = (
                f"https://www.cardmarket.com/{offer.language or 'es'}/{game_url_path}/Products?idProduct={cm_id}"
            )
            offer.product_type = resolved_pt
            offer.assigned_state = "assigned"
            logger.debug(
                "Matched non-single (%s) '%s' -> ID %s (%s) [Score: %.2f]",
                offer_game,
                offer.title,
                cm_id,
                best["name"],
                best["score"],
            )
        else:
            offer.assigned_state = "not_assigned"
            logger.debug("No confident Cardmarket non-single match for '%s'", offer.title)

        if progress_callback:
            progress_callback(idx, total, offer)

    return offers
