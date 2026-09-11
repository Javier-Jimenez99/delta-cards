"""TCGDex integration module for Delta Cards.

Matches single card offers from Wallapop to TCGDex cards using the official TCGDex Python SDK,
and extracts TCGDex IDs and Cardmarket Product IDs.
"""

from .parser import ParsedCardInfo, parse_card_title, extract_card_number
from .matcher import TCGDexMatcher, MatchResult, build_cardmarket_url
from .assigner import (
    is_card_offer,
    assign_tcgdex_to_offers,
    enrich_card_offers_in_db,
    EXCLUDED_PRODUCT_TYPES,
)

__all__ = [
    "ParsedCardInfo",
    "parse_card_title",
    "extract_card_number",
    "TCGDexMatcher",
    "MatchResult",
    "build_cardmarket_url",
    "is_card_offer",
    "assign_tcgdex_to_offers",
    "enrich_card_offers_in_db",
    "EXCLUDED_PRODUCT_TYPES",
]
