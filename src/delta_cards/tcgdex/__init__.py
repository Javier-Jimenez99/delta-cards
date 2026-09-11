"""TCGDex integration module for Delta Cards.

Assigns single card offers from Wallapop to TCGDex cards and Cardmarket products
using CardRecognitionPipeline.
"""

from ..recognition.tcgdex_resolver import build_cardmarket_url
from .assigner import (
    EXCLUDED_PRODUCT_TYPES,
    LOT_PATTERNS,
    assign_tcgdex_to_offers,
    enrich_card_offers_in_db,
    is_card_offer,
)

__all__ = [
    "build_cardmarket_url",
    "is_card_offer",
    "assign_tcgdex_to_offers",
    "enrich_card_offers_in_db",
    "EXCLUDED_PRODUCT_TYPES",
    "LOT_PATTERNS",
]
