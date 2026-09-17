"""Non-singles / Sealed Pokémon products matching package."""

from .assigner import (
    FORMAT_TO_CATEGORIES,
    MIN_SCORE_THRESHOLD,
    NonSinglesMatcher,
    assign_nonsingles_to_offers,
    resolve_product_type_from_category,
)

__all__ = [
    "FORMAT_TO_CATEGORIES",
    "MIN_SCORE_THRESHOLD",
    "NonSinglesMatcher",
    "assign_nonsingles_to_offers",
    "resolve_product_type_from_category",
]
