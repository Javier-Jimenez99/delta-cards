"""Non-singles / Sealed products (Pokémon & Magic: The Gathering) matching package."""

from .assigner import (
    FORMAT_TO_CATEGORIES,
    MAGIC_CATEGORY_TO_PRODUCT_TYPE,
    MAGIC_FORMAT_TO_CATEGORIES,
    MIN_SCORE_THRESHOLD,
    NonSinglesMatcher,
    POKEMON_CATEGORY_TO_PRODUCT_TYPE,
    POKEMON_FORMAT_TO_CATEGORIES,
    assign_nonsingles_to_offers,
    resolve_product_type_from_category,
)

__all__ = [
    "FORMAT_TO_CATEGORIES",
    "MAGIC_CATEGORY_TO_PRODUCT_TYPE",
    "MAGIC_FORMAT_TO_CATEGORIES",
    "MIN_SCORE_THRESHOLD",
    "NonSinglesMatcher",
    "POKEMON_CATEGORY_TO_PRODUCT_TYPE",
    "POKEMON_FORMAT_TO_CATEGORIES",
    "assign_nonsingles_to_offers",
    "resolve_product_type_from_category",
]
