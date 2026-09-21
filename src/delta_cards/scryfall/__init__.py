"""Scryfall and Magic: The Gathering catalog, pricing, and card assigner module."""

from .client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
)
from .vector_cache import (
    ScryfallVectorCache,
    get_default_scryfall_vector_cache,
    get_scryfall_image_url,
)
from .assigner import (
    assign_scryfall_to_offers,
    enrich_magic_offers_in_db,
    extract_magic_identifiers,
    resolve_scryfall_card,
    select_best_magic_printing,
    select_best_magic_printing_visual,
)

__all__ = [
    "ScryfallClient",
    "build_magic_cardmarket_url",
    "extract_scryfall_cardmarket_prices",
    "fetch_scryfall_card_pricing",
    "ScryfallVectorCache",
    "get_default_scryfall_vector_cache",
    "get_scryfall_image_url",
    "assign_scryfall_to_offers",
    "enrich_magic_offers_in_db",
    "extract_magic_identifiers",
    "resolve_scryfall_card",
    "select_best_magic_printing",
    "select_best_magic_printing_visual",
]

