"""Scryfall and Magic: The Gathering catalog, pricing, and card assigner module."""

from .client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
)
from .assigner import (
    assign_scryfall_to_offers,
    enrich_magic_offers_in_db,
    extract_magic_identifiers,
    resolve_scryfall_card,
)

__all__ = [
    "ScryfallClient",
    "build_magic_cardmarket_url",
    "extract_scryfall_cardmarket_prices",
    "fetch_scryfall_card_pricing",
    "assign_scryfall_to_offers",
    "enrich_magic_offers_in_db",
    "extract_magic_identifiers",
    "resolve_scryfall_card",
]
