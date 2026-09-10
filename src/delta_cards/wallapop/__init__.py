"""Wallapop card scraping integration."""

from .client import CardSearchClient
from .scraper import CardOffersScraper, run_scraper, raw_to_offer_listing
from .filters import is_offer_valid

# Aliases
WallapopClient = CardSearchClient
WallapopScraper = CardOffersScraper

__all__ = [
    "CardSearchClient",
    "WallapopClient",
    "CardOffersScraper",
    "WallapopScraper",
    "run_scraper",
    "raw_to_offer_listing",
    "is_offer_valid",
]
