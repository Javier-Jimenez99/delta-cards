"""Delta Cards: Marketplace card offers scraper configured via JSON."""

from .models import SearchConfig, OfferListing, SearchProfile, WallapopSettings, OutputSettings
from .config_loader import load_config
from .wallapop import CardSearchClient, CardOffersScraper, WallapopClient, WallapopScraper, run_scraper
from .tcgdex import TCGDexMatcher, assign_tcgdex_to_offers, enrich_card_offers_in_db, is_card_offer
from .cli import main

__all__ = [
    "SearchConfig",
    "OfferListing",
    "SearchProfile",
    "WallapopSettings",
    "OutputSettings",
    "load_config",
    "CardSearchClient",
    "CardOffersScraper",
    "WallapopClient",
    "WallapopScraper",
    "run_scraper",
    "TCGDexMatcher",
    "assign_tcgdex_to_offers",
    "enrich_card_offers_in_db",
    "is_card_offer",
    "main",
]

