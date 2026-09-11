"""Delta Cards: Marketplace card offers scraper configured via JSON."""

from .models import SearchConfig, OfferListing, SearchProfile, WallapopSettings, OutputSettings
from .config_loader import load_config
from .wallapop import CardSearchClient, CardOffersScraper, WallapopClient, WallapopScraper, run_scraper
from .tcgdex import TCGDexMatcher, assign_tcgdex_to_offers, enrich_card_offers_in_db, is_card_offer
from .recognition import (
    CardNotDetectedError,
    CardOCRReader,
    CardRecognitionPipeline,
    EntityParser,
    ExtractedCardMetadata,
    TCGDexResolver,
    VisualCardDetector,
    build_cardmarket_url,
    extract_text,
    isolate_card,
    parse_listing,
    process_card_listing,
    resolve_card,
)
from .settings import Settings, get_settings
from .alerts import DiscordAlertService, evaluate_alert_condition, AlertConditionResult
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
    "CardNotDetectedError",
    "CardOCRReader",
    "CardRecognitionPipeline",
    "EntityParser",
    "ExtractedCardMetadata",
    "TCGDexResolver",
    "VisualCardDetector",
    "build_cardmarket_url",
    "extract_text",
    "isolate_card",
    "parse_listing",
    "process_card_listing",
    "resolve_card",
    "Settings",
    "get_settings",
    "DiscordAlertService",
    "evaluate_alert_condition",
    "AlertConditionResult",
    "main",
]


