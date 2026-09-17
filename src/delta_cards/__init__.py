"""Delta Cards: Marketplace card offers scraper configured via JSON."""

from .models import SearchConfig, OfferListing, SearchProfile, WallapopSettings, VintedSettings, OutputSettings
from .config_loader import load_config
from .wallapop import CardSearchClient, CardOffersScraper, WallapopClient, WallapopScraper, run_scraper
from .vinted import VintedSearchClient, VintedScraper
from .tcgdex import assign_tcgdex_to_offers, enrich_card_offers_in_db, is_card_offer, build_cardmarket_url
from .scryfall import (
    assign_scryfall_to_offers,
    enrich_magic_offers_in_db,
    fetch_scryfall_card_pricing,
    build_magic_cardmarket_url,
)
from .recognition import (
    CardNotDetectedError,
    VisualCardDetector,
    isolate_card,
    DINOv3Embedder,
    USearchCardIndex,
    CardOCR,
    CardOCRResult,
    CandidateOCRCache,
    CardTextService,
    CardTextMatcher,
    CardRecognitionPipeline,
    process_card_listing,
)
from .settings import Settings, get_settings
from .alerts import (
    DiscordAlertService,
    AlertWorkerService,
    evaluate_alert_condition,
    AlertConditionResult,
)
from .cli import main, main_scrape, main_assign, main_alert, main_scrape_vinted

__all__ = [
    "SearchConfig",
    "OfferListing",
    "SearchProfile",
    "WallapopSettings",
    "VintedSettings",
    "OutputSettings",
    "load_config",
    "CardSearchClient",
    "CardOffersScraper",
    "WallapopClient",
    "WallapopScraper",
    "run_scraper",
    "VintedSearchClient",
    "VintedScraper",
    "assign_tcgdex_to_offers",
    "enrich_card_offers_in_db",
    "is_card_offer",
    "build_cardmarket_url",
    "assign_scryfall_to_offers",
    "enrich_magic_offers_in_db",
    "fetch_scryfall_card_pricing",
    "build_magic_cardmarket_url",
    "CardNotDetectedError",
    "VisualCardDetector",
    "isolate_card",
    "DINOv3Embedder",
    "USearchCardIndex",
    "CardOCR",
    "CardOCRResult",
    "CandidateOCRCache",
    "CardTextService",
    "CardTextMatcher",
    "CardRecognitionPipeline",
    "process_card_listing",
    "Settings",
    "get_settings",
    "DiscordAlertService",
    "AlertWorkerService",
    "evaluate_alert_condition",
    "AlertConditionResult",
    "main",
    "main_scrape",
    "main_assign",
    "main_alert",
    "main_scrape_vinted",
]
