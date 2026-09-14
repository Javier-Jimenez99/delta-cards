"""Delta Cards: Marketplace card offers scraper configured via JSON."""

from .models import SearchConfig, OfferListing, SearchProfile, WallapopSettings, OutputSettings
from .config_loader import load_config
from .wallapop import CardSearchClient, CardOffersScraper, WallapopClient, WallapopScraper, run_scraper
from .tcgdex import assign_tcgdex_to_offers, enrich_card_offers_in_db, is_card_offer, build_cardmarket_url
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
from .cli import main, main_scrape, main_assign, main_alert

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
    "assign_tcgdex_to_offers",
    "enrich_card_offers_in_db",
    "is_card_offer",
    "build_cardmarket_url",
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
]
