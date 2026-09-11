"""Pokémon Trading Card Game (TCG) visual recognition, OCR, and live resolution package."""

from .entity_parser import EntityParser, ExtractedCardMetadata, parse_listing
from .ocr_reader import CardOCRReader, extract_text
from .pipeline import CardRecognitionPipeline, process_card_listing
from .tcgdex_resolver import TCGDexResolver, build_cardmarket_url, resolve_card
from .visual_detector import CardNotDetectedError, VisualCardDetector, isolate_card

__all__ = [
    "VisualCardDetector",
    "CardNotDetectedError",
    "isolate_card",
    "CardOCRReader",
    "extract_text",
    "EntityParser",
    "ExtractedCardMetadata",
    "parse_listing",
    "TCGDexResolver",
    "resolve_card",
    "build_cardmarket_url",
    "CardRecognitionPipeline",
    "process_card_listing",
]
