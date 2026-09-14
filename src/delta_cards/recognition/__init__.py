"""Pokémon Trading Card Game (TCG) visual recognition, OCR, and live resolution package."""

from .card_matcher import CardTextMatcher
from .card_ocr import CardOCR, CardOCRResult
from .card_ocr_cache import CandidateOCRCache
from .card_text_service import CardTextService
from .dinov3_embedder import DINOv3Embedder
from .llm_verifier import LLMCardVerifier, LLMVerificationResult, get_llm_verifier
from .pipeline import CardRecognitionPipeline, process_card_listing
from .vector_index import MultiLanguageCardIndex, USearchCardIndex
from .visual_detector import CardNotDetectedError, VisualCardDetector, isolate_card

__all__ = [
    "VisualCardDetector",
    "CardNotDetectedError",
    "isolate_card",
    "DINOv3Embedder",
    "USearchCardIndex",
    "MultiLanguageCardIndex",
    "CardOCR",
    "CardOCRResult",
    "CandidateOCRCache",
    "CardTextService",
    "CardTextMatcher",
    "CardRecognitionPipeline",
    "process_card_listing",
    "LLMCardVerifier",
    "LLMVerificationResult",
    "get_llm_verifier",
]
