"""Simple card recognition pipeline."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .card_isolator import CardIsolator, CardNotDetectedError
from .card_ocr import CardOCR
from .card_ocr_cache import CandidateOCRCache
from .dinov3_embedder import DINOv3Embedder
from .text_matcher import rank_candidates
from .vector_index import USearchCardIndex


class CardRecognitionPipeline:
    """Isolate a card, find visual candidates, then compare complete OCR text."""

    def __init__(self, yolo_model_path: Optional[str] = None, dinov3_model_path: Optional[str] = None, index_path: Optional[str] = None, vector_index: Optional[USearchCardIndex] = None, default_language: str = "es", acceptance_threshold: float = 0.60, use_fgir: bool = True):
        self.default_language = default_language
        self.acceptance_threshold = acceptance_threshold
        self.isolator = CardIsolator(model_path=yolo_model_path)
        self.embedder = DINOv3Embedder(model_path=dinov3_model_path)
        self.vector_index = vector_index or USearchCardIndex(index_path=index_path, view_only=True)
        self.ocr = CardOCR()
        self.ocr_cache = CandidateOCRCache()

    async def process_card_listing(self, image_source: Union[str, bytes, np.ndarray], title: str = "", description: str = "", image_url: Optional[str] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            card_image, annotated_image = await asyncio.to_thread(self.isolator.isolate_card_with_debug, image_source)
            isolation_status = "isolated"
        except CardNotDetectedError:
            try:
                card_image = await asyncio.to_thread(self.isolator.load_image, image_source)
                annotated_image, isolation_status = None, "fallback_full_frame"
            except Exception:
                return self._empty_response("failed", image_url, started)
        except Exception:
            return self._empty_response("failed", image_url, started)
        if not len(self.vector_index):
            return self._empty_response(isolation_status, image_url, started)

        vector = await asyncio.to_thread(self.embedder.embed_image, card_image)
        candidates = await asyncio.to_thread(self.vector_index.search, vector, 15)
        query_ocr = await asyncio.to_thread(self.ocr.extract, card_image)
        candidate_texts = await asyncio.to_thread(self.ocr_cache.get_candidate_texts, candidates)
        match = rank_candidates(query_ocr.full_text, candidates, candidate_texts, self.acceptance_threshold)
        return self._response(match, query_ocr.full_text, query_ocr.detected_language, card_image, annotated_image, isolation_status, image_source, image_url, started)

    def _response(self, match: Dict[str, Any], query_text: str, query_language: str, card_image: np.ndarray, annotated_image: Optional[np.ndarray], isolation_status: str, image_source: Any, image_url: Optional[str], started: float) -> Dict[str, Any]:
        winner = match["verified_candidate"]
        ranked = match["ranked_candidates"]
        card = winner.get("card", {}) if winner else {}
        verified_card = self._card_payload(card) if winner else None
        response_image_url = image_url or (image_source if isinstance(image_source, str) and image_source.startswith("http") else None)
        return {
            "status": "success" if winner else "rejected", "extraction_method": "full_ocr_text" if winner else "rejected", "image_url": response_image_url, "isolation_status": isolation_status, "card_rectified": True, "ocr_text_snippet": query_text[:300],
            "extracted_metadata": {"card_name": card.get("name"), "set_code": card.get("set_id"), "collector_number": card.get("local_id"), "language": card.get("lang"), "certainty": match["confidence"]} if winner else None,
            "strategy": "visual_candidates_full_ocr_text", "card": verified_card, "verified_card": verified_card,
            "query_ocr": {"full_text": query_text, "detected_language": query_language}, "candidate_ocr": {"full_text": winner["candidate_text"]} if winner else {"full_text": ""}, "confidence": match["confidence"], "verification_method": "full_ocr_text" if winner else "rejected_below_threshold", "cardmarket": self._cardmarket(card) if winner else None,
            "debug_trace": {"isolation_status": isolation_status, "card_image": card_image, "annotated_image": annotated_image, "query_full_text": query_text, "ocr_detected_language": query_language, "candidates": [{"rank": index + 1, "combined_score": candidate["combined_score"], "tcgdex_id": candidate.get("card", {}).get("tcgdex_id"), "full_text": candidate["candidate_text"]} for index, candidate in enumerate(ranked)]},
            "profiling": {"total_s": round(time.perf_counter() - started, 3)},
        }

    def _card_payload(self, card: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": card.get("tcgdex_id"), "lang": card.get("lang"), "name": card.get("name"), "localId": card.get("local_id"), "image": card.get("image_url"), "rarity": card.get("rarity"), "hp": card.get("hp"), "set": {"id": card.get("set_id"), "name": card.get("set_name"), "cardCount": {"total": card.get("total_cards")}}}

    def _cardmarket(self, card: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {"idProduct": card["cardmarket_id"], "url": f"https://www.cardmarket.com/{card.get('lang', 'en')}/Pokemon/Products/Singles/{card.get('set_id')}/{card.get('name')}"} if card.get("cardmarket_id") else None

    def _empty_response(self, isolation_status: str, image_url: Optional[str], started: float) -> Dict[str, Any]:
        return {"status": "unrecognized", "extraction_method": "no_card_detected", "image_url": image_url, "isolation_status": isolation_status, "card_rectified": False, "ocr_text_snippet": "", "extracted_metadata": None, "strategy": "rejected_no_card", "card": None, "verified_card": None, "confidence": 0.0, "verification_method": "no_card_detected", "cardmarket": None, "debug_trace": {"isolation_status": isolation_status, "candidates": []}, "profiling": {"total_s": round(time.perf_counter() - started, 3)}}

    async def process_offers(self, offers: List[Any], progress_callback: Optional[Any] = None, max_concurrency: int = 4) -> List[Any]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_offer(offer: Any) -> Any:
            if getattr(offer, "tcgdex_id", None):
                return offer
            async with semaphore:
                result = await self.process_card_listing(getattr(offer, "image_url", None), getattr(offer, "title", ""), getattr(offer, "description", ""), getattr(offer, "image_url", None))
            if result["verified_card"]:
                offer.tcgdex_id = result["verified_card"]["id"]
                offer.language = result["verified_card"]["lang"]
                offer._query_ocr = result["query_ocr"]
                offer._candidate_ocr = result["candidate_ocr"]
                offer._tcgdex_card = result["verified_card"]
            if progress_callback:
                progress_callback(0, len(offers), offer)
            return offer

        return await asyncio.gather(*(process_offer(offer) for offer in offers))

    def process_offers_sync(self, offers: List[Any], progress_callback: Optional[Any] = None, max_concurrency: int = 4) -> List[Any]:
        return asyncio.run(self.process_offers(offers, progress_callback, max_concurrency))


_shared_pipeline: Optional[CardRecognitionPipeline] = None


async def process_card_listing(image_source: Union[str, bytes, np.ndarray], title: str = "", description: str = "", image_url: Optional[str] = None, pipeline: Optional[CardRecognitionPipeline] = None) -> Dict[str, Any]:
    global _shared_pipeline
    active_pipeline = pipeline or _shared_pipeline or CardRecognitionPipeline()
    if pipeline is None:
        _shared_pipeline = active_pipeline
    return await active_pipeline.process_card_listing(image_source, title, description, image_url)