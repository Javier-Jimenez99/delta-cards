"""End-to-end clean pipeline for Pokémon card visual recognition, full OCR, and TCGdex resolution."""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from fuzzywuzzy import fuzz

from .card_matcher import CardTextMatcher
from .card_ocr import CardOCR, CardOCRResult
from .card_ocr_cache import CandidateOCRCache
from .dinov3_embedder import DINOv3Embedder
from .vector_index import MultiLanguageCardIndex, USearchCardIndex
from .visual_detector import CardNotDetectedError, VisualCardDetector

logger = logging.getLogger(__name__)


class CardRecognitionPipeline:
    """Asynchronous orchestrator coordinating Visual Detection, Full-Card OCR, Language Routing, USearch, and TCGdex Fuzzy Matching."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        dinov3_model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        vector_index: Optional[Union[USearchCardIndex, MultiLanguageCardIndex]] = None,
        default_language: str = "es",
        acceptance_threshold: float = 0.60,
        use_fgir: bool = True,
    ):
        """Initializes pipeline components."""
        self.use_fgir = use_fgir
        self.default_language = default_language
        self.detector = VisualCardDetector(model_path=yolo_model_path)
        self.embedder = DINOv3Embedder(model_path=dinov3_model_path)
        if vector_index is not None:
            self.vector_index = vector_index
        elif index_path:
            self.vector_index = USearchCardIndex(index_path=index_path, view_only=True)
        else:
            self.vector_index = USearchCardIndex(view_only=True)
        self.ocr = CardOCR()
        self.ocr_cache = CandidateOCRCache()
        self.text_service = self.ocr_cache  # Backward compatibility
        self.matcher = CardTextMatcher(acceptance_threshold=acceptance_threshold)

    async def process_card_listing(
        self,
        image_source: Union[str, bytes, np.ndarray],
        title: str = "",
        description: str = "",
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executes the unified card recognition pipeline.

        1. Visual Card Isolation & Perspective Rectification (YOLO + warp).
        2. DINOv3 image visual embedding on card crop.
        3. USearch visual nearest neighbor search on unified index (all cards & languages).
        4. Full-Card OCR on the rectified card crop (RapidOCR ONNX).
        5. TCGdex candidate text reconstruction for all USearch candidates (SQLite cached).
        6. Multimodal fuzzy match comparison between OCR and all candidate texts to pick winner.

        Args:
            image_source: Path to image file, raw bytes, or NumPy BGR array.
            title: Marketplace seller listing title.
            description: Marketplace seller listing description.
            image_url: Optional direct image URL.

        Returns:
            Consolidated dictionary with verified card, Cardmarket product ID, and profiling metrics.
        """
        t_start = time.perf_counter()
        logger.info(f"Starting card recognition for listing: '{title}'")

        # Step 1: Visual Card Isolation & Perspective Rectification
        card_image = None
        annotated_original = None
        isolation_status = "isolated"
        t0 = time.perf_counter()

        if image_source is not None:
            try:
                card_image, annotated_original = await asyncio.to_thread(
                    self.detector.isolate_card_with_debug, image_source
                )
            except CardNotDetectedError as exc:
                logger.warning(f"Card boundaries not clearly detected ({exc}). Fallback to full frame.")
                isolation_status = "fallback_full_frame"
                try:
                    card_image = await asyncio.to_thread(self.detector._load_image, image_source)
                except Exception:
                    card_image = None
            except Exception as exc:
                logger.warning(f"Error in card isolation: {exc}")
                isolation_status = "failed"
                card_image = None
        else:
            isolation_status = "skipped_no_image"

        t_isolation = time.perf_counter() - t0

        if card_image is None or len(self.vector_index) == 0:
            t_total = time.perf_counter() - t_start
            return self._build_empty_response(
                isolation_status=isolation_status,
                image_url=image_url or (image_source if isinstance(image_source, str) and image_source.startswith("http") else None),
                t_total=t_total,
                t_isolation=t_isolation,
            )

        # Step 2: DINOv3 Visual Embedding (Card Crop)
        t_emb0 = time.perf_counter()
        query_vector = await asyncio.to_thread(self.embedder.embed_image, card_image)
        t_embedding = time.perf_counter() - t_emb0

        # Step 3: USearch Nearest Neighbor Visual Search on Unified Index
        t_srch0 = time.perf_counter()
        raw_srch = await asyncio.to_thread(self.vector_index.search, query_vector, 15)
        visual_cands = raw_srch[0] if isinstance(raw_srch, tuple) else raw_srch
        t_search = time.perf_counter() - t_srch0

        # Step 4: Full-Card OCR on the rectified card crop
        t_ocr0 = time.perf_counter()
        ocr_result: CardOCRResult = await asyncio.to_thread(self.ocr.extract, card_image)
        t_ocr = time.perf_counter() - t_ocr0

        # Step 4b: Hybrid Candidate Injection (OCR Fraction / Promo / Name from SQLite)
        candidates = self._inject_ocr_candidates(visual_cands, ocr_result, max_total=15)

        # Step 5: Retrieve Canonical Texts for all Candidate Options (SQLite Cached)
        t_recon0 = time.perf_counter()
        reconstructed_map = await self.ocr_cache.get_reconstructed_cards_batch(candidates)
        t_reconstruct = time.perf_counter() - t_recon0

        # Step 6: Fuzzy Matching Comparison between Query OCR and all USearch Candidates
        t_match0 = time.perf_counter()
        match_result = self.matcher.match_candidates(
            ocr_result=ocr_result,
            candidates=candidates,
            reconstructed_data_map=reconstructed_map,
        )
        t_match = time.perf_counter() - t_match0

        winner = match_result.get("verified_candidate")
        confidence = float(match_result.get("confidence", 0.0))
        is_verified = bool(match_result.get("is_verified", False))
        method = match_result.get("method", "rejected")
        ranked_candidates = match_result.get("ranked_candidates", [])

        t_total = time.perf_counter() - t_start

        inferred_img_url = image_url or (image_source if isinstance(image_source, str) and image_source.startswith("http") else None)
        winner_lang = winner.get("card", {}).get("lang") if winner else "en"

        debug_trace = {
            "isolation_status": isolation_status,
            "card_image": card_image,
            "annotated_image": annotated_original,
            "query_full_text": ocr_result.full_text,
            "ocr_lines": ocr_result.lines,
            "query_top_text": ocr_result.top_text,
            "query_bottom_text": ocr_result.bottom_text,
            "query_hp": ocr_result.hp,
            "query_fractions": [f[2] for f in ocr_result.fractions],
            "query_promos": ocr_result.promo_codes,
            "query_set_codes": ocr_result.promo_codes,
            "winner_language": winner_lang,
            "queried_index_language": winner_lang,
            "detected_language": winner_lang,
            "verification_method": method,
            "certainty": confidence,
            "is_verified": is_verified,
            "candidates": [
                {
                    "rank": idx + 1,
                    "similarity": round(float(c.get("similarity", 0.0)), 4),
                    "combined_score": round(float(c.get("combined_score", 0.0)), 4),
                    "tcgdex_id": c.get("card", {}).get("tcgdex_id"),
                    "lang": c.get("card", {}).get("lang"),
                    "name": c.get("card", {}).get("name"),
                    "set_name": c.get("card", {}).get("set_name"),
                    "set_id": c.get("card", {}).get("set_id"),
                    "local_id": c.get("card", {}).get("local_id"),
                    "card_fraction": c.get("card", {}).get("card_fraction"),
                    "image_url": c.get("card", {}).get("image_url"),
                    "high_image_url": c.get("card", {}).get("image_url"),
                    "fuzzy_metrics": c.get("fuzzy_metrics", {}),
                    "verification_method": c.get("verification_method", ""),
                }
                for idx, c in enumerate(ranked_candidates[:15])
            ],
        }

        if is_verified and winner:
            card_info = winner.get("card", {})
            rec_info = winner.get("reconstructed_card", {})

            verified_card = {
                "id": card_info.get("tcgdex_id"),
                "lang": winner_lang,
                "name": rec_info.get("name") or card_info.get("name"),
                "localId": card_info.get("local_id"),
                "image": card_info.get("image_url"),
                "rarity": card_info.get("rarity"),
                "hp": rec_info.get("hp") or card_info.get("hp"),
                "set": {
                    "id": card_info.get("set_id"),
                    "name": rec_info.get("set_name") or card_info.get("set_name"),
                    "cardCount": {"total": card_info.get("total_cards")},
                },
            }

            cm_data = None
            if card_info.get("cardmarket_id"):
                cm_data = {
                    "idProduct": card_info["cardmarket_id"],
                    "url": f"https://www.cardmarket.com/{winner_lang}/Pokemon/Products/Singles/{card_info.get('set_id')}/{card_info.get('name')}",
                }
            else:
                try:
                    from ..tcgdex.assigner import fetch_tcgdex_card_pricing
                    fetched = fetch_tcgdex_card_pricing(verified_card["id"], language=winner_lang)
                    if fetched:
                        cm_data = fetched
                except Exception as err:
                    logger.debug(f"Could not fetch online pricing for card {verified_card.get('id')}: {err}")

            logger.info(
                f"Card Recognized: '{verified_card['name']}' ({verified_card['id']} [{winner_lang.upper()}]) "
                f"confidence={confidence:.3f}, method={method} in {t_total*1000:.1f}ms"
            )

            return {
                "status": "success",
                "extraction_method": f"fgir_{method}",
                "image_url": inferred_img_url,
                "isolation_status": isolation_status,
                "card_rectified": card_image is not None,
                "ocr_text_snippet": ocr_result.full_text[:300],
                "extracted_metadata": {
                    "card_name": verified_card["name"],
                    "set_code": card_info.get("set_id"),
                    "collector_number": card_info.get("local_id"),
                    "language": winner_lang,
                    "certainty": confidence,
                },
                "strategy": "usearch_image_ocr_tcgdex_fuzzymatch",
                "card": verified_card,
                "verified_card": verified_card,
                "query_ocr": {
                    "full_text": ocr_result.full_text,
                    "top_text": ocr_result.top_text,
                    "bottom_text": ocr_result.bottom_text,
                    "hp": ocr_result.hp,
                    "fractions": [f[2] for f in ocr_result.fractions],
                    "promo_codes": ocr_result.promo_codes,
                    "lines": ocr_result.lines,
                },
                "candidate_ocr": {
                    "full_text": rec_info.get("full_text", ""),
                    "top_text": rec_info.get("top_text", ""),
                    "bottom_text": rec_info.get("bottom_text", ""),
                    "hp": rec_info.get("hp", ""),
                    "fractions": rec_info.get("fractions", []),
                    "promo_codes": rec_info.get("promo_codes", []),
                    "lines": rec_info.get("lines", []),
                },
                "confidence": confidence,
                "verification_method": method,
                "cardmarket": cm_data,
                "debug_trace": debug_trace,
                "profiling": {
                    "isolation_s": round(t_isolation, 3),
                    "embedding_s": round(t_embedding, 3),
                    "search_s": round(t_search, 4),
                    "ocr_s": round(t_ocr, 3),
                    "reconstruct_s": round(t_reconstruct, 3),
                    "matching_s": round(t_match, 4),
                    "micro_ocr_s": round(t_ocr + t_reconstruct + t_match, 3),
                    "total_s": round(t_total, 3),
                },
            }

        # Below threshold -> Unrecognized (safely reject counterfeit / unindexed)
        logger.info(
            f"Card unverified / rejected (best score={confidence:.3f}, "
            f"method={method}) in {t_total*1000:.1f}ms"
        )
        return {
            "status": "rejected",
            "extraction_method": "rejected",
            "image_url": inferred_img_url,
            "isolation_status": isolation_status,
            "card_rectified": card_image is not None,
            "ocr_text_snippet": ocr_result.full_text[:300],
            "extracted_metadata": None,
            "strategy": "rejected_below_threshold",
            "card": None,
            "verified_card": None,
            "confidence": confidence,
            "verification_method": method,
            "cardmarket": None,
            "debug_trace": debug_trace,
            "profiling": {
                "isolation_s": round(t_isolation, 3),
                "embedding_s": round(t_embedding, 3),
                "search_s": round(t_search, 4),
                "ocr_s": round(t_ocr, 3),
                "reconstruct_s": round(t_reconstruct, 3),
                "matching_s": round(t_match, 4),
                "micro_ocr_s": round(t_ocr + t_reconstruct + t_match, 3),
                "total_s": round(t_total, 3),
            },
        }

    def _inject_ocr_candidates(
        self,
        visual_candidates: List[Dict[str, Any]],
        ocr_result: CardOCRResult,
        max_total: int = 15,
    ) -> List[Dict[str, Any]]:
        """Injects high-confidence text-matched candidates from SQLite into visual candidates."""
        cand_map: Dict[str, Dict[str, Any]] = {}
        for c in visual_candidates:
            cid = c.get("card", {}).get("tcgdex_id") or c.get("tcgdex_id")
            if cid:
                cand_map[cid] = dict(c)

        q_top = (ocr_result.top_text or "").lower()
        q_full = (ocr_result.full_text or "").lower()

        # 1. Inject by detected collector fraction / local_id
        for q_num, q_tot, q_frac in ocr_result.fractions:
            matched_cards = self.vector_index.get_cards_by_local_id(q_num)
            scored_matches = []
            clean_q_top = re.sub(r"[\s\-_]", "", q_top)
            clean_q_full = re.sub(r"[\s\-_]", "", q_full)
            for mc in matched_cards:
                mc_name = (mc.get("name") or "").lower()
                clean_mc = re.sub(r"[\s\-_]", "", mc_name)
                n_sim = fuzz.partial_ratio(clean_mc, clean_q_top) if clean_q_top else 0.0
                if n_sim < 60.0 and clean_mc in clean_q_full:
                    n_sim = 85.0
                scored_matches.append((n_sim, mc))
            scored_matches.sort(key=lambda x: x[0], reverse=True)
            for n_sim, mc in scored_matches[:4]:
                cid = mc.get("tcgdex_id")
                if cid and cid not in cand_map:
                    cand_map[cid] = {
                        "key": mc.get("key", 0),
                        "similarity": 0.88 if n_sim >= 60.0 else 0.70,
                        "distance": 0.12 if n_sim >= 60.0 else 0.30,
                        "card": mc,
                        "source": "ocr_fraction",
                    }

        # 2. Inject by detected promo code
        for promo in ocr_result.promo_codes:
            matched_promos = self.vector_index.get_cards_by_local_id(promo)
            for mc in matched_promos[:4]:
                cid = mc.get("tcgdex_id")
                if cid and cid not in cand_map:
                    cand_map[cid] = {
                        "key": mc.get("key", 0),
                        "similarity": 0.85,
                        "distance": 0.15,
                        "card": mc,
                        "source": "ocr_promo",
                    }

        # 3. Inject by prominent Pokémon name in top_text if visual candidates have low similarity
        best_vis = max([float(c.get("similarity", 0.0)) for c in visual_candidates], default=0.0)
        if best_vis < 0.75:
            raw_words = re.findall(r"[A-Za-z]{3,}", ocr_result.top_text or "")
            words = []
            for rw in raw_words:
                words.append(rw)
                if rw.startswith("M") and len(rw) > 4 and rw[1].isupper():
                    words.append(rw[1:])
            ignore = {"pokemon", "basico", "basic", "stage", "fase", "nivel", "level", "vmax", "evoluciona", "evolves"}
            for w in words:
                if w.lower() in ignore:
                    continue
                matched_names = self.vector_index.search_cards_by_name(w, limit=4)
                for mc in matched_names:
                    cid = mc.get("tcgdex_id")
                    if cid and cid not in cand_map and len(cand_map) < max_total + 4:
                        cand_map[cid] = {
                            "key": mc.get("key", 0),
                            "similarity": 0.75,
                            "distance": 0.25,
                            "card": mc,
                            "source": "ocr_name",
                        }

        # Select top candidates, guaranteeing injected fraction/name matches are retained
        injected_cands = [c for c in cand_map.values() if c.get("source") in ("ocr_fraction", "ocr_promo")]
        vis_cands = [c for c in cand_map.values() if c.get("source") not in ("ocr_fraction", "ocr_promo")]
        vis_cands.sort(key=lambda x: float(x.get("similarity", 0.0)), reverse=True)

        selected = injected_cands + vis_cands
        return selected[:max_total]

    def _build_empty_response(
        self, isolation_status: str, image_url: Optional[str], t_total: float, t_isolation: float
    ) -> Dict[str, Any]:
        """Returns standard response when image loading or index fails."""
        return {
            "status": "unrecognized",
            "extraction_method": "no_card_detected",
            "image_url": image_url,
            "isolation_status": isolation_status,
            "card_rectified": False,
            "ocr_text_snippet": "",
            "extracted_metadata": None,
            "strategy": "rejected_no_card",
            "card": None,
            "verified_card": None,
            "confidence": 0.0,
            "verification_method": "no_card_detected",
            "cardmarket": None,
            "debug_trace": {
                "isolation_status": isolation_status,
                "card_image": None,
                "annotated_image": None,
                "candidates": [],
                "certainty": 0.0,
                "is_verified": False,
                "queried_index_language": "en",
                "detected_language": "en",
            },
            "profiling": {
                "isolation_s": round(t_isolation, 3),
                "total_s": round(t_total, 3),
            },
        }

    async def process_offers(
        self,
        offers: List[Any],
        progress_callback: Optional[Any] = None,
        max_concurrency: int = 4,
    ) -> List[Any]:
        """Processes a list of Wallapop offer listings concurrently with bounded parallelism."""
        try:
            from ..models import is_card_offer
        except ImportError:
            is_card_offer = None

        sem = asyncio.Semaphore(max_concurrency)
        counter = [0]
        total = len(offers)

        async def _process_single(offer):
            ptype = getattr(offer, "product_type", None)
            title = getattr(offer, "title", "")
            desc = getattr(offer, "description", "") or ""

            if is_card_offer and not is_card_offer(ptype, title=title, description=desc):
                return offer

            if getattr(offer, "tcgdex_id", None) or getattr(offer, "cardmarket_product_id", None):
                return offer

            async with sem:
                try:
                    result = await self.process_card_listing(
                        image_source=getattr(offer, "image_url", None),
                        title=title,
                        description=desc,
                        image_url=getattr(offer, "image_url", None),
                    )
                    if result.get("status") == "success" and result.get("verified_card"):
                        vc = result["verified_card"]
                        cm = result.get("cardmarket", {}) or {}
                        offer.tcgdex_id = vc.get("id")
                        if cm.get("idProduct") and hasattr(offer, "cardmarket_product_id"):
                            offer.cardmarket_product_id = cm.get("idProduct")
                        if hasattr(offer, "cardmarket_url"):
                            try:
                                offer.cardmarket_url = cm.get("url")
                            except Exception:
                                pass
                        prices = cm.get("prices", {}) or {}
                        if hasattr(offer, "cardmarket_low_price") and prices.get("low") is not None:
                            try:
                                offer.cardmarket_low_price = prices.get("low")
                            except Exception:
                                pass
                        if hasattr(offer, "cardmarket_avg_price") and prices.get("avg") is not None:
                            try:
                                offer.cardmarket_avg_price = prices.get("avg")
                            except Exception:
                                pass
                        if hasattr(offer, "cardmarket_trend_price") and prices.get("trend") is not None:
                            try:
                                offer.cardmarket_trend_price = prices.get("trend")
                            except Exception:
                                pass
                        if cm.get("url"):
                            object.__setattr__(offer, "_temp_cm_url", cm.get("url"))
                        if prices and prices.get("low") is not None:
                            object.__setattr__(offer, "_temp_cm_low_price", prices.get("low"))
                        if prices and prices.get("avg") is not None:
                            object.__setattr__(offer, "_temp_cm_avg_price", prices.get("avg"))
                        if prices and prices.get("trend") is not None:
                            object.__setattr__(offer, "_temp_cm_trend_price", prices.get("trend"))
                        meta = result.get("extracted_metadata", {}) or {}
                        if meta.get("language") and hasattr(offer, "language"):
                            offer.language = meta["language"]
                        try:
                            object.__setattr__(offer, "_query_ocr", result.get("query_ocr"))
                            object.__setattr__(offer, "_candidate_ocr", result.get("candidate_ocr"))
                            object.__setattr__(offer, "_tcgdex_card", vc)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning(f"Pipeline error processing offer {getattr(offer, 'id', '')}: {exc}")
                finally:
                    counter[0] += 1
                    if progress_callback:
                        progress_callback(counter[0], total, offer)
            return offer

        await asyncio.gather(*[_process_single(o) for o in offers])
        return offers

    def process_offers_sync(
        self,
        offers: List[Any],
        progress_callback: Optional[Any] = None,
        max_concurrency: int = 4,
    ) -> List[Any]:
        """Synchronous wrapper for batch processing offers."""
        return asyncio.run(
            self.process_offers(offers, progress_callback=progress_callback, max_concurrency=max_concurrency)
        )


_shared_pipeline: Optional[CardRecognitionPipeline] = None


async def process_card_listing(
    image_source: Union[str, bytes, np.ndarray],
    title: str = "",
    description: str = "",
    image_url: Optional[str] = None,
    pipeline: Optional[CardRecognitionPipeline] = None,
) -> Dict[str, Any]:
    """Convenience function to run card listing recognition with singleton pipeline."""
    global _shared_pipeline
    active = pipeline or _shared_pipeline
    if active is None:
        active = CardRecognitionPipeline()
        if pipeline is None:
            _shared_pipeline = active

    return await active.process_card_listing(
        image_source=image_source,
        title=title,
        description=description,
        image_url=image_url,
    )
