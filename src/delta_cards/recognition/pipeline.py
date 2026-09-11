"""End-to-end asynchronous orchestration pipeline for Pokémon card visual recognition, OCR, entity extraction, and TCGdex resolution."""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from .entity_parser import EntityParser, ExtractedCardMetadata
from .ocr_reader import CardOCRReader
from .tcgdex_resolver import TCGDexResolver
from .visual_detector import CardNotDetectedError, VisualCardDetector

logger = logging.getLogger(__name__)


class CardRecognitionPipeline:
    """Asynchronous orchestrator coordinating Visual Detection, OCR, LLM Parsing, and TCGdex Resolution."""

    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        default_language: str = "es",
    ):
        """Initializes pipeline components."""
        self.detector = VisualCardDetector(model_path=yolo_model_path)
        self.ocr_reader = CardOCRReader()
        self.entity_parser = EntityParser(model_path=llm_model_path)
        self.resolver = TCGDexResolver(default_language=default_language)

    async def process_card_listing(
        self,
        image_source: Union[str, bytes, np.ndarray],
        title: str,
        description: str,
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:

        """Executes the full recognition pipeline asynchronously.

        1. Visual Isolation & Perspective Deskew (CPU via asyncio.to_thread).
        2. Image Preprocessing & Multi-pass OCR (CPU via asyncio.to_thread).
        3. Local LLM Entity Extraction & Normalization (CPU via asyncio.to_thread).
        4. TCGdex Live API Cascade Resolution (Async I/O).

        Args:
            image_source: Path to image file, raw bytes, or NumPy BGR array.
            title: Marketplace seller listing title.
            description: Marketplace seller listing description.
            image_url: Optional direct marketplace image URL.

        Returns:
            Consolidated dictionary with verified card, Cardmarket product ID, and profiling metrics.
        """
        import time

        logger.info(f"Starting card recognition for listing: '{title}'")
        t_start = time.perf_counter()

        # Step 1: Visual Card Isolation & Rectification (in thread pool)
        card_image = None
        isolation_status = "isolated"
        t0 = time.perf_counter()
        try:
            card_image = await asyncio.to_thread(self.detector.isolate_card, image_source)
        except CardNotDetectedError as exc:
            logger.warning(f"Card isolation notice: {exc}. Proceeding with fallback image processing.")
            isolation_status = "fallback_full_frame"
            # Fall back to using the raw image directly for OCR
            card_image = await asyncio.to_thread(self.detector._load_image, image_source)
        except Exception as exc:
            logger.error(f"Unexpected error in card isolation: {exc}")
            isolation_status = "failed"
            card_image = None
        t_isolation = time.perf_counter() - t0

        # Step 2: Full-Card OCR Extraction (in thread pool)
        ocr_text = ""
        t1 = time.perf_counter()
        if card_image is not None:
            try:
                ocr_text = await asyncio.to_thread(self.ocr_reader.extract_text, card_image)
            except Exception as exc:
                logger.error(f"Error during OCR extraction: {exc}")
        t_ocr = time.perf_counter() - t1

        logger.debug(f"Extracted OCR text length: {len(ocr_text)} characters")

        # Step 3: Hybrid Extraction & Resolution
        # Fast-Path: Strict Deterministic Extraction (0.001s)
        t_hybrid_start = time.perf_counter()
        strict_meta = self.entity_parser.try_strict_deterministic_extraction(
            ocr_text=ocr_text,
            seller_title=title,
            seller_description=description,
        )

        metadata = None
        resolution_result = None
        extraction_method = "llm"
        t_llm = 0.0
        t_tcgdex = 0.0

        if strict_meta is not None:
            # Attempt strict direct resolution against TCGdex (Strategy A with zero-false-positive validation)
            t_strict_tcg = time.perf_counter()
            strict_resolution = await self.resolver.resolve_card(
                strict_meta, strict_verification=True
            )
            t_tcgdex = time.perf_counter() - t_strict_tcg
            if strict_resolution.get("status") == "success":
                metadata = strict_meta
                resolution_result = strict_resolution
                extraction_method = "strict_deterministic"
                t_llm = max(0.001, time.perf_counter() - t_hybrid_start - t_tcgdex)
                logger.info(
                    f"Hybrid fast-path SUCCEEDED (strict deterministic): "
                    f"card='{metadata.card_name}', id='{resolution_result['verified_card']['id']}'"
                )

        if resolution_result is None:
            # Fallback to local LLM when deterministic criteria are not met or card is not strictly confirmed
            t2 = time.perf_counter()
            metadata = await asyncio.to_thread(
                self.entity_parser.parse_listing,
                ocr_text=ocr_text,
                seller_title=title,
                seller_description=description,
                use_fast_path=False,
            )
            t_llm = time.perf_counter() - t2

            logger.info(
                f"LLM fallback extracted entities: name='{metadata.card_name}', "
                f"set='{metadata.set_code}', num='{metadata.collector_number}', lang='{metadata.language}'"
            )

            # Step 4: TCGdex Live Resolver (cascade search)
            t3 = time.perf_counter()
            resolution_result = await self.resolver.resolve_card(
                metadata, strict_verification=False
            )
            t_tcgdex = time.perf_counter() - t3

        t_total = time.perf_counter() - t_start

        # Resolved image URL reference
        inferred_img_url = image_url
        if not inferred_img_url and isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            inferred_img_url = image_source

        # Step 5: Consolidate Final Output Summary
        consolidated = {
            "status": resolution_result.get("status", "unknown"),
            "extraction_method": extraction_method,
            "image_url": inferred_img_url,
            "isolation_status": isolation_status,
            "card_rectified": card_image is not None,
            "ocr_text_snippet": ocr_text[:300] if ocr_text else "",
            "extracted_metadata": metadata.model_dump() if metadata else {},
            "strategy": resolution_result.get("strategy"),
            "verified_card": resolution_result.get("verified_card"),
            "cardmarket": resolution_result.get("cardmarket"),
            "profiling": {
                "isolation_s": round(t_isolation, 3),
                "ocr_s": round(t_ocr, 3),
                "entity_extraction_s": round(t_llm, 3),
                "tcgdex_resolution_s": round(t_tcgdex, 3),
                "total_s": round(t_total, 3),
            },
        }

        return consolidated


# Singleton pipeline instance for shared lifecycle
_shared_pipeline: Optional[CardRecognitionPipeline] = None


async def process_card_listing(
    image_source: Union[str, bytes, np.ndarray],
    title: str,
    description: str,
    image_url: Optional[str] = None,
    pipeline: Optional[CardRecognitionPipeline] = None,
) -> Dict[str, Any]:
    """Asynchronous entry point for processing a marketplace card listing."""
    global _shared_pipeline
    active_pipeline = pipeline or _shared_pipeline
    if active_pipeline is None:
        active_pipeline = CardRecognitionPipeline()
        if pipeline is None:
            _shared_pipeline = active_pipeline

    return await active_pipeline.process_card_listing(
        image_source=image_source,
        title=title,
        description=description,
        image_url=image_url,
    )



if __name__ == "__main__":
    import json

    print("=" * 70)
    print(" Delta Cards - Pokémon TCG Recognition & Resolution Pipeline (CPU)")
    print("=" * 70)

    # 1. Create a synthetic test card image for demonstration
    mock_img_path = "/tmp/mock_pokemon_card.jpg"
    synthetic_card = np.full((1050, 750, 3), 245, dtype=np.uint8)

    # Draw card border and text mimicking a physical Furret card
    cv2.rectangle(synthetic_card, (20, 20), (730, 1030), (40, 40, 40), 4)
    cv2.putText(
        synthetic_card,
        "Furret",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (20, 20, 20),
        3,
    )
    cv2.putText(
        synthetic_card,
        "110 HP",
        (550, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
    )
    cv2.putText(
        synthetic_card,
        "Quick Attack 30+",
        (80, 600),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
    )
    cv2.putText(
        synthetic_card,
        "Tail Smash 90",
        (80, 720),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
    )
    cv2.putText(
        synthetic_card,
        "136/201",
        (560, 990),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (30, 30, 30),
        2,
    )
    cv2.imwrite(mock_img_path, synthetic_card)
    print(f"[+] Mock card image generated at: {mock_img_path}")

    # 2. Mock Spanish Wallapop marketplace inputs
    wallapop_title = "Carta Pokemon Furret 136/201 Oscuridad Incandescente Español Holo"
    wallapop_desc = (
        "Vendo carta original de Pokémon TCG Furret número 136/201 "
        "de la colección Darkness Ablaze / Oscuridad Incandescente. Estado Mint, versión holográfica."
    )

    print(f"[+] Seller Title:       '{wallapop_title}'")
    print(f"[+] Seller Description: '{wallapop_desc}'")
    print("\nExecuting end-to-end asynchronous pipeline...")

    result = asyncio.run(
        process_card_listing(
            image_source=mock_img_path,
            title=wallapop_title,
            description=wallapop_desc,
        )
    )

    print("\nPipeline Result Summary:")
    print("-" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)
