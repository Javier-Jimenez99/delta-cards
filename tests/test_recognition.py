"""Unit and integration tests for the Pokémon TCG recognition pipeline."""

import asyncio
import cv2
import numpy as np
import pytest

from delta_cards.recognition.card_ocr import CardOCR
from delta_cards.recognition.card_ocr_cache import CandidateOCRCache
from delta_cards.recognition.pipeline import CardRecognitionPipeline, process_card_listing
from delta_cards.recognition.card_isolator import CardIsolator, CardNotDetectedError


@pytest.fixture
def synthetic_card_image():
    """Generates a synthetic high-contrast card image with border and text."""
    img = np.full((1050, 750, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (725, 1025), (30, 30, 30), 4)
    cv2.putText(img, "Furret", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(img, "110 HP", (540, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Quick Attack 30+", (80, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "Tail Smash 90", (80, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "136/201", (560, 990), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


class TestCardIsolator:
    """Tests for CardIsolator."""

    def test_isolate_real_card(self):
        detector = CardIsolator()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"
        crop = detector.isolate_card(img_path)
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3
        assert crop.shape[0] > 100
        assert crop.shape[1] > 100

    def test_isolate_with_debug(self):
        detector = CardIsolator()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"
        crop, annotated = detector.isolate_card_with_debug(img_path)
        assert isinstance(crop, np.ndarray)
        assert annotated.shape[:2] == cv2.imread(img_path).shape[:2]

    def test_card_not_detected_error(self):
        detector = CardIsolator()
        blank = np.full((500, 500, 3), 128, dtype=np.uint8)
        with pytest.raises(CardNotDetectedError):
            detector.isolate_card(blank)


class TestCardOCRComponent:
    """Tests for CardOCR engine."""

    def test_extract_synthetic(self, synthetic_card_image):
        ocr = CardOCR()
        res = ocr.extract(synthetic_card_image)
        assert "furret" in res.full_text.lower()
        assert res.detected_language in ("en", "es")


class TestCandidateOCRCache:
    """Tests for full-text OCR caching."""

    def test_cache_stores_only_full_text(self, tmp_path):
        cache = CandidateOCRCache(db_path=str(tmp_path / "ocr.sqlite"))
        cache.save_text("sv03-001", "es", "Oddish 50 PS 001/230")
        assert cache.get_cached_text("sv03-001", "es") == "Oddish 50 PS 001/230"


class TestPipelineEndToEnd:
    """Tests for end-to-end CardRecognitionPipeline."""

    @pytest.mark.asyncio
    async def test_process_card_listing_synthetic(self, synthetic_card_image):
        pipe = CardRecognitionPipeline()
        result = await pipe.process_card_listing(
            image_source=synthetic_card_image,
            title="Carta Pokemon Furret 136/201 Darkness Ablaze",
            description="",
        )
        assert "status" in result
        assert "profiling" in result
        assert "total_s" in result["profiling"]

    @pytest.mark.asyncio
    async def test_process_card_listing_real_image(self):
        pipe = CardRecognitionPipeline()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"

        result = await pipe.process_card_listing(
            image_source=img_path,
            title="Blastoise Base Set",
            description="",
        )

        assert result["status"] in ("success", "rejected")
        assert "full_text" in result["query_ocr"]
