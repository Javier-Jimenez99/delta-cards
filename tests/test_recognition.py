"""Unit and integration tests for the Pokémon TCG recognition pipeline."""

import asyncio
import cv2
import numpy as np
import pytest

from delta_cards.recognition.card_ocr import CardOCR
from delta_cards.recognition.card_text_service import CardTextService
from delta_cards.recognition.pipeline import CardRecognitionPipeline, process_card_listing
from delta_cards.recognition.visual_detector import CardNotDetectedError, VisualCardDetector


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


class TestVisualDetector:
    """Tests for VisualCardDetector."""

    def test_isolate_real_card(self):
        detector = VisualCardDetector()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"
        crop = detector.isolate_card(img_path)
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3
        assert crop.shape[0] > 100
        assert crop.shape[1] > 100

    def test_isolate_with_debug(self):
        detector = VisualCardDetector()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"
        crop, annotated = detector.isolate_card_with_debug(img_path)
        assert isinstance(crop, np.ndarray)
        assert annotated.shape[:2] == cv2.imread(img_path).shape[:2]

    def test_card_not_detected_error(self):
        detector = VisualCardDetector()
        blank = np.full((500, 500, 3), 128, dtype=np.uint8)
        with pytest.raises(CardNotDetectedError):
            detector.isolate_card(blank)


class TestCardOCRComponent:
    """Tests for CardOCR engine."""

    def test_extract_synthetic(self, synthetic_card_image):
        ocr = CardOCR()
        res = ocr.extract(synthetic_card_image)
        assert "furret" in res.full_text.lower()
        assert res.hp == 110
        assert any("136/201" in f[2] for f in res.fractions)


class TestCardTextServiceComponent:
    """Tests for CardTextService text retrieval and caching."""

    @pytest.mark.asyncio
    async def test_service_reconstruction(self):
        service = CardTextService()
        data = await service.get_reconstructed_card("sv03-001", lang="es")
        assert data is not None
        assert "Oddish" in data["name"]
        assert data["card_fraction"] is not None
        assert len(data["reconstructed_text"]) > 0


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
        assert "ocr_s" in result["profiling"]

    @pytest.mark.asyncio
    async def test_process_card_listing_real_image(self):
        pipe = CardRecognitionPipeline()
        img_path = "data/evaluation_set/images/sample_001_wallapop_base1-2.jpg"

        result = await pipe.process_card_listing(
            image_source=img_path,
            title="Blastoise Base Set",
            description="",
        )

        assert result["status"] == "success"
        assert result["confidence"] >= pipe.matcher.acceptance_threshold
        assert result["verified_card"] is not None
        assert "base1" in result["verified_card"]["id"].lower() or "blastoise" in result["verified_card"]["name"].lower()
