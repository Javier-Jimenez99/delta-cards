"""Unit and integration tests for the Pokémon TCG recognition pipeline."""

import asyncio
import numpy as np
import pytest
import cv2

from delta_cards.recognition.visual_detector import (
    VisualCardDetector,
    CardNotDetectedError,
    order_points,
)
from delta_cards.recognition.ocr_reader import CardOCRReader
from delta_cards.recognition.entity_parser import (
    EntityParser,
    ExtractedCardMetadata,
)
from delta_cards.recognition.tcgdex_resolver import (
    TCGDexResolver,
    build_cardmarket_url,
)
from delta_cards.recognition.pipeline import (
    CardRecognitionPipeline,
    process_card_listing,
)


@pytest.fixture
def synthetic_card_image():
    """Generates a synthetic high-contrast card image with border and text."""
    img = np.full((1050, 750, 3), 250, dtype=np.uint8)
    # Draw card outer frame
    cv2.rectangle(img, (25, 25), (725, 1025), (30, 30, 30), 4)
    # Card Header
    cv2.putText(img, "Furret", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(img, "110 HP", (540, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    # Attacks
    cv2.putText(img, "Quick Attack 30+", (80, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "Tail Smash 90", (80, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    # Collector Number
    cv2.putText(img, "136/201", (560, 990), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


class TestVisualDetector:
    """Tests for VisualCardDetector and perspective deskewing."""

    def test_order_points(self):
        # Coordinates in random order
        pts = np.array([[100, 500], [10, 10], [100, 10], [10, 500]], dtype="float32")
        ordered = order_points(pts)
        # Expected: TL=(10, 10), TR=(100, 10), BR=(100, 500), BL=(10, 500)
        np.testing.assert_array_almost_equal(ordered[0], [10, 10])
        np.testing.assert_array_almost_equal(ordered[1], [100, 10])
        np.testing.assert_array_almost_equal(ordered[2], [100, 500])
        np.testing.assert_array_almost_equal(ordered[3], [10, 500])

    def test_isolate_synthetic_card(self, synthetic_card_image):
        detector = VisualCardDetector()
        rectified = detector.isolate_card(synthetic_card_image)
        assert isinstance(rectified, np.ndarray)
        assert rectified.shape == (1050, 750, 3)

    def test_card_not_detected_error(self):
        detector = VisualCardDetector()
        # Uniform solid blank image with no borders or features
        blank = np.full((500, 500, 3), 128, dtype=np.uint8)
        with pytest.raises(CardNotDetectedError):
            detector.isolate_card(blank)


class TestCardOCRReader:
    """Tests for CardOCRReader image preprocessing and Tesseract extraction."""

    def test_preprocess_image(self, synthetic_card_image):
        reader = CardOCRReader()
        binary = reader.preprocess_image(synthetic_card_image)
        assert binary.ndim == 2
        assert binary.shape == (1050, 750)
        # Check that text edges are present (contains both 0 and 255 pixels)
        assert 0 in binary
        assert 255 in binary

    def test_extract_text(self, synthetic_card_image):
        reader = CardOCRReader()
        text = reader.extract_text(synthetic_card_image)
        assert isinstance(text, str)
        assert "Furret" in text or "FURRET" in text.upper()
        assert "136/201" in text or "136" in text


class TestEntityParser:
    """Tests for EntityParser and ExtractedCardMetadata schema."""

    def test_metadata_schema_validation(self):
        metadata = ExtractedCardMetadata(
            card_name="Charizard ex",
            set_code="sv03",
            collector_number="125/197",
            attacks=["Slash", "Burning Darkness"],
            language="ES",
            is_holo=True,
        )
        data = metadata.model_dump()
        assert data["card_name"] == "Charizard ex"
        assert data["set_code"] == "sv03"
        assert data["is_holo"] is True
        assert len(data["attacks"]) == 2

    def test_detect_card_language(self):
        # 1. OCR text in English with Spanish seller prose -> must be EN
        ocr_en = "Furret 110 HP Quick Attack Tail Smash Discard all Energy from this Pokémon"
        seller_title_es = "Carta Pokémon Furret Holo Impecable"
        seller_desc_es = "Vendo carta de mi colección personal en perfecto estado, envío rápido y protegido."
        assert EntityParser.detect_card_language(ocr_en, seller_title_es, seller_desc_es) == "en"

        # 2. OCR text in Spanish -> must be ES
        ocr_es = "Pikachu ex 200 PS Corazón Resuelto Rayo Top Descarta todas las Energías"
        assert EntityParser.detect_card_language(ocr_es, seller_title_es, seller_desc_es) == "es"

        # 3. Japanese card OCR text with Kanji/Kana -> must be JA
        ocr_ja = "ピカチュウ 60 HP でんきショック 10"
        assert EntityParser.detect_card_language(ocr_ja, seller_title_es, seller_desc_es) == "ja"

        # 4. Short OCR with HP / PS markers
        assert EntityParser.detect_card_language("Mewtwo 120 HP", "Mewtwo holo", "Colección") == "en"
        assert EntityParser.detect_card_language("Mewtwo 120 PS", "Mewtwo holo", "Colección") == "es"
        assert EntityParser.detect_card_language("Mewtwo 120 KP", "Mewtwo holo", "Colección") == "de"

        # 5. Explicit seller declaration overriding default
        assert EntityParser.detect_card_language("", "Pikachu VMAX en inglés", "Carta original") == "en"
        assert EntityParser.detect_card_language("", "Pikachu VMAX versión japonesa", "Carta original") == "ja"

        # 6. Default fallback
        assert EntityParser.detect_card_language("", "Pikachu", "") == "es"

    def test_parse_listing_with_model(self):
        parser = EntityParser()
        ocr_text = "Furret 110 HP Quick Attack 30+ Tail Smash 90 136/201"
        title = "Carta Pokemon Furret 136/201 Darkness Ablaze Holo"
        desc = "Carta original en español en perfecto estado."

        result = parser.parse_listing(ocr_text, title, desc)
        assert isinstance(result, ExtractedCardMetadata)
        assert "Furret" in result.card_name
        assert "136" in (result.collector_number or "")
        assert result.is_holo is True


class TestTCGDexResolver:
    """Tests for TCGDexResolver live queries and Cardmarket ID retrieval."""

    @pytest.mark.asyncio
    async def test_strategy_a_direct_lookup(self):
        resolver = TCGDexResolver(default_language="es")
        metadata = ExtractedCardMetadata(
            card_name="Furret",
            set_code="swsh3",
            collector_number="136/201",
            attacks=["Quick Attack", "Tail Smash"],
            language="ES",
        )
        res = await resolver.resolve_card(metadata)
        assert res["status"] == "success"
        assert "Strategy A" in res["strategy"]
        assert res["verified_card"]["id"] == "swsh3-136"
        assert res["verified_card"]["localId"] == "136"
        assert res["cardmarket"]["idProduct"] == "483559"
        assert "idProduct=483559" in res["cardmarket"]["url"]

    @pytest.mark.asyncio
    async def test_strategy_b_filtered_search(self):
        resolver = TCGDexResolver(default_language="en")
        # Provide name and attacks without set code to force Strategy B
        metadata = ExtractedCardMetadata(
            card_name="Furret",
            set_code=None,
            collector_number="136",
            attacks=["Quick Attack", "Tail Smash"],
            language="EN",
        )
        res = await resolver.resolve_card(metadata)
        assert res["status"] == "success"
        assert "Strategy B" in res["strategy"]
        assert res["verified_card"]["name"] == "Furret"
        assert res["cardmarket"]["idProduct"] is not None

    def test_build_cardmarket_url(self):
        assert (
            build_cardmarket_url("483559", "es")
            == "https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559"
        )
        assert build_cardmarket_url(None) is None


class TestPipelineEndToEnd:
    """Tests for end-to-end CardRecognitionPipeline."""

    @pytest.mark.asyncio
    async def test_process_card_listing_early_text_fastpath(self):
        """Verifies early text fast-path succeeds in ~0.05s without needing an image."""
        title = "Carta Pokemon Furret 136/201 Darkness Ablaze Español"
        desc = "Vendo carta Furret 136/201 en perfecto estado."

        result = await process_card_listing(
            image_source=None,
            title=title,
            description=desc,
        )

        assert result["status"] == "success"
        assert result["extraction_method"] == "strict_deterministic_text"
        assert result["card_rectified"] is False
        assert result["verified_card"]["id"] == "swsh3-136"
        assert result["cardmarket"]["idProduct"] == "483559"
        assert result["cardmarket"]["prices"] is not None

    @pytest.mark.asyncio
    async def test_process_card_listing_async(self, synthetic_card_image):
        """Verifies visual isolation and OCR path when listing title does not contain the fraction."""
        title = "Carta Pokemon Furret Darkness Ablaze Español"
        desc = "Vendo carta Furret en perfecto estado."

        result = await process_card_listing(
            image_source=synthetic_card_image,
            title=title,
            description=desc,
        )

        assert result["status"] == "success"
        assert result["card_rectified"] is True
        assert result["verified_card"]["id"] == "swsh3-136"
        assert result["cardmarket"]["idProduct"] == "483559"
        assert result["cardmarket"]["prices"] is not None

