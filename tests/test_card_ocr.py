"""Unit tests for full-text OCR and text ranking."""

import numpy as np
import pytest
import cv2

from delta_cards.recognition.card_ocr import CardOCR, CardOCRResult
from delta_cards.recognition.text_matcher import rank_candidates


@pytest.fixture
def sample_card_image():
    """Generates a synthetic high-contrast card image with border and text."""
    img = np.full((1050, 750, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (25, 25), (725, 1025), (30, 30, 30), 4)
    cv2.putText(img, "Furret", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(img, "110 HP", (540, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "Quick Attack 30+", (80, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "Tail Smash 90", (80, 720), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "136/201", (560, 990), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


def test_card_ocr_extraction(sample_card_image):
    ocr = CardOCR()
    res = ocr.extract(sample_card_image)
    assert isinstance(res, CardOCRResult)
    assert "furret" in res.full_text.lower()
    assert res.detected_language == "en"


def test_card_ocr_language_detection():
    ocr = CardOCR()

    # Spanish card
    text_es = "Oddish 50 PS Básico Buen Rollito Roba 1 carta. Estampida 10 001/230 Llamas Obsidianas"
    lang_es = ocr._detect_language(text_es, "Oddish 50 PS", 50)
    assert lang_es == "es"

    # English card
    text_en = "Oddish 50 HP Basic Feelin' Fine Draw a card. Stampede 10 001/197 Obsidian Flames"
    lang_en = ocr._detect_language(text_en, "Oddish 50 HP", 50)
    assert lang_en == "en"

    # Japanese card
    text_ja = "ピカチュウ 60 HP でんきショック 10"
    lang_ja = ocr._detect_language(text_ja, "ピカチュウ", 60)
    assert lang_ja == "ja"

    # German card
    text_de = "Glurak 120 KP Basis Flammenwurf 90"
    lang_de = ocr._detect_language(text_de, "Glurak 120 KP", 120)
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    assert lang_de == detect(text_de)


def test_full_text_ranking():
    query_text = "Oddish 50 HP Basic Feelin' Fine Draw a card. 001/197"

    candidates = [
        {"similarity": 0.89, "card": {"tcgdex_id": "sv03-001", "name": "Oddish", "card_fraction": "001/197", "set_id": "sv03"}},
        {"similarity": 0.85, "card": {"tcgdex_id": "sv03-002", "name": "Gloom", "card_fraction": "002/197", "set_id": "sv03"}},
    ]

    candidate_texts = {
        "sv03-001_en": "Oddish 50 HP Feelin' Fine Draw a card 001/197",
        "sv03-002_en": "Gloom 80 HP 002/197",
    }

    match_result = rank_candidates(query_text, candidates, candidate_texts, 0.6)
    assert match_result["is_verified"] is True
    assert match_result["verified_candidate"]["card"]["tcgdex_id"] == "sv03-001"
    assert match_result["confidence"] > 0.9


def test_full_text_ranking_rejects_unrelated_text():

    candidates = [
        {"similarity": 0.35, "card": {"tcgdex_id": "sv03-001", "name": "Oddish", "card_fraction": "001/197", "set_id": "sv03"}},
    ]

    candidate_texts = {"sv03-001_en": "Oddish 50 HP Feelin' Fine 001/197"}

    match_result = rank_candidates("Unrelated text not matching any card", candidates, candidate_texts, 0.65)
    assert match_result["is_verified"] is False
    assert match_result["verified_candidate"] is None
