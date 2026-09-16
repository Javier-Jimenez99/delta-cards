"""Unit tests for deterministic card matching by edition fraction and Pokémon name."""

import sqlite3
import pytest
from unittest.mock import MagicMock

from delta_cards.recognition.deterministic_matcher import (
    _clean_text,
    _FRACTION_REGEX,
    match_card_by_index_and_name,
)
from delta_cards.recognition.pipeline import CardRecognitionPipeline


def test_clean_text_normalization():
    """Verifies that text is converted to lowercase, stripped of accents and punctuation."""
    assert _clean_text("Carta Pokémon: Blastoise, 2/102!") == " carta pokemon blastoise 2 102 "
    assert _clean_text("Flabébé & Farfetch'd") == " flabebe farfetch d "
    assert _clean_text("   Emolga-EX  (Base 2014)  ") == " emolga ex base 2014 "


def test_fraction_regex():
    """Verifies that number/number is extracted correctly and dates are ignored."""
    # Valid fractions
    assert _FRACTION_REGEX.findall("Pikachu 4/142") == [("4", "142")]
    assert _FRACTION_REGEX.findall("Enamorus 083/076 AR") == [("083", "076")]
    assert _FRACTION_REGEX.findall("Carta 07/23 promo") == [("07", "23")]
    assert _FRACTION_REGEX.findall("Carta 4 / 142 con espacios") == [("4", "142")]

    # Dates must NOT be matched as card fractions
    assert _FRACTION_REGEX.findall("Comprada el 12/10/2023 en tienda") == []
    assert _FRACTION_REGEX.findall("Temporada 2024/2025/2026") == []


@pytest.fixture
def mock_card_db():
    """In-memory SQLite database mimicking card_metadata."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE card_metadata (
            key INTEGER PRIMARY KEY,
            tcgdex_id TEXT NOT NULL,
            lang TEXT NOT NULL DEFAULT 'en',
            name TEXT NOT NULL,
            set_id TEXT NOT NULL,
            set_name TEXT NOT NULL,
            local_id TEXT NOT NULL,
            total_cards TEXT,
            card_fraction TEXT,
            rarity TEXT,
            hp INTEGER,
            cardmarket_id INTEGER,
            image_url TEXT,
            raw_json TEXT
        );
        """
    )
    conn.execute("CREATE INDEX idx_card_fraction ON card_metadata (card_fraction);")

    # Insert test cards
    cards = [
        # base1-2: Blastoise 2/102
        (1, "base1-2", "en", "Blastoise", "base1", "Base Set", "2", "102", "2/102", "Rare Holo", 100, 1001, "http://img/blastoise.png", "{}"),
        (2, "base1-2", "es", "Blastoise", "base1", "Base Set", "2", "102", "2/102", "Rare Holo", 100, 1001, "http://img/blastoise.png", "{}"),
        # bw3-2: Swadloon 2/102 (different pokemon in same fraction)
        (3, "bw3-2", "es", "Swadloon", "bw3", "Noble Victories", "2", "102", "2/102", "Uncommon", 70, 1002, "http://img/swadloon.png", "{}"),
        # Colliding cards sharing same name and fraction across different sets
        (4, "setA-4", "es", "Pikachu", "setA", "Set Alpha", "4", "100", "4/100", "Common", 60, 2001, "http://img/pikaA.png", "{}"),
        (5, "setB-4", "es", "Pikachu", "setB", "Set Beta", "4", "100", "4/100", "Common", 60, 2002, "http://img/pikaB.png", "{}"),
        # Padded fraction card: 083/076
        (6, "sv02-083", "es", "Tinkatuff", "sv02", "Paldea Evolved", "083", "193", "083/193", "Uncommon", 80, 3001, "http://img/tinka.png", "{}"),
        # swsh7-95: Umbreon VMAX 95/237 (official set size 203)
        (7, "swsh7-95", "en", "Umbreon VMAX", "swsh7", "Evolving Skies", "95", "237", "95/237", "Ultra Rare", 310, 4001, "http://img/umbreon.png", "{}"),
        (8, "swsh7-95", "es", "Umbreon VMAX", "swsh7", "Cielos Evolutivos", "95", "237", "95/237", "Ultra Rare", 310, 4001, "http://img/umbreon.png", "{}"),
        # swshp-SWSH261: Charizard VMAX Promo
        (9, "swshp-SWSH261", "es", "Charizard VMAX", "swshp", "SWSH Black Star Promos", "SWSH261", "307", "SWSH261/307", "Promo", 330, 5001, "http://img/zard_promo.png", "{}"),
    ]
    conn.executemany(
        """
        INSERT INTO card_metadata (
            key, tcgdex_id, lang, name, set_id, set_name, local_id, total_cards, card_fraction, rarity, hp, cardmarket_id, image_url, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        cards,
    )
    conn.commit()
    yield conn
    conn.close()



def test_match_card_by_index_and_name_success(mock_card_db):
    """Verifies that exact match works when fraction and name match uniquely."""
    res = match_card_by_index_and_name(
        title="Carta Pokémon Blastoise 2/102 Holo",
        description="En perfecto estado",
        db_conn=mock_card_db,
    )
    assert res is not None
    assert res["tcgdex_id"] == "base1-2"
    assert res["name"] == "Blastoise"


def test_match_card_by_index_and_name_name_in_description(mock_card_db):
    """Verifies that matching works when fraction is in title and name is in description."""
    res = match_card_by_index_and_name(
        title="Carta 2/102 vintage",
        description="Es una carta de Blastoise muy bien conservada",
        db_conn=mock_card_db,
    )
    assert res is not None
    assert res["tcgdex_id"] == "base1-2"


def test_match_card_by_index_and_name_collision_defers_to_visual(mock_card_db):
    """Verifies that if two different cards share the exact same name and fraction, returns None."""
    # Both setA-4 and setB-4 are Pikachu 4/100
    res = match_card_by_index_and_name(
        title="Pikachu 4/100 Pokemon card",
        description="",
        db_conn=mock_card_db,
    )
    # Must defer to visual pipeline
    assert res is None


def test_match_card_by_index_and_name_no_name_match(mock_card_db):
    """Verifies that if fraction matches in DB but pokemon name does not, returns None."""
    res = match_card_by_index_and_name(
        title="Carta Charizard 2/102",  # 2/102 has Blastoise and Swadloon, NOT Charizard
        description="",
        db_conn=mock_card_db,
    )
    assert res is None


def test_match_card_by_index_and_name_no_fraction(mock_card_db):
    """Verifies that listings without fraction return None."""
    res = match_card_by_index_and_name(
        title="Carta Charizard Base Set Holo",
        description="",
        db_conn=mock_card_db,
    )
    assert res is None


def test_match_card_by_official_set_count(mock_card_db):
    """Verifies that matching works when seller writes printed official set size (e.g. 95/203)."""
    # swsh7 official count is 203, but in DB total_cards is 237
    res = match_card_by_index_and_name(
        title="Umbreon VMAX 095/203 – Cielos Evolutivos Pokémon",
        description="Estado perfecto",
        db_conn=mock_card_db,
    )
    assert res is not None
    assert res["tcgdex_id"] == "swsh7-95"
    assert res["name"] == "Umbreon VMAX"


def test_match_card_by_promo_code(mock_card_db):
    """Verifies that matching works for promo codes (e.g. SWSH261)."""
    res = match_card_by_index_and_name(
        title="Charizard Vmax SWSH261 ita NM",
        description="",
        db_conn=mock_card_db,
    )
    assert res is not None
    assert res["tcgdex_id"] == "swshp-SWSH261"
    assert res["name"] == "Charizard VMAX"



@pytest.mark.asyncio
async def test_pipeline_process_card_listing_deterministic_fast_path(monkeypatch, mock_card_db):
    """Verifies that CardRecognitionPipeline uses deterministic fast path without invoking detector/OCR."""
    pipeline = CardRecognitionPipeline.__new__(CardRecognitionPipeline)
    pipeline.default_language = "es"
    pipeline.vector_index = MagicMock()
    pipeline.vector_index._db_conn = mock_card_db
    pipeline.vector_index.meta_path = None
    pipeline.detector = MagicMock()
    pipeline.ocr = MagicMock()
    pipeline.embedder = MagicMock()

    result = await pipeline.process_card_listing(
        image_source=None,
        title="Blastoise 2/102 Holo Base Set",
        description="Excelente estado",
    )

    assert result["status"] == "success"
    assert result["extraction_method"] == "deterministic_text_match"
    assert result["confidence"] == 1.0
    assert result["verified_card"]["id"] == "base1-2"
    assert result["verified_card"]["name"] == "Blastoise"

    # Detector and OCR must NOT be called
    pipeline.detector.isolate_card_with_debug.assert_not_called()
    pipeline.ocr.extract.assert_not_called()
    pipeline.embedder.embed_image.assert_not_called()
