"""Unit tests for USearch vector index and metadata store."""

import os
import numpy as np
import pytest

from delta_cards.recognition.vector_index import USearchCardIndex


def test_usearch_add_search_and_mmap(tmp_path):
    idx_path = str(tmp_path / "cards.usearch")
    meta_path = str(tmp_path / "cards_meta.sqlite")

    index = USearchCardIndex(index_path=idx_path, meta_path=meta_path, view_only=False)

    # Add 3 test card vectors
    v0 = np.random.randn(384).astype(np.float32)
    v0 /= np.linalg.norm(v0)

    v1 = np.random.randn(384).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    index.add_card(0, v0, {
        "tcgdex_id": "sv01-001",
        "name": "Sprigatito",
        "set_id": "sv01",
        "set_name": "Scarlet & Violet",
        "local_id": "001",
        "total_cards": "198",
        "card_fraction": "001/198",
    })
    index.add_card(1, v1, {
        "tcgdex_id": "sv01-002",
        "name": "Floragato",
        "set_id": "sv01",
        "set_name": "Scarlet & Violet",
        "local_id": "002",
        "total_cards": "198",
        "card_fraction": "002/198",
    })

    assert len(index) == 2
    index.save()
    assert os.path.isfile(idx_path)
    assert os.path.isfile(meta_path)

    # Test Mmap view loading (Zero-RAM runtime mode)
    mmap_index = USearchCardIndex(index_path=idx_path, meta_path=meta_path, view_only=True)
    assert len(mmap_index) == 2

    # Query with v0: should return Sprigatito as Top-1 with similarity near 1.0
    results = mmap_index.search(v0, top_k=2)
    assert len(results) == 2
    assert results[0]["key"] == 0
    assert results[0]["card"]["name"] == "Sprigatito"
    assert results[0]["card"]["tcgdex_id"] == "sv01-001"
    assert np.isclose(results[0]["similarity"], 1.0, atol=1e-3)


def test_multi_language_card_index(tmp_path):
    from delta_cards.recognition.vector_index import MultiLanguageCardIndex

    # Create an 'es' index and an 'en' index in tmp_path
    es_idx_path = str(tmp_path / "pokemon_cards_es.usearch")
    es_meta_path = str(tmp_path / "pokemon_cards_es_meta.sqlite")
    es_idx = USearchCardIndex(index_path=es_idx_path, meta_path=es_meta_path, lang="es", view_only=False)

    v_es = np.random.randn(384).astype(np.float32)
    v_es /= np.linalg.norm(v_es)
    es_idx.add_card(0, v_es, {
        "tcgdex_id": "sv01-001",
        "lang": "es",
        "name": "Sprigatito (ES)",
        "set_id": "sv01",
        "set_name": "Escarlata y Púrpura",
        "local_id": "001",
    })
    es_idx.save()

    # Instantiate MultiLanguageCardIndex
    multi = MultiLanguageCardIndex(base_dir=str(tmp_path), default_lang="es")

    # Search in Spanish
    res_es, queried_lang = multi.search(v_es, lang="es", top_k=1)
    assert queried_lang == "es"
    assert len(res_es) == 1
    assert res_es[0]["card"]["name"] == "Sprigatito (ES)"

    # Search in French (not yet indexed) -> should gracefully fallback to default_lang ('es')
    res_fr, queried_fallback = multi.search(v_es, lang="fr", top_k=1)
    assert queried_fallback == "es"
    assert len(res_fr) == 1
    assert res_fr[0]["card"]["name"] == "Sprigatito (ES)"

