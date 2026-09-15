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


