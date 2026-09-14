"""Unit tests for DINOv3 INT8 embedder."""

import numpy as np
import pytest

from delta_cards.recognition.dinov3_embedder import DINOv3Embedder, get_physical_cpu_cores


def test_get_physical_cpu_cores():
    cores = get_physical_cpu_cores()
    assert isinstance(cores, int)
    assert cores >= 1


def test_dinov3_embed_image():
    embedder = DINOv3Embedder()
    dummy_img = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)

    vec = embedder.embed_image(dummy_img)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # Vector should be unit length
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_dinov3_embed_batch():
    embedder = DINOv3Embedder()
    imgs = [
        np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
        np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
    ]

    batch_vecs = embedder.embed_batch(imgs)
    assert batch_vecs.shape == (2, 384)
    for vec in batch_vecs:
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)
