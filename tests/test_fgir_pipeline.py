"""Unit tests for end-to-end FGIR recognition pipeline."""

import os
import pytest
from delta_cards.recognition.pipeline import CardRecognitionPipeline


@pytest.mark.asyncio
async def test_pipeline_fgir_identification():
    pipeline = CardRecognitionPipeline()
    assert pipeline.use_fgir is True
    assert len(pipeline.vector_index) > 0

    eval_images_dir = (
        "data/tests/evaluation_set/images"
        if os.path.exists("data/tests/evaluation_set/images")
        else "data/evaluation_set/images"
    )
    sample_img = os.path.join(eval_images_dir, "sample_015_spanish_xy6-075.jpg")
    if not os.path.exists(sample_img):
        for root, _, files in os.walk(eval_images_dir):
            for f in sorted(files):
                if f.endswith((".png", ".jpg", ".webp")) and "fake" not in f:
                    sample_img = os.path.join(root, f)
                    break
            if sample_img:
                break
    assert os.path.exists(sample_img)

    result = await pipeline.process_card_listing(
        image_source=sample_img,
        title="Pokemon Card For Sale",
        description="Authentic Pokemon trading card",
    )

    assert result["status"] == "success"
    assert result["extraction_method"].startswith("fgir_")
    assert result["verified_card"] is not None
    assert "id" in result["verified_card"]
    assert "profiling" in result
    assert "search_s" in result["profiling"]
    assert "embedding_s" in result["profiling"]
    assert "debug_trace" in result
    assert "queried_index_language" in result["debug_trace"]
    assert "detected_language" in result["debug_trace"]
