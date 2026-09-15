"""Unit tests for end-to-end FGIR recognition pipeline."""

import os
import pytest
from delta_cards.recognition.pipeline import CardRecognitionPipeline


@pytest.mark.asyncio
async def test_pipeline_fgir_identification():
    pipeline = CardRecognitionPipeline()
    assert len(pipeline.vector_index) > 0

    # Pick an indexed card image
    sample_img = "data/evaluation_set/images/sample_015_spanish_xy6-075.jpg"
    if not os.path.exists(sample_img):
        for root, _, files in os.walk("data/evaluation_set/images"):
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

    assert result["status"] in ("success", "rejected")
    assert result["extraction_method"] in ("full_ocr_text", "rejected")
    assert "profiling" in result
    assert "total_s" in result["profiling"]
    assert "debug_trace" in result
    assert "query_full_text" in result["debug_trace"]
