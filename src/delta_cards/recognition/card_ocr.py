"""Full-card OCR extraction and language detection."""

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

@dataclass
class CardOCRResult:
    """The complete OCR text and its language."""

    full_text: str
    detected_language: str = "en"


class CardOCR:
    """Extracts complete card text with RapidOCR."""

    def __init__(self, limit_side_len: int = 960):
        """Initializes the RapidOCR engine."""
        from rapidocr_onnxruntime import RapidOCR

        models_dir = os.path.join(os.getcwd(), "models", "ocr")
        rec_model = os.path.join(models_dir, "japan_PP-OCRv4_rec.onnx")
        rec_keys = os.path.join(models_dir, "japan_dict.txt")

        # Auto-download multilingual model if not present
        if not (os.path.isfile(rec_model) and os.path.isfile(rec_keys)):
            os.makedirs(models_dir, exist_ok=True)
            import urllib.request
            try:
                if not os.path.isfile(rec_keys):
                    urllib.request.urlretrieve(
                        "https://huggingface.co/cycloneboy/japan_PP-OCRv4_rec_infer/resolve/main/japan_dict.txt",
                        rec_keys,
                    )
                if not os.path.isfile(rec_model):
                    urllib.request.urlretrieve(
                        "https://huggingface.co/cycloneboy/japan_PP-OCRv4_rec_infer/resolve/main/model.onnx",
                        rec_model,
                    )
            except Exception:
                pass

        rapid_kwargs = {
            "use_cls": False,
            "det_limit_side_len": limit_side_len,
        }
        if os.path.isfile(rec_model) and os.path.isfile(rec_keys):
            rapid_kwargs["rec_model_path"] = rec_model
            rapid_kwargs["rec_keys_path"] = rec_keys

        self.ocr = RapidOCR(**rapid_kwargs)

    def extract(self, card_image: np.ndarray) -> CardOCRResult:
        """Returns only the full text produced by OCR."""
        if card_image is None or card_image.size == 0:
            return CardOCRResult(full_text="")

        results, _ = self.ocr(card_image)
        if not results:
            return CardOCRResult(full_text="")
        full_text = " ".join(str(item[1]).strip() for item in results if len(item) > 1 and item[1]).strip()
        return CardOCRResult(full_text=full_text, detected_language=self._detect_language(full_text))

    def _detect_language(self, full_text: str, *args: Any, **kwargs: Any) -> str:
        """Detects language exclusively with langdetect."""
        try:
            from langdetect import DetectorFactory, detect
            DetectorFactory.seed = 0
            return detect(full_text) if full_text.strip() else "en"
        except Exception:
            return "en"
