"""Locate and crop the main card in an image."""

import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "riftbound_regions.onnx"


class CardNotDetectedError(Exception):
    """Raised when no card is found."""


def ensure_card_isolator_model(model_path: Optional[str] = None) -> str:
    if model_path and os.path.isfile(model_path):
        return model_path
    local_path = os.path.join(os.getcwd(), "models", DEFAULT_MODEL)
    if os.path.isfile(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="Adrihp06/TCGscanner-detector", filename=DEFAULT_MODEL)


class CardIsolator:
    """YOLO-backed card cropper."""

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model = YOLO(ensure_card_isolator_model(model_path), task="detect")

    def load_image(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(image_source, np.ndarray):
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR) if image_source.ndim == 2 else image_source.copy()
        if isinstance(image_source, bytes):
            image = cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(image_source, timeout=10) as response:
                image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not load card image")
        return image

    def isolate_card(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        image, _ = self.isolate_card_with_debug(image_source)
        return image

    def isolate_card_with_debug(self, image_source: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        image = self.load_image(image_source)
        result = self.model(image, conf=self.conf_threshold, verbose=False)[0]
        if result.boxes is None or not len(result.boxes):
            raise CardNotDetectedError("No trading card detected")
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        x1, y1, x2, y2 = boxes[int(np.argmax(scores))].round().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            raise CardNotDetectedError("Invalid card bounds")
        annotated = image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return image[y1:y2, x1:x2].copy(), annotated


def isolate_card(image_source: Union[str, bytes, np.ndarray], model_path: Optional[str] = None) -> np.ndarray:
    return CardIsolator(model_path=model_path).isolate_card(image_source)