"""Visual card boundary isolation using Adrihp06/TCGscanner-detector (ONNX YOLO).

Performs direct, fast rectangular cropping of the physical card with zero complex
rotation or perspective dewarping, feeding directly into the DINOv3 + USearch visual pipeline.
"""

import logging
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_DETECTOR_REPO_ID = "Adrihp06/TCGscanner-detector"
DEFAULT_ONNX_FILENAME = "riftbound_regions.onnx"


class CardNotDetectedError(Exception):
    """Raised when no trading card can be detected with sufficient confidence."""


def ensure_detector_model(model_path: Optional[str] = None) -> str:
    """Ensures the ONNX Adrihp06 detector model exists locally.

    Returns:
        Absolute path to the local ONNX model file.
    """
    if model_path and os.path.isfile(model_path):
        return model_path

    # Candidate paths in project and cache
    candidate_paths = [
        os.path.join(os.getcwd(), "models", "riftbound_regions.onnx"),
        os.path.expanduser("~/.cache/delta_cards/models/riftbound_regions.onnx"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
            return p

    # Download from HuggingFace
    dest_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "riftbound_regions.onnx")

    logger.info(f"Downloading {DEFAULT_ONNX_FILENAME} from HuggingFace ({DEFAULT_DETECTOR_REPO_ID})...")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=DEFAULT_DETECTOR_REPO_ID,
        filename=DEFAULT_ONNX_FILENAME,
    )
    import shutil
    shutil.copyfile(downloaded, dest_path)
    return dest_path


# Backward compatibility alias
ensure_cardcaptor_model = ensure_detector_model
ensure_cardcaptor_onnx_model = ensure_detector_model


class VisualCardDetector:
    """Card detector and direct rectangular cropper powered by Adrihp06/TCGscanner-detector.

    Localizes the primary trading card boundary on CPU in ~150-190ms and extracts a clean
    rectangular crop directly for downstream embedding retrieval (USearch), with no
    lossy or unstable perspective rotations.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
    ):
        """Initializes the Adrihp06 YOLO card detector.

        Args:
            model_path: Path to riftbound_regions.onnx (downloaded automatically if None).
            conf_threshold: Minimum detection confidence threshold.
        """
        self.conf_threshold = conf_threshold
        self.model_path = ensure_detector_model(model_path)
        logger.info(f"Loading Adrihp06 ONNX card detector from '{self.model_path}' on CPU...")
        self.model = YOLO(self.model_path, task="detect")

    def _load_image(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Loads and converts various input image formats into a BGR NumPy array."""
        if isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
            return image_source.copy()

        if isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from bytes.")
            return img

        if isinstance(image_source, str):
            if image_source.startswith(("http://", "https://")):
                import urllib.request

                req = urllib.request.Request(
                    image_source,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image from URL: {image_source}")
                return img

            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image path does not exist: {image_source}")
            img = cv2.imread(image_source, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image from file: {image_source}")
            return img

        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    def detect_card_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """Detects the primary card bounding box using Adrihp06 detector.

        Args:
            image: BGR NumPy array.

        Returns:
            Tuple of (x1, y1, x2, y2, confidence) clamped to image boundaries, or None.
        """
        h, w = image.shape[:2]
        results = self.model(image, conf=self.conf_threshold, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        # Extract all valid candidate boxes
        candidates = []
        for b in results.boxes:
            conf = float(b.conf[0]) if hasattr(b.conf, "__len__") else float(b.conf)
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            x1 = max(0, min(w - 1, int(round(xyxy[0]))))
            y1 = max(0, min(h - 1, int(round(xyxy[1]))))
            x2 = max(0, min(w, int(round(xyxy[2]))))
            y2 = max(0, min(h, int(round(xyxy[3]))))
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 10 or box_h <= 10:
                continue

            area = box_w * box_h
            candidates.append((conf, area, (x1, y1, x2, y2)))

        if not candidates:
            return None

        # Choose the best candidate: prefer high-confidence box, and if multiple, largest card area
        max_conf = max(c[0] for c in candidates)
        # Candidates within 0.15 of max confidence
        strong_candidates = [c for c in candidates if c[0] >= max_conf - 0.15]
        best_candidate = max(strong_candidates, key=lambda c: c[1])

        x1, y1, x2, y2 = best_candidate[2]
        return (x1, y1, x2, y2, best_candidate[0])

    def isolate_card(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Isolates and crops the card directly without rotation or perspective transforms.

        Args:
            image_source: Image path, bytes, or BGR array.

        Returns:
            Cropped card image as a BGR NumPy array.

        Raises:
            CardNotDetectedError: If no card is detected with sufficient confidence.
        """
        image = self._load_image(image_source)
        bbox = self.detect_card_bbox(image)
        if bbox is None:
            raise CardNotDetectedError(
                f"No trading card detected with confidence >= {self.conf_threshold}"
            )

        x1, y1, x2, y2, _ = bbox
        card_crop = image[y1:y2, x1:x2].copy()
        return card_crop

    def isolate_card_with_debug(
        self, image_source: Union[str, bytes, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Isolates the card and produces an annotated original image for debugging/UI.

        Args:
            image_source: Image path, bytes, or BGR array.

        Returns:
            Tuple of (card_crop, annotated_original_image).
        """
        image = self._load_image(image_source)
        bbox = self.detect_card_bbox(image)
        if bbox is None:
            raise CardNotDetectedError(
                f"No trading card detected with confidence >= {self.conf_threshold}"
            )

        x1, y1, x2, y2, conf = bbox
        card_crop = image[y1:y2, x1:x2].copy()

        # Draw green bounding box on copy of original
        annotated = image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Card {conf:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return card_crop, annotated


def isolate_card(
    image_source: Union[str, bytes, np.ndarray],
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Convenience function to isolate a card using VisualCardDetector."""
    detector = VisualCardDetector(model_path=model_path, conf_threshold=conf_threshold)
    return detector.isolate_card(image_source)
