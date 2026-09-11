"""Visual card isolation and deskewing using ONNX YOLO Oriented Bounding Box (OBB) and OpenCV."""

import logging
import os
import shutil
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ONNX_REPO_ID = "AlecKarfonta/cardcaptor-v3"
DEFAULT_ONNX_FILENAME = "onnx/cardcaptor_v3_best.onnx"


class CardNotDetectedError(Exception):
    """Raised when no trading card can be detected with sufficient confidence."""


def ensure_cardcaptor_onnx_model(model_path: Optional[str] = None) -> str:
    """Ensures cardcaptor_v3_best.onnx exists locally, downloading from HF if necessary."""
    if model_path and os.path.isfile(model_path):
        return model_path

    candidate_paths = [
        os.path.join(os.getcwd(), "onnx", "cardcaptor_v3_best.onnx"),
        os.path.join(os.getcwd(), "data", "models", "onnx", "cardcaptor_v3_best.onnx"),
        os.path.join(os.getcwd(), "data", "models", "cardcaptor_v3_best.onnx"),
        os.path.expanduser("~/.cache/delta_cards/models/cardcaptor_v3_best.onnx"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 10_000_000:
            return p

    dest_dir = os.path.join(os.getcwd(), "onnx")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "cardcaptor_v3_best.onnx")

    logger.info(f"Downloading {DEFAULT_ONNX_FILENAME} from HuggingFace ({DEFAULT_ONNX_REPO_ID})...")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=DEFAULT_ONNX_REPO_ID,
        filename=DEFAULT_ONNX_FILENAME,
    )
    shutil.copyfile(downloaded, dest_path)
    return dest_path


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points clockwise: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: NumPy array of shape (4, 2) containing (x, y) coordinates.

    Returns:
        NumPy array of shape (4, 2) in [TL, TR, BR, BL] order.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: Top-left has smallest sum (x+y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference (y - x): Top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


class VisualCardDetector:
    """Card detector and perspective rectification engine optimized for CPU execution.

    Uses an optimized ONNX YOLOv11m OBB model (cardcaptor_v3_best.onnx) running via
    onnxruntime with CPUExecutionProvider and multi-threading for maximum CPU performance,
    with backwards compatibility for PyTorch (.pt) weights.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        target_size: Tuple[int, int] = (750, 1050),
    ):
        """Initializes the visual detector.

        Args:
            model_path: Path to ONNX model (default: onnx/cardcaptor_v3_best.onnx) or PyTorch .pt model.
            conf_threshold: Minimum detection confidence threshold (0.0 - 1.0).
            target_size: Output resolution tuple (width, height) matching ~2.5x3.5 ratio.
        """
        self.conf_threshold = conf_threshold
        self.target_size = target_size

        if model_path is None:
            self.model_path = ensure_cardcaptor_onnx_model()
        else:
            self.model_path = model_path

        if self.model_path.endswith(".onnx"):
            self.is_onnx = True
            logger.info(f"Loading ONNX card detector from '{self.model_path}' on CPU...")
            import onnxruntime as ort

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = os.cpu_count() or 4
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            input_shape = self.session.get_inputs()[0].shape
            # Default to 1088 for cardcaptor_v3_best.onnx
            self.target_dim = (
                input_shape[2]
                if len(input_shape) >= 3 and isinstance(input_shape[2], int)
                else 1088
            )
            self.model = None
        else:
            self.is_onnx = False
            logger.info(f"Loading YOLO PyTorch model '{self.model_path}' on CPU...")
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self.session = None

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

    def _detect_onnx(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Runs ONNX inference on CPU and extracts rotated bounding box 4-corners."""
        h_orig, w_orig = image.shape[:2]
        target_dim = self.target_dim

        # Letterbox image with aspect ratio preservation
        scale = min(target_dim / h_orig, target_dim / w_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_img = np.full((target_dim, target_dim, 3), 114, dtype=np.uint8)
        px = (target_dim - nw) // 2
        py = (target_dim - nh) // 2
        pad_img[py : py + nh, px : px + nw] = resized

        # BGR -> RGB, float32, normalize [0, 1], NCHW
        blob = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self.session.run(None, {self.input_name: blob})
        dets = outputs[0][0]  # shape (300, 7): [cx, cy, w, h, conf, cls, angle]
        if len(dets) == 0:
            return None

        # Filter by confidence threshold and realistic trading card aspect ratio (~0.714)
        candidates = []
        for d in dets:
            cx, cy, w, h, conf, cls, angle = d
            if conf < self.conf_threshold or w <= 0 or h <= 0:
                continue
            ar = min(w, h) / max(w, h)
            # Trading cards have standard aspect ratio between 0.60 and 0.84
            if 0.58 <= ar <= 0.84:
                candidates.append((conf, d))

        if not candidates:
            # If no candidates meet strict aspect ratio, accept only high-confidence (>= 0.70)
            high_conf = [d for d in dets if d[4] >= 0.70 and d[2] > 0 and d[3] > 0]
            if not high_conf:
                return None
            best_det = max(high_conf, key=lambda x: x[4])
        else:
            best_det = max(candidates, key=lambda x: x[0])[1]

        cx, cy, w, h, conf, cls, angle = best_det

        # Reconstruct corners using standard YOLO OBB vector formulation
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        vec1 = np.array([w / 2.0 * cos_val, w / 2.0 * sin_val], dtype=np.float32)
        vec2 = np.array([-h / 2.0 * sin_val, h / 2.0 * cos_val], dtype=np.float32)
        ctr = np.array([cx, cy], dtype=np.float32)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        canvas_pts = np.stack([pt1, pt2, pt3, pt4], axis=0)

        # Unpad and scale back to original image coordinates
        orig_pts = (canvas_pts - np.array([px, py], dtype=np.float32)) / scale
        orig_pts[:, 0] = np.clip(orig_pts[:, 0], 0, w_orig)
        orig_pts[:, 1] = np.clip(orig_pts[:, 1], 0, h_orig)

        return orig_pts.astype(np.float32)

    def _extract_obb_corners(self, results) -> Optional[np.ndarray]:
        """Extracts 4-corner coordinates of the primary card detection from YOLO OBB PyTorch results."""
        if not results or len(results) == 0:
            return None

        res = results[0]
        obb = getattr(res, "obb", None)
        if obb is None or len(obb) == 0:
            return None

        # Filter by confidence threshold
        confs = obb.conf.cpu().numpy()
        valid_indices = np.where(confs >= self.conf_threshold)[0]
        if len(valid_indices) == 0:
            return None

        # Choose the detection with highest confidence
        best_idx = valid_indices[np.argmax(confs[valid_indices])]

        # xyxyxyxy shape: (N, 4, 2)
        if hasattr(obb, "xyxyxyxy"):
            corners = obb.xyxyxyxy[best_idx].cpu().numpy()
            if corners.shape == (4, 2):
                return corners.astype("float32")

        # Fallback to xywhr if xyxyxyxy is not directly present
        if hasattr(obb, "xywhr"):
            xywhr = obb.xywhr[best_idx].cpu().numpy()
            cx, cy, w, h, angle_rad = xywhr
            angle_deg = np.degrees(angle_rad)
            box = cv2.boxPoints(((cx, cy), (w, h), angle_deg))
            return np.array(box, dtype="float32")

        return None

    def _find_card_contour_fallback(self, image: np.ndarray) -> Optional[np.ndarray]:
        """High-precision OpenCV contour fallback to detect rectangular card quadrilateral.

        Used when model lacks a detection or for edge cases.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-scale edge detection
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        best_quad = None
        max_area = 0.0

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            area = cv2.contourArea(cnt)
            # A card must occupy a reasonable portion of marketplace image (>= 5% and <= 98%)
            if area < (0.05 * image_area) or area > (0.98 * image_area):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")
                # Validate aspect ratio (~0.5 to 0.88)
                rect = order_points(pts)
                width_a = np.linalg.norm(rect[1] - rect[0])
                width_b = np.linalg.norm(rect[2] - rect[3])
                max_w = max(width_a, width_b)

                height_a = np.linalg.norm(rect[3] - rect[0])
                height_b = np.linalg.norm(rect[2] - rect[1])
                max_h = max(height_a, height_b)

                if max_w > 0 and max_h > 0:
                    ar = min(max_w, max_h) / max(max_w, max_h)
                    # Standard Pokemon card ratio is 2.5 / 3.5 = ~0.714
                    if 0.50 <= ar <= 0.88 and area > max_area:
                        max_area = area
                        best_quad = pts

        return best_quad

    def rectify_card(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Applies perspective transformation to rectify and deskew the card upright.

        Args:
            image: Source BGR image.
            corners: (4, 2) array of quadrilateral corner coordinates.

        Returns:
            Upright rectified card image with target dimensions.
        """
        rect = order_points(corners)
        target_w, target_h = self.target_size

        # Compute widths and heights
        width_top = np.linalg.norm(rect[1] - rect[0])
        width_bottom = np.linalg.norm(rect[2] - rect[3])
        height_left = np.linalg.norm(rect[3] - rect[0])
        height_right = np.linalg.norm(rect[2] - rect[1])

        avg_w = (width_top + width_bottom) / 2.0
        avg_h = (height_left + height_right) / 2.0

        # Destination coordinates in standard portrait orientation
        dst = np.array(
            [
                [0, 0],
                [target_w - 1, 0],
                [target_w - 1, target_h - 1],
                [0, target_h - 1],
            ],
            dtype="float32",
        )

        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            image, m, (target_w, target_h), flags=cv2.INTER_LANCZOS4
        )

        # If detected region was sideways (landscape), rotate upright
        if avg_w > avg_h:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            warped = cv2.resize(
                warped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )

        return warped

    def isolate_card(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Isolates, crops, and deskews the primary card upright to standard aspect ratio.

        Args:
            image_source: Image filepath, raw image bytes, or NumPy BGR array.

        Returns:
            Rectified card image as a NumPy array (BGR).

        Raises:
            CardNotDetectedError: If no card is detected with sufficient confidence.
        """
        image = self._load_image(image_source)

        corners = None
        # 1. Primary detection via ONNX or PyTorch
        if self.is_onnx:
            try:
                corners = self._detect_onnx(image)
            except Exception as exc:
                logger.warning(
                    f"ONNX detector inference error: {exc}. Trying fallback detector."
                )
                corners = None
        else:
            try:
                results = self.model.predict(
                    image,
                    device="cpu",
                    verbose=False,
                    conf=self.conf_threshold,
                )
                corners = self._extract_obb_corners(results)
            except Exception as exc:
                logger.warning(
                    f"YOLO OBB inference error: {exc}. Trying fallback detector."
                )
                corners = None

        # 2. If no corners detected, try high-precision contour fallback
        if corners is None:
            corners = self._find_card_contour_fallback(image)

        # 3. If still no card detected, raise custom exception
        if corners is None:
            raise CardNotDetectedError(
                "No Pokémon card detected with sufficient confidence in the provided image."
            )

        # 4. Rectify and deskew upright
        rectified = self.rectify_card(image, corners)
        return rectified


def isolate_card(
    image_source: Union[str, bytes, np.ndarray],
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Convenience function to isolate a card using the VisualCardDetector."""
    detector = VisualCardDetector(model_path=model_path, conf_threshold=conf_threshold)
    return detector.isolate_card(image_source)
