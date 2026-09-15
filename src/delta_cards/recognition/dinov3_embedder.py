"""DINOv3 ONNX image embeddings."""

import os
from typing import List, Optional, Union

import cv2
import numpy as np

DEFAULT_INT8_MODEL_FILENAME = "dinov3_vits16_int8.onnx"
EMBEDDING_DIM = 384


def ensure_dinov3_onnx_model(model_path: Optional[str] = None) -> str:
    """Finds the local DINOv3 ONNX model."""
    if model_path and os.path.isfile(model_path):
        return model_path

    candidate_paths = [
        os.path.join(os.getcwd(), "models", DEFAULT_INT8_MODEL_FILENAME),
        os.path.join(os.getcwd(), "models", "onnx", DEFAULT_INT8_MODEL_FILENAME),
        os.path.join(os.getcwd(), "onnx", DEFAULT_INT8_MODEL_FILENAME),
        os.path.expanduser(f"~/.cache/delta_cards/models/{DEFAULT_INT8_MODEL_FILENAME}"),
    ]
    for p in candidate_paths:
        if os.path.isfile(p) and os.path.getsize(p) > 5_000_000:
            return p

    dest_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, DEFAULT_INT8_MODEL_FILENAME)

    raise FileNotFoundError(f"DINOv3 ONNX model not found: {dest_path}")


class DINOv3Embedder:
    """Generates normalized 384-dimensional image embeddings."""

    # ImageNet mean & std for channels-first RGB
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        """Loads the local ONNX model."""
        self.model_path = ensure_dinov3_onnx_model(model_path)
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        if num_threads:
            sess_opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """Converts a card image to DINOv3 input."""
        img = self._load_image(image)

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]
        tensor = (tensor - self.MEAN) / self.STD
        return tensor.astype(np.float32)

    def embed_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """Returns one normalized visual embedding."""
        embedding = self.session.run([self.output_name], {self.input_name: self.preprocess_image(image)})[0][0]
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def embed_batch(self, images: List[Union[np.ndarray, str, bytes]]) -> np.ndarray:
        """Returns embeddings for several images."""
        if not images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        tensors = [self.preprocess_image(img) for img in images]
        batch = np.concatenate(tensors, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        embeddings = outputs[0]  # shape (N, 384)

        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (embeddings / norms).astype(np.float32)

    def _load_image(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Loads an image into BGR format."""
        if isinstance(image_source, np.ndarray):
            return image_source

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

        raise TypeError(f"Unsupported image type: {type(image_source)}")
