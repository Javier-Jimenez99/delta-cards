"""Global feature extraction using Meta DINOv3 ViT-Small with Dynamic INT8 Quantization on CPU."""

import logging
import os
import shutil
from typing import List, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_INT8_MODEL_FILENAME = "dinov3_vits16_int8.onnx"
DEFAULT_FP32_MODEL_FILENAME = "dinov3_vits16_fp32.onnx"
EMBEDDING_DIM = 384


def get_physical_cpu_cores() -> int:
    """Returns the number of physical CPU cores, excluding hyperthreading."""
    try:
        # Check /proc/cpuinfo for physical cores on Linux
        core_ids = set()
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.strip().startswith("core id"):
                    core_ids.add(line.strip().split(":")[1].strip())
        if core_ids:
            return len(core_ids)
    except Exception:
        pass

    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if count:
            return count
    except Exception:
        pass

    logical = os.cpu_count() or 4
    # On hyperthreaded CPUs, physical is typically logical // 2
    return max(1, logical // 2 if logical > 1 else 1)


def ensure_dinov3_onnx_model(
    model_path: Optional[str] = None,
    repo_id: str = DEFAULT_MODEL_REPO,
) -> str:
    """Ensures dynamic INT8 quantized DINOv3 ONNX model exists, generating or downloading if needed."""
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

    logger.info(f"Exporting DINOv3 INT8 ONNX model from '{repo_id}' to '{dest_path}'...")
    _export_and_quantize_dinov3(repo_id, dest_path)
    return dest_path


def _export_and_quantize_dinov3(repo_id: str, dest_int8_path: str) -> None:
    """Exports DINOv3 ViT-Small to ONNX opset 18 and applies Dynamic INT8 Quantization."""
    import torch
    from transformers import AutoModel
    from onnxruntime.quantization import quantize_dynamic, QuantType

    class DINOv3EmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            # Token 0 is CLS token (384-dim)
            cls_embed = outputs.last_hidden_state[:, 0]
            norm = torch.norm(cls_embed, p=2, dim=-1, keepdim=True)
            return cls_embed / (norm + 1e-12)

    logger.info(f"Loading base PyTorch model '{repo_id}'...")
    base_model = AutoModel.from_pretrained(repo_id)
    base_model.eval()
    wrapper = DINOv3EmbeddingWrapper(base_model)
    wrapper.eval()

    tmp_dir = os.path.join(os.path.dirname(dest_int8_path), ".export_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    fp32_path = os.path.join(tmp_dir, DEFAULT_FP32_MODEL_FILENAME)

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    logger.info("Tracing and exporting FP32 ONNX graph (opset 18)...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        fp32_path,
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=18,
    )

    logger.info("Applying Dynamic INT8 Quantization...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=dest_int8_path,
        weight_type=QuantType.QInt8,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Successfully generated INT8 DINOv3 model: {dest_int8_path} ({os.path.getsize(dest_int8_path)} bytes)")


class DINOv3Embedder:
    """CPU-optimized feature extractor generating 384-dimensional DINOv3 dense embeddings."""

    # ImageNet mean & std for channels-first RGB
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        """Initializes the DINOv3 ONNX Runtime session with strict physical CPU core threading."""
        self.model_path = ensure_dinov3_onnx_model(model_path)
        self.physical_cores = num_threads or get_physical_cpu_cores()
        logger.info(f"Initializing DINOv3 INT8 Embedder on CPU ({self.physical_cores} physical cores)...")

        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        # Strictly configure intra_op_num_threads to physical cores (disabling hyperthreading overhead)
        sess_opts.intra_op_num_threads = self.physical_cores
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """Prepares an image for DINOv3 ViT: BGR -> RGB, resize 224x224, ImageNet normalization.

        Returns:
            NumPy array with shape (1, 3, 224, 224), float32.
        """
        img = self._load_image(image)

        # Convert to RGB
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Transpose HWC -> CHW, scale to [0, 1]
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        # Normalize with ImageNet mean and std
        tensor = (tensor - self.MEAN) / self.STD
        return tensor.astype(np.float32)

    def embed_image(self, image: Union[np.ndarray, str, bytes]) -> np.ndarray:
        """Extracts a 384-dimensional unit-length L2-normalized embedding vector.

        Returns:
            NumPy array of shape (384,), float32.
        """
        tensor = self.preprocess_image(image)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        embedding = outputs[0][0]  # shape (384,)

        # Guarantee unit norm
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def embed_batch(self, images: List[Union[np.ndarray, str, bytes]]) -> np.ndarray:
        """Batch extraction of embeddings for offline indexing.

        Returns:
            NumPy array of shape (N, 384), float32.
        """
        if not images:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        tensors = [self.preprocess_image(img) for img in images]
        batch = np.concatenate(tensors, axis=0)
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        embeddings = outputs[0]  # shape (N, 384)

        # Normalize each row
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (embeddings / norms).astype(np.float32)

    def _load_image(self, image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Loads image into BGR NumPy array."""
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
