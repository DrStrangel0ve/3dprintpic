from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image


DAVID_SOURCE_REVISION = "20a3eb66f61d0e1caff42489c6775e142179bd46"
DAVID_LICENSE = "MIT"
DAVID_MODEL_URL = (
    "https://facesyntheticspubwedata.z6.web.core.windows.net/"
    "iccv-2025/models/depth-model-vitb16_384.onnx"
)
DAVID_MODEL_SHA256 = (
    "c8dc49821c95f7ab56a30150e58eba989b1d67ce988d11f083dc7350c54b883b"
)
DAVID_MODEL_BYTES = 449_231_953
DAVID_MODEL_MAX_BYTES = 512 * 1024 * 1024
DAVID_INPUT_SIZE = 512


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_david_depth_model() -> Path:
    configured = str(os.getenv("DAVID_DEPTH_MODEL_PATH") or "").strip()
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home()
        / ".cache"
        / "3dprintpic"
        / "david_depth_vitb16_384.onnx"
    )
    if path.is_file():
        size = path.stat().st_size
        checksum = _sha256_file(path)
        if size != DAVID_MODEL_BYTES:
            raise RuntimeError(
                f"DAViD model size mismatch: expected={DAVID_MODEL_BYTES}, "
                f"actual={size}"
            )
        if checksum != DAVID_MODEL_SHA256:
            raise RuntimeError(
                "DAViD model checksum mismatch: "
                f"expected={DAVID_MODEL_SHA256}, actual={checksum}"
            )
        return path
    if configured:
        raise FileNotFoundError("Configured DAViD model is unavailable")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".part",
            delete=False,
        ) as partial:
            partial_path = Path(partial.name)
            total_bytes = 0
            with urllib.request.urlopen(DAVID_MODEL_URL, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total_bytes += len(block)
                    if total_bytes > DAVID_MODEL_MAX_BYTES:
                        raise RuntimeError("DAViD model download exceeded size limit")
                    partial.write(block)
        size = partial_path.stat().st_size
        checksum = _sha256_file(partial_path)
        if size != DAVID_MODEL_BYTES or checksum != DAVID_MODEL_SHA256:
            raise RuntimeError(
                "DAViD model download verification failed: "
                f"size={size}, sha256={checksum}"
            )
        os.replace(partial_path, path)
        partial_path = None
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
    return path


def _prepare_bgr_input(
    image_rgb: np.ndarray,
    *,
    size: int = DAVID_INPUT_SIZE,
) -> tuple[np.ndarray, dict]:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("DAViD input must be an RGB image")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = np.asarray(image, dtype=np.float32)
        if float(np.nanmax(image)) > 1.0:
            image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    image = image[:, :, ::-1]
    height, width = image.shape[:2]
    side = max(height, width)
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_REPLICATE,
    )
    resized = cv2.resize(
        padded,
        (int(size), int(size)),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor = np.transpose(resized, (2, 0, 1))[None].astype(np.float32)
    return tensor, {
        "original_shape": [height, width],
        "square_size": side,
        "padding": [pad_top, pad_bottom, pad_left, pad_right],
    }


def _restore_dense_map(values: np.ndarray, metadata: dict) -> np.ndarray:
    values = np.squeeze(np.asarray(values, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError("DAViD output must reduce to a two-dimensional map")
    side = int(metadata["square_size"])
    square = cv2.resize(values, (side, side), interpolation=cv2.INTER_CUBIC)
    pad_top, pad_bottom, pad_left, pad_right = (
        int(value) for value in metadata["padding"]
    )
    y1 = side - pad_bottom if pad_bottom else side
    x1 = side - pad_right if pad_right else side
    restored = square[pad_top:y1, pad_left:x1]
    height, width = (int(value) for value in metadata["original_shape"])
    if restored.shape != (height, width):
        restored = cv2.resize(
            restored,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    return restored.astype(np.float32)


def _normalize_relative_depth(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("DAViD depth contains no finite values")
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-8:
        raise ValueError("DAViD depth has no usable relative span")
    normalized = (values - low) / (high - low)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("DAViD normalized depth contains non-finite values")
    return normalized.astype(np.float32)


def _device_free_memory_bytes(device: str) -> int | None:
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
        return int(free_bytes)
    except Exception:
        return None


def _create_session(model_path: Path, device: str):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    available = ort.get_available_providers()
    if str(device).startswith("cuda"):
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("DAViD requested CUDA but ONNX CUDA is unavailable")
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        requested = ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=requested,
    )
    active = session.get_providers()
    if str(device).startswith("cuda") and (
        not active or active[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError("DAViD ONNX session silently fell back from CUDA")
    return session, ort.__version__, list(available), list(active)


class DAViDFaceDepth:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "cuda",
    ) -> None:
        self.model_path = (
            Path(model_path) if model_path is not None else resolve_david_depth_model()
        )
        if _sha256_file(self.model_path) != DAVID_MODEL_SHA256:
            raise RuntimeError("DAViD model checksum mismatch")
        self.device = str(device)
        self._free_before_load = _device_free_memory_bytes(self.device)
        started = time.perf_counter()
        (
            self.session,
            self.onnxruntime_version,
            self.available_providers,
            self.active_providers,
        ) = _create_session(self.model_path, self.device)
        self.load_seconds = time.perf_counter() - started
        self.input_name = self.session.get_inputs()[0].name
        self.timings: list[float] = []
        self._free_after_first_inference = None
        self.last_inference: dict = {}

    def infer_array(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor, metadata = _prepare_bgr_input(image_rgb)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        self.timings.append(time.perf_counter() - started)
        if not outputs:
            raise RuntimeError("DAViD ONNX session returned no outputs")
        raw = _restore_dense_map(outputs[0], metadata)
        # 3dprintpic's DAv2 maps use the model's z-depth direction. The official
        # DAViD demo optionally negates this output for inverse-depth display,
        # which is the opposite of the fusion contract used here.
        normalized = _normalize_relative_depth(raw)
        if len(self.timings) == 1:
            self._free_after_first_inference = _device_free_memory_bytes(
                self.device
            )
        self.last_inference = {
            "input_shape": list(np.asarray(image_rgb).shape),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "normalized_min": float(np.min(normalized)),
            "normalized_max": float(np.max(normalized)),
            "output_convention": "raw-z-depth-minmax",
            "runtime_seconds": self.timings[-1],
        }
        return normalized

    def infer(self, image_path: str | Path) -> np.ndarray:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        return self.infer_array(image)

    def provenance(self) -> dict:
        memory_gib = None
        if (
            self._free_before_load is not None
            and self._free_after_first_inference is not None
        ):
            memory_gib = max(
                0.0,
                float(
                    self._free_before_load
                    - self._free_after_first_inference
                )
                / (1024**3),
            )
        return {
            "provider": "microsoft-david-base-relative-depth",
            "source_revision": DAVID_SOURCE_REVISION,
            "license": DAVID_LICENSE,
            "model_url": DAVID_MODEL_URL,
            "model_sha256": DAVID_MODEL_SHA256,
            "model_bytes": DAVID_MODEL_BYTES,
            "onnxruntime_version": self.onnxruntime_version,
            "available_execution_providers": self.available_providers,
            "active_execution_providers": self.active_providers,
            "device": self.device,
            "output_convention": "raw-z-depth-minmax",
            "load_seconds": self.load_seconds,
            "inference_calls": len(self.timings),
            "mean_inference_seconds": (
                float(np.mean(self.timings)) if self.timings else None
            ),
            "approximate_peak_device_memory_gib": memory_gib,
            "memory_measurement_scope": (
                "session-load-through-first-fixed-shape-inference"
            ),
        }
