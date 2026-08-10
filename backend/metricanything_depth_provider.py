from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


METRICANYTHING_MODEL_ID = "yjh001/metricanything_student_pointmap"
METRICANYTHING_MODEL_REVISION = "8f1f08c53c683d4e19864601dfaf78515f29a63b"
METRICANYTHING_CHECKPOINT_SHA256 = (
    "f3b49bf78bc2699e79a7c8a53498d4520b30d2c1620c5c4f6f02cc7e94229d58"
)
METRICANYTHING_SOURCE_REPOSITORY = "https://github.com/metric-anything/metric-anything"
METRICANYTHING_SOURCE_COMMIT = "616a5e6762f5fc40d1a4ef990fee04c800532f59"
METRICANYTHING_LICENSE = "Apache-2.0"
METRICANYTHING_RESOLUTION_LEVEL = 9
METRICANYTHING_DEPTHMAP_MODEL_ID = "yjh001/metricanything_student_depthmap"
METRICANYTHING_DEPTHMAP_MODEL_REVISION = "cae9b4eb052e827048c9b385366c6d0dce83fb01"
METRICANYTHING_DEPTHMAP_CHECKPOINT_SHA256 = (
    "4553a58de46664c26616829183cfb38db09e986d706c4c43e06f6a37e2bebddc"
)

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_DEPTHMAP_LOAD_LOCK = threading.Lock()


def _default_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "deps" / "metric-anything"


def _resolve_source_dir(source_dir: str | Path | None = None) -> Path:
    configured = source_dir or os.environ.get("METRICANYTHING_DIR") or _default_source_dir()
    return Path(configured).expanduser().resolve()


def _resolve_checkpoint_path(
    source_dir: Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    configured = checkpoint_path or os.environ.get("METRICANYTHING_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    return source_dir / "checkpoints" / "student_pointmap" / "student_pointmap.pt"


def _source_provenance(source_dir: Path) -> dict:
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"MetricAnything source is not a git checkout: {source_dir}")
    try:
        revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "-C", str(source_dir), "status", "--short", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to validate the MetricAnything source checkout") from exc
    if revision != METRICANYTHING_SOURCE_COMMIT:
        raise RuntimeError(
            "MetricAnything source checkout does not match the pinned official commit"
        )
    if tracked_status:
        raise RuntimeError("MetricAnything source checkout has modified tracked files")
    return {
        "repository_path": str(source_dir),
        "revision": revision,
        "clean_tracked_files": True,
        "revision_verified": True,
    }


def _checkpoint_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"MetricAnything checkpoint not found: {path}")
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _CHECKPOINT_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CHECKPOINT_HASH_CACHE.clear()
    _CHECKPOINT_HASH_CACHE[key] = value
    return value


def _resolve_cuda_device(torch_module, requested: str | int) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        normalized = "cuda"
    if normalized in {"cuda", "0"}:
        normalized = "cuda:0"
    elif normalized.isdigit():
        normalized = f"cuda:{normalized}"
    if not normalized.startswith("cuda"):
        raise RuntimeError("MetricAnything Student PointMap requires CUDA")
    if not torch_module.cuda.is_available():
        raise RuntimeError("MetricAnything requested CUDA, but CUDA is unavailable")
    return normalized


def _load_model_class(source_dir: Path):
    package_root = (source_dir / "models" / "student_pointmap").resolve()
    package_text = str(package_root)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    try:
        module = importlib.import_module("moge.model.v2")
    except ImportError as exc:
        raise RuntimeError(
            "MetricAnything requires the pinned Student PointMap source and dependencies"
        ) from exc
    module_path = Path(module.__file__).resolve()
    if package_root not in module_path.parents:
        raise RuntimeError(
            "A different moge package is already loaded; refusing an ambiguous MetricAnything run"
        )
    return module.MoGeModel


def _load_depthmap_model_class(source_dir: Path):
    package_root = (source_dir / "models" / "student_depthmap").resolve()
    package_text = str(package_root)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    try:
        module = importlib.import_module("depth_model")
    except ImportError as exc:
        raise RuntimeError(
            "MetricAnything DepthMap requires the pinned official source and dependencies"
        ) from exc
    module_path = Path(module.__file__).resolve()
    if package_root not in module_path.parents:
        raise RuntimeError("A different depth_model module is already loaded")
    return module.MetricAnythingDepthMap, package_root


def release_metricanything_models() -> None:
    _MODEL_CACHE.clear()


def infer_metricanything_depth(
    image: Image.Image | str | Path,
    *,
    device: str | int = "auto",
    resolution_level: int = METRICANYTHING_RESOLUTION_LEVEL,
    source_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    if not 0 <= int(resolution_level) <= 9:
        raise ValueError("MetricAnything resolution level must be between 0 and 9")
    resolved_source = _resolve_source_dir(source_dir)
    source = _source_provenance(resolved_source)
    resolved_checkpoint = _resolve_checkpoint_path(resolved_source, checkpoint_path)
    checkpoint_sha256 = _checkpoint_sha256(resolved_checkpoint)
    if checkpoint_sha256 != METRICANYTHING_CHECKPOINT_SHA256:
        raise RuntimeError("MetricAnything checkpoint does not match the pinned public artifact")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MetricAnything requires PyTorch with CUDA support") from exc
    resolved_device = _resolve_cuda_device(torch, device)
    model_class = _load_model_class(resolved_source)

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
    image_array = np.array(pil_image, dtype=np.float32, copy=True) / 255.0
    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .to(resolved_device)
    )

    cuda_device = torch.device(resolved_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    cache_key = (str(resolved_checkpoint), resolved_device)
    load_started = time.perf_counter()
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = (
            model_class.from_pretrained(str(resolved_checkpoint))
            .to(resolved_device)
            .eval()
        )
    model = _MODEL_CACHE[cache_key]
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    output = model.infer(
        image_tensor,
        resolution_level=int(resolution_level),
        apply_mask=False,
        use_fp16=True,
    )
    torch.cuda.synchronize(cuda_device)
    inference_seconds = time.perf_counter() - inference_started

    depth = output["depth"].detach().float().cpu().numpy().astype(np.float32, copy=False)
    mask = output.get("mask")
    intrinsics = output.get("intrinsics")
    if depth.ndim != 2 or depth.size == 0 or not np.all(np.isfinite(depth)):
        raise RuntimeError("MetricAnything returned an invalid metric-depth surface")
    if float(np.min(depth)) <= 0.0 or float(np.ptp(depth)) <= 1e-8:
        raise RuntimeError("MetricAnything returned non-positive or constant metric depth")

    mask_fraction = (
        float(mask.detach().float().mean().item()) if mask is not None else None
    )
    focal_normalized = None
    if intrinsics is not None:
        intrinsic_values = intrinsics.detach().float().cpu().numpy()
        focal_normalized = float(
            (intrinsic_values[0, 0] + intrinsic_values[1, 1]) * 0.5
        )
    metadata = {
        "provider": "metricanything-student-pointmap",
        "model": METRICANYTHING_MODEL_ID,
        "model_revision": METRICANYTHING_MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "source_repository": METRICANYTHING_SOURCE_REPOSITORY,
        "source_commit": METRICANYTHING_SOURCE_COMMIT,
        "source_checkout": source,
        "license": METRICANYTHING_LICENSE,
        "depth_value_semantics": "metric_distance_far_high",
        "relief_value_transform": "inverse-depth",
        "external_scale_model_used": False,
        "predicted_mask_applied": False,
        "predicted_mask_fraction": mask_fraction,
        "resolution_level": int(resolution_level),
        "input_shape": [int(image_array.shape[0]), int(image_array.shape[1])],
        "device": resolved_device,
        "torch_version": str(torch.__version__),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(cuda_device) / (1024**3)),
        "depth": {
            "minimum": float(np.min(depth)),
            "median": float(np.median(depth)),
            "maximum": float(np.max(depth)),
        },
        "intrinsics": {
            "available": intrinsics is not None,
            "normalized_focal": focal_normalized,
        },
    }
    return depth, metadata


def infer_metricanything_depthmap(
    image: Image.Image | str | Path,
    *,
    device: str | int = "auto",
    source_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    resolved_source = _resolve_source_dir(source_dir)
    source = _source_provenance(resolved_source)
    resolved_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else resolved_source
        / "checkpoints"
        / "student_depthmap"
        / "student_depthmap.pt"
    )
    checkpoint_sha256 = _checkpoint_sha256(resolved_checkpoint)
    if checkpoint_sha256 != METRICANYTHING_DEPTHMAP_CHECKPOINT_SHA256:
        raise RuntimeError(
            "MetricAnything DepthMap checkpoint does not match the pinned public artifact"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MetricAnything DepthMap requires PyTorch with CUDA support") from exc
    resolved_device = _resolve_cuda_device(torch, device)
    model_class, package_root = _load_depthmap_model_class(resolved_source)

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
    image_array = np.array(pil_image, dtype=np.float32, copy=True) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    image_tensor = ((image_tensor - mean) / std).to(resolved_device)

    cuda_device = torch.device(resolved_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    cache_key = (str(resolved_checkpoint), f"depthmap:{resolved_device}")
    load_started = time.perf_counter()
    if cache_key not in _MODEL_CACHE:
        # The official factory resolves its local torch.hub checkout from cwd.
        with _DEPTHMAP_LOAD_LOCK:
            previous_cwd = Path.cwd()
            try:
                os.chdir(package_root)
                _MODEL_CACHE[cache_key] = (
                    model_class.from_pretrained(str(resolved_checkpoint))
                    .to(resolved_device)
                    .eval()
                )
            finally:
                os.chdir(previous_cwd)
    model = _MODEL_CACHE[cache_key]
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    output = model.infer(image_tensor, f_px=float(pil_image.width))
    torch.cuda.synchronize(cuda_device)
    inference_seconds = time.perf_counter() - inference_started
    depth = output["depth"].detach().float().cpu().numpy().astype(np.float32, copy=False)
    if depth.ndim != 2 or depth.size == 0 or not np.all(np.isfinite(depth)):
        raise RuntimeError("MetricAnything DepthMap returned an invalid metric-depth surface")
    if float(np.min(depth)) <= 0.0 or float(np.ptp(depth)) <= 1e-8:
        raise RuntimeError("MetricAnything DepthMap returned non-positive or constant depth")

    metadata = {
        "provider": "metricanything-student-depthmap",
        "model": METRICANYTHING_DEPTHMAP_MODEL_ID,
        "model_revision": METRICANYTHING_DEPTHMAP_MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "source_repository": METRICANYTHING_SOURCE_REPOSITORY,
        "source_commit": METRICANYTHING_SOURCE_COMMIT,
        "source_checkout": source,
        "license": METRICANYTHING_LICENSE,
        "depth_value_semantics": "metric_distance_far_high",
        "relief_value_transform": "inverse-depth",
        "focal_length_source": "official image-width fallback",
        "focal_px": float(pil_image.width),
        "external_scale_model_used": False,
        "input_shape": [int(image_array.shape[0]), int(image_array.shape[1])],
        "device": resolved_device,
        "torch_version": str(torch.__version__),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(cuda_device) / (1024**3)),
        "depth": {
            "minimum": float(np.min(depth)),
            "median": float(np.median(depth)),
            "maximum": float(np.max(depth)),
        },
    }
    return depth, metadata
