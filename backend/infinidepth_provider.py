from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


INFINIDEPTH_MODEL_ID = "ritianyu/InfiniDepth"
INFINIDEPTH_MODEL_REVISION = "b387ad877e922468fcd85190f24e5b78b28dcd66"
INFINIDEPTH_CHECKPOINT_SHA256 = (
    "6f123506b266f61b9261878548946cfb8822e6233aaf53527dfc4b3dab1e7f9c"
)
INFINIDEPTH_SOURCE_REPOSITORY = "https://github.com/zju3dv/InfiniDepth"
INFINIDEPTH_SOURCE_COMMIT = "36c6e0c31887fafc210184ee43ca475230704095"
INFINIDEPTH_LICENSE = "Apache-2.0"
INFINIDEPTH_PROCESS_LONG_SIDE = 512
INFINIDEPTH_QUERY_LONG_SIDE = 512

_MODEL_CACHE: dict[tuple[str, str], object] = {}
_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _default_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "deps" / "infinidepth"


def _resolve_source_dir(source_dir: str | Path | None = None) -> Path:
    configured = source_dir or os.environ.get("INFINIDEPTH_DIR") or _default_source_dir()
    return Path(configured).expanduser().resolve()


def _resolve_checkpoint_path(
    source_dir: Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    configured = checkpoint_path or os.environ.get("INFINIDEPTH_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()
    return source_dir / "checkpoints" / "depth" / "infinidepth.ckpt"


def _source_provenance(source_dir: Path) -> dict:
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"InfiniDepth source is not a git checkout: {source_dir}")
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
        raise RuntimeError("Unable to validate the InfiniDepth source checkout") from exc
    if revision != INFINIDEPTH_SOURCE_COMMIT:
        raise RuntimeError(
            "InfiniDepth source checkout does not match the pinned official commit"
        )
    if tracked_status:
        raise RuntimeError("InfiniDepth source checkout has modified tracked files")
    return {
        "repository_path": str(source_dir),
        "revision": revision,
        "clean_tracked_files": True,
        "revision_verified": True,
    }


def _checkpoint_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"InfiniDepth checkpoint not found: {path}")
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


def _scaled_shape(width: int, height: int, long_side: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("InfiniDepth input dimensions must be positive")
    if int(long_side) < 256:
        raise ValueError("InfiniDepth long-side resolution must be at least 256 pixels")
    scale = float(long_side) / float(max(width, height))
    scaled_width = max(16, int(round(width * scale / 16.0)) * 16)
    scaled_height = max(16, int(round(height * scale / 16.0)) * 16)
    return scaled_width, scaled_height


def _resolve_cuda_device(torch_module, requested: str | int) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        normalized = "cuda"
    if normalized in {"cuda", "0"}:
        normalized = "cuda:0"
    elif normalized.isdigit():
        normalized = f"cuda:{normalized}"
    if not normalized.startswith("cuda"):
        raise RuntimeError("InfiniDepth's official inference implementation requires CUDA")
    if not torch_module.cuda.is_available():
        raise RuntimeError("InfiniDepth requested CUDA, but CUDA is unavailable")
    return normalized


def release_infinidepth_models() -> None:
    _MODEL_CACHE.clear()


def infer_infinidepth_disparity(
    image: Image.Image | str | Path,
    *,
    device: str | int = "auto",
    process_long_side: int = INFINIDEPTH_PROCESS_LONG_SIDE,
    query_long_side: int = INFINIDEPTH_QUERY_LONG_SIDE,
    source_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    resolved_source = _resolve_source_dir(source_dir)
    source = _source_provenance(resolved_source)
    resolved_checkpoint = _resolve_checkpoint_path(resolved_source, checkpoint_path)
    checkpoint_sha256 = _checkpoint_sha256(resolved_checkpoint)
    if checkpoint_sha256 != INFINIDEPTH_CHECKPOINT_SHA256:
        raise RuntimeError("InfiniDepth checkpoint does not match the pinned public artifact")

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("InfiniDepth requires PyTorch with CUDA support") from exc
    resolved_device = _resolve_cuda_device(torch, device)

    source_text = str(resolved_source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from InfiniDepth.utils.model_utils import build_model
        from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS
    except ImportError as exc:
        raise RuntimeError(
            "InfiniDepth requires the pinned official source and its isolated dependencies"
        ) from exc

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
    original_width, original_height = pil_image.size
    process_width, process_height = _scaled_shape(
        original_width,
        original_height,
        int(process_long_side),
    )
    query_width, query_height = _scaled_shape(
        original_width,
        original_height,
        int(query_long_side),
    )
    resized = pil_image.resize((process_width, process_height), Image.Resampling.BICUBIC)
    image_array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(resolved_device)
    )

    cuda_device = torch.device(resolved_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    cache_key = (str(resolved_checkpoint), resolved_device)
    load_started = time.perf_counter()
    if cache_key not in _MODEL_CACHE:
        with torch.cuda.device(cuda_device):
            _MODEL_CACHE[cache_key] = (
                build_model(
                    "InfiniDepth",
                    model_path=str(resolved_checkpoint),
                )
                .to(resolved_device)
                .eval()
            )
    model = _MODEL_CACHE[cache_key]
    load_seconds = time.perf_counter() - load_started

    query = SAMPLING_METHODS["2d_uniform"]((query_height, query_width))
    query = query.unsqueeze(0).to(resolved_device)
    inference_started = time.perf_counter()
    with torch.cuda.device(cuda_device):
        predicted_depth, predicted_disparity = model.inference(
            image=image_tensor,
            query_coord=query,
        )
        torch.cuda.synchronize(cuda_device)
    inference_seconds = time.perf_counter() - inference_started

    disparity = (
        predicted_disparity.detach()
        .float()
        .reshape(1, query_height, query_width)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    depth = predicted_depth.detach().float()
    if disparity.ndim != 2 or disparity.size == 0 or not np.all(np.isfinite(disparity)):
        raise RuntimeError("InfiniDepth returned an invalid relative-disparity surface")
    if float(np.ptp(disparity)) <= 1e-8:
        raise RuntimeError("InfiniDepth returned a constant relative-disparity surface")

    metadata = {
        "provider": "infinidepth",
        "model": INFINIDEPTH_MODEL_ID,
        "model_revision": INFINIDEPTH_MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "source_repository": INFINIDEPTH_SOURCE_REPOSITORY,
        "source_commit": INFINIDEPTH_SOURCE_COMMIT,
        "source_checkout": source,
        "license": INFINIDEPTH_LICENSE,
        "depth_value_semantics": "relative_disparity_near_high",
        "relief_value_transform": "linear",
        "metric_scale_model_used": False,
        "process_shape": [process_height, process_width],
        "query_shape": [query_height, query_width],
        "aspect_ratio_preserved": True,
        "device": resolved_device,
        "torch_version": str(torch.__version__),
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(cuda_device) / (1024**3)),
        "disparity": {
            "minimum": float(np.min(disparity)),
            "median": float(np.median(disparity)),
            "maximum": float(np.max(disparity)),
            "negative_fraction": float(np.mean(disparity < 0.0)),
        },
        "official_depth_clamp_fraction": float(torch.mean((depth >= 199.999).float()).item()),
    }
    return disparity, metadata
