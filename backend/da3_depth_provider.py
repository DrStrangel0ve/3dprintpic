from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np


DA3_LARGE_MODEL_ID = "depth-anything/DA3-LARGE-1.1"
DA3_LARGE_MODEL_REVISION = "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
DA3_MONO_MODEL_ID = "depth-anything/DA3MONO-LARGE"
DA3_MONO_MODEL_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
DA3_SOURCE_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3"
DA3_SOURCE_COMMIT = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
DA3_PROCESS_RESOLUTION = 504
DA3_MODEL_SPECS = {
    DA3_LARGE_MODEL_ID: {
        "revision": DA3_LARGE_MODEL_REVISION,
        "license": "CC BY-NC 4.0",
        "role": "any-view relative depth",
    },
    DA3_MONO_MODEL_ID: {
        "revision": DA3_MONO_MODEL_REVISION,
        "license": "Apache-2.0",
        "role": "monocular relative depth",
    },
}
DA3_MODEL_IDS = frozenset(DA3_MODEL_SPECS)

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def is_da3_model(model_id: str | None) -> bool:
    return str(model_id or "").strip() in DA3_MODEL_IDS


def _resolved_device(torch_module, requested: str | int) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized in {"cuda", "0"}:
        if not torch_module.cuda.is_available():
            raise RuntimeError("Depth Anything V3 Large requested CUDA, but CUDA is unavailable")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda:"):
        if not torch_module.cuda.is_available():
            raise RuntimeError("Depth Anything V3 Large requested CUDA, but CUDA is unavailable")
        return normalized
    raise ValueError(f"Unsupported Depth Anything V3 device: {requested}")


def _source_provenance(module_file: str | Path) -> dict:
    module_path = Path(module_file).resolve()
    for parent in module_path.parents:
        if not (parent / ".git").exists():
            continue
        try:
            revision = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(parent), "status", "--short"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            break
        if revision != DA3_SOURCE_COMMIT:
            raise RuntimeError(
                "Depth Anything V3 source checkout does not match the pinned official commit"
            )
        return {
            "repository_path": str(parent),
            "revision": revision,
            "clean": not bool(status),
            "revision_verified": True,
        }
    return {
        "repository_path": None,
        "revision": DA3_SOURCE_COMMIT,
        "clean": None,
        "revision_verified": False,
    }


def release_da3_models() -> None:
    _MODEL_CACHE.clear()


def infer_da3_depth(
    image,
    *,
    model_id: str = DA3_LARGE_MODEL_ID,
    device: str | int = "auto",
    process_resolution: int = DA3_PROCESS_RESOLUTION,
) -> tuple[np.ndarray, dict]:
    if not is_da3_model(model_id):
        raise ValueError(f"Unsupported Depth Anything V3 model: {model_id}")
    if int(process_resolution) < 196:
        raise ValueError("Depth Anything V3 process resolution must be at least 196 pixels")

    try:
        import torch
        import depth_anything_3.api as da3_api
        from depth_anything_3.api import DepthAnything3
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything V3 Large requires the pinned official depth-anything-3 package"
        ) from exc

    resolved_device = _resolved_device(torch, device)
    source = _source_provenance(da3_api.__file__)
    model_spec = DA3_MODEL_SPECS[model_id]
    snapshot = snapshot_download(
        model_id,
        revision=model_spec["revision"],
        allow_patterns=("config.json", "model.safetensors"),
    )

    cache_key = (str(Path(snapshot).resolve()), resolved_device)
    load_started = time.perf_counter()
    if cache_key not in _MODEL_CACHE:
        model = DepthAnything3.from_pretrained(snapshot).to(resolved_device)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    model = _MODEL_CACHE[cache_key]
    load_seconds = time.perf_counter() - load_started

    cuda_device = None
    if resolved_device.startswith("cuda"):
        cuda_device = torch.device(resolved_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)

    inference_started = time.perf_counter()
    prediction = model.inference(
        [image],
        process_res=int(process_resolution),
        process_res_method="upper_bound_resize",
    )
    inference_seconds = time.perf_counter() - inference_started

    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0 or not np.all(np.isfinite(depth)):
        raise RuntimeError("Depth Anything V3 returned an invalid depth surface")

    confidence = (
        np.asarray(prediction.conf[0], dtype=np.float32)
        if getattr(prediction, "conf", None) is not None
        else None
    )
    intrinsics = (
        np.asarray(prediction.intrinsics[0], dtype=np.float32)
        if getattr(prediction, "intrinsics", None) is not None
        else None
    )
    metadata = {
        "provider": "depth-anything-3",
        "model": model_id,
        "model_revision": model_spec["revision"],
        "model_role": model_spec["role"],
        "source_repository": DA3_SOURCE_REPOSITORY,
        "source_commit": DA3_SOURCE_COMMIT,
        "source_checkout": source,
        "license": model_spec["license"],
        "depth_value_semantics": "relative_distance_far_high",
        "process_resolution": int(process_resolution),
        "process_resolution_method": "upper_bound_resize",
        "device": resolved_device,
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_vram_gb": (
            float(torch.cuda.max_memory_allocated(cuda_device) / (1024**3))
            if cuda_device is not None
            else 0.0
        ),
        "confidence": {
            "available": confidence is not None,
            "minimum": float(np.min(confidence)) if confidence is not None else None,
            "median": float(np.median(confidence)) if confidence is not None else None,
            "maximum": float(np.max(confidence)) if confidence is not None else None,
        },
        "intrinsics": {
            "available": intrinsics is not None,
            "focal_px": (
                float((intrinsics[0, 0] + intrinsics[1, 1]) * 0.5)
                if intrinsics is not None
                else None
            ),
        },
    }
    return depth, metadata
