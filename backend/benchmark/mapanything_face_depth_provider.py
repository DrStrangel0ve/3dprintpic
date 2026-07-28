"""Strict, offline MapAnything Apache depth adapter for benchmark runtimes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


SOURCE_REPOSITORY = "https://github.com/facebookresearch/map-anything.git"
SOURCE_REVISION = "c845b8f4f6cde0c20aecd87573656c3f69f5b2b0"
SOURCE_LICENSE = "Apache-2.0"
MODEL_ID = "facebook/map-anything-apache"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
MODEL_LICENSE = "Apache-2.0"
MODEL_CONFIG_SHA256 = (
    "65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5"
)
MODEL_SAFETENSORS_SHA256 = (
    "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
)
MODEL_SAFETENSORS_SIZE = 4_914_062_480

DINOV2_REPOSITORY = "https://github.com/facebookresearch/dinov2.git"
DINOV2_TORCH_HUB_REPOSITORY = "facebookresearch/dinov2"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_LICENSE = "Apache-2.0"
DINOV2_MODEL_NAME = "dinov2_vitg14"
UNICEPTION_VERSION = "0.1.7"

PROVIDER = "mapanything-apache"
INFERENCE_SHAPE = (518, 518)
OUTPUT_DEPTH_FILENAME = "output_depth_data.npy"
METADATA_FILENAME = "metadata.json"
OUTPUT_ORIENTATION = "camera-z-positive-away"
REQUIRED_PREDICTION_KEYS = (
    "depth_z",
    "pts3d_cam",
    "ray_directions",
    "intrinsics",
    "camera_poses",
)
REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "pyproject.toml",
    "mapanything/__init__.py",
    "mapanything/models/__init__.py",
    "mapanything/models/mapanything/model.py",
    "mapanything/utils/image.py",
)
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors")
REQUIRED_DINOV2_SOURCE_FILES = (
    "LICENSE",
    "hubconf.py",
    "dinov2/hub/backbones.py",
    "dinov2/models/vision_transformer.py",
)

DEPTH_POINT_ATOL = 1e-6
DEPTH_POINT_RTOL = 1e-5
RAY_UNIT_ATOL = 5e-3
RAY_UNIT_RTOL = 5e-3
RAY_NONZERO_EPSILON = 1e-8
POSE_HOMOGENEOUS_ATOL = 1e-5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _installed_distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"MapAnything config repeats key {key!r}")
        result[key] = value
    return result


def _official_json_constant(value: str) -> float:
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    raise ValueError(f"MapAnything config contains invalid constant {value!r}")


def parse_snapshot_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("MapAnything config is unexpectedly large")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_official_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("MapAnything config must be a JSON object")
    return payload


def _nested_value(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(
                "MapAnything config is missing " + ".".join(path)
            )
        value = value[key]
    return value


def validate_snapshot_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the architecture-bearing portion of the pinned HF config."""
    if not isinstance(config, Mapping):
        raise TypeError("MapAnything config must be a mapping")

    for key in (
        "encoder_config",
        "geometric_input_config",
        "info_sharing_config",
        "pred_head_config",
    ):
        if not isinstance(config.get(key), Mapping):
            raise ValueError(f"MapAnything config {key} must be an object")

    expected_values = {
        ("name",): "mapanything",
        ("encoder_config", "encoder_str"): "dinov2",
        ("encoder_config", "data_norm_type"): "dinov2",
        ("encoder_config", "size"): "giant",
        ("encoder_config", "keep_first_n_layers"): 24,
        ("encoder_config", "uses_torch_hub"): True,
        ("pred_head_config", "type"): "dpt+pose",
        ("pred_head_config", "adaptor_type"):
            "raydirs+depth+pose+confidence+mask",
        (
            "pred_head_config",
            "adaptor_config",
            "scene_rep_type",
        ): "raydirs+depth+pose",
        ("use_register_tokens_from_encoder",): True,
        ("load_specific_pretrained_submodules",): False,
        ("specific_pretrained_submodules",): [],
        ("torch_hub_force_reload",): False,
    }
    for path, expected in expected_values.items():
        actual = _nested_value(config, path)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(
                f"MapAnything config {'.'.join(path)} is not pinned"
            )

    def reject_external_checkpoints(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_path = (*path, str(key))
                if key == "pretrained_checkpoint_path" and nested is not None:
                    raise ValueError(
                        "MapAnything config references an external checkpoint at "
                        + ".".join(nested_path)
                    )
                if key == "torch_hub_force_reload" and nested is not False:
                    raise ValueError(
                        "MapAnything config enables a Torch Hub reload at "
                        + ".".join(nested_path)
                    )
                reject_external_checkpoints(nested, nested_path)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                reject_external_checkpoints(nested, (*path, str(index)))

    reject_external_checkpoints(config, ())
    return dict(config)


def mapanything_preflight(
    source_root: str | Path,
    snapshot_root: str | Path,
    dinov2_source_root: str | Path,
    *,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    snapshot_root = Path(snapshot_root).resolve()
    dinov2_source_root = Path(dinov2_source_root).resolve()
    missing_source = [
        relative
        for relative in REQUIRED_SOURCE_FILES
        if not (source_root / relative).is_file()
    ]

    source_revision = None
    source_status = None
    source_error = None
    if source_root.is_dir():
        try:
            source_revision = _git_output(source_root, "rev-parse", "HEAD")
            source_status = _git_output(
                source_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            source_error = f"{type(exc).__name__}: {exc}"

    missing_dinov2_source = [
        relative
        for relative in REQUIRED_DINOV2_SOURCE_FILES
        if not (dinov2_source_root / relative).is_file()
    ]
    dinov2_source_revision = None
    dinov2_source_status = None
    dinov2_source_error = None
    if dinov2_source_root.is_dir():
        try:
            dinov2_source_revision = _git_output(
                dinov2_source_root,
                "rev-parse",
                "HEAD",
            )
            dinov2_source_status = _git_output(
                dinov2_source_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            dinov2_source_error = f"{type(exc).__name__}: {exc}"

    config_path = snapshot_root / "config.json"
    weights_path = snapshot_root / "model.safetensors"
    missing_model = [
        relative
        for relative in REQUIRED_MODEL_FILES
        if not (snapshot_root / relative).is_file()
    ]
    config_sha256 = None
    config_valid = False
    config_error = None
    if config_path.is_file():
        try:
            config_sha256 = sha256_file(config_path)
            validate_snapshot_config(parse_snapshot_config(config_path))
            config_valid = True
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            config_error = f"{type(exc).__name__}: {exc}"

    weights_size = None
    weights_sha256 = None
    weights_error = None
    if weights_path.is_file():
        try:
            weights_size = int(weights_path.stat().st_size)
            weights_sha256 = sha256_file(weights_path)
        except OSError as exc:
            weights_error = f"{type(exc).__name__}: {exc}"

    uniception_version = _installed_distribution_version("uniception")

    checks = {
        "source_exists": source_root.is_dir(),
        "source_revision_pinned": source_revision == SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "dinov2_source_exists": dinov2_source_root.is_dir(),
        "dinov2_source_revision_pinned": (
            dinov2_source_revision == DINOV2_REVISION
        ),
        "dinov2_source_clean": dinov2_source_status == "",
        "dinov2_source_files_complete": not missing_dinov2_source,
        "model_id_pinned": model_id == MODEL_ID,
        "model_revision_pinned": model_revision == MODEL_REVISION,
        "snapshot_exists": snapshot_root.is_dir(),
        "model_files_complete": not missing_model,
        "config_valid": config_valid,
        "config_hash_pinned": config_sha256 == MODEL_CONFIG_SHA256,
        "model_size_pinned": weights_size == MODEL_SAFETENSORS_SIZE,
        "model_hash_pinned": weights_sha256 == MODEL_SAFETENSORS_SHA256,
        "uniception_version_pinned": (
            uniception_version == UNICEPTION_VERSION
        ),
    }
    return {
        "schema_version": 1,
        "provider": PROVIDER,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "expected_revision": SOURCE_REVISION,
            "actual_revision": source_revision,
            "license": SOURCE_LICENSE,
            "root": str(source_root),
            "status": source_status,
            "error": source_error,
            "missing_files": missing_source,
        },
        "dinov2_source": {
            "repository": DINOV2_REPOSITORY,
            "torch_hub_repository": DINOV2_TORCH_HUB_REPOSITORY,
            "expected_revision": DINOV2_REVISION,
            "actual_revision": dinov2_source_revision,
            "license": DINOV2_LICENSE,
            "root": str(dinov2_source_root),
            "status": dinov2_source_status,
            "error": dinov2_source_error,
            "missing_files": missing_dinov2_source,
        },
        "model": {
            "id": model_id,
            "expected_id": MODEL_ID,
            "revision": model_revision,
            "expected_revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "snapshot_root": str(snapshot_root),
            "missing_files": missing_model,
            "config": {
                "path": str(config_path),
                "sha256": config_sha256,
                "expected_sha256": MODEL_CONFIG_SHA256,
                "valid": config_valid,
                "error": config_error,
            },
            "safetensors": {
                "path": str(weights_path),
                "size": weights_size,
                "expected_size": MODEL_SAFETENSORS_SIZE,
                "sha256": weights_sha256,
                "expected_sha256": MODEL_SAFETENSORS_SHA256,
                "error": weights_error,
            },
        },
        "runtime_dependencies": {
            "uniception_version": uniception_version,
            "expected_uniception_version": UNICEPTION_VERSION,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def require_runnable_preflight(evidence: Mapping[str, Any]) -> None:
    checks = evidence.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("MapAnything preflight evidence has no checks")
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise RuntimeError("MapAnything preflight failed: " + ", ".join(failed))
    if evidence.get("runnable") is not True:
        raise RuntimeError("MapAnything preflight did not declare the adapter runnable")


def select_amp_dtype(
    device_name: str,
    compute_capability: Sequence[int],
) -> str:
    """Return the required inference AMP name for a CUDA device."""
    if (
        not isinstance(device_name, str)
        or len(compute_capability) != 2
        or any(isinstance(value, bool) for value in compute_capability)
    ):
        raise ValueError("CUDA device identity is invalid")
    try:
        major, minor = (int(value) for value in compute_capability)
    except (TypeError, ValueError) as exc:
        raise ValueError("CUDA compute capability is invalid") from exc
    if major < 1 or minor < 0:
        raise ValueError("CUDA compute capability is invalid")
    is_blackwell = "blackwell" in device_name.casefold() or major in {10, 12}
    return "bf16" if is_blackwell else "fp16"


def _real_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        raise TypeError(f"{name} must not be a masked array")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    elif value.__class__.__module__.split(".", 1)[0] == "torch" and all(
        hasattr(value, method) for method in ("detach", "float", "cpu", "numpy")
    ):
        array = value.detach().float().cpu().numpy()
    else:
        raise TypeError(f"{name} must be a NumPy array or Torch tensor")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def normalize_batch_one_array(
    value: Any,
    *,
    name: str,
    expected_shape: Sequence[int],
) -> np.ndarray:
    """Normalize an exact unbatched or batch-one array to its unbatched shape."""
    shape = tuple(int(dimension) for dimension in expected_shape)
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("expected_shape must contain positive dimensions")
    array = _real_numpy(value, name)
    if array.shape == shape:
        return array
    if array.shape == (1, *shape):
        return array[0]
    raise ValueError(
        f"{name} has unsupported shape {array.shape}; expected {shape} or "
        f"{(1, *shape)}"
    )


def _normalize_depth_z(value: Any) -> np.ndarray:
    array = _real_numpy(value, "depth_z")
    if array.ndim == 2:
        return array
    if array.ndim == 4 and array.shape[0] == 1 and array.shape[-1] == 1:
        return array[0, ..., 0]
    if array.ndim == 3:
        batch_candidate = array[0] if array.shape[0] == 1 else None
        channel_candidate = array[..., 0] if array.shape[-1] == 1 else None
        if batch_candidate is not None and channel_candidate is None:
            return batch_candidate
        if channel_candidate is not None and batch_candidate is None:
            return channel_candidate
    raise ValueError(
        "depth_z has an unsupported shape; expected (H, W), (1, H, W), "
        "(H, W, 1), or (1, H, W, 1)"
    )


def _float32_checked(value: np.ndarray, name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} cannot be represented as finite float32")
    return result


def validate_prediction(
    prediction: Mapping[str, Any],
    *,
    expected_shape: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Normalize and validate one MapAnything prediction without importing Torch."""
    if not isinstance(prediction, Mapping):
        raise TypeError("MapAnything prediction must be a mapping")
    missing = [key for key in REQUIRED_PREDICTION_KEYS if key not in prediction]
    if missing:
        raise ValueError(
            "MapAnything prediction is missing required keys: " + ", ".join(missing)
        )

    depth = _normalize_depth_z(prediction["depth_z"])
    if depth.ndim != 2 or min(depth.shape) <= 0:
        raise ValueError("depth_z must normalize to a non-empty 2D array")
    height, width = (int(value) for value in depth.shape)
    if expected_shape is not None:
        requested_shape = tuple(int(value) for value in expected_shape)
        if requested_shape != (height, width):
            raise ValueError(
                f"depth_z shape {(height, width)} does not match "
                f"{requested_shape}"
            )
    if np.any(depth <= 0.0):
        raise ValueError("depth_z must be strictly positive")

    points = normalize_batch_one_array(
        prediction["pts3d_cam"],
        name="pts3d_cam",
        expected_shape=(height, width, 3),
    )
    rays = normalize_batch_one_array(
        prediction["ray_directions"],
        name="ray_directions",
        expected_shape=(height, width, 3),
    )
    intrinsics = normalize_batch_one_array(
        prediction["intrinsics"],
        name="intrinsics",
        expected_shape=(3, 3),
    )
    pose = normalize_batch_one_array(
        prediction["camera_poses"],
        name="camera_poses",
        expected_shape=(4, 4),
    )

    depth_error = np.abs(points[..., 2] - depth)
    depth_tolerance = DEPTH_POINT_ATOL + DEPTH_POINT_RTOL * np.abs(depth)
    if np.any(depth_error > depth_tolerance):
        raise ValueError("pts3d_cam z does not agree with depth_z")

    ray_norm = np.linalg.norm(rays, axis=-1)
    if not np.all(np.isfinite(ray_norm)) or np.any(ray_norm <= RAY_NONZERO_EPSILON):
        raise ValueError("ray_directions must be finite and nonzero")
    if not np.allclose(
        ray_norm,
        1.0,
        rtol=RAY_UNIT_RTOL,
        atol=RAY_UNIT_ATOL,
    ):
        raise ValueError("ray_directions must be approximately unit length")

    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    principal_x = float(intrinsics[0, 2])
    principal_y = float(intrinsics[1, 2])
    if not (0.0 <= principal_x < width and 0.0 <= principal_y < height):
        raise ValueError("intrinsics principal point must be within the image")

    if not np.allclose(
        pose[3],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=POSE_HOMOGENEOUS_ATOL,
    ):
        raise ValueError("camera_poses has an invalid homogeneous row")

    return {
        "depth_z": _float32_checked(depth, "depth_z"),
        "pts3d_cam": _float32_checked(points, "pts3d_cam"),
        "ray_directions": _float32_checked(rays, "ray_directions"),
        "intrinsics": _float32_checked(intrinsics, "intrinsics"),
        "camera_poses": _float32_checked(pose, "camera_poses"),
    }


def load_full_rgb_image(image: str | Path | Any) -> np.ndarray:
    from PIL import Image, ImageOps

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        with Image.open(path) as loaded:
            if int(getattr(loaded, "n_frames", 1)) != 1:
                raise ValueError("MapAnything expects exactly one still RGB image")
            image_rgb = np.asarray(
                ImageOps.exif_transpose(loaded).convert("RGB"),
                dtype=np.uint8,
            )
    elif isinstance(image, Image.Image):
        if int(getattr(image, "n_frames", 1)) != 1:
            raise ValueError("MapAnything expects exactly one still RGB image")
        image_rgb = np.asarray(
            ImageOps.exif_transpose(image).convert("RGB"),
            dtype=np.uint8,
        )
    elif isinstance(image, np.ndarray):
        image_rgb = image
    else:
        raise TypeError("MapAnything expects a path, PIL image, or NumPy RGB image")

    if (
        image_rgb.dtype != np.uint8
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.shape[0] <= 0
        or image_rgb.shape[1] <= 0
    ):
        raise ValueError("MapAnything expects one non-empty HxWx3 uint8 RGB image")
    return np.ascontiguousarray(image_rgb).copy()


def resize_depth_to_original(
    raw_depth_z: np.ndarray,
    original_shape: Sequence[int],
) -> np.ndarray:
    """Resize raw camera-Z depth once, without changing its sign or scale."""
    depth = np.asarray(raw_depth_z)
    if depth.dtype != np.float32 or depth.ndim != 2:
        raise ValueError("Raw depth must be a 2D float32 array")
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("Raw depth must be finite and strictly positive")
    if len(original_shape) != 2 or any(
        isinstance(value, bool) for value in original_shape
    ):
        raise ValueError("Original image shape must be (height, width)")
    height, width = (int(value) for value in original_shape)
    if height <= 0 or width <= 0:
        raise ValueError("Original image dimensions must be positive")

    import cv2

    resized = cv2.resize(
        depth,
        (width, height),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    resized = np.ascontiguousarray(resized, dtype=np.float32)
    if resized.shape != (height, width):
        raise ValueError("OpenCV returned an unexpected depth shape")
    if not np.all(np.isfinite(resized)) or np.any(resized <= 0.0):
        raise ValueError("Resized depth is not finite and strictly positive")
    return resized


def save_depth_and_metadata(
    depth: np.ndarray,
    output_directory: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output = np.asarray(depth)
    if output.dtype != np.float32 or output.ndim != 2:
        raise ValueError("Output depth must be a 2D float32 array")
    if not np.all(np.isfinite(output)) or np.any(output <= 0.0):
        raise ValueError("Output depth must be finite and strictly positive")
    if not isinstance(metadata, Mapping):
        raise TypeError("MapAnything metadata must be a mapping")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    depth_path = output_directory / OUTPUT_DEPTH_FILENAME
    depth_temporary = output_directory / f".{OUTPUT_DEPTH_FILENAME}.tmp"
    with depth_temporary.open("wb") as destination:
        np.save(destination, output, allow_pickle=False)
    os.replace(depth_temporary, depth_path)

    payload = dict(metadata)
    payload.update(
        {
            "output_depth_data_filename": OUTPUT_DEPTH_FILENAME,
            "output_depth_data_sha256": sha256_file(depth_path),
            "output_depth_data_dtype": "float32",
            "output_shape": [int(value) for value in output.shape],
        }
    )
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"
    metadata_path = output_directory / METADATA_FILENAME
    metadata_temporary = output_directory / f".{METADATA_FILENAME}.tmp"
    metadata_temporary.write_text(serialized, encoding="utf-8")
    os.replace(metadata_temporary, metadata_path)
    return payload


def _path_is_within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _import_verified_mapanything(source_root: Path):
    for name, module in tuple(sys.modules.items()):
        if name != "mapanything" and not name.startswith("mapanything."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None or not _path_is_within(module_path, source_root):
            raise RuntimeError(
                "An unverified MapAnything module is already loaded: " + name
            )

    sys.path.insert(0, str(source_root))
    try:
        models_module = importlib.import_module("mapanything.models")
        image_module = importlib.import_module("mapanything.utils.image")
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass

    model_class = getattr(models_module, "MapAnything", None)
    preprocess_inputs = getattr(image_module, "preprocess_inputs", None)
    if model_class is None or not callable(preprocess_inputs):
        raise RuntimeError("Pinned MapAnything source lacks the required API")
    for value, label in (
        (model_class, "MapAnything"),
        (preprocess_inputs, "preprocess_inputs"),
    ):
        source_path = inspect.getsourcefile(value)
        if source_path is None or not _path_is_within(source_path, source_root):
            raise RuntimeError(f"{label} was not imported from the verified source")
    return model_class, preprocess_inputs


def _reject_unverified_namespace_modules(namespace: str, source_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != namespace and not name.startswith(namespace + "."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None or not _path_is_within(module_path, source_root):
            raise RuntimeError(
                f"An unverified {namespace} module is already loaded: {name}"
            )


@contextmanager
def _offline_model_load_environment():
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update({key: "1" for key in keys})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _verified_local_dinov2_hub(torch_module: Any, source_root: Path):
    """Route UniCeption's sole Torch Hub request to verified local source."""
    _reject_unverified_namespace_modules("dinov2", source_root)
    original_load = torch_module.hub.load
    calls: list[dict[str, Any]] = []

    def local_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != DINOV2_TORCH_HUB_REPOSITORY:
            raise RuntimeError(
                "MapAnything attempted an unexpected Torch Hub repository"
            )
        if model != DINOV2_MODEL_NAME:
            raise RuntimeError("MapAnything attempted an unexpected DINOv2 model")
        if kwargs.get("force_reload", False) is not False:
            raise RuntimeError("MapAnything attempted a forced Torch Hub reload")
        if kwargs.get("pretrained", True) is not False:
            raise RuntimeError("MapAnything attempted external DINOv2 weights")
        if kwargs.get("source", "github") != "github":
            raise RuntimeError("MapAnything supplied an unexpected Torch Hub source")

        local_kwargs = dict(kwargs)
        local_kwargs.pop("force_reload", None)
        local_kwargs["source"] = "local"
        local_kwargs["pretrained"] = False
        calls.append(
            {
                "repository": repo_or_dir,
                "model": model,
                "source_revision": DINOV2_REVISION,
                "pretrained": False,
            }
        )
        return original_load(str(source_root), model, *args, **local_kwargs)

    torch_module.hub.load = local_load
    try:
        yield calls
    finally:
        torch_module.hub.load = original_load


class MapAnythingFaceDepthProvider:
    def __init__(
        self,
        source_root: str | Path,
        snapshot_root: str | Path,
        dinov2_source_root: str | Path,
        *,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        device: str = "cuda",
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.snapshot_root = Path(snapshot_root).resolve()
        self.dinov2_source_root = Path(dinov2_source_root).resolve()
        self.preflight = mapanything_preflight(
            self.source_root,
            self.snapshot_root,
            self.dinov2_source_root,
            model_id=model_id,
            model_revision=model_revision,
        )
        require_runnable_preflight(self.preflight)

        config_path = self.snapshot_root / "config.json"
        if sha256_file(config_path) != self.preflight["model"]["config"]["sha256"]:
            raise RuntimeError("MapAnything config changed after preflight")
        config = validate_snapshot_config(parse_snapshot_config(config_path))

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("MapAnything benchmark inference requires CUDA")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("MapAnything benchmark inference requires a CUDA device")
        device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        if not 0 <= int(device_index) < int(torch.cuda.device_count()):
            raise ValueError("Requested CUDA device does not exist")
        self.device = torch.device("cuda", int(device_index))
        self.device_name = str(torch.cuda.get_device_name(self.device))
        self.compute_capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(self.device)
        )
        self.amp_dtype = select_amp_dtype(
            self.device_name,
            self.compute_capability,
        )

        MapAnything, preprocess_inputs = _import_verified_mapanything(
            self.source_root
        )
        encoder_config = copy.deepcopy(config["encoder_config"])
        encoder_config["torch_hub_pretrained"] = False
        with _offline_model_load_environment():
            with _verified_local_dinov2_hub(
                torch,
                self.dinov2_source_root,
            ) as dinov2_hub_calls:
                model = MapAnything.from_pretrained(
                    str(self.snapshot_root),
                    revision=MODEL_REVISION,
                    local_files_only=True,
                    force_download=False,
                    strict=True,
                    encoder_config=encoder_config,
                )
        if len(dinov2_hub_calls) != 1:
            raise RuntimeError(
                "MapAnything did not make exactly one verified DINOv2 source load"
            )
        self.dinov2_hub_call = dict(dinov2_hub_calls[0])
        self.model = model.to(self.device).eval()
        self._preprocess_inputs = preprocess_inputs
        self._torch = torch

    def infer(self, image: str | Path | Any) -> dict[str, Any]:
        image_rgb = load_full_rgb_image(image)
        original_shape = tuple(int(value) for value in image_rgb.shape[:2])
        views = self._preprocess_inputs(
            [{"img": image_rgb}],
            resize_mode="fixed_size",
            size=(INFERENCE_SHAPE[1], INFERENCE_SHAPE[0]),
            norm_type="dinov2",
            patch_size=14,
            resolution_set=518,
            verbose=False,
        )
        if not isinstance(views, list) or len(views) != 1:
            raise ValueError("Official preprocessing did not return exactly one view")
        view_image = views[0].get("img") if isinstance(views[0], Mapping) else None
        if tuple(getattr(view_image, "shape", ())) != (1, 3, 518, 518):
            raise ValueError("Official preprocessing did not produce 1x3x518x518")

        cuda = self._torch.cuda
        cuda.synchronize(self.device)
        cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        try:
            predictions = self.model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype=self.amp_dtype,
                apply_mask=False,
                mask_edges=False,
            )
        finally:
            cuda.synchronize(self.device)
        runtime_seconds = float(time.perf_counter() - started)
        peak_vram_bytes = int(cuda.max_memory_allocated(self.device))
        if peak_vram_bytes < 0:
            raise RuntimeError("CUDA returned an invalid peak allocation")

        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError("MapAnything must return exactly one prediction")
        normalized = validate_prediction(
            predictions[0],
            expected_shape=INFERENCE_SHAPE,
        )
        output_depth = resize_depth_to_original(
            normalized["depth_z"],
            original_shape,
        )
        metadata = {
            "schema_version": 1,
            "provider": PROVIDER,
            "orientation": OUTPUT_ORIENTATION,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_license": SOURCE_LICENSE,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
            "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
            "model_safetensors_size": MODEL_SAFETENSORS_SIZE,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "dinov2_repository": DINOV2_REPOSITORY,
            "dinov2_revision": DINOV2_REVISION,
            "dinov2_license": DINOV2_LICENSE,
            "dinov2_model_name": DINOV2_MODEL_NAME,
            "uniception_version": UNICEPTION_VERSION,
            "offline_model_load": True,
            "dinov2_source_load": self.dinov2_hub_call,
            "original_shape": [int(value) for value in original_shape],
            "inference_shape": [int(value) for value in INFERENCE_SHAPE],
            "runtime_seconds": runtime_seconds,
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_gib": float(peak_vram_bytes / (1024**3)),
            "device": str(self.device),
            "device_name": self.device_name,
            "compute_capability": [int(value) for value in self.compute_capability],
            "amp_dtype": self.amp_dtype,
            "memory_efficient_inference": True,
            "minibatch_size": 1,
            "use_amp": True,
            "apply_mask": False,
            "mask_edges": False,
            "depth_resize_interpolation": "cv2.INTER_LINEAR_EXACT",
        }
        return {"depth": output_depth, "metadata": metadata}

    def run(
        self,
        image: str | Path | Any,
        output_directory: str | Path,
    ) -> dict[str, Any]:
        inference = self.infer(image)
        metadata = save_depth_and_metadata(
            inference["depth"],
            output_directory,
            inference["metadata"],
        )
        output_directory = Path(output_directory).resolve()
        return {
            "depth_path": str(output_directory / OUTPUT_DEPTH_FILENAME),
            "metadata_path": str(output_directory / METADATA_FILENAME),
            "metadata": metadata,
        }


def write_depth_outputs(
    source_root: str | Path,
    snapshot_root: str | Path,
    dinov2_source_root: str | Path,
    image: str | Path | Any,
    output_directory: str | Path,
    *,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    device: str = "cuda",
) -> dict[str, Any]:
    provider = MapAnythingFaceDepthProvider(
        source_root,
        snapshot_root,
        dinov2_source_root,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
    )
    return provider.run(image, output_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--source-root", required=True)
    preflight_parser.add_argument("--snapshot-root", required=True)
    preflight_parser.add_argument("--dinov2-source-root", required=True)
    preflight_parser.add_argument("--model-id", default=MODEL_ID)
    preflight_parser.add_argument("--model-revision", default=MODEL_REVISION)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--source-root", required=True)
    run_parser.add_argument("--snapshot-root", required=True)
    run_parser.add_argument("--dinov2-source-root", required=True)
    run_parser.add_argument("--input-image", required=True)
    run_parser.add_argument("--output-directory", required=True)
    run_parser.add_argument("--model-id", default=MODEL_ID)
    run_parser.add_argument("--model-revision", default=MODEL_REVISION)
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.command == "preflight":
        result = mapanything_preflight(
            args.source_root,
            args.snapshot_root,
            args.dinov2_source_root,
            model_id=args.model_id,
            model_revision=args.model_revision,
        )
    else:
        result = write_depth_outputs(
            args.source_root,
            args.snapshot_root,
            args.dinov2_source_root,
            args.input_image,
            args.output_directory,
            model_id=args.model_id,
            model_revision=args.model_revision,
            device=args.device,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
