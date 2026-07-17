"""Run pinned SHeaP expressive FLAME inference in the source camera."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import math
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image


SHEAP_SOURCE_URL = "https://github.com/nlml/SHeaP.git"
SHEAP_SOURCE_REVISION = "33f21125a1c353e1cb534cf9776fefd9e6cce719"
SHEAP_RELEASE = "v1.0.0"
SHEAP_CHECKPOINT_FILENAME = "model_expressive.pt"
SHEAP_CHECKPOINT_URL = (
    "https://github.com/nlml/SHeaP/releases/download/"
    f"{SHEAP_RELEASE}/{SHEAP_CHECKPOINT_FILENAME}"
)
SHEAP_CHECKPOINT_SIZE_BYTES = 348_292_433
SHEAP_CHECKPOINT_SHA256 = (
    "4d769f493072aa2e98770ed1b71db784bc3ee0a2132a0fd36aab841ee591c5e2"
)
SHEAP_PAPER_CHECKPOINT_FILENAME = "model_paper.pt"
SHEAP_PAPER_CHECKPOINT_SIZE_BYTES = 348_352_867
SHEAP_PAPER_CHECKPOINT_SHA256 = (
    "e79addb1b56beb1cf5198020da27e790fa79681544ce1cece58460af9a923d25"
)
SHEAP_CHECKPOINT_VARIANTS = {
    "expressive": {
        "filename": SHEAP_CHECKPOINT_FILENAME,
        "size_bytes": SHEAP_CHECKPOINT_SIZE_BYTES,
        "sha256": SHEAP_CHECKPOINT_SHA256,
    },
    "paper": {
        "filename": SHEAP_PAPER_CHECKPOINT_FILENAME,
        "size_bytes": SHEAP_PAPER_CHECKPOINT_SIZE_BYTES,
        "sha256": SHEAP_PAPER_CHECKPOINT_SHA256,
    },
}
SHEAP_FLAME_SOURCE_SIZE_BYTES = 53_023_716
SHEAP_FLAME_SOURCE_SHA256 = (
    "efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b"
)
SHEAP_FLAME_TENSOR_SIZE_BYTES = 26_784_481
SHEAP_FLAME_TENSOR_SHA256 = (
    "0d3082630291dcdb7186e8553f02083062cce37389c89ad913b47661ed20021f"
)
SHEAP_EYELIDS_SHA256 = (
    "d5d5a2abbc71384b203451085337b0f9a581619bc839838a92b32a80d76ad9fa"
)
SHEAP_IMAGE_SIZE = 224
SHEAP_CROP_SCALE = 1.0
SHEAP_FACE_CROP_MARGIN = 0.9
SHEAP_FACE_CROP_SHIFT_UP = 0.5
SHEAP_VERTICAL_FOV_DEGREES = 14.2539
SHEAP_CAMERA_WORLD_Z = 1.0
SHEAP_REQUIRED_SOURCE_FILES = (
    "LICENSE.txt",
    "README.md",
    "gradio_demo.py",
    "video_demo.py",
    "sheap/fa_landmark_utils.py",
    "sheap/landmark_utils.py",
    "sheap/load_model.py",
    "sheap/render.py",
    "sheap/tiny_flame.py",
)
SHEAP_REQUIRED_DEPENDENCIES = (
    "cv2",
    "PIL",
    "roma",
    "torch",
    "torchvision",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _asset_record(
    path: Path,
    *,
    expected_size_bytes: int | None,
    expected_sha256: str,
) -> dict:
    exists = path.is_file()
    size_bytes = int(path.stat().st_size) if exists else None
    sha256 = _sha256(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "expected_size_bytes": expected_size_bytes,
        "size_pinned": (
            size_bytes == expected_size_bytes
            if expected_size_bytes is not None
            else None
        ),
        "sha256": sha256,
        "expected_sha256": expected_sha256,
        "hash_pinned": sha256 == expected_sha256,
    }


def sheap_preflight(
    provider_root: str | Path,
    checkpoint_path: str | Path,
    flame_tensor_path: str | Path,
    flame_source_path: str | Path,
    eyelids_path: str | Path,
    *,
    model_type: str = "expressive",
) -> dict:
    """Verify every source and model input before loading executable weights."""

    provider_root = Path(provider_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    flame_tensor_path = Path(flame_tensor_path).resolve()
    flame_source_path = Path(flame_source_path).resolve()
    eyelids_path = Path(eyelids_path).resolve()
    model_type = str(model_type)
    if model_type not in SHEAP_CHECKPOINT_VARIANTS:
        raise ValueError(f"Unsupported SHeaP model type: {model_type}")
    checkpoint_spec = SHEAP_CHECKPOINT_VARIANTS[model_type]
    missing_source = [
        relative
        for relative in SHEAP_REQUIRED_SOURCE_FILES
        if not (provider_root / relative).is_file()
    ]
    source_revision = None
    source_status = None
    source_error = None
    try:
        source_revision = _git_output(provider_root, "rev-parse", "HEAD")
        source_status = _git_output(
            provider_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        source_error = f"{type(exc).__name__}: {exc}"

    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in SHEAP_REQUIRED_DEPENDENCIES
    }
    checkpoint = _asset_record(
        checkpoint_path,
        expected_size_bytes=checkpoint_spec["size_bytes"],
        expected_sha256=checkpoint_spec["sha256"],
    )
    flame_source = _asset_record(
        flame_source_path,
        expected_size_bytes=SHEAP_FLAME_SOURCE_SIZE_BYTES,
        expected_sha256=SHEAP_FLAME_SOURCE_SHA256,
    )
    flame_tensor = _asset_record(
        flame_tensor_path,
        expected_size_bytes=SHEAP_FLAME_TENSOR_SIZE_BYTES,
        expected_sha256=SHEAP_FLAME_TENSOR_SHA256,
    )
    eyelids = _asset_record(
        eyelids_path,
        expected_size_bytes=None,
        expected_sha256=SHEAP_EYELIDS_SHA256,
    )
    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": source_revision == SHEAP_SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "checkpoint_exists": checkpoint["exists"],
        "checkpoint_size_pinned": checkpoint["size_pinned"] is True,
        "checkpoint_hash_pinned": checkpoint["hash_pinned"],
        "flame_source_exists": flame_source["exists"],
        "flame_source_size_pinned": flame_source["size_pinned"] is True,
        "flame_source_hash_pinned": flame_source["hash_pinned"],
        "flame_tensor_exists": flame_tensor["exists"],
        "flame_tensor_size_pinned": flame_tensor["size_pinned"] is True,
        "flame_tensor_hash_pinned": flame_tensor["hash_pinned"],
        "eyelids_exists": eyelids["exists"],
        "eyelids_hash_pinned": eyelids["hash_pinned"],
        "dependencies_available": all(dependencies.values()),
    }
    return {
        "schema_version": 1,
        "provider": f"sheap-{model_type}",
        "source": {
            "url": SHEAP_SOURCE_URL,
            "expected_revision": SHEAP_SOURCE_REVISION,
            "actual_revision": source_revision,
            "status": source_status,
            "error": source_error,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "checkpoint": {
            "release": SHEAP_RELEASE,
            "model_type": model_type,
            "filename": checkpoint_spec["filename"],
            "url": (
                "https://github.com/nlml/SHeaP/releases/download/"
                f"{SHEAP_RELEASE}/{checkpoint_spec['filename']}"
            ),
            **checkpoint,
        },
        "flame_source": flame_source,
        "flame_tensor": {
            "conversion": "official convert_flame.py from pinned source",
            **flame_tensor,
        },
        "eyelids": eyelids,
        "dependencies": dependencies,
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "camera": {
            "network_size": SHEAP_IMAGE_SIZE,
            "vertical_fov_degrees": SHEAP_VERTICAL_FOV_DEGREES,
            "camera_world_z": SHEAP_CAMERA_WORLD_Z,
            "projection": (
                "pyrender-perspective-then-torchvision-half-pixel-"
                "inverse-resize"
            ),
            "front_surface": "minimum-positive-camera-depth",
            "depth_interpolation": "perspective-correct-reciprocal-depth",
        },
        "license": {
            "source_and_weights": "CC BY-NC 4.0",
            "flame_model": (
                "Research-only authenticated local asset; never redistribute"
            ),
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def sheap_crop_transform(
    bbox_xyxy: list[float] | tuple[float, ...],
    *,
    image_width: int,
    image_height: int,
    crop_scale: float = SHEAP_CROP_SCALE,
) -> dict:
    """Build the integer-slice crop and half-pixel resize transform."""

    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError("SHeaP target bbox must contain four finite values")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("SHeaP target bbox is empty")
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width < 2 or image_height < 2:
        raise ValueError("SHeaP source image is too small")
    if x1 <= 0 or y1 <= 0 or x0 >= image_width or y0 >= image_height:
        raise ValueError("SHeaP target bbox does not intersect the source image")
    crop_scale = float(crop_scale)
    if not math.isfinite(crop_scale) or crop_scale <= 0:
        raise ValueError("SHeaP crop scale must be positive and finite")
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    half_width = 0.5 * (x1 - x0) * crop_scale
    half_height = 0.5 * (y1 - y0) * crop_scale
    left = max(0, int(center_x - half_width))
    top = max(0, int(center_y - half_height))
    right = min(image_width, int(center_x + half_width))
    bottom = min(image_height, int(center_y + half_height))
    crop_width = right - left
    crop_height = bottom - top
    if crop_width <= 1 or crop_height <= 1:
        raise ValueError("SHeaP integer crop is too small")
    scale_x = crop_width / SHEAP_IMAGE_SIZE
    scale_y = crop_height / SHEAP_IMAGE_SIZE
    network_to_source = np.array(
        [
            [scale_x, 0.0, left - 0.5 + 0.5 * scale_x],
            [0.0, scale_y, top - 0.5 + 0.5 * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_to_network = np.linalg.inv(network_to_source)
    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "crop_scale": crop_scale,
        "integer_crop_box_xyxy": [left, top, right, bottom],
        "crop_width_pixels": crop_width,
        "crop_height_pixels": crop_height,
        "crop_clamped": bool(
            left != int(center_x - half_width)
            or top != int(center_y - half_height)
            or right != int(center_x + half_width)
            or bottom != int(center_y + half_height)
        ),
        "network_size": SHEAP_IMAGE_SIZE,
        "image_shape": [image_height, image_width],
        "pixel_center_convention": "torchvision-half-pixel-resize",
        "resize": {
            "library": "torchvision.transforms.functional.resize",
            "interpolation": "bilinear",
            "antialias": True,
        },
        "source_to_network_matrix": source_to_network.astype(np.float64),
        "network_to_source_matrix": network_to_source.astype(np.float64),
    }


def sheap_head_crop_bbox(
    face_bbox_xyxy: list[float] | tuple[float, ...],
    *,
    image_width: int,
    image_height: int,
    margin: float = SHEAP_FACE_CROP_MARGIN,
    shift_up: float = SHEAP_FACE_CROP_SHIFT_UP,
) -> list[int]:
    """Approximate SHeaP's landmark crop from tight bbox extrema."""

    bbox = np.asarray(face_bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError("SHeaP face bbox must contain four finite values")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("SHeaP face bbox is empty")
    return sheap_head_crop_bbox_from_landmarks(
        np.array([[x0, y0], [x1, y1]], dtype=np.float64),
        image_width=image_width,
        image_height=image_height,
        margin=margin,
        shift_up=shift_up,
    )


def sheap_head_crop_bbox_from_landmarks(
    landmarks_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    margin: float = SHEAP_FACE_CROP_MARGIN,
    shift_up: float = SHEAP_FACE_CROP_SHIFT_UP,
) -> list[int]:
    """Apply the upstream video crop formula to detected landmark extrema."""

    landmarks_xy = np.asarray(landmarks_xy, dtype=np.float64)
    if (
        landmarks_xy.ndim != 2
        or landmarks_xy.shape[1] != 2
        or len(landmarks_xy) < 2
        or not np.all(np.isfinite(landmarks_xy))
    ):
        raise ValueError("SHeaP crop landmarks must have shape [N, 2]")
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width < 2 or image_height < 2:
        raise ValueError("SHeaP source image is too small")
    margin = float(margin)
    shift_up = float(shift_up)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("SHeaP face crop margin must be non-negative")
    if not math.isfinite(shift_up):
        raise ValueError("SHeaP face crop shift must be finite")
    normalized = landmarks_xy / np.array(
        [image_width, image_height],
        dtype=np.float64,
    )
    minimum = np.min(normalized, axis=0)
    maximum = np.max(normalized, axis=0)
    center = 0.5 * (minimum + maximum)
    base_half_size = 0.5 * float(np.max(maximum - minimum))
    half_size = base_half_size * (1.0 + margin)
    shifted_center_y = center[1] - base_half_size * shift_up
    aspect_ratio = image_width / image_height
    normalized_crop = [
        center[0] - half_size / aspect_ratio,
        shifted_center_y - half_size,
        center[0] + half_size / aspect_ratio,
        shifted_center_y + half_size,
    ]
    normalized_crop = np.clip(normalized_crop, 0.0, 1.0)
    pixel_crop = normalized_crop * np.array(
        [image_width, image_height, image_width, image_height],
        dtype=np.float64,
    )
    crop = [int(value) for value in pixel_crop]
    if crop[2] - crop[0] <= 1 or crop[3] - crop[1] <= 1:
        raise ValueError("SHeaP landmark crop is too small")
    return crop


def project_sheap_vertices(
    vertices: np.ndarray,
    crop: dict,
    *,
    vertical_fov_degrees: float = SHEAP_VERTICAL_FOV_DEGREES,
    camera_world_z: float = SHEAP_CAMERA_WORLD_Z,
) -> np.ndarray:
    """Project SHeaP world vertices into source pixels and camera depth."""

    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("SHeaP vertices must have shape [V, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("SHeaP vertices contain non-finite values")
    vertical_fov_degrees = float(vertical_fov_degrees)
    camera_world_z = float(camera_world_z)
    if (
        not math.isfinite(vertical_fov_degrees)
        or vertical_fov_degrees <= 0
        or vertical_fov_degrees >= 180
    ):
        raise ValueError("SHeaP vertical FOV must be between 0 and 180 degrees")
    if not math.isfinite(camera_world_z):
        raise ValueError("SHeaP camera world Z must be finite")
    network_size = int(crop.get("network_size", SHEAP_IMAGE_SIZE))
    if network_size < 2:
        raise ValueError("SHeaP network size must be at least two pixels")
    network_to_source = np.asarray(
        crop.get("network_to_source_matrix"),
        dtype=np.float64,
    )
    if network_to_source.shape != (3, 3) or not np.all(
        np.isfinite(network_to_source)
    ):
        raise ValueError("SHeaP crop has no finite inverse transform")

    camera_depth = camera_world_z - vertices[:, 2]
    if np.any(camera_depth <= 0) or not np.all(np.isfinite(camera_depth)):
        raise ValueError("SHeaP vertices cross or leave the camera plane")
    center = 0.5 * (network_size - 1)
    focal = (0.5 * network_size) / math.tan(
        math.radians(vertical_fov_degrees) * 0.5
    )
    x_network = focal * vertices[:, 0] / camera_depth + center
    y_network = center - focal * vertices[:, 1] / camera_depth
    network_h = np.column_stack(
        (x_network, y_network, np.ones(len(vertices), dtype=np.float64))
    )
    source_h = network_h @ network_to_source.T
    source_xy = source_h[:, :2] / source_h[:, 2:3]
    projected = np.column_stack((source_xy, camera_depth))
    if not np.all(np.isfinite(projected)):
        raise ValueError("SHeaP projection produced non-finite coordinates")
    return projected.astype(np.float32)


def fit_sheap_landmark_similarity(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit a deterministic orientation-preserving 2D similarity transform."""

    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if (
        source_xy.ndim != 2
        or source_xy.shape[1] != 2
        or target_xy.shape != source_xy.shape
        or len(source_xy) < 3
    ):
        raise ValueError("SHeaP landmark arrays must have matching [N, 2] shapes")
    if not np.all(np.isfinite(source_xy)) or not np.all(
        np.isfinite(target_xy)
    ):
        raise ValueError("SHeaP landmarks contain non-finite values")
    source_center = np.mean(source_xy, axis=0)
    target_center = np.mean(target_xy, axis=0)
    source_centered = source_xy - source_center
    target_centered = target_xy - target_center
    source_energy = float(np.sum(source_centered * source_centered))
    if source_energy <= np.finfo(np.float64).eps:
        raise ValueError("SHeaP source landmarks have no spatial extent")
    covariance = source_centered.T @ target_centered
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1.0
        singular_values[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    scale = float(np.sum(singular_values) / source_energy)
    translation = target_center - scale * (rotation @ source_center)
    matrix = np.column_stack(
        (scale * rotation, translation.reshape(2, 1))
    )
    fitted = np.column_stack(
        (source_xy, np.ones(len(source_xy), dtype=np.float64))
    ) @ matrix.T
    errors = np.linalg.norm(fitted - target_xy, axis=1)
    rotation_degrees = math.degrees(
        math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    )
    stats = {
        "method": "all-correspondence-orthogonal-procrustes-similarity",
        "correspondences": int(len(source_xy)),
        "scale": scale,
        "rotation_degrees": float(rotation_degrees),
        "translation_xy": [float(value) for value in translation],
        "residual_median_pixels": float(np.median(errors)),
        "residual_p95_pixels": float(np.percentile(errors, 95.0)),
        "residual_maximum_pixels": float(np.max(errors)),
    }
    return matrix.astype(np.float32), stats


def apply_sheap_image_similarity(
    projected_vertices: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Apply a 2D registration while retaining metric camera depth."""

    projected_vertices = np.asarray(projected_vertices, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float64)
    if projected_vertices.ndim != 2 or projected_vertices.shape[1] != 3:
        raise ValueError("SHeaP projected vertices must have shape [V, 3]")
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("SHeaP image similarity must have shape [2, 3]")
    transformed = projected_vertices.copy()
    transformed[:, :2] = np.column_stack(
        (
            projected_vertices[:, :2],
            np.ones(len(projected_vertices), dtype=np.float32),
        )
    ) @ matrix.T
    if not np.all(np.isfinite(transformed)):
        raise ValueError("SHeaP image similarity produced non-finite vertices")
    return transformed


def rasterize_sheap_camera_depth(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict]:
    """Rasterize positive camera depth with perspective-correct interpolation."""

    from backend.benchmark import vggheads_depth_provider

    projected_vertices = np.asarray(projected_vertices, dtype=np.float64)
    if projected_vertices.ndim != 2 or projected_vertices.shape[1] != 3:
        raise ValueError("SHeaP projected vertices must have shape [V, 3]")
    if not np.all(np.isfinite(projected_vertices)):
        raise ValueError("SHeaP projected vertices contain non-finite values")
    if np.any(projected_vertices[:, 2] <= 0.05):
        raise ValueError("SHeaP camera depth crosses the 0.05 near plane")
    reciprocal_vertices = projected_vertices.copy()
    reciprocal_vertices[:, 2] = 1.0 / reciprocal_vertices[:, 2]
    reciprocal_depth, raster = (
        vggheads_depth_provider.rasterize_projected_mesh_depth(
        reciprocal_vertices,
        faces,
        height=height,
        width=width,
        front_surface="maximum-z",
        )
    )
    finite = np.isfinite(reciprocal_depth) & (reciprocal_depth > 0)
    camera_depth = np.full(reciprocal_depth.shape, np.nan, dtype=np.float32)
    camera_depth[finite] = 1.0 / reciprocal_depth[finite]
    return camera_depth, {
        **raster,
        "front_surface": "minimum-positive-camera-depth",
        "interpolation": "perspective-correct-reciprocal-depth",
        "internal_representation": "maximum-reciprocal-camera-depth",
        "background": "nan",
        "backface_culling": False,
        "near_plane": 0.05,
        "adapter_sha256": _sha256(Path(__file__).resolve()),
        "shared_rasterizer_sha256": _sha256(
            Path(vggheads_depth_provider.__file__).resolve()
        ),
    }


def _load_tiny_flame_module(provider_root: Path):
    module_path = provider_root / "sheap" / "tiny_flame.py"
    spec = importlib.util.spec_from_file_location(
        "_sheap_pinned_tiny_flame",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned SHeaP module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_norm(tensor) -> float:
    import torch

    return float(torch.linalg.vector_norm(tensor.detach().float()).cpu())


class SHeaPProvider:
    """Pinned expressive SHeaP model with camera-faithful mesh projection."""

    def __init__(
        self,
        provider_root: str | Path,
        checkpoint_path: str | Path,
        flame_tensor_path: str | Path,
        flame_source_path: str | Path,
        eyelids_path: str | Path,
        *,
        device: str = "cuda",
        crop_scale: float = SHEAP_CROP_SCALE,
        model_type: str = "expressive",
    ):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        self.provider_root = Path(provider_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.flame_tensor_path = Path(flame_tensor_path).resolve()
        self.flame_source_path = Path(flame_source_path).resolve()
        self.eyelids_path = Path(eyelids_path).resolve()
        self.device = str(device)
        self.crop_scale = float(crop_scale)
        self.model_type = str(model_type)
        self.preflight = sheap_preflight(
            self.provider_root,
            self.checkpoint_path,
            self.flame_tensor_path,
            self.flame_source_path,
            self.eyelids_path,
            model_type=self.model_type,
        )
        if not self.preflight["runnable"]:
            failed = [
                name
                for name, passed in self.preflight["checks"].items()
                if not passed
            ]
            raise RuntimeError("SHeaP preflight failed: " + ", ".join(failed))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SHeaP requested CUDA but CUDA is unavailable")
        if not math.isfinite(self.crop_scale) or self.crop_scale <= 0:
            raise ValueError("SHeaP crop scale must be positive and finite")

        torch.manual_seed(0)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        import cv2
        import PIL
        import roma
        import torchvision

        cuda_runtime = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": None,
            "device_capability": None,
        }
        if self.device.startswith("cuda"):
            cuda_runtime["device_name"] = torch.cuda.get_device_name(
                self.device
            )
            cuda_runtime["device_capability"] = list(
                torch.cuda.get_device_capability(self.device)
            )
        self.preflight["runtime"] = {
            "python": platform.python_version(),
            "packages": {
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "pillow": PIL.__version__,
                "roma": (
                    getattr(roma, "__version__", None)
                    or _distribution_version("roma")
                ),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
            },
            "cuda": cuda_runtime,
            "requested_device": self.device,
        }
        self.preflight["determinism"] = {
            "seed": 0,
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(
                torch.backends.cudnn.deterministic
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        }

        tiny_flame_module = _load_tiny_flame_module(self.provider_root)
        self._pose_components_to_rotmats = (
            tiny_flame_module.pose_components_to_rotmats
        )
        self.model = torch.jit.load(
            str(self.checkpoint_path),
            map_location=self.device,
        ).to(self.device)
        self.model.eval()
        self.flame = tiny_flame_module.TinyFlame(
            self.flame_tensor_path,
            eyelids_ckpt=self.eyelids_path,
        ).to(self.device)
        self.flame.eval()
        self.faces = self.flame.faces.detach().long().cpu().numpy()
        if self.faces.shape != (9976, 3):
            raise ValueError("SHeaP FLAME faces must have shape [9976, 3]")
        if int(np.min(self.faces)) < 0 or int(np.max(self.faces)) >= 5023:
            raise ValueError("SHeaP FLAME faces reference invalid vertices")
        self.calls: list[dict] = []
        self.allocated_before_inference = (
            int(torch.cuda.memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _preprocess(
        self,
        image_rgb: np.ndarray,
        target_bbox_xyxy: list[float] | tuple[float, ...],
    ):
        import torch
        import torchvision
        from torchvision.transforms import functional as tv_functional

        height, width = image_rgb.shape[:2]
        crop = sheap_crop_transform(
            target_bbox_xyxy,
            image_width=width,
            image_height=height,
            crop_scale=self.crop_scale,
        )
        left, top, right, bottom = crop["integer_crop_box_xyxy"]
        image_tensor = (
            torch.from_numpy(np.array(image_rgb, copy=True, order="C"))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        cropped = image_tensor[:, top:bottom, left:right]
        if cropped.shape[1] <= 1 or cropped.shape[2] <= 1:
            raise ValueError("SHeaP integer crop has insufficient pixels")
        tensor = tv_functional.resize(
            cropped,
            [SHEAP_IMAGE_SIZE, SHEAP_IMAGE_SIZE],
            antialias=True,
        ).unsqueeze(0).to(self.device)
        crop["resize"]["torch_version"] = torch.__version__
        crop["resize"]["torchvision_version"] = torchvision.__version__
        serializable_crop = {
            **crop,
            "source_to_network_matrix": crop[
                "source_to_network_matrix"
            ].tolist(),
            "network_to_source_matrix": crop[
                "network_to_source_matrix"
            ].tolist(),
        }
        return tensor, crop, serializable_crop

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        target_bbox_xyxy: list[float] | tuple[float, ...],
    ) -> dict:
        import torch

        if isinstance(image, (str, Path)):
            image_rgb = np.asarray(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_rgb = np.asarray(image.convert("RGB"))
        else:
            image_rgb = np.asarray(image)
        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.dtype != np.uint8
        ):
            raise ValueError("SHeaP expects an RGB uint8 image")
        tensor, crop, serializable_crop = self._preprocess(
            image_rgb,
            target_bbox_xyxy,
        )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(tensor)
            required = {
                "shape_from_facenet",
                "expr",
                "eyelids",
                "cam_trans",
                "torso_pose",
                "neck_pose",
                "jaw_pose",
                "eye_l_pose",
                "eye_r_pose",
            }
            missing = sorted(required - set(outputs))
            if missing:
                raise RuntimeError(
                    "SHeaP checkpoint omitted outputs: " + ", ".join(missing)
                )
            expected_shapes = {
                "shape_from_facenet": (1, 300),
                "expr": (1, 100),
                "eyelids": (1, 2),
                "cam_trans": (1, 3),
                "torso_pose": (1, 3),
                "neck_pose": (1, 3),
                "jaw_pose": (1, 3),
                "eye_l_pose": (1, 3),
                "eye_r_pose": (1, 3),
            }
            invalid_shapes = [
                name
                for name, shape in expected_shapes.items()
                if tuple(outputs[name].shape) != shape
            ]
            if invalid_shapes:
                raise RuntimeError(
                    "SHeaP checkpoint output shapes changed: "
                    + ", ".join(invalid_shapes)
                )
            non_finite = [
                name
                for name in expected_shapes
                if not bool(torch.all(torch.isfinite(outputs[name])))
            ]
            if non_finite:
                raise RuntimeError(
                    "SHeaP checkpoint emitted non-finite outputs: "
                    + ", ".join(non_finite)
                )
            world_vertices_tensor = self.flame(
                shape=outputs["shape_from_facenet"],
                expression=outputs["expr"],
                pose=self._pose_components_to_rotmats(outputs),
                eyelids=outputs["eyelids"],
                translation=outputs["cam_trans"],
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        world_vertices = (
            world_vertices_tensor[0].detach().float().cpu().numpy()
        )
        if world_vertices.shape != (5023, 3) or not np.all(
            np.isfinite(world_vertices)
        ):
            raise RuntimeError("SHeaP FLAME output must be finite [5023, 3]")
        projected_vertices = project_sheap_vertices(world_vertices, crop)
        if float(np.min(projected_vertices[:, 2])) <= 0.05:
            raise RuntimeError("SHeaP mesh crosses the renderer near plane")
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        height, width = image_rgb.shape[:2]
        in_frame = (
            (projected_vertices[:, 0] >= 0)
            & (projected_vertices[:, 0] <= width - 1)
            & (projected_vertices[:, 1] >= 0)
            & (projected_vertices[:, 1] <= height - 1)
        )
        output_shapes = {
            name: list(value.shape)
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
        optional_scalars = {}
        for name in ("uncertainty", "uncertainty_neck"):
            value = outputs.get(name)
            if isinstance(value, torch.Tensor):
                if tuple(value.shape) != (1, 1) or not bool(
                    torch.all(torch.isfinite(value))
                ):
                    raise RuntimeError(
                        f"SHeaP optional output {name} changed shape or finiteness"
                    )
                optional_scalars[name] = float(value[0, 0].detach().cpu())
        metadata = {
            "provider": f"sheap-{self.model_type}",
            "runtime_seconds": float(elapsed),
            "peak_vram_gib": float(peak_allocated / (1024**3)),
            "incremental_peak_vram_gib": float(
                max(0, peak_allocated - self.allocated_before_inference)
                / (1024**3)
            ),
            "device": self.device,
            "crop": serializable_crop,
            "camera": {
                "vertical_fov_degrees": SHEAP_VERTICAL_FOV_DEGREES,
                "camera_world_z": SHEAP_CAMERA_WORLD_Z,
                "depth_semantics": "positive-camera-distance",
                "near_is_smaller": True,
                "units": "native-nonmetric",
                "near_plane": 0.05,
                "pixel_projection": (
                    "pyrender-perspective-then-torchvision-half-pixel-"
                    "inverse-resize"
                ),
            },
            "cam_trans": [
                float(value)
                for value in outputs["cam_trans"][0]
                .detach()
                .float()
                .cpu()
            ],
            "shape_l2": _tensor_norm(outputs["shape_from_facenet"][0]),
            "expression_l2": _tensor_norm(outputs["expr"][0]),
            "optional_scalars": optional_scalars,
            "blendshapes_l2": (
                _tensor_norm(outputs["blendshapes"][0])
                if isinstance(outputs.get("blendshapes"), torch.Tensor)
                else None
            ),
            "eyelids": [
                float(value)
                for value in outputs["eyelids"][0].detach().float().cpu()
            ],
            "pose_l2": {
                name: _tensor_norm(outputs[name][0])
                for name in (
                    "torso_pose",
                    "neck_pose",
                    "jaw_pose",
                    "eye_l_pose",
                    "eye_r_pose",
                )
            },
            "output_shapes": output_shapes,
            "vertices": int(len(projected_vertices)),
            "faces": int(len(self.faces)),
            "projected_bbox_xyxy": [
                float(np.min(projected_vertices[:, 0])),
                float(np.min(projected_vertices[:, 1])),
                float(np.max(projected_vertices[:, 0])),
                float(np.max(projected_vertices[:, 1])),
            ],
            "camera_depth_min": float(np.min(projected_vertices[:, 2])),
            "camera_depth_max": float(np.max(projected_vertices[:, 2])),
            "in_frame_vertex_ratio": float(np.mean(in_frame)),
        }
        self.calls.append(metadata)
        return {
            "vertices": projected_vertices,
            "world_vertices": world_vertices.astype(np.float32),
            "faces": self.faces.copy(),
            "metadata": metadata,
        }

    def provenance(self) -> dict:
        return {
            "provider": f"sheap-{self.model_type}",
            "preflight": self.preflight,
            "calls": self.calls,
        }
