"""Validate and normalize Pixel3DMM full-fit camera geometry."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import subprocess

import numpy as np


PIXEL3DMM_SOURCE_URL = "https://github.com/SimonGiebenhain/pixel3dmm.git"
PIXEL3DMM_SOURCE_REVISION = "fcd1fa973c7715b02a8948dfc679dff53cf85924"
PIXEL3DMM_TRACKING_SIZE = 256
PIXEL3DMM_PREDICTION_SIZE = 512
PIXEL3DMM_UV_CHECKPOINT_SIZE_BYTES = 2_246_794_577
PIXEL3DMM_UV_CHECKPOINT_SHA256 = (
    "dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8"
)
PIXEL3DMM_NORMAL_CHECKPOINT_SIZE_BYTES = 1_469_022_184
PIXEL3DMM_NORMAL_CHECKPOINT_SHA256 = (
    "e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da"
)
PIXEL3DMM_REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "README.md",
    "configs/tracking.yaml",
    "install_preprocessing_pipeline.sh",
    "scripts/network_inference.py",
    "scripts/run_preprocessing.py",
    "scripts/track.py",
    "scripts/viz_head_centric_cameras.py",
    "src/pixel3dmm/preprocessing/pipnet_utils.py",
    "src/pixel3dmm/tracking/nvdiffrast_util.py",
    "src/pixel3dmm/tracking/renderer_nvdiffrast.py",
    "src/pixel3dmm/tracking/tracker.py",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
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


def _asset_record(
    path: Path,
    *,
    expected_size_bytes: int,
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
        "size_pinned": size_bytes == expected_size_bytes,
        "sha256": sha256,
        "expected_sha256": expected_sha256,
        "hash_pinned": sha256 == expected_sha256,
    }


def pixel3dmm_transfer_preflight(
    provider_root: str | Path,
    uv_checkpoint_path: str | Path,
    normal_checkpoint_path: str | Path,
) -> dict:
    """Verify public source and checkpoints without claiming runtime readiness."""

    provider_root = Path(provider_root).resolve()
    uv_checkpoint_path = Path(uv_checkpoint_path).resolve()
    normal_checkpoint_path = Path(normal_checkpoint_path).resolve()
    missing_source = [
        relative
        for relative in PIXEL3DMM_REQUIRED_SOURCE_FILES
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

    uv_checkpoint = _asset_record(
        uv_checkpoint_path,
        expected_size_bytes=PIXEL3DMM_UV_CHECKPOINT_SIZE_BYTES,
        expected_sha256=PIXEL3DMM_UV_CHECKPOINT_SHA256,
    )
    normal_checkpoint = _asset_record(
        normal_checkpoint_path,
        expected_size_bytes=PIXEL3DMM_NORMAL_CHECKPOINT_SIZE_BYTES,
        expected_sha256=PIXEL3DMM_NORMAL_CHECKPOINT_SHA256,
    )
    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": source_revision == PIXEL3DMM_SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "uv_checkpoint_size_pinned": uv_checkpoint["size_pinned"] is True,
        "uv_checkpoint_hash_pinned": uv_checkpoint["hash_pinned"],
        "normal_checkpoint_size_pinned": (
            normal_checkpoint["size_pinned"] is True
        ),
        "normal_checkpoint_hash_pinned": normal_checkpoint["hash_pinned"],
    }
    return {
        "schema_version": 1,
        "provider": "pixel3dmm-full-fit",
        "source": {
            "url": PIXEL3DMM_SOURCE_URL,
            "expected_revision": PIXEL3DMM_SOURCE_REVISION,
            "revision": source_revision,
            "status": source_status,
            "error": source_error,
            "missing_files": missing_source,
        },
        "checkpoints": {
            "uv": uv_checkpoint,
            "normal": normal_checkpoint,
        },
        "checks": checks,
        "transfer_ready": all(checks.values()),
        "runtime_ready": False,
        "runtime_blockers_not_assessed": [
            "registered FLAME/MICA assets and license acknowledgements",
            "isolated Linux CUDA dependency environment",
            "source-landmark projection replay",
        ],
        "license": {
            "source": "CC BY-NC 4.0 with additional notices in copied files",
            "flame_assets": "registered research terms",
            "research_only": True,
            "production_eligible": False,
        },
    }


def _rotation_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    gram = matrix @ matrix.T
    determinant = float(np.linalg.det(matrix))
    if not np.allclose(gram, np.eye(3), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{name} is not orthonormal")
    if not math.isclose(determinant, 1.0, abs_tol=1e-4, rel_tol=1e-4):
        raise ValueError(f"{name} is not a proper rotation")
    return matrix


def _translation(value: np.ndarray, *, name: str) -> np.ndarray:
    translation = np.asarray(value, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError(f"{name} must contain three finite values")
    return translation


def transform_pixel3dmm_camera_vertices(
    posed_vertices: np.ndarray,
    *,
    head_rotation: np.ndarray,
    head_translation: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    near_plane: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """Apply the fitted rigid head pose and OpenGL world-to-camera transform."""

    vertices = np.asarray(posed_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Pixel3DMM posed vertices must have shape [V, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Pixel3DMM posed vertices contain non-finite values")
    head_rotation = _rotation_matrix(head_rotation, name="head_rotation")
    camera_rotation = _rotation_matrix(
        camera_rotation,
        name="camera_rotation",
    )
    head_translation = _translation(
        head_translation,
        name="head_translation",
    )
    camera_translation = _translation(
        camera_translation,
        name="camera_translation",
    )
    near_plane = float(near_plane)
    if not math.isfinite(near_plane) or near_plane <= 0:
        raise ValueError("Pixel3DMM near plane must be positive and finite")

    world_vertices = vertices @ head_rotation.T + head_translation
    camera_vertices = world_vertices @ camera_rotation.T + camera_translation
    camera_depth = -camera_vertices[:, 2]
    if not np.all(np.isfinite(camera_vertices)) or np.any(
        camera_depth <= near_plane
    ):
        raise ValueError("Pixel3DMM mesh crosses the OpenGL camera near plane")
    return camera_vertices.astype(np.float32), {
        "coordinate_convention": "OpenGL world-to-camera",
        "depth_semantics": "positive -z_cam",
        "near_is_smaller": True,
        "units": "FLAME model scale; not guaranteed metric",
        "near_plane": near_plane,
        "depth_min": float(np.min(camera_depth)),
        "depth_max": float(np.max(camera_depth)),
    }


def pixel3dmm_crop_transform(
    crop_ymin_ymax_xmin_xmax: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    source_sha256: str,
    crop_source_sha256: str,
    frame_id: int,
    crop_frame_id: int,
    checkpoint_image_size: np.ndarray,
    tracking_size: int = PIXEL3DMM_TRACKING_SIZE,
) -> dict:
    """Build the exact pixel-center inverse for Pixel3DMM's stored crop."""

    bounds = np.asarray(crop_ymin_ymax_xmin_xmax)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("Pixel3DMM crop metadata must have four finite values")
    if not np.all(bounds == np.round(bounds)):
        raise ValueError("Pixel3DMM crop metadata must contain integer bounds")
    ymin, ymax, xmin, xmax = [int(value) for value in bounds]
    source_height = int(source_height)
    source_width = int(source_width)
    tracking_size = int(tracking_size)
    if source_height < 2 or source_width < 2 or tracking_size < 2:
        raise ValueError("Pixel3DMM image sizes must be at least two pixels")
    if not (0 <= xmin < xmax <= source_width):
        raise ValueError("Pixel3DMM horizontal crop is outside the source image")
    if not (0 <= ymin < ymax <= source_height):
        raise ValueError("Pixel3DMM vertical crop is outside the source image")
    crop_width = xmax - xmin
    crop_height = ymax - ymin
    if crop_width < 2 or crop_height < 2:
        raise ValueError("Pixel3DMM crop has insufficient pixels")

    source_sha256 = str(source_sha256).lower()
    crop_source_sha256 = str(crop_source_sha256).lower()
    if not _SHA256_PATTERN.fullmatch(source_sha256):
        raise ValueError("Pixel3DMM source SHA256 is invalid")
    if not _SHA256_PATTERN.fullmatch(crop_source_sha256):
        raise ValueError("Pixel3DMM crop source SHA256 is invalid")
    if source_sha256 != crop_source_sha256:
        raise ValueError("Pixel3DMM crop belongs to a different source image")
    frame_id = int(frame_id)
    crop_frame_id = int(crop_frame_id)
    if frame_id < 0 or crop_frame_id < 0 or frame_id != crop_frame_id:
        raise ValueError("Pixel3DMM crop frame ID does not match the checkpoint")
    checkpoint_image_size = np.asarray(checkpoint_image_size).reshape(-1)
    if (
        checkpoint_image_size.shape != (2,)
        or not np.all(np.isfinite(checkpoint_image_size))
        or not np.all(checkpoint_image_size == np.round(checkpoint_image_size))
        or not np.all(checkpoint_image_size == tracking_size)
    ):
        raise ValueError(
            "Pixel3DMM checkpoint image size does not match the tracking grid"
        )

    scale_x = crop_width / tracking_size
    scale_y = crop_height / tracking_size
    tracking_to_source = np.array(
        [
            [scale_x, 0.0, xmin + 0.5 * scale_x - 0.5],
            [0.0, scale_y, ymin + 0.5 * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_to_tracking = np.linalg.inv(tracking_to_source)
    return {
        "stored_order": "ymin,ymax,xmin,xmax",
        "stored_bounds_semantics": "exclusive NumPy slice upper bounds",
        "stored_bounds_yxyx": [ymin, ymax, xmin, xmax],
        "source_height": source_height,
        "source_width": source_width,
        "source_sha256": source_sha256,
        "crop_source_sha256": crop_source_sha256,
        "frame_id": frame_id,
        "crop_frame_id": crop_frame_id,
        "checkpoint_image_size": [
            int(checkpoint_image_size[0]),
            int(checkpoint_image_size[1]),
        ],
        "prediction_size": PIXEL3DMM_PREDICTION_SIZE,
        "tracking_size": tracking_size,
        "crop_height_pixels": crop_height,
        "crop_width_pixels": crop_width,
        "tracking_to_source_matrix": tracking_to_source,
        "source_to_tracking_matrix": source_to_tracking,
        "resize_coordinate_convention": "half-pixel centers",
    }


def project_pixel3dmm_camera_vertices(
    camera_vertices: np.ndarray,
    crop: dict,
    *,
    focal_length: float,
    principal_point: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Project OpenGL camera vertices to original source-image pixels."""

    vertices = np.asarray(camera_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Pixel3DMM camera vertices must have shape [V, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Pixel3DMM camera vertices contain non-finite values")
    depth = -vertices[:, 2]
    if np.any(depth <= 0):
        raise ValueError("Pixel3DMM camera vertices must have negative OpenGL Z")
    tracking_size = int(crop.get("tracking_size", 0))
    if tracking_size < 2:
        raise ValueError("Pixel3DMM crop has no valid tracking size")
    tracking_to_source = np.asarray(
        crop.get("tracking_to_source_matrix"),
        dtype=np.float64,
    )
    if tracking_to_source.shape != (3, 3) or not np.all(
        np.isfinite(tracking_to_source)
    ):
        raise ValueError("Pixel3DMM crop has no finite inverse transform")
    focal_length = float(focal_length)
    if not math.isfinite(focal_length) or focal_length <= 0:
        raise ValueError("Pixel3DMM focal length must be positive and finite")
    principal_point = np.asarray(principal_point, dtype=np.float64).reshape(-1)
    if principal_point.shape != (2,) or not np.all(
        np.isfinite(principal_point)
    ):
        raise ValueError("Pixel3DMM principal point must contain two values")

    focal_pixels = focal_length * tracking_size
    center_scale = tracking_size / 2.0 + 0.5
    center = center_scale + principal_point * center_scale
    # Nvdiffrast maps NDC pixel centers to half-integer viewport coordinates;
    # this adapter stores array indices, so remove the final half-pixel.
    x_tracking = focal_pixels * vertices[:, 0] / depth + center[0] - 0.5
    y_tracking = -focal_pixels * vertices[:, 1] / depth + center[1] - 0.5
    tracking_h = np.column_stack(
        (
            x_tracking,
            y_tracking,
            np.ones(len(vertices), dtype=np.float64),
        )
    )
    source_h = tracking_h @ tracking_to_source.T
    source_xy = source_h[:, :2] / source_h[:, 2:3]
    projected = np.column_stack((source_xy, depth))
    if not np.all(np.isfinite(projected)):
        raise ValueError("Pixel3DMM projection produced non-finite values")
    return projected.astype(np.float32), {
        "tracking_size": tracking_size,
        "focal_length_normalized": focal_length,
        "focal_length_pixels": float(focal_pixels),
        "principal_point_normalized": [
            float(principal_point[0]),
            float(principal_point[1]),
        ],
        "principal_point_tracking_pixels": [
            float(center[0]),
            float(center[1]),
        ],
        "viewport_to_array_index_offset": -0.5,
        "projection": "OpenGL pinhole with positive -z_cam depth",
        "source_mapping": "stored crop plus half-pixel center inversion",
    }


def rasterize_pixel3dmm_camera_depth(
    projected_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict]:
    """Rasterize source-aligned depth using reciprocal-depth interpolation."""

    from backend.benchmark import vggheads_depth_provider

    projected_vertices = np.asarray(projected_vertices, dtype=np.float64)
    if projected_vertices.ndim != 2 or projected_vertices.shape[1] != 3:
        raise ValueError("Pixel3DMM projected vertices must have shape [V, 3]")
    if not np.all(np.isfinite(projected_vertices)):
        raise ValueError("Pixel3DMM projected vertices contain non-finite values")
    if np.any(projected_vertices[:, 2] <= 0):
        raise ValueError("Pixel3DMM projected depths must be positive")
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
        "background": "nan",
        "silhouette_depth_antialiasing": False,
        "source_alignment_requires_landmark_replay": True,
        "adapter_sha256": _sha256(Path(__file__).resolve()),
        "shared_rasterizer_sha256": _sha256(
            Path(vggheads_depth_provider.__file__).resolve()
        ),
    }
