"""Decode pinned MHR coefficients into crop-aligned floating camera-Z."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.mhr_face_training_corpus import (
    MHR_EXPRESSION_DIMENSION,
    MHR_HEAD_IDENTITY_SLICE,
    MHR_HEAD_MASK_RELATIVE_PATH,
    MHR_IDENTITY_DIMENSION,
    MHR_MODEL_RELATIVE_PATH,
    _decode_head_coefficients,
    _load_model,
    preflight_mhr_root,
)
from backend.benchmark.mesh_rendering import CameraSpec, camera_transform
from backend.benchmark.vggheads_depth_provider import (
    rasterize_projected_mesh_depth,
)


CAMERA_FOV_Y_DEGREES = 32.0
MINIMUM_TARGET_FACE_PIXELS = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_head_mask(mhr_root: str | Path) -> np.ndarray:
    root = Path(mhr_root)
    preflight = preflight_mhr_root(root)
    if not preflight["runnable"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError("MHR preflight failed: " + ", ".join(failed))
    with np.load(root / MHR_HEAD_MASK_RELATIVE_PATH) as masks:
        head_mask = np.asarray(masks["head_mask"], dtype=np.float32)
    return head_mask


def load_mhr_geometry_runtime(
    mhr_root: str | Path,
    *,
    device: str = "cpu",
):
    root = Path(mhr_root)
    head_mask = load_head_mask(root)
    model = _load_model(root, device)
    return (
        model,
        head_mask,
        {
            "model_sha256": _sha256(root / MHR_MODEL_RELATIVE_PATH),
            "head_mask_sha256": _sha256(root / MHR_HEAD_MASK_RELATIVE_PATH),
            "device": str(device),
        },
    )


def expand_head_identity(head_coefficients: np.ndarray) -> np.ndarray:
    values = np.asarray(head_coefficients, dtype=np.float32).reshape(-1)
    expected = MHR_HEAD_IDENTITY_SLICE[1] - MHR_HEAD_IDENTITY_SLICE[0]
    if values.shape != (expected,) or not np.all(np.isfinite(values)):
        raise ValueError(f"MHR head identity must contain {expected} finite values")
    identity = np.zeros(MHR_IDENTITY_DIMENSION, dtype=np.float32)
    identity[MHR_HEAD_IDENTITY_SLICE[0] : MHR_HEAD_IDENTITY_SLICE[1]] = values
    return identity


def validate_expression(expression: np.ndarray) -> np.ndarray:
    values = np.asarray(expression, dtype=np.float32).reshape(-1)
    if values.shape != (MHR_EXPRESSION_DIMENSION,) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"MHR expression must contain {MHR_EXPRESSION_DIMENSION} finite values"
        )
    return values


def _target_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    support = np.asarray(mask) > 0
    if support.ndim != 2:
        raise ValueError("MHR target support must be two-dimensional")
    rows, columns = np.where(support)
    if len(rows) < MINIMUM_TARGET_FACE_PIXELS:
        raise ValueError("MHR target support is too small")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def camera_space_vertices(
    vertices: np.ndarray,
    *,
    yaw_degrees: float,
    elevation_degrees: float,
    distance: float,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("MHR vertices must have finite shape [N, 3]")
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    center = 0.5 * (lower + upper)
    extent = float(np.max(upper - lower))
    if extent <= 1e-8:
        raise ValueError("MHR vertices have no usable extent")
    scale = 1.45 / extent
    homogeneous = np.column_stack((values, np.ones(len(values))))
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, :3] *= scale
    normalization[:3, 3] = -scale * center
    rotation = camera_transform(
        CameraSpec(float(yaw_degrees), float(elevation_degrees))
    )
    render_to_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    render_to_camera[2, 3] = float(distance)
    object_to_camera = render_to_camera @ rotation @ normalization
    camera = homogeneous @ object_to_camera.T
    return camera[:, :3], {
        "center_native": center.tolist(),
        "scale_to_max_extent_1_45": scale,
        "object_to_camera": object_to_camera.tolist(),
    }


def project_camera_vertices(
    camera_vertices: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(camera_vertices, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Camera vertices must have shape [N, 3]")
    if matrix.shape != (3, 3):
        raise ValueError("Camera intrinsics must have shape [3, 3]")
    if np.any(vertices[:, 2] <= 1e-6):
        raise ValueError("MHR camera vertices cross the near plane")
    pixels = vertices @ matrix.T
    pixels[:, :2] /= pixels[:, 2:3]
    pixels[:, 2] = vertices[:, 2]
    return pixels


def project_head_to_support(
    vertices: np.ndarray,
    support_mask: np.ndarray,
    *,
    yaw_degrees: float,
    elevation_degrees: float,
    fov_y_degrees: float = CAMERA_FOV_Y_DEGREES,
    iterations: int = 6,
) -> tuple[np.ndarray, dict]:
    support = np.asarray(support_mask) > 0
    height, width = support.shape
    x0, y0, x1, y1 = _target_bbox(support)
    target_height = float(y1 - y0)
    target_center_x = 0.5 * (x0 + x1 - 1)
    target_center_y = 0.5 * (y0 + y1 - 1)
    fov = float(fov_y_degrees)
    if not 5.0 <= fov <= 120.0:
        raise ValueError("MHR projection FOV is invalid")
    focal = 0.5 * max(height - 1, 1) / np.tan(np.deg2rad(0.5 * fov))
    distance = max(1.8, focal * 1.45 / max(target_height, 1.0))
    projected = None
    transform = None
    intrinsics = None
    for _ in range(max(1, int(iterations))):
        camera, transform = camera_space_vertices(
            vertices,
            yaw_degrees=yaw_degrees,
            elevation_degrees=elevation_degrees,
            distance=distance,
        )
        intrinsics = np.asarray(
            (
                (focal, 0.0, 0.5 * (width - 1)),
                (0.0, focal, 0.5 * (height - 1)),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        projected = project_camera_vertices(camera, intrinsics)
        rendered_height = float(np.ptp(projected[:, 1]))
        if rendered_height <= 1e-6:
            raise ValueError("MHR projection has no vertical span")
        distance *= rendered_height / target_height
    camera, transform = camera_space_vertices(
        vertices,
        yaw_degrees=yaw_degrees,
        elevation_degrees=elevation_degrees,
        distance=distance,
    )
    projected = project_camera_vertices(camera, intrinsics)
    projected_center_x = 0.5 * (
        float(np.min(projected[:, 0])) + float(np.max(projected[:, 0]))
    )
    projected_center_y = 0.5 * (
        float(np.min(projected[:, 1])) + float(np.max(projected[:, 1]))
    )
    intrinsics = intrinsics.copy()
    intrinsics[0, 2] += target_center_x - projected_center_x
    intrinsics[1, 2] += target_center_y - projected_center_y
    projected = project_camera_vertices(camera, intrinsics)
    return projected.astype(np.float32), {
        "projection": "perspective-opencv-floating-camera-z",
        "yaw_degrees": float(yaw_degrees),
        "elevation_degrees": float(elevation_degrees),
        "fov_y_degrees": fov,
        "distance_normalized_units": float(distance),
        "intrinsics": intrinsics.tolist(),
        "target_bbox_xyxy": [x0, y0, x1, y1],
        "projected_bbox_xyxy": [
            float(np.min(projected[:, 0])),
            float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])),
            float(np.max(projected[:, 1])),
        ],
        "normalization": transform,
    }


def render_mhr_camera_depth(
    model,
    head_mask: np.ndarray,
    head_identity: np.ndarray,
    expression: np.ndarray,
    support_mask: np.ndarray,
    *,
    yaw_degrees: float,
    elevation_degrees: float,
    device: str = "cpu",
) -> tuple[np.ndarray, dict]:
    identity = expand_head_identity(head_identity)
    expression = validate_expression(expression)
    mesh = _decode_head_coefficients(
        model,
        identity,
        expression,
        np.asarray(head_mask, dtype=np.float32),
        str(device),
    )
    projected, camera = project_head_to_support(
        np.asarray(mesh.vertices, dtype=np.float32),
        support_mask,
        yaw_degrees=yaw_degrees,
        elevation_degrees=elevation_degrees,
    )
    depth, raster = rasterize_projected_mesh_depth(
        projected,
        np.asarray(mesh.faces, dtype=np.int32),
        height=int(np.asarray(support_mask).shape[0]),
        width=int(np.asarray(support_mask).shape[1]),
        front_surface="minimum-z",
    )
    finite = np.isfinite(depth)
    return depth, {
        "camera": camera,
        "raster": raster,
        "finite_pixels": int(np.count_nonzero(finite)),
        "coverage_ratio": float(np.mean(finite)),
        "head_vertices": int(len(mesh.vertices)),
        "head_faces": int(len(mesh.faces)),
    }


def crop_mask(mask_path: str | Path, bbox: tuple[int, int, int, int]) -> np.ndarray:
    values = np.asarray(Image.open(mask_path).convert("L")) >= 128
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if x0 < 0 or y0 < 0 or x1 > values.shape[1] or y1 > values.shape[0]:
        raise ValueError("MHR crop bbox lies outside the mask")
    cropped = values[y0:y1, x0:x1]
    _target_bbox(cropped)
    return cropped
