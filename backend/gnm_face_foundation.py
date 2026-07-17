from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
import urllib.request

import cv2
import numpy as np
from scipy.optimize import least_squares


GNM_REVISION = "9c9419f191edd68644ef5cb6572e238248e9a81c"
GNM_LICENSE = "Apache-2.0"
GNM_MODEL_URL = (
    "https://raw.githubusercontent.com/google/gnm/"
    f"{GNM_REVISION}/gnm/shape/data/versions/v3_0/gnm_head.npz"
)
GNM_MODEL_SHA256 = "e50a702789af51347531e13721df1e5a26dcfc30ced9e243c7cede9fcac5db43"
GNM_MODEL_MAX_BYTES = 64 * 1024 * 1024
GNM_LANDMARKS_URL = (
    "https://raw.githubusercontent.com/google/gnm/"
    f"{GNM_REVISION}/gnm/shape/data/landmarks/head_sparse_68.txt"
)
GNM_LANDMARKS_SHA256 = "8b4b759042cae8b67062794306dae9d60fc7ba11ddad60461ba3e2bfaaeac222"
GNM_LANDMARKS_MAX_BYTES = 16 * 1024
MEDIAPIPE_DLIB_MAPPING_REVISION = "14e7480964c564b35debd878bde359c204388919"
MEDIAPIPE_DLIB_MAPPING_LICENSE = "MIT"
GNM_MAX_FACE_SUPPORT_PIXELS = 55
GNM_MINIMUM_RENDER_COVERAGE = 0.85
GNM_MAXIMUM_POSE_RMS = 0.06
GNM_MAXIMUM_CORRECTION_RATIO = 0.50
GNM_MINIMUM_ALIGNMENT_CORRELATION = 0.65
GNM_MAXIMUM_ALIGNMENT_NORMALIZED_RMSE = 0.35
GNM_HIGH_CONFIDENCE_ALIGNMENT_CORRELATION = 0.80
GNM_HIGH_CONFIDENCE_ALIGNMENT_NORMALIZED_RMSE = 0.26
GNM_GUARDED_CORRECTION_STRENGTH = 0.25
GNM_CENTRAL_CORRECTION_DILATION_PIXELS = 3
GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS = 0.5
GNM_IDENTITY_DIMENSION = 253
GNM_EXPRESSION_DIMENSION = 383

# MediaPipe-to-dlib68 correspondence from PeizhiYan/Mediapipe_2_Dlib_Landmarks.
# A tuple with two entries is averaged before fitting.
MEDIAPIPE_TO_DLIB68 = (
    (127,),
    (234,),
    (93,),
    (132, 58),
    (58, 172),
    (136,),
    (150,),
    (176,),
    (152,),
    (400,),
    (379,),
    (365,),
    (397, 288),
    (361,),
    (323,),
    (454,),
    (356,),
    (70,),
    (63,),
    (105,),
    (66,),
    (107,),
    (336,),
    (296,),
    (334,),
    (293,),
    (300,),
    (168, 6),
    (197, 195),
    (5,),
    (4,),
    (75,),
    (97,),
    (2,),
    (326,),
    (305,),
    (33,),
    (160,),
    (158,),
    (133,),
    (153,),
    (144,),
    (362,),
    (385,),
    (387,),
    (263,),
    (373,),
    (380,),
    (61,),
    (39,),
    (37,),
    (0,),
    (267,),
    (269,),
    (291,),
    (321,),
    (314,),
    (17,),
    (84,),
    (91,),
    (78,),
    (82,),
    (13,),
    (312,),
    (308,),
    (317,),
    (14,),
    (87,),
)


@dataclass(frozen=True)
class GNMSurfaceCandidateSet:
    candidate_surface: np.ndarray
    candidate_stats: dict
    fallback_surface: np.ndarray
    fallback_stats: dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_verified_asset(
    *,
    environment_name: str,
    cache_name: str,
    url: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> Path:
    configured = str(os.getenv(environment_name) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"Configured GNM asset is unavailable: {environment_name}"
            )
        try:
            size = configured_path.stat().st_size
            checksum = _sha256_file(configured_path)
        except OSError as exc:
            raise RuntimeError(
                f"Configured GNM asset could not be verified: {environment_name}"
            ) from exc
        if size > int(maximum_bytes):
            raise RuntimeError(
                f"Configured GNM asset exceeds the size limit: {environment_name}"
            )
        if checksum != expected_sha256:
            raise RuntimeError(
                f"Configured GNM asset checksum mismatch: {environment_name}"
            )
        return configured_path

    cache_path = Path.home() / ".cache" / "3dprintpic" / cache_name
    if cache_path.is_file() and _sha256_file(cache_path) == expected_sha256:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f"{cache_path.name}.",
            suffix=".part",
            delete=False,
        ) as partial_file:
            partial_path = Path(partial_file.name)
            total_bytes = 0
            with urllib.request.urlopen(url, timeout=30) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total_bytes += len(block)
                    if total_bytes > int(maximum_bytes):
                        raise RuntimeError(
                            f"GNM asset download exceeded {int(maximum_bytes)} bytes"
                        )
                    partial_file.write(block)
        checksum = _sha256_file(partial_path)
        if checksum != expected_sha256:
            raise RuntimeError(
                "GNM asset checksum mismatch: "
                f"expected={expected_sha256}, actual={checksum}"
            )
        os.replace(partial_path, cache_path)
        partial_path = None
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
    return cache_path


def resolve_gnm_model() -> Path:
    return _resolve_verified_asset(
        environment_name="GNM_HEAD_MODEL_PATH",
        cache_name="gnm_head_v3.npz",
        url=GNM_MODEL_URL,
        expected_sha256=GNM_MODEL_SHA256,
        maximum_bytes=GNM_MODEL_MAX_BYTES,
    )


def resolve_gnm_landmarks() -> Path:
    return _resolve_verified_asset(
        environment_name="GNM_HEAD_LANDMARKS_PATH",
        cache_name="gnm_head_sparse_68.txt",
        url=GNM_LANDMARKS_URL,
        expected_sha256=GNM_LANDMARKS_SHA256,
        maximum_bytes=GNM_LANDMARKS_MAX_BYTES,
    )


def mediapipe_to_dlib68(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 468:
        raise ValueError("GNM fitting requires at least 468 MediaPipe landmarks")
    mapped = np.asarray(
        [np.mean(points[list(indices), :2], axis=0) for indices in MEDIAPIPE_TO_DLIB68],
        dtype=np.float64,
    )
    if mapped.shape != (68, 2) or not np.all(np.isfinite(mapped)):
        raise ValueError("MediaPipe-to-dlib68 mapping produced invalid landmarks")
    return mapped


def _load_landmark_vertices(path: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    weights = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) != 6:
            raise ValueError("GNM landmark definition must contain three index-weight pairs")
        row_indices = [int(tokens[index]) for index in (0, 2, 4)]
        row_weights = [float(tokens[index]) for index in (1, 3, 5)]
        if min(row_indices) < 0 or max(row_indices) >= vertex_count:
            raise ValueError("GNM landmark definition references an invalid vertex")
        if not math.isclose(sum(row_weights), 1.0, abs_tol=0.01):
            raise ValueError("GNM landmark barycentric weights do not sum to one")
        indices.append(row_indices)
        weights.append(row_weights)
    if len(indices) != 68:
        raise ValueError("GNM sparse landmark definition must contain 68 rows")
    return np.asarray(indices, dtype=np.int32), np.asarray(weights, dtype=np.float64)


def _axis_angle_rotation(axis_angle: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(axis_angle, dtype=np.float64).reshape(3, 1))
    return rotation


def _rasterize_front_surface(
    projected_vertices: np.ndarray,
    triangles: np.ndarray,
    mask: np.ndarray,
    *,
    chunk_size: int = 96,
) -> np.ndarray:
    mask = np.asarray(mask) > 0
    surface = np.full(mask.shape, np.nan, dtype=np.float32)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return surface

    vertices = np.asarray(projected_vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int32)
    triangle_xy = vertices[faces, :2]
    triangle_z = vertices[faces, 2]
    minimum = np.min(triangle_xy, axis=1)
    maximum = np.max(triangle_xy, axis=1)
    denominator = (
        (triangle_xy[:, 1, 1] - triangle_xy[:, 2, 1])
        * (triangle_xy[:, 0, 0] - triangle_xy[:, 2, 0])
        + (triangle_xy[:, 2, 0] - triangle_xy[:, 1, 0])
        * (triangle_xy[:, 0, 1] - triangle_xy[:, 2, 1])
    )
    usable = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)

    for start in range(0, len(rows), max(1, int(chunk_size))):
        stop = min(len(rows), start + max(1, int(chunk_size)))
        x = columns[start:stop].astype(np.float64)[:, None]
        y = rows[start:stop].astype(np.float64)[:, None]
        candidates = (
            usable[None, :]
            & (x >= minimum[None, :, 0] - 1e-7)
            & (x <= maximum[None, :, 0] + 1e-7)
            & (y >= minimum[None, :, 1] - 1e-7)
            & (y <= maximum[None, :, 1] + 1e-7)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            first = (
                (triangle_xy[None, :, 1, 1] - triangle_xy[None, :, 2, 1])
                * (x - triangle_xy[None, :, 2, 0])
                + (triangle_xy[None, :, 2, 0] - triangle_xy[None, :, 1, 0])
                * (y - triangle_xy[None, :, 2, 1])
            ) / denominator[None, :]
            second = (
                (triangle_xy[None, :, 2, 1] - triangle_xy[None, :, 0, 1])
                * (x - triangle_xy[None, :, 2, 0])
                + (triangle_xy[None, :, 0, 0] - triangle_xy[None, :, 2, 0])
                * (y - triangle_xy[None, :, 2, 1])
            ) / denominator[None, :]
        third = 1.0 - first - second
        inside = (
            candidates
            & (first >= -1e-7)
            & (second >= -1e-7)
            & (third >= -1e-7)
        )
        interpolated = (
            first * triangle_z[None, :, 0]
            + second * triangle_z[None, :, 1]
            + third * triangle_z[None, :, 2]
        )
        interpolated[~inside] = -np.inf
        front = np.max(interpolated, axis=1)
        valid = np.isfinite(front)
        surface[rows[start:stop][valid], columns[start:stop][valid]] = front[valid]
    return surface


def _apply_geometry_coefficients(
    template_vertices: np.ndarray,
    identity_basis: np.ndarray,
    expression_basis: np.ndarray,
    identity: np.ndarray,
    expression: np.ndarray,
) -> np.ndarray:
    template = np.asarray(template_vertices, dtype=np.float64)
    identity_basis = np.asarray(identity_basis)
    expression_basis = np.asarray(expression_basis)
    identity = np.asarray(identity, dtype=np.float64)
    expression = np.asarray(expression, dtype=np.float64)
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("GNM template vertices have an invalid shape")
    if identity_basis.shape != (GNM_IDENTITY_DIMENSION, *template.shape):
        raise ValueError("GNM identity basis has an invalid shape")
    if expression_basis.shape != (GNM_EXPRESSION_DIMENSION, *template.shape):
        raise ValueError("GNM expression basis has an invalid shape")
    if identity.shape != (GNM_IDENTITY_DIMENSION,):
        raise ValueError("GNM identity coefficients have an invalid shape")
    if expression.shape != (GNM_EXPRESSION_DIMENSION,):
        raise ValueError("GNM expression coefficients have an invalid shape")
    if not np.all(np.isfinite(identity)) or not np.all(np.isfinite(expression)):
        raise ValueError("GNM geometry coefficients must be finite")
    vertices = (
        template
        + np.einsum("i,ivk->vk", identity, identity_basis)
        + np.einsum("i,ivk->vk", expression, expression_basis)
    )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("GNM conditioned vertices must be finite")
    return vertices


def gnm_surface_alignment_quality(
    depth: np.ndarray,
    surface: np.ndarray,
    face_mask: np.ndarray,
) -> dict:
    depth = np.asarray(depth, dtype=np.float32)
    surface = np.asarray(surface, dtype=np.float32)
    mask = np.asarray(face_mask) > 0
    if depth.ndim != 2 or surface.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("GNM alignment inputs must be matching two-dimensional arrays")
    valid = mask & np.isfinite(depth) & np.isfinite(surface)
    face_pixels = int(np.count_nonzero(mask))
    coverage = float(np.count_nonzero(valid) / max(face_pixels, 1))
    if coverage < GNM_MINIMUM_RENDER_COVERAGE:
        raise ValueError(
            f"GNM alignment surface coverage failed: {coverage:.6f} "
            f"< {GNM_MINIMUM_RENDER_COVERAGE:.6f}"
        )
    design = np.column_stack(
        (
            surface[valid].astype(np.float64),
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        )
    )
    scale, offset = np.linalg.lstsq(
        design,
        depth[valid].astype(np.float64),
        rcond=None,
    )[0]
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
        raise ValueError("GNM surface alignment is not positive and finite")
    aligned = surface.astype(np.float64) * float(scale) + float(offset)
    active_values = depth[valid]
    low, high = np.percentile(active_values, (5.0, 95.0))
    active_span = float(high - low)
    if not math.isfinite(active_span) or active_span <= 1e-8:
        raise ValueError("GNM alignment depth has no usable span")
    aligned_valid = aligned[valid]
    depth_valid = depth[valid].astype(np.float64)
    aligned_centered = aligned_valid - np.mean(aligned_valid)
    depth_centered = depth_valid - np.mean(depth_valid)
    denominator = float(
        np.linalg.norm(aligned_centered) * np.linalg.norm(depth_centered)
    )
    correlation = (
        float(np.dot(aligned_centered, depth_centered) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(aligned_valid - depth_valid))) / active_span
    )
    return {
        "coverage_ratio": coverage,
        "alignment_scale": float(scale),
        "alignment_offset": float(offset),
        "active_span": active_span,
        "alignment_depth_correlation": correlation,
        "alignment_normalized_rmse": normalized_rmse,
    }


def select_gnm_surface_candidate(
    depth: np.ndarray,
    candidates: GNMSurfaceCandidateSet,
    face_mask: np.ndarray,
    *,
    minimum_strict_improvement: float = 1e-6,
) -> tuple[np.ndarray, dict, dict]:
    fallback_quality = gnm_surface_alignment_quality(
        depth,
        candidates.fallback_surface,
        face_mask,
    )
    try:
        candidate_quality = gnm_surface_alignment_quality(
            depth,
            candidates.candidate_surface,
            face_mask,
        )
        correlation_delta = float(
            candidate_quality["alignment_depth_correlation"]
            - fallback_quality["alignment_depth_correlation"]
        )
        rmse_delta = float(
            candidate_quality["alignment_normalized_rmse"]
            - fallback_quality["alignment_normalized_rmse"]
        )
        pareto = bool(
            correlation_delta >= 0.0
            and rmse_delta <= 0.0
            and (
                correlation_delta >= float(minimum_strict_improvement)
                or rmse_delta <= -float(minimum_strict_improvement)
            )
        )
        candidate_error = None
    except Exception as exc:
        candidate_quality = None
        correlation_delta = None
        rmse_delta = None
        pareto = False
        candidate_error = type(exc).__name__
    if pareto:
        surface = candidates.candidate_surface
        provider_stats = candidates.candidate_stats
        selected = "conditioned"
        reason = "candidate_pareto_dominates_live_depth_alignment"
    else:
        surface = candidates.fallback_surface
        provider_stats = candidates.fallback_stats
        selected = "mean_fallback"
        reason = (
            "candidate_alignment_error"
            if candidate_error is not None
            else "candidate_does_not_pareto_dominate_live_depth_alignment"
        )
    return surface, provider_stats, {
        "selected": selected,
        "reason": reason,
        "minimum_strict_improvement": float(minimum_strict_improvement),
        "candidate": candidate_quality,
        "fallback": fallback_quality,
        "correlation_delta": correlation_delta,
        "normalized_rmse_delta": rmse_delta,
        "candidate_error_type": candidate_error,
    }


class GNMMeanFaceFoundation:
    def __init__(
        self,
        model_path: str | Path | None = None,
        landmarks_path: str | Path | None = None,
        *,
        load_geometry_bases: bool = False,
    ) -> None:
        model_path = Path(model_path) if model_path is not None else resolve_gnm_model()
        landmarks_path = (
            Path(landmarks_path) if landmarks_path is not None else resolve_gnm_landmarks()
        )
        with np.load(model_path, allow_pickle=False) as data:
            required = {
                "template_vertex_positions",
                "triangles",
                "vertex_groups",
                "vertex_group_names",
            }
            if load_geometry_bases:
                required.update(
                    {"vertex_identity_basis", "expression_basis"}
                )
            if not required.issubset(data.files):
                raise ValueError("GNM model is missing required geometry arrays")
            vertices = np.asarray(data["template_vertex_positions"], dtype=np.float64)
            identity_basis = (
                np.asarray(data["vertex_identity_basis"], dtype=np.float32)
                if load_geometry_bases
                else None
            )
            expression_basis = (
                np.asarray(data["expression_basis"], dtype=np.float32)
                if load_geometry_bases
                else None
            )
            triangles = np.asarray(data["triangles"], dtype=np.int32)
            groups = np.asarray(data["vertex_groups"], dtype=np.float32)
            group_names = [str(value) for value in data["vertex_group_names"].tolist()]
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("GNM template vertices have an invalid shape")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
            raise ValueError("GNM triangles have an invalid shape")
        if np.min(triangles) < 0 or np.max(triangles) >= len(vertices):
            raise ValueError("GNM triangles reference invalid vertices")
        if groups.ndim != 2 or groups.shape[1] != len(vertices):
            raise ValueError("GNM vertex groups have an invalid shape")
        if "skin" not in group_names:
            raise ValueError("GNM model does not define a skin vertex group")
        landmark_indices, landmark_weights = _load_landmark_vertices(
            landmarks_path,
            len(vertices),
        )
        landmark_vertices = vertices[landmark_indices]
        landmarks = np.sum(landmark_vertices * landmark_weights[..., None], axis=1)
        skin = groups[group_names.index("skin")] > 0.5
        skin_triangles = triangles[np.all(skin[triangles], axis=1)]
        if not len(skin_triangles):
            raise ValueError("GNM model has no skin triangles")

        self.vertices = vertices
        self.identity_basis = identity_basis
        self.expression_basis = expression_basis
        self.skin_triangles = skin_triangles
        self.landmarks = landmarks
        self.landmark_indices = landmark_indices
        self.landmark_weights = landmark_weights

    def fit_and_render(
        self,
        media_pipe_landmarks_xy: np.ndarray,
        face_mask: np.ndarray,
        *,
        identity: np.ndarray | None = None,
        expression: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        geometry_conditioned = identity is not None or expression is not None
        if geometry_conditioned:
            if identity is None or expression is None:
                raise ValueError(
                    "GNM conditioning requires both identity and expression"
                )
            if self.identity_basis is None or self.expression_basis is None:
                raise RuntimeError(
                    "GNM geometry bases were not requested for this provider"
                )
            vertices = _apply_geometry_coefficients(
                self.vertices,
                self.identity_basis,
                self.expression_basis,
                identity,
                expression,
            )
            landmark_vertices = vertices[self.landmark_indices]
            landmarks = np.sum(
                landmark_vertices * self.landmark_weights[..., None],
                axis=1,
            )
        else:
            vertices = self.vertices
            landmarks = self.landmarks
        target_pixels = mediapipe_to_dlib68(media_pipe_landmarks_xy)
        face_mask = (np.asarray(face_mask) > 0).astype(np.uint8)
        if face_mask.ndim != 2 or not np.any(face_mask):
            raise ValueError("GNM face mask must be a non-empty 2D array")

        target_center = np.mean(target_pixels, axis=0)
        target_scale = float(np.max(np.ptp(target_pixels, axis=0)))
        if not math.isfinite(target_scale) or target_scale <= 1e-6:
            raise ValueError("GNM target landmarks have no usable scale")
        target = (target_pixels - target_center) / target_scale

        weights = np.ones(68, dtype=np.float64)
        weights[:17] = 0.25
        weights[17:27] = 0.70
        sqrt_weights = np.sqrt(weights)[:, None]

        def residual(parameters: np.ndarray) -> np.ndarray:
            rotation = _axis_angle_rotation(parameters[:3])
            rotated = landmarks @ rotation.T
            scale = math.exp(float(parameters[3]))
            projected = scale * np.column_stack(
                (rotated[:, 0], -rotated[:, 1])
            ) + parameters[4:6]
            return ((projected - target) * sqrt_weights).reshape(-1)

        result = least_squares(
            residual,
            np.zeros(6, dtype=np.float64),
            loss="soft_l1",
            f_scale=0.04,
            max_nfev=200,
            bounds=(
                np.asarray(
                    [-math.pi, -math.pi, -math.pi, -2.5, -5.0, -5.0],
                    dtype=np.float64,
                ),
                np.asarray(
                    [math.pi, math.pi, math.pi, 5.0, 5.0, 5.0],
                    dtype=np.float64,
                ),
            ),
        )
        weighted_rms = float(np.sqrt(np.mean(residual(result.x) ** 2)))
        if (
            not result.success
            or not math.isfinite(weighted_rms)
            or weighted_rms > GNM_MAXIMUM_POSE_RMS
        ):
            raise ValueError(
                f"GNM pose fit exceeded the residual gate: rms={weighted_rms:.6f}"
            )

        rotation = _axis_angle_rotation(result.x[:3])
        rotated_vertices = vertices @ rotation.T
        normalized_scale = math.exp(float(result.x[3]))
        projected_normalized = normalized_scale * np.column_stack(
            (rotated_vertices[:, 0], -rotated_vertices[:, 1])
        ) + result.x[4:6]
        projected_pixels = projected_normalized * target_scale + target_center
        projected_depth = normalized_scale * rotated_vertices[:, 2] * target_scale
        projected_vertices = np.column_stack((projected_pixels, projected_depth))
        surface = _rasterize_front_surface(
            projected_vertices,
            self.skin_triangles,
            face_mask,
        )
        face_pixels = int(np.count_nonzero(face_mask))
        covered_pixels = int(np.count_nonzero(np.isfinite(surface) & (face_mask > 0)))
        coverage = float(covered_pixels / max(face_pixels, 1))
        if coverage < GNM_MINIMUM_RENDER_COVERAGE:
            raise ValueError(
                f"GNM surface coverage failed: {coverage:.6f} "
                f"< {GNM_MINIMUM_RENDER_COVERAGE:.6f}"
            )
        return surface, {
            "enabled": True,
            "method": (
                "gnm-conditioned-head-weak-perspective-zbuffer"
                if geometry_conditioned
                else "gnm-mean-head-weak-perspective-zbuffer"
            ),
            "geometry_conditioned": geometry_conditioned,
            "gnm_revision": GNM_REVISION,
            "gnm_license": GNM_LICENSE,
            "mapping_revision": MEDIAPIPE_DLIB_MAPPING_REVISION,
            "mapping_license": MEDIAPIPE_DLIB_MAPPING_LICENSE,
            "pose_weighted_rms": weighted_rms,
            "rotation_axis_angle": [float(value) for value in result.x[:3]],
            "normalized_scale": normalized_scale,
            "normalized_translation": [float(value) for value in result.x[4:6]],
            "face_pixels": face_pixels,
            "covered_pixels": covered_pixels,
            "coverage_ratio": coverage,
            "skin_triangles": int(len(self.skin_triangles)),
            "identity_coefficient_rms": (
                float(np.sqrt(np.mean(np.square(identity))))
                if identity is not None
                else 0.0
            ),
            "expression_coefficient_rms": (
                float(np.sqrt(np.mean(np.square(expression))))
                if expression is not None
                else 0.0
            ),
        }


_DEFAULT_PROVIDER: GNMMeanFaceFoundation | None = None


def active_gnm_conditioned_feature_requirements() -> dict:
    provider = _DEFAULT_PROVIDER
    return {
        "face_blendshapes": bool(
            provider is not None
            and getattr(provider, "requires_face_blendshapes", False)
        )
    }


def active_gnm_detection_kwargs() -> dict:
    requirements = active_gnm_conditioned_feature_requirements()
    return (
        {"output_face_blendshapes": True}
        if requirements["face_blendshapes"]
        else {}
    )


def get_gnm_mean_face_foundation() -> GNMMeanFaceFoundation:
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = GNMMeanFaceFoundation()
    return _DEFAULT_PROVIDER


def fuse_gnm_face_foundation(
    depth: np.ndarray,
    gnm_surface: np.ndarray,
    face_mask: np.ndarray,
    *,
    correction_region_mask: np.ndarray | None = None,
    preserve_detail_mask: np.ndarray | None = None,
    max_face_support_pixels: int = GNM_MAX_FACE_SUPPORT_PIXELS,
    maximum_correction_ratio: float = GNM_MAXIMUM_CORRECTION_RATIO,
) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    surface = np.asarray(gnm_surface, dtype=np.float32)
    mask = np.asarray(face_mask) > 0
    if depth.ndim != 2 or surface.shape != depth.shape or mask.shape != depth.shape:
        raise ValueError("GNM fusion inputs must be matching two-dimensional arrays")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("GNM fusion face mask is empty")
    support_height = int(rows.max() - rows.min() + 1)
    support_width = int(columns.max() - columns.min() + 1)
    if max(support_height, support_width) > int(max_face_support_pixels):
        return depth.copy(), np.zeros_like(depth), {
            "enabled": False,
            "reason": "face_support_above_small_face_gate",
            "support_height_pixels": support_height,
            "support_width_pixels": support_width,
            "maximum_support_pixels": int(max_face_support_pixels),
        }

    valid = mask & np.isfinite(depth) & np.isfinite(surface)
    coverage = float(np.count_nonzero(valid) / max(np.count_nonzero(mask), 1))
    if coverage < GNM_MINIMUM_RENDER_COVERAGE:
        raise ValueError(
            f"GNM fusion surface coverage failed: {coverage:.6f} "
            f"< {GNM_MINIMUM_RENDER_COVERAGE:.6f}"
        )
    design = np.column_stack(
        (
            surface[valid].astype(np.float64),
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        )
    )
    scale, offset = np.linalg.lstsq(
        design,
        depth[valid].astype(np.float64),
        rcond=None,
    )[0]
    if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
        raise ValueError("GNM surface alignment is not positive and finite")

    aligned = surface.astype(np.float64) * float(scale) + float(offset)
    active_values = depth[valid]
    low, high = np.percentile(active_values, (5.0, 95.0))
    active_span = float(high - low)
    if not math.isfinite(active_span) or active_span <= 1e-8:
        raise ValueError("GNM fusion face depth has no usable span")
    aligned_valid = aligned[valid]
    depth_valid = depth[valid].astype(np.float64)
    aligned_centered = aligned_valid - np.mean(aligned_valid)
    depth_centered = depth_valid - np.mean(depth_valid)
    correlation_denominator = float(
        np.linalg.norm(aligned_centered) * np.linalg.norm(depth_centered)
    )
    alignment_correlation = (
        float(np.dot(aligned_centered, depth_centered) / correlation_denominator)
        if correlation_denominator > 1e-12
        else 0.0
    )
    alignment_normalized_rmse = float(
        np.sqrt(np.mean(np.square(aligned_valid - depth_valid)))
        / active_span
    )
    reliability = {
        "alignment_depth_correlation": alignment_correlation,
        "alignment_normalized_rmse": alignment_normalized_rmse,
        "minimum_alignment_correlation": GNM_MINIMUM_ALIGNMENT_CORRELATION,
        "maximum_alignment_normalized_rmse": (
            GNM_MAXIMUM_ALIGNMENT_NORMALIZED_RMSE
        ),
        "high_confidence_alignment_correlation": (
            GNM_HIGH_CONFIDENCE_ALIGNMENT_CORRELATION
        ),
        "high_confidence_alignment_normalized_rmse": (
            GNM_HIGH_CONFIDENCE_ALIGNMENT_NORMALIZED_RMSE
        ),
    }
    if (
        alignment_correlation < GNM_MINIMUM_ALIGNMENT_CORRELATION
        or alignment_normalized_rmse > GNM_MAXIMUM_ALIGNMENT_NORMALIZED_RMSE
    ):
        return depth.copy(), np.zeros_like(depth), {
            "enabled": False,
            "reason": "alignment_reliability_gate",
            "support_height_pixels": support_height,
            "support_width_pixels": support_width,
            "maximum_support_pixels": int(max_face_support_pixels),
            "coverage_ratio": coverage,
            "alignment_scale": float(scale),
            "alignment_offset": float(offset),
            "active_span": active_span,
            "reliability_tier": "skip",
            **reliability,
        }
    high_confidence = bool(
        alignment_correlation >= GNM_HIGH_CONFIDENCE_ALIGNMENT_CORRELATION
        and alignment_normalized_rmse
        <= GNM_HIGH_CONFIDENCE_ALIGNMENT_NORMALIZED_RMSE
    )
    correction_strength = (
        1.0 if high_confidence else GNM_GUARDED_CORRECTION_STRENGTH
    )
    correction_limit = active_span * max(0.0, float(maximum_correction_ratio))
    correction = np.clip(
        aligned - depth.astype(np.float64),
        -correction_limit,
        correction_limit,
    )
    correction[~valid] = 0.0

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance - 1.0) / 0.5, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight *= mask
    correction *= weight * correction_strength
    correction_region_weight = mask.astype(np.float32)
    correction_region_core = np.zeros_like(mask, dtype=bool)
    if correction_region_mask is not None:
        correction_region_core = np.asarray(correction_region_mask) > 0
        if correction_region_core.shape != depth.shape:
            raise ValueError(
                "GNM correction region must match the face-depth crop"
            )
        correction_region_core &= mask
        if not np.any(correction_region_core):
            return depth.copy(), np.zeros_like(depth), {
                "enabled": False,
                "reason": "empty_correction_region",
                "support_height_pixels": support_height,
                "support_width_pixels": support_width,
                "maximum_support_pixels": int(max_face_support_pixels),
                "coverage_ratio": coverage,
                "alignment_scale": float(scale),
                "alignment_offset": float(offset),
                "active_span": active_span,
                "reliability_tier": "high" if high_confidence else "guarded",
                "correction_strength": 0.0,
                "requested_correction_strength": correction_strength,
                "correction_region": {
                    "enabled": False,
                    "method": "detected-central-face-parts-dilated-feather",
                    "core_pixels": 0,
                    "weighted_pixels": 0,
                    "dilation_radius_pixels": (
                        GNM_CENTRAL_CORRECTION_DILATION_PIXELS
                    ),
                    "feather_sigma_pixels": (
                        GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS
                    ),
                },
                **reliability,
            }
        diameter = GNM_CENTRAL_CORRECTION_DILATION_PIXELS * 2 + 1
        expanded = cv2.dilate(
            correction_region_core.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (diameter, diameter),
            ),
        ).astype(np.float32)
        correction_region_weight = cv2.GaussianBlur(
            expanded,
            (0, 0),
            sigmaX=GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS,
            sigmaY=GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS,
            borderType=cv2.BORDER_REPLICATE,
        )
        correction_region_weight = np.clip(
            correction_region_weight,
            0.0,
            1.0,
        )
        correction_region_weight *= mask.astype(np.float32)
        correction *= correction_region_weight
    detail_preservation_weight = np.zeros_like(depth, dtype=np.float32)
    detail_preservation_core = np.zeros_like(mask, dtype=bool)
    if preserve_detail_mask is not None:
        detail_preservation_core = np.asarray(preserve_detail_mask) > 0
        if detail_preservation_core.shape != depth.shape:
            raise ValueError(
                "GNM detail-preservation mask must match the face-depth crop"
            )
        detail_preservation_core &= mask
        if np.any(detail_preservation_core):
            expanded = cv2.dilate(
                detail_preservation_core.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ).astype(np.float32)
            detail_preservation_weight = cv2.GaussianBlur(
                expanded,
                (0, 0),
                sigmaX=0.75,
                sigmaY=0.75,
                borderType=cv2.BORDER_REPLICATE,
            )
            maximum_weight = float(np.max(detail_preservation_weight))
            if maximum_weight > 1e-8:
                detail_preservation_weight /= maximum_weight
            detail_preservation_weight = np.maximum(
                detail_preservation_weight,
                detail_preservation_core.astype(np.float32),
            )
            detail_preservation_weight *= mask.astype(np.float32)
            correction *= 1.0 - detail_preservation_weight
    refined = depth.copy()
    finite_depth = np.isfinite(depth)
    refined[finite_depth] = (
        depth[finite_depth].astype(np.float64) + correction[finite_depth]
    ).astype(np.float32)
    boundary = mask & (distance <= 1.0 + 1e-6)
    applied_weight = weight.astype(np.float32) * correction_region_weight
    applied_weight *= 1.0 - detail_preservation_weight
    return refined, applied_weight, {
        "enabled": True,
        "method": "bounded-camera-aligned-gnm-mean-face",
        "support_height_pixels": support_height,
        "support_width_pixels": support_width,
        "maximum_support_pixels": int(max_face_support_pixels),
        "coverage_ratio": coverage,
        "alignment_scale": float(scale),
        "alignment_offset": float(offset),
        "active_span": active_span,
        "reliability_tier": "high" if high_confidence else "guarded",
        "correction_strength": correction_strength,
        "correction_region": {
            "enabled": correction_region_mask is not None,
            "method": "detected-central-face-parts-dilated-feather",
            "core_pixels": int(np.count_nonzero(correction_region_core)),
            "weighted_pixels": int(
                np.count_nonzero(correction_region_weight > 1e-4)
            ),
            "dilation_radius_pixels": (
                GNM_CENTRAL_CORRECTION_DILATION_PIXELS
            ),
            "feather_sigma_pixels": (
                GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS
            ),
        },
        "detail_preservation": {
            "enabled": bool(np.any(detail_preservation_core)),
            "method": "detected-expression-parts-hard-core-feathered-margin",
            "core_pixels": int(np.count_nonzero(detail_preservation_core)),
            "weighted_pixels": int(
                np.count_nonzero(detail_preservation_weight > 1e-4)
            ),
            "maximum_core_abs_correction": (
                float(np.max(np.abs(correction[detail_preservation_core])))
                if np.any(detail_preservation_core)
                else 0.0
            ),
        },
        **reliability,
        "maximum_correction_ratio": float(maximum_correction_ratio),
        "correction_limit": correction_limit,
        "effective_correction_limit": correction_limit * correction_strength,
        "max_abs_correction": float(np.max(np.abs(correction))),
        "mean_abs_correction": float(np.mean(np.abs(correction[mask]))),
        "boundary_max_abs_correction": (
            float(np.max(np.abs(correction[boundary]))) if np.any(boundary) else 0.0
        ),
    }
