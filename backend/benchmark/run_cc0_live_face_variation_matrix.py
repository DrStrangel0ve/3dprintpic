"""Run a privacy-safe generated relief-context matrix against a live backend.

All rendered inputs, request records, and raw responses are kept below the
specified clean server checkout's ignored ``backend/output`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np
from PIL import Image

from backend.benchmark.face_part_metrics import (
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.makehuman_face_fixture import (
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import (
    CameraSpec,
    RenderConfig,
    make_procedural_mesh,
    render_mesh,
)
from backend.benchmark.run_makehuman_face_depth_smoke import (
    _correlation,
    _make_scene,
    _resize_nan_aware,
)
from backend.benchmark.run_background_photo_detail_sweep import _detail_metrics
from backend.benchmark.run_private_background_photo_detail_replay import (
    _require_ignored,
    _selection_source_fingerprint,
    _sha256,
)
from backend.benchmark.run_relief_scene_regression import _boundary_shape_metrics
from backend.benchmark.run_relief_visual_sweep import (
    BACKGROUND_APPEARANCE_GATES,
    FACE_APPEARANCE_GATES,
    SELECTION_APPEARANCE_GATES,
    _appearance_checks,
    _stl_heightfield_agreement,
)
from backend.benchmark.summarize_private_live_api_background_replay import (
    _independent_background_checks,
    _independent_cap_checks,
    _topology_record,
)
from backend.face_depth_refinement import FACE_PART_NAMES


DEFAULT_ASSET_DIR = Path(__file__).parent / "assets" / "makehuman_cc0_heads"
DEFAULT_OUTPUT_NAME = "cc0-live-face-variation-matrix"
BACKGROUND_FIELDS = (
    "background_photo_detail_mm",
    "selection_background_depth_ratio",
)
BASELINE_DETAIL_MM = 0.0
CANDIDATE_DEFAULT_DETAIL_MM = 0.60
DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO = 0.65
RELIEF_HEIGHT_MM = 30.0
MAX_XY_SIZE_MM = 96.0
EXACT_FACE_DEPTH_GATES = {
    "minimum_coverage_ratio": 0.98,
    "minimum_shape_correlation": 0.60,
    "minimum_gradient_correlation": 0.20,
    "maximum_normalized_rmse": 0.35,
}
BACKGROUND_PROFILES = {
    "structured_room",
    "deep_shelves",
    "layered_studio",
}
LIGHTING_PROFILES = {
    "soft_left": {
        "ambient": 0.42,
        "diffuse": 0.53,
        "specular": 0.035,
        "shininess": 48.0,
        "direction": (-0.35, -0.20, 0.90),
    },
    "side_right": {
        "ambient": 0.34,
        "diffuse": 0.61,
        "specular": 0.025,
        "shininess": 42.0,
        "direction": (0.62, -0.12, 0.78),
    },
    "overhead": {
        "ambient": 0.38,
        "diffuse": 0.57,
        "specular": 0.03,
        "shininess": 54.0,
        "direction": (-0.08, -0.68, 0.73),
    },
}
_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
PRODUCER_PATHS = (
    "backend/benchmark/run_cc0_live_face_variation_matrix.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/run_private_background_photo_detail_replay.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
    "backend/benchmark/summarize_private_live_api_background_replay.py",
    "backend/face_depth_refinement.py",
    "backend/benchmark/assets/makehuman_cc0_heads/asset.json",
)


@dataclass(frozen=True)
class OccluderSpec:
    """A normalized, deliberately small rectangular foreground occluder."""

    left: float
    top: float
    right: float
    bottom: float
    rgb: tuple[int, int, int] = (47, 71, 83)
    anchor: str = "image"
    opacity: float = 1.0


@dataclass(frozen=True)
class FaceSceneSpec:
    row_id: str
    profile_name: str
    target_dimension: int
    camera_yaw_deg: float
    camera_distance: float
    camera_elevation_deg: float = 0.0
    camera_scale: float = 1.0
    horizontal_offset: float = 0.0
    background_profile: str = "structured_room"
    lighting_profile: str = "soft_left"
    occluder: OccluderSpec | None = None


@dataclass(frozen=True)
class ObjectSceneSpec:
    row_id: str
    procedural_index: int
    target_dimension: int
    camera_yaw_deg: float
    camera_distance: float
    camera_elevation_deg: float = 0.0
    camera_scale: float = 1.0
    horizontal_offset: float = 0.0
    background_profile: str = "deep_shelves"
    lighting_profile: str = "side_right"


SceneSpec = FaceSceneSpec | ObjectSceneSpec


# Keep the recommended cheap smoke first: a small, off-axis, yawed face at 256 px.
DEFAULT_MATRIX = (
    FaceSceneSpec(
        row_id="small_off_axis_yaw_256",
        profile_name="asian_female_asymmetric",
        target_dimension=256,
        camera_yaw_deg=24.0,
        camera_distance=7.5,
        horizontal_offset=0.16,
    ),
    FaceSceneSpec(
        row_id="centered_neutral_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=0.0,
        camera_distance=3.6,
    ),
    FaceSceneSpec(
        row_id="offset_occluded_turn_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=-20.0,
        camera_distance=4.0,
        camera_scale=1.05,
        horizontal_offset=-0.13,
        occluder=OccluderSpec(
            0.39,
            0.49,
            0.61,
            0.62,
            rgb=(0, 0, 0),
            anchor="eye_band",
            opacity=0.40,
        ),
    ),
    FaceSceneSpec(
        row_id="close_positive_turn_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=14.0,
        camera_distance=3.25,
        camera_scale=1.08,
        horizontal_offset=0.06,
    ),
)


VARIED_CONTEXT_MATRIX = (
    FaceSceneSpec(
        row_id="small_side_lit_shelves_256",
        profile_name="asian_female_asymmetric",
        target_dimension=256,
        camera_yaw_deg=29.0,
        camera_elevation_deg=-4.0,
        camera_distance=7.2,
        horizontal_offset=0.18,
        background_profile="deep_shelves",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="eyewear_overhead_panel_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=-17.0,
        camera_elevation_deg=3.0,
        camera_distance=4.1,
        horizontal_offset=-0.11,
        background_profile="layered_studio",
        lighting_profile="overhead",
        occluder=OccluderSpec(
            0.39,
            0.49,
            0.61,
            0.62,
            rgb=(8, 13, 18),
            anchor="eye_band",
            opacity=0.60,
        ),
    ),
    FaceSceneSpec(
        row_id="strong_turn_layered_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=32.0,
        camera_elevation_deg=-2.0,
        camera_distance=3.55,
        camera_scale=1.04,
        horizontal_offset=0.08,
        background_profile="layered_studio",
        lighting_profile="soft_left",
    ),
    ObjectSceneSpec(
        row_id="multilobe_object_shelves_384",
        procedural_index=5,
        target_dimension=384,
        camera_yaw_deg=31.0,
        camera_elevation_deg=-9.0,
        camera_distance=3.8,
        camera_scale=1.08,
        horizontal_offset=-0.06,
        background_profile="deep_shelves",
        lighting_profile="side_right",
    ),
)


# Privacy-safe fitting rows for feature-local face depth experiments. Keep these
# separate from VARIED_CONTEXT_MATRIX, which remains the held-out regression set.
FACE_FUSION_TRAIN_MATRIX = (
    FaceSceneSpec(
        row_id="fusion_asian_small_left_256",
        profile_name="asian_female_asymmetric",
        target_dimension=256,
        camera_yaw_deg=-28.0,
        camera_elevation_deg=2.0,
        camera_distance=7.3,
        horizontal_offset=-0.15,
        background_profile="structured_room",
        lighting_profile="soft_left",
    ),
    FaceSceneSpec(
        row_id="fusion_asian_small_front_384",
        profile_name="asian_female_asymmetric",
        target_dimension=384,
        camera_yaw_deg=0.0,
        camera_elevation_deg=-3.0,
        camera_distance=6.8,
        horizontal_offset=0.14,
        background_profile="deep_shelves",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="fusion_asian_medium_right_256",
        profile_name="asian_female_asymmetric",
        target_dimension=256,
        camera_yaw_deg=18.0,
        camera_distance=4.7,
        background_profile="layered_studio",
        lighting_profile="overhead",
    ),
    FaceSceneSpec(
        row_id="fusion_asian_medium_left_384",
        profile_name="asian_female_asymmetric",
        target_dimension=384,
        camera_yaw_deg=-17.0,
        camera_elevation_deg=-2.0,
        camera_distance=4.4,
        horizontal_offset=-0.08,
        background_profile="deep_shelves",
        lighting_profile="overhead",
    ),
    FaceSceneSpec(
        row_id="fusion_asian_close_right_384",
        profile_name="asian_female_asymmetric",
        target_dimension=384,
        camera_yaw_deg=18.0,
        camera_elevation_deg=2.0,
        camera_distance=3.35,
        camera_scale=1.04,
        horizontal_offset=0.06,
        background_profile="structured_room",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="fusion_african_small_right_256",
        profile_name="african_male_neutral",
        target_dimension=256,
        camera_yaw_deg=31.0,
        camera_elevation_deg=3.0,
        camera_distance=7.1,
        horizontal_offset=0.16,
        background_profile="layered_studio",
        lighting_profile="overhead",
    ),
    FaceSceneSpec(
        row_id="fusion_african_small_left_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=-30.0,
        camera_elevation_deg=-3.0,
        camera_distance=6.7,
        horizontal_offset=-0.10,
        background_profile="structured_room",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="fusion_african_medium_front_256",
        profile_name="african_male_neutral",
        target_dimension=256,
        camera_yaw_deg=0.0,
        camera_elevation_deg=-2.0,
        camera_distance=4.5,
        horizontal_offset=-0.10,
        background_profile="deep_shelves",
        lighting_profile="soft_left",
    ),
    FaceSceneSpec(
        row_id="fusion_african_medium_right_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=16.0,
        camera_elevation_deg=1.0,
        camera_distance=4.2,
        horizontal_offset=0.08,
        background_profile="structured_room",
        lighting_profile="overhead",
    ),
    FaceSceneSpec(
        row_id="fusion_african_close_left_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=-15.0,
        camera_elevation_deg=3.0,
        camera_distance=3.3,
        camera_scale=1.05,
        horizontal_offset=-0.05,
        background_profile="layered_studio",
        lighting_profile="soft_left",
    ),
    FaceSceneSpec(
        row_id="fusion_caucasian_small_left_256",
        profile_name="caucasian_female_smile",
        target_dimension=256,
        camera_yaw_deg=-32.0,
        camera_elevation_deg=-3.0,
        camera_distance=7.4,
        horizontal_offset=-0.17,
        background_profile="deep_shelves",
        lighting_profile="overhead",
    ),
    FaceSceneSpec(
        row_id="fusion_caucasian_small_right_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=28.0,
        camera_elevation_deg=2.0,
        camera_distance=6.9,
        horizontal_offset=0.11,
        background_profile="layered_studio",
        lighting_profile="soft_left",
    ),
    FaceSceneSpec(
        row_id="fusion_caucasian_medium_front_256",
        profile_name="caucasian_female_smile",
        target_dimension=256,
        camera_yaw_deg=0.0,
        camera_elevation_deg=-1.0,
        camera_distance=4.6,
        horizontal_offset=0.10,
        background_profile="structured_room",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="fusion_caucasian_medium_left_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=-16.0,
        camera_elevation_deg=2.0,
        camera_distance=4.3,
        horizontal_offset=-0.07,
        background_profile="layered_studio",
        lighting_profile="side_right",
    ),
    FaceSceneSpec(
        row_id="fusion_caucasian_close_front_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=0.0,
        camera_distance=3.25,
        camera_scale=1.06,
        background_profile="deep_shelves",
        lighting_profile="soft_left",
    ),
)
OBJECT_SMOKE_MATRIX = (VARIED_CONTEXT_MATRIX[-1],)


def _validate_scene_spec(spec: SceneSpec) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", spec.row_id):
        raise ValueError(f"Unsafe row identifier: {spec.row_id!r}")
    if not 64 <= int(spec.target_dimension) <= 1024:
        raise ValueError("target_dimension must be between 64 and 1024")
    numeric = (
        spec.camera_yaw_deg,
        spec.camera_elevation_deg,
        spec.camera_distance,
        spec.camera_scale,
        spec.horizontal_offset,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("Scene camera controls must be finite")
    if not -75.0 <= float(spec.camera_yaw_deg) <= 75.0:
        raise ValueError("camera_yaw_deg must be between -75 and 75")
    if not -45.0 <= float(spec.camera_elevation_deg) <= 45.0:
        raise ValueError("camera_elevation_deg must be between -45 and 45")
    if not 1.5 <= float(spec.camera_distance) <= 12.0:
        raise ValueError("camera_distance must be between 1.5 and 12")
    if not 0.5 <= float(spec.camera_scale) <= 2.0:
        raise ValueError("camera_scale must be between 0.5 and 2")
    if not -0.45 <= float(spec.horizontal_offset) <= 0.45:
        raise ValueError("horizontal_offset must be between -0.45 and 0.45")
    if spec.background_profile not in BACKGROUND_PROFILES:
        raise ValueError("Unknown background_profile")
    if spec.lighting_profile not in LIGHTING_PROFILES:
        raise ValueError("Unknown lighting_profile")
    if isinstance(spec, ObjectSceneSpec):
        if int(spec.procedural_index) < 0:
            raise ValueError("procedural_index must be non-negative")
        return
    if spec.occluder is not None:
        occ = spec.occluder
        coordinates = (occ.left, occ.top, occ.right, occ.bottom)
        if not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError("Occluder bounds must be finite")
        if not (
            0.0 <= occ.left < occ.right <= 1.0 and 0.0 <= occ.top < occ.bottom <= 1.0
        ):
            raise ValueError("Occluder bounds must be ordered normalized coordinates")
        width = float(occ.right - occ.left)
        height = float(occ.bottom - occ.top)
        if width > 0.35 or height > 0.35 or width * height > 0.10:
            raise ValueError("Occluder exceeds the bounded foreground allowance")
        if len(occ.rgb) != 3 or any(not 0 <= int(value) <= 255 for value in occ.rgb):
            raise ValueError("Occluder RGB values must be bytes")
        if occ.anchor not in {"image", "eye_band"}:
            raise ValueError("Occluder anchor must be image or eye_band")
        if not 0.0 < float(occ.opacity) <= 1.0:
            raise ValueError("Occluder opacity must be in (0, 1]")


def _translate(values: np.ndarray, columns: int, fill) -> np.ndarray:
    array = np.asarray(values)
    shifted = np.empty_like(array)
    shifted[...] = fill
    if columns == 0:
        shifted[...] = array
    elif columns > 0 and columns < array.shape[1]:
        shifted[:, columns:] = array[:, : array.shape[1] - columns]
    elif columns < 0 and -columns < array.shape[1]:
        shifted[:, :columns] = array[:, -columns:]
    return shifted


def _scene_phase(row_id: str) -> float:
    # A stable identifier-derived phase avoids Python's randomized hash seed.
    import hashlib

    prefix = hashlib.sha256(row_id.encode("ascii")).digest()[:8]
    return int.from_bytes(prefix, "big") / float(2**64) * 2.0 * math.pi


def _background_rgb(profile: str) -> tuple[float, float, float]:
    return {
        "structured_room": (0.84, 0.87, 0.91),
        "deep_shelves": (0.76, 0.80, 0.84),
        "layered_studio": (0.82, 0.78, 0.73),
    }[profile]


def _render_config(spec: SceneSpec, effective_distance: float) -> RenderConfig:
    lighting = LIGHTING_PROFILES[spec.lighting_profile]
    return RenderConfig(
        size=int(spec.target_dimension),
        projection="perspective",
        perspective_fov_y_deg=32.0,
        camera_distance=effective_distance,
        background_rgb=_background_rgb(spec.background_profile),
        ambient=float(lighting["ambient"]),
        diffuse=float(lighting["diffuse"]),
        specular=float(lighting["specular"]),
        shininess=float(lighting["shininess"]),
        light_direction=tuple(float(value) for value in lighting["direction"]),
    )


def _compose_scene(
    rendered_depth: np.ndarray,
    selection_mask: np.ndarray,
    selection_rgb: np.ndarray,
    *,
    phase: float,
    background_profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    if background_profile == "structured_room":
        return _make_scene(
            rendered_depth,
            selection_mask,
            selection_rgb,
            phase=phase,
        )

    rows, columns = np.indices(rendered_depth.shape, dtype=np.float32)
    x = 2.0 * columns / max(rendered_depth.shape[1] - 1, 1) - 1.0
    y = 2.0 * rows / max(rendered_depth.shape[0] - 1, 1) - 1.0
    if background_profile == "deep_shelves":
        depth = 0.73 + 0.075 * x + 0.055 * y
        depth += 0.025 * np.sin(2.2 * np.pi * x + phase)
        left_bay = (x < -0.30) & (y > -0.72) & (y < 0.72)
        right_bay = (x > 0.34) & (y > -0.58) & (y < 0.82)
        shelf_a = (y > -0.18) & (y < -0.09) & (x < 0.48)
        shelf_b = (y > 0.38) & (y < 0.48) & (x > -0.70)
        depth = np.where(left_bay, depth - 0.105, depth)
        depth = np.where(right_bay, depth + 0.085, depth)
        depth = np.where(shelf_a | shelf_b, depth - 0.055, depth)
        depth = np.clip(depth, 0.50, 0.96)
        rgb = np.stack(
            (
                0.34 + 0.34 * (1.0 - depth) + 0.05 * np.cos(4.2 * y),
                0.40 + 0.23 * x + 0.05 * np.sin(3.4 * y + phase),
                0.46 + 0.20 * y + 0.04 * np.cos(3.1 * x),
            ),
            axis=-1,
        )
        rgb[left_bay] *= np.asarray((0.82, 0.72, 0.62), dtype=np.float32)
        rgb[right_bay] = 0.72 * rgb[right_bay] + 0.28 * np.asarray(
            (0.38, 0.55, 0.66), dtype=np.float32
        )
        rgb[shelf_a | shelf_b] *= np.asarray((0.62, 0.58, 0.54), dtype=np.float32)
    elif background_profile == "layered_studio":
        depth = 0.74 + 0.065 * x - 0.045 * y
        depth += 0.035 * np.cos(2.0 * np.pi * y - 0.4 * phase)
        diagonal = y > 0.46 * x + 0.16
        inset = (x > -0.76) & (x < -0.22) & (y > -0.58) & (y < 0.42)
        plinth = (y > 0.46) & (x > -0.18) & (x < 0.78)
        depth = np.where(diagonal, depth + 0.075, depth)
        depth = np.where(inset, depth - 0.095, depth)
        depth = np.where(plinth, depth - 0.045, depth)
        depth = np.clip(depth, 0.52, 0.96)
        rgb = np.stack(
            (
                0.48 + 0.22 * (1.0 - depth) + 0.04 * np.sin(3.3 * x),
                0.43 + 0.13 * x + 0.06 * np.cos(2.7 * y + phase),
                0.38 + 0.16 * y + 0.04 * np.sin(2.1 * x - y),
            ),
            axis=-1,
        )
        rgb[inset] = 0.70 * rgb[inset] + 0.30 * np.asarray(
            (0.31, 0.46, 0.56), dtype=np.float32
        )
        rgb[plinth] *= np.asarray((0.70, 0.78, 0.86), dtype=np.float32)
    else:
        raise ValueError(f"Unknown background profile: {background_profile}")

    texture = 0.014 * np.sin(67.0 * x + 41.0 * y + phase)
    rgb = np.clip(rgb + texture[..., None], 0.0, 1.0).astype(np.float32)
    exact_depth = np.where(
        selection_mask,
        0.08 + 0.50 * np.asarray(rendered_depth, dtype=np.float32),
        depth,
    ).astype(np.float32)
    rgb[selection_mask] = selection_rgb[selection_mask]
    return exact_depth, rgb


def _selection_geometry_record(mask: np.ndarray) -> dict:
    pixels = int(np.count_nonzero(mask))
    if pixels == 0:
        raise ValueError("Rendered selection mask is empty")
    rows, columns = np.where(mask)
    bbox = [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return {
        "visible_selection_pixels": pixels,
        "visible_selection_fraction": pixels / float(mask.size),
        "selection_bbox_xyxy": bbox,
        "selection_bbox_width_pixels": int(width),
        "selection_bbox_height_pixels": int(height),
        "selection_bbox_width_ratio": float(width / mask.shape[1]),
        "selection_bbox_height_ratio": float(height / mask.shape[0]),
    }


def _render_scene_arrays(
    spec: SceneSpec, fixture: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None, dict]:
    _validate_scene_spec(spec)
    if isinstance(spec, ObjectSceneSpec):
        return _render_object_scene_arrays(spec)
    profile = fixture["profiles"].get(spec.profile_name)
    if profile is None:
        raise ValueError(f"Unknown MakeHuman profile: {spec.profile_name}")
    colors = make_profile_vertex_colors(
        profile["mesh"].vertices,
        profile["skin_tone"],
        fixture["part_weights"],
        fixture["surface_weights"],
    )
    effective_distance = float(spec.camera_distance) / float(spec.camera_scale)
    background = _background_rgb(spec.background_profile)
    rendered = render_mesh(
        profile["mesh"],
        CameraSpec(
            azimuth_deg=float(spec.camera_yaw_deg),
            elevation_deg=float(spec.camera_elevation_deg),
        ),
        _render_config(spec, effective_distance),
        (180, 120, 100),
        vertex_part_weights=fixture["part_weights"],
        vertex_colors=colors,
    )
    offset_columns = int(round(float(spec.horizontal_offset) * spec.target_dimension))
    face_mask = _translate(
        np.asarray(rendered.silhouette, dtype=bool), offset_columns, False
    )
    face_rgb = _translate(
        np.asarray(rendered.rgb, dtype=np.float32), offset_columns, background
    )
    rendered_depth = _translate(
        np.asarray(rendered.depth, dtype=np.float32), offset_columns, 1.0
    )
    rendered_part_masks = rendered.part_masks or {}
    required_part_names = set(FACE_PART_NAMES)
    if set(rendered_part_masks) != required_part_names:
        missing = sorted(required_part_names - set(rendered_part_masks))
        unexpected = sorted(set(rendered_part_masks) - required_part_names)
        raise ValueError(
            "Rendered face-part masks must contain exactly the required parts; "
            f"missing={missing}, unexpected={unexpected}"
        )
    exact_part_masks = {}
    for name in FACE_PART_NAMES:
        raw_part = np.asarray(rendered_part_masks[name], dtype=bool)
        if raw_part.shape != face_mask.shape:
            raise ValueError(
                f"Rendered face-part mask {name!r} has shape {raw_part.shape}, "
                f"expected {face_mask.shape}"
            )
        translated = _translate(raw_part, offset_columns, False) & face_mask
        if not np.any(translated):
            raise ValueError(f"Rendered face-part mask {name!r} is empty")
        exact_part_masks[name] = translated
    occluder_pixels = 0
    occluder_bounds = None
    if spec.occluder is not None:
        occ = spec.occluder
        size = int(spec.target_dimension)
        if occ.anchor == "eye_band":
            eye_mask = np.zeros_like(face_mask)
            for name in ("left_eye", "right_eye"):
                eye_mask |= exact_part_masks[name]
            eye_rows, eye_columns = np.where(eye_mask & face_mask)
            face_rows, face_columns = np.where(face_mask)
            if not len(eye_rows) or not len(face_rows):
                raise ValueError("Eye-band occluder requires visible rendered eyes")
            face_width = int(face_columns.max() - face_columns.min() + 1)
            face_height = int(face_rows.max() - face_rows.min() + 1)
            center_y = 0.5 * float(eye_rows.min() + eye_rows.max() + 1)
            half_height = max(3, int(round(face_height * 0.05)))
            horizontal_margin = max(2, int(round(face_width * 0.05)))
            left = max(0, int(eye_columns.min()) - horizontal_margin)
            right = min(size, int(eye_columns.max() + 1) + horizontal_margin)
            top = max(0, int(round(center_y)) - half_height)
            bottom = min(size, int(round(center_y)) + half_height)
        else:
            left = max(0, min(size - 1, int(math.floor(occ.left * size))))
            top = max(0, min(size - 1, int(math.floor(occ.top * size))))
            right = max(left + 1, min(size, int(math.ceil(occ.right * size))))
            bottom = max(top + 1, min(size, int(math.ceil(occ.bottom * size))))
        occluder_pixels = int((right - left) * (bottom - top))
        occluder_bounds = (top, bottom, left, right)

    exact_depth, source_rgb = _compose_scene(
        rendered_depth,
        face_mask,
        face_rgb,
        phase=_scene_phase(spec.row_id),
        background_profile=spec.background_profile,
    )
    if occluder_bounds is not None:
        top, bottom, left, right = occluder_bounds
        opacity = float(spec.occluder.opacity)
        ink = np.asarray(spec.occluder.rgb, dtype=np.float32) / 255.0
        source_rgb[top:bottom, left:right] = (1.0 - opacity) * source_rgb[
            top:bottom, left:right
        ] + opacity * ink

    selection_record = _selection_geometry_record(face_mask)
    mask_pixels = int(selection_record["visible_selection_pixels"])
    source = np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)
    mask = face_mask.astype(np.uint8) * 255
    face_bbox = selection_record["selection_bbox_xyxy"]
    face_width = selection_record["selection_bbox_width_pixels"]
    face_height = selection_record["selection_bbox_height_pixels"]
    background_values = exact_depth[~face_mask]
    return (
        source,
        mask,
        exact_depth.astype(np.float32, copy=False),
        exact_part_masks,
        {
            "scene_kind": "face",
            "background_profile": spec.background_profile,
            "lighting_profile": spec.lighting_profile,
            "effective_camera_distance": effective_distance,
            "horizontal_offset_columns": offset_columns,
            **selection_record,
            "visible_face_pixels": mask_pixels,
            "visible_face_fraction": mask_pixels / float(mask.size),
            "face_bbox_xyxy": face_bbox,
            "face_bbox_width_pixels": int(face_width),
            "face_bbox_height_pixels": int(face_height),
            "face_bbox_width_ratio": float(face_width / spec.target_dimension),
            "face_bbox_height_ratio": float(face_height / spec.target_dimension),
            "background_depth_p02": float(np.percentile(background_values, 2.0)),
            "background_depth_p98": float(np.percentile(background_values, 98.0)),
            "background_depth_span": float(
                np.percentile(background_values, 98.0)
                - np.percentile(background_values, 2.0)
            ),
            "occluder_pixels": occluder_pixels,
            "occluder_anchor": (
                spec.occluder.anchor if spec.occluder is not None else None
            ),
            "occluder_bounds_tblr": (
                list(occluder_bounds) if occluder_bounds is not None else None
            ),
        },
    )


def _render_object_scene_arrays(
    spec: ObjectSceneSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None, dict]:
    effective_distance = float(spec.camera_distance) / float(spec.camera_scale)
    rendered = render_mesh(
        make_procedural_mesh(int(spec.procedural_index)),
        CameraSpec(
            azimuth_deg=float(spec.camera_yaw_deg),
            elevation_deg=float(spec.camera_elevation_deg),
        ),
        _render_config(spec, effective_distance),
        (132, 177, 118),
    )
    background = _background_rgb(spec.background_profile)
    offset_columns = int(round(float(spec.horizontal_offset) * spec.target_dimension))
    selection_mask = _translate(
        np.asarray(rendered.silhouette, dtype=bool), offset_columns, False
    )
    selection_rgb = _translate(
        np.asarray(rendered.rgb, dtype=np.float32), offset_columns, background
    )
    rendered_depth = _translate(
        np.asarray(rendered.depth, dtype=np.float32), offset_columns, 1.0
    )
    exact_depth, source_rgb = _compose_scene(
        rendered_depth,
        selection_mask,
        selection_rgb,
        phase=_scene_phase(spec.row_id),
        background_profile=spec.background_profile,
    )
    selection_record = _selection_geometry_record(selection_mask)
    background_values = exact_depth[~selection_mask]
    return (
        np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8),
        selection_mask.astype(np.uint8) * 255,
        exact_depth.astype(np.float32, copy=False),
        None,
        {
            "scene_kind": "object",
            "procedural_index": int(spec.procedural_index),
            "background_profile": spec.background_profile,
            "lighting_profile": spec.lighting_profile,
            "effective_camera_distance": effective_distance,
            "horizontal_offset_columns": offset_columns,
            **selection_record,
            "background_depth_p02": float(np.percentile(background_values, 2.0)),
            "background_depth_p98": float(np.percentile(background_values, 98.0)),
            "background_depth_span": float(
                np.percentile(background_values, 98.0)
                - np.percentile(background_values, 2.0)
            ),
            "occluder_pixels": 0,
            "occluder_anchor": None,
            "occluder_bounds_tblr": None,
        },
    )


def _artifact_record(path: Path, root: Path) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _git_output(command: list[str], repository: Path) -> str:
    try:
        return subprocess.run(
            command,
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Clean server repository must be a readable git checkout"
        ) from exc


def _clean_server_provenance(repository: Path) -> dict:
    repository = repository.resolve()
    top_level = Path(
        _git_output(["git", "rev-parse", "--show-toplevel"], repository)
    ).resolve()
    if top_level != repository:
        raise ValueError("Clean server repository must be the git checkout root")
    revision = _git_output(["git", "rev-parse", "HEAD"], repository).lower()
    status = _git_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], repository
    )
    provenance = {
        "available": bool(re.fullmatch(r"[a-f0-9]{40}", revision)),
        "revision": revision,
        "clean": not bool(status),
        "status": status.splitlines() if status else [],
    }
    if not provenance["available"] or not provenance["clean"]:
        raise ValueError("Clean server repository provenance is unavailable or dirty")
    return provenance


def _producer_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]
    provenance = _clean_server_provenance(repository)
    provenance["files"] = [
        {
            "path": relative,
            "sha256": _sha256(repository / relative),
        }
        for relative in PRODUCER_PATHS
    ]
    return provenance


def _prepare_output_directory(
    clean_repository: Path, output_dir: Path | None
) -> tuple[Path, Path]:
    server_output = (clean_repository / "backend" / "output").resolve()
    selected = (
        (server_output / DEFAULT_OUTPUT_NAME).resolve()
        if output_dir is None
        else output_dir.resolve()
    )
    try:
        relative = selected.relative_to(server_output)
    except ValueError as exc:
        raise ValueError(
            "Harness output must be beneath the clean server backend/output"
        ) from exc
    if not relative.parts:
        raise ValueError("Harness output must use a child directory of backend/output")
    _require_ignored(server_output, clean_repository)
    _require_ignored(selected, clean_repository)
    selected.mkdir(parents=True, exist_ok=True)
    return server_output, selected


def _json_object(content: bytes, label: str) -> dict:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} did not return a JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return payload


def _request_json(
    client,
    *,
    method: str,
    endpoint: str,
    output_dir: Path,
    record_name: str,
    data: dict[str, str] | None = None,
    upload_path: Path | None = None,
    omitted_background_fields: Iterable[str] = (),
) -> tuple[dict, dict, Path]:
    started = time.perf_counter()
    if upload_path is None:
        response = client.get(endpoint)
        upload_record = None
    else:
        with upload_path.open("rb") as upload_handle:
            response = client.post(
                endpoint,
                files={"file": (upload_path.name, upload_handle, "image/png")},
                data=data or {},
            )
        upload_record = {
            "field": "file",
            "filename": upload_path.name,
            "content_type": "image/png",
            "size_bytes": int(upload_path.stat().st_size),
            "sha256": _sha256(upload_path),
        }
    elapsed = time.perf_counter() - started
    response_path = output_dir / f"{record_name}_response.json"
    response_path.write_bytes(bytes(response.content))
    fields = {str(key): str(value) for key, value in sorted((data or {}).items())}
    record = {
        "schema_version": 1,
        "method": method.upper(),
        "endpoint": endpoint,
        "request_fields": fields,
        "posted_form_fields": sorted(fields),
        "omitted_background_fields": sorted(set(omitted_background_fields)),
        "upload": upload_record,
        "http_status": int(response.status_code),
        "elapsed_seconds": round(float(elapsed), 6),
        "response_sha256": _sha256(response_path),
        "response_size_bytes": int(response_path.stat().st_size),
    }
    record_path = output_dir / f"{record_name}_request.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not 200 <= int(response.status_code) < 300:
        text = bytes(response.content)[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{endpoint} failed with HTTP {response.status_code}: {text}"
        )
    return _json_object(bytes(response.content), endpoint), record, record_path


def _server_runtime_provenance(payload: dict) -> dict:
    provenance = payload.get("runtime", {}).get("implementation_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Server response has no runtime implementation provenance")
    return provenance


def _assert_runtime_provenance(payload: dict, expected: dict) -> dict:
    actual = _server_runtime_provenance(payload)
    if not (
        actual.get("available") is True
        and actual.get("clean") is True
        and str(actual.get("revision", "")).lower() == expected["revision"]
        and not actual.get("status")
    ):
        raise RuntimeError(
            "Live server runtime provenance is dirty or does not match the clean checkout"
        )
    return actual


def _assert_number(payload: dict, key: str, expected: float) -> None:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Live response has no numeric {key}") from exc
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Live response {key} mismatch: expected {expected}, got {value}"
        )


def _assert_process_contract(
    payload: dict,
    *,
    spec: SceneSpec,
    selection_job_id: str,
    expected_provenance: dict,
    expected_detail_mm: float,
) -> None:
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise RuntimeError("Live process response has no valid job identifier")
    requested_dimension = payload.get(
        "requested_target_dimension", payload.get("target_dimension")
    )
    if int(requested_dimension) != int(spec.target_dimension):
        raise RuntimeError("Live response target_dimension does not match the request")
    _assert_number(payload, "z_scale", RELIEF_HEIGHT_MM)
    _assert_number(payload, "max_xy_size", MAX_XY_SIZE_MM)
    _assert_number(payload, "background_photo_detail_mm", expected_detail_mm)
    _assert_number(
        payload,
        "selection_background_depth_ratio",
        DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    )
    context = payload.get("selection_depth_context", {})
    if context.get("selection_job_id") != selection_job_id:
        raise RuntimeError("Live response does not preserve the composed selection job")
    _assert_runtime_provenance(payload, expected_provenance)


def _stage_scene(
    row_dir: Path, spec: SceneSpec, fixture: dict
) -> tuple[Path, Path, Path, dict[str, Path] | None, dict]:
    source, mask, exact_depth, exact_part_masks, render_record = _render_scene_arrays(
        spec, fixture
    )
    source_path = row_dir / "source.png"
    mask_path = row_dir / "selection_mask.png"
    exact_depth_path = row_dir / "exact_depth.npy"
    Image.fromarray(source, mode="RGB").save(source_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    np.save(exact_depth_path, exact_depth)
    part_paths = None
    if exact_part_masks is not None:
        if set(exact_part_masks) != set(FACE_PART_NAMES):
            raise ValueError("Staged face row has incomplete exact face-part masks")
        part_dir = row_dir / "exact_face_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_paths = {}
        for name in FACE_PART_NAMES:
            path = part_dir / f"{name}.png"
            Image.fromarray(
                np.asarray(exact_part_masks[name], dtype=np.uint8) * 255,
                mode="L",
            ).save(path)
            part_paths[name] = path
    return source_path, mask_path, exact_depth_path, part_paths, render_record


def _process_form(
    spec: SceneSpec, selection_job_id: str, *, baseline: bool
) -> dict[str, str]:
    form = {
        "selection_job_id": selection_job_id,
        "target_dimension": str(int(spec.target_dimension)),
        "z_scale": f"{RELIEF_HEIGHT_MM:g}",
        "max_xy_size": f"{MAX_XY_SIZE_MM:g}",
    }
    if baseline:
        form["background_photo_detail_mm"] = f"{BASELINE_DETAIL_MM:g}"
    if not baseline and any(field in form for field in BACKGROUND_FIELDS):
        raise AssertionError("Candidate request must omit both background fields")
    return form


def _job_artifacts(server_output: Path, response: dict) -> dict[str, Path]:
    job_id = str(response.get("job_id", ""))
    if not _JOB_ID.fullmatch(job_id):
        raise RuntimeError("Cannot resolve artifacts for an invalid job identifier")
    job_dir = (server_output / job_id).resolve()
    if job_dir.parent != server_output.resolve():
        raise RuntimeError("Live job artifacts escaped the clean server output root")
    artifacts = {
        "surface": job_dir / "output_surface.npy",
        "reference_surface": job_dir / "output_reference_surface.npy",
        "stl": job_dir / "output_model.stl",
    }
    refinement = response.get("face_refinement", {})
    depth_name = str(refinement.get("depth_file") or "")
    if depth_name:
        if Path(depth_name).name != depth_name:
            raise RuntimeError("Live face refinement has an unsafe depth artifact name")
        artifacts["refined_depth"] = job_dir / depth_name
    elif bool(refinement.get("applied", False)):
        raise RuntimeError("Applied live face refinement has no depth artifact name")
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Live job is missing artifacts: " + ", ".join(missing))
    return artifacts


def _mask_on_emitted_grid(
    mask_path: Path,
    surface_grid_transform: dict,
    emitted_shape: tuple[int, int],
) -> np.ndarray:
    required = {
        "input_depth_shape",
        "target_depth_shape",
        "mesh_shape_before_crop",
        "crop_bbox_rc",
        "emitted_shape",
        "flip_x",
    }
    missing = sorted(required - set(surface_grid_transform))
    if missing:
        raise ValueError("Surface-grid transform is missing: " + ", ".join(missing))

    input_rows, input_columns = (
        int(value) for value in surface_grid_transform["input_depth_shape"]
    )
    target_rows, target_columns = (
        int(value) for value in surface_grid_transform["target_depth_shape"]
    )
    with Image.open(mask_path) as loaded:
        mask = (
            np.asarray(
                loaded.convert("L").resize(
                    (input_columns, input_rows), Image.Resampling.NEAREST
                )
            )
            >= 128
        )
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (target_columns, target_rows), Image.Resampling.NEAREST
            )
        )
        >= 128
    )
    if bool(surface_grid_transform["flip_x"]):
        mask = np.flip(mask, axis=1)

    mesh_rows, mesh_columns = (
        int(value) for value in surface_grid_transform["mesh_shape_before_crop"]
    )
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask = (
        np.asarray(
            mask_image.resize((mesh_columns, mesh_rows), Image.Resampling.NEAREST)
        )
        >= 128
    )
    top, left, bottom, right = (
        int(value) for value in surface_grid_transform["crop_bbox_rc"]
    )
    mask = mask[top:bottom, left:right]
    transform_shape = tuple(
        int(value) for value in surface_grid_transform["emitted_shape"]
    )
    if mask.shape != transform_shape or mask.shape != emitted_shape:
        raise ValueError(
            f"Transformed selection mask has shape {mask.shape}, expected "
            f"{transform_shape} and emitted surface shape {emitted_shape}"
        )
    return mask


def _exact_face_depth_quality(
    refined_depth_path: Path,
    exact_depth_path: Path,
    mask_path: Path,
    *,
    expected_scale_sign: float,
    part_mask_paths: dict[str, Path] | None = None,
    require_face_parts: bool = True,
) -> dict:
    predicted = np.load(refined_depth_path).astype(np.float32)
    exact = np.load(exact_depth_path).astype(np.float32)
    with Image.open(mask_path) as loaded:
        face = (
            np.asarray(
                loaded.convert("L").resize(
                    (exact.shape[1], exact.shape[0]), Image.Resampling.NEAREST
                )
            )
            >= 128
        )
    if predicted.shape != exact.shape:
        predicted = _resize_nan_aware(predicted, exact.shape)
    reference_valid = face & np.isfinite(exact)
    candidate_valid = face & np.isfinite(predicted)
    valid = reference_valid & candidate_valid
    reference_coverage = float(
        np.count_nonzero(reference_valid) / max(np.count_nonzero(face), 1)
    )
    coverage = float(
        np.count_nonzero(valid) / max(np.count_nonzero(reference_valid), 1)
    )
    if np.count_nonzero(reference_valid) < 64 or np.count_nonzero(valid) < 64:
        return {
            "available": False,
            "coverage_ratio": coverage,
            "reference_coverage_ratio": reference_coverage,
            "reason": (
                "insufficient_reference_samples"
                if np.count_nonzero(reference_valid) < 64
                else "insufficient_candidate_samples"
            ),
            "gates": dict(EXACT_FACE_DEPTH_GATES),
            "checks": {"passed": False},
        }

    reference = exact.astype(np.float64)
    candidate = predicted.astype(np.float64)
    design = np.column_stack((candidate[valid], np.ones(np.count_nonzero(valid))))
    scale, shift = np.linalg.lstsq(design, reference[valid], rcond=None)[0]
    aligned = candidate * float(scale) + float(shift)
    reference_span = float(
        np.percentile(reference[valid], 95.0) - np.percentile(reference[valid], 5.0)
    )
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(aligned[valid] - reference[valid])))
        / max(reference_span, 1e-8)
    )

    interior = valid.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    interior[1:-1, 1:-1] &= (
        valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    )
    reference_gradients = np.gradient(reference)
    candidate_gradients = np.gradient(aligned)
    gradient_correlation = min(
        _correlation(ref[interior], cand[interior])
        for ref, cand in zip(reference_gradients, candidate_gradients)
    )
    metrics = {
        "available": True,
        "coverage_ratio": coverage,
        "reference_coverage_ratio": reference_coverage,
        "shape_correlation": _correlation(reference[valid], aligned[valid]),
        "gradient_correlation": float(gradient_correlation),
        "normalized_rmse": normalized_rmse,
        "affine_scale": float(scale),
        "affine_shift": float(shift),
        "samples": int(np.count_nonzero(valid)),
        "gates": dict(EXACT_FACE_DEPTH_GATES),
    }
    named_part_shape = None
    named_part_affine_mm = None
    if require_face_parts:
        if part_mask_paths is None or set(part_mask_paths) != set(FACE_PART_NAMES):
            metrics["face_part_reason"] = "missing_exact_face_part_masks"
        else:
            part_masks = {}
            for name in FACE_PART_NAMES:
                with Image.open(part_mask_paths[name]) as loaded:
                    part = np.asarray(
                        loaded.convert("L").resize(
                            (exact.shape[1], exact.shape[0]),
                            Image.Resampling.NEAREST,
                        )
                    ) >= 128
                if not np.any(part & face):
                    raise ValueError(f"Exact face-part mask {name!r} is empty")
                part_masks[name] = part & face

            exact_signal = 1.0 - reference
            exact_min = float(np.min(exact_signal[reference_valid]))
            exact_span = float(
                np.max(exact_signal[reference_valid]) - exact_min
            )
            if not np.isfinite(exact_span) or exact_span <= 1e-8:
                raise ValueError("Exact depth has no usable global span")
            reference_surface_mm = np.full(reference.shape, np.nan, dtype=np.float64)
            reference_surface_mm[reference_valid] = (
                exact_signal[reference_valid] - exact_min
            ) * (RELIEF_HEIGHT_MM / exact_span)
            predicted_signal = (
                1.0 - candidate if float(expected_scale_sign) > 0 else candidate
            )
            design = np.column_stack(
                (predicted_signal[valid], np.ones(np.count_nonzero(valid)))
            )
            part_scale, part_shift = np.linalg.lstsq(
                design, reference_surface_mm[valid], rcond=None
            )[0]
            aligned_surface_mm = np.full(candidate.shape, np.nan, dtype=np.float64)
            aligned_surface_mm[candidate_valid] = (
                predicted_signal[candidate_valid] * float(part_scale)
                + float(part_shift)
            )
            pitch_mm = float(MAX_XY_SIZE_MM / max(exact.shape[1] - 1, 1))
            named_part_shape = face_part_cross_height_metrics(
                reference_surface_mm,
                aligned_surface_mm,
                face,
                part_masks,
                sample_pitch_mm=pitch_mm,
            )
            named_part_affine_mm = face_part_affine_surface_error_metrics(
                reference_surface_mm,
                aligned_surface_mm,
                face,
                part_masks,
            )
            metrics.update(
                {
                    "exact_reference_mapping": {
                        "method": "one-minus-depth-global-span-to-relief-mm",
                        "relief_height_mm": float(RELIEF_HEIGHT_MM),
                        "exact_signal_min": exact_min,
                        "exact_signal_span": exact_span,
                    },
                    "named_part_alignment": {
                        "method": "single-global-face-affine-fit",
                        "scale": float(part_scale),
                        "shift": float(part_shift),
                        "sample_pitch_mm": pitch_mm,
                    },
                    "named_part_shape": named_part_shape,
                    "named_part_affine_mm": named_part_affine_mm,
                }
            )
    checks = {
        "reference_coverage": reference_coverage
        >= EXACT_FACE_DEPTH_GATES["minimum_coverage_ratio"],
        "coverage": coverage >= EXACT_FACE_DEPTH_GATES["minimum_coverage_ratio"],
        "depth_semantics_orientation": bool(
            np.isfinite(scale) and scale * float(expected_scale_sign) > 0
        ),
        "shape_correlation": metrics["shape_correlation"]
        >= EXACT_FACE_DEPTH_GATES["minimum_shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"]
        >= EXACT_FACE_DEPTH_GATES["minimum_gradient_correlation"],
        "normalized_rmse": normalized_rmse
        <= EXACT_FACE_DEPTH_GATES["maximum_normalized_rmse"],
    }
    if require_face_parts:
        checks["named_part_shape"] = bool(
            named_part_shape and named_part_shape.get("passed", False)
        )
        checks["named_part_affine_mm"] = bool(
            named_part_affine_mm and named_part_affine_mm.get("passed", False)
        )
    checks["passed"] = bool(all(checks.values()))
    return {**metrics, "checks": checks}


def _emitted_face_part_retention(
    reference_surface: np.ndarray,
    emitted_surface: np.ndarray,
    mask_path: Path,
    part_mask_paths: dict[str, Path] | None,
    postprocess: dict,
    *,
    required: bool,
) -> dict:
    if not required:
        return {"required": False, "available": False, "checks": {"passed": True}}
    if part_mask_paths is None or set(part_mask_paths) != set(FACE_PART_NAMES):
        return {
            "required": True,
            "available": False,
            "reason": "missing_exact_face_part_masks",
            "checks": {"passed": False},
        }
    transform = postprocess.get("surface_grid_transform", {})
    emitted_shape = tuple(int(value) for value in emitted_surface.shape)
    face = _mask_on_emitted_grid(mask_path, transform, emitted_shape)
    parts = {
        name: _mask_on_emitted_grid(part_mask_paths[name], transform, emitted_shape)
        & face
        for name in FACE_PART_NAMES
    }
    if any(not np.any(mask) for mask in parts.values()):
        raise ValueError("An emitted exact face-part mask is empty")
    try:
        pitch_mm = float(postprocess["mesh_sample_pitch_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid emitted mesh sample pitch") from exc
    if not np.isfinite(pitch_mm) or pitch_mm <= 0:
        raise ValueError("Emitted mesh sample pitch must be positive and finite")
    shape = face_part_cross_height_metrics(
        reference_surface,
        emitted_surface,
        face,
        parts,
        sample_pitch_mm=pitch_mm,
    )
    affine_mm = face_part_affine_surface_error_metrics(
        reference_surface,
        emitted_surface,
        face,
        parts,
    )
    checks = {
        "named_part_shape": bool(shape.get("passed", False)),
        "named_part_affine_mm": bool(affine_mm.get("passed", False)),
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "required": True,
        "available": True,
        "sample_pitch_mm": pitch_mm,
        "named_part_shape": shape,
        "named_part_affine_mm": affine_mm,
        "checks": checks,
    }


def _occlusion_handling(refinement: dict, *, required: bool) -> dict:
    if not required:
        return {
            "required": False,
            "eyewear_detected": False,
            "deoccluded_faces": int(refinement.get("eyewear_deoccluded_faces", 0)),
            "already_consistent_faces": 0,
            "passed": True,
        }
    records = [
        face.get("eyewear_deocclusion", {})
        for face in refinement.get("faces", [])
        if isinstance(face, dict)
    ]
    detected = [
        record
        for record in records
        if bool(record.get("detection", {}).get("enabled", False))
    ]
    consistent = [
        record
        for record in detected
        if record.get("reason") == "source_depth_already_consistent"
    ]
    safely_rejected = [
        record
        for record in detected
        if record.get("reason") == "quality_gate_failed"
        and "bilateral_eye_detail_retention"
        in record.get("quality_gates", {}).get("failures", [])
        and not bool(
            record.get("bilateral_eye_detail_retention", {}).get("passed", True)
        )
    ]
    deoccluded = int(refinement.get("eyewear_deoccluded_faces", 0))
    return {
        "required": True,
        "eyewear_detected": bool(detected),
        "deoccluded_faces": deoccluded,
        "already_consistent_faces": len(consistent),
        "safely_rejected_corrections": len(safely_rejected),
        "passed": bool(
            detected and (deoccluded > 0 or consistent or safely_rejected)
        ),
    }


def _score_variant(
    response: dict,
    *,
    server_output: Path,
    exact_depth_path: Path,
    mask_path: Path,
    require_occlusion: bool,
    part_mask_paths: dict[str, Path] | None = None,
    spec: SceneSpec | None = None,
) -> dict:
    artifacts = _job_artifacts(server_output, response)
    surface = np.load(artifacts["surface"])
    reference = np.load(artifacts["reference_surface"])
    surfaces_valid = bool(
        surface.ndim == 2
        and surface.shape == reference.shape
        and min(surface.shape, default=0) >= 2
        and np.all(np.isfinite(surface))
        and np.all(np.isfinite(reference))
    )

    refinement = response.get("face_refinement", {})
    detected_faces = int(refinement.get("detected_faces", 0))
    refined_faces = int(refinement.get("refined_faces", 0))
    validated_faces = min(detected_faces, refined_faces)
    selected_detail = int(refinement.get("refined_selection_detail_regions", 0))
    selection_fallback = int(refinement.get("selection_detail_fallback_regions", 0))
    refined_regions_total = int(refinement.get("refined_regions_total", 0))
    is_face = not isinstance(spec, ObjectSceneSpec)
    semantic_key = "face" if is_face else "selection_nonface"
    appearance_gates = FACE_APPEARANCE_GATES if is_face else SELECTION_APPEARANCE_GATES
    postprocess = response.get("relief_postprocess", {})
    appearance = postprocess.get("surface_appearance_agreement", {}).get(
        semantic_key, {}
    )
    appearance_checks = _appearance_checks(appearance, appearance_gates)
    background_appearance = postprocess.get("surface_appearance_agreement", {}).get(
        "background", {}
    )
    background_appearance_checks = _appearance_checks(
        background_appearance, BACKGROUND_APPEARANCE_GATES
    )
    background_checks = _independent_background_checks(
        postprocess.get("background_depth_preservation", {})
    )
    cap_checks = _independent_cap_checks(
        postprocess.get("selection_background_physical_cap", {})
    )
    boundary_shape = _boundary_shape_metrics(
        postprocess.get("background_depth_preservation", {}),
        postprocess.get("selection_background_physical_cap", {}),
    )
    topology = _topology_record(response.get("stl_diagnostics", {}))
    shell = _stl_heightfield_agreement(
        artifacts["stl"],
        artifacts["surface"],
        expected_max_xy_size_mm=MAX_XY_SIZE_MM,
    )
    exact_subject_depth = (
        _exact_face_depth_quality(
            artifacts["refined_depth"],
            exact_depth_path,
            mask_path,
            expected_scale_sign=1.0 if bool(response.get("invert", False)) else -1.0,
            part_mask_paths=part_mask_paths,
            require_face_parts=is_face,
        )
        if "refined_depth" in artifacts
        else {
            "available": False,
            "reason": "no_refined_depth_artifact",
            "gates": dict(EXACT_FACE_DEPTH_GATES),
            "checks": {"passed": False},
        }
    )
    emitted_face_part_retention = _emitted_face_part_retention(
        reference,
        surface,
        mask_path,
        part_mask_paths,
        postprocess,
        required=is_face,
    )
    occlusion = _occlusion_handling(
        refinement,
        required=bool(is_face and require_occlusion),
    )
    refinement_route_passed = bool(
        validated_faces > 0
        if is_face
        else selected_detail > 0
        and selection_fallback >= selected_detail
        and refined_regions_total == selected_detail
        and detected_faces == 0
        and refined_faces == 0
    )
    occlusion_passed = bool(
        occlusion["passed"] and (validated_faces > 0 if is_face else True)
    )
    checks = {
        "finite_surface_contract": surfaces_valid,
        "refinement_applied": bool(refinement.get("applied", False)),
        "subject_refinement_route": refinement_route_passed,
        "exact_subject_depth": bool(exact_subject_depth["checks"].get("passed", False)),
        "semantic_appearance": bool(appearance_checks.get("passed", False)),
        "background_appearance": bool(
            background_appearance_checks.get("passed", False)
        ),
        "background_depth": bool(background_checks.get("passed", False)),
        "physical_cap": bool(cap_checks.get("passed", False)),
        "boundary_shape": bool(boundary_shape.get("passed", False)),
        "topology": bool(topology["checks"].get("passed", False)),
        "exact_stl_shell": bool(shell.get("passed", False)),
        "occlusion_deoccluded": occlusion_passed,
        "emitted_face_part_retention": bool(
            emitted_face_part_retention["checks"].get("passed", False)
        ),
    }
    if is_face:
        checks["validated_human_face_refined"] = validated_faces > 0
    else:
        checks["generic_selection_refined"] = selected_detail > 0
        checks["generic_fallback_accounted"] = selection_fallback >= selected_detail
        checks["refined_region_total"] = refined_regions_total == selected_detail
        checks["no_false_human_face"] = detected_faces == 0 and refined_faces == 0
    checks["passed"] = bool(all(checks.values()))
    return {
        "semantic_scope": semantic_key,
        "validated_face_regions": validated_faces,
        "detected_faces": detected_faces,
        "refined_faces": refined_faces,
        "selected_detail_regions": selected_detail,
        "selection_detail_fallback_regions": selection_fallback,
        "refined_regions_total": refined_regions_total,
        "eyewear_deoccluded_faces": int(refinement.get("eyewear_deoccluded_faces", 0)),
        "occlusion_handling": occlusion,
        "appearance": appearance,
        "appearance_checks": appearance_checks,
        "background_appearance": background_appearance,
        "background_appearance_checks": background_appearance_checks,
        "background_depth": postprocess.get("background_depth_preservation", {}),
        "background_depth_checks": background_checks,
        "physical_cap": postprocess.get("selection_background_physical_cap", {}),
        "physical_cap_checks": cap_checks,
        "boundary_shape": boundary_shape,
        "topology": topology,
        "stl_heightfield_agreement": shell,
        "emitted_face_part_retention": emitted_face_part_retention,
        "exact_subject_depth": exact_subject_depth,
        (
            "exact_face_depth" if is_face else "exact_selection_depth"
        ): exact_subject_depth,
        "artifacts": {
            name: _artifact_record(path, server_output)
            for name, path in artifacts.items()
        },
        "checks": checks,
    }


def _score_pair(
    baseline_response: dict,
    candidate_response: dict,
    *,
    source_path: Path,
    mask_path: Path,
    server_output: Path,
) -> dict:
    baseline_artifacts = _job_artifacts(server_output, baseline_response)
    candidate_artifacts = _job_artifacts(server_output, candidate_response)
    baseline_surface = np.load(baseline_artifacts["surface"])
    candidate_surface = np.load(candidate_artifacts["surface"])
    transform = candidate_response.get("relief_postprocess", {}).get(
        "surface_grid_transform", {}
    )
    baseline_transform = baseline_response.get("relief_postprocess", {}).get(
        "surface_grid_transform", {}
    )
    transforms_match = baseline_transform == transform
    face_mask = _mask_on_emitted_grid(mask_path, transform, candidate_surface.shape)
    detail_telemetry = candidate_response.get("relief_postprocess", {}).get(
        "background_photo_detail", {}
    )
    protection_halo_px = float(detail_telemetry.get("protection_halo_px", 13.0))
    detail = _detail_metrics(
        baseline_surface,
        candidate_surface,
        face_mask,
        source_path,
        CANDIDATE_DEFAULT_DETAIL_MM,
        protection_halo_px,
        surface_grid_transform=transform,
    )
    checks = {
        "shared_surface_shape": baseline_surface.shape == candidate_surface.shape,
        "shared_surface_grid_transform": transforms_match,
        "background_photo_detail": bool(detail["checks"].get("passed", False)),
    }
    checks["passed"] = bool(all(checks.values()))
    return {"background_photo_detail": detail, "checks": checks}


def _matrix_coverage(specs: tuple[SceneSpec, ...]) -> dict:
    rows = [spec.row_id for spec in specs]
    dimensions = sorted({int(spec.target_dimension) for spec in specs})
    yaw_signs = sorted(
        {
            "negative"
            if spec.camera_yaw_deg < 0
            else "positive"
            if spec.camera_yaw_deg > 0
            else "neutral"
            for spec in specs
        }
    )
    return {
        "row_ids": rows,
        "scene_kinds": sorted(
            {
                "object" if isinstance(spec, ObjectSceneSpec) else "face"
                for spec in specs
            }
        ),
        "target_dimensions": dimensions,
        "yaw_signs": yaw_signs,
        "background_profiles": sorted({spec.background_profile for spec in specs}),
        "lighting_profiles": sorted({spec.lighting_profile for spec in specs}),
        "occluded_rows": [
            spec.row_id for spec in specs if getattr(spec, "occluder", None) is not None
        ],
        "unique_rows": len(rows) == len(set(rows)),
    }


def run(
    clean_server_repository: str | Path,
    *,
    output_dir: str | Path | None = None,
    base_url: str = "http://127.0.0.1:8005",
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    limit: int | None = None,
    timeout_seconds: float = 900.0,
    specs: Iterable[SceneSpec] = DEFAULT_MATRIX,
    client=None,
) -> dict:
    clean_repository = Path(clean_server_repository).resolve()
    expected_provenance = _clean_server_provenance(clean_repository)
    server_output, run_output = _prepare_output_directory(
        clean_repository,
        Path(output_dir) if output_dir is not None else None,
    )
    all_specs = tuple(specs)
    selected_specs = all_specs
    if not all_specs:
        raise ValueError("Face variation matrix must contain at least one row")
    if len({spec.row_id for spec in selected_specs}) != len(selected_specs):
        raise ValueError("Face variation matrix row identifiers must be unique")
    for spec in selected_specs:
        _validate_scene_spec(spec)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be at least one")
        selected_specs = selected_specs[: int(limit)]

    owns_client = client is None
    live_client = client or httpx.Client(
        base_url=base_url.rstrip("/"), timeout=float(timeout_seconds)
    )
    try:
        health, health_record, health_record_path = _request_json(
            live_client,
            method="GET",
            endpoint="/health",
            output_dir=run_output,
            record_name="health",
        )
        runtime_provenance = _assert_runtime_provenance(health, expected_provenance)
        reported_output = health.get("output_dir")
        if (
            not isinstance(reported_output, str)
            or Path(reported_output).resolve() != server_output
        ):
            raise RuntimeError(
                "Live server output directory does not match the clean checkout"
            )
        producer_provenance = _producer_provenance()
        if producer_provenance["revision"] != expected_provenance["revision"]:
            raise RuntimeError(
                "Matrix producer revision does not match the clean live server"
            )

        if any(isinstance(spec, FaceSceneSpec) for spec in selected_specs):
            fixture = load_makehuman_face_fixture(asset_dir)
        else:
            fixture = {
                "profiles": {},
                "manifest": {
                    "source": "deterministic_generated_procedural_mesh",
                    "generator": (
                        "backend.benchmark.mesh_rendering.make_procedural_mesh"
                    ),
                    "indices": sorted(
                        {
                            int(spec.procedural_index)
                            for spec in selected_specs
                            if isinstance(spec, ObjectSceneSpec)
                        }
                    ),
                },
            }
        rows = []
        for spec in selected_specs:
            row_dir = run_output / "rows" / spec.row_id
            row_dir.mkdir(parents=True, exist_ok=True)
            (
                source_path,
                mask_path,
                exact_depth_path,
                part_mask_paths,
                render_record,
            ) = _stage_scene(row_dir, spec, fixture)
            mask_reference = mask_path.resolve().relative_to(server_output).as_posix()
            compose_form = {
                "mask_paths_json": json.dumps([mask_reference], separators=(",", ":")),
                "background_mode": "neutral",
            }
            compose, compose_record, compose_record_path = _request_json(
                live_client,
                method="POST",
                endpoint="/selection/compose",
                output_dir=row_dir,
                record_name="compose",
                data=compose_form,
                upload_path=source_path,
            )
            selection_job_id = compose.get("job_id")
            if not isinstance(selection_job_id, str) or not _JOB_ID.fullmatch(
                selection_job_id
            ):
                raise RuntimeError(
                    "Selection compose response has no valid job identifier"
                )
            if (
                compose.get("model_status") != "composed-clicked-masks"
                or int(compose.get("mask_count", -1)) != 1
                or compose.get("source_fingerprint")
                != _selection_source_fingerprint(source_path)
            ):
                raise RuntimeError(
                    "Selection compose response does not match the staged source and mask"
                )

            variants = {}
            response_payloads = {}
            for variant, baseline in (("baseline", True), ("candidate", False)):
                form = _process_form(spec, selection_job_id, baseline=baseline)
                omitted = [field for field in BACKGROUND_FIELDS if field not in form]
                response, request_record, request_record_path = _request_json(
                    live_client,
                    method="POST",
                    endpoint="/process_image",
                    output_dir=row_dir,
                    record_name=variant,
                    data=form,
                    upload_path=source_path,
                    omitted_background_fields=omitted,
                )
                expected_detail = (
                    BASELINE_DETAIL_MM if baseline else CANDIDATE_DEFAULT_DETAIL_MM
                )
                _assert_process_contract(
                    response,
                    spec=spec,
                    selection_job_id=selection_job_id,
                    expected_provenance=expected_provenance,
                    expected_detail_mm=expected_detail,
                )
                response_payloads[variant] = response
                variants[variant] = {
                    "job_id": response["job_id"],
                    "request": request_record,
                    "request_artifact": _artifact_record(
                        request_record_path, run_output
                    ),
                    "response_artifact": _artifact_record(
                        row_dir / f"{variant}_response.json", run_output
                    ),
                    "runtime_provenance": _server_runtime_provenance(response),
                }

            for variant, response in response_payloads.items():
                variants[variant]["quality"] = _score_variant(
                    response,
                    server_output=server_output,
                    exact_depth_path=exact_depth_path,
                    mask_path=mask_path,
                    part_mask_paths=part_mask_paths,
                    require_occlusion=getattr(spec, "occluder", None) is not None,
                    spec=spec,
                )
            pair_quality = _score_pair(
                response_payloads["baseline"],
                response_payloads["candidate"],
                source_path=source_path,
                mask_path=mask_path,
                server_output=server_output,
            )
            row_checks = {
                "baseline": bool(variants["baseline"]["quality"]["checks"]["passed"]),
                "candidate": bool(variants["candidate"]["quality"]["checks"]["passed"]),
                "paired_background_detail": bool(pair_quality["checks"]["passed"]),
                "input_background_depth_span": bool(
                    float(render_record.get("background_depth_span", 0.0)) >= 0.15
                ),
            }
            row_checks["passed"] = bool(all(row_checks.values()))

            rows.append(
                {
                    "row_id": spec.row_id,
                    "scene": asdict(spec),
                    "render": render_record,
                    "selection_job_id": selection_job_id,
                    "source": _artifact_record(source_path, run_output),
                    "selection_mask": _artifact_record(mask_path, run_output),
                    "exact_depth": _artifact_record(exact_depth_path, run_output),
                    "exact_face_part_masks": {
                        "applicable": part_mask_paths is not None,
                        "complete": bool(
                            part_mask_paths is not None
                            and set(part_mask_paths) == set(FACE_PART_NAMES)
                        ),
                        "files": {
                            name: _artifact_record(path, run_output)
                            for name, path in sorted((part_mask_paths or {}).items())
                        },
                    },
                    "compose": {
                        "request": compose_record,
                        "request_artifact": _artifact_record(
                            compose_record_path, run_output
                        ),
                        "response_artifact": _artifact_record(
                            row_dir / "compose_response.json", run_output
                        ),
                    },
                    "variants": variants,
                    "pair_quality": pair_quality,
                    "checks": row_checks,
                }
            )
    finally:
        if owns_client:
            live_client.close()

    final_provenance = _clean_server_provenance(clean_repository)
    if final_provenance["revision"] != expected_provenance["revision"]:
        raise RuntimeError("Clean server repository changed revision during the run")
    final_producer_provenance = _producer_provenance()
    if final_producer_provenance != producer_provenance:
        raise RuntimeError("Matrix producer provenance changed during the run")
    coverage = _matrix_coverage(selected_specs)
    all_rows_passed = bool(rows) and all(row["checks"]["passed"] for row in rows)
    summary = {
        "schema_version": 1,
        "run_kind": (
            "cc0_generated_live_relief_context_matrix"
            if any(isinstance(spec, ObjectSceneSpec) for spec in selected_specs)
            else "cc0_makehuman_live_face_variation_matrix"
        ),
        "privacy": (
            "CC0 MakeHuman synthetic heads, generated procedural objects, and "
            "deterministic procedural scene materials only"
        ),
        "server_provenance": expected_provenance,
        "server_runtime_provenance": runtime_provenance,
        "producer_provenance": producer_provenance,
        "final_producer_provenance": final_producer_provenance,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "httpx": httpx.__version__,
            "numpy": np.__version__,
        },
        "fixture": fixture["manifest"],
        "matrix": {
            "available_rows": len(all_specs),
            "executed_rows": len(rows),
            "limit": int(limit) if limit is not None else None,
            "z_scale": RELIEF_HEIGHT_MM,
            "max_xy_size": MAX_XY_SIZE_MM,
            "baseline_background_photo_detail_mm": BASELINE_DETAIL_MM,
            "candidate_omitted_background_fields": list(BACKGROUND_FIELDS),
            "coverage": coverage,
        },
        "health": {
            "request": health_record,
            "request_artifact": _artifact_record(health_record_path, run_output),
            "response_artifact": _artifact_record(
                run_output / "health_response.json", run_output
            ),
        },
        "checks": {
            "server_repository_clean": True,
            "server_revision_unchanged": True,
            "server_runtime_matches_checkout": True,
            "producer_matches_server_revision": True,
            "producer_unchanged_during_run": True,
            "raw_artifacts_beneath_ignored_output": True,
            "requested_rows_executed_once": bool(
                coverage["unique_rows"]
                and coverage["row_ids"] == [row["row_id"] for row in rows]
            ),
            "all_behavioral_rows_passed": all_rows_passed,
            "passed": all_rows_passed,
        },
        "rows": rows,
    }
    summary_path = run_output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-server-repository", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument(
        "--matrix",
        choices=(
            "default",
            "varied-context",
            "face-fusion-train",
            "procedural-object",
        ),
        default="default",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    selected_matrix = {
        "default": DEFAULT_MATRIX,
        "varied-context": VARIED_CONTEXT_MATRIX,
        "face-fusion-train": FACE_FUSION_TRAIN_MATRIX,
        "procedural-object": OBJECT_SMOKE_MATRIX,
    }[args.matrix]
    summary = run(
        args.clean_server_repository,
        output_dir=args.output_dir,
        base_url=args.base_url,
        asset_dir=args.asset_dir,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        specs=selected_matrix,
    )
    print(
        json.dumps(
            {
                "passed": summary["checks"]["passed"],
                "executed_rows": summary["matrix"]["executed_rows"],
                "first_row": summary["rows"][0]["row_id"],
            },
            indent=2,
        )
    )
    if not summary["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
