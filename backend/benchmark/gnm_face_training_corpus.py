"""Build a deterministic Google GNM face-depth corpus with sealed identities."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import trimesh

from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    _background_rgb,
    _compose_scene,
    _scene_phase,
    _selection_geometry_record,
    _translate,
)


CORPUS_SEED = 20260717
GNM_SOURCE_REVISION = "9c9419f191edd68644ef5cb6572e238248e9a81c"
GNM_MODEL_SHA256 = "e50a702789af51347531e13721df1e5a26dcfc30ced9e243c7cede9fcac5db43"
GNM_IDENTITY_DECODER_SHA256 = (
    "1f069bf4975620e25a053772719a35678c9cfe6121e58bc4276a3515b53ac142"
)
GNM_EXPRESSION_DECODER_SHA256 = (
    "5eba165f8a414f73b24be96963d0a17e708c0856739ed85a19031f318dfb51e6"
)
GNM_LICENSE = "Apache-2.0"
GNM_MODEL_RELATIVE_PATH = (
    Path("gnm") / "shape" / "data" / "versions" / "v3_0" / "gnm_head.npz"
)
GNM_SAMPLER_RELATIVE_ROOT = (
    Path("gnm") / "shape" / "data" / "semantic_sampler"
)
GNM_IDENTITY_DECODER = "identity_decoder_model.h5"
GNM_EXPRESSION_DECODER = "expression_decoder_model.h5"
NOVEL_IDENTITY_SELECTION_STRATEGY = "novel-identity-stratified"
ROW_SELECTION_STRATEGIES = (
    "linspace",
    "identity-stratified",
    NOVEL_IDENTITY_SELECTION_STRATEGY,
)

DEMOGRAPHICS = (
    ("female", "middle_eastern"),
    ("female", "asian"),
    ("female", "white"),
    ("female", "black"),
    ("male", "middle_eastern"),
    ("male", "asian"),
    ("male", "white"),
    ("male", "black"),
)
EXPRESSIONS = (
    "surprise",
    "happy",
    "smile_wide",
    "mouth_left",
    "wink_left",
    "mouth_right",
    "corners_down",
    "wink_right",
)
EXPRESSION_INDEX = {
    "surprise": 0,
    "happy": 5,
    "smile_wide": 10,
    "corners_down": 11,
    "wink_left": 13,
    "wink_right": 14,
    "mouth_left": 15,
    "mouth_right": 16,
}
ETHNICITY_INDEX = {
    "middle_eastern": 0,
    "asian": 1,
    "white": 2,
    "black": 3,
}
SKIN_TONES = {
    "middle_eastern": np.asarray((0.57, 0.36, 0.27), dtype=np.float32),
    "asian": np.asarray((0.66, 0.45, 0.34), dtype=np.float32),
    "white": np.asarray((0.73, 0.50, 0.40), dtype=np.float32),
    "black": np.asarray((0.34, 0.20, 0.15), dtype=np.float32),
}
LIGHTING = {
    "soft_left": {
        "ambient": 0.48,
        "diffuse": 0.48,
        "specular": 0.04,
        "shininess": 48.0,
        "direction": (-0.48, -0.20, 0.85),
    },
    "side_right": {
        "ambient": 0.38,
        "diffuse": 0.58,
        "specular": 0.04,
        "shininess": 42.0,
        "direction": (0.70, -0.10, 0.70),
    },
    "overhead": {
        "ambient": 0.43,
        "diffuse": 0.53,
        "specular": 0.04,
        "shininess": 52.0,
        "direction": (0.08, 0.72, 0.69),
    },
}


@dataclass(frozen=True)
class GNMIdentitySpec:
    identity_group: str
    split: str
    gender: str
    ethnicity: str
    variant_index: int
    latent_seed: int


@dataclass(frozen=True)
class GNMSceneCondition:
    target_dimension: int
    camera_yaw_deg: float
    camera_elevation_deg: float
    camera_distance: float
    horizontal_offset: float
    background_profile: str
    lighting_profile: str
    occlusion: str | None


@dataclass(frozen=True)
class GNMTrainingSceneSpec:
    row_id: str
    split: str
    identity_group: str
    gender: str
    ethnicity: str
    identity_variant_index: int
    identity_latent_seed: int
    expression: str
    expression_latent_seed: int
    target_dimension: int
    camera_yaw_deg: float
    camera_elevation_deg: float
    camera_distance: float
    horizontal_offset: float
    background_profile: str
    lighting_profile: str
    occlusion: str | None


SCENE_CONDITIONS = (
    GNMSceneCondition(
        256,
        -38.0,
        -4.0,
        9.2,
        -0.16,
        "deep_shelves",
        "side_right",
        None,
    ),
    GNMSceneCondition(
        384,
        38.0,
        3.0,
        13.8,
        0.16,
        "layered_studio",
        "soft_left",
        None,
    ),
    GNMSceneCondition(
        256,
        -30.0,
        2.0,
        7.6,
        -0.12,
        "structured_room",
        "overhead",
        "eye_band",
    ),
    GNMSceneCondition(
        384,
        22.0,
        -4.0,
        8.55,
        0.08,
        "structured_room",
        "overhead",
        "eye_band",
    ),
    GNMSceneCondition(
        256,
        13.0,
        -1.0,
        2.9,
        0.03,
        "layered_studio",
        "side_right",
        None,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_split(variant_index: int) -> str:
    if variant_index < 3:
        return "train"
    if variant_index == 3:
        return "validation"
    if variant_index == 4:
        return "sealed"
    raise ValueError("GNM identity variants must be in [0, 4]")


def build_identity_specs() -> tuple[GNMIdentitySpec, ...]:
    identities = []
    for demographic_index, (gender, ethnicity) in enumerate(DEMOGRAPHICS):
        for variant_index in range(5):
            identities.append(
                GNMIdentitySpec(
                    identity_group=(
                        f"gnm_{gender}_{ethnicity}_v{variant_index:02d}"
                    ),
                    split=_identity_split(variant_index),
                    gender=gender,
                    ethnicity=ethnicity,
                    variant_index=variant_index,
                    latent_seed=(
                        CORPUS_SEED
                        + demographic_index * 1009
                        + variant_index * 7919
                    ),
                )
            )
    return tuple(identities)


IDENTITIES = build_identity_specs()


def build_training_matrix(
    identities: tuple[GNMIdentitySpec, ...] = IDENTITIES,
) -> tuple[GNMTrainingSceneSpec, ...]:
    rows = []
    for identity_index, identity in enumerate(identities):
        for expression_index, expression in enumerate(EXPRESSIONS):
            for scene_index in range(len(SCENE_CONDITIONS)):
                shifted_scene_index = (
                    scene_index + identity_index * 2 + expression_index
                ) % len(SCENE_CONDITIONS)
                scene = SCENE_CONDITIONS[shifted_scene_index]
                rows.append(
                    GNMTrainingSceneSpec(
                        row_id=(
                            f"{identity.identity_group}__{expression}_"
                            f"{scene_index:02d}"
                        ),
                        split=identity.split,
                        identity_group=identity.identity_group,
                        gender=identity.gender,
                        ethnicity=identity.ethnicity,
                        identity_variant_index=identity.variant_index,
                        identity_latent_seed=identity.latent_seed,
                        expression=expression,
                        expression_latent_seed=(
                            identity.latent_seed
                            + 100_003
                            + expression_index * 12_289
                        ),
                        target_dimension=scene.target_dimension,
                        camera_yaw_deg=scene.camera_yaw_deg,
                        camera_elevation_deg=scene.camera_elevation_deg,
                        camera_distance=scene.camera_distance,
                        horizontal_offset=scene.horizontal_offset,
                        background_profile=scene.background_profile,
                        lighting_profile=scene.lighting_profile,
                        occlusion=scene.occlusion,
                    )
                )
    return tuple(rows)


TRAINING_MATRIX = build_training_matrix()


def build_novel_training_identities() -> tuple[GNMIdentitySpec, ...]:
    variant_index = 5
    identities = []
    for demographic_index, (gender, ethnicity) in enumerate(DEMOGRAPHICS):
        identities.append(
            GNMIdentitySpec(
                identity_group=(
                    f"gnm_{gender}_{ethnicity}_v{variant_index:02d}"
                ),
                split="train",
                gender=gender,
                ethnicity=ethnicity,
                variant_index=variant_index,
                latent_seed=(
                    CORPUS_SEED
                    + demographic_index * 1009
                    + variant_index * 7919
                ),
            )
        )
    return tuple(identities)


NOVEL_TRAINING_IDENTITIES = build_novel_training_identities()
NOVEL_TRAINING_MATRIX = build_training_matrix(NOVEL_TRAINING_IDENTITIES)


def select_training_rows(
    *,
    split: str | None = None,
    limit: int | None = None,
    strategy: str = "linspace",
) -> tuple[GNMTrainingSceneSpec, ...]:
    if strategy == NOVEL_IDENTITY_SELECTION_STRATEGY:
        if split not in (None, "train"):
            raise ValueError(
                "Novel identity rows are a training-only cohort"
            )
        rows = NOVEL_TRAINING_MATRIX
        if limit is None or int(limit) >= len(rows):
            return rows
        count = max(int(limit), 0)
        if count == 0:
            return ()
        return _select_identity_stratified_rows(rows, count)
    rows = tuple(
        row for row in TRAINING_MATRIX if split is None or row.split == split
    )
    if strategy not in ROW_SELECTION_STRATEGIES:
        raise ValueError(
            f"Unsupported row selection strategy {strategy!r}; "
            f"expected one of {ROW_SELECTION_STRATEGIES}"
        )
    if limit is None or int(limit) >= len(rows):
        return rows
    count = max(int(limit), 0)
    if count == 0:
        return ()
    if strategy == "identity-stratified":
        return _select_identity_stratified_rows(rows, count)
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    return tuple(rows[int(index)] for index in indices)


def _scene_condition_index(row: GNMTrainingSceneSpec) -> int:
    for index, condition in enumerate(SCENE_CONDITIONS):
        if (
            row.target_dimension == condition.target_dimension
            and row.camera_yaw_deg == condition.camera_yaw_deg
            and row.camera_elevation_deg == condition.camera_elevation_deg
            and row.camera_distance == condition.camera_distance
            and row.horizontal_offset == condition.horizontal_offset
            and row.background_profile == condition.background_profile
            and row.lighting_profile == condition.lighting_profile
            and row.occlusion == condition.occlusion
        ):
            return index
    raise ValueError(f"Row {row.row_id!r} does not match a GNM scene condition")


def _select_identity_stratified_rows(
    rows: tuple[GNMTrainingSceneSpec, ...],
    count: int,
) -> tuple[GNMTrainingSceneSpec, ...]:
    identity_groups = tuple(dict.fromkeys(row.identity_group for row in rows))
    if not identity_groups:
        return ()
    base_quota, remainder = divmod(int(count), len(identity_groups))
    if base_quota > len(EXPRESSIONS) * len(SCENE_CONDITIONS):
        raise ValueError("Requested identity-stratified quota exceeds matrix size")
    extra_indices = set(
        int(index)
        for index in (
            np.linspace(
                0,
                len(identity_groups) - 1,
                num=remainder,
                dtype=np.int64,
            )
            if remainder
            else ()
        )
    )
    by_identity: dict[
        str, dict[tuple[str, int], GNMTrainingSceneSpec]
    ] = {}
    for row in rows:
        by_identity.setdefault(row.identity_group, {})[
            (row.expression, _scene_condition_index(row))
        ] = row

    selected = []
    global_slot = 0
    for identity_index, identity_group in enumerate(identity_groups):
        quota = base_quota + int(identity_index in extra_indices)
        available = by_identity[identity_group]
        for _ in range(quota):
            expression = EXPRESSIONS[global_slot % len(EXPRESSIONS)]
            scene_index = global_slot % len(SCENE_CONDITIONS)
            key = (expression, scene_index)
            if key not in available:
                raise ValueError(
                    f"Identity {identity_group!r} is missing matrix cell {key!r}"
                )
            selected.append(available[key])
            global_slot += 1
    if len(selected) != int(count) or len({row.row_id for row in selected}) != len(
        selected
    ):
        raise RuntimeError("Identity-stratified row selection is incomplete")
    return tuple(selected)


def _asset_paths(gnm_root: Path) -> dict[str, Path]:
    sampler_root = gnm_root / GNM_SAMPLER_RELATIVE_ROOT
    return {
        "model": gnm_root / GNM_MODEL_RELATIVE_PATH,
        "identity_decoder": sampler_root / GNM_IDENTITY_DECODER,
        "expression_decoder": sampler_root / GNM_EXPRESSION_DECODER,
    }


def _git_revision(gnm_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=gnm_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_tracked_clean(gnm_root: Path) -> bool:
    try:
        return (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--"],
                cwd=gnm_root,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
    except OSError:
        return False


def preflight_gnm_root(gnm_root: str | Path) -> dict:
    root = Path(gnm_root).resolve()
    paths = _asset_paths(root)
    expected_hashes = {
        "model": GNM_MODEL_SHA256,
        "identity_decoder": GNM_IDENTITY_DECODER_SHA256,
        "expression_decoder": GNM_EXPRESSION_DECODER_SHA256,
    }
    assets = {}
    for name, path in paths.items():
        exists = path.is_file()
        actual_hash = _sha256(path) if exists else None
        assets[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": int(path.stat().st_size) if exists else None,
            "sha256": actual_hash,
            "expected_sha256": expected_hashes[name],
            "hash_matches": bool(
                actual_hash is not None and actual_hash == expected_hashes[name]
            ),
        }
    revision = _git_revision(root)
    checks = {
        "source_root_exists": root.is_dir(),
        "source_revision_matches": revision == GNM_SOURCE_REVISION,
        "tracked_source_clean": _git_tracked_clean(root),
        "assets_exist": all(asset["exists"] for asset in assets.values()),
        "asset_hashes_match": all(
            asset["hash_matches"] for asset in assets.values()
        ),
    }
    return {
        "runnable": bool(all(checks.values())),
        "root": str(root),
        "revision": revision,
        "expected_revision": GNM_SOURCE_REVISION,
        "license": GNM_LICENSE,
        "checks": checks,
        "assets": assets,
    }


def _decode_h5(
    path: Path,
    latent: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "GNM corpus generation requires h5py; install it in the "
            "benchmark environment before rendering"
        ) from exc

    values = np.concatenate((latent, condition), axis=-1).astype(np.float32)
    with h5py.File(path, "r") as source:
        layers = []
        for name in source["model_weights"].keys():
            match = re.fullmatch(r"dense_(\d+)", name)
            if match:
                layers.append((int(match.group(1)), name))
        if not layers:
            raise ValueError(f"No dense decoder layers found in {path}")
        for index, (_, name) in enumerate(sorted(layers)):
            group = source["model_weights"][name][name]
            values = (
                values @ np.asarray(group["kernel:0"], dtype=np.float32)
                + np.asarray(group["bias:0"], dtype=np.float32)
            )
            if index + 1 < len(layers):
                values = np.maximum(values, 0.0)
    return values.astype(np.float32)


def _identity_condition(spec: GNMTrainingSceneSpec) -> np.ndarray:
    condition = np.zeros((1, 6), dtype=np.float32)
    condition[0, 0 if spec.gender == "female" else 1] = 1.0
    condition[0, 2 + ETHNICITY_INDEX[spec.ethnicity]] = 1.0
    return condition


def _expression_condition(spec: GNMTrainingSceneSpec) -> np.ndarray:
    condition = np.zeros((1, 20), dtype=np.float32)
    condition[0, EXPRESSION_INDEX[spec.expression]] = 1.0
    return condition


def _load_gnm_model(gnm_root: Path):
    if str(gnm_root) not in sys.path:
        sys.path.insert(0, str(gnm_root))
    try:
        from gnm.shape import gnm_numpy
    except ImportError as exc:
        raise RuntimeError(
            "GNM corpus generation requires the official GNM source and its "
            "lightweight dependencies etils and immutabledict"
        ) from exc
    module_path = Path(gnm_numpy.__file__).resolve()
    if not module_path.is_relative_to(gnm_root):
        raise RuntimeError(
            "Imported GNM module is outside the pinned source checkout: "
            f"{module_path}"
        )
    return gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )


def _face_part_weights(model) -> dict[str, np.ndarray]:
    def maximum(*names: str) -> np.ndarray:
        return np.maximum.reduce(
            [
                np.asarray(model.vertex_group(name), dtype=np.float32)
                for name in names
            ]
        )

    return {
        "left_eye": maximum("left_eye", "left_orbital_region"),
        "right_eye": maximum("right_eye", "right_orbital_region"),
        "left_eyebrow": maximum(
            "left_brow_region", "left_orbital_region"
        )
        * 0.85,
        "right_eyebrow": maximum(
            "right_brow_region", "right_orbital_region"
        )
        * 0.85,
        "nose": maximum("nose_region"),
        "mouth": maximum(
            "upper_lip_region",
            "lower_lip_region",
            "mouth_sock",
        ),
    }


def _vertex_colors(model, spec: GNMTrainingSceneSpec) -> np.ndarray:
    skin = SKIN_TONES[spec.ethnicity].copy()
    if spec.gender == "male":
        skin *= np.asarray((0.94, 0.94, 0.95), dtype=np.float32)
    colors = np.broadcast_to(skin, (model.num_vertices, 3)).copy()

    def blend(
        group: str,
        color: tuple[float, float, float],
        amount: float = 1.0,
    ) -> None:
        weight = np.asarray(
            model.vertex_group(group), dtype=np.float32
        )[:, None]
        weight = np.clip(weight * amount, 0.0, 1.0)
        colors[:] = colors * (1.0 - weight) + np.asarray(color) * weight

    blend("scleras", (0.91, 0.92, 0.90))
    blend("irises", (0.18, 0.28, 0.31))
    blend("pupils", (0.025, 0.025, 0.025))
    blend(
        "upper_lip_region",
        tuple(np.clip(skin * (0.78, 0.62, 0.66), 0, 1)),
        0.55,
    )
    blend(
        "lower_lip_region",
        tuple(np.clip(skin * (0.86, 0.68, 0.71), 0, 1)),
        0.62,
    )
    blend("teeth", (0.88, 0.86, 0.79))
    blend("tongue", (0.53, 0.20, 0.23))
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def _render_config(spec: GNMTrainingSceneSpec) -> RenderConfig:
    lighting = LIGHTING[spec.lighting_profile]
    return RenderConfig(
        size=int(spec.target_dimension),
        projection="perspective",
        perspective_fov_y_deg=32.0,
        camera_distance=float(spec.camera_distance),
        background_rgb=_background_rgb(spec.background_profile),
        ambient=float(lighting["ambient"]),
        diffuse=float(lighting["diffuse"]),
        specular=float(lighting["specular"]),
        shininess=float(lighting["shininess"]),
        light_direction=tuple(lighting["direction"]),
    )


def _apply_eye_band(
    rgb: np.ndarray,
    face_mask: np.ndarray,
    parts: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    eye_mask = (parts["left_eye"] | parts["right_eye"]) & face_mask
    eye_rows, eye_columns = np.where(eye_mask)
    face_rows, face_columns = np.where(face_mask)
    if not len(eye_rows) or not len(face_rows):
        raise ValueError("GNM eye-band row requires visible eye masks")
    face_width = int(face_columns.max() - face_columns.min() + 1)
    face_height = int(face_rows.max() - face_rows.min() + 1)
    center_y = 0.5 * float(eye_rows.min() + eye_rows.max() + 1)
    half_height = max(3, int(round(face_height * 0.05)))
    margin = max(2, int(round(face_width * 0.05)))
    left = max(0, int(eye_columns.min()) - margin)
    right = min(rgb.shape[1], int(eye_columns.max() + 1) + margin)
    top = max(0, int(round(center_y)) - half_height)
    bottom = min(rgb.shape[0], int(round(center_y)) + half_height)
    output = rgb.copy()
    output[top:bottom, left:right] = (
        0.42 * output[top:bottom, left:right]
        + 0.58 * np.asarray((0.03, 0.05, 0.07), dtype=np.float32)
    )
    return output, [top, bottom, left, right]


def _render_row(
    spec: GNMTrainingSceneSpec,
    *,
    model,
    part_weights: dict[str, np.ndarray],
    asset_paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    identity_rng = np.random.default_rng(spec.identity_latent_seed)
    expression_rng = np.random.default_rng(spec.expression_latent_seed)
    identity = _decode_h5(
        asset_paths["identity_decoder"],
        identity_rng.normal(size=(1, 64)).astype(np.float32),
        _identity_condition(spec),
    )[0]
    expression = _decode_h5(
        asset_paths["expression_decoder"],
        expression_rng.normal(size=(1, 64)).astype(np.float32),
        _expression_condition(spec),
    )[0]
    vertices = model(
        identity=identity,
        expression=expression,
        rotations=np.zeros((model.num_joints, 3), dtype=np.float32),
        translation=np.zeros(3, dtype=np.float32),
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(model.triangles, dtype=np.int64),
        process=False,
    )
    rendered = render_mesh(
        mesh,
        CameraSpec(
            azimuth_deg=float(spec.camera_yaw_deg),
            elevation_deg=float(spec.camera_elevation_deg),
        ),
        _render_config(spec),
        (180, 120, 100),
        vertex_part_weights=part_weights,
        vertex_colors=_vertex_colors(model, spec),
    )
    offset_columns = int(
        round(float(spec.horizontal_offset) * spec.target_dimension)
    )
    background = _background_rgb(spec.background_profile)
    face_mask = _translate(
        rendered.silhouette.astype(bool), offset_columns, False
    )
    face_rgb = _translate(
        rendered.rgb.astype(np.float32), offset_columns, background
    )
    rendered_depth = _translate(
        rendered.depth.astype(np.float32), offset_columns, 1.0
    )
    exact_parts = {
        name: _translate(mask.astype(bool), offset_columns, False) & face_mask
        for name, mask in (rendered.part_masks or {}).items()
    }
    missing_parts = sorted(
        name for name, mask in exact_parts.items() if not np.any(mask)
    )
    if missing_parts:
        raise ValueError(f"{spec.row_id} has empty face parts: {missing_parts}")
    exact_depth, source_rgb = _compose_scene(
        rendered_depth,
        face_mask,
        face_rgb,
        phase=_scene_phase(spec.row_id),
        background_profile=spec.background_profile,
    )
    occluder_bounds = None
    if spec.occlusion == "eye_band":
        source_rgb, occluder_bounds = _apply_eye_band(
            source_rgb, face_mask, exact_parts
        )
    geometry = _selection_geometry_record(face_mask)
    background_values = exact_depth[~face_mask]
    geometry.update(
        {
            "face_bbox_xyxy": geometry["selection_bbox_xyxy"],
            "face_bbox_width_pixels": geometry[
                "selection_bbox_width_pixels"
            ],
            "face_bbox_height_pixels": geometry[
                "selection_bbox_height_pixels"
            ],
            "background_depth_p02": float(
                np.percentile(background_values, 2.0)
            ),
            "background_depth_p98": float(
                np.percentile(background_values, 98.0)
            ),
            "background_depth_span": float(
                np.percentile(background_values, 98.0)
                - np.percentile(background_values, 2.0)
            ),
            "horizontal_offset_columns": offset_columns,
            "occluder_bounds_tblr": occluder_bounds,
            "identity_coefficient_rms": float(
                np.sqrt(np.mean(identity**2))
            ),
            "expression_coefficient_rms": float(
                np.sqrt(np.mean(expression**2))
            ),
            "source_mesh_vertices": int(len(mesh.vertices)),
            "source_mesh_faces": int(len(mesh.faces)),
            "source_mesh_degenerate_faces": int(
                np.count_nonzero(mesh.area_faces <= 1e-12)
            ),
        }
    )
    return (
        np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8),
        face_mask.astype(np.uint8) * 255,
        exact_depth.astype(np.float32),
        exact_parts,
        geometry,
    )


def render_training_slice(
    gnm_root: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    selection_strategy: str = "linspace",
) -> dict:
    root = Path(gnm_root).resolve()
    preflight = preflight_gnm_root(root)
    if not preflight["runnable"]:
        raise ValueError(
            "GNM source preflight failed: "
            + json.dumps(preflight["checks"], sort_keys=True)
        )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    asset_paths = _asset_paths(root)
    model = _load_gnm_model(root)
    part_weights = _face_part_weights(model)
    selected = select_training_rows(
        split=split,
        limit=limit,
        strategy=selection_strategy,
    )
    selected_matrix = (
        NOVEL_TRAINING_MATRIX
        if selection_strategy == NOVEL_IDENTITY_SELECTION_STRATEGY
        else TRAINING_MATRIX
    )
    rows = []
    started = time.perf_counter()
    for spec in selected:
        source, selection, exact, parts, geometry = _render_row(
            spec,
            model=model,
            part_weights=part_weights,
            asset_paths=asset_paths,
        )
        row_dir = output_root / "rows" / spec.row_id
        parts_dir = row_dir / "exact_face_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        source_path = row_dir / "source.png"
        selection_path = row_dir / "selection_mask.png"
        exact_path = row_dir / "exact_depth.npy"
        Image.fromarray(source).save(source_path)
        Image.fromarray(selection).save(selection_path)
        np.save(exact_path, exact)
        part_records = {}
        for name, mask in sorted(parts.items()):
            part_path = parts_dir / f"{name}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(part_path)
            part_records[name] = {
                "path": part_path.relative_to(output_root).as_posix(),
                "sha256": _sha256(part_path),
                "pixels": int(np.count_nonzero(mask)),
            }
        rows.append(
            {
                "row_id": spec.row_id,
                "split": spec.split,
                "identity_group": spec.identity_group,
                "expression": spec.expression,
                "spec": asdict(spec),
                "render": geometry,
                "source": {
                    "path": source_path.relative_to(output_root).as_posix(),
                    "sha256": _sha256(source_path),
                },
                "selection_mask": {
                    "path": selection_path.relative_to(
                        output_root
                    ).as_posix(),
                    "sha256": _sha256(selection_path),
                },
                "exact_depth": {
                    "path": exact_path.relative_to(output_root).as_posix(),
                    "sha256": _sha256(exact_path),
                },
                "exact_face_parts": part_records,
            }
        )
    summary = {
        "schema_version": 1,
        "provider": "google-gnm-head-v3",
        "source_revision": GNM_SOURCE_REVISION,
        "license": GNM_LICENSE,
        "privacy": (
            "scan-derived parametric GNM samples and deterministic "
            "procedural scenes only"
        ),
        "source_geometry_training_and_evaluation_only": True,
        "corpus_seed": CORPUS_SEED,
        "matrix_row_count": len(selected_matrix),
        "matrix_split_counts": {
            name: sum(row.split == name for row in selected_matrix)
            for name in ("train", "validation", "sealed")
        },
        "requested_split": split,
        "requested_limit": limit,
        "selection_strategy": selection_strategy,
        "row_count": len(rows),
        "runtime_seconds": float(time.perf_counter() - started),
        "preflight": preflight,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnm-root", required=True)
    parser.add_argument("--render-output")
    parser.add_argument("--split", choices=("train", "validation", "sealed"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-strategy",
        choices=ROW_SELECTION_STRATEGIES,
        default="linspace",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight_gnm_root(args.gnm_root)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["runnable"] else 2)
    if not args.render_output:
        parser.error("--render-output is required unless --preflight-only is set")
    summary = render_training_slice(
        args.gnm_root,
        args.render_output,
        split=args.split,
        limit=args.limit,
        selection_strategy=args.selection_strategy,
    )
    print(
        json.dumps(
            {
                "row_count": summary["row_count"],
                "runtime_seconds": summary["runtime_seconds"],
                "output": str(Path(args.render_output) / "summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
