"""Render a deterministic, privacy-safe face-depth corpus from Meta MHR v1.0.1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image
import trimesh

from backend.benchmark.mesh_rendering import (
    CameraSpec,
    RenderConfig,
    camera_transform,
    render_mesh,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    _background_rgb,
    _compose_scene,
    _scene_phase,
    _selection_geometry_record,
    _translate,
)


CORPUS_SEED = 20260718
MHR_PROVIDER = "meta-mhr-v1.0.1-camera-depth"
MHR_SOURCE_URL = "https://github.com/facebookresearch/MHR"
MHR_RELEASE_URL = "https://github.com/facebookresearch/MHR/releases/tag/v1.0.1"
MHR_SOURCE_REVISION = "4998cec385b1aaa07abdefba71bfba2f83c7db32"
MHR_RELEASE_ARCHIVE_SHA256 = (
    "e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94"
)
MHR_MODEL_SHA256 = (
    "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
)
MHR_HEAD_MASK_SHA256 = (
    "60a64a305731dcc1c09f167dfa844246443b3115ae05abebe3813aa64a9ba2ae"
)
MHR_SOURCE_LICENSE_SHA256 = (
    "3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5"
)
MHR_ASSET_LICENSE_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)
MHR_LICENSE = "Apache-2.0"
MHR_MODEL_RELATIVE_PATH = Path("assets") / "mhr_model.pt"
MHR_HEAD_MASK_RELATIVE_PATH = (
    Path("tools")
    / "mhr_smpl_conversion"
    / "assets"
    / "head_hand_mask.npz"
)
MHR_SOURCE_LICENSE_RELATIVE_PATH = Path("LICENSE")
MHR_ASSET_LICENSE_RELATIVE_PATH = Path("assets") / "LICENSE.txt"
MHR_MODEL_SIZE_BYTES = 696_110_248
MHR_IDENTITY_DIMENSION = 45
MHR_HEAD_IDENTITY_SLICE = (20, 40)
MHR_EXPRESSION_DIMENSION = 72
MHR_POSE_DIMENSION = 204
MIN_HEAD_VERTICES = 5_000
MIN_HEAD_FACES = 10_000
APPEARANCE_KINDS = ("clay", "semantic-procedural")
SEMANTIC_APPEARANCE_VERSION = 2
FACE_PART_NAMES = (
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "nose",
    "mouth",
)

LIGHTING = {
    "soft_left": {
        "ambient": 0.48,
        "diffuse": 0.48,
        "specular": 0.04,
        "shininess": 48.0,
        "direction": (-0.48, -0.20, 0.85),
    },
    "side_right": {
        "ambient": 0.40,
        "diffuse": 0.56,
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
SKIN_TONES = (
    (0.70, 0.48, 0.38),
    (0.56, 0.36, 0.27),
    (0.39, 0.24, 0.18),
    (0.76, 0.55, 0.44),
)
TARGET_DIMENSIONS = (256, 384, 256, 384)
FACE_HEIGHTS = (64, 75, 96, 128)
CAMERA_YAWS = (-40.0, -20.0, 20.0, 40.0)
CAMERA_ELEVATIONS = (-4.0, 2.0, -2.0, 4.0)
BACKGROUND_PROFILES = (
    "deep_shelves",
    "structured_room",
    "layered_studio",
    "structured_room",
)
LIGHTING_PROFILES = ("side_right", "soft_left", "overhead", "soft_left")
OCCLUSIONS = (None, "eye_band", None, "cheek_patch")
TRAINING_TARGET_DIMENSIONS = (256, 384, 256, 384, 384, 256, 384, 256)
TRAINING_FACE_HEIGHTS = (75, 64, 96, 128, 75, 64, 96, 128)
TRAINING_CAMERA_YAWS = (-40.0, 40.0, -20.0, 20.0, 40.0, -40.0, 20.0, -20.0)
TRAINING_CAMERA_ELEVATIONS = (-4.0, 3.0, 2.0, -2.0, 4.0, -3.0, 1.0, -1.0)
TRAINING_BACKGROUNDS = (
    "deep_shelves",
    "structured_room",
    "layered_studio",
    "structured_room",
    "layered_studio",
    "deep_shelves",
    "structured_room",
    "layered_studio",
)
TRAINING_LIGHTING = (
    "side_right",
    "soft_left",
    "overhead",
    "soft_left",
    "overhead",
    "side_right",
    "soft_left",
    "side_right",
)
TRAINING_OCCLUSIONS = (None, "cheek_patch", "eye_band", None, None, None, None, None)


@dataclass(frozen=True)
class MHRSceneSpec:
    row_id: str
    split: str
    identity_group: str
    identity_seed: int
    expression_seed: int
    expression_profile: str
    target_dimension: int
    face_height_pixels: int
    camera_yaw_deg: float
    camera_elevation_deg: float
    horizontal_offset: float
    background_profile: str
    lighting_profile: str
    occlusion: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_split(identity_index: int) -> str:
    return ("train", "train", "validation", "sealed")[identity_index]


def build_scene_matrix() -> tuple[MHRSceneSpec, ...]:
    rows = []
    for identity_index in range(4):
        identity_group = f"mhr_pilot_identity_{identity_index:02d}"
        identity_seed = CORPUS_SEED + identity_index * 7_919
        for scene_index in range(4):
            condition = (scene_index + identity_index) % 4
            expression_seed = identity_seed + 100_003 + scene_index * 12_289
            expression_profile = (
                "neutral" if scene_index == 0 else f"pca-seed-{expression_seed}"
            )
            rows.append(
                MHRSceneSpec(
                    row_id=f"{identity_group}__scene_{scene_index:02d}",
                    split=_identity_split(identity_index),
                    identity_group=identity_group,
                    identity_seed=identity_seed,
                    expression_seed=expression_seed,
                    expression_profile=expression_profile,
                    target_dimension=TARGET_DIMENSIONS[condition],
                    face_height_pixels=FACE_HEIGHTS[(condition + 1) % 4],
                    camera_yaw_deg=CAMERA_YAWS[condition],
                    camera_elevation_deg=CAMERA_ELEVATIONS[condition],
                    horizontal_offset=(-0.08, 0.06, -0.04, 0.08)[condition],
                    background_profile=BACKGROUND_PROFILES[condition],
                    lighting_profile=LIGHTING_PROFILES[condition],
                    occlusion=OCCLUSIONS[condition],
                )
            )
    return tuple(rows)


SCENE_MATRIX = build_scene_matrix()


def _training_identity_split(identity_index: int) -> str:
    if identity_index < 30:
        return "train"
    if identity_index < 35:
        return "validation"
    if identity_index < 40:
        return "sealed"
    raise ValueError("MHR training identity index must be in [0, 39]")


def build_training_matrix() -> tuple[MHRSceneSpec, ...]:
    rows = []
    for identity_index in range(40):
        identity_group = f"mhr_training_identity_{identity_index:03d}"
        identity_seed = CORPUS_SEED + 1_000_003 + identity_index * 7_919
        for scene_index in range(8):
            condition = (scene_index + identity_index * 3) % 8
            expression_seed = identity_seed + 100_003 + scene_index * 12_289
            expression_profile = (
                "neutral" if scene_index == 0 else f"pca-seed-{expression_seed}"
            )
            rows.append(
                MHRSceneSpec(
                    row_id=f"{identity_group}__scene_{scene_index:02d}",
                    split=_training_identity_split(identity_index),
                    identity_group=identity_group,
                    identity_seed=identity_seed,
                    expression_seed=expression_seed,
                    expression_profile=expression_profile,
                    target_dimension=TRAINING_TARGET_DIMENSIONS[condition],
                    face_height_pixels=TRAINING_FACE_HEIGHTS[condition],
                    camera_yaw_deg=TRAINING_CAMERA_YAWS[condition],
                    camera_elevation_deg=TRAINING_CAMERA_ELEVATIONS[condition],
                    horizontal_offset=(-0.08, 0.08, -0.05, 0.04, 0.07, -0.07, 0.03, -0.03)[
                        condition
                    ],
                    background_profile=TRAINING_BACKGROUNDS[condition],
                    lighting_profile=TRAINING_LIGHTING[condition],
                    occlusion=TRAINING_OCCLUSIONS[condition],
                )
            )
    return tuple(rows)


TRAINING_MATRIX = build_training_matrix()
MATRICES = {"pilot": SCENE_MATRIX, "training": TRAINING_MATRIX}


def select_scene_rows(
    limit: int | None = None,
    matrix_kind: str = "pilot",
) -> tuple[MHRSceneSpec, ...]:
    try:
        matrix = MATRICES[matrix_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown MHR matrix kind: {matrix_kind}") from exc
    if limit is None or int(limit) >= len(matrix):
        return matrix
    count = max(0, int(limit))
    if count == 0:
        return ()
    if matrix_kind == "training" and count <= 40:
        identity_indices = np.linspace(0, 39, num=count, dtype=np.int64)
        selected = []
        for slot, identity_index in enumerate(identity_indices):
            condition = slot % 8
            scene_index = (condition - int(identity_index) * 3) % 8
            selected.append(matrix[int(identity_index) * 8 + scene_index])
        return tuple(selected)
    indices = np.linspace(0, len(matrix) - 1, num=count, dtype=np.int64)
    return tuple(matrix[int(index)] for index in indices)


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_tracked_clean(root: Path) -> bool:
    try:
        return (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--"],
                cwd=root,
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )
    except OSError:
        return False


def _asset_paths(root: Path) -> dict[str, Path]:
    return {
        "model": root / MHR_MODEL_RELATIVE_PATH,
        "head_mask": root / MHR_HEAD_MASK_RELATIVE_PATH,
        "source_license": root / MHR_SOURCE_LICENSE_RELATIVE_PATH,
        "asset_license": root / MHR_ASSET_LICENSE_RELATIVE_PATH,
    }


def preflight_mhr_root(mhr_root: str | Path) -> dict:
    root = Path(mhr_root).resolve()
    paths = _asset_paths(root)
    expected_hashes = {
        "model": MHR_MODEL_SHA256,
        "head_mask": MHR_HEAD_MASK_SHA256,
        "source_license": MHR_SOURCE_LICENSE_SHA256,
        "asset_license": MHR_ASSET_LICENSE_SHA256,
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
            "hash_matches": actual_hash == expected_hashes[name],
        }
    revision = _git_revision(root)
    checks = {
        "source_root_exists": root.is_dir(),
        "source_revision_matches": revision == MHR_SOURCE_REVISION,
        "tracked_source_clean": _git_tracked_clean(root),
        "assets_exist": all(asset["exists"] for asset in assets.values()),
        "asset_hashes_match": all(
            asset["hash_matches"] for asset in assets.values()
        ),
        "model_size_matches": (
            assets["model"]["size_bytes"] == MHR_MODEL_SIZE_BYTES
        ),
    }
    return {
        "runnable": bool(all(checks.values())),
        "root": str(root),
        "revision": revision,
        "expected_revision": MHR_SOURCE_REVISION,
        "license": MHR_LICENSE,
        "checks": checks,
        "assets": assets,
    }


def _portable_preflight(preflight: dict, root: Path) -> dict:
    assets = {}
    for name, record in preflight["assets"].items():
        path = Path(record["path"])
        assets[name] = {
            **record,
            "path": path.relative_to(root).as_posix(),
        }
    return {
        **preflight,
        "root": "external-pinned-mhr-checkout",
        "assets": assets,
        "torchscript_only_no_source_imports": True,
    }


def _load_model(root: Path, device: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MHR corpus generation requires PyTorch") from exc
    model = torch.jit.load(str(root / MHR_MODEL_RELATIVE_PATH), map_location=device)
    model.eval()
    if int(model.get_num_identity_blendshapes()) != MHR_IDENTITY_DIMENSION:
        raise ValueError("Pinned MHR identity dimension changed")
    if int(model.get_num_face_expression_blendshapes()) != MHR_EXPRESSION_DIMENSION:
        raise ValueError("Pinned MHR expression dimension changed")
    return model


def coefficients_for_spec(spec: MHRSceneSpec) -> tuple[np.ndarray, np.ndarray]:
    identity = np.zeros(MHR_IDENTITY_DIMENSION, dtype=np.float32)
    identity_rng = np.random.Generator(np.random.PCG64(spec.identity_seed))
    identity[MHR_HEAD_IDENTITY_SLICE[0] : MHR_HEAD_IDENTITY_SLICE[1]] = np.clip(
        identity_rng.normal(
            0.0,
            0.72,
            MHR_HEAD_IDENTITY_SLICE[1] - MHR_HEAD_IDENTITY_SLICE[0],
        ),
        -1.8,
        1.8,
    )
    expression = np.zeros(MHR_EXPRESSION_DIMENSION, dtype=np.float32)
    if spec.expression_profile != "neutral":
        expression_rng = np.random.Generator(
            np.random.PCG64(spec.expression_seed)
        )
        expression[:] = np.clip(
            expression_rng.normal(0.0, 0.20, MHR_EXPRESSION_DIMENSION),
            -0.60,
            0.60,
        )
    return identity, expression


def _compact_head_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    head_mask: np.ndarray,
) -> trimesh.Trimesh:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.asarray(head_mask, dtype=np.float32) > 0.5
    if mask.shape != (len(vertices),):
        raise ValueError("MHR head mask does not match the model vertex count")
    selected_faces = faces[np.all(mask[faces], axis=1)]
    used = np.unique(selected_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    mesh = trimesh.Trimesh(
        vertices=vertices[used],
        faces=remap[selected_faces],
        process=False,
    )
    if len(mesh.vertices) < MIN_HEAD_VERTICES or len(mesh.faces) < MIN_HEAD_FACES:
        raise ValueError(
            "Pinned MHR head segmentation produced unexpectedly small geometry"
        )
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError("MHR head geometry contains non-finite vertices")
    return mesh


def _decode_head(
    model,
    spec: MHRSceneSpec,
    head_mask: np.ndarray,
    device: str,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    identity, expression = coefficients_for_spec(spec)
    mesh = _decode_head_coefficients(
        model,
        identity,
        expression,
        head_mask,
        device,
    )
    return mesh, identity, expression


def _decode_head_coefficients(
    model,
    identity: np.ndarray,
    expression: np.ndarray,
    head_mask: np.ndarray,
    device: str,
) -> trimesh.Trimesh:
    import torch

    identity_tensor = torch.from_numpy(identity[None]).to(device)
    expression_tensor = torch.from_numpy(expression[None]).to(device)
    pose_tensor = torch.zeros((1, MHR_POSE_DIMENSION), device=device)
    with torch.inference_mode():
        vertices, _ = model(
            identity_tensor,
            pose_tensor,
            expression_tensor,
            False,
        )
    faces = model.character_torch.mesh.faces.detach().cpu().numpy()
    return _compact_head_mesh(
        vertices[0].detach().cpu().numpy(),
        faces,
        head_mask,
    )


def _normalized_vertex_coordinates(mesh: trimesh.Trimesh) -> tuple[np.ndarray, ...]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if not len(vertices):
        raise ValueError("MHR face-part weights require non-empty geometry")
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    center_x = 0.5 * (lower[0] + upper[0])
    half_width = max(0.5 * float(upper[0] - lower[0]), 1e-6)
    x = (vertices[:, 0] - center_x) / half_width
    y = (vertices[:, 1] - lower[1]) / max(float(upper[1] - lower[1]), 1e-6)
    z = (vertices[:, 2] - lower[2]) / max(float(upper[2] - lower[2]), 1e-6)
    return x, y, z


def _gaussian_part(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    spatial = np.exp(
        -0.5
        * (
            ((x - center_x) / sigma_x) ** 2
            + ((y - center_y) / sigma_y) ** 2
        )
    )
    front = np.clip((z - 0.35) / 0.30, 0.0, 1.0)
    return np.clip(spatial * front, 0.0, 1.0).astype(np.float32)


def _face_part_weights(mesh: trimesh.Trimesh) -> dict[str, np.ndarray]:
    x, y, z = _normalized_vertex_coordinates(mesh)
    return {
        "left_eye": _gaussian_part(
            x, y, z, center_x=-0.30, center_y=0.62, sigma_x=0.16, sigma_y=0.065
        ),
        "right_eye": _gaussian_part(
            x, y, z, center_x=0.30, center_y=0.62, sigma_x=0.16, sigma_y=0.065
        ),
        "left_eyebrow": _gaussian_part(
            x, y, z, center_x=-0.30, center_y=0.70, sigma_x=0.20, sigma_y=0.055
        ),
        "right_eyebrow": _gaussian_part(
            x, y, z, center_x=0.30, center_y=0.70, sigma_x=0.20, sigma_y=0.055
        ),
        "nose": _gaussian_part(
            x, y, z, center_x=0.0, center_y=0.50, sigma_x=0.18, sigma_y=0.15
        ),
        "mouth": _gaussian_part(
            x, y, z, center_x=0.0, center_y=0.36, sigma_x=0.28, sigma_y=0.075
        ),
    }


def _vertex_colors(
    mesh: trimesh.Trimesh,
    identity_index: int,
    *,
    appearance_kind: str = "clay",
    part_weights: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if appearance_kind not in APPEARANCE_KINDS:
        raise ValueError(f"Unknown MHR appearance kind: {appearance_kind}")
    skin = np.asarray(SKIN_TONES[identity_index % len(SKIN_TONES)], dtype=np.float32)
    colors = np.broadcast_to(skin, (len(mesh.vertices), 3)).copy()
    x, y, z = _normalized_vertex_coordinates(mesh)
    variation = 0.96 + 0.035 * y + 0.015 * z - 0.008 * x
    if appearance_kind == "semantic-procedural":
        phase = (identity_index * 0.754877666) % (2.0 * np.pi)
        variation += 0.018 * np.sin(17.0 * x + 11.0 * y + phase)
        variation += 0.010 * np.sin(31.0 * x - 23.0 * y + 0.7 * phase)
    colors *= variation[:, None]
    if appearance_kind == "clay":
        return np.clip(colors, 0.0, 1.0).astype(np.float32)
    if part_weights is None or set(part_weights) != set(FACE_PART_NAMES):
        raise ValueError("Semantic MHR appearance requires all topology-stable parts")

    def blend(weight: np.ndarray, color: np.ndarray, strength: float) -> None:
        amount = np.clip(np.asarray(weight) * strength, 0.0, 1.0)[:, None]
        colors[:] = colors * (1.0 - amount) + color[None] * amount

    brows = np.maximum(
        part_weights["left_eyebrow"],
        part_weights["right_eyebrow"],
    )
    front = np.clip((z - 0.50) / 0.22, 0.0, 1.0)

    def paired_eye_core(sigma_x: float, sigma_y: float) -> np.ndarray:
        left = np.exp(
            -0.5
            * (
                ((x + 0.30) / sigma_x) ** 2
                + ((y - 0.62) / sigma_y) ** 2
            )
        )
        right = np.exp(
            -0.5
            * (
                ((x - 0.30) / sigma_x) ** 2
                + ((y - 0.62) / sigma_y) ** 2
            )
        )
        return np.maximum(left, right) * front

    sclera = paired_eye_core(0.105, 0.026)
    blend(sclera, np.asarray((0.80, 0.79, 0.75), dtype=np.float32), 0.62)
    iris_tones = (
        (0.16, 0.24, 0.25),
        (0.25, 0.18, 0.10),
        (0.20, 0.30, 0.38),
        (0.12, 0.12, 0.10),
    )
    iris = paired_eye_core(0.031, 0.021)
    blend(
        iris,
        np.asarray(iris_tones[identity_index % len(iris_tones)], dtype=np.float32),
        0.95,
    )
    blend(
        paired_eye_core(0.014, 0.014),
        np.asarray((0.025, 0.022, 0.020), dtype=np.float32),
        0.92,
    )
    blend(brows**1.65, np.asarray((0.12, 0.075, 0.05), dtype=np.float32), 0.60)
    lip = np.clip(skin * np.asarray((1.02, 0.80, 0.84)), 0.0, 1.0)
    mouth_core = np.exp(
        -0.5 * ((x / 0.22) ** 2 + ((y - 0.36) / 0.036) ** 2)
    ) * front
    blend(mouth_core, lip, 0.58)

    if identity_index % 5 != 4:
        hairline = 0.77 + 0.025 * np.sin(5.0 * x + phase)
        top = np.clip((y - hairline) / 0.07, 0.0, 1.0)
        sides = np.clip((np.abs(x) - 0.70) / 0.20, 0.0, 1.0) * np.clip(
            (y - 0.55) / 0.18,
            0.0,
            1.0,
        )
        back = np.clip((0.64 - z) / 0.18, 0.0, 1.0) * np.clip(
            (y - 0.60) / 0.18,
            0.0,
            1.0,
        )
        hair = np.maximum.reduce((top, sides, back))
        hair_tones = (
            (0.055, 0.040, 0.030),
            (0.13, 0.085, 0.045),
            (0.025, 0.025, 0.022),
            (0.20, 0.15, 0.09),
        )
        blend(
            hair,
            np.asarray(
                hair_tones[identity_index % len(hair_tones)],
                dtype=np.float32,
            ),
            0.96,
        )
    return np.clip(colors, 0.0, 1.0).astype(np.float32)


def _render_config(spec: MHRSceneSpec, camera_distance: float) -> RenderConfig:
    lighting = LIGHTING[spec.lighting_profile]
    return RenderConfig(
        size=spec.target_dimension,
        projection="perspective",
        perspective_fov_y_deg=32.0,
        camera_distance=float(camera_distance),
        background_rgb=_background_rgb(spec.background_profile),
        ambient=float(lighting["ambient"]),
        diffuse=float(lighting["diffuse"]),
        specular=float(lighting["specular"]),
        shininess=float(lighting["shininess"]),
        light_direction=tuple(lighting["direction"]),
    )


def _initial_camera_distance(spec: MHRSceneSpec) -> float:
    focal = 0.5 * (spec.target_dimension - 1) / np.tan(np.deg2rad(16.0))
    return max(1.8, float(focal * 1.45 / spec.face_height_pixels))


def _render_at_target_height(
    mesh: trimesh.Trimesh,
    spec: MHRSceneSpec,
    part_weights: dict[str, np.ndarray],
    vertex_colors: np.ndarray,
):
    distance = _initial_camera_distance(spec)
    rendered = None
    for _ in range(5):
        rendered = render_mesh(
            mesh,
            CameraSpec(spec.camera_yaw_deg, spec.camera_elevation_deg),
            _render_config(spec, distance),
            (180, 120, 100),
            vertex_part_weights=part_weights,
            vertex_colors=vertex_colors,
        )
        height = _selection_geometry_record(rendered.silhouette)[
            "selection_bbox_height_pixels"
        ]
        if abs(height - spec.face_height_pixels) <= 1:
            break
        distance *= max(height, 1) / float(spec.face_height_pixels)
    if rendered is None:
        raise RuntimeError("MHR renderer did not produce a frame")
    actual_height = _selection_geometry_record(rendered.silhouette)[
        "selection_bbox_height_pixels"
    ]
    if abs(actual_height - spec.face_height_pixels) > 2:
        raise ValueError(
            f"{spec.row_id} face height {actual_height} missed target "
            f"{spec.face_height_pixels}"
        )
    return rendered, float(distance)


def _apply_occlusion(
    rgb: np.ndarray,
    face_mask: np.ndarray,
    parts: dict[str, np.ndarray],
    profile: str | None,
) -> tuple[np.ndarray, list[int] | None]:
    if profile is None:
        return rgb, None
    output = rgb.copy()
    face_rows, face_columns = np.where(face_mask)
    if not len(face_rows):
        raise ValueError("MHR occlusion requires a visible face")
    if profile == "eye_band":
        eyes = (parts["left_eye"] | parts["right_eye"]) & face_mask
        rows, columns = np.where(eyes)
        if not len(rows):
            raise ValueError("MHR eye-band requires visible eye masks")
        half_height = max(2, int(round((np.ptp(face_rows) + 1) * 0.045)))
        center = int(round(0.5 * (rows.min() + rows.max())))
        top, bottom = max(0, center - half_height), min(len(rgb), center + half_height)
        left = max(0, int(columns.min()) - 3)
        right = min(rgb.shape[1], int(columns.max()) + 4)
    elif profile == "cheek_patch":
        top = int(face_rows.min() + 0.54 * (np.ptp(face_rows) + 1))
        bottom = int(face_rows.min() + 0.76 * (np.ptp(face_rows) + 1))
        left = int(face_columns.min() + 0.08 * (np.ptp(face_columns) + 1))
        right = int(face_columns.min() + 0.34 * (np.ptp(face_columns) + 1))
    else:
        raise ValueError(f"Unknown MHR occlusion profile: {profile}")
    output[top:bottom, left:right] = (
        0.35 * output[top:bottom, left:right]
        + 0.65 * np.asarray((0.04, 0.06, 0.08), dtype=np.float32)
    )
    return output, [top, bottom, left, right]


def _camera_record(
    mesh: trimesh.Trimesh,
    spec: MHRSceneSpec,
    distance: float,
    offset_columns: int,
) -> dict:
    lower, upper = mesh.bounds
    center = 0.5 * (lower + upper)
    scale = 1.45 / max(float(np.max(upper - lower)), 1e-8)
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, :3] *= scale
    normalization[:3, 3] = -scale * center
    rotation = camera_transform(
        CameraSpec(spec.camera_yaw_deg, spec.camera_elevation_deg)
    )
    render_to_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    render_to_camera[2, 3] = distance
    object_to_camera = render_to_camera @ rotation @ normalization
    focal = 0.5 * (spec.target_dimension - 1) / np.tan(np.deg2rad(16.0))
    intrinsics = np.asarray(
        (
            (focal, 0.0, 0.5 * (spec.target_dimension - 1) + offset_columns),
            (0.0, focal, 0.5 * (spec.target_dimension - 1)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return {
        "projection": "perspective-opencv",
        "fov_y_degrees": 32.0,
        "intrinsics": intrinsics.tolist(),
        "object_to_camera": object_to_camera.tolist(),
        "camera_distance_normalized_units": distance,
        "native_model_units": "official demo describes vertex offsets as cm",
        "metric_scale_claimed": False,
        "normalization": {
            "center_native": center.tolist(),
            "scale_to_max_extent_1_45": scale,
        },
    }


def _camera_space_normals(
    camera_z: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    values = np.asarray(camera_z, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError("MHR camera intrinsics must be 3x3")
    rows, columns = np.indices(values.shape, dtype=np.float32)
    points = np.stack(
        (
            (columns - intrinsics[0, 2]) * values / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * values / intrinsics[1, 1],
            values,
        ),
        axis=-1,
    )
    interior = np.zeros_like(valid)
    interior[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    tangent_u = np.zeros_like(points)
    tangent_v = np.zeros_like(points)
    tangent_u[:, 1:-1] = points[:, 2:] - points[:, :-2]
    tangent_v[1:-1] = points[2:] - points[:-2]
    normals = np.cross(tangent_v, tangent_u)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    normals[~interior] = np.nan
    return normals.astype(np.float32)


def _render_row(
    model,
    head_mask: np.ndarray,
    spec: MHRSceneSpec,
    device: str,
    part_weights: dict[str, np.ndarray],
    appearance_kind: str,
) -> tuple[dict[str, np.ndarray], dict, np.ndarray, np.ndarray]:
    mesh, identity, expression = _decode_head(model, spec, head_mask, device)
    if any(len(values) != len(mesh.vertices) for values in part_weights.values()):
        raise ValueError("MHR topology-stable part weights no longer match the head mesh")
    identity_index = int(spec.identity_group.rsplit("_", 1)[-1])
    rendered, distance = _render_at_target_height(
        mesh,
        spec,
        part_weights,
        _vertex_colors(
            mesh,
            identity_index,
            appearance_kind=appearance_kind,
            part_weights=part_weights,
        ),
    )
    offset_columns = int(round(spec.horizontal_offset * spec.target_dimension))
    background = _background_rgb(spec.background_profile)
    face_mask = _translate(rendered.silhouette.astype(bool), offset_columns, False)
    face_rgb = _translate(rendered.rgb.astype(np.float32), offset_columns, background)
    relative_depth = _translate(
        rendered.depth.astype(np.float32), offset_columns, 1.0
    )
    camera_z = _translate(
        -rendered.surface_z.astype(np.float32), offset_columns, np.nan
    )
    parts = {
        name: _translate(mask.astype(bool), offset_columns, False) & face_mask
        for name, mask in (rendered.part_masks or {}).items()
    }
    if set(parts) != set(FACE_PART_NAMES):
        raise ValueError(
            f"{spec.row_id} emitted unexpected face parts: {sorted(parts)}"
        )
    missing_parts = sorted(name for name in FACE_PART_NAMES if not np.any(parts[name]))
    if missing_parts:
        raise ValueError(f"{spec.row_id} has empty face parts: {missing_parts}")
    exact_depth, source_rgb = _compose_scene(
        relative_depth,
        face_mask,
        face_rgb,
        phase=_scene_phase(spec.row_id),
        background_profile=spec.background_profile,
    )
    source_rgb, occluder = _apply_occlusion(
        source_rgb, face_mask, parts, spec.occlusion
    )
    camera = _camera_record(mesh, spec, distance, offset_columns)
    geometry = _selection_geometry_record(face_mask)
    geometry.update(
        {
            "face_bbox_xyxy": geometry["selection_bbox_xyxy"],
            "face_bbox_width_pixels": geometry["selection_bbox_width_pixels"],
            "face_bbox_height_pixels": geometry["selection_bbox_height_pixels"],
            "camera": camera,
            "horizontal_offset_columns": offset_columns,
            "occluder_bounds_tblr": occluder,
            "source_mesh_vertices": int(len(mesh.vertices)),
            "source_mesh_faces": int(len(mesh.faces)),
            "source_mesh_degenerate_faces": int(
                np.count_nonzero(mesh.area_faces <= 1e-12)
            ),
            "identity_head_coefficient_rms": float(
                np.sqrt(np.mean(identity[20:40] ** 2))
            ),
            "expression_coefficient_rms": float(
                np.sqrt(np.mean(expression**2))
            ),
        }
    )
    arrays = {
        "source": np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8),
        "selection_mask": face_mask.astype(np.uint8) * 255,
        "exact_depth": exact_depth.astype(np.float32),
        "exact_camera_depth": camera_z.astype(np.float32),
        "exact_camera_normals": _camera_space_normals(
            camera_z,
            face_mask,
            np.asarray(camera["intrinsics"], dtype=np.float32),
        ),
        **{f"part:{name}": parts[name] for name in FACE_PART_NAMES},
    }
    return arrays, geometry, identity, expression


def _write_array_record(path: Path, root: Path, values: np.ndarray) -> dict:
    if path.suffix.lower() == ".png":
        Image.fromarray(values).save(path)
    else:
        np.save(path, values, allow_pickle=False)
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}


def _write_row(
    root: Path,
    spec: MHRSceneSpec,
    arrays: dict[str, np.ndarray],
    geometry: dict,
    identity: np.ndarray,
    expression: np.ndarray,
) -> tuple[dict, dict | None]:
    row_dir = root / "rows" / spec.row_id
    parts_dir = row_dir / "exact_face_parts"
    parts_dir.mkdir(parents=True)
    source = _write_array_record(row_dir / "source.png", root, arrays["source"])
    selection = _write_array_record(
        row_dir / "selection_mask.png", root, arrays["selection_mask"]
    )
    exact_depth = _write_array_record(
        row_dir / "exact_depth.npy", root, arrays["exact_depth"]
    )
    camera_depth = _write_array_record(
        row_dir / "exact_camera_depth.npy", root, arrays["exact_camera_depth"]
    )
    camera_normals = _write_array_record(
        row_dir / "exact_camera_normals.npy",
        root,
        arrays["exact_camera_normals"],
    )
    parts = {
        name: _write_array_record(
            parts_dir / f"{name}.png",
            root,
            arrays[f"part:{name}"].astype(np.uint8) * 255,
        )
        for name in FACE_PART_NAMES
    }
    targets = None
    if spec.split == "train":
        target_dir = row_dir / "geometry_targets"
        target_dir.mkdir()
        targets = {
            "identity": _write_array_record(
                target_dir / "identity.npy", root, identity.astype("<f4")
            ),
            "expression": _write_array_record(
                target_dir / "expression.npy", root, expression.astype("<f4")
            ),
        }
    record = {
        "row_id": spec.row_id,
        "split": spec.split,
        "identity_group": spec.identity_group,
        "expression": spec.expression_profile,
        "spec": asdict(spec),
        "render": geometry,
        "source": source,
        "selection_mask": selection,
        "exact_depth": exact_depth,
        "exact_camera_depth": camera_depth,
        "exact_camera_normals": camera_normals,
        "exact_face_parts": parts,
    }
    supervision = (
        {
            "row_id": spec.row_id,
            "split": spec.split,
            "geometry_targets": targets,
        }
        if targets is not None
        else None
    )
    return record, supervision


def _write_training_supervision_manifest(
    root: Path,
    supervision_rows: list[dict],
) -> dict:
    manifest = {
        "schema_version": 1,
        "provider": MHR_PROVIDER,
        "training_only": True,
        "must_be_excluded_from_compact_evidence": True,
        "row_count": len(supervision_rows),
        "rows": supervision_rows,
    }
    path = root / "training_supervision.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "row_count": len(supervision_rows),
        "excluded_from_compact_evidence": True,
    }


def render_training_corpus(
    mhr_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    limit: int | None = None,
    matrix_kind: str = "pilot",
    appearance_kind: str = "clay",
) -> dict:
    import torch

    mhr_root = Path(mhr_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing corpus: {output_dir}")
    preflight = preflight_mhr_root(mhr_root)
    if not preflight["runnable"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError(f"MHR preflight failed: {', '.join(failed)}")
    rows = select_scene_rows(limit, matrix_kind)
    matrix = MATRICES[matrix_kind]
    if appearance_kind not in APPEARANCE_KINDS:
        raise ValueError(f"Unknown MHR appearance kind: {appearance_kind}")
    if not rows:
        raise ValueError("MHR corpus selected zero rows")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    started = time.perf_counter()
    try:
        model = _load_model(mhr_root, device)
        with np.load(mhr_root / MHR_HEAD_MASK_RELATIVE_PATH) as masks:
            head_mask = np.asarray(masks["head_mask"], dtype=np.float32)
        template_mesh = _decode_head_coefficients(
            model,
            np.zeros(MHR_IDENTITY_DIMENSION, dtype=np.float32),
            np.zeros(MHR_EXPRESSION_DIMENSION, dtype=np.float32),
            head_mask,
            device,
        )
        part_weights = _face_part_weights(template_mesh)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        records = []
        supervision_rows = []
        for index, spec in enumerate(rows, start=1):
            arrays, geometry, identity, expression = _render_row(
                model,
                head_mask,
                spec,
                device,
                part_weights,
                appearance_kind,
            )
            record, supervision = _write_row(
                temporary,
                spec,
                arrays,
                geometry,
                identity,
                expression,
            )
            records.append(record)
            if supervision is not None:
                supervision_rows.append(supervision)
            print(f"rendered {index}/{len(rows)} MHR rows", flush=True)
        supervision_manifest = _write_training_supervision_manifest(
            temporary,
            supervision_rows,
        )
        corpus_complete = tuple(row.row_id for row in rows) == tuple(
            row.row_id for row in matrix
        )
        deterministic_generation = str(device) == "cpu"
        split_identities = {
            split: sorted(
                {row.identity_group for row in rows if row.split == split}
            )
            for split in ("train", "validation", "sealed")
        }
        summary = {
            "schema_version": 1,
            "provider": MHR_PROVIDER,
            "privacy_safe_synthetic": True,
            "experimental_corpus": True,
            "pilot_corpus": matrix_kind == "pilot",
            "matrix_kind": matrix_kind,
            "appearance": {
                "kind": appearance_kind,
                "semantic_procedural_version": (
                    SEMANTIC_APPEARANCE_VERSION
                    if appearance_kind == "semantic-procedural"
                    else None
                ),
                "topology_stable_part_weights_used_for_albedo": (
                    appearance_kind == "semantic-procedural"
                ),
                "photorealism_claimed": False,
            },
            "corpus_complete": corpus_complete,
            "training_eligible": bool(
                matrix_kind == "training"
                and corpus_complete
                and deterministic_generation
            ),
            "production_training_eligible": False,
            "promotion_eligible": False,
            "identity_disjoint_split_design": True,
            "identity_disjoint_splits": corpus_complete,
            "source_geometry_training_and_evaluation_only": True,
            "source_revision": MHR_SOURCE_REVISION,
            "corpus_seed": CORPUS_SEED,
            "row_count": len(records),
            "full_balanced_row_count": len(matrix),
            "selection_strategy": (
                "full-matrix" if corpus_complete else "stratified-smoke"
            ),
            "split_identities": split_identities,
            "source": {
                "repository": MHR_SOURCE_URL,
                "release": MHR_RELEASE_URL,
                "license": MHR_LICENSE,
                "release_archive_sha256": MHR_RELEASE_ARCHIVE_SHA256,
            },
            "preflight": _portable_preflight(preflight, mhr_root),
            "depth_target_provenance": {
                "exact_depth": "deterministic normalized scene depth",
                "exact_camera_depth": (
                    "floating camera-Z after max-extent normalization"
                ),
                "metric_scale_claimed": False,
                "camera_convention": "OpenCV +X right, +Y down, +Z forward",
                "exact_camera_normals": (
                    "camera-space finite-difference normals from unprojected "
                    "camera-Z and pinned intrinsics; -Z faces the camera"
                ),
            },
            "face_part_provenance": {
                "representation": "fixed LOD1 topology weights",
                "source": "neutral-template anatomical proxy regions",
                "stable_across_identity_and_expression": True,
                "not_claimed": "official MHR semantic segmentation",
            },
            "geometry_target_contract": {
                "training_rows_only": True,
                "identity_layout": {
                    "dimension": MHR_IDENTITY_DIMENSION,
                    "head_slice": list(MHR_HEAD_IDENTITY_SLICE),
                    "body_and_hand_coefficients_zero": True,
                },
                "expression_dimension": MHR_EXPRESSION_DIMENSION,
                "compact_evidence_must_exclude_raw_targets": True,
                "supervision_manifest": supervision_manifest,
            },
            "deterministic_generation": {
                "required_device_for_training": "cpu",
                "device_requirement_met": deterministic_generation,
                "runtime_excluded_from_manifest": True,
            },
            "rows": records,
        }
        summary_path = temporary / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generation_report = {
            "schema_version": 1,
            "summary_sha256": _sha256(summary_path),
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "peak_torch_allocated_gib": (
                float(torch.cuda.max_memory_allocated(device) / (1024**3))
                if str(device).startswith("cuda")
                else 0.0
            ),
        }
        (temporary / "generation_report.json").write_text(
            json.dumps(generation_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhr-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--matrix", choices=tuple(MATRICES), default="pilot")
    parser.add_argument(
        "--appearance",
        choices=APPEARANCE_KINDS,
        default="clay",
    )
    args = parser.parse_args()
    summary = render_training_corpus(
        args.mhr_root,
        args.output_dir,
        device=args.device,
        limit=args.limit,
        matrix_kind=args.matrix,
        appearance_kind=args.appearance,
    )
    print(json.dumps({"row_count": summary["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
