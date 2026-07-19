"""Gate a photoreal RGB transfer against exact MHR face geometry."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

from backend.benchmark.mhr_face_training_corpus import FACE_PART_NAMES
from backend.face_depth_refinement import detect_face_regions_in_roi


DIFFSYNTH_SOURCE_URL = "https://github.com/modelscope/DiffSynth-Studio.git"
DIFFSYNTH_SOURCE_REVISION = "fb337fbb90945ff829de69dbd44ded618f73e889"
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_MODEL_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
ZIMAGE_CONTROLNET_ID = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
ZIMAGE_CONTROLNET_REVISION = "5155fc56d17821007d6f62ac192c09e0f0e72016"
ZIMAGE_CONTROLNET_FILENAME = (
    "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
)
ZIMAGE_CONTROLNET_SIZE_BYTES = 6_712_485_600
ZIMAGE_CONTROLNET_SHA256 = (
    "d1251cc7bc3486bc61d25c3be498ef394c31c85ddf4ee9137d2e933411f4a689"
)
DEFAULT_ROW_ID = "mhr_training_identity_000__scene_00"
DEFAULT_TRAINING_SUPERVISION_SHA256 = (
    "42cb6169170f0dd31a8715aefdf45229749d4418d54d19e4d5c48b8452c59acb"
)
DEPTH_EVALUATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_EVALUATOR_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
DEPTH_EVALUATOR_REQUIRED_FILES = {
    "config.json": (
        950,
        "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
    ),
    "model.safetensors": (
        99_173_660,
        "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
    ),
    "preprocessor_config.json": (
        775,
        "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
    ),
}
DEPTH_EVALUATOR_DEPENDENCY_VERSIONS = {
    "torch": "2.7.1+cu128",
    "transformers": "5.13.0",
}
DEFAULT_SEED = 20260719
DEFAULT_DENOISING_STRENGTH = 0.35
DEFAULT_CONTROL_SCALE = 0.90
DEFAULT_STEPS = 8
DEFAULT_PROMPT = (
    "A natural documentary photograph of the same adult person and the same "
    "scene shown in the input. Preserve the exact camera, head pose, facial "
    "expression, silhouette, eye positions, nose, mouth, ears, lighting "
    "direction, occlusions, and background geometry. Replace only the "
    "synthetic face material with realistic human skin, hair, eyes, and lips. "
    "No retouching, no beautification, no face reshaping, no added objects."
)

DIFFSYNTH_REQUIRED_FILES = (
    "LICENSE",
    "pyproject.toml",
    "diffsynth/core/loader/model.py",
    "diffsynth/core/vram/disk_map.py",
    "diffsynth/core/vram/layers.py",
    "diffsynth/models/z_image_controlnet.py",
    "diffsynth/models/z_image_dit.py",
    "diffsynth/pipelines/z_image.py",
)

# Every file consumed from the pinned model snapshot is content-addressed.
ZIMAGE_REQUIRED_FILES = {
    "text_encoder/config.json": (
        726,
        "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    ),
    "text_encoder/generation_config.json": (
        239,
        "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    ),
    "text_encoder/model-00001-of-00003.safetensors": (
        3_957_900_840,
        "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    ),
    "text_encoder/model-00002-of-00003.safetensors": (
        3_987_450_520,
        "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    ),
    "text_encoder/model-00003-of-00003.safetensors": (
        99_630_640,
        "7ca841ee75b9c61267c0c6148fd8d096d3d21b6d3e161256a9b878154f91fc52",
    ),
    "text_encoder/model.safetensors.index.json": (
        32_819,
        "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2",
    ),
    "tokenizer/merges.txt": (
        1_671_853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "tokenizer/tokenizer.json": (
        11_422_654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    "tokenizer/tokenizer_config.json": (
        9_732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
    "tokenizer/vocab.json": (
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    "transformer/config.json": (
        473,
        "06d49d1ae533f825c4704cc616fc8bc7826e9d449dde292cc7a95163c69af24a",
    ),
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": (
        9_973_693_184,
        "95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",
    ),
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": (
        9_973_714_824,
        "a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",
    ),
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": (
        4_672_282_880,
        "aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",
    ),
    "transformer/diffusion_pytorch_model.safetensors.index.json": (
        48_969,
        "182a119d8018bfc61c9a62685a384af3b7d4a2f8aabbe766e07c9e1eda5b97ab",
    ),
    "vae/config.json": (
        805,
        "e80af1e64a71883a9d10c3159d2e493e5934508da57852f6a180ae6ae63b14bd",
    ),
    "vae/diffusion_pytorch_model.safetensors": (
        167_666_902,
        "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",
    ),
}
ZIMAGE_TEXT_ENCODER_SHARDS = tuple(
    name
    for name in ZIMAGE_REQUIRED_FILES
    if name.startswith("text_encoder/model-") and name.endswith(".safetensors")
)
ZIMAGE_TRANSFORMER_SHARDS = tuple(
    name
    for name in ZIMAGE_REQUIRED_FILES
    if name.startswith("transformer/diffusion_pytorch_model-")
    and name.endswith(".safetensors")
)

ALIGNMENT_THRESHOLDS = {
    "minimum_face_mask_iou": 0.98,
    "maximum_face_boundary_h95_pixels": 1.5,
    "maximum_landmark_median_pixels": 1.0,
    "maximum_landmark_p95_pixels": 2.0,
    "maximum_face_center_displacement_ratio": 0.015,
    "minimum_face_height_retention": 0.98,
    "maximum_face_height_retention": 1.02,
    "minimum_part_mask_iou": 0.95,
    "maximum_part_boundary_h95_pixels": 1.5,
    "maximum_part_centroid_displacement_pixels": 1.0,
    "minimum_inside_rgb_rms": 3.0,
    "minimum_inside_changed_fraction": 0.20,
    "maximum_exact_part_iou_regression": 0.02,
    "maximum_exact_part_boundary_h95_increase_pixels": 1.0,
    "maximum_exact_part_centroid_increase_pixels": 1.0,
}
EXACT_PART_ABSOLUTE_THRESHOLDS = {
    "left_eye": (0.0, 18.0, 18.0),
    "right_eye": (0.0, 12.0, 10.0),
    "left_eyebrow": (0.0, 8.0, 5.0),
    "right_eyebrow": (0.20, 8.0, 7.0),
    "nose": (0.30, 11.0, 4.0),
    "mouth": (0.12, 8.0, 4.0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _asset_record(path: Path, size: int, sha256: str) -> dict:
    exists = path.is_file()
    actual_size = int(path.stat().st_size) if exists else None
    actual_sha256 = _sha256(path) if exists and actual_size == size else None
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": actual_size,
        "expected_size_bytes": size,
        "size_pinned": actual_size == size,
        "sha256": actual_sha256,
        "expected_sha256": sha256,
        "hash_pinned": actual_sha256 == sha256,
    }


def _adapter_repository_provenance() -> dict:
    adapter_path = Path(__file__).resolve()
    root_text = _git_output(adapter_path.parent, "rev-parse", "--show-toplevel")
    root = Path(root_text).resolve() if root_text else None
    revision = _git_output(root, "rev-parse", "HEAD") if root else None
    status = (
        _git_output(root, "status", "--porcelain=v1", "--untracked-files=all")
        if root
        else None
    )
    try:
        relative = adapter_path.relative_to(root).as_posix() if root else None
    except ValueError:
        relative = None
    tracked = (
        _git_output(root, "ls-files", "--error-unmatch", relative) == relative
        if root and relative
        else False
    )
    return {
        "root": str(root) if root else None,
        "revision": revision,
        "clean": status == "",
        "status": status.splitlines() if status else [],
        "adapter_relative_path": relative,
        "adapter_tracked": tracked,
    }


def zimage_preflight(
    provider_root: str | Path,
    model_root: str | Path,
    controlnet_path: str | Path,
) -> dict:
    """Verify the exact executable source and every model file before load."""

    provider_root = Path(provider_root).resolve()
    model_root = Path(model_root).resolve()
    controlnet_path = Path(controlnet_path).resolve()
    revision = _git_output(provider_root, "rev-parse", "HEAD")
    status = _git_output(
        provider_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    missing_source = [
        relative
        for relative in DIFFSYNTH_REQUIRED_FILES
        if not (provider_root / relative).is_file()
    ]
    model_files = {
        relative: _asset_record(model_root / relative, size, sha256)
        for relative, (size, sha256) in ZIMAGE_REQUIRED_FILES.items()
    }
    expected_safetensors = {
        relative
        for relative in ZIMAGE_REQUIRED_FILES
        if relative.endswith(".safetensors")
    }
    actual_safetensors = {
        path.relative_to(model_root).as_posix()
        for path in model_root.rglob("*.safetensors")
        if path.is_file()
    }
    unexpected_safetensors = sorted(actual_safetensors - expected_safetensors)
    actual_consumed_files = {
        path.relative_to(model_root).as_posix()
        for directory in ("text_encoder", "tokenizer", "transformer", "vae")
        for path in (model_root / directory).rglob("*")
        if path.is_file()
    }
    unexpected_consumed_files = sorted(
        actual_consumed_files - set(ZIMAGE_REQUIRED_FILES)
    )
    controlnet = _asset_record(
        controlnet_path,
        ZIMAGE_CONTROLNET_SIZE_BYTES,
        ZIMAGE_CONTROLNET_SHA256,
    )
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "einops",
            "modelscope",
            "numpy",
            "PIL",
            "safetensors",
            "torch",
            "transformers",
        )
    }
    adapter_repository = _adapter_repository_provenance()
    checks = {
        "source_revision_pinned": revision == DIFFSYNTH_SOURCE_REVISION,
        "source_clean": status == "",
        "source_files_complete": not missing_source,
        "model_files_complete": all(item["exists"] for item in model_files.values()),
        "model_sizes_pinned": all(item["size_pinned"] for item in model_files.values()),
        "model_hashes_pinned": all(
            item["hash_pinned"] for item in model_files.values()
        ),
        "model_has_no_unexpected_safetensors": not unexpected_safetensors,
        "model_has_no_unexpected_consumed_files": not unexpected_consumed_files,
        "controlnet_exists": controlnet["exists"],
        "controlnet_size_pinned": controlnet["size_pinned"],
        "controlnet_hash_pinned": controlnet["hash_pinned"],
        "dependencies_available": all(dependencies.values()),
        "adapter_repository_clean": adapter_repository["clean"],
        "adapter_is_tracked": adapter_repository["adapter_tracked"],
    }
    adapter_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "provider": "zimage-turbo-controlnet-union-2.1-2602-8steps",
        "source": {
            "url": DIFFSYNTH_SOURCE_URL,
            "expected_revision": DIFFSYNTH_SOURCE_REVISION,
            "actual_revision": revision,
            "tracked_status": status,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "model": {
            "id": ZIMAGE_MODEL_ID,
            "revision": ZIMAGE_MODEL_REVISION,
            "root": str(model_root),
            "files": model_files,
            "unexpected_safetensors": unexpected_safetensors,
            "unexpected_consumed_files": unexpected_consumed_files,
        },
        "controlnet": {
            "id": ZIMAGE_CONTROLNET_ID,
            "revision": ZIMAGE_CONTROLNET_REVISION,
            "filename": ZIMAGE_CONTROLNET_FILENAME,
            **controlnet,
        },
        "dependencies": dependencies,
        "dependency_versions": {
            name: _distribution_version(name)
            for name in (
                "diffusers",
                "einops",
                "modelscope",
                "numpy",
                "Pillow",
                "safetensors",
                "torch",
                "transformers",
            )
        },
        "adapter": {"path": str(adapter_path), "sha256": _sha256(adapter_path)},
        "adapter_repository": adapter_repository,
        "license": {
            "source": "Apache-2.0",
            "model": "Apache-2.0",
            "controlnet": "Apache-2.0",
            "production_eligible": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def depth_evaluator_preflight(root: str | Path) -> dict:
    root = Path(root).resolve()
    files = {
        relative: _asset_record(root / relative, size, sha256)
        for relative, (size, sha256) in DEPTH_EVALUATOR_REQUIRED_FILES.items()
    }
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    unexpected_files = sorted(actual_files - set(DEPTH_EVALUATOR_REQUIRED_FILES))
    dependency_versions = {
        name: _distribution_version(name)
        for name in DEPTH_EVALUATOR_DEPENDENCY_VERSIONS
    }
    checks = {
        "files_complete": all(item["exists"] for item in files.values()),
        "sizes_pinned": all(item["size_pinned"] for item in files.values()),
        "hashes_pinned": all(item["hash_pinned"] for item in files.values()),
        "no_unexpected_files": not unexpected_files,
        "dependency_versions_pinned": dependency_versions
        == DEPTH_EVALUATOR_DEPENDENCY_VERSIONS,
    }
    return {
        "id": DEPTH_EVALUATOR_ID,
        "revision": DEPTH_EVALUATOR_REVISION,
        "root": str(root),
        "files": files,
        "unexpected_files": unexpected_files,
        "dependency_versions": dependency_versions,
        "expected_dependency_versions": DEPTH_EVALUATOR_DEPENDENCY_VERSIONS,
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def _safe_verified_asset(root: Path, record: dict) -> Path:
    relative = Path(str(record.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("MHR asset path must be a safe relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("MHR asset resolves outside the corpus") from exc
    if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
        raise ValueError(f"MHR asset hash mismatch: {relative.as_posix()}")
    return path


def load_parent_row(
    corpus_root: str | Path,
    expected_summary_sha256: str,
    row_id: str = DEFAULT_ROW_ID,
    expected_training_supervision_sha256: str = DEFAULT_TRAINING_SUPERVISION_SHA256,
) -> tuple[dict, dict, dict]:
    corpus_root = Path(corpus_root).resolve()
    summary_path = corpus_root / "summary.json"
    actual_summary_sha256 = _sha256(summary_path)
    if actual_summary_sha256 != expected_summary_sha256:
        raise ValueError("Parent MHR summary SHA256 does not match the pinned input")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [row for row in summary.get("rows", ()) if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one parent MHR row named {row_id}")
    row = matches[0]
    if row.get("split") != "train" or row.get("spec", {}).get("split") != "train":
        raise ValueError("Photoreal domain transfer is restricted to train-only rows")
    supervision_path = corpus_root / "training_supervision.json"
    supervision_sha256 = _sha256(supervision_path)
    if supervision_sha256 != expected_training_supervision_sha256:
        raise ValueError(
            "MHR training supervision SHA256 does not match the pinned input"
        )
    supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
    supervision_rows = [
        item for item in supervision.get("rows", ()) if item.get("row_id") == row_id
    ]
    if len(supervision_rows) != 1 or supervision_rows[0].get("split") != "train":
        raise ValueError("Expected one train-only MHR supervision row")
    supervision_row = supervision_rows[0]
    geometry_targets = supervision_row.get("geometry_targets") or {}
    if not geometry_targets:
        raise ValueError("MHR supervision row has no geometry targets")
    records = {
        "source": row["source"],
        "selection_mask": row["selection_mask"],
        "exact_depth": row["exact_depth"],
        "exact_camera_depth": row["exact_camera_depth"],
        "exact_camera_normals": row["exact_camera_normals"],
        **{f"part:{name}": row["exact_face_parts"][name] for name in FACE_PART_NAMES},
        **{
            f"geometry_target:{name}": record
            for name, record in geometry_targets.items()
        },
    }
    paths = {
        name: _safe_verified_asset(corpus_root, record)
        for name, record in records.items()
    }
    provenance = {
        "root": str(corpus_root),
        "summary_path": str(summary_path),
        "summary_sha256": actual_summary_sha256,
        "training_supervision_path": str(supervision_path),
        "training_supervision_sha256": supervision_sha256,
        "row_id": row_id,
        "asset_sha256": {name: records[name]["sha256"] for name in records},
    }
    return row, paths, provenance


def make_inverse_depth_control(exact_depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(exact_depth, dtype=np.float32)
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError("Exact scene depth must be a finite 2D array")
    if float(np.min(depth)) < -1e-6 or float(np.max(depth)) > 1.0 + 1e-6:
        raise ValueError(
            "Exact scene depth must use the pinned normalized [0, 1] range"
        )
    inverse = np.rint((1.0 - np.clip(depth, 0.0, 1.0)) * 255.0).astype(np.uint8)
    return np.repeat(inverse[..., None], 3, axis=2)


def composite_face_only(
    parent_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    selection_mask: np.ndarray,
) -> np.ndarray:
    parent = np.asarray(parent_rgb, dtype=np.uint8)
    generated = np.asarray(generated_rgb, dtype=np.uint8)
    mask = np.asarray(selection_mask) > 0
    if parent.ndim != 3 or parent.shape[2] != 3:
        raise ValueError("Parent source must be RGB")
    if generated.shape != parent.shape:
        raise ValueError(
            "Generated RGB dimensions changed; automatic resize is forbidden"
        )
    if mask.shape != parent.shape[:2]:
        raise ValueError("Selection mask dimensions do not match the source")
    output = parent.copy()
    output[mask] = generated[mask]
    return output


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))


def _boundary_h95(first: np.ndarray, second: np.ndarray) -> float:
    first_boundary = _boundary(first)
    second_boundary = _boundary(second)
    if not np.any(first_boundary) or not np.any(second_boundary):
        return math.inf
    first_to_second = distance_transform_edt(~second_boundary)[first_boundary]
    second_to_first = distance_transform_edt(~first_boundary)[second_boundary]
    return float(
        max(np.percentile(first_to_second, 95), np.percentile(second_to_first, 95))
    )


def _centroid(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    return np.mean(points[:, ::-1], axis=0) if len(points) else np.full(2, np.nan)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(np.asarray(mask, dtype=bool))
    if not len(rows):
        raise ValueError("Face alignment mask is empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    )


def _landmarks_pixels(region: dict, shape: tuple[int, int]) -> np.ndarray:
    landmarks = np.asarray(region.get("landmarks_xyz"), dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[0] < 468 or landmarks.shape[1] < 2:
        raise ValueError("Alignment requires a complete 468-landmark face")
    height, width = shape
    points = landmarks[:468, :2].copy()
    points[:, 0] *= max(width - 1, 1)
    points[:, 1] *= max(height - 1, 1)
    if not np.all(np.isfinite(points)):
        raise ValueError("Alignment landmarks are non-finite")
    return points


def _strict_region(
    image_rgb: np.ndarray,
    selection_mask: np.ndarray,
    detector: Callable | None,
) -> tuple[dict, dict, list[str]]:
    kwargs = {
        "max_faces": 2,
        "min_face_pixels": 32,
        "allow_selection_detail_fallback": False,
    }
    if detector is not None:
        kwargs["detector"] = detector
    regions, errors, telemetry = detect_face_regions_in_roi(
        image_rgb,
        selection_mask,
        **kwargs,
    )
    complete = [
        region
        for region in regions
        if int(region.get("landmark_count") or 0) >= 468
        and region.get("landmarks_xyz") is not None
        and all(
            np.any(np.asarray((region.get("part_masks") or {}).get(name, 0)) > 0)
            for name in FACE_PART_NAMES
        )
    ]
    if len(complete) != 1:
        raise RuntimeError(
            f"Expected exactly one complete face in the selected ROI; found {len(complete)}"
        )
    return complete[0], telemetry, errors


def _anatomical_ordering_signature(region: dict) -> dict[str, int] | None:
    parts = {name: _centroid(region["part_masks"][name]) for name in FACE_PART_NAMES}
    if not all(np.all(np.isfinite(value)) for value in parts.values()):
        return None
    eye_y = 0.5 * (parts["left_eye"][1] + parts["right_eye"][1])
    brow_y = 0.5 * (parts["left_eyebrow"][1] + parts["right_eyebrow"][1])
    deltas = {
        "eye_left_to_right_x": parts["right_eye"][0] - parts["left_eye"][0],
        "eyebrow_left_to_right_x": (
            parts["right_eyebrow"][0] - parts["left_eyebrow"][0]
        ),
        "eyebrow_to_eye_y": eye_y - brow_y,
        "eye_to_nose_y": parts["nose"][1] - eye_y,
        "nose_to_mouth_y": parts["mouth"][1] - parts["nose"][1],
    }
    signature = {name: int(np.sign(value)) for name, value in deltas.items()}
    return signature if all(value != 0 for value in signature.values()) else None


def _anatomical_ordering_matches(parent_region: dict, candidate_region: dict) -> bool:
    parent = _anatomical_ordering_signature(parent_region)
    candidate = _anatomical_ordering_signature(candidate_region)
    return parent is not None and candidate == parent


def measure_rgb_alignment(
    parent_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    selection_mask: np.ndarray,
    *,
    detector: Callable | None = None,
    thresholds: dict | None = None,
    exact_part_masks: dict[str, np.ndarray] | None = None,
) -> dict:
    """Measure unregistered RGB motion before a row can become training data."""

    thresholds = dict(ALIGNMENT_THRESHOLDS if thresholds is None else thresholds)
    parent = np.asarray(parent_rgb, dtype=np.uint8)
    candidate = np.asarray(candidate_rgb, dtype=np.uint8)
    mask = np.asarray(selection_mask) > 0
    if candidate.shape != parent.shape:
        return {
            "schema_version": 1,
            "checks": {"output_dimensions_exact": False},
            "passed": False,
            "reason": "output_dimensions_changed",
        }
    parent_region, parent_detection, parent_errors = _strict_region(
        parent, mask, detector
    )
    candidate_region, candidate_detection, candidate_errors = _strict_region(
        candidate, mask, detector
    )
    parent_face = np.asarray(parent_region["face_mask"]) > 0
    candidate_face = np.asarray(candidate_region["face_mask"]) > 0
    face_bbox = _bbox(mask)
    face_height = float(face_bbox[3] - face_bbox[1])
    parent_bbox = _bbox(parent_face)
    candidate_bbox = _bbox(candidate_face)
    parent_center = np.asarray(
        ((parent_bbox[0] + parent_bbox[2]) / 2, (parent_bbox[1] + parent_bbox[3]) / 2)
    )
    candidate_center = np.asarray(
        (
            (candidate_bbox[0] + candidate_bbox[2]) / 2,
            (candidate_bbox[1] + candidate_bbox[3]) / 2,
        )
    )
    center_ratio = float(np.linalg.norm(candidate_center - parent_center) / face_height)
    height_retention = float(
        (candidate_bbox[3] - candidate_bbox[1])
        / max(parent_bbox[3] - parent_bbox[1], 1)
    )
    parent_landmarks = _landmarks_pixels(parent_region, parent.shape[:2])
    candidate_landmarks = _landmarks_pixels(candidate_region, candidate.shape[:2])
    landmark_error = np.linalg.norm(candidate_landmarks - parent_landmarks, axis=1)
    part_metrics = {}
    exact_part_metrics = {}
    if exact_part_masks is not None and set(exact_part_masks) != set(FACE_PART_NAMES):
        raise ValueError("Exact alignment requires all six named face-part masks")
    for name in FACE_PART_NAMES:
        parent_part = np.asarray(parent_region["part_masks"][name]) > 0
        candidate_part = np.asarray(candidate_region["part_masks"][name]) > 0
        part_metrics[name] = {
            "mask_iou": _mask_iou(parent_part, candidate_part),
            "boundary_h95_pixels": _boundary_h95(parent_part, candidate_part),
            "centroid_displacement_pixels": float(
                np.linalg.norm(_centroid(candidate_part) - _centroid(parent_part))
            ),
        }
        if exact_part_masks is not None:
            exact_part = np.asarray(exact_part_masks[name]) > 0
            if exact_part.shape != mask.shape or not np.any(exact_part):
                raise ValueError(f"Exact face-part mask {name!r} is empty or mis-sized")
            parent_exact = {
                "mask_iou": _mask_iou(exact_part, parent_part),
                "boundary_h95_pixels": _boundary_h95(exact_part, parent_part),
                "centroid_displacement_pixels": float(
                    np.linalg.norm(_centroid(parent_part) - _centroid(exact_part))
                ),
            }
            candidate_exact = {
                "mask_iou": _mask_iou(exact_part, candidate_part),
                "boundary_h95_pixels": _boundary_h95(exact_part, candidate_part),
                "centroid_displacement_pixels": float(
                    np.linalg.norm(_centroid(candidate_part) - _centroid(exact_part))
                ),
            }
            minimum_iou, maximum_h95, maximum_centroid = EXACT_PART_ABSOLUTE_THRESHOLDS[
                name
            ]
            exact_part_metrics[name] = {
                "parent": parent_exact,
                "candidate": candidate_exact,
                "absolute_thresholds": {
                    "minimum_mask_iou": minimum_iou,
                    "maximum_boundary_h95_pixels": maximum_h95,
                    "maximum_centroid_displacement_pixels": maximum_centroid,
                },
                "absolute_passed": bool(
                    candidate_exact["mask_iou"] >= minimum_iou
                    and candidate_exact["boundary_h95_pixels"] <= maximum_h95
                    and candidate_exact["centroid_displacement_pixels"]
                    <= maximum_centroid
                ),
                "iou_regression": float(
                    parent_exact["mask_iou"] - candidate_exact["mask_iou"]
                ),
                "boundary_h95_increase_pixels": float(
                    candidate_exact["boundary_h95_pixels"]
                    - parent_exact["boundary_h95_pixels"]
                ),
                "centroid_increase_pixels": float(
                    candidate_exact["centroid_displacement_pixels"]
                    - parent_exact["centroid_displacement_pixels"]
                ),
            }
    difference = candidate.astype(np.float32) - parent.astype(np.float32)
    inside = difference[mask]
    outside = difference[~mask]
    inside_rms = float(np.sqrt(np.mean(inside**2))) if inside.size else 0.0
    changed_fraction = (
        float(np.mean(np.max(np.abs(inside), axis=1) >= 2.0)) if inside.size else 0.0
    )
    metrics = {
        "face_mask_iou": _mask_iou(parent_face, candidate_face),
        "face_boundary_h95_pixels": _boundary_h95(parent_face, candidate_face),
        "face_center_displacement_ratio": center_ratio,
        "face_height_retention": height_retention,
        "landmark_median_pixels": float(np.median(landmark_error)),
        "landmark_p95_pixels": float(np.percentile(landmark_error, 95)),
        "inside_rgb_rms": inside_rms,
        "inside_changed_fraction": changed_fraction,
        "outside_changed_pixels": int(np.count_nonzero(np.max(np.abs(outside), axis=1)))
        if outside.size
        else 0,
        "parts": part_metrics,
        "exact_part_alignment": {
            "available": exact_part_masks is not None,
            "comparison": "candidate-and-parent-detector-masks-versus-exact-projection",
            "parts": exact_part_metrics,
        },
        "parent_anatomical_ordering": _anatomical_ordering_signature(parent_region),
        "candidate_anatomical_ordering": _anatomical_ordering_signature(
            candidate_region
        ),
        "anatomical_ordering_preserved": _anatomical_ordering_matches(
            parent_region, candidate_region
        ),
    }
    checks = {
        "output_dimensions_exact": True,
        "outside_selection_bit_exact": metrics["outside_changed_pixels"] == 0,
        "face_mask_iou": metrics["face_mask_iou"]
        >= thresholds["minimum_face_mask_iou"],
        "face_boundary_h95": metrics["face_boundary_h95_pixels"]
        <= thresholds["maximum_face_boundary_h95_pixels"],
        "landmark_median": metrics["landmark_median_pixels"]
        <= thresholds["maximum_landmark_median_pixels"],
        "landmark_p95": metrics["landmark_p95_pixels"]
        <= thresholds["maximum_landmark_p95_pixels"],
        "face_center": metrics["face_center_displacement_ratio"]
        <= thresholds["maximum_face_center_displacement_ratio"],
        "face_height": thresholds["minimum_face_height_retention"]
        <= metrics["face_height_retention"]
        <= thresholds["maximum_face_height_retention"],
        "all_parts": all(
            item["mask_iou"] >= thresholds["minimum_part_mask_iou"]
            and item["boundary_h95_pixels"]
            <= thresholds["maximum_part_boundary_h95_pixels"]
            and item["centroid_displacement_pixels"]
            <= thresholds["maximum_part_centroid_displacement_pixels"]
            for item in part_metrics.values()
        ),
        "anatomical_ordering": metrics["anatomical_ordering_preserved"],
        "exact_part_geometry": (
            exact_part_masks is None
            or all(
                item["iou_regression"]
                <= thresholds["maximum_exact_part_iou_regression"]
                and item["absolute_passed"]
                and item["boundary_h95_increase_pixels"]
                <= thresholds["maximum_exact_part_boundary_h95_increase_pixels"]
                and item["centroid_increase_pixels"]
                <= thresholds["maximum_exact_part_centroid_increase_pixels"]
                for item in exact_part_metrics.values()
            )
        ),
        "non_noop_rgb_transfer": (
            inside_rms >= thresholds["minimum_inside_rgb_rms"]
            and changed_fraction >= thresholds["minimum_inside_changed_fraction"]
        ),
    }
    return {
        "schema_version": 1,
        "comparison": "candidate-versus-parent-detector-with-no-registration",
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "detection": {
            "parent": parent_detection,
            "candidate": candidate_detection,
            "parent_errors": parent_errors,
            "candidate_errors": candidate_errors,
            "parent_detector": parent_region.get("detector"),
            "candidate_detector": candidate_region.get("detector"),
        },
        "passed": bool(all(checks.values())),
    }


def _path_is_within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _prepare_pinned_diffsynth_import(provider_root: Path) -> None:
    loaded = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name == "diffsynth" or name.startswith("diffsynth.")
    }
    conflicts = {
        name: path
        for name, path in loaded.items()
        if path is None or not _path_is_within(path, provider_root)
    }
    if conflicts:
        raise RuntimeError(
            "A non-pinned DiffSynth module is already loaded: "
            + ", ".join(f"{name}={path}" for name, path in sorted(conflicts.items()))
        )
    retained = []
    for entry in sys.path:
        try:
            matches = Path(entry or ".").resolve() == provider_root
        except OSError:
            matches = False
        if not matches:
            retained.append(entry)
    sys.path[:] = [str(provider_root), *retained]


def _run_zimage(
    preflight: dict,
    parent_rgb: np.ndarray,
    control_rgb: np.ndarray,
    *,
    prompt: str,
    seed: int,
    denoising_strength: float,
    control_scale: float,
    steps: int,
) -> tuple[np.ndarray, dict]:
    import torch

    provider_root = Path(preflight["source"]["root"]).resolve()
    _prepare_pinned_diffsynth_import(provider_root)
    from diffsynth.pipelines.z_image import ControlNetInput, ModelConfig, ZImagePipeline
    import diffsynth.pipelines.z_image as zimage_module

    imported_files = {
        "pipeline": str(Path(zimage_module.__file__).resolve()),
        "pipeline_package": str(Path(sys.modules["diffsynth"].__file__).resolve()),
    }
    if not all(
        _path_is_within(path, provider_root) for path in imported_files.values()
    ):
        raise RuntimeError("DiffSynth resolved outside the pinned provider checkout")

    model_root = Path(preflight["model"]["root"])
    controlnet_path = Path(preflight["controlnet"]["path"])
    device = torch.device("cuda", torch.cuda.current_device())
    disk_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": str(device),
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    transformer_paths = [model_root / name for name in ZIMAGE_TRANSFORMER_SHARDS]
    text_encoder_paths = [model_root / name for name in ZIMAGE_TEXT_ENCODER_SHARDS]
    vae_path = model_root / "vae" / "diffusion_pytorch_model.safetensors"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=str(device),
        model_configs=[
            ModelConfig(path=str(controlnet_path), skip_download=True, **disk_config),
            ModelConfig(
                path=[str(path) for path in transformer_paths],
                skip_download=True,
                **disk_config,
            ),
            ModelConfig(
                path=[str(path) for path in text_encoder_paths],
                skip_download=True,
                **disk_config,
            ),
            ModelConfig(path=str(vae_path), skip_download=True, **disk_config),
        ],
        tokenizer_config=ModelConfig(
            path=str(model_root / "tokenizer"), skip_download=True
        ),
        vram_limit=min(
            10.5,
            torch.cuda.mem_get_info(device)[1] / (1024**3) - 0.5,
        ),
    )
    torch.cuda.synchronize(device)
    loaded_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    output = pipe(
        prompt=prompt,
        input_image=Image.fromarray(parent_rgb),
        denoising_strength=denoising_strength,
        height=int(parent_rgb.shape[0]),
        width=int(parent_rgb.shape[1]),
        seed=seed,
        rand_device="cpu",
        num_inference_steps=steps,
        controlnet_inputs=[
            ControlNetInput(image=Image.fromarray(control_rgb), scale=control_scale)
        ],
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    output_rgb = np.asarray(output.convert("RGB"), dtype=np.uint8)
    peak_allocated = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    peak_reserved = float(torch.cuda.max_memory_reserved(device) / (1024**3))
    telemetry = {
        "load_seconds": float(loaded_seconds),
        "inference_seconds": float(inference_seconds),
        "total_seconds": float(time.perf_counter() - started),
        "peak_vram_gib": peak_allocated,
        "peak_allocated_vram_gib": peak_allocated,
        "peak_reserved_vram_gib": peak_reserved,
        "device_index": int(device.index),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "device_wide_peak_available": False,
        "provider_import_files": imported_files,
        "disk_offload": True,
        "disk_offload_preparing_device": str(device),
        "disk_offload_reason": (
            "official CUDA-staged layer offload on the pinned torch 2.7.1 runtime; "
            "CPU staging exceeds the Windows paging-file mapping limit"
        ),
        "dtype": "bfloat16",
    }
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return output_rgb, telemetry


def _raw_depth_geometry_gate(
    parent_source_path: Path,
    candidate_source_path: Path,
    exact_camera_depth_path: Path,
    selection_mask_path: Path,
    exact_part_paths: dict[str, Path],
    evaluator_root: Path,
    output_dir: Path,
) -> dict:
    import torch

    from backend.benchmark.run_cc0_live_face_variation_matrix import (
        _exact_face_depth_quality,
    )
    from backend.pic_to_3d import process_image_get_depth_data

    device = torch.device("cuda", torch.cuda.current_device())
    evaluator_root = evaluator_root.resolve()
    device_argument = str(device.index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    parent_depth_path = Path(
        process_image_get_depth_data(
            parent_source_path,
            output_dir=output_dir / "parent_raw_depth",
            provider="depth-anything-v2",
            model_name=str(evaluator_root),
            device=device_argument,
        )
    )
    candidate_depth_path = Path(
        process_image_get_depth_data(
            candidate_source_path,
            output_dir=output_dir / "candidate_raw_depth",
            provider="depth-anything-v2",
            model_name=str(evaluator_root),
            device=device_argument,
        )
    )
    torch.cuda.synchronize(device)
    parent = _exact_face_depth_quality(
        parent_depth_path,
        exact_camera_depth_path,
        selection_mask_path,
        expected_scale_sign=-1.0,
        part_mask_paths=exact_part_paths,
        require_face_parts=True,
    )
    candidate = _exact_face_depth_quality(
        candidate_depth_path,
        exact_camera_depth_path,
        selection_mask_path,
        expected_scale_sign=-1.0,
        part_mask_paths=exact_part_paths,
        require_face_parts=True,
    )
    checks = {
        "parent_metrics_available": bool(parent.get("available", False)),
        "candidate_metrics_available": bool(candidate.get("available", False)),
        "candidate_absolute_gates": bool(
            candidate.get("checks", {}).get("passed", False)
        ),
        "candidate_shape_no_regression": float(
            candidate.get("shape_correlation", -math.inf)
        )
        >= float(parent.get("shape_correlation", math.inf)),
        "candidate_raw_gradient_no_regression": float(
            candidate.get("gradient_correlation", -math.inf)
        )
        >= float(parent.get("gradient_correlation", math.inf)),
        "candidate_rmse_no_regression": float(
            candidate.get("normalized_rmse", math.inf)
        )
        <= float(parent.get("normalized_rmse", -math.inf)),
        "candidate_six_part_shape": bool(
            candidate.get("named_part_shape", {}).get("passed", False)
        ),
        "candidate_six_part_affine_mm": bool(
            candidate.get("named_part_affine_mm", {}).get("passed", False)
        ),
    }
    return {
        "schema_version": 1,
        "estimator": {
            "id": DEPTH_EVALUATOR_ID,
            "revision": DEPTH_EVALUATOR_REVISION,
            "root": str(evaluator_root),
        },
        "device_index": int(device.index),
        "comparison": "unregistered-parent-versus-candidate-to-exact-camera-depth",
        "source_geometry_evaluation_only": True,
        "parent": parent,
        "candidate": candidate,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_allocated_vram_gib": float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
        "peak_reserved_vram_gib": float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        ),
        "artifacts": {
            "parent_raw_depth": str(parent_depth_path),
            "candidate_raw_depth": str(candidate_depth_path),
        },
    }


def _tree_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    exclude = set() if exclude is None else set(exclude)
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in exclude
    }


def _publish_derived_corpus(
    parent_root: Path,
    output_root: Path,
    parent_row: dict,
    candidate_rgb: np.ndarray,
    provenance: dict,
    alignment: dict,
    depth_geometry: dict,
) -> dict:
    if not alignment.get("passed", False) or not depth_geometry.get("passed", False):
        raise ValueError("Derived corpus publication requires both geometry gates")
    row_id = parent_row["row_id"]
    parent_row_dir = parent_root / "rows" / row_id
    target_root = output_root / "derived_corpus"
    with tempfile.TemporaryDirectory(dir=output_root, prefix="derived-corpus-") as temp:
        temp_root = Path(temp)
        target_row_dir = temp_root / "rows" / row_id
        shutil.copytree(parent_row_dir, target_row_dir)
        before = _tree_hashes(parent_row_dir, exclude={"source.png"})
        Image.fromarray(candidate_rgb).save(target_row_dir / "source.png")
        after = _tree_hashes(target_row_dir, exclude={"source.png"})
        if before != after:
            raise RuntimeError(
                "A non-RGB parent asset changed during derived-corpus copy"
            )
        derived_row = copy.deepcopy(parent_row)
        derived_row["parent_source"] = copy.deepcopy(parent_row["source"])
        derived_row["source"] = {
            "path": f"rows/{row_id}/source.png",
            "sha256": _sha256(target_row_dir / "source.png"),
        }
        derived_row["domain_transfer"] = {
            "provider": provenance["provider"],
            "configuration_sha256": provenance["configuration_sha256"],
            "alignment_passed": True,
            "raw_depth_geometry_passed": True,
            "raw_depth_geometry_sha256": _json_sha256(depth_geometry),
        }
        supervision_path = parent_root / "training_supervision.json"
        if (
            _sha256(supervision_path)
            != provenance["parent"]["training_supervision_sha256"]
        ):
            raise RuntimeError("Pinned MHR training supervision changed before publish")
        parent_supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
        supervision_rows = [
            row for row in parent_supervision["rows"] if row.get("row_id") == row_id
        ]
        if len(supervision_rows) != 1:
            raise RuntimeError("Selected MHR training supervision row is missing")
        supervision = {
            **parent_supervision,
            "row_count": 1,
            "rows": supervision_rows,
            "derived_from_sha256": _sha256(supervision_path),
        }
        (temp_root / "training_supervision.json").write_text(
            json.dumps(supervision, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "provider": provenance["provider"],
            "experimental_corpus": True,
            "privacy_safe_synthetic": True,
            "source_geometry_training_and_evaluation_only": True,
            "training_eligible": False,
            "promotion_eligible": False,
            "reason": "one-row alignment smoke only",
            "row_count": 1,
            "parent": provenance["parent"],
            "configuration_sha256": provenance["configuration_sha256"],
            "alignment": alignment,
            "raw_depth_geometry": depth_geometry,
            "raw_depth_geometry_sha256": _json_sha256(depth_geometry),
            "rows": [derived_row],
        }
        (temp_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target_root.exists():
            raise FileExistsError(f"Derived corpus already exists: {target_root}")
        shutil.move(str(temp_root), str(target_root))
    return {
        "path": str(target_root),
        "summary_sha256": _sha256(target_root / "summary.json"),
        "non_rgb_files_preserved": len(before),
    }


def run_smoke(
    *,
    parent_corpus_root: str | Path,
    expected_parent_summary_sha256: str,
    expected_training_supervision_sha256: str = DEFAULT_TRAINING_SUPERVISION_SHA256,
    provider_root: str | Path,
    model_root: str | Path,
    controlnet_path: str | Path,
    depth_evaluator_root: str | Path,
    output_dir: str | Path,
    row_id: str = DEFAULT_ROW_ID,
    prompt: str = DEFAULT_PROMPT,
    seed: int = DEFAULT_SEED,
    denoising_strength: float = DEFAULT_DENOISING_STRENGTH,
    control_scale: float = DEFAULT_CONTROL_SCALE,
    steps: int = DEFAULT_STEPS,
    preflight_only: bool = False,
) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if expected_training_supervision_sha256 != DEFAULT_TRAINING_SUPERVISION_SHA256:
        raise ValueError(
            "Training supervision hash must equal the adapter's immutable corpus pin"
        )
    preflight = zimage_preflight(provider_root, model_root, controlnet_path)
    evaluator_preflight = depth_evaluator_preflight(depth_evaluator_root)
    preflight["depth_evaluator"] = evaluator_preflight
    preflight["checks"].update(
        {
            f"depth_evaluator_{name}": passed
            for name, passed in evaluator_preflight["checks"].items()
        }
    )
    preflight["runnable"] = bool(all(preflight["checks"].values()))
    (output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if preflight_only:
        return preflight
    if not preflight["runnable"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError("Z-Image preflight failed: " + ", ".join(failed))
    parent_root = Path(parent_corpus_root).resolve()
    row, paths, parent_provenance = load_parent_row(
        parent_root,
        expected_parent_summary_sha256,
        row_id,
        expected_training_supervision_sha256,
    )
    parent_rgb = np.asarray(Image.open(paths["source"]).convert("RGB"), dtype=np.uint8)
    selection_mask = np.asarray(Image.open(paths["selection_mask"])) > 0
    exact_depth = np.load(paths["exact_depth"], allow_pickle=False)
    control_rgb = make_inverse_depth_control(exact_depth)
    Image.fromarray(control_rgb).save(output_dir / "depth_control.png")
    configuration = {
        "prompt": prompt,
        "seed": int(seed),
        "denoising_strength": float(denoising_strength),
        "control_scale": float(control_scale),
        "steps": int(steps),
        "row_id": row_id,
        "output_dimensions": list(parent_rgb.shape[:2]),
        "resize_or_warp": None,
        "depth_control": "uint8_rgb_round(255*(1-exact_normalized_scene_depth))",
    }
    generated_rgb, runtime = _run_zimage(
        preflight,
        parent_rgb,
        control_rgb,
        prompt=prompt,
        seed=seed,
        denoising_strength=denoising_strength,
        control_scale=control_scale,
        steps=steps,
    )
    Image.fromarray(generated_rgb).save(output_dir / "raw_generated.png")
    candidate_rgb = composite_face_only(parent_rgb, generated_rgb, selection_mask)
    candidate_path = output_dir / "candidate_face_only.png"
    Image.fromarray(candidate_rgb).save(candidate_path)
    exact_part_masks = {
        name: np.asarray(Image.open(paths[f"part:{name}"]).convert("L")) > 0
        for name in FACE_PART_NAMES
    }
    alignment = measure_rgb_alignment(
        parent_rgb,
        candidate_rgb,
        selection_mask,
        exact_part_masks=exact_part_masks,
    )
    depth_geometry = (
        _raw_depth_geometry_gate(
            paths["source"],
            candidate_path,
            paths["exact_camera_depth"],
            paths["selection_mask"],
            {name: paths[f"part:{name}"] for name in FACE_PART_NAMES},
            Path(depth_evaluator_root),
            output_dir,
        )
        if alignment["passed"]
        else {
            "schema_version": 1,
            "available": False,
            "reason": "rgb_alignment_gate_failed",
            "passed": False,
        }
    )
    provenance = {
        "provider": preflight["provider"],
        "parent": parent_provenance,
        "configuration": configuration,
        "configuration_sha256": _json_sha256(configuration),
        "preflight_sha256": _sha256(output_dir / "preflight.json"),
        "control_image_sha256": _sha256(output_dir / "depth_control.png"),
        "raw_generated_sha256": _sha256(output_dir / "raw_generated.png"),
        "candidate_sha256": _sha256(output_dir / "candidate_face_only.png"),
    }
    derived = None
    if alignment["passed"] and depth_geometry["passed"]:
        derived = _publish_derived_corpus(
            parent_root,
            output_dir,
            row,
            candidate_rgb,
            provenance,
            alignment,
            depth_geometry,
        )
    result = {
        "schema_version": 1,
        "status": (
            "pass" if alignment["passed"] and depth_geometry["passed"] else "hold"
        ),
        "expansion_authorized": False,
        "reason": (
            "one-row gate passed; three-seed repeat required"
            if alignment["passed"] and depth_geometry["passed"]
            else (
                "unregistered RGB alignment gate failed"
                if not alignment["passed"]
                else "exact raw-depth geometry gate failed"
            )
        ),
        "provenance": provenance,
        "runtime": runtime,
        "alignment": alignment,
        "raw_depth_geometry": depth_geometry,
        "derived_corpus": derived,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-corpus-root", required=True)
    parser.add_argument("--expected-parent-summary-sha256", required=True)
    parser.add_argument(
        "--expected-training-supervision-sha256",
        default=DEFAULT_TRAINING_SUPERVISION_SHA256,
    )
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--controlnet-path", required=True)
    parser.add_argument("--depth-evaluator-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-id", default=DEFAULT_ROW_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--denoising-strength", type=float, default=DEFAULT_DENOISING_STRENGTH
    )
    parser.add_argument("--control-scale", type=float, default=DEFAULT_CONTROL_SCALE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run_smoke(
        parent_corpus_root=args.parent_corpus_root,
        expected_parent_summary_sha256=args.expected_parent_summary_sha256,
        expected_training_supervision_sha256=(
            args.expected_training_supervision_sha256
        ),
        provider_root=args.provider_root,
        model_root=args.model_root,
        controlnet_path=args.controlnet_path,
        depth_evaluator_root=args.depth_evaluator_root,
        output_dir=args.output_dir,
        row_id=args.row_id,
        prompt=args.prompt,
        seed=args.seed,
        denoising_strength=args.denoising_strength,
        control_scale=args.control_scale,
        steps=args.steps,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
