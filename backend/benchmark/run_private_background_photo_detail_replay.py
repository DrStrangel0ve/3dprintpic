"""Replay private relief scenes without persisting private evidence in git.

The scene config and all generated artifacts must live below a gitignored output
directory.  The optional aggregate summary contains metrics and scene labels only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import zoom

from backend.benchmark.run_background_photo_detail_provider_smoke import (
    _detail_telemetry_checks,
)
from backend.benchmark.run_background_photo_detail_sweep import _detail_metrics
from backend.benchmark.run_relief_scene_regression import _git_provenance, _mesh_topology
from backend.benchmark.run_relief_visual_sweep import _stl_heightfield_agreement
from backend.pic_to_3d import (
    BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


DETAIL_LEVELS_MM = (0.0, 0.60)
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/run_background_photo_detail_provider_smoke.py",
    "backend/benchmark/run_private_background_photo_detail_replay.py",
)
_SAFE_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PRIVATE_PATH_FIELDS = (
    "source_image",
    "depth_npy",
    "selection_mask",
    "selection_metadata",
    "request_metadata",
    "face_region_mask",
    "feature_weight_mask",
    "face_metadata",
)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    target = (int(shape[0]), int(shape[1]))
    if binary.shape == target:
        return binary
    factors = (target[0] / binary.shape[0], target[1] / binary.shape[1])
    resized = zoom(binary.astype(np.uint8), factors, order=0) > 0
    resized = resized[: target[0], : target[1]]
    padding = (target[0] - resized.shape[0], target[1] - resized.shape[1])
    if padding[0] > 0 or padding[1] > 0:
        resized = np.pad(
            resized,
            ((0, max(0, padding[0])), (0, max(0, padding[1]))),
            mode="edge",
        )
    return resized


def _transform_mask_to_surface_grid(mask: np.ndarray, transform: dict) -> np.ndarray:
    required = {
        "input_depth_shape",
        "target_depth_shape",
        "mesh_shape_before_crop",
        "crop_bbox_rc",
        "emitted_shape",
    }
    missing = sorted(required - set(transform))
    if missing:
        raise ValueError("Surface-grid transform is missing: " + ", ".join(missing))
    input_shape = tuple(int(value) for value in transform["input_depth_shape"])
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != input_shape:
        binary = _resize_mask(binary, input_shape)
    binary = _resize_mask(binary, tuple(transform["target_depth_shape"]))
    if bool(transform.get("flip_x", False)):
        binary = np.flip(binary, axis=1)
    binary = _resize_mask(binary, tuple(transform["mesh_shape_before_crop"]))
    top, left, bottom, right = (
        int(value) for value in transform["crop_bbox_rc"]
    )
    binary = binary[top:bottom, left:right]
    emitted_shape = tuple(int(value) for value in transform["emitted_shape"])
    if binary.shape != emitted_shape:
        raise ValueError(
            f"Transformed mask has shape {binary.shape}, expected {emitted_shape}"
        )
    return binary


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_source_fingerprint(path: Path) -> str:
    with Image.open(path) as image:
        source = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{source.width}x{source.height}:".encode("ascii"))
    digest.update(source.tobytes())
    return digest.hexdigest()


def _require_ignored(path: Path, repository: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError("Private replay output must be inside the repository") from exc
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=repository,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Private replay output must be covered by .gitignore")


def _verify_private_checksums(
    label: str, paths: dict[str, Path], expected_hashes: object
) -> None:
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(paths):
        raise ValueError(f"Scene {label!r} must checksum-pin every private input path")
    for key, path in paths.items():
        expected = expected_hashes[key]
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError(f"Scene {label!r} has an invalid checksum for {key}")
        if _sha256(path) != expected:
            raise ValueError(f"Scene {label!r} checksum mismatch for {key}")


def _validate_scene(scene: dict, repository: Path) -> dict:
    allowed = {
        "label",
        "source_image",
        "depth_npy",
        "selection_mask",
        "selection_metadata",
        "request_metadata",
        "expected_depth_model",
        "input_sha256",
        "face_region_mask",
        "feature_weight_mask",
        "face_metadata",
        "expected_face_count",
    }
    unexpected = sorted(set(scene) - allowed)
    if unexpected:
        raise ValueError("Unexpected scene fields: " + ", ".join(unexpected))
    label = scene.get("label")
    if (
        not isinstance(label, str)
        or not _SAFE_LABEL.fullmatch(label)
        or not re.fullmatch(r"scene-[0-9]{2}", label)
    ):
        raise ValueError("Scene label must use the non-semantic scene-NN form")
    for key in (
        "source_image",
        "depth_npy",
        "selection_mask",
        "selection_metadata",
        "request_metadata",
    ):
        if not isinstance(scene.get(key), str) or not Path(scene[key]).is_file():
            raise ValueError(f"Scene {label!r} has no readable {key}")
    for key in ("face_region_mask", "feature_weight_mask", "face_metadata"):
        value = scene.get(key)
        if value is not None and (not isinstance(value, str) or not Path(value).is_file()):
            raise ValueError(f"Scene {label!r} has no readable {key}")
    present_paths = {
        key: Path(scene[key]) for key in _PRIVATE_PATH_FIELDS if scene.get(key)
    }
    for path in present_paths.values():
        _require_ignored(path, repository)
    _verify_private_checksums(label, present_paths, scene.get("input_sha256"))

    source_path = present_paths["source_image"]
    selection_path = present_paths["selection_mask"]
    selection_metadata = json.loads(
        present_paths["selection_metadata"].read_text(encoding="utf-8")
    )
    request_metadata = json.loads(
        present_paths["request_metadata"].read_text(encoding="utf-8")
    )
    source_fingerprint = _selection_source_fingerprint(source_path)
    request_context = request_metadata.get("selection_depth_context", {})
    selection_job_id = selection_metadata.get("job_id")
    recorded_mask = str(request_context.get("selection_mask", "")).replace("\\", "/")
    expected_mask_suffix = f"selection/{selection_job_id}/selection_mask.png"
    if (
        selection_metadata.get("model_status") != "composed-clicked-masks"
        or selection_metadata.get("source_fingerprint") != source_fingerprint
        or request_context.get("source_fingerprint") != source_fingerprint
        or not isinstance(selection_job_id, str)
        or not re.fullmatch(r"[a-f0-9]{32}", selection_job_id)
        or request_context.get("selection_job_id") != selection_job_id
        or recorded_mask != expected_mask_suffix
    ):
        raise ValueError(f"Scene {label!r} selection provenance does not match")
    expected_depth_model = scene.get("expected_depth_model")
    if (
        not isinstance(expected_depth_model, str)
        or request_metadata.get("depth_model") != expected_depth_model
        or request_metadata.get("depth_metadata", {}).get("effective_model")
        != expected_depth_model
    ):
        raise ValueError(f"Scene {label!r} depth-model provenance does not match")
    historical_controls = {
        "target_dimension": 512,
        "sigma": 0.35,
        "z_scale": 30.0,
        "max_xy_size": 128.0,
        "relief_gamma": 0.75,
        "detail_boost": 0.8,
        "background_detail_boost": 2.4,
        "selection_background_depth_ratio": 0.45,
        "background_photo_detail_mm": 0.12,
    }
    if any(
        request_metadata.get(key) is None
        or abs(float(request_metadata[key]) - expected) > 1e-9
        for key, expected in historical_controls.items()
    ):
        raise ValueError(f"Scene {label!r} historical request controls do not match")
    with Image.open(source_path) as source_image, Image.open(selection_path) as mask_image:
        source_shape = (source_image.height, source_image.width)
        if source_image.size != mask_image.size:
            raise ValueError(f"Scene {label!r} source and selection mask do not align")
    expected_face_count = scene.get("expected_face_count")
    face_metadata = scene.get("face_metadata")
    has_face_region = scene.get("face_region_mask") is not None
    if not (
        (expected_face_count is None and face_metadata is None and not has_face_region)
        or (expected_face_count is not None and face_metadata is not None and has_face_region)
    ):
        raise ValueError(
            f"Scene {label!r} must provide face_region_mask, face_metadata, and "
            "expected_face_count together"
        )
    if expected_face_count is not None:
        if not isinstance(expected_face_count, int) or expected_face_count < 1:
            raise ValueError(f"Scene {label!r} has an invalid expected_face_count")
        if not isinstance(face_metadata, str) or not Path(face_metadata).is_file():
            raise ValueError(f"Scene {label!r} has no readable face_metadata")
        metadata = json.loads(Path(face_metadata).read_text(encoding="utf-8"))
        face_mask = _read_mask(Path(scene["face_region_mask"]))
        faces = metadata.get("faces", [])
        if (
            face_mask.shape != source_shape
            or not np.any(face_mask)
            or int(metadata.get("detected_faces", -1)) != expected_face_count
            or int(metadata.get("refined_faces", -1)) != expected_face_count
            or len(faces) != expected_face_count
        ):
            raise ValueError(f"Scene {label!r} face metadata contract does not match")
        for face in faces:
            bbox = face.get("bbox")
            if face.get("status") != "refined" or not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Scene {label!r} has incomplete per-face metadata")
            left, top, right, bottom = (int(value) for value in bbox)
            center_col = int(round((left + right) / 2.0))
            center_row = int(round((top + bottom) / 2.0))
            if not (
                0 <= center_row < face_mask.shape[0]
                and 0 <= center_col < face_mask.shape[1]
                and bool(face_mask[center_row, center_col])
                and np.count_nonzero(face_mask[top:bottom, left:right]) > 0
            ):
                raise ValueError(f"Scene {label!r} face mask misses a refined face")
    return scene


def _compact_shell(shell: dict) -> dict:
    return {
        key: shell.get(key)
        for key in (
            "passed",
            "complete_shell_verified",
            "facet_geometry_passed",
            "coverage_ratio",
            "max_abs_error_mm",
            "rms_error_mm",
            "expected_shell_triangle_count",
            "actual_shell_triangle_count",
            "invalid_shell_triangle_count",
        )
    }


def _finite(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _compact_cap(cap: dict) -> dict:
    return {
        "strict_passed": bool(cap.get("passed", False)),
        "emission_passed": bool(cap.get("emission_passed", False)),
        "far_background_cap_passed": bool(cap.get("far_background_cap_passed", False)),
        "feasible_attachment_constraints_passed": bool(
            cap.get("feasible_attachment_constraints_passed", False)
        ),
        "far_background_max_mm": _finite(cap.get("far_background_max_mm")),
        "far_background_ceiling_mm": _finite(cap.get("far_background_ceiling_mm")),
        "feasible_attachment_jump_max_mm": _finite(
            cap.get("feasible_attachment_jump_max_mm")
        ),
        "attachment_constraint_conflicts": int(
            cap.get("attachment_constraint_conflicts", 0)
        ),
        "far_background_cap_violation_mm": _finite(
            cap.get("far_background_cap_violation_mm")
        ),
    }


def _compact_background_preservation(stats: dict) -> dict:
    return {
        "passed": bool(stats.get("passed", False)),
        "correlation": _finite(stats.get("correlation")),
        "rms_retention": _finite(stats.get("rms_retention")),
        "gradient_correlation": _finite(stats.get("gradient_correlation")),
        "candidate_coverage_ratio": _finite(stats.get("candidate_coverage_ratio")),
    }


def _face_delta_metrics(
    baseline: np.ndarray, candidate: np.ndarray, face_mask: np.ndarray | None
) -> dict:
    if face_mask is None or not np.any(face_mask):
        return {"available": False, "p99_change_mm": None, "max_change_mm": None}
    delta = np.abs(np.asarray(candidate) - np.asarray(baseline))[face_mask]
    return {
        "available": True,
        "p99_change_mm": float(np.percentile(delta, 99.0)),
        "max_change_mm": float(np.max(delta)),
    }


def _assert_aggregate_privacy(summary: dict) -> None:
    serialized = json.dumps(summary, sort_keys=True)
    lowered = serialized.lower()
    forbidden = (
        "source.png",
        "selection_mask.png",
        "backend/output",
        "\\users\\",
        "/users/",
        ".npy",
        ".stl",
        ".png",
        ".jpg",
        ".jpeg",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError("Aggregate summary contains private path material")
    if re.search(r"\b[a-f0-9]{64}\b", lowered):
        raise ValueError("Aggregate summary must not contain content hashes")
    if re.search(r"\b[a-f0-9]{32}\b", lowered):
        raise ValueError("Aggregate summary must not contain private job identifiers")


def run(
    scene_config: str | Path,
    output_dir: str | Path,
    *,
    aggregate_summary: str | Path | None = None,
    relief_height_mm: float = 30.0,
    physical_size_mm: float = 128.0,
    background_depth_ratio: float = 0.65,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    repository = Path(__file__).resolve().parents[2]
    if (
        abs(float(relief_height_mm) - 30.0) > 1e-9
        or abs(float(physical_size_mm) - 128.0) > 1e-9
        or abs(float(background_depth_ratio) - 0.65) > 1e-9
    ):
        raise ValueError(
            "Retained-artifact replay is fixed to 30 mm, 128 mm, and ratio 0.65"
        )
    output_dir = Path(output_dir)
    _require_ignored(output_dir, repository)
    _require_ignored(Path(scene_config), repository)
    config = json.loads(Path(scene_config).read_text(encoding="utf-8"))
    if set(config) != {"scenes"} or not isinstance(config["scenes"], list):
        raise ValueError("Private replay config must contain only a scenes list")
    scenes = [_validate_scene(scene, repository) for scene in config["scenes"]]
    if not scenes or len({scene["label"] for scene in scenes}) != len(scenes):
        raise ValueError("Private replay requires uniquely labeled scenes")

    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _git_provenance(PROVENANCE_PATHS)
    records = []
    for scene in scenes:
        started = time.perf_counter()
        label = scene["label"]
        scene_dir = output_dir / label
        scene_dir.mkdir(parents=True, exist_ok=True)
        depth = np.load(scene["depth_npy"]).astype(np.float32)
        if depth.ndim != 2 or min(depth.shape) < 2:
            raise ValueError(f"Scene {label!r} depth must be a non-empty 2D array")
        selection_mask = _read_mask(Path(scene["selection_mask"]))
        face_contract = {"required": False, "passed": True}
        if scene.get("face_metadata"):
            metadata = json.loads(
                Path(scene["face_metadata"]).read_text(encoding="utf-8")
            )
            expected_faces = int(scene["expected_face_count"])
            face_contract = {
                "required": True,
                "passed": bool(
                    int(metadata.get("detected_faces", -1)) == expected_faces
                    and int(metadata.get("refined_faces", -1)) == expected_faces
                    and len(metadata.get("faces", [])) == expected_faces
                ),
            }
        sample_pitch_mm = float(physical_size_mm) / max(max(depth.shape) - 1, 1)
        composed, compose_stats = compose_selection_depth_with_context(
            depth,
            selection_mask,
            value_transform="linear",
            relief_height_mm=float(relief_height_mm),
            sample_pitch_mm=sample_pitch_mm,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=float(background_depth_ratio),
            background_feather_mm=1.5,
            background_smoothing_mm=0.6,
        )
        composed_path = scene_dir / "composed_depth.npy"
        np.save(composed_path, composed.astype(np.float32, copy=False))
        emitted = []
        for detail_mm in DETAIL_LEVELS_MM:
            row_dir = scene_dir / f"detail_{detail_mm:.2f}".replace(".", "p")
            row_dir.mkdir(parents=True, exist_ok=True)
            stl_path = row_dir / "relief.stl"
            surface_path = row_dir / "surface.npy"
            reference_path = row_dir / "reference_surface.npy"
            postprocess = depth_data_to_3d_model(
                composed_path,
                output_stl_path=str(stl_path),
                target_dimension=512,
                z_scale=float(relief_height_mm),
                invert=False,
                sigma=0.35,
                max_xy_size=float(physical_size_mm),
                relief_gamma=0.75,
                detail_boost=0.8,
                background_detail_boost=2.4,
                low_percentile=1.0,
                high_percentile=99.0,
                base_border_px=2,
                value_transform="linear",
                minimum_feature_mm=0.8,
                max_relief_slope=2.0,
                face_region_mask=scene.get("face_region_mask"),
                selection_region_mask=scene["selection_mask"],
                selection_background_depth_ratio=float(background_depth_ratio),
                source_image=scene["source_image"],
                background_photo_detail_mm=float(detail_mm),
                trim_top_background=False,
                feature_weight_mask=scene.get("feature_weight_mask"),
                printable_feature_depth_mm=0.8,
                feature_bridge_depth_mm=0.8,
                surface_output_path=surface_path,
                reference_surface_output_path=reference_path,
            )
            (row_dir / "postprocess.json").write_text(
                json.dumps(postprocess, indent=2), encoding="utf-8"
            )
            surface = np.load(surface_path)
            transform = postprocess["surface_grid_transform"]
            transformed_selection = _transform_mask_to_surface_grid(
                selection_mask, transform
            )
            face_mask = None
            if scene.get("face_region_mask"):
                face_mask = _transform_mask_to_surface_grid(
                    _read_mask(Path(scene["face_region_mask"])), transform
                )
            topology = _mesh_topology(trimesh.load_mesh(stl_path, process=True))
            shell = _stl_heightfield_agreement(
                stl_path,
                surface_path,
                expected_max_xy_size_mm=float(physical_size_mm),
            )
            emitted.append(
                {
                    "composed_path": composed_path.resolve(),
                    "detail_mm": detail_mm,
                    "surface": surface,
                    "selection_mask": transformed_selection,
                    "face_mask": face_mask,
                    "postprocess": postprocess,
                    "topology": topology,
                    "shell": shell,
                }
            )

        baseline, candidate = emitted
        transform_match = (
            baseline["postprocess"]["surface_grid_transform"]
            == candidate["postprocess"]["surface_grid_transform"]
        )
        selection_match = np.array_equal(
            baseline["selection_mask"], candidate["selection_mask"]
        )
        same_composed_depth = (
            baseline["composed_path"] == candidate["composed_path"]
        )
        face_mask_contract = True
        if baseline["face_mask"] is not None:
            face_pixels = int(np.count_nonzero(baseline["face_mask"]))
            overlap = int(
                np.count_nonzero(
                    baseline["face_mask"] & baseline["selection_mask"]
                )
            )
            face_mask_contract = bool(
                face_pixels > 0 and overlap / max(face_pixels, 1) >= 0.95
            )
        protection_mask = baseline["selection_mask"].copy()
        if baseline["face_mask"] is not None:
            protection_mask |= baseline["face_mask"]
        pitch_mm = float(candidate["postprocess"]["mesh_sample_pitch_mm"])
        metrics = _detail_metrics(
            baseline["surface"],
            candidate["surface"],
            protection_mask,
            Path(scene["source_image"]),
            candidate["detail_mm"],
            protection_halo_px=(
                BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM / max(pitch_mm, 1e-6)
            ),
            surface_grid_transform=baseline["postprocess"][
                "surface_grid_transform"
            ],
        )
        face_delta = _face_delta_metrics(
            baseline["surface"], candidate["surface"], baseline["face_mask"]
        )
        baseline_telemetry = _detail_telemetry_checks(
            baseline["postprocess"]["background_photo_detail"], 0.0
        )
        candidate_telemetry = _detail_telemetry_checks(
            candidate["postprocess"]["background_photo_detail"], 0.60
        )
        row_checks = {
            "same_composed_depth": bool(same_composed_depth),
            "retained_artifact_bundle_verified": True,
            "surface_grid_match": bool(transform_match),
            "selection_mask_match": bool(selection_match),
            "detail_metrics": bool(metrics["checks"]["passed"]),
            "baseline_telemetry": bool(baseline_telemetry["passed"]),
            "candidate_telemetry": bool(candidate_telemetry["passed"]),
            "baseline_physical_cap": bool(
                baseline["postprocess"]["selection_background_physical_cap"].get(
                    "emission_passed", False
                )
            ),
            "candidate_physical_cap": bool(
                candidate["postprocess"]["selection_background_physical_cap"].get(
                    "emission_passed", False
                )
            ),
            "baseline_printable": bool(baseline["topology"]["printable"]),
            "candidate_printable": bool(candidate["topology"]["printable"]),
            "baseline_shell": bool(baseline["shell"]["passed"]),
            "candidate_shell": bool(candidate["shell"]["passed"]),
            "face_delta": bool(
                not face_delta["available"]
                or (
                    face_delta["p99_change_mm"] <= 0.01
                    and face_delta["max_change_mm"] <= 0.05
                )
            ),
            "face_metadata_contract": bool(face_contract["passed"]),
            "face_mask_contract": bool(face_mask_contract),
            "baseline_background_preservation": bool(
                baseline["postprocess"]["background_depth_preservation"].get(
                    "passed", False
                )
            ),
            "candidate_background_preservation": bool(
                candidate["postprocess"]["background_depth_preservation"].get(
                    "passed", False
                )
            ),
        }
        records.append(
            {
                "scene_label": label,
                "runtime_seconds": float(time.perf_counter() - started),
                "compose": {
                    "background_context_enabled": bool(
                        compose_stats.get("background_context_enabled", False)
                    ),
                    "background_output_span_ratio": _finite(
                        compose_stats.get("background_output_span_ratio")
                    ),
                },
                "detail_metrics": metrics,
                "face_delta": face_delta,
                "telemetry_checks": {
                    "baseline": baseline_telemetry,
                    "candidate": candidate_telemetry,
                },
                "baseline": {
                    "topology": baseline["topology"],
                    "shell": _compact_shell(baseline["shell"]),
                    "physical_cap": _compact_cap(
                        baseline["postprocess"]["selection_background_physical_cap"]
                    ),
                    "background_preservation": _compact_background_preservation(
                        baseline["postprocess"]["background_depth_preservation"]
                    ),
                },
                "candidate": {
                    "topology": candidate["topology"],
                    "shell": _compact_shell(candidate["shell"]),
                    "physical_cap": _compact_cap(
                        candidate["postprocess"]["selection_background_physical_cap"]
                    ),
                    "background_preservation": _compact_background_preservation(
                        candidate["postprocess"]["background_depth_preservation"]
                    ),
                },
                "checks": {**row_checks, "passed": bool(all(row_checks.values()))},
            }
        )

    checks = {
        "expected_rows": len(records) == len(scenes),
        "all_scenes_passed": bool(records)
        and all(record["checks"]["passed"] for record in records),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "retained_private_artifact_background_photo_detail_replay",
        "privacy": (
            "aggregate metrics only; source images, masks, paths, hashes, surfaces, "
            "and meshes remain in gitignored output"
        ),
        "implementation_provenance": provenance,
        "matrix": {
            "scene_labels": [scene["label"] for scene in scenes],
            "protection_mask_semantics": (
                "the exact production selection-plus-face union is excluded from "
                "background scoring; an available face region is gated separately"
            ),
            "detail_levels_mm": list(DETAIL_LEVELS_MM),
            "relief_height_mm": float(relief_height_mm),
            "physical_size_mm": float(physical_size_mm),
            "background_depth_ratio": float(background_depth_ratio),
            "replay_semantics": (
                "locally pinned retained artifacts with current detail/context "
                "overrides; not an authenticated historical-request reproduction"
            ),
            "historical_controls_verified": {
                "target_dimension": 512,
                "sigma": 0.35,
                "relief_height_mm": 30.0,
                "physical_size_mm": 128.0,
                "relief_gamma": 0.75,
                "detail_boost": 0.8,
                "background_detail_boost": 2.4,
            },
            "intentional_overrides": {
                "selection_background_depth_ratio": {
                    "historical": 0.45,
                    "replay": 0.65,
                },
                "background_photo_detail_mm": {
                    "historical": 0.12,
                    "replay_pair": list(DETAIL_LEVELS_MM),
                },
            },
        },
        "records": records,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["all_scenes_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    _assert_aggregate_privacy(summary)
    private_summary = output_dir / "aggregate_summary.json"
    private_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if aggregate_summary is not None:
        aggregate_path = Path(aggregate_summary)
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Private background photo-detail replay failed its gates")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aggregate-summary")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.scene_config,
        args.output_dir,
        aggregate_summary=args.aggregate_summary,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )
    print(json.dumps(summary["checks"], indent=2))
    return 0 if summary["checks"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
