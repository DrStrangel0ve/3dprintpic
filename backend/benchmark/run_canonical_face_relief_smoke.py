"""Render a known canonical face and measure absolute relief fidelity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from backend.benchmark.face_part_metrics import (
    FACE_PART_AFFINE_MM_GATES,
    FACE_PART_GATES,
    FACE_PART_METRIC_SCHEMA_VERSION,
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
    load_persisted_face_part_masks,
)
from backend.benchmark.mesh_rendering import (
    CameraSpec,
    RenderConfig,
    render_mesh,
)
from backend.benchmark.run_relief_scene_regression import (
    _git_provenance,
    _mesh_topology,
    _scene_checks,
)
from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    _appearance_checks,
    _stl_heightfield_agreement,
)
from backend.face_depth_refinement import (
    FACE_PART_INDEX_GROUPS,
    FACE_PART_NAMES,
)
from backend.pic_to_3d import (
    _surface_lighting_agreement_metrics,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "mediapipe_canonical_face"
CANONICAL_FACE_PATH = ASSET_DIR / "canonical_face_model.obj"
CANONICAL_FACE_LICENSE_PATH = ASSET_DIR / "LICENSE"
CANONICAL_FACE_SOURCE_COMMIT = "a908d668c730da128dfa8d9f6bd25d519d006692"
CANONICAL_FACE_SHA256 = "8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618"
CANONICAL_FACE_LICENSE_SHA256 = (
    "8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8"
)
CANONICAL_FACE_SOURCE_URL = (
    "https://github.com/google-ai-edge/mediapipe/blob/"
    f"{CANONICAL_FACE_SOURCE_COMMIT}/mediapipe/modules/face_geometry/data/"
    "canonical_face_model.obj"
)
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/face_depth_refinement.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
    "backend/benchmark/run_canonical_face_relief_smoke.py",
    "backend/benchmark/assets/mediapipe_canonical_face",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_canonical_face_fixture() -> tuple[trimesh.Trimesh, dict]:
    """Load the ordered 468-landmark fixture after strict provenance checks."""
    asset_sha256 = _sha256(CANONICAL_FACE_PATH)
    license_sha256 = _sha256(CANONICAL_FACE_LICENSE_PATH)
    if asset_sha256 != CANONICAL_FACE_SHA256:
        raise ValueError(
            f"Canonical face checksum mismatch: {asset_sha256} != {CANONICAL_FACE_SHA256}"
        )
    if license_sha256 != CANONICAL_FACE_LICENSE_SHA256:
        raise ValueError(
            "Canonical face license checksum mismatch: "
            f"{license_sha256} != {CANONICAL_FACE_LICENSE_SHA256}"
        )
    loaded = trimesh.load(
        CANONICAL_FACE_PATH,
        process=False,
        maintain_order=True,
    )
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Canonical face fixture is not a mesh: {type(loaded)!r}")
    mesh = loaded
    if len(mesh.vertices) != 468 or len(mesh.faces) != 898:
        raise ValueError(
            "Canonical face topology mismatch: "
            f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}"
        )
    if int(np.min(mesh.faces)) != 0 or int(np.max(mesh.faces)) != 467:
        raise ValueError("Canonical face fixture does not use the 468 ordered vertices")
    components = len(mesh.split(only_watertight=False))
    if components != 1:
        raise ValueError(f"Canonical face fixture has {components} components")
    metadata = {
        "source_url": CANONICAL_FACE_SOURCE_URL,
        "source_commit": CANONICAL_FACE_SOURCE_COMMIT,
        "asset_sha256": asset_sha256,
        "license": "Apache-2.0",
        "license_sha256": license_sha256,
        "coordinate_unit": "centimeter",
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "component_count": components,
        "watertight": bool(mesh.is_watertight),
        "bounds_cm": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extents_cm": np.asarray(mesh.extents, dtype=np.float64).tolist(),
    }
    return mesh, metadata


def canonical_vertex_part_weights(vertex_count: int = 468) -> dict[str, np.ndarray]:
    weights = {}
    for name in FACE_PART_NAMES:
        values = np.zeros(int(vertex_count), dtype=np.float32)
        indices = np.asarray(FACE_PART_INDEX_GROUPS[name], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= int(vertex_count)):
            raise ValueError(f"Canonical part {name!r} has out-of-range indices")
        values[indices] = 1.0
        weights[name] = values
    return weights


def canonical_face_masks(
    silhouette: np.ndarray,
    rendered_part_masks: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    face_mask = np.asarray(silhouette, dtype=bool)
    raw_parts = rendered_part_masks or {}
    parts = {
        name: (np.asarray(raw_parts[name]) > 0) & face_mask
        for name in FACE_PART_NAMES
        if name in raw_parts
    }
    if len(parts) != len(FACE_PART_NAMES) or not all(
        np.any(parts.get(name, False)) for name in FACE_PART_NAMES
    ):
        missing = [name for name in FACE_PART_NAMES if not np.any(parts.get(name, False))]
        raise ValueError(f"Canonical render has empty facial parts: {missing}")
    return face_mask, parts


def make_structured_face_scene(
    rendered_depth: np.ndarray,
    face_mask: np.ndarray,
    yaw_deg: float,
    scene_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Place known face depth in front of a deterministic structured background."""
    depth = np.asarray(rendered_depth, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool)
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    x = 2.0 * cols / max(depth.shape[1] - 1, 1) - 1.0
    y = 2.0 * rows / max(depth.shape[0] - 1, 1) - 1.0
    phase = float(scene_phase_rad)
    background = 0.76 + 0.055 * x + 0.035 * y
    background += 0.035 * np.sin(
        2.6 * np.pi * x + np.deg2rad(yaw_deg) + phase
    )
    background += 0.028 * np.cos(2.1 * np.pi * y - 0.4 * x - 0.7 * phase)
    blob_x = -0.52 + 0.14 * np.sin(phase)
    blob_y = 0.22 + 0.12 * np.cos(phase) - 0.12
    background += 0.045 * np.exp(-((x - blob_x) ** 2 + (y - blob_y) ** 2) / 0.08)
    background = np.clip(background, 0.64, 0.92)

    face_depth = 0.08 + 0.50 * depth
    scene = np.where(face, face_depth, background).astype(np.float32)
    background_rgb = np.stack(
        (
            0.34 + 0.30 * (1.0 - background),
            0.42 + 0.25 * x + 0.12 * (1.0 - background),
            0.50 + 0.20 * y + 0.10 * (1.0 - background),
        ),
        axis=-1,
    )
    return scene, np.clip(background_rgb, 0.0, 1.0).astype(np.float32)


def _persist_face_masks(
    output_dir: Path,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
) -> Path:
    part_dir = output_dir / "face_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    face_path = part_dir / "face.png"
    Image.fromarray(np.asarray(face_mask, dtype=np.uint8) * 255).save(face_path)
    files = {}
    for name in FACE_PART_NAMES:
        path = part_dir / f"{name}.png"
        Image.fromarray(np.asarray(part_masks[name], dtype=np.uint8) * 255).save(path)
        files[name] = path.relative_to(output_dir).as_posix()
    metadata = {
        "part_mask_schema_version": 1,
        "part_names": list(FACE_PART_NAMES),
        "faces": [
            {
                "index": 0,
                "detector": "canonical-468-vertex-projection",
                "landmark_count": 468,
                "part_masks": {
                    "schema_version": 1,
                    "coordinate_space": "depth",
                    "face_file": face_path.relative_to(output_dir).as_posix(),
                    "files": files,
                    "complete": True,
                },
            }
        ],
    }
    metadata_path = output_dir / "face_part_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _finite(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _compact_appearance(metrics: dict) -> dict:
    keys = (
        "available",
        "samples",
        "reference_samples",
        "candidate_coverage_ratio",
        "normal_mean_cosine",
        "normal_p05_cosine",
        "normal_angle_p95_deg",
        "minimum_lighting_correlation",
        "maximum_lighting_mae",
        "minimum_lighting_rms_retention",
        "maximum_lighting_rms_retention",
    )
    return {key: metrics.get(key) for key in keys}


def _run_row(
    output_dir: Path,
    mesh: trimesh.Trimesh,
    *,
    yaw_deg: float,
    relief_height_mm: float,
    render_size: int,
    physical_size_mm: float,
    render_ortho_scale: float = 1.8,
    background_phase_rad: float = 0.0,
) -> dict:
    started = time.perf_counter()
    row_id = f"yaw_{yaw_deg:+05.1f}_height_{relief_height_mm:04.1f}mm".replace(
        ".", "p"
    )
    row_dir = output_dir / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    camera = CameraSpec(azimuth_deg=float(yaw_deg), elevation_deg=0.0)
    render_config = RenderConfig(
        size=int(render_size),
        ortho_scale=float(render_ortho_scale),
        background_rgb=(0.84, 0.87, 0.91),
        ambient=0.52,
        diffuse=0.48,
        light_direction=(0.35, -0.45, 0.82),
    )
    rendered = render_mesh(
        mesh,
        camera=camera,
        config=render_config,
        base_color=(188, 146, 128),
        vertex_part_weights=canonical_vertex_part_weights(len(mesh.vertices)),
    )
    face_mask, part_masks = canonical_face_masks(
        rendered.silhouette,
        rendered.part_masks,
    )
    source_depth, background_rgb = make_structured_face_scene(
        rendered.depth,
        face_mask,
        yaw_deg,
        scene_phase_rad=background_phase_rad,
    )
    preview = background_rgb.copy()
    preview[face_mask] = rendered.rgb[face_mask]
    Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8)).save(
        row_dir / "source.png"
    )
    metadata_path = _persist_face_masks(row_dir, face_mask, part_masks)

    input_pitch_mm = float(physical_size_mm) / max(int(render_size) - 1, 1)
    composed, compose_stats = compose_selection_depth_with_context(
        source_depth,
        face_mask,
        relief_height_mm=float(relief_height_mm),
        sample_pitch_mm=input_pitch_mm,
        max_slope_mm_per_mm=2.0,
        background_depth_ratio=0.50,
        background_feather_mm=1.5,
        background_smoothing_mm=0.6,
    )
    depth_path = row_dir / "composed_depth.npy"
    surface_path = row_dir / "emitted_surface.npy"
    reference_surface_path = row_dir / "reference_surface.npy"
    stl_path = row_dir / "relief.stl"
    np.save(depth_path, composed.astype(np.float32, copy=False))
    feature_weight = np.zeros(face_mask.shape, dtype=np.float32)
    for mask in part_masks.values():
        feature_weight = np.maximum(feature_weight, np.asarray(mask, dtype=np.float32))

    postprocess = depth_data_to_3d_model(
        depth_path,
        output_stl_path=str(stl_path),
        target_dimension=-1,
        z_scale=float(relief_height_mm),
        invert=True,
        sigma=0.0,
        max_xy_size=float(physical_size_mm),
        relief_gamma=1.0,
        detail_boost=0.0,
        low_percentile=0.0,
        high_percentile=100.0,
        base_border_px=1,
        value_transform="linear",
        minimum_feature_mm=0.8,
        max_relief_slope=2.0,
        face_region_mask=face_mask,
        selection_region_mask=face_mask,
        selection_background_depth_ratio=0.50,
        source_image=row_dir / "source.png",
        background_photo_detail_mm=0.0,
        feature_weight_mask=feature_weight,
        printable_feature_depth_mm=0.8,
        feature_bridge_depth_mm=0.8,
        surface_output_path=surface_path,
        reference_surface_output_path=reference_surface_path,
    )
    postprocess_path = row_dir / "postprocess.json"
    postprocess_path.write_text(json.dumps(postprocess, indent=2), encoding="utf-8")
    emitted = np.load(surface_path).astype(np.float64)
    reference = np.load(reference_surface_path).astype(np.float64)
    transformed_masks = load_persisted_face_part_masks(
        metadata_path,
        surface_grid_transform=postprocess["surface_grid_transform"],
    )
    if len(transformed_masks) != 1:
        raise ValueError(
            f"Canonical row expected one transformed face, got {len(transformed_masks)}"
        )
    masks = transformed_masks[0]
    pitch_mm = float(postprocess["mesh_sample_pitch_mm"])
    absolute_face = _surface_lighting_agreement_metrics(
        reference,
        emitted,
        masks["face_mask"],
        sample_pitch_mm=pitch_mm,
        component_metrics=False,
    )
    absolute_parts = face_part_cross_height_metrics(
        reference,
        emitted,
        masks["face_mask"],
        masks["part_masks"],
        sample_pitch_mm=pitch_mm,
    )
    absolute_part_mm = face_part_affine_surface_error_metrics(
        reference,
        emitted,
        masks["face_mask"],
        masks["part_masks"],
    )
    appearance_checks = _appearance_checks(absolute_face, FACE_APPEARANCE_GATES)
    topology = _mesh_topology(trimesh.load_mesh(stl_path, process=True))
    shell = _stl_heightfield_agreement(
        stl_path,
        surface_path,
        expected_max_xy_size_mm=float(physical_size_mm),
    )
    scene_checks = _scene_checks(compose_stats, postprocess, topology)
    feature_stats = postprocess["printable_feature_depth"]
    checks = {
        "source_context": bool(
            scene_checks["context_composed"] and scene_checks["source_context_signal"]
        ),
        "absolute_face_appearance": bool(appearance_checks["passed"]),
        "absolute_named_parts": bool(absolute_parts["passed"]),
        "absolute_named_part_mm_error": bool(absolute_part_mm["passed"]),
        "background_preservation": bool(scene_checks["background_preservation"]),
        "localized_background_structure": bool(
            scene_checks["localized_background_structure"]
        ),
        "physical_emission": bool(scene_checks["physical_emission"]),
        "feasible_attachment": bool(scene_checks["feasible_attachment"]),
        "printable_mesh": bool(topology["printable"]),
        "complete_shell": bool(shell["passed"]),
        "screened_face_reconstruction": bool(
            postprocess["face_boundary_alignment"].get("method")
            == "screened_gradient_domain_compression"
        ),
        "redundant_emboss_suppressed": bool(
            feature_stats.get("suppressed_after_screened_face_reconstruction", False)
            and float(postprocess["effective_printable_feature_depth_mm"]) == 0.0
        ),
        "feature_bridge_requested": bool(
            float(postprocess["feature_bridge_depth_mm"]) == 0.8
        ),
    }
    background = postprocess["background_depth_preservation"]
    cap = postprocess["selection_background_physical_cap"]
    part_records = [
        {
            "name": record["name"],
            "passed": bool(record["passed"]),
            "samples": int(record["samples"]),
            "visible_samples_before_boundary_exclusion": int(
                record.get("visible_samples_before_boundary_exclusion", 0)
            ),
            "boundary_exclusion_applied": bool(
                record.get("boundary_exclusion_applied", False)
            ),
            "boundary_exclusion_reason": record.get(
                "boundary_exclusion_reason"
            ),
            "boundary_exclusion_eroded_samples": int(
                record.get("boundary_exclusion_eroded_samples", 0)
            ),
            "boundary_exclusion_attempted_retained_fraction": _finite(
                record.get("boundary_exclusion_attempted_retained_fraction")
            ),
            "boundary_exclusion_retained_fraction": _finite(
                record.get("boundary_exclusion_retained_fraction")
            ),
            "shape_correlation": _finite(record.get("shape_correlation")),
            "face_normalized_shape_rmse": _finite(
                record.get("face_normalized_shape_rmse")
            ),
            "minimum_gradient_correlation": _finite(
                record.get("minimum_gradient_correlation")
            ),
            "minimum_raw_gradient_correlation": _finite(
                record.get("minimum_raw_gradient_correlation")
            ),
            "minimum_all_scale_gradient_correlation": _finite(
                record.get("minimum_all_scale_gradient_correlation")
            ),
            "minimum_slope_q95_retention": _finite(
                record.get("minimum_slope_q95_retention")
            ),
            "maximum_slope_q95_retention": _finite(
                record.get("maximum_slope_q95_retention")
            ),
            "minimum_curvature_q95_retention": _finite(
                record.get("minimum_curvature_q95_retention")
            ),
            "maximum_curvature_q95_retention": _finite(
                record.get("maximum_curvature_q95_retention")
            ),
        }
        for record in absolute_parts["parts"]
    ]
    part_mm_records = [
        {
            "name": record["name"],
            "available": bool(record["available"]),
            "passed": bool(record["passed"]),
            "samples": int(record["samples"]),
            "visible_samples_before_boundary_exclusion": int(
                record.get("visible_samples_before_boundary_exclusion", 0)
            ),
            "boundary_exclusion_applied": bool(
                record.get("boundary_exclusion_applied", False)
            ),
            "boundary_exclusion_reason": record.get(
                "boundary_exclusion_reason"
            ),
            "boundary_exclusion_eroded_samples": int(
                record.get("boundary_exclusion_eroded_samples", 0)
            ),
            "boundary_exclusion_attempted_retained_fraction": _finite(
                record.get("boundary_exclusion_attempted_retained_fraction")
            ),
            "boundary_exclusion_retained_fraction": _finite(
                record.get("boundary_exclusion_retained_fraction")
            ),
            "rmse_mm": _finite(record.get("rmse_mm")),
            "bias_mm": _finite(record.get("bias_mm")),
            "p95_absolute_error_mm": _finite(
                record.get("p95_absolute_error_mm")
            ),
            "span_retention": _finite(record.get("span_retention")),
            "reason": record.get("reason"),
        }
        for record in absolute_part_mm["parts"]
    ]
    visible_surface_z = np.asarray(rendered.surface_z)[rendered.silhouette]
    return {
        "row_id": row_id,
        "yaw_deg": float(yaw_deg),
        "relief_height_mm": float(relief_height_mm),
        "render_size": int(render_size),
        "physical_size_mm": float(physical_size_mm),
        "render_ortho_scale": float(render_ortho_scale),
        "background_phase_rad": float(background_phase_rad),
        "runtime_seconds": float(time.perf_counter() - started),
        "checks": {**checks, "passed": bool(all(checks.values()))},
        "compose": {
            "context_pixels": int(compose_stats["background_context_pixels"]),
            "normalized_context_correlation": _finite(
                compose_stats.get("background_context_normalized_correlation")
            ),
            "normalized_context_rms_retention": _finite(
                compose_stats.get("background_context_normalized_rms_retention")
            ),
            "recoverable_context_coverage_ratio": _finite(
                compose_stats.get("background_context_recoverable_coverage_ratio")
            ),
        },
        "absolute_face": {
            **_compact_appearance(absolute_face),
            "checks": appearance_checks,
        },
        "absolute_named_parts": {
            "schema_version": int(absolute_parts["schema_version"]),
            "passed": bool(absolute_parts["passed"]),
            "failed_parts": absolute_parts["failed_parts"],
            "gates": FACE_PART_GATES,
            "parts": part_records,
        },
        "absolute_named_part_mm_error": {
            "schema_version": int(absolute_part_mm["schema_version"]),
            "passed": bool(absolute_part_mm["passed"]),
            "failed_parts": absolute_part_mm["failed_parts"],
            "gates": FACE_PART_AFFINE_MM_GATES,
            "face_affine_fit": absolute_part_mm.get("face_affine_fit"),
            "parts": part_mm_records,
        },
        "render_depth_provenance": {
            "source_coordinate_unit": "canonical centimeters before normalization",
            "camera_frame_z_near": float(np.max(visible_surface_z)),
            "camera_frame_z_far": float(np.min(visible_surface_z)),
            "camera_frame_z_span": float(np.ptp(visible_surface_z)),
            "depth_convention": "near=0, far=1 within visible canonical face",
            "part_labels": "barycentric vertex weights sharing the primary z-buffer",
        },
        "background": {
            "passed": bool(background.get("passed", False)),
            "correlation": _finite(background.get("correlation")),
            "rms_retention": _finite(background.get("rms_retention")),
            "span_retention": _finite(background.get("span_retention")),
            "gradient_correlation": _finite(background.get("gradient_correlation")),
            "gradient_rms_retention": _finite(
                background.get("gradient_rms_retention")
            ),
            "candidate_coverage_ratio": _finite(
                background.get("candidate_coverage_ratio")
            ),
            "localized_structure": background.get("localized_structure"),
        },
        "physical_cap": {
            "emission_passed": bool(cap.get("emission_passed", False)),
            "far_background_max_mm": _finite(cap.get("far_background_max_mm")),
            "far_background_ceiling_mm": _finite(
                cap.get("far_background_ceiling_mm")
            ),
            "feasible_attachment_jump_max_mm": _finite(
                cap.get("feasible_attachment_jump_max_mm")
            ),
        },
        "feature_handling": {
            "requested_feature_depth_mm": float(
                postprocess["printable_feature_depth_mm"]
            ),
            "effective_feature_depth_mm": float(
                postprocess["effective_printable_feature_depth_mm"]
            ),
            "emboss_suppressed": bool(
                feature_stats.get(
                    "suppressed_after_screened_face_reconstruction", False
                )
            ),
            "bridge_enabled": bool(postprocess["feature_bridge"].get("enabled")),
            "bridge_reason": postprocess["feature_bridge"].get("reason"),
        },
        "topology": topology,
        "shell": {
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
        },
        "artifacts": {
            path.relative_to(row_dir).as_posix(): {
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in (
                row_dir / "source.png",
                depth_path,
                reference_surface_path,
                surface_path,
                stl_path,
                postprocess_path,
                metadata_path,
                *sorted((row_dir / "face_parts").glob("*.png")),
            )
        },
    }


def run(
    output_dir: str | Path,
    *,
    yaws_deg: tuple[float, ...] = (0.0,),
    relief_heights_mm: tuple[float, ...] = (30.0,),
    render_size: int = 256,
    physical_size_mm: float = 96.0,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh, asset = load_canonical_face_fixture()
    provenance = _git_provenance(PROVENANCE_PATHS)
    rows = [
        _run_row(
            output_dir,
            mesh,
            yaw_deg=yaw,
            relief_height_mm=height,
            render_size=render_size,
            physical_size_mm=physical_size_mm,
        )
        for yaw, height in product(yaws_deg, relief_heights_mm)
    ]
    checks = {
        "expected_rows": len(rows) == len(yaws_deg) * len(relief_heights_mm),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "all_row_gates_passed": bool(rows)
        and all(row["checks"]["passed"] for row in rows),
    }
    summary = {
        "schema_version": 2,
        "named_face_part_metric_schema_version": (
            FACE_PART_METRIC_SCHEMA_VERSION
        ),
        "run_kind": "canonical_face_absolute_relief_fidelity",
        "privacy": "official canonical mesh and deterministic analytic background only",
        "asset": asset,
        "implementation_provenance": provenance,
        "allow_dirty": bool(allow_dirty),
        "allow_failures": bool(allow_failures),
        "matrix": {
            "yaws_deg": [float(value) for value in yaws_deg],
            "relief_heights_mm": [float(value) for value in relief_heights_mm],
            "render_size": int(render_size),
            "physical_size_mm": float(physical_size_mm),
            "expected_rows": len(yaws_deg) * len(relief_heights_mm),
            "completed_rows": len(rows),
        },
        "face_appearance_gates": FACE_APPEARANCE_GATES,
        "named_face_part_gates": FACE_PART_GATES,
        "named_face_part_affine_mm_gates": FACE_PART_AFFINE_MM_GATES,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["all_row_gates_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failures = []
    if not allow_dirty and not checks["implementation_provenance_clean"]:
        failures.append("implementation_provenance_clean")
    if not checks["all_row_gates_passed"]:
        failures.append("all_row_gates_passed")
    if failures and not allow_failures:
        raise RuntimeError(
            "Canonical face relief smoke failed: " + ", ".join(failures)
        )
    return summary


def _float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="backend/output/canonical-face-relief-smoke",
    )
    parser.add_argument("--yaws-deg", type=_float_tuple, default=(0.0,))
    parser.add_argument(
        "--relief-heights-mm",
        type=_float_tuple,
        default=(30.0,),
    )
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--physical-size-mm", type=float, default=96.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        yaws_deg=args.yaws_deg,
        relief_heights_mm=args.relief_heights_mm,
        render_size=args.render_size,
        physical_size_mm=args.physical_size_mm,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
