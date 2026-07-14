"""Run deterministic portrait-and-background relief regressions at 30 mm."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.benchmark.mesh_rendering import load_mesh
from backend.pic_to_3d import (
    _background_relief_preservation_metrics,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/mesh_rendering.py",
)


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    background: str
    center_col: float = 60.0
    center_row: float = 44.0
    yaw: float = 0.0
    expression: float = 0.0


def _scene_specs() -> tuple[SceneSpec, ...]:
    return (
        SceneSpec("planar_room_center", "planar_room"),
        SceneSpec(
            "framed_wall_left_portrait",
            "framed_wall",
            center_col=39.0,
            yaw=-0.35,
            expression=0.2,
        ),
        SceneSpec(
            "shelving_right_portrait",
            "shelving",
            center_col=80.0,
            yaw=0.28,
            expression=-0.2,
        ),
        SceneSpec(
            "soft_outdoor_smile",
            "soft_outdoor",
            center_col=55.0,
            expression=0.65,
        ),
        SceneSpec(
            "split_depth_turn",
            "split_depth",
            center_col=66.0,
            yaw=0.55,
            expression=0.1,
        ),
        SceneSpec(
            "low_contrast_close_background",
            "low_contrast",
            center_col=47.0,
            center_row=41.0,
            yaw=-0.18,
            expression=-0.55,
        ),
    )


def _background_surface(kind: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if kind == "planar_room":
        surface = 0.07 + 0.0019 * cols + 0.00035 * rows
        surface += 0.020 * np.sin(cols / 8.0) + 0.012 * np.cos(rows / 11.0)
        surface += 0.065 * np.exp(-((rows - 31.0) ** 2 + (cols - 98.0) ** 2) / 330.0)
        return surface
    if kind == "framed_wall":
        surface = 0.085 + 0.00065 * rows + 0.00035 * cols
        surface += 0.090 * np.exp(-((cols - 16.0) / 5.5) ** 2)
        surface += 0.075 * np.exp(-((cols - 105.0) / 6.0) ** 2)
        surface += 0.035 * np.exp(-((rows - 19.0) / 6.0) ** 2)
        return surface
    if kind == "shelving":
        surface = 0.065 + 0.0008 * cols + 0.015 * np.sin(rows / 9.0)
        surface += 0.045 * ((rows > 24.0) & (rows < 31.0))
        surface += 0.070 * ((rows > 58.0) & (rows < 65.0))
        surface += 0.055 * ((cols > 9.0) & (cols < 16.0))
        surface += 0.060 * ((cols > 103.0) & (cols < 111.0))
        return surface
    if kind == "soft_outdoor":
        surface = 0.075 + 0.016 * np.sin(cols / 13.0) + 0.012 * np.cos(rows / 15.0)
        for center_row, center_col, amplitude, radius in (
            (25.0, 18.0, 0.11, 260.0),
            (18.0, 92.0, 0.08, 330.0),
            (73.0, 105.0, 0.12, 420.0),
            (92.0, 19.0, 0.07, 300.0),
        ):
            surface += amplitude * np.exp(
                -((rows - center_row) ** 2 + (cols - center_col) ** 2) / radius
            )
        return surface
    if kind == "split_depth":
        near_wall = 1.0 / (1.0 + np.exp(-(cols - 74.0) / 4.0))
        surface = 0.055 + 0.205 * near_wall + 0.00045 * rows
        surface += 0.018 * np.sin(rows / 8.5) + 0.012 * np.cos(cols / 10.0)
        return surface
    if kind == "low_contrast":
        surface = 0.315 + 0.055 * np.sin(cols / 15.0) + 0.035 * np.cos(rows / 17.0)
        surface += 0.070 * np.exp(-((rows - 29.0) ** 2 + (cols - 93.0) ** 2) / 500.0)
        return surface
    raise ValueError(f"Unknown synthetic background: {kind}")


def _synthetic_scene(spec: SceneSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (121, 121)
    rows, cols = np.indices(shape, dtype=np.float64)
    x = (cols - spec.center_col) / 24.0
    y = (rows - spec.center_row) / 30.0
    face = np.square(x) + np.square(y) <= 1.0

    neck_start = spec.center_row + 22.0
    neck_end = spec.center_row + 48.0
    neck_half_width = 11.0 - 0.10 * np.clip(rows - neck_start, 0.0, 22.0)
    neck = (
        (rows >= neck_start)
        & (rows <= neck_end)
        & (np.abs(cols - spec.center_col) <= neck_half_width)
    )
    torso = (
        np.square((cols - spec.center_col) / 47.0)
        + np.square((rows - (spec.center_row + 69.0)) / 38.0)
        <= 1.0
    )
    subject = face | neck | torso

    relief = _background_surface(spec.background, rows, cols)
    relief[torso] = 0.36 + 0.0009 * cols[torso]
    relief[neck] = 0.49 + 0.0006 * cols[neck]

    yaw_offset = 0.11 * spec.yaw
    facial_shape = 0.040 * np.exp(-(np.square(x / 0.72) + np.square(y / 0.88)))
    facial_shape += 0.077 * np.exp(
        -(
            np.square((x - yaw_offset) / 0.17)
            + np.square((y - 0.02) / 0.28)
        )
    )
    eye_balance = 0.0045 * spec.yaw
    facial_shape -= (0.014 - eye_balance) * np.exp(
        -(np.square((x - 0.30) / 0.13) + np.square((y + 0.20) / 0.08))
    )
    facial_shape -= (0.014 + eye_balance) * np.exp(
        -(np.square((x + 0.30) / 0.13) + np.square((y + 0.20) / 0.08))
    )
    mouth_curve = y - 0.44 - 0.025 * spec.expression * np.square(x / 0.34)
    facial_shape += (0.013 + 0.003 * abs(spec.expression)) * np.exp(
        -(np.square(x / 0.29) + np.square(mouth_curve / 0.07))
    )
    facial_shape += 0.006 * spec.yaw * x
    relief[face] = 0.58 + facial_shape[face]
    return relief.astype(np.float32), face, subject


def _scene_checks(compose: dict, postprocess: dict, topology: dict) -> dict[str, bool]:
    background = postprocess["background_depth_preservation"]
    cap = postprocess["selection_background_physical_cap"]
    face = postprocess["face_detail_guard"].get("final", {})
    far_max = cap.get("far_background_max_mm")
    far_ceiling = cap.get("far_background_ceiling_mm")
    return {
        "context_composed": bool(
            compose.get("background_context_enabled")
            and compose.get("background_context_pixels", 0) > 0
        ),
        "source_context_signal": bool(
            float(compose.get("background_context_normalized_correlation", 0.0)) >= 0.95
            and 0.8
            <= float(compose.get("background_context_normalized_rms_retention", 0.0))
            <= 1.2
            and float(compose.get("background_context_recoverable_coverage_ratio", 0.0))
            >= 0.6
        ),
        "background_preservation": bool(
            background.get("available", False)
            and background.get("passed", False)
        ),
        "background_coverage": float(background.get("candidate_coverage_ratio", 0.0)) >= 1.0,
        "localized_background_structure": bool(
            background.get("localized_structure", {}).get("passed", False)
        ),
        "face_detail": bool(
            face.get("available", False)
            and float(face.get("correlation", 0.0)) >= 0.8
            and 0.6 <= float(face.get("rms_retention", 0.0)) <= 2.0
        ),
        "physical_emission": bool(cap.get("emission_passed", False)),
        "far_background_cap": bool(
            far_max is not None
            and far_ceiling is not None
            and float(far_max) <= float(far_ceiling) + 1e-5
        ),
        "feasible_attachment": bool(
            cap.get("feasible_attachment_constraints_passed", False)
        ),
        "printable_mesh": bool(topology.get("printable", False)),
    }


def _finite_metric(record: dict, key: str) -> float | None:
    value = record.get(key)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _mesh_topology(mesh) -> dict:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold_edges = int(np.count_nonzero(edge_counts != 2))
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    degenerate_faces = int(
        np.count_nonzero((~np.isfinite(face_areas)) | (face_areas <= 1e-12))
    )
    components = len(mesh.split(only_watertight=False))
    try:
        volume = float(mesh.volume)
    except Exception:
        volume = float("nan")
    positive_volume = bool(np.isfinite(volume) and volume > 0)
    bbox_has_volume = bool(
        np.asarray(mesh.extents).shape == (3,)
        and np.all(np.isfinite(mesh.extents))
        and np.all(np.asarray(mesh.extents) > 0)
    )
    topology = {
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "component_count": int(components),
        "nonmanifold_edge_count": nonmanifold_edges,
        "degenerate_face_count": degenerate_faces,
        "positive_volume": positive_volume,
        "bbox_has_volume": bbox_has_volume,
        "face_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
    }
    topology["printable"] = bool(
        topology["watertight"]
        and topology["is_volume"]
        and topology["winding_consistent"]
        and topology["component_count"] == 1
        and nonmanifold_edges == 0
        and degenerate_faces == 0
        and positive_volume
        and bbox_has_volume
    )
    return topology


def _git_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]

    def run_git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        revision = run_git("rev-parse", "HEAD")
        status = run_git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *PROVENANCE_PATHS,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "clean": False,
            "revision": None,
            "paths": list(PROVENANCE_PATHS),
            "status": [],
            "error": type(exc).__name__,
        }
    return {
        "available": True,
        "clean": not bool(status),
        "revision": revision,
        "paths": list(PROVENANCE_PATHS),
        "status": status.splitlines() if status else [],
    }


def _negative_controls() -> dict:
    rows, cols = np.indices((80, 100), dtype=np.float32)
    foreground = ((rows - 47.0) ** 2 / 210.0 + (cols - 52.0) ** 2 / 330.0) <= 1.0
    background = ~foreground
    reference = (
        2.0
        + 0.035 * cols
        + 0.9 * np.sin(cols / 9.0)
        + 0.5 * np.cos(rows / 7.0)
        + 4.0 * foreground
    ).astype(np.float32)
    flattened = reference.copy()
    flattened[background] = float(np.median(reference[background]))
    flattened_metrics = _background_relief_preservation_metrics(
        reference,
        flattened,
        foreground,
        sample_pitch_mm=0.4,
    )

    bump = 2.5 * np.exp(-((rows - 24.0) ** 2 + (cols - 82.0) ** 2) / 28.0)
    localized_reference = reference.copy()
    localized_reference[background] += bump[background]
    localized_loss = localized_reference.copy()
    localized_loss[background] -= bump[background]
    localized_metrics = _background_relief_preservation_metrics(
        localized_reference,
        localized_loss,
        foreground,
        sample_pitch_mm=0.4,
    )

    narrow_foreground = np.ones((40, 40), dtype=bool)
    narrow_foreground[:3, :] = False
    narrow_reference = (2.0 + 0.02 * rows[:40, :40] + 0.01 * cols[:40, :40]).astype(
        np.float32
    )
    unavailable_metrics = _background_relief_preservation_metrics(
        narrow_reference,
        narrow_reference.copy(),
        narrow_foreground,
        sample_pitch_mm=0.4,
    )
    checks = {
        "whole_background_flattening_rejected": bool(
            not flattened_metrics["passed"]
            and "correlation" in flattened_metrics["quality_failures"]
        ),
        "localized_object_loss_rejected": bool(
            not localized_metrics["passed"]
            and "localized_structure" in localized_metrics["quality_failures"]
        ),
        "insufficient_context_unavailable": bool(
            not unavailable_metrics["available"]
            and not unavailable_metrics["passed"]
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "localized_loss_global_correlation": _finite_metric(
            localized_metrics, "correlation"
        ),
        "localized_loss_global_gradient_correlation": _finite_metric(
            localized_metrics, "gradient_correlation"
        ),
        "localized_loss_failed_windows": int(
            localized_metrics["localized_structure"]["failed_window_count"]
        ),
        "insufficient_context_reason": unavailable_metrics.get("reason"),
    }


def _extreme(rows: list[dict], section: str, key: str, function) -> float | None:
    values = [
        float(row[section][key])
        for row in rows
        if row[section].get(key) is not None
        and np.isfinite(float(row[section][key]))
    ]
    return float(function(values)) if values else None


def _certification_checks(
    rows: list[dict],
    *,
    expected_scene_count: int,
    full_scene_count: int,
    provenance: dict,
    negative_controls: dict,
) -> dict[str, bool]:
    required_metrics = (
        ("compose", "normalized_context_correlation"),
        ("compose", "normalized_context_rms_retention"),
        ("compose", "recoverable_context_coverage_ratio"),
        ("background", "correlation"),
        ("background", "gradient_correlation"),
        ("face", "correlation"),
        ("physical_cap", "far_background_max_mm"),
    )
    return {
        "full_scene_matrix_complete": len(rows) == int(full_scene_count),
        "expected_scene_count": len(rows) == int(expected_scene_count),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "required_telemetry_complete": bool(rows)
        and all(
            row.get(section, {}).get(key) is not None
            and np.isfinite(float(row[section][key]))
            for row in rows
            for section, key in required_metrics
        ),
        "negative_controls_passed": bool(
            negative_controls.get("checks", {}).get("passed", False)
        ),
        "all_scene_gates_passed": bool(rows)
        and all(row["checks"]["passed"] for row in rows),
        "all_background_gates_passed": bool(rows)
        and all(row["checks"]["background_preservation"] for row in rows),
        "all_face_gates_passed": bool(rows)
        and all(row["checks"]["face_detail"] for row in rows),
        "all_physical_emission_gates_passed": bool(rows)
        and all(row["checks"]["physical_emission"] for row in rows),
        "all_meshes_printable": bool(rows)
        and all(row["checks"]["printable_mesh"] for row in rows),
    }


def run(
    output_dir: str | Path,
    summary_path: str | Path | None = None,
    limit: int = 6,
    allow_dirty: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_specs = _scene_specs()
    specs = all_specs[: max(0, int(limit))]
    if not specs:
        raise ValueError("Relief scene regression requires at least one scene")

    rows = []
    started = time.perf_counter()
    for spec in specs:
        scene_started = time.perf_counter()
        scene_dir = output_dir / spec.scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        source, face_mask, subject_mask = _synthetic_scene(spec)
        composed, compose_stats = compose_selection_depth_with_context(
            source,
            subject_mask,
            relief_height_mm=30.0,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
            background_feather_mm=1.5,
            background_smoothing_mm=0.6,
        )
        depth_path = scene_dir / "composed_depth.npy"
        stl_path = scene_dir / "relief_30mm.stl"
        np.save(depth_path, composed)
        postprocess = depth_data_to_3d_model(
            depth_path,
            output_stl_path=str(stl_path),
            target_dimension=-1,
            z_scale=30.0,
            max_xy_size=48.0,
            sigma=0.0,
            relief_gamma=1.0,
            detail_boost=0.0,
            low_percentile=0.0,
            high_percentile=100.0,
            base_border_px=1,
            value_transform="linear",
            minimum_feature_mm=0.8,
            max_relief_slope=2.0,
            face_region_mask=face_mask,
            selection_region_mask=subject_mask,
            selection_background_depth_ratio=0.45,
        )

        mesh = load_mesh(stl_path)
        topology = _mesh_topology(mesh)
        background = postprocess["background_depth_preservation"]
        cap = postprocess["selection_background_physical_cap"]
        face = postprocess["face_detail_guard"].get("final", {})
        checks = _scene_checks(compose_stats, postprocess, topology)
        rows.append(
            {
                "scene_id": spec.scene_id,
                "background_archetype": spec.background,
                "runtime_seconds": float(time.perf_counter() - scene_started),
                "checks": {**checks, "passed": all(checks.values())},
                "compose": {
                    "context_pixels": int(compose_stats["background_context_pixels"]),
                    "context_coverage_ratio": float(
                        compose_stats["background_context_coverage_ratio"]
                    ),
                    "context_correlation": _finite_metric(
                        compose_stats, "background_context_correlation"
                    ),
                    "normalized_context_correlation": _finite_metric(
                        compose_stats, "background_context_normalized_correlation"
                    ),
                    "normalized_context_rms_retention": _finite_metric(
                        compose_stats,
                        "background_context_normalized_rms_retention",
                    ),
                    "recoverable_context_coverage_ratio": _finite_metric(
                        compose_stats,
                        "background_context_recoverable_coverage_ratio",
                    ),
                    "output_span_ratio": float(
                        compose_stats["background_output_span_ratio"]
                    ),
                },
                "background": {
                    "correlation": _finite_metric(background, "correlation"),
                    "rms_retention": _finite_metric(background, "rms_retention"),
                    "span_retention": _finite_metric(background, "span_retention"),
                    "gradient_correlation": _finite_metric(
                        background, "gradient_correlation"
                    ),
                    "gradient_rms_retention": _finite_metric(
                        background, "gradient_rms_retention"
                    ),
                    "mean_shift_mm": _finite_metric(background, "mean_shift_mm"),
                    "candidate_coverage_ratio": _finite_metric(
                        background, "candidate_coverage_ratio"
                    ),
                    "fallback_used": "fallback" in background,
                    "localized_structure": background.get("localized_structure"),
                },
                "face": {
                    "correlation": _finite_metric(face, "correlation"),
                    "rms_retention": _finite_metric(face, "rms_retention"),
                    "component_count": int(face.get("component_count", 0)),
                },
                "physical_cap": {
                    "far_background_ceiling_mm": _finite_metric(
                        cap, "far_background_ceiling_mm"
                    ),
                    "far_background_max_mm": _finite_metric(
                        cap, "far_background_max_mm"
                    ),
                    "far_background_cap_violation_mm": _finite_metric(
                        cap, "far_background_cap_violation_mm"
                    ),
                    "feasible_attachment_jump_max_mm": _finite_metric(
                        cap, "feasible_attachment_jump_max_mm"
                    ),
                    "attachment_constraint_conflicts": int(
                        cap.get("attachment_constraint_conflicts", 0)
                    ),
                    "strict_passed": bool(cap.get("passed", False)),
                    "emission_passed": bool(cap.get("emission_passed", False)),
                },
                "topology": topology,
            }
        )

    provenance = _git_provenance()
    negative_controls = _negative_controls()
    checks = _certification_checks(
        rows,
        expected_scene_count=len(specs),
        full_scene_count=len(all_specs),
        provenance=provenance,
        negative_controls=negative_controls,
    )
    summary = {
        "schema_version": 2,
        "run_kind": "deterministic_privacy_safe_relief_scene_regression",
        "privacy": "all inputs are analytic arrays; no photos, masks, or private meshes are used",
        "implementation_provenance": provenance,
        "allow_dirty": bool(allow_dirty),
        "relief_height_mm": 30.0,
        "background_depth_ratio": 0.45,
        "sample_pitch_mm": 0.4,
        "scene_count": len(rows),
        "runtime_seconds": float(time.perf_counter() - started),
        "checks": {**checks, "passed": all(checks.values())},
        "aggregate": {
            "minimum_normalized_context_correlation": _extreme(
                rows, "compose", "normalized_context_correlation", min
            ),
            "minimum_recoverable_context_coverage_ratio": _extreme(
                rows, "compose", "recoverable_context_coverage_ratio", min
            ),
            "minimum_background_correlation": _extreme(
                rows, "background", "correlation", min
            ),
            "minimum_background_rms_retention": _extreme(
                rows, "background", "rms_retention", min
            ),
            "minimum_background_span_retention": _extreme(
                rows, "background", "span_retention", min
            ),
            "minimum_background_gradient_correlation": _extreme(
                rows, "background", "gradient_correlation", min
            ),
            "minimum_face_correlation": _extreme(rows, "face", "correlation", min),
            "minimum_face_rms_retention": _extreme(
                rows, "face", "rms_retention", min
            ),
            "maximum_far_background_mm": _extreme(
                rows, "physical_cap", "far_background_max_mm", max
            ),
            "maximum_feasible_attachment_jump_mm": _extreme(
                rows, "physical_cap", "feasible_attachment_jump_max_mm", max
            ),
            "total_attachment_constraint_conflicts": sum(
                row["physical_cap"]["attachment_constraint_conflicts"] for row in rows
            ),
            "fallback_count": sum(row["background"]["fallback_used"] for row in rows),
        },
        "negative_controls": negative_controls,
        "rows": rows,
    }
    destination = Path(summary_path) if summary_path else output_dir / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["checks"]["passed"] and not allow_dirty:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Relief scene regression failed: {', '.join(failed)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="backend/output/relief-scene-regression-local-n6",
    )
    parser.add_argument("--summary-path")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        summary_path=args.summary_path,
        limit=args.limit,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
