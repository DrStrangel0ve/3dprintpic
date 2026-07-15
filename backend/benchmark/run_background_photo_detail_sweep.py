"""Measure face-protected photometric background relief at printable amplitudes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, gaussian_filter, maximum_filter

from backend.benchmark.makehuman_face_fixture import load_makehuman_face_fixture
from backend.benchmark.run_makehuman_face_depth_smoke import DEFAULT_ASSET_DIR, _correlation
from backend.benchmark.run_makehuman_face_relief_smoke import (
    DEFAULT_SCENES,
    SceneSpec,
    _emit_row,
)
from backend.benchmark.run_relief_scene_regression import _git_provenance
from backend.pic_to_3d import (
    BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
    DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
)


DETAIL_LEVELS_MM = (0.0, 0.12, 0.30, 0.60)
SWEEP_SCENES = DEFAULT_SCENES
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/run_makehuman_face_relief_smoke.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/assets/makehuman_cc0_heads",
)
DETAIL_GATES = {
    "minimum_intended_background_source_detail_correlation": 0.30,
    "minimum_realized_p95_ratio": 0.12,
    "maximum_realized_p95_ratio": 1.05,
    "maximum_face_interior_p99_change_mm": 0.01,
    "maximum_face_interior_change_mm": 0.05,
    "maximum_attachment_boundary_change_mm": 0.10,
    "minimum_background_coverage_ratio": 0.35,
}


def _photo_detail_signal(source_path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(source_path).convert("RGB")
    image = image.resize((shape[1], shape[0]), Image.Resampling.LANCZOS)
    rgb = np.flip(np.asarray(image, dtype=np.float32) / 255.0, axis=1)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    fine = gray - gaussian_filter(gray, sigma=1.1)
    medium = gray - gaussian_filter(gray, sigma=3.5)
    return (0.72 * fine + 0.28 * medium).astype(np.float32)


def _detail_metrics(
    baseline_surface: np.ndarray,
    candidate_surface: np.ndarray,
    face_mask: np.ndarray,
    source_path: Path,
    requested_detail_mm: float,
    protection_halo_px: float = 13.0,
) -> dict:
    baseline = np.asarray(baseline_surface, dtype=np.float64)
    candidate = np.asarray(candidate_surface, dtype=np.float64)
    face = np.asarray(face_mask, dtype=bool)
    if baseline.shape != candidate.shape or face.shape != baseline.shape:
        raise ValueError("Detail sweep surfaces and face mask must share one grid")
    delta = candidate - baseline
    source_detail = _photo_detail_signal(source_path, baseline.shape)
    background = ~face
    halo_radius_px = max(1, int(np.ceil(float(protection_halo_px))))
    protection_feather = gaussian_filter(
        maximum_filter(face.astype(np.float32), size=2 * halo_radius_px + 1),
        sigma=max(1.0, 0.625 * halo_radius_px),
    )
    intended_background = background & (protection_feather < 0.5)
    intended_source = np.abs(source_detail[intended_background])
    scale = (
        float(np.percentile(intended_source, 98.0))
        if intended_source.size
        else 0.0
    )
    finite_source_delta = np.isfinite(delta) & np.isfinite(source_detail)
    usable_all = background & finite_source_delta
    usable = intended_background & finite_source_delta
    if scale > 1e-8:
        usable_all &= np.abs(source_detail) >= 0.04 * scale
        usable &= np.abs(source_detail) >= 0.04 * scale
    background_count = max(int(np.count_nonzero(intended_background)), 1)
    coverage = float(np.count_nonzero(usable) / background_count)
    absolute_all = np.abs(delta[background])
    absolute = np.abs(delta[intended_background])
    p95 = float(np.percentile(absolute, 95.0)) if absolute.size else 0.0
    rms = (
        float(np.sqrt(np.mean(np.square(delta[intended_background]))))
        if absolute.size
        else 0.0
    )
    face_interior = binary_erosion(face, iterations=3)
    face_boundary = face & ~face_interior
    interior_delta = np.abs(delta[face_interior])
    boundary_delta = np.abs(delta[face_boundary])
    max_face_interior_change = (
        float(np.max(interior_delta)) if interior_delta.size else 0.0
    )
    face_interior_p99_change = (
        float(np.percentile(interior_delta, 99.0)) if interior_delta.size else 0.0
    )
    max_attachment_boundary_change = (
        float(np.max(boundary_delta)) if boundary_delta.size else 0.0
    )
    source_correlation_all = (
        _correlation(source_detail[usable_all], delta[usable_all])
        if np.count_nonzero(usable_all) >= 64
        else 0.0
    )
    source_correlation_intended = (
        _correlation(source_detail[usable], delta[usable])
        if np.count_nonzero(usable) >= 64
        else 0.0
    )
    active = usable.copy()
    if requested_detail_mm > 0:
        active &= np.abs(delta) >= 0.04 * float(requested_detail_mm)
    active_coverage = float(np.count_nonzero(active) / background_count)
    source_aligned = active & (delta * source_detail > 0)
    source_aligned_capture_ratio = float(
        np.count_nonzero(source_aligned) / max(int(np.count_nonzero(usable)), 1)
    )
    source_correlation = (
        _correlation(source_detail[active], delta[active])
        if np.count_nonzero(active) >= 64
        else 0.0
    )
    if requested_detail_mm > 0:
        realized_ratio = p95 / float(requested_detail_mm)
        checks = {
            "intended_background_source_detail_correlation": source_correlation_intended
            >= DETAIL_GATES["minimum_intended_background_source_detail_correlation"],
            "realized_p95": DETAIL_GATES["minimum_realized_p95_ratio"]
            <= realized_ratio
            <= DETAIL_GATES["maximum_realized_p95_ratio"],
            "face_interior": bool(
                face_interior_p99_change
                <= DETAIL_GATES["maximum_face_interior_p99_change_mm"]
                and max_face_interior_change
                <= DETAIL_GATES["maximum_face_interior_change_mm"]
            ),
            "attachment_boundary": max_attachment_boundary_change
            <= DETAIL_GATES["maximum_attachment_boundary_change_mm"],
            "coverage": coverage >= DETAIL_GATES["minimum_background_coverage_ratio"],
        }
    else:
        realized_ratio = 0.0
        checks = {
            "intended_background_source_detail_correlation": True,
            "realized_p95": p95 <= 1e-9,
            "face_interior": max_face_interior_change <= 1e-9,
            "attachment_boundary": max_attachment_boundary_change <= 1e-9,
            "coverage": coverage >= DETAIL_GATES["minimum_background_coverage_ratio"],
        }
    return {
        "requested_detail_mm": float(requested_detail_mm),
        "active_source_detail_correlation": float(source_correlation),
        "source_detail_correlation_intended_background": float(
            source_correlation_intended
        ),
        "source_detail_correlation_all_background": float(source_correlation_all),
        "background_delta_rms_mm": rms,
        "background_delta_p95_mm": p95,
        "background_delta_p95_all_background_mm": (
            float(np.percentile(absolute_all, 95.0)) if absolute_all.size else 0.0
        ),
        "background_delta_max_mm": (
            float(np.max(absolute_all)) if absolute_all.size else 0.0
        ),
        "realized_p95_ratio": float(realized_ratio),
        "face_interior_p99_change_mm": face_interior_p99_change,
        "max_face_interior_change_mm": max_face_interior_change,
        "max_attachment_boundary_change_mm": max_attachment_boundary_change,
        "background_coverage_ratio": coverage,
        "active_detail_coverage_ratio": active_coverage,
        "source_aligned_capture_ratio": source_aligned_capture_ratio,
        "protection_halo_px": float(protection_halo_px),
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    scenes: tuple[SceneSpec, ...] = SWEEP_SCENES,
    detail_levels_mm: tuple[float, ...] = DETAIL_LEVELS_MM,
    relief_height_mm: float = 30.0,
    render_size: int = 384,
    crop_size: int = 256,
    physical_size_mm: float = 96.0,
    background_depth_ratio: float = DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    levels = tuple(float(value) for value in detail_levels_mm)
    if not levels or levels[0] != 0.0 or any(value < 0 for value in levels):
        raise ValueError("Detail levels must start with a zero baseline and be non-negative")
    if len(set(levels)) != len(levels):
        raise ValueError("Detail levels must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    rows = []
    records = []
    input_pitch_mm = float(physical_size_mm) / max(int(crop_size) - 1, 1)
    protection_halo_px = BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM / max(
        input_pitch_mm, 1e-6
    )
    for spec in scenes:
        baseline_surface = None
        baseline_context = None
        baseline_source = None
        for detail_mm in levels:
            provider_name = f"photo_detail_{detail_mm:.2f}mm".replace(".", "p")
            row, context = _emit_row(
                output_dir,
                fixture,
                spec,
                relief_height_mm=float(relief_height_mm),
                render_size=render_size,
                crop_size=crop_size,
                physical_size_mm=physical_size_mm,
                background_depth_ratio=background_depth_ratio,
                provider_name=provider_name,
                background_photo_detail_mm=detail_mm,
            )
            surface = np.load(context["surface_path"])
            source_path = context["surface_path"].parent / "source.png"
            if detail_mm == 0.0:
                baseline_surface = surface
                baseline_context = context
                baseline_source = source_path
            metrics = _detail_metrics(
                baseline_surface,
                surface,
                baseline_context["face_mask"],
                baseline_source,
                detail_mm,
                protection_halo_px=protection_halo_px,
            )
            record_checks = {
                "detail": bool(metrics["checks"]["passed"]),
                "downstream": bool(row["checks"]["passed"]),
                "physical_cap": bool(row["physical_cap"]["emission_passed"]),
                "printable": bool(row["topology"]["printable"]),
                "shell": bool(row["shell"]["passed"]),
            }
            records.append(
                {
                    "profile_name": spec.profile_name,
                    "framing": spec.framing,
                    "detail_mm": detail_mm,
                    "row_id": row["row_id"],
                    "metrics": metrics,
                    "checks": {
                        **record_checks,
                        "passed": bool(all(record_checks.values())),
                    },
                }
            )
            rows.append(row)

    level_summaries = []
    for detail_mm in levels:
        level_records = [record for record in records if record["detail_mm"] == detail_mm]
        level_summaries.append(
            {
                "detail_mm": detail_mm,
                "minimum_intended_background_source_detail_correlation": float(
                    min(
                        record["metrics"][
                            "source_detail_correlation_intended_background"
                        ]
                        for record in level_records
                    )
                ),
                "minimum_active_source_detail_correlation": float(
                    min(
                        record["metrics"]["active_source_detail_correlation"]
                        for record in level_records
                    )
                ),
                "minimum_background_delta_rms_mm": float(
                    min(record["metrics"]["background_delta_rms_mm"] for record in level_records)
                ),
                "maximum_face_interior_change_mm": float(
                    max(
                        record["metrics"]["max_face_interior_change_mm"]
                        for record in level_records
                    )
                ),
                "maximum_attachment_boundary_change_mm": float(
                    max(
                        record["metrics"]["max_attachment_boundary_change_mm"]
                        for record in level_records
                    )
                ),
                "all_passed": bool(level_records)
                and all(record["checks"]["passed"] for record in level_records),
            }
        )
    eligible = [
        record["detail_mm"]
        for record in level_summaries
        if record["detail_mm"] > 0 and record["all_passed"]
    ]
    recommendation = max(eligible) if eligible else None
    provenance = _git_provenance(PROVENANCE_PATHS)
    checks = {
        "expected_rows": len(rows) == len(scenes) * len(levels),
        "unique_rows": len({row["row_id"] for row in rows}) == len(rows),
        "all_baselines_passed": all(
            record["checks"]["passed"] for record in records if record["detail_mm"] == 0.0
        ),
        "at_least_one_enhanced_level_passed": recommendation is not None,
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "face_protected_background_photo_detail_sweep",
        "privacy": "checksum-pinned CC0 generated heads and deterministic analytic backgrounds only",
        "implementation_provenance": provenance,
        "matrix": {
            "scenes": [spec.__dict__ for spec in scenes],
            "detail_levels_mm": list(levels),
            "relief_height_mm": float(relief_height_mm),
            "physical_size_mm": float(physical_size_mm),
            "background_depth_ratio": float(background_depth_ratio),
        },
        "gates": DETAIL_GATES,
        "recommendation": {
            "background_photo_detail_mm": recommendation,
            "selection_policy": "largest level passing every scene and printability gate",
        },
        "levels": level_summaries,
        "records": records,
        "rows": rows,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["unique_rows"]
                and checks["all_baselines_passed"]
                and checks["at_least_one_enhanced_level_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Background photo-detail sweep failed")
    return summary


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--detail-levels-mm", type=_float_tuple, default=DETAIL_LEVELS_MM)
    parser.add_argument("--relief-height-mm", type=float, default=30.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        asset_dir=args.asset_dir,
        detail_levels_mm=args.detail_levels_mm,
        relief_height_mm=args.relief_height_mm,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
