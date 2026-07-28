"""Run absolute relief fidelity checks over deterministic canonical face variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

from backend.benchmark.canonical_face_variants import (
    CANONICAL_FACE_VARIANTS,
    canonical_face_variant_by_name,
    deform_canonical_face,
)
from backend.benchmark.face_part_metrics import FACE_PART_METRIC_SCHEMA_VERSION
from backend.benchmark.run_canonical_face_relief_smoke import (
    PROVENANCE_PATHS,
    _float_tuple,
    _git_provenance,
    _run_row,
    load_canonical_face_fixture,
)


VARIANT_PROVENANCE_PATHS = PROVENANCE_PATHS + (
    "backend/benchmark/canonical_face_variants.py",
    "backend/benchmark/run_canonical_face_variant_smoke.py",
)
VARIANT_RENDER_SETTINGS = {
    "neutral": {"ortho_scale": 1.80, "background_phase_rad": 0.00},
    "broad_shallow_nose": {"ortho_scale": 1.92, "background_phase_rad": 0.73},
    "narrow_projected_nose": {"ortho_scale": 1.68, "background_phase_rad": 1.46},
    "smile_high_cheek": {"ortho_scale": 1.84, "background_phase_rad": 2.19},
    "asymmetric_expression": {"ortho_scale": 1.74, "background_phase_rad": 2.92},
    "deep_set_angular": {"ortho_scale": 1.96, "background_phase_rad": 3.65},
}
DIVERSITY_GATES = {
    "minimum_unique_geometry_count": 6,
    "minimum_face_width_range_cm": 2.5,
    "minimum_nose_projection_range_cm": 0.8,
    "minimum_interocular_range_cm": 0.5,
    "minimum_mouth_corner_mean_range_cm": 0.2,
    "minimum_maximum_absolute_mouth_tilt_cm": 0.25,
}


def _diversity_metrics(variant_audits: list[dict]) -> dict:
    geometry_hashes = {audit["geometry_sha256"] for audit in variant_audits}
    shape_metrics = [audit["shape_metrics"] for audit in variant_audits]

    def value_range(name: str) -> float:
        values = [float(metrics[name]) for metrics in shape_metrics]
        return float(max(values) - min(values))

    metrics = {
        "unique_geometry_count": len(geometry_hashes),
        "face_width_range_cm": value_range("face_width_cm"),
        "nose_projection_range_cm": value_range(
            "nose_projection_from_eye_plane_cm"
        ),
        "interocular_range_cm": value_range("interocular_distance_cm"),
        "mouth_corner_mean_range_cm": value_range("mouth_corner_mean_y_cm"),
        "maximum_absolute_mouth_tilt_cm": max(
            abs(float(item["mouth_corner_tilt_cm"])) for item in shape_metrics
        ),
    }
    checks = {
        "unique_geometry": metrics["unique_geometry_count"]
        >= DIVERSITY_GATES["minimum_unique_geometry_count"],
        "face_width": metrics["face_width_range_cm"]
        >= DIVERSITY_GATES["minimum_face_width_range_cm"],
        "nose_projection": metrics["nose_projection_range_cm"]
        >= DIVERSITY_GATES["minimum_nose_projection_range_cm"],
        "interocular_distance": metrics["interocular_range_cm"]
        >= DIVERSITY_GATES["minimum_interocular_range_cm"],
        "mouth_corner_height": metrics["mouth_corner_mean_range_cm"]
        >= DIVERSITY_GATES["minimum_mouth_corner_mean_range_cm"],
        "mouth_asymmetry": metrics["maximum_absolute_mouth_tilt_cm"]
        >= DIVERSITY_GATES["minimum_maximum_absolute_mouth_tilt_cm"],
    }
    return {
        **metrics,
        "gates": DIVERSITY_GATES,
        "checks": {**checks, "passed": bool(all(checks.values()))},
        "passed": bool(all(checks.values())),
    }


def run(
    output_dir: str | Path,
    *,
    variant_names: tuple[str, ...] = tuple(
        variant.name for variant in CANONICAL_FACE_VARIANTS
    ),
    yaws_deg: tuple[float, ...] = (-45.0, 0.0, 45.0),
    relief_heights_mm: tuple[float, ...] = (30.0,),
    render_size: int = 256,
    physical_size_mm: float = 96.0,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mesh, asset = load_canonical_face_fixture()
    variants = [canonical_face_variant_by_name(name) for name in variant_names]
    variant_records = []
    rows = []
    for variant in variants:
        mesh, audit = deform_canonical_face(source_mesh, variant)
        settings = VARIANT_RENDER_SETTINGS[variant.name]
        variant_records.append(
            {
                "name": variant.name,
                "parameters": asdict(variant),
                "render_settings": settings,
                "deformation_audit": audit,
            }
        )
        for yaw_deg, relief_height_mm in product(yaws_deg, relief_heights_mm):
            row = _run_row(
                output_dir / variant.name,
                mesh,
                yaw_deg=yaw_deg,
                relief_height_mm=relief_height_mm,
                render_size=render_size,
                physical_size_mm=physical_size_mm,
                render_ortho_scale=settings["ortho_scale"],
                background_phase_rad=settings["background_phase_rad"],
            )
            local_row_id = row["row_id"]
            row["variant_name"] = variant.name
            row["variant_geometry_sha256"] = audit["geometry_sha256"]
            row["artifact_subdirectory"] = (
                Path(variant.name) / local_row_id
            ).as_posix()
            row["row_id"] = f"{variant.name}__{local_row_id}"
            rows.append(row)

    provenance = _git_provenance(VARIANT_PROVENANCE_PATHS)
    deformation_audits = [
        record["deformation_audit"] for record in variant_records
    ]
    diversity = _diversity_metrics(deformation_audits)
    expected_rows = len(variants) * len(yaws_deg) * len(relief_heights_mm)
    checks = {
        "expected_rows": len(rows) == expected_rows,
        "unique_rows": len({row["row_id"] for row in rows}) == len(rows),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "all_deformations_passed": bool(deformation_audits)
        and all(audit["passed"] for audit in deformation_audits),
        "diversity_passed": bool(diversity["passed"]),
        "all_row_gates_passed": bool(rows)
        and all(row["checks"]["passed"] for row in rows),
    }
    summary = {
        "schema_version": 2,
        "named_face_part_metric_schema_version": (
            FACE_PART_METRIC_SCHEMA_VERSION
        ),
        "run_kind": "canonical_face_variant_absolute_relief_fidelity",
        "privacy": "pinned canonical mesh plus deterministic deformations and analytic backgrounds only",
        "asset": asset,
        "implementation_provenance": provenance,
        "allow_dirty": bool(allow_dirty),
        "allow_failures": bool(allow_failures),
        "matrix": {
            "variant_names": [variant.name for variant in variants],
            "yaws_deg": [float(value) for value in yaws_deg],
            "relief_heights_mm": [float(value) for value in relief_heights_mm],
            "render_size": int(render_size),
            "physical_size_mm": float(physical_size_mm),
            "expected_rows": expected_rows,
            "completed_rows": len(rows),
        },
        "diversity": diversity,
        "variants": variant_records,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["unique_rows"]
                and checks["all_deformations_passed"]
                and checks["diversity_passed"]
                and checks["all_row_gates_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    failures = [
        name
        for name, passed in checks.items()
        if not passed and not (allow_dirty and name == "implementation_provenance_clean")
    ]
    if failures and not allow_failures:
        raise RuntimeError(
            "Canonical face variant smoke failed: " + ", ".join(failures)
        )
    return summary


def _string_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one variant name is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="backend/output/canonical-face-variant-smoke",
    )
    parser.add_argument(
        "--variants",
        type=_string_tuple,
        default=tuple(variant.name for variant in CANONICAL_FACE_VARIANTS),
    )
    parser.add_argument("--yaws-deg", type=_float_tuple, default=(-45.0, 0.0, 45.0))
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
        variant_names=args.variants,
        yaws_deg=args.yaws_deg,
        relief_heights_mm=args.relief_heights_mm,
        render_size=args.render_size,
        physical_size_mm=args.physical_size_mm,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
