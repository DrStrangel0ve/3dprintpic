"""Validate protected background photo detail on production DA2 depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from backend.benchmark.makehuman_face_fixture import load_makehuman_face_fixture
from backend.benchmark.run_background_photo_detail_sweep import _detail_metrics
from backend.benchmark.run_makehuman_face_depth_smoke import DEFAULT_ASSET_DIR
from backend.benchmark.run_makehuman_face_provider_relief_smoke import (
    DA2_PROVIDER,
    _infer_cached_provider,
    _prepare_inference_scene,
)
from backend.benchmark.run_makehuman_face_relief_smoke import (
    DEFAULT_SCENES,
    SceneSpec,
    _emit_row,
    _sha256,
)
from backend.benchmark.run_relief_scene_regression import _git_provenance
from backend.pic_to_3d import (
    BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
    DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
)


DETAIL_LEVELS_MM = (0.0, 0.60)
SMOKE_SCENES = (DEFAULT_SCENES[2],)
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/run_makehuman_face_relief_smoke.py",
    "backend/benchmark/run_makehuman_face_provider_relief_smoke.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/run_background_photo_detail_provider_smoke.py",
    "backend/benchmark/assets/makehuman_cc0_heads",
)


def _portable_inference_manifest(manifest: dict) -> dict:
    portable = json.loads(json.dumps(manifest))
    metadata = portable.get("provider_metadata")
    model_id = portable.get("model_id")
    if isinstance(metadata, dict) and isinstance(model_id, str):
        for key in ("model", "effective_model"):
            if key in metadata:
                metadata[key] = model_id
    return portable


def _detail_telemetry_checks(stats: dict, expected_detail_mm: float) -> dict:
    requested = stats.get("requested_detail_mm")
    effective = stats.get("effective_detail_mm")
    checks = {
        "requested_detail": requested is not None
        and abs(float(requested) - float(expected_detail_mm)) <= 1e-9,
        "effective_detail": effective is not None
        and abs(float(effective) - float(expected_detail_mm)) <= 1e-9,
        "not_unprotected_limited": not bool(
            stats.get("unprotected_detail_limited", False)
        ),
    }
    if expected_detail_mm > 0:
        checks["protected"] = bool(stats.get("face_protected", False))
        checks["enabled"] = bool(stats.get("enabled", False))
    return {**checks, "passed": bool(all(checks.values()))}


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    scenes: tuple[SceneSpec, ...] = SMOKE_SCENES,
    detail_levels_mm: tuple[float, ...] = DETAIL_LEVELS_MM,
    relief_height_mm: float = 30.0,
    render_size: int = 384,
    crop_size: int = 256,
    physical_size_mm: float = 96.0,
    background_depth_ratio: float = DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    device: str = "cuda",
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    levels = tuple(float(value) for value in detail_levels_mm)
    if levels != DETAIL_LEVELS_MM:
        raise ValueError("Provider smoke requires the fixed 0.00/0.60 mm pair")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    provenance = _git_provenance(PROVENANCE_PATHS)
    rows = []
    records = []
    inference_manifests = []
    protection_halo_px = BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM / max(
        float(physical_size_mm) / max(int(crop_size) - 1, 1),
        1e-6,
    )

    for spec in scenes:
        scene = _prepare_inference_scene(
            output_dir,
            fixture,
            spec,
            render_size=render_size,
            crop_size=crop_size,
        )
        prediction, manifest = _infer_cached_provider(
            scene,
            provider=DA2_PROVIDER,
            device=device,
        )
        inference_manifests.append(
            {"scene": spec.__dict__, **_portable_inference_manifest(manifest)}
        )
        emitted = []
        for detail_mm in levels:
            provider_name = f"{DA2_PROVIDER}_photo_detail_{detail_mm:.2f}mm".replace(
                ".", "p"
            )
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
                input_depth=prediction,
                value_transform=manifest["value_transform"],
                invert=manifest["value_transform"] == "inverse-depth",
                low_percentile=1.0,
                high_percentile=99.0,
                use_part_feature_weight=False,
                background_photo_detail_mm=detail_mm,
            )
            row_dir = output_dir / row["row_id"]
            postprocess = json.loads(
                (row_dir / "postprocess.json").read_text(encoding="utf-8")
            )
            emitted.append(
                {
                    "detail_mm": detail_mm,
                    "row": row,
                    "context": context,
                    "row_dir": row_dir,
                    "postprocess": postprocess,
                }
            )
            rows.append(row)

        baseline, candidate = emitted
        baseline_surface = np.load(baseline["context"]["surface_path"])
        candidate_surface = np.load(candidate["context"]["surface_path"])
        detail_metrics = _detail_metrics(
            baseline_surface,
            candidate_surface,
            baseline["context"]["face_mask"],
            baseline["row_dir"] / "source.png",
            candidate["detail_mm"],
            protection_halo_px=protection_halo_px,
        )
        baseline_stats = baseline["postprocess"]["background_photo_detail"]
        candidate_stats = candidate["postprocess"]["background_photo_detail"]
        baseline_telemetry = _detail_telemetry_checks(
            baseline_stats, baseline["detail_mm"]
        )
        candidate_telemetry = _detail_telemetry_checks(
            candidate_stats, candidate["detail_mm"]
        )
        baseline_input_hash = baseline["row"]["artifacts"]["provider_depth.npy"][
            "sha256"
        ]
        candidate_input_hash = candidate["row"]["artifacts"]["provider_depth.npy"][
            "sha256"
        ]
        pair_checks = {
            "source_match": (
                _sha256(baseline["row_dir"] / "source.png")
                == _sha256(candidate["row_dir"] / "source.png")
                == scene["source_sha256"]
            ),
            "provider_depth_match": baseline_input_hash
            == candidate_input_hash
            == manifest["resized_sha256"],
            "surface_grid_match": baseline["context"]["surface_grid_transform"]
            == candidate["context"]["surface_grid_transform"],
            "sample_pitch_match": abs(
                float(baseline["context"]["sample_pitch_mm"])
                - float(candidate["context"]["sample_pitch_mm"])
            )
            <= 1e-12,
            "face_mask_match": bool(
                np.array_equal(
                    baseline["context"]["face_mask"],
                    candidate["context"]["face_mask"],
                )
            ),
            "baseline_telemetry": bool(baseline_telemetry["passed"]),
            "candidate_telemetry": bool(candidate_telemetry["passed"]),
            "detail_delta": bool(detail_metrics["checks"]["passed"]),
            "baseline_physical_cap": bool(
                baseline["row"]["physical_cap"]["emission_passed"]
            ),
            "baseline_printable": bool(baseline["row"]["topology"]["printable"]),
            "baseline_shell": bool(baseline["row"]["shell"]["passed"]),
            "physical_cap": bool(candidate["row"]["physical_cap"]["emission_passed"]),
            "printable": bool(candidate["row"]["topology"]["printable"]),
            "shell": bool(candidate["row"]["shell"]["passed"]),
        }
        records.append(
            {
                "profile_name": spec.profile_name,
                "framing": spec.framing,
                "baseline_row_id": baseline["row"]["row_id"],
                "candidate_row_id": candidate["row"]["row_id"],
                "detail_metrics": detail_metrics,
                "baseline_detail_telemetry": baseline_stats,
                "candidate_detail_telemetry": candidate_stats,
                "telemetry_checks": {
                    "baseline": baseline_telemetry,
                    "candidate": candidate_telemetry,
                },
                "diagnostic_downstream_checks": {
                    "baseline": baseline["row"]["checks"],
                    "candidate": candidate["row"]["checks"],
                },
                "checks": {**pair_checks, "passed": bool(all(pair_checks.values()))},
            }
        )

    checks = {
        "expected_rows": len(rows) == len(scenes) * len(levels),
        "one_inference_per_scene": len(inference_manifests) == len(scenes),
        "all_pairs_passed": bool(records)
        and all(record["checks"]["passed"] for record in records),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "production_depth_background_photo_detail_smoke",
        "privacy": "checksum-pinned CC0 generated head and analytic background only",
        "implementation_provenance": provenance,
        "matrix": {
            "provider": DA2_PROVIDER,
            "scenes": [spec.__dict__ for spec in scenes],
            "detail_levels_mm": list(levels),
            "relief_height_mm": float(relief_height_mm),
            "physical_size_mm": float(physical_size_mm),
            "background_depth_ratio": float(background_depth_ratio),
            "device": device,
        },
        "inference_manifests": inference_manifests,
        "records": records,
        "rows": rows,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["one_inference_per_scene"]
                and checks["all_pairs_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Production-depth background photo-detail smoke failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.output_dir,
        asset_dir=args.asset_dir,
        scenes=DEFAULT_SCENES if args.all_scenes else SMOKE_SCENES,
        device=args.device,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )
    print(json.dumps(summary["checks"], indent=2))


if __name__ == "__main__":
    main()
