"""Analyze localized face quality from an exact CC0 live-matrix run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark import run_cc0_live_face_variation_matrix as matrix
from backend.benchmark.face_relief_local_metrics import (
    globally_align_face_surface,
    localized_face_surface_metrics,
    transform_mask_to_emitted_grid,
)
from backend.benchmark.makehuman_face_fixture import (
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import CameraSpec, render_mesh


ANALYSIS_SCHEMA_VERSION = 2


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {Path(path).name}")
    return payload


def _resolve_artifact(record: dict, root: Path, label: str) -> Path:
    relative = record.get("path")
    expected = record.get("sha256")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError(f"Unsafe {label} artifact path")
    path = (root / relative).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{label} artifact is missing or has the wrong checksum")
    return path


def _face_spec(scene: dict) -> matrix.FaceSceneSpec:
    values = dict(scene)
    occluder = values.get("occluder")
    if isinstance(occluder, dict):
        values["occluder"] = matrix.OccluderSpec(**occluder)
    elif occluder is not None:
        raise ValueError("Face scene has invalid occluder metadata")
    return matrix.FaceSceneSpec(**values)


def _rerender_part_masks(
    spec: matrix.FaceSceneSpec,
    fixture: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    rendered = render_mesh(
        profile["mesh"],
        CameraSpec(
            azimuth_deg=float(spec.camera_yaw_deg),
            elevation_deg=float(spec.camera_elevation_deg),
        ),
        matrix._render_config(spec, effective_distance),
        (180, 120, 100),
        vertex_part_weights=fixture["part_weights"],
        vertex_colors=colors,
    )
    offset = int(round(float(spec.horizontal_offset) * spec.target_dimension))
    face = matrix._translate(np.asarray(rendered.silhouette, dtype=bool), offset, False)
    parts = {
        name: matrix._translate(np.asarray(mask, dtype=bool), offset, False)
        for name, mask in rendered.part_masks.items()
    }
    return face, parts


def _compact_part_metrics(metrics: dict) -> dict:
    cross_parts = {
        record["name"]: record for record in metrics["cross_height"].get("parts", [])
    }
    affine_parts = {
        record["name"]: record for record in metrics["affine_mm"].get("parts", [])
    }
    appearance_parts = {
        record["name"]: record
        for record in metrics["multi_light_appearance"].get("parts", [])
    }
    rows = []
    for name in metrics["region_names"]:
        cross = cross_parts.get(name, {})
        affine = affine_parts.get(name, {})
        appearance = appearance_parts.get(name, {})
        lighting = appearance.get("metrics", {})
        rows.append(
            {
                "name": name,
                "passed": bool(
                    cross.get("passed", False)
                    and affine.get("passed", False)
                    and appearance.get("passed", False)
                ),
                "shape_correlation": cross.get("shape_correlation"),
                "face_normalized_shape_rmse": cross.get("face_normalized_shape_rmse"),
                "minimum_raw_gradient_correlation": cross.get(
                    "minimum_raw_gradient_correlation"
                ),
                "minimum_gradient_correlation": cross.get(
                    "minimum_gradient_correlation"
                ),
                "rmse_mm": affine.get("rmse_mm"),
                "p95_absolute_error_mm": affine.get("p95_absolute_error_mm"),
                "span_retention": affine.get("span_retention"),
                "normal_mean_cosine": lighting.get("normal_mean_cosine"),
                "normal_angle_p95_deg": lighting.get("normal_angle_p95_deg"),
                "minimum_lighting_correlation": lighting.get(
                    "minimum_lighting_correlation"
                ),
                "minimum_lighting_rms_retention": lighting.get(
                    "minimum_lighting_rms_retention"
                ),
                "maximum_lighting_rms_retention": lighting.get(
                    "maximum_lighting_rms_retention"
                ),
                "failed_checks": {
                    "cross_height": [
                        key
                        for key, value in cross.get("checks", {}).items()
                        if not value
                    ],
                    "affine_mm": [
                        key
                        for key, value in affine.get("checks", {}).items()
                        if not value
                    ],
                    "appearance": [
                        key
                        for key, value in appearance.get("checks", {}).items()
                        if key != "passed" and not value
                    ],
                },
            }
        )
    return {
        "passed": bool(metrics["checks"]["passed"]),
        "failed_regions": [row["name"] for row in rows if not row["passed"]],
        "regions": rows,
    }


def _analyze_row(
    row: dict,
    *,
    run_dir: Path,
    server_output: Path,
    fixture: dict,
) -> dict:
    spec = _face_spec(row["scene"])
    source_path = _resolve_artifact(row["source"], run_dir, "source")
    mask_path = _resolve_artifact(row["selection_mask"], run_dir, "selection mask")
    exact_path = _resolve_artifact(row["exact_depth"], run_dir, "exact depth")
    candidate = row["variants"]["candidate"]
    response_path = _resolve_artifact(
        candidate["response_artifact"], run_dir, "candidate response"
    )
    response = _load_json(response_path)
    quality = candidate["quality"]
    artifacts = {
        name: _resolve_artifact(record, server_output, f"candidate {name}")
        for name, record in quality["artifacts"].items()
    }

    staged_source = np.asarray(Image.open(source_path).convert("RGB"))
    staged_mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
    staged_exact = np.load(exact_path).astype(np.float64)
    rerendered_source, rerendered_mask, rerendered_exact, staged_parts, _ = (
        matrix._render_scene_arrays(spec, fixture)
    )
    rendered_face, named_parts = _rerender_part_masks(spec, fixture)
    provenance_checks = {
        "source_pixels_exact": bool(np.array_equal(staged_source, rerendered_source)),
        "selection_mask_exact": bool(
            np.array_equal(staged_mask, rerendered_mask >= 128)
            and np.array_equal(staged_mask, rendered_face)
        ),
        "exact_depth_exact": bool(
            np.array_equal(staged_exact.astype(np.float32), rerendered_exact)
        ),
        "exact_face_parts_exact": bool(
            staged_parts is not None
            and set(staged_parts) == set(named_parts)
            and all(
                np.array_equal(staged_parts[name], named_parts[name])
                for name in named_parts
            )
        ),
    }
    if not all(provenance_checks.values()):
        raise ValueError(f"Rerendered fixture does not match row {spec.row_id}")

    refined = np.load(artifacts["refined_depth"]).astype(np.float64)
    if refined.shape != staged_exact.shape:
        refined = matrix._resize_nan_aware(refined, staged_exact.shape)
    invert = bool(response.get("invert", False))
    exact_signal = 1.0 - staged_exact
    exact_min = float(np.nanmin(exact_signal))
    exact_span = float(np.nanmax(exact_signal) - exact_min)
    if not np.isfinite(exact_span) or exact_span <= 1e-8:
        raise ValueError(f"Exact depth has no usable span for row {spec.row_id}")
    exact_surface_mm = (exact_signal - exact_min) * (
        matrix.RELIEF_HEIGHT_MM / exact_span
    )
    predicted_signal = 1.0 - refined if invert else refined
    aligned, alignment = globally_align_face_surface(
        exact_surface_mm,
        predicted_signal,
        staged_mask,
        expected_scale_sign=1.0,
    )
    input_pitch = float(matrix.MAX_XY_SIZE_MM / max(spec.target_dimension - 1, 1))
    exact_metrics = localized_face_surface_metrics(
        exact_surface_mm,
        aligned,
        staged_mask,
        named_parts,
        sample_pitch_mm=input_pitch,
    )

    surface = np.load(artifacts["surface"]).astype(np.float64)
    reference_surface = np.load(artifacts["reference_surface"]).astype(np.float64)
    transform = response.get("relief_postprocess", {}).get("surface_grid_transform", {})
    emitted_face = transform_mask_to_emitted_grid(staged_mask, transform, surface.shape)
    emitted_parts = {
        name: transform_mask_to_emitted_grid(mask, transform, surface.shape)
        for name, mask in named_parts.items()
    }
    output_pitch = float(
        response.get("relief_postprocess", {}).get("mesh_sample_pitch_mm")
    )
    retention_metrics = localized_face_surface_metrics(
        reference_surface,
        surface,
        emitted_face,
        emitted_parts,
        sample_pitch_mm=output_pitch,
    )
    checks = {
        "provenance": bool(all(provenance_checks.values())),
        "exact_reconstruction": bool(exact_metrics["checks"]["passed"]),
        "postprocess_retention": bool(retention_metrics["checks"]["passed"]),
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "row_id": spec.row_id,
        "profile_name": spec.profile_name,
        "yaw_deg": float(spec.camera_yaw_deg),
        "target_dimension": int(spec.target_dimension),
        "input_sample_pitch_mm": input_pitch,
        "output_sample_pitch_mm": output_pitch,
        "global_exact_alignment": alignment,
        "exact_reference_mapping": {
            "method": "one-minus-depth-global-span-to-relief-mm",
            "relief_height_mm": float(matrix.RELIEF_HEIGHT_MM),
            "exact_signal_min": exact_min,
            "exact_signal_span": exact_span,
            "prediction_transform": "one-minus-depth" if invert else "linear",
        },
        "provenance_checks": provenance_checks,
        "exact_reconstruction": _compact_part_metrics(exact_metrics),
        "postprocess_retention": _compact_part_metrics(retention_metrics),
        "checks": checks,
    }


def analyze(
    run_dir: str | Path,
    *,
    asset_dir: str | Path = matrix.DEFAULT_ASSET_DIR,
) -> dict:
    selected_run = Path(run_dir).resolve()
    summary_path = selected_run / "summary.json"
    summary = _load_json(summary_path)
    if summary.get("run_kind") != "cc0_generated_live_relief_context_matrix":
        raise ValueError("Input is not a CC0 live relief-context matrix")
    if summary.get("privacy") != (
        "CC0 MakeHuman synthetic heads, generated procedural objects, and "
        "deterministic procedural scene materials only"
    ):
        raise ValueError("Input run does not have the expected public privacy contract")
    if summary.get("checks", {}).get("passed") is not True:
        raise ValueError("Input live matrix did not pass its original gates")
    fixture = load_makehuman_face_fixture(asset_dir)
    server_output = selected_run.parent
    rows = [
        _analyze_row(
            row,
            run_dir=selected_run,
            server_output=server_output,
            fixture=fixture,
        )
        for row in summary.get("rows", [])
        if row.get("render", {}).get("scene_kind") == "face"
    ]
    if len(rows) != 3:
        raise ValueError(f"Expected exactly three public face rows, got {len(rows)}")
    checks = {
        "three_face_rows": len(rows) == 3,
        "all_rerenders_exact": all(row["checks"]["provenance"] for row in rows),
        "all_exact_reconstruction_parts": all(
            row["checks"]["exact_reconstruction"] for row in rows
        ),
        "all_postprocess_parts": all(
            row["checks"]["postprocess_retention"] for row in rows
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_kind": "cc0_face_local_quality_analysis",
        "input_revision": summary.get("server_provenance", {}).get("revision"),
        "input_summary_sha256": _sha256(summary_path),
        "privacy": summary.get("privacy"),
        "metric_sources": [
            "backend/benchmark/face_relief_local_metrics.py",
            "backend/benchmark/face_part_metrics.py",
            "backend/pic_to_3d.py:_surface_lighting_agreement_metrics",
        ],
        "rows": rows,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--asset-dir", default=str(matrix.DEFAULT_ASSET_DIR))
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_dir, asset_dir=args.asset_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(report["rows"]),
                "passed": report["checks"]["passed"],
            },
            indent=2,
        )
    )
    if args.require_pass and not report["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
