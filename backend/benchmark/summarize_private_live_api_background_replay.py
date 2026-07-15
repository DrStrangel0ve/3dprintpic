"""Summarize a privacy-safe exact-input live API relief replay.

The response and every generated artifact must remain under ignored output.
Only scalar measurements, non-semantic scene labels, and implementation
provenance are allowed into the tracked aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from backend.benchmark.run_private_background_photo_detail_replay import (
    _assert_aggregate_privacy,
    _compact_background_preservation,
    _compact_cap,
    _finite,
    _require_ignored,
    _sha256,
    _validate_scene,
)
from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    _appearance_checks,
    _stl_heightfield_agreement,
)


EXPECTED_DETAIL_METHOD = "euclidean_inner_guard_smoothstep_to_full_gain"
EXPECTED_REQUEST = {
    "target_dimension": 512,
    "z_scale": 30.0,
    "max_xy_size": 128.0,
    "sigma": 0.35,
    "background_photo_detail_mm": 0.60,
    "selection_background_depth_ratio": 0.65,
}
PROVENANCE_PATHS = (
    "backend/main.py",
    "backend/pic_to_3d.py",
    "backend/face_depth_refinement.py",
)
EXPECTED_POSTED_FORM_FIELDS = {
    "selection_job_id",
    "target_dimension",
    "z_scale",
    "max_xy_size",
    "sigma",
}
OMITTED_BACKGROUND_FIELDS = {
    "background_photo_detail_mm",
    "selection_background_depth_ratio",
}


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_provenance(repository: Path, expected_revision: str) -> dict:
    revision = _git(repository, "rev-parse", "HEAD").lower()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if revision != expected_revision.lower():
        raise ValueError(
            f"Clean API repository revision mismatch: expected {expected_revision}, got {revision}"
        )
    if status:
        raise ValueError("Clean API repository has uncommitted non-ignored changes")
    return {
        "available": True,
        "clean": True,
        "revision": revision,
        "paths": list(PROVENANCE_PATHS),
        "status": [],
    }


def _require_job_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Live API response has no valid private job identifier")
    return value


def _same_file(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and _sha256(left) == _sha256(right)


def _request_checks(metadata: dict, response: dict) -> dict:
    observed = {
        "target_dimension": metadata.get("target_dimension"),
        "z_scale": metadata.get("z_scale"),
        "max_xy_size": metadata.get("max_xy_size"),
        "sigma": metadata.get("sigma"),
        "background_photo_detail_mm": response.get("background_photo_detail_mm"),
        "selection_background_depth_ratio": response.get(
            "selection_background_depth_ratio"
        ),
    }
    checks = {}
    for key, expected in EXPECTED_REQUEST.items():
        value = observed.get(key)
        try:
            checks[key] = abs(float(value) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            checks[key] = False
    checks["response_metadata_match"] = all(
        response.get(key) == metadata.get(key)
        for key in (
            "background_photo_detail_mm",
            "selection_background_depth_ratio",
            "target_dimension",
            "relief_sample_pitch_mm",
        )
    )
    checks["passed"] = all(checks.values())
    return checks


def _request_record_checks(record: dict, response_path: Path, job_id: str) -> dict:
    posted = set(record.get("posted_form_fields", []))
    omitted = set(record.get("omitted_background_fields", []))
    checks = {
        "schema": record.get("schema_version") == 1,
        "endpoint": record.get("endpoint") == "/process_image",
        "http_status": record.get("http_status") == 200,
        "response_hash": record.get("response_sha256") == _sha256(response_path),
        "response_job": record.get("response_job_id") == job_id,
        "exact_posted_fields": posted == EXPECTED_POSTED_FORM_FIELDS,
        "background_fields_omitted": omitted == OMITTED_BACKGROUND_FIELDS,
        "omitted_not_posted": not bool(posted & OMITTED_BACKGROUND_FIELDS),
    }
    checks["passed"] = all(checks.values())
    return checks


def _finite_between(value: object, minimum: object, maximum: object) -> bool:
    numeric = _finite(value)
    lower = _finite(minimum)
    upper = _finite(maximum)
    return (
        numeric is not None
        and lower is not None
        and upper is not None
        and lower <= numeric <= upper
    )


def _independent_height_checks(compression: dict) -> dict:
    quality = compression.get("quality_gates", {})
    detail = compression.get("detail_preservation", {})
    checks = {
        "detail_correlation": (
            _finite(detail.get("correlation")) is not None
            and _finite(quality.get("minimum_detail_correlation")) is not None
            and float(detail["correlation"])
            >= float(quality["minimum_detail_correlation"])
        ),
        "detail_rms_retention": _finite_between(
            detail.get("rms_retention"),
            quality.get("minimum_detail_rms_retention"),
            quality.get("maximum_detail_rms_retention"),
        ),
        "component_detail_correlation": (
            _finite(detail.get("minimum_component_correlation")) is not None
            and _finite(quality.get("minimum_detail_correlation")) is not None
            and float(detail["minimum_component_correlation"])
            >= float(quality["minimum_detail_correlation"])
        ),
        "component_detail_rms_retention": _finite_between(
            detail.get("minimum_component_rms_retention"),
            quality.get("minimum_detail_rms_retention"),
            quality.get("maximum_detail_rms_retention"),
        ),
        "height_span_ratio": _finite_between(
            compression.get("height_span_ratio"),
            quality.get("minimum_height_span_ratio"),
            quality.get("maximum_height_span_ratio"),
        ),
        "correction_span_ratio": (
            _finite(compression.get("correction_span_ratio")) is not None
            and _finite(quality.get("maximum_correction_span_ratio")) is not None
            and float(compression["correction_span_ratio"])
            <= float(quality["maximum_correction_span_ratio"])
        ),
        "edge_p99_ratio": (
            _finite(compression.get("output_edge_ratio_p99")) is not None
            and _finite(quality.get("maximum_output_edge_p99_ratio")) is not None
            and float(compression["output_edge_ratio_p99"])
            <= float(quality["maximum_output_edge_p99_ratio"])
        ),
        "edge_max_ratio": (
            _finite(compression.get("output_edge_ratio_max")) is not None
            and _finite(quality.get("maximum_output_edge_ratio")) is not None
            and float(compression["output_edge_ratio_max"])
            <= float(quality["maximum_output_edge_ratio"])
        ),
    }
    checks["producer_quality_agrees"] = bool(quality.get("passed", False))
    checks["passed"] = all(checks.values())
    return checks


def _independent_background_checks(stats: dict) -> dict:
    localized = stats.get("localized_structure", {})
    checks = {
        "available": bool(stats.get("available", False)),
        "coverage": (
            _finite(stats.get("candidate_coverage_ratio")) is not None
            and _finite(stats.get("minimum_coverage_ratio")) is not None
            and float(stats["candidate_coverage_ratio"])
            >= float(stats["minimum_coverage_ratio"])
        ),
        "correlation": (
            _finite(stats.get("correlation")) is not None
            and _finite(stats.get("minimum_correlation")) is not None
            and float(stats["correlation"]) >= float(stats["minimum_correlation"])
        ),
        "rms_retention": _finite_between(
            stats.get("rms_retention"),
            stats.get("minimum_rms_retention"),
            stats.get("maximum_rms_retention"),
        ),
        "span_retention": _finite_between(
            stats.get("span_retention"),
            stats.get("minimum_span_retention"),
            stats.get("maximum_span_retention"),
        ),
        "gradient_correlation": (
            _finite(stats.get("gradient_correlation")) is not None
            and _finite(stats.get("minimum_gradient_correlation")) is not None
            and float(stats["gradient_correlation"])
            >= float(stats["minimum_gradient_correlation"])
        ),
        "gradient_rms_retention": _finite_between(
            stats.get("gradient_rms_retention"),
            stats.get("minimum_gradient_rms_retention"),
            stats.get("maximum_gradient_rms_retention"),
        ),
        "mean_shift": (
            _finite(stats.get("mean_shift_mm")) is not None
            and _finite(stats.get("maximum_mean_shift_mm")) is not None
            and abs(float(stats["mean_shift_mm"]))
            <= float(stats["maximum_mean_shift_mm"])
        ),
        "boundary_p99": (
            _finite(stats.get("output_boundary_jump_p99_mm")) is not None
            and _finite(stats.get("effective_boundary_jump_p99_limit_mm"))
            is not None
            and float(stats["output_boundary_jump_p99_mm"])
            <= float(stats["effective_boundary_jump_p99_limit_mm"])
        ),
        "boundary_max": (
            _finite(stats.get("output_boundary_jump_max_mm")) is not None
            and _finite(stats.get("effective_boundary_jump_max_limit_mm"))
            is not None
            and float(stats["output_boundary_jump_max_mm"])
            <= float(stats["effective_boundary_jump_max_limit_mm"])
        ),
        "localized_structure": bool(localized.get("passed", False)),
        "no_reported_failures": not bool(stats.get("quality_failures", [])),
        "producer_quality_agrees": bool(stats.get("passed", False)),
    }
    checks["passed"] = all(checks.values())
    return checks


def _independent_cap_checks(cap: dict) -> dict:
    tolerance = 1e-6
    checks = {
        "emission": bool(cap.get("emission_passed", False)),
        "far_background_flag": bool(cap.get("far_background_cap_passed", False)),
        "feasible_attachment_flag": bool(
            cap.get("feasible_attachment_constraints_passed", False)
        ),
        "zero_far_violation": (
            _finite(cap.get("far_background_cap_violation_mm")) is not None
            and float(cap["far_background_cap_violation_mm"]) <= tolerance
        ),
        "far_height_within_ceiling": (
            _finite(cap.get("far_background_max_mm")) is not None
            and _finite(cap.get("far_background_ceiling_mm")) is not None
            and float(cap["far_background_max_mm"])
            <= float(cap["far_background_ceiling_mm"]) + tolerance
        ),
        "feasible_jump_within_limit": (
            _finite(cap.get("feasible_attachment_jump_max_mm")) is not None
            and _finite(cap.get("attachment_step_limit_mm")) is not None
            and float(cap["feasible_attachment_jump_max_mm"])
            <= float(cap["attachment_step_limit_mm"]) + tolerance
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def _photo_detail_record(postprocess: dict) -> dict:
    detail = postprocess.get("background_photo_detail", {})
    checks = {
        "enabled": bool(detail.get("enabled", False)),
        "face_protected": bool(detail.get("face_protected", False)),
        "method": detail.get("protection_method") == EXPECTED_DETAIL_METHOD,
        "requested_detail": abs(float(detail.get("requested_detail_mm", -1)) - 0.60)
        <= 1e-9,
        "effective_detail": abs(float(detail.get("effective_detail_mm", -1)) - 0.60)
        <= 1e-9,
        "halo": abs(float(detail.get("protection_halo_mm", -1)) - 5.0) <= 1e-9,
        "zero_guard": abs(float(detail.get("protection_zero_guard_mm", -1)) - 2.0)
        <= 1e-9,
        "finite_positive_scale": (
            _finite(detail.get("detail_scale")) is not None
            and float(detail.get("detail_scale")) > 0
        ),
        "nonempty_background": (
            _finite(detail.get("background_coverage_ratio")) is not None
            and 0 < float(detail.get("background_coverage_ratio")) <= 1
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "enabled": bool(detail.get("enabled", False)),
        "method": detail.get("protection_method"),
        "requested_detail_mm": _finite(detail.get("requested_detail_mm")),
        "effective_detail_mm": _finite(detail.get("effective_detail_mm")),
        "protection_halo_mm": _finite(detail.get("protection_halo_mm")),
        "protection_zero_guard_mm": _finite(
            detail.get("protection_zero_guard_mm")
        ),
        "input_sample_pitch_mm": _finite(detail.get("input_sample_pitch_mm")),
        "detail_scale": _finite(detail.get("detail_scale")),
        "background_coverage_ratio": _finite(
            detail.get("background_coverage_ratio")
        ),
        "checks": checks,
    }


def _face_quality_record(postprocess: dict) -> dict:
    appearance = postprocess.get("surface_appearance_agreement", {}).get("face", {})
    appearance_checks = _appearance_checks(appearance, FACE_APPEARANCE_GATES)
    compression = (
        postprocess.get("face_height_stabilization", {})
        .get("gradient_compression", {})
    )
    detail = compression.get("detail_preservation", {})
    independent_height_checks = _independent_height_checks(compression)
    checks = {
        "appearance": bool(appearance_checks.get("passed", False)),
        "height_quality": bool(independent_height_checks.get("passed", False)),
        "all_components_measured": (
            int(appearance.get("component_count", 0)) > 0
            and int(appearance.get("measured_component_count", -1))
            == int(appearance.get("component_count", 0))
            and int(appearance.get("unavailable_component_count", -1)) == 0
        ),
        "full_coverage": _finite(appearance.get("candidate_coverage_ratio")) == 1.0,
    }
    checks["passed"] = all(checks.values())
    return {
        "normal_mean_cosine": _finite(appearance.get("normal_mean_cosine")),
        "normal_p05_cosine": _finite(appearance.get("normal_p05_cosine")),
        "normal_angle_p95_deg": _finite(appearance.get("normal_angle_p95_deg")),
        "minimum_lighting_correlation": _finite(
            appearance.get("minimum_lighting_correlation")
        ),
        "maximum_lighting_mae": _finite(appearance.get("maximum_lighting_mae")),
        "minimum_lighting_rms_retention": _finite(
            appearance.get("minimum_lighting_rms_retention")
        ),
        "maximum_lighting_rms_retention": _finite(
            appearance.get("maximum_lighting_rms_retention")
        ),
        "candidate_coverage_ratio": _finite(
            appearance.get("candidate_coverage_ratio")
        ),
        "component_count": int(appearance.get("component_count", 0)),
        "measured_component_count": int(
            appearance.get("measured_component_count", 0)
        ),
        "detail_correlation": _finite(detail.get("correlation")),
        "detail_rms_retention": _finite(detail.get("rms_retention")),
        "minimum_component_detail_correlation": _finite(
            detail.get("minimum_component_correlation")
        ),
        "minimum_component_detail_rms_retention": _finite(
            detail.get("minimum_component_rms_retention")
        ),
        "height_span_ratio": _finite(compression.get("height_span_ratio")),
        "appearance_gates": dict(FACE_APPEARANCE_GATES),
        "height_checks": independent_height_checks,
        "checks": checks,
    }


def _topology_record(diagnostics: dict) -> dict:
    checks = {
        "exists": bool(diagnostics.get("stl_exists", False)),
        "watertight": bool(diagnostics.get("stl_is_watertight", False)),
        "volume": bool(diagnostics.get("stl_is_volume", False)),
        "manifold": bool(diagnostics.get("stl_is_manifold", False)),
        "winding": bool(diagnostics.get("stl_winding_consistent", False)),
        "single_component": int(diagnostics.get("stl_component_count", 0)) == 1,
        "zero_degenerates": int(diagnostics.get("stl_degenerate_face_count", -1))
        == 0,
        "positive_volume": bool(diagnostics.get("stl_positive_volume", False)),
        "bbox_has_volume": bool(diagnostics.get("stl_bbox_has_volume", False)),
    }
    checks["passed"] = all(checks.values())
    return {
        "face_count": int(diagnostics.get("stl_faces", 0)),
        "vertex_count": int(diagnostics.get("stl_vertices", 0)),
        "component_count": int(diagnostics.get("stl_component_count", 0)),
        "nonmanifold_edge_count": int(
            diagnostics.get("stl_nonmanifold_edge_count", 0)
        ),
        "degenerate_face_count": int(
            diagnostics.get("stl_degenerate_face_count", 0)
        ),
        "volume_fill_ratio": _finite(diagnostics.get("stl_volume_fill_ratio")),
        "z_range_mm": _finite(diagnostics.get("stl_z_range")),
        "checks": checks,
    }


def summarize(
    scene_config: str | Path,
    response_json: str | Path,
    request_record: str | Path,
    clean_repository: str | Path,
    direct_context_depth: str | Path,
    direct_reference_surface: str | Path,
    expected_revision: str,
    *,
    scene_label: str = "scene-01",
) -> dict:
    repository = Path(__file__).resolve().parents[2]
    clean_repository = Path(clean_repository).resolve()
    response_path = Path(response_json).resolve()
    request_record_path = Path(request_record).resolve()
    direct_context_path = Path(direct_context_depth).resolve()
    direct_reference_path = Path(direct_reference_surface).resolve()
    config_path = Path(scene_config).resolve()
    _require_ignored(config_path, repository)
    _require_ignored(direct_context_path, repository)
    _require_ignored(direct_reference_path, repository)
    _require_ignored(response_path, clean_repository)
    _require_ignored(request_record_path, clean_repository)

    config = _load_json(config_path)
    matches = [
        scene
        for scene in config.get("scenes", [])
        if scene.get("label") == scene_label
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one private scene labeled {scene_label}")
    scene = _validate_scene(matches[0], repository)
    response = _load_json(response_path)
    job_id = _require_job_id(response.get("job_id"))
    record = _load_json(request_record_path)
    request_record_checks = _request_record_checks(record, response_path, job_id)
    if not request_record_checks["passed"]:
        raise ValueError("Private live API request record does not bind the response")
    job_dir = clean_repository / "backend" / "output" / job_id
    metadata_path = job_dir / "metadata.json"
    diagnostics_path = job_dir / "diagnostics.json"
    stl_path = job_dir / "output_model.stl"
    surface_path = job_dir / "output_surface.npy"
    reference_path = job_dir / "output_reference_surface.npy"
    for path in (metadata_path, diagnostics_path, stl_path, surface_path, reference_path):
        if not path.is_file():
            raise ValueError(f"Live API output is missing {path.name}")
        _require_ignored(path, clean_repository)

    metadata = _load_json(metadata_path)
    diagnostics = _load_json(diagnostics_path)
    if metadata.get("job_id") != job_id or diagnostics.get("job_id") != job_id:
        raise ValueError("Live API response and artifact job identifiers disagree")
    runtime_provenance = (
        metadata.get("runtime", {}).get("implementation_provenance", {})
    )
    runtime_provenance_checks = {
        "available": bool(runtime_provenance.get("available", False)),
        "clean": runtime_provenance.get("clean") is True,
        "revision": runtime_provenance.get("revision") == expected_revision.lower(),
        "source": runtime_provenance.get("source") in {"git-worktree", "environment"},
        "empty_status": not bool(runtime_provenance.get("status", [])),
    }
    runtime_provenance_checks["passed"] = all(runtime_provenance_checks.values())

    selection_metadata = _load_json(Path(scene["selection_metadata"]))
    selection_id = _require_job_id(selection_metadata.get("job_id"))
    clean_selection = clean_repository / "backend" / "output" / "selection" / selection_id
    source_exact = _same_file(clean_selection / "source.png", Path(scene["source_image"]))
    selection_exact = _same_file(
        clean_selection / "selection_mask.png", Path(scene["selection_mask"])
    )
    refined_depth_exact = _same_file(
        job_dir / "output_depth_data_face_refined.npy", Path(scene["depth_npy"])
    )
    context_depth_exact = _same_file(
        job_dir / "output_depth_data_selected_context.npy", direct_context_path
    )
    reference_surface_exact = _same_file(reference_path, direct_reference_path)
    face_region_exact = _same_file(
        job_dir / "output_face_refinement_region.png",
        Path(scene["face_region_mask"]),
    )
    feature_weight_exact = _same_file(
        job_dir / "output_face_refinement_weight.png",
        Path(scene["feature_weight_mask"]),
    )
    exact_input_checks = {
        "pinned_private_bundle_verified": True,
        "source_exact": source_exact,
        "selection_mask_exact": selection_exact,
        "refined_depth_exact": refined_depth_exact,
        "composed_context_depth_exact": context_depth_exact,
        "reference_surface_exact": reference_surface_exact,
        "face_region_exact": face_region_exact,
        "feature_weight_exact": feature_weight_exact,
        "effective_model_exact": (
            metadata.get("depth_model") == scene.get("expected_depth_model")
            and metadata.get("depth_fallback_reason") is None
        ),
    }
    exact_input_checks["passed"] = all(exact_input_checks.values())

    postprocess = metadata.get("relief_postprocess", {})
    request_checks = _request_checks(metadata, response)
    photo_detail = _photo_detail_record(postprocess)
    face_quality = _face_quality_record(postprocess)
    raw_background = postprocess.get("background_depth_preservation", {})
    background = _compact_background_preservation(raw_background)
    background["checks"] = _independent_background_checks(raw_background)
    raw_cap = postprocess.get("selection_background_physical_cap", {})
    cap = _compact_cap(raw_cap)
    cap["checks"] = _independent_cap_checks(raw_cap)
    topology = _topology_record(diagnostics)
    shell = _stl_heightfield_agreement(
        stl_path,
        surface_path,
        expected_max_xy_size_mm=128.0,
    )
    shell_record = {
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
    surface = np.load(surface_path)
    reference = np.load(reference_path)
    surface_checks = {
        "same_shape": surface.shape == reference.shape,
        "finite_surface": bool(np.all(np.isfinite(surface))),
        "finite_reference": bool(np.all(np.isfinite(reference))),
        "reference_emitted": bool(
            postprocess.get("reference_surface", {}).get("emitted", False)
        ),
    }
    surface_checks["passed"] = all(surface_checks.values())

    provenance = _clean_provenance(clean_repository, expected_revision)
    checks = {
        "request_contract": bool(request_checks["passed"]),
        "request_record": bool(request_record_checks["passed"]),
        "exact_input": bool(exact_input_checks["passed"]),
        "photo_detail": bool(photo_detail["checks"]["passed"]),
        "face_quality": bool(face_quality["checks"]["passed"]),
        "background_preservation": bool(background["checks"]["passed"]),
        "physical_cap": bool(cap["checks"]["passed"]),
        "topology": bool(topology["checks"]["passed"]),
        "surface_arrays": bool(surface_checks["passed"]),
        "serialized_shell": bool(shell_record.get("passed", False)),
        "implementation_provenance_clean": bool(provenance["clean"]),
        "runtime_provenance_bound": bool(runtime_provenance_checks["passed"]),
    }
    checks["passed"] = all(checks.values())
    summary = {
        "schema_version": 1,
        "run_kind": "retained_private_exact_input_live_api_background_replay",
        "privacy": (
            "aggregate metrics only; source images, masks, paths, hashes, job "
            "identifiers, surfaces, and meshes remain in gitignored output"
        ),
        "implementation_provenance": provenance,
        "matrix": {
            "scene_label": scene_label,
            "relief_height_mm": 30.0,
            "physical_size_mm": 128.0,
            "target_dimension": 512,
            "sigma": 0.35,
            "background_fields_omitted_by_launcher": bool(
                request_record_checks["background_fields_omitted"]
                and request_record_checks["omitted_not_posted"]
            ),
            "observed_background_photo_detail_mm": 0.60,
            "observed_selection_background_depth_ratio": 0.65,
            "runtime_seconds": _finite(response.get("timings", {}).get("total_seconds")),
        },
        "request_checks": request_checks,
        "request_record_checks": request_record_checks,
        "runtime_provenance_checks": runtime_provenance_checks,
        "exact_input_checks": exact_input_checks,
        "photo_detail": photo_detail,
        "face_quality": face_quality,
        "background_preservation": background,
        "physical_cap": cap,
        "topology": topology,
        "surface_checks": surface_checks,
        "serialized_shell": shell_record,
        "checks": checks,
    }
    _assert_aggregate_privacy(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--scene-label", default="scene-01")
    parser.add_argument("--response-json", required=True)
    parser.add_argument("--request-record", required=True)
    parser.add_argument("--clean-repository", required=True)
    parser.add_argument("--direct-context-depth", required=True)
    parser.add_argument("--direct-reference-surface", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--aggregate-summary", required=True)
    args = parser.parse_args()
    summary = summarize(
        args.scene_config,
        args.response_json,
        args.request_record,
        args.clean_repository,
        args.direct_context_depth,
        args.direct_reference_surface,
        args.expected_revision,
        scene_label=args.scene_label,
    )
    output = Path(args.aggregate_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["checks"], indent=2))


if __name__ == "__main__":
    main()
