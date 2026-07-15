"""Create privacy-safe evidence from full generated live-relief summaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path


EYEWEAR_ROW_ID = "eyewear_overhead_panel_384"
EXPECTED_ROW_IDS = (
    "small_side_lit_shelves_256",
    EYEWEAR_ROW_ID,
    "strong_turn_layered_384",
    "multilobe_object_shelves_384",
)
EXPECTED_PRODUCER_PATHS = (
    "backend/benchmark/assets/makehuman_cc0_heads/asset.json",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/run_cc0_live_face_variation_matrix.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/run_private_background_photo_detail_replay.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
    "backend/benchmark/summarize_private_live_api_background_replay.py",
)
EXPECTED_PRIVACY = (
    "CC0 MakeHuman synthetic heads, generated procedural objects, and "
    "deterministic procedural scene materials only"
)
EXPECTED_FIXTURE_SHA256 = (
    "91d94d989a017c61cca2b27dddc14e78ad9f045ce6ca449af0b6d482d1622015"
)
EXPECTED_ROW_RENDER = {
    "small_side_lit_shelves_256": ("face", "deep_shelves", "side_right"),
    EYEWEAR_ROW_ID: ("face", "layered_studio", "overhead"),
    "strong_turn_layered_384": ("face", "layered_studio", "soft_left"),
    "multilobe_object_shelves_384": ("object", "deep_shelves", "side_right"),
}
REQUIRED_FULL_CHECKS = {
    "all_behavioral_rows_passed",
    "passed",
    "producer_matches_server_revision",
    "producer_unchanged_during_run",
    "raw_artifacts_beneath_ignored_output",
    "requested_rows_executed_once",
    "server_repository_clean",
    "server_revision_unchanged",
    "server_runtime_matches_checkout",
}
EXPECTED_ROW_CHECKS = {
    "baseline",
    "candidate",
    "input_background_depth_span",
    "paired_background_detail",
    "passed",
}
COMMON_QUALITY_CHECKS = {
    "background_appearance",
    "background_depth",
    "boundary_shape",
    "exact_stl_shell",
    "exact_subject_depth",
    "finite_surface_contract",
    "occlusion_deoccluded",
    "passed",
    "physical_cap",
    "refinement_applied",
    "semantic_appearance",
    "subject_refinement_route",
    "topology",
}
FACE_QUALITY_CHECKS = COMMON_QUALITY_CHECKS | {"validated_human_face_refined"}
OBJECT_QUALITY_CHECKS = COMMON_QUALITY_CHECKS | {
    "generic_fallback_accounted",
    "generic_selection_refined",
    "no_false_human_face",
    "refined_region_total",
}


def _load_json(path: str | Path) -> dict:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved.name}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing finite metric: {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Missing finite metric: {label}")
    return number


def _strict_bool(value, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Expected a boolean for {label}")
    return value


def _hex_digest(value, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
    ):
        raise ValueError(f"Expected a lowercase hexadecimal digest for {label}")
    return value


def _rows(summary: dict) -> list[dict]:
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Full summary has no rows")
    identifiers = [row.get("row_id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(rows) or len(set(identifiers)) != len(rows):
        raise ValueError("Full summary rows are missing unique identifiers")
    return rows


def _validate_public_summary_contract(summary: dict) -> None:
    if summary.get("schema_version") != 1:
        raise ValueError("Full summary has an unsupported schema")
    if summary.get("run_kind") != "cc0_generated_live_relief_context_matrix":
        raise ValueError("Full summary is not the public CC0 context matrix")
    if summary.get("privacy") != EXPECTED_PRIVACY:
        raise ValueError("Full summary has an unexpected privacy declaration")
    fixture = summary.get("fixture")
    if not isinstance(fixture, dict):
        raise ValueError("Full summary has no fixture provenance")
    fixture_record = fixture.get("fixture", {})
    source = fixture.get("source", {})
    if (
        fixture.get("schema_version") != 1
        or fixture.get("license") != "CC0-1.0"
        or fixture_record.get("sha256") != EXPECTED_FIXTURE_SHA256
        or source.get("repository") != "https://github.com/makehumancommunity/makehuman"
    ):
        raise ValueError("Full summary has unexpected fixture provenance")
    _hex_digest(source.get("commit"), 40, "fixture source commit")
    if tuple(row["row_id"] for row in _rows(summary)) != EXPECTED_ROW_IDS:
        raise ValueError("Full summary does not contain the exact public matrix")
    coverage = summary.get("matrix", {}).get("coverage", {})
    if (
        tuple(coverage.get("row_ids", [])) != EXPECTED_ROW_IDS
        or set(coverage.get("background_profiles", []))
        != {"deep_shelves", "layered_studio"}
        or set(coverage.get("lighting_profiles", []))
        != {"overhead", "side_right", "soft_left"}
        or set(coverage.get("scene_kinds", [])) != {"face", "object"}
    ):
        raise ValueError("Full summary coverage is not the exact public matrix")
    for row in _rows(summary):
        render = row.get("render", {})
        actual = (
            render.get("scene_kind"),
            render.get("background_profile"),
            render.get("lighting_profile"),
        )
        if actual != EXPECTED_ROW_RENDER[row["row_id"]]:
            raise ValueError("Full summary contains an unexpected public row profile")


def _row(summary: dict, row_id: str) -> dict:
    matches = [row for row in _rows(summary) if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row named {row_id}")
    return matches[0]


def _quality(row: dict) -> dict:
    try:
        quality = row["variants"]["candidate"]["quality"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Row has no candidate quality record") from exc
    if not isinstance(quality, dict):
        raise ValueError("Row has no candidate quality record")
    return quality


def _verify_response(row: dict, response_path: str | Path) -> dict:
    try:
        expected = row["variants"]["candidate"]["response_artifact"]["sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Row has no candidate response checksum") from exc
    if not isinstance(expected, str) or _sha256(response_path) != expected:
        raise ValueError("Candidate response checksum does not match the full summary")
    return _load_json(response_path)


def _eyewear_detection(response: dict) -> dict:
    try:
        faces = response["face_refinement"]["faces"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Response has no face refinement records") from exc
    detections = []
    for face in faces:
        if not isinstance(face, dict):
            continue
        eyewear = face.get("eyewear_deocclusion", {})
        detection = eyewear.get("detection", {}) if isinstance(eyewear, dict) else {}
        if isinstance(detection, dict) and detection.get("method"):
            detections.append((eyewear, detection))
    if len(detections) != 1:
        raise ValueError("Expected exactly one eyewear detection record")
    eyewear, detection = detections[0]
    return {
        "detected": _strict_bool(detection.get("enabled"), "eyewear detection"),
        "deoccluded": _strict_bool(eyewear.get("enabled"), "eyewear deocclusion"),
        "dark_coverage_ratio": _finite(
            detection.get("dark_coverage_ratio"), "dark coverage"
        ),
        "component_coverage_ratio": _finite(
            detection.get("component_coverage_ratio"), "component coverage"
        ),
        "component_width_ratio": _finite(
            detection.get("component_width_ratio"), "component width"
        ),
        "minimum_dark_coverage_ratio": _finite(
            detection.get("minimum_dark_coverage_ratio"), "dark coverage gate"
        ),
        "minimum_component_coverage_ratio": _finite(
            detection.get("minimum_component_coverage_ratio"),
            "component coverage gate",
        ),
        "minimum_component_width_ratio": _finite(
            detection.get("minimum_component_width_ratio"), "component width gate"
        ),
    }


def _effective_response_configuration(response: dict) -> dict:
    return {
        "background_photo_detail_mm": _finite(
            response.get("background_photo_detail_mm"),
            "effective background photo detail",
        ),
        "selection_background_depth_ratio": _finite(
            response.get("selection_background_depth_ratio"),
            "effective selection background depth ratio",
        ),
        "relief_height_mm": _finite(response.get("z_scale"), "response relief height"),
        "max_xy_size_mm": _finite(response.get("max_xy_size"), "response size"),
        "target_dimension": int(response.get("target_dimension", -1)),
    }


def _validated_producer_provenance(summary: dict, revision: str) -> list[dict]:
    producer = summary.get("producer_provenance")
    final_producer = summary.get("final_producer_provenance")
    if not isinstance(producer, dict) or not isinstance(final_producer, dict):
        raise ValueError("Full summary has no producer provenance")
    if producer != final_producer:
        raise ValueError("Producer provenance changed during the full run")
    if not _strict_bool(
        producer.get("available"), "producer availability"
    ) or not _strict_bool(producer.get("clean"), "producer cleanliness"):
        raise ValueError("Producer provenance is unavailable or dirty")
    if producer.get("revision") != revision or producer.get("status") != []:
        raise ValueError("Producer provenance does not match the server revision")
    files = producer.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Producer provenance has no source records")
    sanitized = []
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("Producer source record is invalid")
        path = record.get("path")
        checksum = record.get("sha256")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError("Producer source record is invalid")
        _hex_digest(checksum, 64, "producer source")
        sanitized.append({"path": path, "sha256": checksum})
    if tuple(sorted(record["path"] for record in sanitized)) != EXPECTED_PRODUCER_PATHS:
        raise ValueError("Producer source records do not match the public matrix")
    return sorted(sanitized, key=lambda record: record["path"])


def _validate_run_provenance(summary: dict, revision: str) -> list[dict]:
    for key in ("server_provenance", "server_runtime_provenance"):
        provenance = summary.get(key)
        if not isinstance(provenance, dict):
            raise ValueError(f"Full summary has no {key}")
        if (
            not _strict_bool(provenance.get("available"), f"{key}.available")
            or not _strict_bool(provenance.get("clean"), f"{key}.clean")
            or provenance.get("revision") != revision
            or provenance.get("status") != []
        ):
            raise ValueError(f"Full summary has invalid {key}")
    return _validated_producer_provenance(summary, revision)


def _strict_checks(summary: dict, *, expected_passed: bool) -> dict:
    checks = summary.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError("Full summary has no checks")
    if set(checks) != REQUIRED_FULL_CHECKS:
        raise ValueError("Full summary is missing required checks")
    strict = {
        key: _strict_bool(value, f"checks.{key}") for key, value in checks.items()
    }
    if strict.get("passed") is not expected_passed:
        raise ValueError("Full summary pass status is inconsistent")
    if strict.get("all_behavioral_rows_passed") is not expected_passed:
        raise ValueError("Full summary behavioral status is inconsistent")
    if not all(
        value
        for key, value in strict.items()
        if key not in {"passed", "all_behavioral_rows_passed"}
    ):
        raise ValueError("Full summary provenance checks did not pass")
    return strict


def _strict_check_mapping(record: object, label: str, expected_keys: set[str]) -> dict:
    if not isinstance(record, dict) or not record:
        raise ValueError(f"Missing {label}")
    if set(record) != expected_keys:
        raise ValueError(f"{label} does not contain the exact required gates")
    return {key: _strict_bool(value, f"{label}.{key}") for key, value in record.items()}


def _validate_final_row_checks(row: dict) -> None:
    row_checks = _strict_check_mapping(
        row.get("checks"), "final row checks", EXPECTED_ROW_CHECKS
    )
    if not all(row_checks.values()):
        raise ValueError("Final row contains a failed nested check")
    for variant in ("baseline", "candidate"):
        quality_keys = (
            OBJECT_QUALITY_CHECKS
            if row.get("render", {}).get("scene_kind") == "object"
            else FACE_QUALITY_CHECKS
        )
        quality_checks = _strict_check_mapping(
            row.get("variants", {}).get(variant, {}).get("quality", {}).get("checks"),
            f"final {variant} quality checks",
            quality_keys,
        )
        if not all(quality_checks.values()):
            raise ValueError("Final row contains a failed nested check")


def _validate_initial_row_checks(row: dict, *, eyewear_failure: bool) -> None:
    row_checks = _strict_check_mapping(
        row.get("checks"), "initial row checks", EXPECTED_ROW_CHECKS
    )
    expected_row_failures = (
        {"baseline", "candidate", "passed"} if eyewear_failure else set()
    )
    if {key for key, value in row_checks.items() if not value} != expected_row_failures:
        raise ValueError("Initial row failure is not isolated to eyewear")
    for variant in ("baseline", "candidate"):
        quality_keys = (
            OBJECT_QUALITY_CHECKS
            if row.get("render", {}).get("scene_kind") == "object"
            else FACE_QUALITY_CHECKS
        )
        quality_checks = _strict_check_mapping(
            row.get("variants", {}).get(variant, {}).get("quality", {}).get("checks"),
            f"initial {variant} quality checks",
            quality_keys,
        )
        expected_quality_failures = (
            {"occlusion_deoccluded", "passed"} if eyewear_failure else set()
        )
        if {
            key for key, value in quality_checks.items() if not value
        } != expected_quality_failures:
            raise ValueError("Initial eyewear failure is not isolated to occlusion")


def _compact_row(row: dict) -> dict:
    scene = row.get("scene", {})
    render = row.get("render", {})
    quality = _quality(row)
    exact = quality.get("exact_subject_depth", {})
    appearance = quality.get("appearance", {})
    background = quality.get("background_depth", {})
    cap = quality.get("physical_cap", {})
    boundary = quality.get("boundary_shape", {})
    topology = quality.get("topology", {})
    pair = row.get("pair_quality", {}).get("background_photo_detail", {})
    checks = row.get("checks", {})
    compact = {
        "row_id": row.get("row_id"),
        "scene_kind": render.get("scene_kind"),
        "target_dimension": int(scene.get("target_dimension")),
        "yaw_deg": _finite(scene.get("camera_yaw_deg"), "camera yaw"),
        "elevation_deg": _finite(scene.get("camera_elevation_deg"), "camera elevation"),
        "background_profile": render.get("background_profile"),
        "lighting_profile": render.get("lighting_profile"),
        "selection_bbox_ratio": [
            _finite(render.get("selection_bbox_width_ratio"), "selection width"),
            _finite(render.get("selection_bbox_height_ratio"), "selection height"),
        ],
        "input_background_depth_span": _finite(
            render.get("background_depth_span"), "input background depth span"
        ),
        "detected_faces": int(quality.get("detected_faces", -1)),
        "refined_faces": int(quality.get("refined_faces", -1)),
        "selected_detail_regions": int(quality.get("selected_detail_regions", -1)),
        "exact_shape_correlation": _finite(
            exact.get("shape_correlation"), "exact shape correlation"
        ),
        "exact_gradient_correlation": _finite(
            exact.get("gradient_correlation"), "exact gradient correlation"
        ),
        "exact_normalized_rmse": _finite(
            exact.get("normalized_rmse"), "exact normalized RMSE"
        ),
        "appearance_normal_mean_cosine": _finite(
            appearance.get("normal_mean_cosine"), "appearance normal cosine"
        ),
        "appearance_minimum_lighting_correlation": _finite(
            appearance.get("minimum_lighting_correlation"),
            "appearance lighting correlation",
        ),
        "background_depth_correlation": _finite(
            background.get("correlation"), "background depth correlation"
        ),
        "background_gradient_correlation": _finite(
            background.get("gradient_correlation"),
            "background gradient correlation",
        ),
        "background_rms_retention": _finite(
            background.get("rms_retention"), "background RMS retention"
        ),
        "paired_source_detail_correlation": _finite(
            pair.get("source_detail_correlation_intended_background"),
            "paired source detail correlation",
        ),
        "paired_source_aligned_capture_ratio": _finite(
            pair.get("source_aligned_capture_ratio"), "source-aligned capture"
        ),
        "paired_face_interior_p99_change_mm": _finite(
            pair.get("face_interior_p99_change_mm"), "face interior movement"
        ),
        "paired_attachment_boundary_max_change_mm": _finite(
            pair.get("max_attachment_boundary_change_mm"),
            "attachment boundary movement",
        ),
        "far_background_cap_violation_mm": _finite(
            cap.get("far_background_cap_violation_mm"), "far cap violation"
        ),
        "feasible_attachment_jump_max_mm": _finite(
            cap.get("feasible_attachment_jump_max_mm"), "attachment jump"
        ),
        "attachment_constraint_conflicts": int(
            cap.get("attachment_constraint_conflicts", -1)
        ),
        "boundary_p99_to_limit_ratio": _finite(
            boundary.get("output_p99_to_limit_ratio"), "boundary p99 ratio"
        ),
        "boundary_max_to_limit_ratio": _finite(
            boundary.get("output_max_to_limit_ratio"), "boundary max ratio"
        ),
        "topology_passed": _strict_bool(
            topology.get("checks", {}).get("passed"), "topology pass status"
        ),
        "exact_stl_shell_passed": _strict_bool(
            quality.get("stl_heightfield_agreement", {}).get("passed"),
            "STL shell pass status",
        ),
        "passed": _strict_bool(checks.get("passed"), "row pass status"),
    }
    occlusion = quality.get("occlusion_handling", {})
    if _strict_bool(occlusion.get("required"), "occlusion requirement"):
        compact["eyewear_detected"] = _strict_bool(
            occlusion.get("eyewear_detected"), "eyewear detected status"
        )
        compact["eyewear_deoccluded_faces"] = int(occlusion.get("deoccluded_faces", 0))
    if compact["scene_kind"] not in {"face", "object"}:
        raise ValueError("Row has no supported scene kind")
    return compact


def _aggregate(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "face_rows": sum(row["scene_kind"] == "face" for row in rows),
        "object_rows": sum(row["scene_kind"] == "object" for row in rows),
        "passed_rows": sum(row["passed"] for row in rows),
        "minimum_input_background_depth_span": min(
            row["input_background_depth_span"] for row in rows
        ),
        "maximum_input_background_depth_span": max(
            row["input_background_depth_span"] for row in rows
        ),
        "minimum_exact_shape_correlation": min(
            row["exact_shape_correlation"] for row in rows
        ),
        "minimum_exact_gradient_correlation": min(
            row["exact_gradient_correlation"] for row in rows
        ),
        "maximum_exact_normalized_rmse": max(
            row["exact_normalized_rmse"] for row in rows
        ),
        "minimum_appearance_lighting_correlation": min(
            row["appearance_minimum_lighting_correlation"] for row in rows
        ),
        "minimum_background_depth_correlation": min(
            row["background_depth_correlation"] for row in rows
        ),
        "minimum_background_gradient_correlation": min(
            row["background_gradient_correlation"] for row in rows
        ),
        "minimum_background_rms_retention": min(
            row["background_rms_retention"] for row in rows
        ),
        "maximum_background_rms_retention": max(
            row["background_rms_retention"] for row in rows
        ),
        "minimum_paired_source_detail_correlation": min(
            row["paired_source_detail_correlation"] for row in rows
        ),
        "minimum_paired_source_aligned_capture_ratio": min(
            row["paired_source_aligned_capture_ratio"] for row in rows
        ),
        "maximum_face_interior_p99_change_mm": max(
            row["paired_face_interior_p99_change_mm"] for row in rows
        ),
        "maximum_attachment_boundary_change_mm": max(
            row["paired_attachment_boundary_max_change_mm"] for row in rows
        ),
        "maximum_far_background_cap_violation_mm": max(
            row["far_background_cap_violation_mm"] for row in rows
        ),
        "maximum_feasible_attachment_jump_mm": max(
            row["feasible_attachment_jump_max_mm"] for row in rows
        ),
        "maximum_boundary_p99_to_limit_ratio": max(
            row["boundary_p99_to_limit_ratio"] for row in rows
        ),
        "maximum_boundary_max_to_limit_ratio": max(
            row["boundary_max_to_limit_ratio"] for row in rows
        ),
        "all_topology_checks_passed": all(row["topology_passed"] for row in rows),
        "all_exact_stl_shell_checks_passed": all(
            row["exact_stl_shell_passed"] for row in rows
        ),
        "passed": all(row["passed"] for row in rows),
    }


def summarize(
    final_summary_path: str | Path,
    initial_summary_path: str | Path,
    final_eyewear_response_path: str | Path,
    initial_eyewear_response_path: str | Path,
) -> dict:
    final = _load_json(final_summary_path)
    initial = _load_json(initial_summary_path)
    _validate_public_summary_contract(final)
    _validate_public_summary_contract(initial)
    final_checks = _strict_checks(final, expected_passed=True)
    _strict_checks(initial, expected_passed=False)
    final_rows = _rows(final)
    for row in final_rows:
        _validate_final_row_checks(row)
    initial_rows = _rows(initial)
    if {row["row_id"] for row in initial_rows} != {row["row_id"] for row in final_rows}:
        raise ValueError("Initial and final summaries cover different rows")
    failed_initial_rows = [
        row
        for row in initial_rows
        if not _strict_bool(
            row.get("checks", {}).get("passed"), "initial row pass status"
        )
    ]
    if (
        len(failed_initial_rows) != 1
        or failed_initial_rows[0].get("row_id") != EYEWEAR_ROW_ID
    ):
        raise ValueError("Initial summary does not isolate the eyewear row")
    initial_row = _row(initial, EYEWEAR_ROW_ID)
    for row in initial_rows:
        _validate_initial_row_checks(
            row, eyewear_failure=row["row_id"] == EYEWEAR_ROW_ID
        )

    final_row = _row(final, EYEWEAR_ROW_ID)
    initial_response = _verify_response(initial_row, initial_eyewear_response_path)
    final_response = _verify_response(final_row, final_eyewear_response_path)
    initial_detection = _eyewear_detection(initial_response)
    final_detection = _eyewear_detection(final_response)
    if initial_detection["detected"] or initial_detection["deoccluded"]:
        raise ValueError("Initial response unexpectedly handled eyewear")
    if not final_detection["detected"] or not final_detection["deoccluded"]:
        raise ValueError("Final response did not detect and reconstruct eyewear")
    for key in (
        "minimum_dark_coverage_ratio",
        "minimum_component_coverage_ratio",
        "minimum_component_width_ratio",
    ):
        if final_detection[key] != initial_detection[key]:
            raise ValueError("Production eyewear gates changed between runs")

    if initial.get("matrix") != final.get("matrix"):
        raise ValueError("Matrix configuration changed between runs")
    if initial.get("fixture") != final.get("fixture"):
        raise ValueError("Fixture provenance changed between runs")
    for initial_candidate, final_candidate in zip(
        initial_rows, final_rows, strict=True
    ):
        initial_scene = copy.deepcopy(initial_candidate.get("scene"))
        final_scene = copy.deepcopy(final_candidate.get("scene"))
        if initial_candidate["row_id"] == EYEWEAR_ROW_ID:
            initial_scene.get("occluder", {}).pop("opacity", None)
            final_scene.get("occluder", {}).pop("opacity", None)
        if initial_scene != final_scene:
            raise ValueError("Scene configuration changed beyond eyewear opacity")

    initial_configuration = _effective_response_configuration(initial_response)
    final_configuration = _effective_response_configuration(final_response)
    if initial_configuration != final_configuration:
        raise ValueError("Effective API configuration changed between runs")
    initial_opacity = _finite(
        initial_row.get("scene", {}).get("occluder", {}).get("opacity"),
        "initial occluder opacity",
    )
    final_opacity = _finite(
        final_row.get("scene", {}).get("occluder", {}).get("opacity"),
        "final occluder opacity",
    )
    if initial_opacity != 0.42 or final_opacity != 0.60:
        raise ValueError("Fixture correction is not the measured 0.42 to 0.60 change")

    compact_rows = [_compact_row(row) for row in final_rows]
    aggregate = _aggregate(compact_rows)
    matrix = final.get("matrix", {})
    coverage = matrix.get("coverage", {})
    final_revision = final.get("server_provenance", {}).get("revision")
    initial_revision = initial.get("server_provenance", {}).get("revision")
    _hex_digest(final_revision, 40, "final revision")
    _hex_digest(initial_revision, 40, "initial revision")
    producer_files = _validate_run_provenance(final, final_revision)
    _validate_run_provenance(initial, initial_revision)
    if (
        final_configuration["relief_height_mm"]
        != _finite(matrix.get("z_scale"), "relief height")
        or final_configuration["max_xy_size_mm"]
        != _finite(matrix.get("max_xy_size"), "physical size")
        or final_configuration["target_dimension"]
        != int(final_row.get("scene", {}).get("target_dimension", -1))
    ):
        raise ValueError("Effective response does not match the matrix configuration")
    compact = {
        "schema_version": 1,
        "run_kind": "cc0_generated_live_relief_context_matrix_compact",
        "revision": final_revision,
        "full_summary_sha256": _sha256(final_summary_path),
        "generator": {
            "path": "backend/benchmark/summarize_cc0_live_relief_context.py",
            "sha256": _sha256(Path(__file__)),
        },
        "producer_files": producer_files,
        "privacy": EXPECTED_PRIVACY,
        "configuration": {
            "relief_height_mm": _finite(matrix.get("z_scale"), "relief height"),
            "max_xy_size_mm": _finite(matrix.get("max_xy_size"), "physical size"),
            "candidate_background_photo_detail_mm": final_configuration[
                "background_photo_detail_mm"
            ],
            "baseline_background_photo_detail_mm": _finite(
                matrix.get("baseline_background_photo_detail_mm"),
                "baseline background detail",
            ),
            "selection_background_depth_ratio": final_configuration[
                "selection_background_depth_ratio"
            ],
            "background_profiles": coverage.get("background_profiles", []),
            "lighting_profiles": coverage.get("lighting_profiles", []),
        },
        "measured_fixture_correction": {
            "initial_revision": initial_revision,
            "initial_full_summary_sha256": _sha256(initial_summary_path),
            "initial_passed_rows": sum(
                _strict_bool(
                    row.get("checks", {}).get("passed"), "initial row pass status"
                )
                for row in initial_rows
            ),
            "failed_row_id": EYEWEAR_ROW_ID,
            "only_failed_gate": "occlusion_deoccluded",
            "initial_occluder_opacity": _finite(
                initial_opacity, "initial occluder opacity"
            ),
            "final_occluder_opacity": _finite(final_opacity, "final occluder opacity"),
            "initial_detection": initial_detection,
            "final_detection": final_detection,
            "production_gates_unchanged": True,
            "passed": True,
        },
        "aggregate": aggregate,
        "rows": compact_rows,
        "checks": {
            "server_and_producer_provenance_passed": all(
                value for key, value in final_checks.items() if key != "passed"
            ),
            "all_behavioral_rows_passed": final_checks["all_behavioral_rows_passed"],
            "compact_rows_match_full_rows": len(compact_rows)
            == int(matrix.get("executed_rows", -1)),
            "response_checksums_match": True,
            "initial_failure_isolated": True,
            "production_gates_unchanged": True,
        },
    }
    checks = compact["checks"]
    checks["passed"] = (
        all(checks.values())
        and aggregate["passed"]
        and aggregate["all_topology_checks_passed"]
        and aggregate["all_exact_stl_shell_checks_passed"]
    )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-summary", required=True)
    parser.add_argument("--initial-summary", required=True)
    parser.add_argument("--final-eyewear-response", required=True)
    parser.add_argument("--initial-eyewear-response", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compact = summarize(
        args.final_summary,
        args.initial_summary,
        args.final_eyewear_response,
        args.initial_eyewear_response,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": compact["aggregate"]["rows"],
                "passed": compact["checks"]["passed"],
            },
            indent=2,
        )
    )
    if not compact["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
