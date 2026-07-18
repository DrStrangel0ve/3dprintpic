"""Emit and compare baseline, VGGHeads, and oracle 30 mm face relief STLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from backend.benchmark.face_part_metrics import (
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    MAX_XY_SIZE_MM,
    RELIEF_HEIGHT_MM,
    _mask_on_emitted_grid,
)
from backend.benchmark.run_relief_scene_regression import _mesh_topology
from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    _appearance_checks,
    _stl_heightfield_agreement,
)
from backend.benchmark.summarize_private_live_api_background_replay import (
    _independent_background_checks,
    _independent_cap_checks,
)
from backend.pic_to_3d import depth_data_to_3d_model
from backend.stl_diagnostics import (
    json_safe_stl_diagnostics,
    stl_diagnostics,
)


METHOD = "vggheads-small-face-30mm-production-emission-replay"
HARD_SMALL_FACE_ROW = "small_side_lit_shelves_256"
MINIMUM_BACKGROUND_CORRELATION = 0.999
MINIMUM_BACKGROUND_RMS_RETENTION = 0.98
MAXIMUM_BACKGROUND_RMS_RETENTION = 1.02
MAXIMUM_RELIEF_HEIGHT_MM = RELIEF_HEIGHT_MM + 0.011
PRINTABLE_FEATURE_DEPTH_MM = 0.4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def _named_part_failure_count(
    shape: dict,
    affine_mm: dict,
) -> int:
    return int(
        len(shape.get("failed_parts", []))
        + len(affine_mm.get("failed_parts", []))
    )


def _emitted_masks(
    selection_mask_path: Path,
    part_mask_paths: dict[str, Path],
    postprocess: dict,
    emitted_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    transform = postprocess["surface_grid_transform"]
    face = _mask_on_emitted_grid(
        selection_mask_path,
        transform,
        emitted_shape,
    )
    parts = {
        name: (
            _mask_on_emitted_grid(
                part_mask_paths[name],
                transform,
                emitted_shape,
            )
            & face
        )
        for name in FACE_PART_NAMES
    }
    if not np.any(face) or any(not np.any(mask) for mask in parts.values()):
        raise ValueError("An emitted exact face or named-part mask is empty")
    return face, parts


def _emit_variant(
    name: str,
    depth_path: Path,
    output_dir: Path,
    *,
    invert: bool,
    source_path: Path,
    selection_mask_path: Path,
    face_region_path: Path,
    feature_weight_path: Path,
    feature_exclusion_path: Path,
    normalization_reference_depth: Path | None = None,
    relief_height_mm: float = RELIEF_HEIGHT_MM,
) -> dict:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    stl_path = variant_dir / "relief.stl"
    surface_path = variant_dir / "surface.npy"
    reference_path = variant_dir / "reference_surface.npy"
    started = time.perf_counter()
    postprocess = depth_data_to_3d_model(
        depth_path,
        output_stl_path=str(stl_path),
        target_dimension=256,
        z_scale=relief_height_mm,
        invert=invert,
        sigma=0.6,
        max_xy_size=MAX_XY_SIZE_MM,
        relief_gamma=0.75,
        detail_boost=0.8,
        background_detail_boost=2.4,
        source_image=source_path,
        background_photo_detail_mm=0.60,
        trim_top_background=False,
        feature_weight_mask=feature_weight_path,
        printable_feature_depth_mm=PRINTABLE_FEATURE_DEPTH_MM,
        feature_bridge_depth_mm=0.8,
        feature_exclusion_mask=feature_exclusion_path,
        detail_radius=2.0,
        detail_edge_threshold=0.12,
        low_percentile=1.0,
        high_percentile=99.0,
        base_border_px=2,
        value_transform="linear",
        minimum_feature_mm=0.8,
        max_relief_slope=2.0,
        face_region_mask=face_region_path,
        selection_region_mask=selection_mask_path,
        selection_background_depth_ratio=0.65,
        surface_output_path=surface_path,
        reference_surface_output_path=reference_path,
        normalization_reference_depth=normalization_reference_depth,
    )
    surface = np.load(surface_path).astype(np.float64)
    reference = np.load(reference_path).astype(np.float64)
    mesh = trimesh.load_mesh(stl_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Emitted STL is not one mesh: {type(mesh)!r}")
    topology = _mesh_topology(mesh)
    shell = _stl_heightfield_agreement(
        stl_path,
        surface_path,
        expected_max_xy_size_mm=MAX_XY_SIZE_MM,
    )
    diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
    record = {
        "name": name,
        "invert": bool(invert),
        "runtime_seconds": float(time.perf_counter() - started),
        "input_depth_sha256": _sha256(depth_path),
        "stl_sha256": _sha256(stl_path),
        "surface_sha256": _sha256(surface_path),
        "reference_surface_sha256": _sha256(reference_path),
        "surface_shape": [int(value) for value in surface.shape],
        "surface_min_mm": float(np.min(surface)),
        "surface_max_mm": float(np.max(surface)),
        "surface_span_mm": float(np.max(surface) - np.min(surface)),
        "postprocess": postprocess,
        "topology": topology,
        "shell": shell,
        "stl_diagnostics": diagnostics,
    }
    (variant_dir / "postprocess.json").write_text(
        json.dumps(postprocess, indent=2) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "record.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        **record,
        "_surface": surface,
        "_reference": reference,
        "_surface_path": surface_path,
        "_stl_path": stl_path,
    }


def _variant_quality(
    variant: dict,
    oracle: dict,
    selection_mask_path: Path,
    part_mask_paths: dict[str, Path],
) -> dict:
    surface = variant["_surface"]
    oracle_surface = oracle["_surface"]
    if surface.shape != oracle_surface.shape:
        raise ValueError("Variant and oracle emitted grids differ")
    face, parts = _emitted_masks(
        selection_mask_path,
        part_mask_paths,
        variant["postprocess"],
        surface.shape,
    )
    pitch_mm = float(variant["postprocess"]["mesh_sample_pitch_mm"])
    shape = face_part_cross_height_metrics(
        oracle_surface,
        surface,
        face,
        parts,
        sample_pitch_mm=pitch_mm,
    )
    affine_mm = face_part_affine_surface_error_metrics(
        oracle_surface,
        surface,
        face,
        parts,
    )
    appearance = variant["postprocess"]["surface_appearance_agreement"][
        "face"
    ]
    appearance_checks = _appearance_checks(
        appearance,
        FACE_APPEARANCE_GATES,
    )
    background = _independent_background_checks(
        variant["postprocess"]["background_depth_preservation"]
    )
    cap = _independent_cap_checks(
        variant["postprocess"]["selection_background_physical_cap"]
    )
    checks = {
        "named_part_shape": bool(shape.get("passed", False)),
        "named_part_affine_mm": bool(affine_mm.get("passed", False)),
        "face_appearance": bool(appearance_checks.get("passed", False)),
        "background_depth": bool(background.get("passed", False)),
        "physical_cap": bool(cap.get("passed", False)),
        "topology": bool(variant["topology"].get("printable", False)),
        "complete_shell": bool(variant["shell"].get("passed", False)),
        "height_cap": bool(
            variant["surface_max_mm"] <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
    }
    required_checks = (
        "face_appearance",
        "background_depth",
        "physical_cap",
        "topology",
        "complete_shell",
        "height_cap",
    )
    checks["passed"] = bool(
        all(checks[name] for name in required_checks)
    )
    return {
        "combined_named_part_failures": _named_part_failure_count(
            shape,
            affine_mm,
        ),
        "named_part_shape": shape,
        "named_part_affine_mm": affine_mm,
        "face_appearance": appearance,
        "face_appearance_checks": appearance_checks,
        "background_depth_checks": background,
        "physical_cap_checks": cap,
        "checks": checks,
    }


def _paired_background(
    baseline: dict,
    candidate: dict,
    selection_mask_path: Path,
) -> dict:
    baseline_surface = baseline["_surface"]
    candidate_surface = candidate["_surface"]
    if baseline_surface.shape != candidate_surface.shape:
        raise ValueError("Baseline and candidate emitted grids differ")
    selection = _mask_on_emitted_grid(
        selection_mask_path,
        baseline["postprocess"]["surface_grid_transform"],
        baseline_surface.shape,
    )
    if not np.any(selection) or np.all(selection):
        raise ValueError("Paired background selection mask is invalid")
    background = ~selection
    baseline_values = baseline_surface[background]
    candidate_values = candidate_surface[background]
    baseline_centered = baseline_values - np.median(baseline_values)
    candidate_centered = candidate_values - np.median(candidate_values)
    baseline_rms = float(
        np.sqrt(np.mean(np.square(baseline_centered)))
    )
    candidate_rms = float(
        np.sqrt(np.mean(np.square(candidate_centered)))
    )
    rms_retention = candidate_rms / max(baseline_rms, 1e-12)
    correlation = _correlation(baseline_values, candidate_values)
    checks = {
        "correlation": correlation >= MINIMUM_BACKGROUND_CORRELATION,
        "rms_retention": (
            MINIMUM_BACKGROUND_RMS_RETENTION
            <= rms_retention
            <= MAXIMUM_BACKGROUND_RMS_RETENTION
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "samples": int(np.count_nonzero(background)),
        "correlation": correlation,
        "rms_retention": float(rms_retention),
        "maximum_absolute_difference_mm": float(
            np.max(np.abs(candidate_values - baseline_values))
        ),
        "checks": checks,
    }


def _compact_variant(variant: dict) -> dict:
    return {
        key: value
        for key, value in variant.items()
        if not key.startswith("_")
    }


def _eligible_for_replay(evidence: dict) -> bool:
    decision = evidence.get("decision", {})
    if "eligible_for_full_stl_replay" in decision:
        return bool(decision["eligible_for_full_stl_replay"])
    if "eligible_for_30mm_stl_replay" in decision:
        return bool(decision["eligible_for_30mm_stl_replay"])
    return bool(
        decision.get("checks", {}).get(
            "eligible_for_30mm_stl_replay",
            False,
        )
    )


def evaluate(
    exact_gate_root: str | Path,
    baseline_root: str | Path,
    output_dir: str | Path,
    *,
    candidate_depth_path: str | Path | None = None,
    eligibility_evidence_path: str | Path | None = None,
) -> dict:
    exact_gate_root = Path(exact_gate_root)
    baseline_root = Path(baseline_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_row = (
        exact_gate_root / "rows" / HARD_SMALL_FACE_ROW / "exact"
    )
    source_path = exact_row / "source.png"
    selection_mask_path = exact_row / "selection_mask.png"
    exact_depth_path = exact_row / "exact_depth.npy"
    part_mask_paths = {
        name: exact_row / "exact_face_parts" / f"{name}.png"
        for name in FACE_PART_NAMES
    }
    baseline_dir = baseline_root / HARD_SMALL_FACE_ROW / "baseline"
    baseline_depth_path = (
        baseline_dir / "output_depth_data_face_refined.npy"
    )
    candidate_depth_path = (
        Path(candidate_depth_path)
        if candidate_depth_path is not None
        else (
            exact_gate_root
            / "rows"
            / HARD_SMALL_FACE_ROW
            / "candidate_depth.npy"
        )
    )
    eligibility_evidence_path = (
        Path(eligibility_evidence_path)
        if eligibility_evidence_path is not None
        else exact_gate_root / "evidence.json"
    )
    face_region_path = (
        baseline_dir / "output_face_refinement_region.png"
    )
    feature_weight_path = (
        baseline_dir / "output_face_refinement_weight.png"
    )
    feature_exclusion_path = (
        baseline_dir / "output_face_refinement_occlusion.png"
    )
    required = (
        source_path,
        selection_mask_path,
        exact_depth_path,
        baseline_depth_path,
        candidate_depth_path,
        face_region_path,
        feature_weight_path,
        feature_exclusion_path,
        eligibility_evidence_path,
        *part_mask_paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "30 mm VGGHeads replay artifacts are unavailable: "
            + ", ".join(missing)
        )

    started = time.perf_counter()
    oracle = _emit_variant(
        "oracle",
        exact_depth_path,
        output_dir,
        invert=True,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
    )
    baseline = _emit_variant(
        "baseline",
        baseline_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
    )
    candidate = _emit_variant(
        "candidate",
        candidate_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
        normalization_reference_depth=baseline_depth_path,
    )
    baseline_quality = _variant_quality(
        baseline,
        oracle,
        selection_mask_path,
        part_mask_paths,
    )
    candidate_quality = _variant_quality(
        candidate,
        oracle,
        selection_mask_path,
        part_mask_paths,
    )
    paired_background = _paired_background(
        baseline,
        candidate,
        selection_mask_path,
    )
    checks = {
        "exact_gate_eligible": bool(
            _eligible_for_replay(
                json.loads(
                    eligibility_evidence_path.read_text(
                        encoding="utf-8"
                    )
                )
            )
        ),
        "candidate_named_parts_improve": bool(
            candidate_quality["combined_named_part_failures"]
            < baseline_quality["combined_named_part_failures"]
        ),
        "candidate_variant_passes": bool(
            candidate_quality["checks"]["passed"]
        ),
        "oracle_variant_passes": bool(
            oracle["topology"].get("printable", False)
            and oracle["shell"].get("passed", False)
            and oracle["surface_max_mm"] <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
        "paired_background_preserved": bool(
            paired_background["checks"]["passed"]
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    evidence = {
        "schema_version": 1,
        "status": "pass" if checks["passed"] else "hold",
        "method": METHOD,
        "privacy": (
            "CC0 MakeHuman synthetic head and deterministic procedural "
            "scene only"
        ),
        "production_changed": False,
        "configuration": {
            "relief_height_mm": RELIEF_HEIGHT_MM,
            "max_xy_size_mm": MAX_XY_SIZE_MM,
            "target_dimension": 256,
            "background_depth_ratio": 0.65,
            "background_photo_detail_mm": 0.60,
            "background_detail_boost": 2.4,
            "maximum_relief_height_mm": MAXIMUM_RELIEF_HEIGHT_MM,
        },
        "artifacts": {
            "source_sha256": _sha256(source_path),
            "selection_mask_sha256": _sha256(selection_mask_path),
            "exact_depth_sha256": _sha256(exact_depth_path),
            "baseline_depth_sha256": _sha256(baseline_depth_path),
            "candidate_depth_sha256": _sha256(candidate_depth_path),
            "eligibility_evidence_sha256": _sha256(
                eligibility_evidence_path
            ),
        },
        "oracle": _compact_variant(oracle),
        "baseline": {
            **_compact_variant(baseline),
            "quality": baseline_quality,
        },
        "candidate": {
            **_compact_variant(candidate),
            "quality": candidate_quality,
        },
        "paired_background": paired_background,
        "checks": checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-gate-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-depth")
    parser.add_argument("--eligibility-evidence")
    args = parser.parse_args()
    evidence = evaluate(
        args.exact_gate_root,
        args.baseline_root,
        args.output_dir,
        candidate_depth_path=args.candidate_depth,
        eligibility_evidence_path=args.eligibility_evidence,
    )
    print(json.dumps(evidence["checks"], indent=2))
    if not evidence["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
