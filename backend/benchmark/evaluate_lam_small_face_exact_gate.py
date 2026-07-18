"""Gate pinned LAM camera depth on the hardest exact CC0 face row."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_sheap_small_face_exact_gate import (
    COMPARISON_EPSILON,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_EXACT_DEPTH_SHA256,
    EXPECTED_FACE_BBOX_XYXY,
    EXPECTED_FACE_MASK_SHA256,
    EXPECTED_LOCAL_FACE_DEPTH_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_PART_MASK_SHA256,
    EXPECTED_SELECTION_BBOX_XYXY,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SOURCE_SHA256,
    ROW_ID,
    _compact_quality,
    _compare_parts,
    _quality,
    _write_depth_preview,
)
from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    SUBJECT_BOUNDARY_PIXELS,
    build_small_face_candidate,
)
from backend.benchmark.lam_depth_provider import (
    LAM_MODEL_REVISION,
    LAM_MODEL_SHA256,
    LAM_SOURCE_REVISION,
    normalize_lam_comp_depth,
    validate_lam_camera_depth,
    warp_lam_depth_to_source,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    _exact_face_depth_quality,
)


METHOD = "lam-20k-camera-depth-low-frequency-fusion"
PROVIDER_ALPHAS = (
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
)
MINIMUM_PROVIDER_FACE_COVERAGE = 0.50
EXPECTED_PROVIDER_HASHES = {
    "lam_preflight.json": (
        "9313dc1138e76b27339a2e36b2dc4abcebc06de05324c6f879be97ceb789f333"
    ),
    "pre_render_audit.json": (
        "4f7c109cdc4640dde39e4e5fb175f7a5e83c5e1e5846bbbdf9e2c77fef66d9b1"
    ),
    "provider_depth_render.npy": (
        "e0d4265bae65e61db6e47d719c7f6a9a9d5a3c6593c277a373185a8192d2f4f7"
    ),
    "provider_depth_render_weighted.npy": (
        "a39e73c1db52cb96a06fcdfc035d5e38373e884450180c13884f5dbc070c4aaa"
    ),
    "provider_depth_source.npy": (
        "38c64ae56a8672fa0fa43c43ee432e00f911ddf87708490fdab5687d917866fd"
    ),
    "provider_mask_render.npy": (
        "70da2b441c225ed2d93b953a3c4eed9ad64931a2a39ab991dd308e2bd09f8dce"
    ),
    "provider_mask_source.npy": (
        "81362c22d6ff1af05befb20a2c020099128b44bed1eb1228447ed03c1e9def44"
    ),
    "provider_metadata.json": (
        "d18cd5dc8f290a6fc85eddd204ad14a575c9602822d60b7dfbf35b947d793f40"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _selection_bbox(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(x):
        raise ValueError("LAM exact selection mask is empty")
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def _expected_input_hashes() -> dict[str, str]:
    expected = {
        "source": EXPECTED_SOURCE_SHA256,
        "selection_mask": EXPECTED_SELECTION_SHA256,
        "exact_depth": EXPECTED_EXACT_DEPTH_SHA256,
        "baseline": EXPECTED_BASELINE_SHA256,
        "metadata": EXPECTED_METADATA_SHA256,
        "local_face_depth": EXPECTED_LOCAL_FACE_DEPTH_SHA256,
        "face_mask": EXPECTED_FACE_MASK_SHA256,
    }
    expected.update(
        {
            f"part_mask:{name}": sha256
            for name, sha256 in EXPECTED_PART_MASK_SHA256.items()
        }
    )
    expected.update(
        {
            f"provider:{name}": sha256
            for name, sha256 in EXPECTED_PROVIDER_HASHES.items()
        }
    )
    return expected


def _verify_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    expected = _expected_input_hashes()
    if set(paths) != set(expected):
        raise ValueError("LAM exact gate input path schema changed")
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LAM exact gate inputs are missing: {missing}")
    actual = {name: _sha256(path) for name, path in paths.items()}
    mismatches = [name for name in expected if actual[name] != expected[name]]
    if mismatches:
        raise ValueError(f"LAM exact gate input hashes changed: {mismatches}")
    return actual


def _provider_arrays_and_validation(
    provider_dir: Path,
    selection_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict, dict]:
    metadata = json.loads(
        (provider_dir / "provider_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema_version") != 2:
        raise ValueError("LAM provider metadata schema changed")
    if metadata.get("provider") != "lam-20k-camera-depth":
        raise ValueError("LAM provider identity changed")
    if metadata.get("production_changed") is not False:
        raise ValueError("LAM research run changed production")
    model = metadata.get("model", {})
    if (
        model.get("revision") != LAM_MODEL_REVISION
        or model.get("sha256") != LAM_MODEL_SHA256
    ):
        raise ValueError("LAM provider model pin changed")
    if metadata.get("license", {}).get("production_eligible") is not False:
        raise ValueError("LAM non-commercial weights became production eligible")
    if (
        metadata.get("license", {}).get("all_asset_terms_established")
        is not False
    ):
        raise ValueError("LAM auxiliary-asset license guard changed")
    camera = metadata.get("camera", {})
    if (
        camera.get("native_depth_field") != "alpha-weighted comp_depth"
        or camera.get("depth_normalization")
        != "comp_depth / comp_mask where comp_mask >= 0.01"
        or camera.get("source_depth_resampling")
        != (
            "bilinear comp_depth and comp_mask warp, then division where "
            "warped comp_mask >= 0.01"
        )
        or camera.get("near_is_smaller") is not True
    ):
        raise ValueError("LAM camera-depth semantics changed")

    names = {
        "render_depth_weighted": "provider_depth_render_weighted.npy",
        "render_depth": "provider_depth_render.npy",
        "render_mask": "provider_mask_render.npy",
        "source_depth": "provider_depth_source.npy",
        "source_mask": "provider_mask_source.npy",
    }
    arrays = {
        name: np.load(provider_dir / filename, allow_pickle=False)
        for name, filename in names.items()
    }
    reconstructed_render = normalize_lam_comp_depth(
        arrays["render_depth_weighted"],
        arrays["render_mask"],
    )
    render_exact = bool(
        np.array_equal(
            np.isnan(reconstructed_render),
            np.isnan(arrays["render_depth"]),
        )
        and np.allclose(
            reconstructed_render,
            arrays["render_depth"],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )
    tracker_crop = metadata.get("detector", {}).get("tracker_crop_xyxy")
    preprocessor_crop = camera.get("preprocessor_crop_xywh")
    source_depth, source_mask = warp_lam_depth_to_source(
        arrays["render_depth_weighted"],
        arrays["render_mask"],
        selection_mask.shape,
        tracker_crop,
        preprocessor_crop,
    )
    source_depth_exact = bool(
        np.allclose(
            source_depth,
            arrays["source_depth"],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    )
    source_mask_exact = bool(
        np.array_equal(source_mask, arrays["source_mask"])
    )
    validation = validate_lam_camera_depth(
        arrays["render_depth"],
        arrays["render_mask"],
        arrays["source_depth"],
        arrays["source_mask"],
        selection_mask=selection_mask.astype(np.float32),
    )
    checks = {
        "weighted_depth_normalization_exact": render_exact,
        "captured_affine_source_depth_exact": source_depth_exact,
        "captured_affine_source_mask_exact": source_mask_exact,
        "host_validation_reproduced": validation
        == metadata.get("host_depth_validation"),
    }
    if not all(checks.values()):
        raise ValueError(f"LAM provider replay failed: {checks}")
    return arrays, metadata, {"checks": checks, "depth": validation}


def _measurement_source_manifest() -> dict:
    functions = (
        build_small_face_candidate,
        _exact_face_depth_quality,
        normalize_lam_comp_depth,
        validate_lam_camera_depth,
        warp_lam_depth_to_source,
        _compact_quality,
        _compare_parts,
    )
    repo_root = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve()}
    for function in functions:
        source = inspect.getsourcefile(function)
        if source is None:
            raise RuntimeError(f"Cannot locate measurement source: {function}")
        paths.add(Path(source).resolve())
    return {
        "schema_version": 1,
        "files": {
            path.relative_to(repo_root).as_posix(): _sha256(path)
            for path in sorted(paths)
        },
        "functions": {
            f"{function.__module__}.{function.__qualname__}": hashlib.sha256(
                inspect.getsource(function).encode("utf-8")
            ).hexdigest()
            for function in functions
        },
    }


def _decision_from_variants(variants: list[dict]) -> tuple[list[dict], dict]:
    if not variants:
        raise ValueError("LAM exact gate produced no variants")
    ordered = sorted(
        variants,
        key=lambda item: (
            item["eligible"],
            item["comparison"]["passed_checks"],
            -item["quality"]["combined_part_failures"],
            item["quality"]["shape_correlation"],
            item["quality"]["gradient_correlation"],
            -item["quality"]["normalized_rmse"],
        ),
        reverse=True,
    )
    selected = ordered[0]
    eligible = bool(selected["eligible"])
    return ordered, {
        "status": "eligible-for-30mm-stl-replay" if eligible else "hold",
        "eligible_for_30mm_stl_replay": eligible,
        "selected_variant": selected["variant_id"],
        "selected_passed_part_metric_checks": int(
            selected["comparison"]["passed_checks"]
        ),
        "selected_total_part_metric_checks": int(
            selected["comparison"]["total_checks"]
        ),
        "failed_checks": [
            name for name, passed in selected["checks"].items() if not passed
        ],
        "production_changed": False,
        "next_action": (
            "run-bounded-30mm-stl-replay"
            if eligible
            else "close-lam-face-depth-lane"
        ),
    }


def evaluate(
    baseline_row_dir: str | Path,
    exact_row_dir: str | Path,
    provider_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    started = time.perf_counter()
    baseline_row_dir = Path(baseline_row_dir).resolve()
    exact_row_dir = Path(exact_row_dir).resolve()
    provider_dir = Path(provider_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = exact_row_dir / "source.png"
    selection_path = exact_row_dir / "selection_mask.png"
    exact_depth_path = exact_row_dir / "exact_depth.npy"
    baseline_path = baseline_row_dir / "output_depth_data_face_refined.npy"
    metadata_path = baseline_row_dir / "output_face_refinement_metadata.json"
    local_depth_path = (
        baseline_row_dir
        / "face_refinement/face_00_depth/output_depth_data.npy"
    )
    face_mask_path = (
        baseline_row_dir / "face_refinement/face_00_parts/face.png"
    )
    part_paths = {
        name: exact_row_dir / "exact_face_parts" / f"{name}.png"
        for name in EXPECTED_PART_MASK_SHA256
    }
    paths = {
        "source": source_path,
        "selection_mask": selection_path,
        "exact_depth": exact_depth_path,
        "baseline": baseline_path,
        "metadata": metadata_path,
        "local_face_depth": local_depth_path,
        "face_mask": face_mask_path,
        **{f"part_mask:{name}": path for name, path in part_paths.items()},
        **{
            f"provider:{name}": provider_dir / name
            for name in EXPECTED_PROVIDER_HASHES
        },
    }
    input_hashes = _verify_input_hashes(paths)

    with Image.open(selection_path) as loaded:
        selection = np.asarray(loaded.convert("L")) > 0
    with Image.open(face_mask_path) as loaded:
        face_mask = np.asarray(loaded.convert("L"))
    if _selection_bbox(selection) != EXPECTED_SELECTION_BBOX_XYXY:
        raise ValueError("LAM exact selection bbox changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    face_record = metadata["faces"][0]
    if [int(value) for value in face_record["bbox"]] != EXPECTED_FACE_BBOX_XYXY:
        raise ValueError("LAM exact face bbox changed")
    crop_bbox = [int(value) for value in face_record["crop_bbox"]]

    arrays, provider_metadata, provider_replay = (
        _provider_arrays_and_validation(provider_dir, selection)
    )
    baseline_depth = np.load(baseline_path, allow_pickle=False).astype(np.float32)
    local_depth = np.load(local_depth_path, allow_pickle=False).astype(np.float32)
    provider_depth = arrays["source_depth"].astype(np.float32)
    if baseline_depth.shape != selection.shape or face_mask.shape != selection.shape:
        raise ValueError("LAM exact gate input grid changed")

    raw_metrics = _exact_face_depth_quality(
        provider_dir / "provider_depth_source.npy",
        exact_depth_path,
        selection_path,
        expected_scale_sign=1.0,
        part_mask_paths=part_paths,
    )
    raw_compact = _compact_quality(raw_metrics)
    _write_json(output_dir / "provider_raw_metrics.json", raw_metrics)
    baseline_metrics = _quality(
        baseline_path,
        exact_depth_path,
        selection_path,
        part_paths,
    )
    baseline_compact = _compact_quality(baseline_metrics)
    _write_json(output_dir / "baseline_metrics.json", baseline_metrics)

    x0, y0, x1, y1 = crop_bbox
    local_face_mask = face_mask[y0:y1, x0:x1] > 0
    provider_crop = provider_depth[y0:y1, x0:x1]
    expected_face_pixels = int(np.count_nonzero(local_face_mask))
    provider_face_pixels = int(
        np.count_nonzero(np.isfinite(provider_crop) & local_face_mask)
    )
    provider_face_coverage = provider_face_pixels / max(expected_face_pixels, 1)
    selection_distance = cv2.distanceTransform(
        selection.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    boundary = (
        (selection_distance > 0)
        & (selection_distance <= SUBJECT_BOUNDARY_PIXELS)
    )

    variants = []
    for alpha in PROVIDER_ALPHAS:
        candidate, fusion = build_small_face_candidate(
            baseline_depth,
            local_depth,
            provider_depth,
            face_mask,
            selection,
            crop_bbox,
            provider_alpha=alpha,
        )
        variant_id = f"camera_a{alpha:.7f}"
        candidate_path = output_dir / f"candidate_{variant_id}.npy"
        np.save(candidate_path, candidate)
        _write_depth_preview(
            output_dir / f"candidate_{variant_id}.png",
            candidate,
        )
        metrics = _quality(
            candidate_path,
            exact_depth_path,
            selection_path,
            part_paths,
        )
        metrics_path = output_dir / f"candidate_{variant_id}_metrics.json"
        _write_json(metrics_path, metrics)
        compact = _compact_quality(metrics)
        comparison = _compare_parts(baseline_compact, compact)
        checks = {
            "provider_face_coverage": provider_face_coverage
            >= MINIMUM_PROVIDER_FACE_COVERAGE,
            "provider_depth_orientation_positive_scale": bool(
                math.isfinite(float(fusion["provider_scale"]))
                and float(fusion["provider_scale"]) > 0
            ),
            "all_six_part_metrics_non_regressing": comparison[
                "all_part_metrics_non_regressing"
            ],
            "all_six_parts_strict_shape_gradient_affine_improvement": (
                comparison[
                    "parts_with_strict_shape_gradient_affine_improvement"
                ]
                == 6
            ),
            "combined_part_failures_non_regressing": (
                compact["combined_part_failures"]
                <= baseline_compact["combined_part_failures"]
            ),
            "overall_shape_non_regressing": (
                compact["shape_correlation"] + COMPARISON_EPSILON
                >= baseline_compact["shape_correlation"]
            ),
            "overall_gradient_non_regressing": (
                compact["gradient_correlation"] + COMPARISON_EPSILON
                >= baseline_compact["gradient_correlation"]
            ),
            "overall_rmse_non_regressing": (
                compact["normalized_rmse"]
                <= baseline_compact["normalized_rmse"] + COMPARISON_EPSILON
            ),
            "background_value_exact": bool(
                np.array_equal(candidate[~selection], baseline_depth[~selection])
            ),
            "attachment_boundary_value_exact": bool(
                not np.any(boundary)
                or np.array_equal(candidate[boundary], baseline_depth[boundary])
            ),
        }
        variants.append(
            {
                "variant_id": variant_id,
                "provider_alpha": float(alpha),
                "provider_face_coverage": float(provider_face_coverage),
                "candidate_sha256": _sha256(candidate_path),
                "metrics_file": metrics_path.name,
                "metrics_sha256": _sha256(metrics_path),
                "quality": compact,
                "comparison": comparison,
                "fusion": fusion,
                "checks": checks,
                "eligible": bool(all(checks.values())),
            }
        )

    variants, decision = _decision_from_variants(variants)
    evidence = {
        "schema_version": 1,
        "method": METHOD,
        "row_id": ROW_ID,
        "privacy": "CC0 MakeHuman synthetic identity and procedural scene only",
        "source_geometry": "evaluation-only",
        "production_changed": False,
        "input_hashes": input_hashes,
        "harness": _measurement_source_manifest(),
        "provider": {
            "current_reference_source_revision": LAM_SOURCE_REVISION,
            "metadata": provider_metadata,
            "replay": provider_replay,
            "face_finite_pixels": provider_face_pixels,
            "face_expected_pixels": expected_face_pixels,
            "face_coverage": float(provider_face_coverage),
            "raw_quality": raw_compact,
        },
        "configuration": {
            "provider_alphas": list(PROVIDER_ALPHAS),
            "minimum_provider_face_coverage": MINIMUM_PROVIDER_FACE_COVERAGE,
            "strict_parts": sorted(EXPECTED_PART_MASK_SHA256),
            "strict_metrics_per_part": 7,
        },
        "baseline": baseline_compact,
        "variants": variants,
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-row-dir", required=True)
    parser.add_argument("--exact-row-dir", required=True)
    parser.add_argument("--provider-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = evaluate(
        args.baseline_row_dir,
        args.exact_row_dir,
        args.provider_dir,
        args.output_dir,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_30mm_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
