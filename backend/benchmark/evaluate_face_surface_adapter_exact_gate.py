"""Replay a trained face-surface adapter through the exact production pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_surface_adapter import (
    CHECKPOINT_SCHEMA_VERSION,
    CRITICAL_PART_CHECKS,
    MODEL_ID,
    MODEL_REVISION,
    _coordinate_channels,
    _strictly_improves,
    build_surface_adapter,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    SUBJECT_SUPPORTED_METHOD,
)
from backend.face_depth_refinement import refine_depth_for_faces


HARD_SMALL_FACE_ROW = "small_side_lit_shelves_256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Local face depth contains no finite values")
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high <= low:
        return values.copy()
    return ((values - low) / (high - low)).astype(np.float32)


def _resize(
    values: np.ndarray,
    shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    return cv2.resize(
        np.asarray(values),
        (int(shape[1]), int(shape[0])),
        interpolation=interpolation,
    )


def _adapter_input(
    image_rgb: np.ndarray,
    local_depth: np.ndarray,
    face_mask: np.ndarray,
    *,
    network_size: int,
) -> np.ndarray:
    shape = (int(network_size), int(network_size))
    rgb = _resize(image_rgb, shape, cv2.INTER_AREA).astype(np.float32)
    rgb = rgb.transpose(2, 0, 1) / 255.0
    baseline = _resize(
        _minmax_normalize(local_depth),
        shape,
        cv2.INTER_CUBIC,
    ).astype(np.float32)
    face = _resize(
        (np.asarray(face_mask) > 0).astype(np.uint8),
        shape,
        cv2.INTER_NEAREST,
    ).astype(np.float32)
    return np.concatenate(
        (
            rgb,
            baseline[None],
            face[None],
            _coordinate_channels(network_size),
        ),
        axis=0,
    ).astype(np.float32)


class CachedLocalFaceDepth:
    """Return one historical local-depth crop and audit crop equivalence."""

    def __init__(self, depth_path: Path, crop_path: Path):
        self.depth = np.squeeze(np.load(depth_path)).astype(np.float32)
        self.expected_crop = np.asarray(Image.open(crop_path).convert("RGB"))
        self.depth_sha256 = _sha256(depth_path)
        self.crop_sha256 = _sha256(crop_path)
        self.calls: list[dict] = []

    def callback(self, image_path, output_dir):
        actual_crop = np.asarray(Image.open(image_path).convert("RGB"))
        equivalent = bool(
            actual_crop.shape == self.expected_crop.shape
            and np.array_equal(actual_crop, self.expected_crop)
        )
        self.calls.append(
            {
                "shape": list(actual_crop.shape),
                "pixel_equivalent": equivalent,
            }
        )
        if not equivalent:
            raise ValueError("Current face crop differs from the exact cached crop")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output_depth_data.npy"
        np.save(path, self.depth)
        return str(path)

    def provenance(self) -> dict:
        return {
            "depth_sha256": self.depth_sha256,
            "crop_sha256": self.crop_sha256,
            "calls": self.calls,
            "all_crops_pixel_equivalent": bool(
                self.calls and all(call["pixel_equivalent"] for call in self.calls)
            ),
        }


class FaceSurfaceResidualProvider:
    def __init__(self, checkpoint_path: Path, device: str):
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        supported_methods = {
            "cc0_camera_aligned_face_surface_residual",
            "cc0_production_native_face_surface_residual",
            SUBJECT_SUPPORTED_METHOD,
        }
        method = checkpoint.get("method")
        if method not in supported_methods:
            raise ValueError(
                f"Unsupported face-surface checkpoint method: {method!r}"
            )
        required = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "input_channels": 7,
            "residual_postprocess": (
                "remove-affine-component-relative-to-local-depth"
            ),
        }
        mismatches = {
            key: {"expected": expected, "actual": checkpoint.get(key)}
            for key, expected in required.items()
            if checkpoint.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "Face-surface checkpoint provenance mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        self.surface_support_mode = str(
            checkpoint.get("surface_support_mode", "detector-face")
        )
        expected_support_mode = (
            "selection-subject"
            if method == SUBJECT_SUPPORTED_METHOD
            else "detector-face"
        )
        if self.surface_support_mode != expected_support_mode:
            raise ValueError(
                "Face-surface checkpoint support contract mismatch: "
                f"expected {expected_support_mode!r}, got "
                f"{self.surface_support_mode!r}"
            )
        alpha = float(checkpoint.get("selected_alpha", -1.0))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Face-surface checkpoint selected alpha is invalid")
        network_size = int(checkpoint.get("network_size", 0))
        if network_size < 32:
            raise ValueError("Face-surface checkpoint network size is invalid")

        self.device = str(device)
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.method = str(method)
        self.alpha = alpha
        self.network_size = network_size
        self.model = build_surface_adapter(
            input_channels=int(checkpoint["input_channels"])
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device, dtype=torch.float32).eval()
        self.timings: list[float] = []
        self.calls: list[dict] = []
        self.allocated_before_inference = (
            int(torch.cuda.memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def infer(
        self,
        image_path: Path,
        local_depth: np.ndarray,
        surface_support_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        import torch

        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        input_values = _adapter_input(
            image_rgb,
            local_depth,
            surface_support_mask,
            network_size=self.network_size,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            residual = self.model(
                torch.from_numpy(input_values)[None].to(
                    self.device,
                    dtype=torch.float32,
                )
            )[0, 0]
        elapsed = time.perf_counter() - started
        self.timings.append(elapsed)
        residual = residual.float().cpu().numpy()
        residual = _resize(
            residual,
            np.asarray(local_depth).shape,
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        residual *= self.alpha
        stats = {
            "provider": "cc0-camera-aligned-face-surface-residual",
            "selected_alpha": self.alpha,
            "network_size": self.network_size,
            "inference_seconds": elapsed,
            "raw_scaled_residual_min": float(np.min(residual)),
            "raw_scaled_residual_max": float(np.max(residual)),
            "raw_scaled_residual_rms": float(
                np.sqrt(np.mean(np.square(residual, dtype=np.float64)))
            ),
            "affine_component_removed_downstream": True,
            "surface_support_mode": self.surface_support_mode,
        }
        self.calls.append(stats)
        return residual, stats

    def callback(
        self,
        image_path,
        local_depth,
        face_mask,
        _feature_mask,
        surface_support_mask,
        output_dir,
    ):
        support = (
            surface_support_mask
            if self.surface_support_mode == "selection-subject"
            else face_mask
        )
        residual, stats = self.infer(
            Path(image_path),
            np.asarray(local_depth, dtype=np.float32),
            np.asarray(support),
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "surface_residual.npy", residual)
        return residual, stats

    def provenance(self) -> dict:
        import torch

        peak_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        return {
            "method": self.method,
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "checkpoint_selected_epoch": int(
                self.checkpoint.get("selected_epoch", 0)
            ),
            "selected_alpha": self.alpha,
            "surface_support_mode": self.surface_support_mode,
            "network_size": self.network_size,
            "parameter_count": int(
                sum(parameter.numel() for parameter in self.model.parameters())
            ),
            "inference_calls": len(self.timings),
            "mean_inference_seconds": (
                float(np.mean(self.timings)) if self.timings else None
            ),
            "allocated_before_inference_gb": float(
                self.allocated_before_inference / (1024**3)
            ),
            "peak_allocated_gb": float(peak_allocated / (1024**3)),
            "incremental_peak_vram_gb": float(
                max(0, peak_allocated - self.allocated_before_inference)
                / (1024**3)
            ),
            "calls": self.calls,
        }


def _quality(run_root: Path, row: dict, candidate_path: Path) -> dict:
    part_paths = {
        name: run_root / row["exact_face_part_masks"]["files"][name]["path"]
        for name in FACE_PART_NAMES
    }
    metrics = _exact_face_depth_quality(
        candidate_path,
        run_root / row["exact_depth"]["path"],
        run_root / row["selection_mask"]["path"],
        expected_scale_sign=-1.0,
        part_mask_paths=part_paths,
    )
    check_failures = Counter()
    for part in metrics["named_part_shape"]["parts"]:
        check_failures.update(
            key
            for key, passed in part.get("checks", {}).items()
            if key != "passed" and not passed
        )
    shape_failed = metrics["named_part_shape"]["failed_parts"]
    affine_failed = metrics["named_part_affine_mm"]["failed_parts"]
    return {
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": shape_failed,
        "affine_failed_parts": affine_failed,
        "combined_part_failures": len(shape_failed) + len(affine_failed),
        "shape_check_failures": dict(sorted(check_failures.items())),
    }


def _summary(rows: list[dict]) -> dict:
    check_failures = Counter()
    for row in rows:
        check_failures.update(row.get("shape_check_failures", {}))
    return {
        "row_count": len(rows),
        "combined_part_failures": int(
            sum(row["combined_part_failures"] for row in rows)
        ),
        "median_shape_correlation": float(
            np.median([row["shape_correlation"] for row in rows])
        ),
        "median_gradient_correlation": float(
            np.median([row["gradient_correlation"] for row in rows])
        ),
        "median_normalized_rmse": float(
            np.median([row["normalized_rmse"] for row in rows])
        ),
        "shape_check_failure_counts": dict(sorted(check_failures.items())),
        "rows": rows,
    }


def _current_baseline_equivalent(
    historical: dict,
    reproduced: dict,
    maximum_differences: list[float],
) -> bool:
    return bool(
        len(historical["rows"]) == len(reproduced["rows"])
        and all(value <= 1e-7 for value in maximum_differences)
        and historical["combined_part_failures"]
        == reproduced["combined_part_failures"]
        and abs(
            historical["median_shape_correlation"]
            - reproduced["median_shape_correlation"]
        )
        <= 1e-9
        and abs(
            historical["median_gradient_correlation"]
            - reproduced["median_gradient_correlation"]
        )
        <= 1e-9
        and abs(
            historical["median_normalized_rmse"]
            - reproduced["median_normalized_rmse"]
        )
        <= 1e-9
    )


def _selected_source_path(job_dir: Path, output_root: Path) -> Path:
    metadata = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    context = metadata.get("depth_metadata", {}).get(
        "selection_depth_context",
        {},
    )
    selection_job_id = str(context.get("selection_job_id") or "").strip()
    if not selection_job_id:
        raise ValueError(
            f"Production job {job_dir.name} has no selection job provenance"
        )
    path = output_root / "selection" / selection_job_id / "selected_image.png"
    if not path.is_file():
        raise FileNotFoundError(
            f"Production selected image is unavailable: {path}"
        )
    return path


def _decision(
    baseline: dict,
    candidate: dict,
    *,
    baseline_equivalent: bool,
    residual_contract_passed: bool,
) -> dict:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    candidate_by_id = {row["row_id"]: row for row in candidate["rows"]}
    per_row_no_regression = bool(
        set(baseline_by_id) == set(candidate_by_id)
        and all(
            candidate_by_id[row_id]["combined_part_failures"]
            <= baseline_by_id[row_id]["combined_part_failures"]
            for row_id in baseline_by_id
        )
    )
    critical_checks_no_regression = all(
        int(candidate["shape_check_failure_counts"].get(name, 0))
        <= int(baseline["shape_check_failure_counts"].get(name, 0))
        for name in CRITICAL_PART_CHECKS
    )
    decision = {
        "current_baseline_matches_historical": bool(baseline_equivalent),
        "aggregate_strictly_improves": _strictly_improves(candidate, baseline),
        "per_row_no_failure_regression": per_row_no_regression,
        "hard_small_face_improves": bool(
            HARD_SMALL_FACE_ROW in baseline_by_id
            and HARD_SMALL_FACE_ROW in candidate_by_id
            and candidate_by_id[HARD_SMALL_FACE_ROW]["combined_part_failures"]
            < baseline_by_id[HARD_SMALL_FACE_ROW]["combined_part_failures"]
        ),
        "critical_part_checks_no_regression": bool(
            critical_checks_no_regression
        ),
        "surface_residual_contract_passed": bool(residual_contract_passed),
    }
    decision["eligible_for_full_stl_replay"] = bool(all(decision.values()))
    return decision


def evaluate(
    run_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict:
    run_root = Path(run_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    adapter = FaceSurfaceResidualProvider(checkpoint_path, device)
    historical_rows = []
    reproduced_rows = []
    candidate_rows = []
    maximum_differences = []
    cache_provenance = []
    residual_contract = []
    started = time.perf_counter()

    for row in run_summary["rows"]:
        if row["render"].get("scene_kind") != "face":
            continue
        row_id = row["row_id"]
        job_dir = run_root.parent / row["variants"]["candidate"]["job_id"]
        cached_depth_path = (
            job_dir
            / "face_refinement"
            / "face_00_depth"
            / "output_depth_data.npy"
        )
        cached_crop_path = (
            job_dir / "face_refinement" / "face_00_input.png"
        )
        historical_path = job_dir / "output_depth_data_face_refined.npy"
        selected_source_path = _selected_source_path(
            job_dir,
            run_root.parent,
        )
        historical_rows.append(
            {
                "row_id": row_id,
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                **_quality(run_root, row, historical_path),
            }
        )

        baseline_cache = CachedLocalFaceDepth(
            cached_depth_path,
            cached_crop_path,
        )
        baseline_dir = output_dir / row_id / "baseline"
        reproduced_path, reproduced_metadata = refine_depth_for_faces(
            selected_source_path,
            job_dir / "output_depth_data.npy",
            baseline_dir,
            infer_depth=baseline_cache.callback,
            mode="on",
            detection_roi_mask=run_root / row["selection_mask"]["path"],
        )
        historical_values = np.load(historical_path).astype(np.float32)
        reproduced_values = np.load(reproduced_path).astype(np.float32)
        maximum_difference = float(
            np.max(np.abs(historical_values - reproduced_values))
        )
        maximum_differences.append(maximum_difference)
        reproduced_rows.append(
            {
                "row_id": row_id,
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "refined_faces": reproduced_metadata["refined_faces"],
                "crop_bbox": reproduced_metadata["faces"][0].get("crop_bbox"),
                "historical_maximum_absolute_difference": maximum_difference,
                **_quality(run_root, row, Path(reproduced_path)),
            }
        )

        candidate_cache = CachedLocalFaceDepth(
            cached_depth_path,
            cached_crop_path,
        )
        candidate_dir = output_dir / row_id / "candidate"
        candidate_path, candidate_metadata = refine_depth_for_faces(
            selected_source_path,
            job_dir / "output_depth_data.npy",
            candidate_dir,
            infer_depth=candidate_cache.callback,
            infer_surface_residual=adapter.callback,
            mode="on",
            detection_roi_mask=run_root / row["selection_mask"]["path"],
        )
        face_record = candidate_metadata["faces"][0]
        residual_stats = face_record.get("surface_residual", {})
        residual_contract.append(
            {
                "row_id": row_id,
                "surface_residual_faces": candidate_metadata.get(
                    "surface_residual_faces"
                ),
                "enabled": residual_stats.get("enabled"),
                "method": residual_stats.get("method"),
                "boundary_max_abs_correction": residual_stats.get(
                    "boundary_max_abs_correction"
                ),
                "alignment_anchor": residual_stats.get("alignment_anchor"),
                "surface_support_mode": residual_stats.get(
                    "surface_support_mode"
                ),
                "surface_support_expansion_ratio": residual_stats.get(
                    "surface_support",
                    {},
                ).get("support_expansion_ratio"),
            }
        )
        candidate_rows.append(
            {
                "row_id": row_id,
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "refined_faces": candidate_metadata["refined_faces"],
                "surface_residual_faces": candidate_metadata.get(
                    "surface_residual_faces"
                ),
                "crop_bbox": face_record.get("crop_bbox"),
                "surface_residual": residual_stats,
                **_quality(run_root, row, Path(candidate_path)),
            }
        )
        cache_provenance.append(
            {
                "row_id": row_id,
                "selected_source_sha256": _sha256(selected_source_path),
                "baseline": baseline_cache.provenance(),
                "candidate": candidate_cache.provenance(),
            }
        )

    historical = _summary(historical_rows)
    reproduced = _summary(reproduced_rows)
    candidate = _summary(candidate_rows)
    baseline_equivalent = _current_baseline_equivalent(
        historical,
        reproduced,
        maximum_differences,
    )
    residual_contract_passed = bool(
        residual_contract
        and all(
            record["surface_residual_faces"] == 1
            and record["enabled"] is True
            and record["method"]
            == "bounded-non-affine-face-surface-residual"
            and float(record["boundary_max_abs_correction"]) <= 1e-7
            and record["alignment_anchor"] == "outer-face-ring"
            and record["surface_support_mode"]
            == adapter.surface_support_mode
            for record in residual_contract
        )
    )
    decision = _decision(
        reproduced,
        candidate,
        baseline_equivalent=baseline_equivalent,
        residual_contract_passed=residual_contract_passed,
    )
    evidence = {
        "schema_version": 1,
        "method": "cc0_face_surface_adapter_through_exact_production_refinement",
        "source_revision": run_summary["server_provenance"]["revision"],
        "source_summary_sha256": _sha256(run_root / "summary.json"),
        "implementation_files": {
            "face_depth_refinement_sha256": _sha256(
                Path(__file__).parents[1] / "face_depth_refinement.py"
            ),
            "adapter_training_sha256": _sha256(
                Path(__file__).with_name("train_face_surface_adapter.py")
            ),
            "exact_gate_sha256": _sha256(Path(__file__)),
        },
        "checkpoint": adapter.provenance(),
        "cached_local_depth": cache_provenance,
        "historical_baseline": historical,
        "reproduced_current_baseline": reproduced,
        "candidate": candidate,
        "baseline_maximum_absolute_differences": maximum_differences,
        "surface_residual_contract": residual_contract,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.run_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_full_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
