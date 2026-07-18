"""Replay a trained DAv2 face head through the exact production face pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _strictly_improves,
)
from backend.face_depth_refinement import refine_depth_for_faces


GNM_FUSION_METHOD = "gnm_dav2_multiscale_fusion_head"
GNM_PRODUCTION_FUSION_METHOD = "gnm_dav2_production_surface_fusion_head"
DIRECT_LOCAL_INTEGRATION = "direct-local"
PAIRED_RESIDUAL_INTEGRATION = "paired-residual"
SIZE_AWARE_PAIRED_RESIDUAL_INTEGRATION = "paired-residual-size-aware"
SMALL_FACE_PAIRED_RESIDUAL_INTEGRATION = "paired-residual-small-face-only"
INTEGRATION_MODES = (
    DIRECT_LOCAL_INTEGRATION,
    PAIRED_RESIDUAL_INTEGRATION,
    SIZE_AWARE_PAIRED_RESIDUAL_INTEGRATION,
    SMALL_FACE_PAIRED_RESIDUAL_INTEGRATION,
)
GNM_TRAINING_FACE_HEIGHT_ANCHOR_PIXELS = 90.0
GNM_DETECTOR_SUPPORT_HEIGHT_RATIO = 0.5
RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS = (
    GNM_TRAINING_FACE_HEIGHT_ANCHOR_PIXELS
    * GNM_DETECTOR_SUPPORT_HEIGHT_RATIO
)
RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS = 55.0
ZERO_STRENGTH_REPLAY_METRIC_TOLERANCE = 0.001


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
        raise ValueError("Depth prediction contains no finite values")
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high <= low:
        return values.copy()
    return ((values - low) / (high - low)).astype(np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _resize_depth(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape == shape:
        return values
    image = Image.fromarray(values, mode="F")
    resized = image.resize(
        (int(shape[1]), int(shape[0])),
        resample=Image.Resampling.BICUBIC,
    )
    return np.asarray(resized, dtype=np.float32)


def _face_residual_scale(
    face_mask: np.ndarray,
    *,
    full_strength_height_pixels: float = (
        RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
    ),
) -> tuple[float, int]:
    rows = np.flatnonzero(np.any(np.asarray(face_mask) > 0, axis=1))
    if not rows.size:
        raise ValueError("Cannot scale a face residual for an empty face mask")
    support_height = int(rows[-1] - rows[0] + 1)
    scale = min(
        1.0,
        max(0.0, float(full_strength_height_pixels))
        / max(float(support_height), 1.0),
    )
    return float(scale), support_height


def _small_face_residual_scale(
    face_mask: np.ndarray,
    *,
    full_strength_height_pixels: float = (
        RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
    ),
    zero_strength_height_pixels: float = (
        RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS
    ),
) -> tuple[float, int]:
    rows = np.flatnonzero(np.any(np.asarray(face_mask) > 0, axis=1))
    if not rows.size:
        raise ValueError("Cannot scale a face residual for an empty face mask")
    support_height = int(rows[-1] - rows[0] + 1)
    full_strength = max(float(full_strength_height_pixels), 1.0)
    zero_strength = max(float(zero_strength_height_pixels), full_strength)
    if support_height <= full_strength:
        scale = 1.0
    elif support_height >= zero_strength:
        scale = 0.0
    else:
        scale = (zero_strength - support_height) / (
            zero_strength - full_strength
        )
    return float(scale), support_height


def _checkpoint_mode_and_alpha(checkpoint: dict) -> tuple[str, float]:
    if checkpoint.get("method") in {
        GNM_FUSION_METHOD,
        GNM_PRODUCTION_FUSION_METHOD,
    }:
        required = ("fusion_stage_state_dict", "head_state_dict")
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(
                "GNM fusion checkpoint is missing state dictionaries: "
                + ", ".join(missing)
            )
        mode = "fusion-head"
        alpha = float(checkpoint.get("selected_alpha", -1.0))
    else:
        if "head_state_dict" not in checkpoint:
            raise ValueError("Face-head checkpoint is missing head_state_dict")
        mode = "head-only"
        alpha = float(checkpoint.get("blend_alpha", -1.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Face-depth checkpoint blend alpha must be in [0, 1]")
    return mode, alpha


class BlendedFaceDepth:
    def __init__(self, checkpoint_path: Path, device: str):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_id") != MODEL_ID:
            raise ValueError("Face-head checkpoint model id does not match DAv2 Large")
        if checkpoint.get("model_revision") != MODEL_REVISION:
            raise ValueError("Face-head checkpoint revision does not match the pinned model")
        mode, alpha = _checkpoint_mode_and_alpha(checkpoint)
        snapshot = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            backend="pil",
        )
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=self.dtype,
        ).to(device)
        self.model.eval()
        self.baseline_head = copy.deepcopy(self.model.head).to(
            device=device,
            dtype=torch.float32,
        )
        self.trained_head = copy.deepcopy(self.model.head).to(
            device=device,
            dtype=torch.float32,
        )
        self.trained_head.load_state_dict(checkpoint["head_state_dict"])
        self.baseline_fusion = None
        self.trained_fusion = None
        if mode == "fusion-head":
            self.baseline_fusion = copy.deepcopy(
                self.model.neck.fusion_stage
            ).to(device=device, dtype=torch.float32)
            self.trained_fusion = copy.deepcopy(
                self.model.neck.fusion_stage
            ).to(device=device, dtype=torch.float32)
            self.trained_fusion.load_state_dict(
                checkpoint["fusion_stage_state_dict"]
            )
            self.baseline_fusion.eval()
            self.trained_fusion.eval()
        self.baseline_head.eval()
        self.trained_head.eval()
        self.mode = mode
        self.alpha = alpha
        self.snapshot = Path(snapshot)
        self.checkpoint = checkpoint
        self.timings = []
        self._prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def _prediction_pair(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        import torch

        image_path = Path(image_path)
        stat = image_path.stat()
        cache_key = (
            str(image_path.resolve()),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(device=self.device, dtype=self.dtype)
        patch_height = pixel_values.shape[-2] // self.model.config.patch_size
        patch_width = pixel_values.shape[-1] // self.model.config.patch_size
        started = time.perf_counter()
        with torch.inference_mode():
            backbone = self.model.backbone.forward_with_filtered_kwargs(pixel_values)
            if self.mode == "fusion-head":
                reassembled = self.model.neck.reassemble_stage(
                    backbone.feature_maps,
                    patch_height,
                    patch_width,
                )
                features = [
                    self.model.neck.convs[index](feature).to(
                        dtype=torch.float32
                    )
                    for index, feature in enumerate(reassembled)
                ]
                baseline_hidden = self.baseline_fusion(features)
                trained_hidden = self.trained_fusion(features)
                baseline = self.baseline_head(
                    baseline_hidden,
                    patch_height,
                    patch_width,
                )
                trained = self.trained_head(
                    trained_hidden,
                    patch_height,
                    patch_width,
                )
            else:
                hidden = self.model.neck(
                    backbone.feature_maps,
                    patch_height,
                    patch_width,
                )
                feature = hidden[-1].to(dtype=torch.float32)
                baseline = self.baseline_head(
                    [feature],
                    patch_height,
                    patch_width,
                )
                trained = self.trained_head(
                    [feature],
                    patch_height,
                    patch_width,
                )
            baseline = torch.nn.functional.interpolate(
                baseline[:, None],
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            trained = torch.nn.functional.interpolate(
                trained[:, None],
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        self.timings.append(time.perf_counter() - started)
        pair = (
            baseline.float().cpu().numpy(),
            trained.float().cpu().numpy(),
        )
        self._prediction_cache[cache_key] = pair
        return pair

    def infer(self, image_path: Path, alpha: float | None = None) -> np.ndarray:
        baseline, trained = self._prediction_pair(Path(image_path))
        blend = self.alpha if alpha is None else float(alpha)
        prediction = (1.0 - blend) * baseline + blend * trained
        return _minmax_normalize(prediction)

    def callback(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output_depth_data.npy"
        np.save(path, self.infer(Path(image_path)))
        return str(path)

    def baseline_callback(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output_depth_data.npy"
        np.save(path, self.infer(Path(image_path), alpha=0.0))
        return str(path)

    def provenance(self) -> dict:
        import torch

        peak_vram = (
            float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            if self.device.startswith("cuda")
            else 0.0
        )
        return {
            "checkpoint_method": self.checkpoint.get("method"),
            "trainable_scope": self.mode,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": _sha256(self.snapshot / "model.safetensors"),
            "checkpoint_selected_epoch": int(
                self.checkpoint.get("selected_epoch", 0)
            ),
            "blend_alpha": self.alpha,
            "inference_calls": len(self.timings),
            "prediction_cache_hits": self.cache_hits,
            "prediction_cache_misses": self.cache_misses,
            "mean_inference_seconds": (
                float(np.mean(self.timings)) if self.timings else None
            ),
            "peak_vram_gb": peak_vram,
        }


class PairedFusionSurfaceResidual:
    def __init__(
        self,
        provider: BlendedFaceDepth,
        *,
        size_aware: bool = False,
        small_face_only: bool = False,
    ):
        if size_aware and small_face_only:
            raise ValueError("Only one paired residual scale mode may be active")
        self.provider = provider
        self.size_aware = bool(size_aware)
        self.small_face_only = bool(small_face_only)
        self.calls = []

    def callback(
        self,
        image_path,
        local_depth,
        face_mask,
        _feature_mask,
        _surface_support_mask,
        output_dir,
    ):
        image_path = Path(image_path)
        baseline = self.provider.infer(image_path, alpha=0.0)
        candidate = self.provider.infer(image_path)
        residual = candidate - baseline
        residual_scale = 1.0
        support_height = None
        if self.small_face_only:
            residual_scale, support_height = _small_face_residual_scale(
                face_mask
            )
            residual = residual * residual_scale
        elif self.size_aware:
            residual_scale, support_height = _face_residual_scale(face_mask)
            residual = residual * residual_scale
        comparable_baseline = _resize_depth(
            baseline,
            np.asarray(local_depth).shape,
        )
        local_depth = np.asarray(local_depth, dtype=np.float32)
        stats = {
            "provider": "gnm-dav2-paired-fusion-depth-delta",
            "selected_alpha": self.provider.alpha,
            "residual_scale": residual_scale,
            "size_aware_scaling": self.size_aware,
            "small_face_only_scaling": self.small_face_only,
            "support_height_pixels": support_height,
            "full_strength_support_height_pixels": (
                RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
                if self.size_aware
                else None
            ),
            "zero_strength_support_height_pixels": (
                RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS
                if self.small_face_only
                else None
            ),
            "surface_support_mode": "detector-face",
            "local_baseline_correlation": _correlation(
                local_depth,
                comparable_baseline,
            ),
            "local_baseline_maximum_absolute_difference": float(
                np.max(np.abs(local_depth - comparable_baseline))
            ),
            "raw_scaled_residual_min": float(np.min(residual)),
            "raw_scaled_residual_max": float(np.max(residual)),
            "raw_scaled_residual_rms": float(
                np.sqrt(np.mean(np.square(residual, dtype=np.float64)))
            ),
            "affine_component_removed_downstream": True,
        }
        self.calls.append(stats)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "surface_residual.npy", residual)
        return residual, stats

    def provenance(self) -> dict:
        return {
            "method": "paired-trained-minus-baseline-depth-residual",
            "size_aware_scaling": self.size_aware,
            "small_face_only_scaling": self.small_face_only,
            "full_strength_support_height_pixels": (
                RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
                if self.size_aware
                else None
            ),
            "zero_strength_support_height_pixels": (
                RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS
                if self.small_face_only
                else None
            ),
            "calls": self.calls,
            "all_local_baselines_equivalent": bool(
                self.calls
                and all(
                    call["local_baseline_correlation"] >= 0.999
                    and call[
                        "local_baseline_maximum_absolute_difference"
                    ]
                    <= 0.02
                    for call in self.calls
                )
            ),
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
    return {
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": metrics["named_part_shape"]["failed_parts"],
        "affine_failed_parts": metrics["named_part_affine_mm"]["failed_parts"],
        "combined_part_failures": len(metrics["named_part_shape"]["failed_parts"])
        + len(metrics["named_part_affine_mm"]["failed_parts"]),
    }


def _summary(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "combined_part_failures": sum(
            row["combined_part_failures"] for row in rows
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
        "rows": rows,
    }


def _small_face_only_decision_contract(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
) -> dict:
    baseline_by_id = {row["row_id"]: row for row in baseline_rows}
    candidate_by_id = {row["row_id"]: row for row in candidate_rows}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Small-face replay rows do not match the baseline")
    affected = []
    zero_strength = []
    invalid_scale_rows = []
    for row_id, candidate in candidate_by_id.items():
        surface = candidate.get("surface_residual") or {}
        scale = float(surface.get("residual_scale", np.nan))
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            invalid_scale_rows.append(row_id)
        elif scale > 0.0:
            affected.append(candidate)
        else:
            zero_strength.append(candidate)
    affected_baseline = [baseline_by_id[row["row_id"]] for row in affected]
    affected_strictly_improves = bool(
        affected
        and _strictly_improves(
            _summary(affected),
            _summary(affected_baseline),
        )
    )
    zero_records = []
    for candidate in zero_strength:
        baseline = baseline_by_id[candidate["row_id"]]
        surface = candidate.get("surface_residual") or {}
        metric_drift = {
            name: abs(float(candidate[name]) - float(baseline[name]))
            for name in (
                "shape_correlation",
                "gradient_correlation",
                "normalized_rmse",
            )
        }
        record = {
            "row_id": candidate["row_id"],
            "combined_part_failures_equal": (
                candidate["combined_part_failures"]
                == baseline["combined_part_failures"]
            ),
            "maximum_absolute_correction": float(
                surface.get("max_abs_correction", np.inf)
            ),
            "metric_drift": metric_drift,
        }
        record["equivalent"] = bool(
            record["combined_part_failures_equal"]
            and record["maximum_absolute_correction"] <= 1e-8
            and all(
                drift <= ZERO_STRENGTH_REPLAY_METRIC_TOLERANCE
                for drift in metric_drift.values()
            )
        )
        zero_records.append(record)
    zero_strength_rows_equivalent = bool(
        zero_records and all(record["equivalent"] for record in zero_records)
    )
    checks = {
        "all_scales_valid": not invalid_scale_rows,
        "at_least_one_affected_row": bool(affected),
        "affected_rows_strictly_improve": affected_strictly_improves,
        "at_least_one_zero_strength_row": bool(zero_records),
        "zero_strength_rows_equivalent": zero_strength_rows_equivalent,
    }
    return {
        "passes": bool(all(checks.values())),
        "checks": checks,
        "affected_row_ids": [row["row_id"] for row in affected],
        "zero_strength_rows": zero_records,
        "invalid_scale_row_ids": invalid_scale_rows,
        "zero_strength_metric_tolerance": ZERO_STRENGTH_REPLAY_METRIC_TOLERANCE,
    }


def evaluate(
    run_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    integration_mode: str = DIRECT_LOCAL_INTEGRATION,
) -> dict:
    run_root = Path(run_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    integration_mode = str(integration_mode).strip().lower()
    if integration_mode not in INTEGRATION_MODES:
        raise ValueError(
            f"Unsupported integration mode {integration_mode!r}; "
            f"expected one of {INTEGRATION_MODES}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    provider = BlendedFaceDepth(checkpoint_path, device)
    surface_provider = (
        PairedFusionSurfaceResidual(
            provider,
            size_aware=(
                integration_mode
                == SIZE_AWARE_PAIRED_RESIDUAL_INTEGRATION
            ),
            small_face_only=(
                integration_mode
                == SMALL_FACE_PAIRED_RESIDUAL_INTEGRATION
            ),
        )
        if integration_mode
        in {
            PAIRED_RESIDUAL_INTEGRATION,
            SIZE_AWARE_PAIRED_RESIDUAL_INTEGRATION,
            SMALL_FACE_PAIRED_RESIDUAL_INTEGRATION,
        }
        else None
    )
    baseline_rows = []
    candidate_rows = []
    equivalence = []
    started = time.perf_counter()
    for row in run_summary["rows"]:
        if row["render"].get("scene_kind") != "face":
            continue
        job_dir = run_root.parent / row["variants"]["candidate"]["job_id"]
        cached_local = np.load(
            job_dir
            / "face_refinement"
            / "face_00_depth"
            / "output_depth_data.npy"
        ).astype(np.float32)
        crop_path = job_dir / "face_refinement" / "face_00_input.png"
        reproduced = provider.infer(crop_path, alpha=0.0)
        equivalence.append(
            {
                "row_id": row["row_id"],
                "cached_shape": list(cached_local.shape),
                "reproduced_shape": list(reproduced.shape),
                "correlation": _correlation(cached_local, reproduced),
                "maximum_absolute_difference": float(
                    np.max(np.abs(cached_local - reproduced))
                ),
            }
        )
        baseline_path = job_dir / "output_depth_data_face_refined.npy"
        baseline_rows.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                **_quality(run_root, row, baseline_path),
            }
        )
        row_output = output_dir / row["row_id"]
        refined_path, metadata = refine_depth_for_faces(
            run_root / row["source"]["path"],
            job_dir / "output_depth_data.npy",
            row_output,
            infer_depth=(
                provider.baseline_callback
                if surface_provider is not None
                else provider.callback
            ),
            mode="on",
            detection_roi_mask=run_root / row["selection_mask"]["path"],
            infer_surface_residual=(
                surface_provider.callback
                if surface_provider is not None
                else None
            ),
        )
        candidate_rows.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "refined_faces": metadata["refined_faces"],
                "crop_bbox": metadata["faces"][0].get("crop_bbox"),
                "max_abs_correction": metadata["faces"][0].get(
                    "max_abs_correction"
                ),
                "surface_residual": metadata["faces"][0].get(
                    "surface_residual"
                ),
                **_quality(run_root, row, Path(refined_path)),
            }
        )
    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    by_id = {row["row_id"]: row for row in candidate_rows}
    baseline_by_id = {row["row_id"]: row for row in baseline_rows}
    per_row_no_regression = all(
        by_id[row_id]["combined_part_failures"]
        <= baseline_by_id[row_id]["combined_part_failures"]
        for row_id in by_id
    )
    hard_row = "small_side_lit_shelves_256"
    decision = {
        "local_baseline_equivalent": bool(
            all(
                row["cached_shape"] == row["reproduced_shape"]
                and row["correlation"] >= 0.999
                and row["maximum_absolute_difference"] <= 0.02
                for row in equivalence
            )
        ),
        "aggregate_strictly_improves": _strictly_improves(candidate, baseline),
        "per_row_no_failure_regression": per_row_no_regression,
        "hard_small_face_improves": bool(
            by_id[hard_row]["combined_part_failures"]
            < baseline_by_id[hard_row]["combined_part_failures"]
        ),
    }
    small_face_only_contract = None
    if surface_provider is not None and surface_provider.small_face_only:
        small_face_only_contract = _small_face_only_decision_contract(
            baseline_rows,
            candidate_rows,
        )
        decision["small_face_only_contract_passes"] = bool(
            small_face_only_contract["passes"]
        )
    if surface_provider is not None:
        decision["paired_local_baseline_equivalent"] = bool(
            surface_provider.provenance()["all_local_baselines_equivalent"]
        )
    if small_face_only_contract is not None:
        decision["eligible_for_full_stl_replay"] = bool(
            decision["local_baseline_equivalent"]
            and decision["paired_local_baseline_equivalent"]
            and decision["per_row_no_failure_regression"]
            and decision["hard_small_face_improves"]
            and decision["small_face_only_contract_passes"]
        )
    else:
        decision["eligible_for_full_stl_replay"] = bool(all(decision.values()))
    if surface_provider is not None and surface_provider.small_face_only:
        method = (
            "trained_dav2_fusion_delta_through_small_face_only_"
            "bounded_surface_residual"
        )
    elif surface_provider is not None and surface_provider.size_aware:
        method = (
            "trained_dav2_fusion_delta_through_size_aware_bounded_"
            "surface_residual"
        )
    elif surface_provider is not None:
        method = (
            "trained_dav2_fusion_delta_through_bounded_surface_residual"
        )
    elif provider.mode == "fusion-head":
        method = (
            "trained_dav2_fusion_head_through_unchanged_face_refinement"
        )
    else:
        method = "trained_dav2_head_blend_through_unchanged_face_refinement"
    evidence = {
        "schema_version": 1,
        "method": method,
        "integration_mode": integration_mode,
        "source_revision": run_summary["server_provenance"]["revision"],
        "source_summary_sha256": _sha256(run_root / "summary.json"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "provider": provider.provenance(),
        "surface_residual_provider": (
            surface_provider.provenance()
            if surface_provider is not None
            else None
        ),
        "local_baseline_equivalence": equivalence,
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "small_face_only_contract": small_face_only_contract,
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
    parser.add_argument(
        "--integration-mode",
        choices=INTEGRATION_MODES,
        default=DIRECT_LOCAL_INTEGRATION,
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.run_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        integration_mode=args.integration_mode,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_full_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
