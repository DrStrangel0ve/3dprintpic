"""Train a bounded face-depth rectified-flow pilot and matched direct control."""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.c3i_synface_corpus import verified_corpus_asset
from backend.benchmark.face_depth_rectified_flow import (
    GeometryLossWeights,
    RectifiedFlowConfig,
    build_static_conditioning,
    build_unet,
    camera_rays_from_intrinsics,
    deterministic_geometry_loss,
    flow_training_state,
    geometric_flow_loss,
    median_absolute_deviation,
    parameter_count,
    posterior_medoid,
    project_flow_state_to_support,
    sample_rectified_flow,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    PHOTO_DOMAIN_AUGMENTATION,
    _augment_training_crop,
    _padded_box,
    _validate_training_corpus_summary,
)


TRAINING_SEED = 20260719
DEPTH_OFFSET = 1.0
MAX_PEAK_VRAM_GB = 11.0
MAX_RUNTIME_SECONDS = 2.0 * 60.0 * 60.0
SMALL_FACE_HEIGHT_MAX_PIXELS = 96
TURNED_FACE_YAW_MIN_DEGREES = 18.0
OWNED_PROVENANCE_PATHS = (
    "backend/benchmark/c3i_synface_corpus.py",
    "backend/benchmark/face_depth_rectified_flow.py",
    "backend/benchmark/mhr_face_training_corpus.py",
    "backend/benchmark/run_cc0_live_face_variation_matrix.py",
    "backend/benchmark/train_face_depth_head.py",
    "backend/benchmark/train_face_depth_rectified_flow.py",
    "backend/tests/test_face_depth_rectified_flow.py",
    "backend/tests/test_train_face_depth_rectified_flow.py",
)


@dataclass(frozen=True)
class PilotConfig:
    steps: int = 1200
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    evaluation_interval: int = 300
    gradient_clip_norm: float = 1.0
    max_runtime_seconds: float = MAX_RUNTIME_SECONDS
    max_peak_vram_gb: float = MAX_PEAK_VRAM_GB
    evaluation_reserve_seconds: float = 600.0
    training_augmentation: str = PHOTO_DOMAIN_AUGMENTATION
    required_gpu_name_substring: str = "RTX 3080 Ti"

    def validated(self) -> "PilotConfig":
        if not 1 <= int(self.steps) <= 60_000:
            raise ValueError("Pilot steps must be in [1, 60000]")
        if not 1 <= int(self.batch_size) <= 64:
            raise ValueError("Pilot batch size must be in [1, 64]")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 0.01:
            raise ValueError("Pilot learning rate must be in (0, 0.01]")
        if not math.isfinite(self.weight_decay) or not 0.0 <= self.weight_decay <= 1.0:
            raise ValueError("Pilot weight decay must be in [0, 1]")
        if not 1 <= int(self.evaluation_interval) <= int(self.steps):
            raise ValueError("Pilot evaluation interval must fit the step budget")
        if not math.isfinite(self.gradient_clip_norm) or not 0.0 < self.gradient_clip_norm <= 100.0:
            raise ValueError("Pilot gradient clip norm must be in (0, 100]")
        if not 60.0 <= float(self.max_runtime_seconds) <= MAX_RUNTIME_SECONDS:
            raise ValueError("Pilot runtime cap must be between 60 seconds and 2 hours")
        if not 0.5 <= float(self.max_peak_vram_gb) <= MAX_PEAK_VRAM_GB:
            raise ValueError("Pilot VRAM cap must be between 0.5 and 11 GiB")
        if not 30.0 <= float(self.evaluation_reserve_seconds) <= 1800.0:
            raise ValueError("Pilot evaluation reserve must be in [30, 1800] seconds")
        if self.training_augmentation != PHOTO_DOMAIN_AUGMENTATION:
            raise ValueError("The bounded pilot requires photo-domain-v1 augmentation")
        if not self.required_gpu_name_substring.strip():
            raise ValueError("The bounded pilot requires an explicit GPU name guard")
        return self


@dataclass
class PreparedFace:
    row: dict
    static: object
    coarse: object
    target: object
    face: object
    parts: tuple[object, ...]
    boundary: object
    rays: object
    bbox: tuple[int, int, int, int]
    source_coarse: np.ndarray
    source_face: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_unit_depth(values: np.ndarray, face_mask: np.ndarray, *, inverse: bool) -> np.ndarray:
    depth = np.asarray(values, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool) & np.isfinite(depth)
    if depth.ndim != 2 or face.shape != depth.shape or np.count_nonzero(face) < 64:
        raise ValueError("Face depth normalization needs at least 64 valid pixels")
    low, high = np.percentile(depth[face], (2.0, 98.0))
    normalized = (depth - float(low)) / max(float(high - low), 1e-6)
    if inverse:
        normalized = 1.0 - normalized
    normalized = np.clip(normalized, -0.5, 1.5)
    return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)


def _attachment_boundary(face_mask: np.ndarray, width: int = 2) -> np.ndarray:
    face = np.asarray(face_mask, dtype=np.uint8)
    width = int(width)
    if face.ndim != 2 or not np.any(face):
        raise ValueError("Attachment boundary needs a nonempty 2D face mask")
    if not 1 <= width <= 8:
        raise ValueError("Attachment boundary width must be in [1, 8]")
    kernel = np.ones((2 * width + 1, 2 * width + 1), dtype=np.uint8)
    eroded = cv2.erode(face, kernel, iterations=1)
    return (face.astype(bool) & ~eroded.astype(bool)).astype(np.float32)


def _nearest_face_fill(values: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    values = np.asarray(values, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or face.shape != values.shape or not np.any(face):
        raise ValueError("Nearest face fill needs finite 2D supported values")
    _, indices = distance_transform_edt(~face, return_indices=True)
    filled = values.copy()
    filled[~face] = values[tuple(indices[:, ~face])]
    if not np.isfinite(filled).all():
        raise ValueError("Nearest face fill produced invalid values")
    return filled


def _select_rows(rows: Sequence[dict], limit_per_split: int | None) -> list[dict]:
    if limit_per_split is None:
        selected = list(rows)
    else:
        limit = int(limit_per_split)
        if not 1 <= limit <= 40:
            raise ValueError("Smoke row limit must be in [1, 40]")
        selected = []
        for split in ("train", "validation", "sealed"):
            selected.extend([row for row in rows if row["split"] == split][:limit])
    if {row["split"] for row in selected} != {"train", "validation", "sealed"}:
        raise ValueError("Pilot selection must retain every identity-disjoint split")
    return selected


def _source_resolution_candidate(
    source_coarse_depth: np.ndarray,
    source_face_mask: np.ndarray,
    low_resolution_residual: np.ndarray,
    low_resolution_support: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    coarse = np.asarray(source_coarse_depth, dtype=np.float32)
    face = np.asarray(source_face_mask, dtype=bool)
    residual = np.asarray(low_resolution_residual, dtype=np.float32)
    support = (
        np.ones(residual.shape, dtype=np.float32)
        if low_resolution_support is None
        else np.asarray(low_resolution_support, dtype=np.float32)
    )
    if coarse.ndim != 2 or face.shape != coarse.shape:
        raise ValueError("Source-resolution depth and face support must match")
    if residual.ndim != 2 or support.shape != residual.shape:
        raise ValueError("Low-resolution residual and support must match")
    if not np.isfinite(coarse).all() or not np.isfinite(residual).all():
        raise ValueError("Source-resolution emission inputs must be finite")
    numerator = cv2.resize(
        residual * support,
        (coarse.shape[1], coarse.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    denominator = cv2.resize(
        support,
        (coarse.shape[1], coarse.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.where(denominator > 1e-3, numerator / np.maximum(denominator, 1e-3), 0.0)
    candidate = np.where(face, coarse + resized, coarse).astype(np.float32)
    return candidate, bool(np.array_equal(candidate[~face], coarse[~face]))


def _resize(values: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(np.asarray(values), (size, size), interpolation=interpolation)


def _load_depth_model(device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    snapshot = Path(
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    )
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dtype = torch.float16
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return processor, model, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_sha256": _sha256(snapshot / "model.safetensors"),
        "processor_sha256": _sha256(snapshot / "preprocessor_config.json"),
        "local_files_only": True,
    }


def _predict_coarse(source: Image.Image, processor, model, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    dtype = next(model.parameters()).dtype
    pixels = processor(images=source, return_tensors="pt")["pixel_values"].to(
        device=device,
        dtype=dtype,
    )
    patch_height = pixels.shape[-2] // model.config.patch_size
    patch_width = pixels.shape[-1] // model.config.patch_size
    context = torch.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()
    with torch.inference_mode(), context:
        backbone = model.backbone.forward_with_filtered_kwargs(pixels)
        hidden = model.neck(backbone.feature_maps, patch_height, patch_width)
        prediction = model.head(hidden, patch_height, patch_width)
        prediction = functional.interpolate(
            prediction[:, None],
            size=(source.height, source.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    return prediction.float().cpu().numpy()


def _prepare_one(
    corpus_root: Path,
    row: dict,
    processor,
    depth_model,
    device: str,
    model_config: RectifiedFlowConfig,
) -> PreparedFace:
    import torch

    source = Image.open(verified_corpus_asset(corpus_root, row["source"])).convert("RGB")
    exact = np.load(verified_corpus_asset(corpus_root, row["exact_camera_depth"])).astype(np.float32)
    full_face = np.asarray(
        Image.open(verified_corpus_asset(corpus_root, row["selection_mask"])).convert("L")
    ) >= 128
    if exact.shape != full_face.shape or exact.shape != (source.height, source.width):
        raise ValueError("Flow source, target, and support shapes must match")
    bbox = _padded_box(row["render"]["face_bbox_xyxy"], source.width, source.height)
    x0, y0, x1, y1 = bbox
    source_face = full_face[y0:y1, x0:x1]
    crop = source.crop(bbox)
    inference_crop = (
        _augment_training_crop(crop, str(row["row_id"]), 1, PHOTO_DOMAIN_AUGMENTATION)
        if row["split"] == "train"
        else crop
    )
    coarse_native = _predict_coarse(inference_crop, processor, depth_model, device)
    source_coarse = DEPTH_OFFSET + _robust_unit_depth(
        coarse_native,
        source_face,
        inverse=True,
    )
    size = int(model_config.sample_size)
    face = _resize(source_face.astype(np.uint8), size, cv2.INTER_NEAREST).astype(bool)
    coarse = _resize(source_coarse, size, cv2.INTER_CUBIC).astype(np.float32)
    target_normalized = _nearest_face_fill(
        _robust_unit_depth(exact, full_face, inverse=False),
        full_face,
    )
    target_depth = DEPTH_OFFSET + _resize(
        target_normalized[y0:y1, x0:x1],
        size,
        cv2.INTER_CUBIC,
    )
    boundary = _attachment_boundary(face)
    target = np.where(face, target_depth - coarse, 0.0).astype(np.float32)
    target[boundary > 0.5] = 0.0
    parts = []
    for name in FACE_PART_NAMES:
        part = np.asarray(
            Image.open(verified_corpus_asset(corpus_root, row["exact_face_parts"][name])).convert("L")
        ) >= 128
        parts.append(
            _resize(part[y0:y1, x0:x1].astype(np.uint8), size, cv2.INTER_NEAREST).astype(np.float32)
        )
    rgb = np.asarray(inference_crop.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
    rgb = (rgb / 127.5) - 1.0
    rays = camera_rays_from_intrinsics(
        size,
        size,
        row["render"]["camera"]["intrinsics"],
        bbox,
        dtype=torch.float32,
    )[0]
    face_tensor = torch.from_numpy(face.astype(np.float32))[None]
    coarse_tensor = torch.from_numpy(coarse)[None]
    static = build_static_conditioning(
        torch.from_numpy(np.moveaxis(rgb, -1, 0))[None],
        coarse_tensor[None],
        torch.ones_like(coarse_tensor)[None],
        face_tensor[None],
        rays[None],
        torch.ones((1,), dtype=torch.float32),
    )[0]
    return PreparedFace(
        row=row,
        static=static.to(torch.float16),
        coarse=coarse_tensor,
        target=torch.from_numpy(target)[None],
        face=face_tensor,
        parts=tuple(torch.from_numpy(part)[None] for part in parts),
        boundary=torch.from_numpy(boundary)[None],
        rays=rays,
        bbox=bbox,
        source_coarse=source_coarse,
        source_face=source_face,
    )


def _prepare_rows(
    corpus_root: Path,
    rows: Sequence[dict],
    device: str,
    model_config: RectifiedFlowConfig,
):
    import torch

    started = time.perf_counter()
    processor, depth_model, provenance = _load_depth_model(device)
    prepared = []
    try:
        for index, row in enumerate(rows, start=1):
            prepared.append(
                _prepare_one(corpus_root, row, processor, depth_model, device, model_config)
            )
            if index % 40 == 0 or index == len(rows):
                print(f"prepared {index}/{len(rows)} {row['split']} rows", flush=True)
    finally:
        depth_model.to("cpu")
        del depth_model
        torch.cuda.empty_cache()
    return prepared, {
        **provenance,
        "row_count": len(prepared),
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
    }


def _batch(items: Sequence[PreparedFace], device: str) -> dict:
    import torch

    def stack(name: str):
        return torch.stack([getattr(item, name) for item in items]).to(
            device=device,
            dtype=torch.float32,
        )

    return {
        "static": stack("static"),
        "coarse": stack("coarse"),
        "target": stack("target"),
        "face": stack("face"),
        "boundary": stack("boundary"),
        "rays": stack("rays"),
        "parts": [
            torch.stack([item.parts[index] for item in items]).to(device=device, dtype=torch.float32)
            for index in range(len(FACE_PART_NAMES))
        ],
    }


def _model_output(model, state, static, time_values):
    import torch

    timestep = torch.as_tensor(time_values, device=state.device, dtype=state.dtype)
    if timestep.ndim == 0:
        timestep = timestep.expand(state.shape[0])
    return model(torch.cat((state, static), dim=1), timestep * 1000.0).sample


def _candidate_quality(
    corpus_root: Path,
    item: PreparedFace,
    residual: np.ndarray,
    output_dir: Path,
    label: str,
) -> dict:
    exact_path = verified_corpus_asset(corpus_root, item.row["exact_camera_depth"])
    exact = np.load(exact_path)
    full_face = np.asarray(
        Image.open(verified_corpus_asset(corpus_root, item.row["selection_mask"])).convert("L")
    ) >= 128
    candidate_crop, crop_bit_exact = _source_resolution_candidate(
        item.source_coarse,
        item.source_face,
        residual,
        item.face[0].numpy(),
    )
    x0, y0, x1, y1 = item.bbox
    correction = np.zeros(exact.shape, dtype=np.float32)
    correction_crop = candidate_crop - item.source_coarse
    correction[y0:y1, x0:x1] = correction_crop
    full_background_bit_exact = bool(
        np.array_equal(correction[~full_face], np.zeros(np.count_nonzero(~full_face), dtype=np.float32))
    )
    candidate = np.full(exact.shape, np.nan, dtype=np.float32)
    candidate[y0:y1, x0:x1] = candidate_crop
    candidate_path = output_dir / label / f"{item.row['row_id']}.npy"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(candidate_path, candidate)
    metrics = _exact_face_depth_quality(
        candidate_path,
        exact_path,
        verified_corpus_asset(corpus_root, item.row["selection_mask"]),
        expected_scale_sign=1.0,
        part_mask_paths={
            name: verified_corpus_asset(corpus_root, item.row["exact_face_parts"][name])
            for name in FACE_PART_NAMES
        },
    )
    return {
        "row_id": item.row["row_id"],
        "identity_group": item.row["identity_group"],
        "expression": item.row["expression"],
        "face_height_pixels": item.row["render"]["face_bbox_height_pixels"],
        "camera_yaw_degrees": item.row["spec"]["camera_yaw_deg"],
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": metrics["named_part_shape"]["failed_parts"],
        "affine_failed_parts": metrics["named_part_affine_mm"]["failed_parts"],
        "crop_background_bit_exact": crop_bit_exact,
        "full_image_correction_background_bit_exact": full_background_bit_exact,
    }


def _summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    return {
        "row_count": len(rows),
        "combined_part_failures": sum(
            len(row["shape_failed_parts"]) + len(row["affine_failed_parts"])
            for row in rows
        ),
        "median_shape_correlation": float(np.median([row["shape_correlation"] for row in rows])),
        "median_gradient_correlation": float(np.median([row["gradient_correlation"] for row in rows])),
        "median_normalized_rmse": float(np.median([row["normalized_rmse"] for row in rows])),
        "source_background_bit_exact": all(
            row["crop_background_bit_exact"]
            and row["full_image_correction_background_bit_exact"]
            for row in rows
        ),
        "rows": list(rows),
    }


def _small_turned(summary: dict) -> dict:
    rows = [
        row
        for row in summary["rows"]
        if int(row["face_height_pixels"]) <= SMALL_FACE_HEIGHT_MAX_PIXELS
        and abs(float(row["camera_yaw_degrees"])) >= TURNED_FACE_YAW_MIN_DEGREES
    ]
    if not rows:
        raise ValueError("Evaluation lacks small turned-face rows")
    return _summarize(rows)


def _quality_rank(summary: dict) -> tuple:
    small = _small_turned(summary)
    return (
        small["combined_part_failures"],
        small["median_normalized_rmse"],
        -small["median_gradient_correlation"],
        -small["median_shape_correlation"],
        summary["combined_part_failures"],
    )


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    from scipy.stats import spearmanr

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.ptp(first) <= 1e-12 or np.ptp(second) <= 1e-12:
        return 0.0
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def _calibrated_uncertainty(
    mad_values: np.ndarray,
    error_values: np.ndarray,
    calibration_scale_90: float | None,
) -> dict:
    mad = np.asarray(mad_values, dtype=np.float64).reshape(-1)
    error = np.asarray(error_values, dtype=np.float64).reshape(-1)
    if mad.shape != error.shape or not mad.size:
        raise ValueError("Uncertainty calibration needs paired nonempty values")
    if not np.isfinite(mad).all() or not np.isfinite(error).all():
        raise ValueError("Uncertainty calibration values must be finite")
    if np.any(mad < 0.0) or np.any(error < 0.0):
        raise ValueError("Uncertainty calibration values must be nonnegative")
    effective_mad = np.maximum(mad, 1e-6)
    mode = "sealed-using-validation-scale"
    scale = calibration_scale_90
    if scale is None:
        scale = float(np.quantile(error / effective_mad, 0.90, method="higher"))
        mode = "validation-fit"
    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("Uncertainty calibration scale must be finite and nonnegative")
    return {
        "mad_absolute_error_spearman": _spearman(mad, error),
        "validation_calibrated_90_interval_coverage": float(
            np.mean(error <= scale * effective_mad)
        ),
        "calibration_scale_90": scale,
        "calibration_mode": mode,
        "supported_pixel_count": int(mad.size),
        "raw_sample_quantile_claimed": False,
    }


def _evaluate_control(
    corpus_root: Path,
    items: Sequence[PreparedFace],
    model,
    device: str,
    output_dir: Path,
    label: str,
) -> dict:
    import torch

    model.eval()
    rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for item in items:
            batch = _batch([item], device)
            zero = torch.zeros_like(batch["target"])
            residual = _model_output(model, zero, batch["static"], zero.new_zeros((1,)))
            rows.append(
                _candidate_quality(
                    corpus_root,
                    item,
                    residual[0, 0].float().cpu().numpy(),
                    output_dir,
                    label,
                )
            )
    result = _summarize(rows)
    result["small_turned"] = _small_turned(result)
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def _evaluate_flow(
    corpus_root: Path,
    items: Sequence[PreparedFace],
    model,
    device: str,
    config: RectifiedFlowConfig,
    output_dir: Path,
    label: str,
    *,
    calibration_scale_90: float | None = None,
) -> dict:
    import torch

    model.eval()
    rows = []
    uncertainty = []
    errors = []
    started = time.perf_counter()
    with torch.inference_mode():
        for item in items:
            batch = _batch([item], device)
            samples = sample_rectified_flow(
                model,
                batch["static"],
                seeds=config.inference_seeds,
                steps=config.inference_steps,
                noise_sigma=config.noise_sigma,
            )
            residual, _ = posterior_medoid(samples, batch["face"])
            rows.append(
                _candidate_quality(
                    corpus_root,
                    item,
                    residual[0, 0].float().cpu().numpy(),
                    output_dir,
                    label,
                )
            )
            support = batch["face"].bool()
            uncertainty.append(median_absolute_deviation(samples)[support].cpu().numpy())
            errors.append(torch.abs(residual - batch["target"])[support].cpu().numpy())
    result = _summarize(rows)
    result["small_turned"] = _small_turned(result)
    result["runtime_seconds"] = time.perf_counter() - started
    result["uncertainty"] = _calibrated_uncertainty(
        np.concatenate(uncertainty),
        np.concatenate(errors),
        calibration_scale_90,
    )
    return result


def _paired_wins(candidate: dict, control: dict) -> dict:
    control_rows = {row["row_id"]: row for row in control["rows"]}
    records = []
    wins = 0
    ties = 0
    for row in candidate["rows"]:
        incumbent = control_rows[row["row_id"]]
        candidate_rank = (
            len(row["shape_failed_parts"]) + len(row["affine_failed_parts"]),
            row["normalized_rmse"],
            -row["gradient_correlation"],
            -row["shape_correlation"],
        )
        control_rank = (
            len(incumbent["shape_failed_parts"]) + len(incumbent["affine_failed_parts"]),
            incumbent["normalized_rmse"],
            -incumbent["gradient_correlation"],
            -incumbent["shape_correlation"],
        )
        outcome = "tie" if candidate_rank == control_rank else (
            "win" if candidate_rank < control_rank else "loss"
        )
        wins += int(outcome == "win")
        ties += int(outcome == "tie")
        records.append({"row_id": row["row_id"], "outcome": outcome})
    return {
        "row_count": len(records),
        "wins": wins,
        "ties": ties,
        "win_rate": float(wins / len(records)) if records else 0.0,
        "rows": records,
    }


def _flow_advance_gate(flow: dict, control: dict, run_checks: dict[str, bool]) -> dict:
    paired = _paired_wins(flow["small_turned"], control["small_turned"])
    control_failures = int(control["small_turned"]["combined_part_failures"])
    flow_failures = int(flow["small_turned"]["combined_part_failures"])
    reduction = (
        float((control_failures - flow_failures) / control_failures)
        if control_failures
        else 0.0
    )
    uncertainty = flow["uncertainty"]
    checks = {
        **{str(name): bool(value) for name, value in run_checks.items()},
        "paired_win_rate_at_least_0_8": paired["win_rate"] >= 0.8,
        "part_failure_reduction_at_least_0_25": reduction >= 0.25,
        "uncertainty_spearman_at_least_0_35": (
            uncertainty["mad_absolute_error_spearman"] >= 0.35
        ),
        "validation_calibrated_90_coverage_in_0_85_0_95": (
            0.85
            <= uncertainty["validation_calibrated_90_interval_coverage"]
            <= 0.95
        ),
        "overall_shape_not_regressed": (
            flow["median_shape_correlation"] >= control["median_shape_correlation"]
        ),
        "overall_gradient_not_regressed": (
            flow["median_gradient_correlation"] >= control["median_gradient_correlation"]
        ),
        "overall_rmse_not_regressed": (
            flow["median_normalized_rmse"] <= control["median_normalized_rmse"]
        ),
        "source_background_bit_exact": (
            flow["source_background_bit_exact"]
            and control["source_background_bit_exact"]
        ),
    }
    return {
        "decision": "advance-to-exact-hard-row" if all(checks.values()) else "hold",
        "checks": checks,
        "paired_small_turned": paired,
        "small_turned_part_failure_reduction": reduction,
    }


def _code_provenance() -> dict:
    repo = Path(__file__).resolve().parents[2]
    files = {}
    for relative in OWNED_PROVENANCE_PATHS:
        path = repo / relative
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ("git", "status", "--porcelain", "--", *OWNED_PROVENANCE_PATHS),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty = ["git-status-unavailable"]
    dependency_files = {}
    for relative in ("backend/requirements.txt", "requirements.txt", "pyproject.toml"):
        path = repo / relative
        if path.is_file():
            dependency_files[relative] = _sha256(path)
    packages = {}
    for name in ("diffusers", "transformers", "numpy", "scipy", "opencv-python", "Pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_revision": revision,
        "owned_git_status": dirty,
        "files": files,
        "dependency_files": dependency_files,
        "package_versions": packages,
    }


def _hardware_provenance(device: str) -> dict:
    import torch

    requested = torch.device(device)
    index = requested.index if requested.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "requested_device": str(device),
        "resolved_cuda_device_index": index,
        "gpu_name": properties.name,
        "gpu_total_vram_gb": float(properties.total_memory / (1024**3)),
        "compute_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def _train_step(
    flow_model,
    control_model,
    flow_optimizer,
    control_optimizer,
    batch: dict,
    model_config: RectifiedFlowConfig,
    loss_weights: GeometryLossWeights,
    generator,
    gradient_clip_norm: float,
) -> dict:
    import torch

    target = batch["target"]
    times = torch.rand(
        (target.shape[0],),
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    noise = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    state, target_velocity = flow_training_state(
        target,
        times,
        noise,
        model_config.noise_sigma,
    )
    state = project_flow_state_to_support(state, batch["static"])
    flow_model.train()
    flow_optimizer.zero_grad(set_to_none=True)
    predicted_velocity = _model_output(flow_model, state, batch["static"], times)
    flow_loss, flow_components, _ = geometric_flow_loss(
        predicted_velocity,
        target_velocity,
        state,
        times,
        target,
        batch["face"],
        batch["parts"],
        batch["boundary"],
        coarse_depth=batch["coarse"],
        camera_rays=batch["rays"],
        weights=loss_weights,
    )
    flow_loss.backward()
    flow_norm = torch.nn.utils.clip_grad_norm_(flow_model.parameters(), gradient_clip_norm)
    flow_optimizer.step()

    control_model.train()
    control_optimizer.zero_grad(set_to_none=True)
    zero = torch.zeros_like(target)
    control_residual = _model_output(
        control_model,
        zero,
        batch["static"],
        zero.new_zeros((target.shape[0],)),
    )
    control_loss, control_components, _ = deterministic_geometry_loss(
        control_residual,
        target,
        batch["face"],
        batch["parts"],
        batch["boundary"],
        coarse_depth=batch["coarse"],
        camera_rays=batch["rays"],
        weights=loss_weights,
    )
    control_loss.backward()
    control_norm = torch.nn.utils.clip_grad_norm_(control_model.parameters(), gradient_clip_norm)
    control_optimizer.step()
    return {
        "flow_total": float(flow_loss.detach()),
        "control_total": float(control_loss.detach()),
        "flow_gradient_norm": float(flow_norm.detach()),
        "control_gradient_norm": float(control_norm.detach()),
        "flow_components": {name: float(value.detach()) for name, value in flow_components.items()},
        "control_components": {
            name: float(value.detach()) for name, value in control_components.items()
        },
    }


def _save_checkpoint(path: Path, payload: dict) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _sha256(path)


def train_pilot(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    model_config: RectifiedFlowConfig = RectifiedFlowConfig(),
    pilot_config: PilotConfig = PilotConfig(),
    loss_weights: GeometryLossWeights = GeometryLossWeights(),
    max_rows_per_split: int | None = None,
) -> dict:
    import diffusers
    import torch

    model_config = model_config.validated()
    pilot_config = pilot_config.validated()
    loss_weights = loss_weights.validated()
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The bounded face rectified-flow pilot requires CUDA")
    requested = torch.device(device)
    if requested.index is not None:
        torch.cuda.set_device(requested.index)
    hardware = _hardware_provenance(device)
    if pilot_config.required_gpu_name_substring.lower() not in hardware["gpu_name"].lower():
        raise RuntimeError(
            f"GPU name guard failed: expected {pilot_config.required_gpu_name_substring!r}, "
            f"got {hardware['gpu_name']!r}"
        )
    corpus_root = Path(corpus_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    summary_sha = _sha256(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    corpus_contract = _validate_training_corpus_summary(summary)
    selected = _select_rows(summary["rows"], max_rows_per_split)
    split_rows = {
        split: [row for row in selected if row["split"] == split]
        for split in ("train", "validation", "sealed")
    }
    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    if max_rows_per_split is None and split_counts != {
        "train": 240,
        "validation": 40,
        "sealed": 40,
    }:
        raise ValueError("Full pilot requires the complete 240/40/40 corpus")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    code = _code_provenance()
    train_validation, preparation = _prepare_rows(
        corpus_root,
        split_rows["train"] + split_rows["validation"],
        device,
        model_config,
    )
    train_items = [item for item in train_validation if item.row["split"] == "train"]
    validation_items = [
        item for item in train_validation if item.row["split"] == "validation"
    ]
    flow_model = build_unet(model_config).to(device=device, dtype=torch.float32)
    control_model = build_unet(model_config).to(device=device, dtype=torch.float32)
    control_model.load_state_dict(copy.deepcopy(flow_model.state_dict()))
    parameter_total = parameter_count(flow_model)
    if parameter_total != parameter_count(control_model):
        raise AssertionError("Flow and direct control parameter counts differ")
    flow_optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=pilot_config.learning_rate,
        weight_decay=pilot_config.weight_decay,
    )
    control_optimizer = torch.optim.AdamW(
        control_model.parameters(),
        lr=pilot_config.learning_rate,
        weight_decay=pilot_config.weight_decay,
    )
    tensor_generator = torch.Generator(device=device).manual_seed(TRAINING_SEED)
    order_generator = random.Random(TRAINING_SEED)
    order = list(range(len(train_items)))
    cursor = len(order)
    evaluations = []
    training_tail = []
    best_flow_state = copy.deepcopy(flow_model.state_dict())
    best_control_state = copy.deepcopy(control_model.state_dict())
    best_flow_rank = None
    best_control_rank = None
    best_flow_step = 0
    best_control_step = 0
    best_calibration_scale = None
    completed_step = 0
    stop_reason = "step-budget-complete"

    for step in range(1, pilot_config.steps + 1):
        if cursor + pilot_config.batch_size > len(order):
            order_generator.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + pilot_config.batch_size]
        cursor += len(indices)
        record = _train_step(
            flow_model,
            control_model,
            flow_optimizer,
            control_optimizer,
            _batch([train_items[index] for index in indices], device),
            model_config,
            loss_weights,
            tensor_generator,
            pilot_config.gradient_clip_norm,
        )
        completed_step = step
        training_tail.append({"step": step, **record})
        training_tail = training_tail[-20:]
        elapsed = time.perf_counter() - started
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if peak > pilot_config.max_peak_vram_gb:
            stop_reason = "peak-vram-cap-exceeded"
            break
        if elapsed > pilot_config.max_runtime_seconds:
            stop_reason = "runtime-cap-exceeded"
            break
        if step % pilot_config.evaluation_interval == 0 or step == pilot_config.steps:
            if (
                time.perf_counter() - started + pilot_config.evaluation_reserve_seconds
                > pilot_config.max_runtime_seconds
            ):
                stop_reason = "runtime-cap-insufficient-validation-reserve"
                break
            flow_validation = _evaluate_flow(
                corpus_root,
                validation_items,
                flow_model,
                device,
                model_config,
                output_dir,
                f"validation_flow_step_{step:05d}",
            )
            control_validation = _evaluate_control(
                corpus_root,
                validation_items,
                control_model,
                device,
                output_dir,
                f"validation_control_step_{step:05d}",
            )
            flow_rank = _quality_rank(flow_validation)
            control_rank = _quality_rank(control_validation)
            if best_flow_rank is None or flow_rank < best_flow_rank:
                best_flow_rank = flow_rank
                best_flow_step = step
                best_flow_state = copy.deepcopy(flow_model.state_dict())
                best_calibration_scale = flow_validation["uncertainty"][
                    "calibration_scale_90"
                ]
            if best_control_rank is None or control_rank < best_control_rank:
                best_control_rank = control_rank
                best_control_step = step
                best_control_state = copy.deepcopy(control_model.state_dict())
            evaluations.append(
                {"step": step, "flow": flow_validation, "control": control_validation}
            )
            print(
                f"evaluation step={step} flow_failures="
                f"{flow_validation['combined_part_failures']} control_failures="
                f"{control_validation['combined_part_failures']}",
                flush=True,
            )
            peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if peak > pilot_config.max_peak_vram_gb:
                stop_reason = "peak-vram-cap-exceeded-after-validation"
                break
            if time.perf_counter() - started > pilot_config.max_runtime_seconds:
                stop_reason = "runtime-cap-exceeded-after-validation"
                break

    training_complete = bool(
        stop_reason == "step-budget-complete"
        and completed_step == pilot_config.steps
        and best_flow_step > 0
        and best_control_step > 0
        and best_calibration_scale is not None
    )
    sealed_preparation = None
    sealed_flow = None
    sealed_control = None
    if training_complete:
        latest = evaluations[-1]
        seconds_per_prepared_row = preparation["runtime_seconds"] / max(
            preparation["row_count"],
            1,
        )
        reserve = 2.0 * seconds_per_prepared_row * len(split_rows["sealed"])
        reserve += max(
            pilot_config.evaluation_reserve_seconds,
            1.5
            * (latest["flow"]["runtime_seconds"] + latest["control"]["runtime_seconds"]),
        )
        if time.perf_counter() - started + reserve > pilot_config.max_runtime_seconds:
            training_complete = False
            stop_reason = "runtime-cap-insufficient-sealed-reserve"
    if training_complete:
        sealed_items, sealed_preparation = _prepare_rows(
            corpus_root,
            split_rows["sealed"],
            device,
            model_config,
        )
        if time.perf_counter() - started > pilot_config.max_runtime_seconds:
            training_complete = False
            stop_reason = "runtime-cap-exceeded-during-sealed-preparation"
        if float(torch.cuda.max_memory_allocated(device) / (1024**3)) > pilot_config.max_peak_vram_gb:
            training_complete = False
            stop_reason = "peak-vram-cap-exceeded-during-sealed-preparation"
    if training_complete:
        flow_model.load_state_dict(best_flow_state)
        control_model.load_state_dict(best_control_state)
        sealed_flow = _evaluate_flow(
            corpus_root,
            sealed_items,
            flow_model,
            device,
            model_config,
            output_dir,
            "sealed_flow",
            calibration_scale_90=float(best_calibration_scale),
        )
        sealed_control = _evaluate_control(
            corpus_root,
            sealed_items,
            control_model,
            device,
            output_dir,
            "sealed_control",
        )

    runtime_seconds = time.perf_counter() - started
    peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    runtime_passed = runtime_seconds <= pilot_config.max_runtime_seconds
    vram_passed = peak_vram_gb <= pilot_config.max_peak_vram_gb
    run_checks = {
        "step_budget_complete": training_complete,
        "runtime_cap_passed": runtime_passed,
        "vram_cap_passed": vram_passed,
        "validation_checkpoint_available": best_flow_step > 0 and best_control_step > 0,
        "sealed_evaluation_complete": sealed_flow is not None and sealed_control is not None,
        "sealed_labels_loaded_after_training": sealed_preparation is not None,
    }
    gate = (
        _flow_advance_gate(sealed_flow, sealed_control, run_checks)
        if sealed_flow is not None and sealed_control is not None
        else {
            "decision": "hold",
            "checks": run_checks,
            "paired_small_turned": None,
            "small_turned_part_failure_reduction": None,
            "sealed_not_run_reason": stop_reason,
        }
    )

    common_checkpoint = {
        "model_config": model_config.provenance(),
        "pilot_config": asdict(pilot_config),
        "loss_weights": asdict(loss_weights),
        "input_contract": "source-support-normalized-camera-z-residual-v3",
        "corpus_summary_sha256": summary_sha,
        "training_seed": TRAINING_SEED,
        "code_provenance": code,
        "hardware_provenance": hardware,
        "run_status": {
            "stop_reason": stop_reason,
            "decision": gate["decision"],
            "runtime_cap_passed": runtime_passed,
            "vram_cap_passed": vram_passed,
            "advance_checks": gate["checks"],
        },
    }
    checkpoint_dir = output_dir / "checkpoints"
    flow_path = checkpoint_dir / "rectified_flow_best.pt"
    control_path = checkpoint_dir / "deterministic_control_best.pt"
    flow_sha = _save_checkpoint(
        flow_path,
        {
            **common_checkpoint,
            "method": "rectified-flow",
            "best_step": best_flow_step,
            "validation_calibration_scale_90": best_calibration_scale,
            "state_dict": best_flow_state,
        },
    )
    control_sha = _save_checkpoint(
        control_path,
        {
            **common_checkpoint,
            "method": "deterministic-control",
            "best_step": best_control_step,
            "state_dict": best_control_state,
        },
    )
    result = {
        "schema_version": 2,
        "run_kind": "face-depth-rectified-flow-vs-endpoint-matched-control",
        "promotion_eligible": False,
        "production_changed": False,
        "decision": gate["decision"],
        "stop_reason": stop_reason,
        "training_seed": TRAINING_SEED,
        "corpus": {
            "summary_sha256": summary_sha,
            "contract": corpus_contract,
            "split_counts": split_counts,
            "smoke_row_limit_per_split": max_rows_per_split,
            "production_training_eligible": summary.get("production_training_eligible"),
        },
        "model": {
            "config": model_config.provenance(),
            "parameter_count": parameter_total,
            "equal_parameter_control": True,
            "endpoint_objective_matched": True,
            "random_initialization": True,
            "latent_vae": False,
            "diffusers_version": diffusers.__version__,
        },
        "pilot": asdict(pilot_config),
        "loss_weights": asdict(loss_weights),
        "preparation": {
            "train_validation": preparation,
            "sealed": sealed_preparation,
        },
        "best_steps": {"flow": best_flow_step, "control": best_control_step},
        "validation_evaluations": evaluations,
        "sealed": {
            "opened_after_training": sealed_preparation is not None,
            "flow": sealed_flow,
            "control": sealed_control,
        },
        "advance_gate": gate,
        "runtime_seconds": runtime_seconds,
        "peak_vram_gb": peak_vram_gb,
        "runtime_cap_passed": runtime_passed,
        "vram_cap_passed": vram_passed,
        "hardware_provenance": hardware,
        "code_provenance": code,
        "checkpoints": {
            "flow": {"path": str(flow_path.relative_to(output_dir)), "sha256": flow_sha},
            "control": {
                "path": str(control_path.relative_to(output_dir)),
                "sha256": control_sha,
            },
        },
        "training_tail": training_tail,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["results_sha256"] = _sha256(result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_root")
    parser.add_argument("output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=PilotConfig.steps)
    parser.add_argument("--batch-size", type=int, default=PilotConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=PilotConfig.learning_rate)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=PilotConfig.evaluation_interval,
    )
    parser.add_argument("--sample-size", type=int, default=RectifiedFlowConfig.sample_size)
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=RectifiedFlowConfig.inference_steps,
    )
    parser.add_argument(
        "--inference-seeds",
        type=int,
        nargs="+",
        default=RectifiedFlowConfig.inference_seeds,
    )
    parser.add_argument("--max-rows-per-split", type=int)
    args = parser.parse_args()
    result = train_pilot(
        args.corpus_root,
        args.output_dir,
        device=args.device,
        model_config=RectifiedFlowConfig(
            sample_size=args.sample_size,
            inference_steps=args.inference_steps,
            inference_seeds=tuple(args.inference_seeds),
        ),
        pilot_config=PilotConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            evaluation_interval=args.evaluation_interval,
        ),
        max_rows_per_split=args.max_rows_per_split,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "runtime_seconds": result["runtime_seconds"],
                "peak_vram_gb": result["peak_vram_gb"],
                "results_sha256": result["results_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
