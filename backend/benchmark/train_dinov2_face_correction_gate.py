"""Train a selective safety gate for the frozen DINOv2 face correction."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.benchmark import train_dinov2_face_spatial_decoder as base
from backend.benchmark.train_face_surface_adapter import (
    TRAINING_SEED,
    _per_row_non_regression,
    _sha256,
    _strictly_improves,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    CachedFusionSurface,
    DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
    DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
    evaluate_exact_surfaces,
    load_cached_surfaces,
)


METHOD = "dinov2_pyramid448_selective_face_correction_gate"
BASE_ENCODER_PROFILE = "pyramid-448"
CHECKPOINT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_EPOCHS = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_OVERFIT_ROWS = 2
DEFAULT_OVERFIT_MAXIMUM_RATIO = 0.97
DEFAULT_ORACLE_WINDOW = 5
DEFAULT_ORACLE_MARGIN_RATIO = 0.02
DEFAULT_ORACLE_LOSS_WEIGHT = 0.50
DEFAULT_PIXEL_RISK_WEIGHT = 4.0
DEFAULT_TOTAL_VARIATION_WEIGHT = 0.05
GATE_THRESHOLDS = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
PRODUCTION_INPUTS = (
    "rgb",
    "local_depth",
    "incumbent_baseline",
    "face_support",
    "fusion_weight",
    "xy_coordinates",
    "frozen_dinov2_pyramid",
    "frozen_candidate_residual",
)
RESEARCH_SOURCES = {
    "selective_net": "https://proceedings.mlr.press/v97/geifman19a.html",
    "adaptive_depth_confidence": (
        "https://openaccess.thecvf.com/content/ICCV2021/html/"
        "Choi_Adaptive_Confidence_Thresholding_for_Monocular_"
        "Depth_Estimation_ICCV_2021_paper.html"
    ),
}
EXECUTION_CRITICAL_PATHS = (
    "backend/benchmark/train_dinov2_face_correction_gate.py",
    "backend/benchmark/train_dinov2_face_spatial_decoder.py",
    "backend/benchmark/train_face_surface_adapter.py",
    "backend/benchmark/train_face_surface_fusion_adapter.py",
    "backend/benchmark/train_gnm_production_face_depth_fusion.py",
)


def _code_provenance() -> dict:
    provenance = base._code_provenance()
    repo = Path(__file__).resolve().parents[2]
    for relative in EXECUTION_CRITICAL_PATHS:
        path = repo / relative
        provenance["files"][relative] = {
            "sha256": base._file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    safe_directory = f"safe.directory={repo.as_posix()}"
    try:
        status = subprocess.run(
            (
                "git",
                "-c",
                safe_directory,
                "status",
                "--porcelain",
                "--",
                *EXECUTION_CRITICAL_PATHS,
            ),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        status = ["git-status-unavailable"]
    provenance["owned_git_status"] = status
    return provenance


def production_conditioning(tensors: dict):
    """Build inference conditioning from an explicit production-only view."""

    required = ("inputs", "baseline", "support_face", "fusion_weight")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(
            "Missing production conditioning tensors: " + ", ".join(missing)
        )
    return base.conditioning_from_tensors(
        {name: tensors[name] for name in required}
    )


def _finite_gradients(values):
    import torch.nn.functional as functional

    dx = values[..., :, 1:] - values[..., :, :-1]
    dy = values[..., 1:, :] - values[..., :-1, :]
    return (
        functional.pad(dx, (0, 1, 0, 0)),
        functional.pad(dy, (0, 0, 0, 1)),
    )


def oracle_correction_gate(
    residual,
    tensors: dict,
    *,
    window: int = DEFAULT_ORACLE_WINDOW,
    margin_ratio: float = DEFAULT_ORACLE_MARGIN_RATIO,
):
    """Return a training-only soft target for locally beneficial correction."""

    import torch
    import torch.nn.functional as functional

    if int(window) <= 0 or int(window) % 2 != 1:
        raise ValueError("Oracle window must be a positive odd integer")
    if float(margin_ratio) < 0.0:
        raise ValueError("Oracle margin ratio must be nonnegative")
    baseline = tensors["baseline"]
    target = tensors["target"]
    support = torch.clamp(tensors["support_face"], 0.0, 1.0)
    if residual.shape != baseline.shape or target.shape != baseline.shape:
        raise ValueError("Oracle residual, baseline, and target shapes disagree")
    direction = target - baseline
    optimal = torch.clamp(
        direction * residual / torch.clamp(residual.square(), min=1e-8),
        0.0,
        1.0,
    )
    candidate = baseline + optimal * residual
    baseline_error = (baseline - target).abs()
    candidate_error = (candidate - target).abs()
    base_dx, base_dy = _finite_gradients(baseline - target)
    cand_dx, cand_dy = _finite_gradients(candidate - target)
    gradient_improvement = 0.5 * (
        base_dx.abs()
        + base_dy.abs()
        - cand_dx.abs()
        - cand_dy.abs()
    )
    improvement = baseline_error - candidate_error + 0.25 * gradient_improvement
    padding = int(window) // 2
    local_improvement = functional.avg_pool2d(
        improvement,
        kernel_size=int(window),
        stride=1,
        padding=padding,
    )
    scale = torch.clamp(tensors["correction_limit"].abs(), min=1e-6)
    margin = float(margin_ratio) * scale
    confidence = torch.sigmoid(
        (local_improvement - margin) / torch.clamp(0.25 * scale, min=1e-6)
    )
    raw = optimal * confidence * (local_improvement > margin).to(optimal.dtype)
    smoothed = functional.avg_pool2d(
        raw,
        kernel_size=int(window),
        stride=1,
        padding=padding,
    )
    return torch.clamp(smoothed, 0.0, 1.0) * support


def selective_gate(probability, support, *, threshold: float, window: int = 5):
    """Convert confidence to an exact-abstention, spatially smooth gate."""

    import torch
    import torch.nn.functional as functional

    value = float(threshold)
    if not 0.0 <= value < 1.0:
        raise ValueError("Gate threshold must be in [0, 1)")
    if int(window) <= 0 or int(window) % 2 != 1:
        raise ValueError("Gate smoothing window must be a positive odd integer")
    probability = torch.clamp(probability, 0.0, 1.0)
    support = torch.clamp(support, 0.0, 1.0)
    smoothed = functional.avg_pool2d(
        probability,
        kernel_size=int(window),
        stride=1,
        padding=int(window) // 2,
    )
    accepted = torch.clamp((smoothed - value) / max(1.0 - value, 1e-6), 0.0, 1.0)
    return accepted * support


def build_correction_gate():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    profile = base.resolve_encoder_profile(BASE_ENCODER_PROFILE)
    feature_channels = int(profile["feature_channels"])

    class CorrectionSafetyGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = nn.Sequential(
                nn.Conv2d(feature_channels, 32, 1),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                base._block(32, 32),
            )
            self.context = base._block(base.CONDITIONING_CHANNELS + 4, 32)
            self.fusion = base._block(64, 32)
            self.output = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.constant_(self.output.bias, -4.0)

        def forward(self, features, conditioning, residual, support):
            if features.ndim != 4 or features.shape[1:] != (
                feature_channels,
                int(profile["feature_size"]),
                int(profile["feature_size"]),
            ):
                raise ValueError("Correction gate received the wrong feature shape")
            if residual.ndim != 4 or residual.shape[1] != 1:
                raise ValueError("Correction gate residual must be [B, 1, H, W]")
            if (
                conditioning.shape
                != (
                    residual.shape[0],
                    base.CONDITIONING_CHANNELS,
                    residual.shape[-2],
                    residual.shape[-1],
                )
                or support.shape != residual.shape
            ):
                raise ValueError("Correction gate inputs disagree")
            dx, dy = _finite_gradients(residual)
            context = torch.cat(
                (conditioning, residual, residual.abs(), dx, dy),
                dim=1,
            )
            encoded = self.feature_projection(features.float())
            encoded = functional.interpolate(
                encoded,
                size=residual.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            hidden = self.fusion(
                torch.cat((encoded, self.context(context.float())), dim=1)
            )
            logits = self.output(hidden)
            probability = torch.sigmoid(logits)
            return logits, probability * torch.clamp(support, 0.0, 1.0)

    return CorrectionSafetyGate()


def validate_base_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    expected_corpus_summary_sha256: str,
    expected_cache_manifest_sha256: str,
    expected_cache_row_sha256: str,
    device: str,
):
    path = Path(checkpoint_path)
    observed = base._file_sha256(path)
    if observed != str(expected_sha256).lower():
        raise ValueError(
            "Base DINOv2 checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    model, payload, profile = base.load_decoder_checkpoint(path, device=device)
    checks = {
        "profile": profile["name"] == BASE_ENCODER_PROFILE,
        "corpus_summary": (
            payload.get("corpus_summary_sha256")
            == str(expected_corpus_summary_sha256).lower()
        ),
        "cache_manifest": (
            payload.get("cache_manifest_sha256")
            == str(expected_cache_manifest_sha256).lower()
        ),
        "ordered_cache_rows": (
            payload.get("ordered_cache_row_content_sha256")
            == str(expected_cache_row_sha256).lower()
        ),
        "model_files": payload.get("model_file_sha256") == base.MODEL_FILE_HASHES,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError("Base DINOv2 checkpoint binding failed: " + failed)
    return model, payload, profile, {
        "path": str(path),
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "checks": checks,
    }


def _residuals_by_id(items, values) -> dict[str, np.ndarray]:
    if len(items) != len(values):
        raise ValueError("Residual count does not match item count")
    result = {}
    for item, value in zip(items, values, strict=True):
        row_id = str(item.row["row_id"])
        array = np.asarray(value, dtype=np.float32)
        if array.shape != item.baseline.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid frozen residual for {row_id}")
        result[row_id] = array
    return result


def _unique_items(items):
    result = []
    seen = set()
    for item in items:
        row_id = str(item.row["row_id"])
        if not row_id or row_id in seen:
            continue
        result.append(item)
        seen.add(row_id)
    return result


def _residual_tensor(items, residuals_by_id: dict):
    import torch

    values = []
    for item in items:
        row_id = str(item.row["row_id"])
        if row_id not in residuals_by_id:
            raise ValueError(f"Missing frozen residual for {row_id}")
        values.append(residuals_by_id[row_id])
    return torch.from_numpy(np.stack(values))[:, None]


def _gate_loss(
    model,
    features,
    conditioning,
    residual,
    tensors: dict,
    *,
    oracle_window: int,
    oracle_margin_ratio: float,
    oracle_loss_weight: float,
    pixel_risk_weight: float,
    total_variation_weight: float,
):
    import torch
    import torch.nn.functional as functional

    logits, probability = model(
        features,
        conditioning,
        residual,
        tensors["support_face"],
    )
    oracle = oracle_correction_gate(
        residual,
        tensors,
        window=oracle_window,
        margin_ratio=oracle_margin_ratio,
    )
    support = torch.clamp(tensors["support_face"], 0.0, 1.0)
    sample_weight = tensors["sample_weight"]
    denominator = torch.clamp((support * sample_weight).sum(), min=1.0)
    oracle_loss = (
        functional.binary_cross_entropy_with_logits(
            logits,
            oracle,
            reduction="none",
        )
        * support
        * sample_weight
    ).sum() / denominator
    gated_residual = probability * residual
    geometry_loss, geometry_details = base._loss(gated_residual, tensors)
    baseline_error = (tensors["baseline"] - tensors["target"]).abs()
    gated_error = (
        tensors["baseline"] + gated_residual - tensors["target"]
    ).abs()
    pixel_risk = (
        functional.relu(gated_error - baseline_error)
        * support
        * sample_weight
    ).sum() / denominator
    dx, dy = _finite_gradients(probability)
    total_variation = (
        (dx.abs() + dy.abs()) * support * sample_weight
    ).sum() / denominator
    total = (
        geometry_loss
        + float(oracle_loss_weight) * oracle_loss
        + float(pixel_risk_weight) * pixel_risk
        + float(total_variation_weight) * total_variation
    )
    with torch.no_grad():
        accepted = probability >= 0.5
        beneficial = oracle >= 0.5
        true_positive = (accepted & beneficial & (support > 0.0)).sum()
        predicted_positive = (accepted & (support > 0.0)).sum()
        oracle_positive = (beneficial & (support > 0.0)).sum()
        coverage = (probability * support).sum() / torch.clamp(
            support.sum(), min=1.0
        )
    return total, {
        **geometry_details,
        "oracle_loss": float(oracle_loss.detach()),
        "pixel_non_regression_risk": float(pixel_risk.detach()),
        "gate_total_variation": float(total_variation.detach()),
        "soft_gate_coverage": float(coverage.detach()),
        "accepted_precision": float(
            true_positive / torch.clamp(predicted_positive, min=1)
        ),
        "accepted_recall": float(
            true_positive / torch.clamp(oracle_positive, min=1)
        ),
        "oracle_coverage": float(
            (oracle * support).sum() / torch.clamp(support.sum(), min=1.0)
        ),
    }


def _prepare_tensors(
    items: Sequence[CachedFusionSurface],
    *,
    small_face_weight_reference_px: float,
    maximum_small_face_sample_weight: float,
    yaw_sample_weight_strength: float,
    maximum_combined_sample_weight: float,
):
    return base._stack_with_curriculum(
        items,
        small_face_weight_reference_px=small_face_weight_reference_px,
        maximum_small_face_sample_weight=maximum_small_face_sample_weight,
        yaw_sample_weight_strength=yaw_sample_weight_strength,
        maximum_combined_sample_weight=maximum_combined_sample_weight,
    )


def _values_to_device(tensors: dict, indices: list[int], device: str) -> dict:
    return {name: value[indices].to(device) for name, value in tensors.items()}


def _validation_loss(
    model,
    items,
    features_by_id,
    residuals_by_id,
    *,
    device: str,
    batch_size: int,
    loss_options: dict,
    curriculum_options: dict,
):
    import torch

    if int(batch_size) <= 0:
        raise ValueError("Validation batch size must be positive")
    tensors, _stats = _prepare_tensors(items, **curriculum_options)
    residuals = _residual_tensor(items, residuals_by_id)
    totals = []
    details = []
    model.eval()
    with torch.inference_mode():
        # Per-row normalization makes selection invariant to batch composition.
        for start in range(len(items)):
            stop = start + 1
            indices = [start]
            values = _values_to_device(tensors, indices, device)
            features = base._features_for(
                items[start:stop],
                features_by_id,
                encoder_profile=BASE_ENCODER_PROFILE,
            ).to(device)
            conditioning = production_conditioning(values)
            total, current = _gate_loss(
                model,
                features,
                conditioning,
                residuals[start:stop].to(device),
                values,
                **loss_options,
            )
            totals.append(float(total))
            details.append(current)
    aggregate = {
        key: float(np.mean([record[key] for record in details]))
        for key in details[0]
    }
    return float(np.mean(totals)), aggregate


def train_gate(
    train_items,
    validation_items,
    features_by_id,
    residuals_by_id,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    loss_options: dict,
    curriculum_options: dict,
):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    train_tensors, train_weight_stats = _prepare_tensors(
        train_items, **curriculum_options
    )
    _validation_tensors, validation_weight_stats = _prepare_tensors(
        validation_items, **curriculum_options
    )
    train_conditioning = production_conditioning(train_tensors)
    train_residuals = _residual_tensor(train_items, residuals_by_id)
    model = build_correction_gate().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_loss, initial_details = _validation_loss(
        model,
        validation_items,
        features_by_id,
        residuals_by_id,
        device=device,
        batch_size=batch_size,
        loss_options=loss_options,
        curriculum_options=curriculum_options,
    )
    best_loss = initial_loss
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(len(train_items), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), max(1, int(batch_size))):
            indices = order[start : start + max(1, int(batch_size))]
            values = _values_to_device(train_tensors, indices, device)
            features = base._features_for(
                [train_items[index] for index in indices],
                features_by_id,
                encoder_profile=BASE_ENCODER_PROFILE,
            ).to(device)
            conditioning = train_conditioning[indices].to(device)
            total, _details = _gate_loss(
                model,
                features,
                conditioning,
                train_residuals[indices].to(device),
                values,
                **loss_options,
            )
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite correction-gate loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach()))
        validation_loss, validation_details = _validation_loss(
            model,
            validation_items,
            features_by_id,
            residuals_by_id,
            device=device,
            batch_size=batch_size,
            loss_options=loss_options,
            curriculum_options=curriculum_options,
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_details": validation_details,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    peak = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    return model.eval(), {
        "initial_validation_loss": initial_loss,
        "initial_validation_details": initial_details,
        "best_validation_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "history": history,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_vram_gib": peak,
        "host_memory": base._process_memory(),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "train_sample_weights": train_weight_stats,
        "validation_sample_weights": validation_weight_stats,
        "loss_options": loss_options,
    }


def predict_gated_residuals(
    model,
    items,
    features_by_id,
    residuals_by_id,
    *,
    device: str,
    batch_size: int,
    threshold: float,
):
    import torch

    tensors, _stats = _prepare_tensors(
        items,
        small_face_weight_reference_px=DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
        maximum_small_face_sample_weight=DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
        yaw_sample_weight_strength=0.0,
        maximum_combined_sample_weight=4.0,
    )
    conditioning = production_conditioning(tensors)
    residuals = _residual_tensor(items, residuals_by_id)
    output = []
    coverages = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(items))
            features = base._features_for(
                items[start:stop],
                features_by_id,
                encoder_profile=BASE_ENCODER_PROFILE,
            ).to(device)
            support = tensors["support_face"][start:stop].to(device)
            _logits, probability = model(
                features,
                conditioning[start:stop].to(device),
                residuals[start:stop].to(device),
                support,
            )
            gate = selective_gate(
                probability,
                support,
                threshold=threshold,
                window=5,
            )
            gated = gate * residuals[start:stop].to(device)
            output.extend(gated[:, 0].float().cpu().numpy())
            coverage = (gate * support).sum(dim=(1, 2, 3)) / torch.clamp(
                support.sum(dim=(1, 2, 3)), min=1.0
            )
            coverages.extend(coverage.float().cpu().tolist())
    return output, {
        "minimum": float(np.min(coverages)),
        "median": float(np.median(coverages)),
        "maximum": float(np.max(coverages)),
    }


def _select_gate_candidate(candidates: Sequence[dict], baseline: dict):
    resolved = []
    for candidate in candidates:
        audit = base._paired_regression_audit(candidate, baseline)
        tagged = audit["named_part_regression_count"] == 0
        whole_face = audit["whole_face_regression_count"] == 0
        eligible = bool(
            tagged
            and whole_face
            and _strictly_improves(candidate, baseline)
            and _per_row_non_regression(candidate, baseline) >= 1.0
        )
        resolved.append(
            {
                **candidate,
                "tagged_part_non_regression": tagged,
                "whole_face_non_regression": whole_face,
                "paired_regression_audit": audit,
                "eligible": eligible,
            }
        )
    eligible = [record for record in resolved if record["eligible"]]
    if not eligible:
        return {
            **baseline,
            "gate_threshold": None,
            "gate_coverage": {
                "minimum": 0.0,
                "median": 0.0,
                "maximum": 0.0,
            },
            "tagged_part_non_regression": True,
            "whole_face_non_regression": True,
            "paired_regression_audit": base._paired_regression_audit(
                baseline, baseline
            ),
            "eligible": False,
        }, resolved
    selected = min(
        eligible,
        key=lambda record: (
            int(record["combined_part_failures"]),
            -float(record["median_shape_correlation"]),
            -float(record["median_gradient_correlation"]),
            float(record["median_normalized_rmse"]),
            -float(record["gate_coverage"]["median"]),
        ),
    )
    return selected, resolved


def _capacity_gate(training: dict, *, maximum_ratio: float) -> dict:
    initial = float(training["initial_validation_loss"])
    best = float(training["best_validation_loss"])
    ratio = best / max(initial, 1e-12)
    checks = {
        "finite": math.isfinite(initial) and math.isfinite(best),
        "positive_initial": initial > 0.0,
        "selected_trained_epoch": int(training["best_epoch"]) > 0,
        "loss_ratio": ratio <= float(maximum_ratio),
    }
    return {
        "checks": checks,
        "initial_loss": initial,
        "best_loss": best,
        "best_to_initial_ratio": ratio,
        "maximum_ratio": float(maximum_ratio),
        "passed": all(checks.values()),
    }


def _checkpoint_payload(
    model,
    *,
    base_checkpoint: dict,
    cache_binding: dict,
    corpus_root: Path,
    cache_root: Path,
    code_provenance: dict,
    training: dict,
):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": METHOD,
        "base_encoder_profile": BASE_ENCODER_PROFILE,
        "base_checkpoint": base_checkpoint,
        "corpus_summary_sha256": _sha256(corpus_root / "summary.json"),
        "cache_manifest_sha256": _sha256(cache_root / "manifest.json"),
        "ordered_cache_row_content_sha256": cache_binding[
            "ordered_row_content_sha256"
        ],
        "production_inputs": list(PRODUCTION_INPUTS),
        "source_geometry_training_and_evaluation_only": True,
        "training": {
            key: training[key]
            for key in (
                "best_epoch",
                "epochs",
                "batch_size",
                "learning_rate",
                "seed",
                "loss_options",
            )
        },
        "code_provenance": code_provenance,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }


def run_experiment(args: argparse.Namespace) -> dict:
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Correction-gate training requested CUDA without a GPU")
    corpus_root = Path(args.corpus_root)
    cache_root = Path(args.cache_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_binding = base.validate_cache_binding(
        corpus_root,
        cache_root,
        expected_corpus_summary_sha256=args.expected_corpus_summary_sha256,
        expected_ordered_row_content_sha256=args.expected_cache_row_sha256,
    )
    base_model, _base_payload, _profile, base_checkpoint = (
        validate_base_checkpoint(
            args.base_checkpoint,
            expected_sha256=args.expected_base_checkpoint_sha256,
            expected_corpus_summary_sha256=args.expected_corpus_summary_sha256,
            expected_cache_manifest_sha256=cache_binding["manifest_sha256"],
            expected_cache_row_sha256=args.expected_cache_row_sha256,
            device=args.device,
        )
    )
    all_items, summary, cache_manifest = load_cached_surfaces(
        corpus_root, cache_root
    )
    by_split = {
        split: [item for item in all_items if item.row.get("split") == split]
        for split in ("train", "validation", "sealed")
    }
    if any(not rows for rows in by_split.values()):
        raise ValueError("Correction-gate corpus must retain all three splits")
    identities = {}
    for item in all_items:
        identities.setdefault(str(item.row["identity_group"]), set()).add(
            str(item.row["split"])
        )
    if any(len(splits) != 1 for splits in identities.values()):
        raise ValueError("Correction-gate identity groups cross split boundaries")
    if args.mode == "overfit":
        train_items = base._select_overfit_items(by_split["train"], args.rows)
        validation_items = train_items
    else:
        train_items = by_split["train"]
        validation_items = by_split["validation"]
    tuning_items = _unique_items([*train_items, *validation_items])
    features, encoder = base.extract_frozen_features(
        tuning_items,
        args.model_root,
        device=args.device,
        batch_size=args.batch_size,
        encoder_profile=BASE_ENCODER_PROFILE,
    )
    base_residual_values = base.predict_residuals(
        base_model,
        tuning_items,
        features,
        device=args.device,
        batch_size=args.batch_size,
        encoder_profile=BASE_ENCODER_PROFILE,
    )
    residuals_by_id = _residuals_by_id(tuning_items, base_residual_values)
    loss_options = {
        "oracle_window": int(args.oracle_window),
        "oracle_margin_ratio": float(args.oracle_margin_ratio),
        "oracle_loss_weight": float(args.oracle_loss_weight),
        "pixel_risk_weight": float(args.pixel_risk_weight),
        "total_variation_weight": float(args.total_variation_weight),
    }
    curriculum_options = {
        "small_face_weight_reference_px": float(
            args.small_face_weight_reference_px
        ),
        "maximum_small_face_sample_weight": float(
            args.maximum_small_face_sample_weight
        ),
        "yaw_sample_weight_strength": float(args.yaw_sample_weight_strength),
        "maximum_combined_sample_weight": float(
            args.maximum_combined_sample_weight
        ),
    }
    gate_model, training = train_gate(
        train_items,
        validation_items,
        features,
        residuals_by_id,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=TRAINING_SEED,
        loss_options=loss_options,
        curriculum_options=curriculum_options,
    )
    code_provenance = _code_provenance()
    checkpoint_path = output / "dinov2_face_correction_gate.pt"
    torch.save(
        _checkpoint_payload(
            gate_model,
            base_checkpoint=base_checkpoint,
            cache_binding=cache_binding,
            corpus_root=corpus_root,
            cache_root=cache_root,
            code_provenance=code_provenance,
            training=training,
        ),
        checkpoint_path,
    )
    common = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "production_changed": False,
        "method": METHOD,
        "research_sources": RESEARCH_SOURCES,
        "production_inputs": list(PRODUCTION_INPUTS),
        "source_geometry_training_and_evaluation_only": True,
        "code_provenance": code_provenance,
        "encoder": encoder,
        "base_checkpoint": base_checkpoint,
        "corpus": {
            "privacy": summary.get("privacy"),
            "summary_sha256": _sha256(corpus_root / "summary.json"),
            "cache_manifest_sha256": _sha256(cache_root / "manifest.json"),
            "cache_method": cache_manifest.get("method"),
            "cache_binding": cache_binding,
            "identity_disjoint": True,
            "split_counts": {
                split: len(items) for split, items in by_split.items()
            },
        },
        "training": training,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": base._file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "device": (
            torch.cuda.get_device_name(args.device)
            if str(args.device).startswith("cuda")
            else "cpu"
        ),
    }
    if args.mode == "overfit":
        gate = _capacity_gate(
            training, maximum_ratio=args.maximum_loss_ratio
        )
        evidence = {
            **common,
            "status": "smoke-pass" if gate["passed"] else "smoke-hold",
            "advance_to_split_training": bool(gate["passed"]),
            "selected_rows": [
                {
                    "row_id": item.row["row_id"],
                    "identity_group": item.row["identity_group"],
                    "face_height_pixels": item.row["render"][
                        "face_bbox_height_pixels"
                    ],
                }
                for item in train_items
            ],
            "capacity_gate": gate,
        }
    else:
        zeros = [
            np.zeros_like(item.baseline, dtype=np.float32)
            for item in validation_items
        ]
        baseline = evaluate_exact_surfaces(
            corpus_root,
            validation_items,
            zeros,
            alpha=0.0,
            output_dir=output / "validation_baseline",
        )
        candidates = []
        for threshold in GATE_THRESHOLDS:
            residuals, coverage = predict_gated_residuals(
                gate_model,
                validation_items,
                features,
                residuals_by_id,
                device=args.device,
                batch_size=args.batch_size,
                threshold=threshold,
            )
            candidate = evaluate_exact_surfaces(
                corpus_root,
                validation_items,
                residuals,
                alpha=1.0,
                output_dir=output / f"validation_threshold_{threshold:g}",
            )
            candidates.append(
                {
                    **candidate,
                    "gate_threshold": float(threshold),
                    "gate_coverage": coverage,
                }
            )
        selected, audited = _select_gate_candidate(candidates, baseline)
        decision = {
            "validation_strictly_improves": bool(selected.get("eligible")),
            "sealed_opened": False,
            "sealed_strictly_improves": False,
            "advance_to_exact_photo": False,
            "advance_to_30mm_physical_replay": False,
        }
        evidence = {
            **common,
            "status": (
                "validation-pass-sealed-pending"
                if decision["validation_strictly_improves"]
                else "hold"
            ),
            "validation": {
                "baseline": baseline,
                "candidates": audited,
                "selected": selected,
            },
            "sealed": None,
            "decision": decision,
        }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "split"), default="overfit")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=DEFAULT_OVERFIT_ROWS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--oracle-window", type=int, default=DEFAULT_ORACLE_WINDOW)
    parser.add_argument(
        "--oracle-margin-ratio",
        type=float,
        default=DEFAULT_ORACLE_MARGIN_RATIO,
    )
    parser.add_argument(
        "--oracle-loss-weight",
        type=float,
        default=DEFAULT_ORACLE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--pixel-risk-weight",
        type=float,
        default=DEFAULT_PIXEL_RISK_WEIGHT,
    )
    parser.add_argument(
        "--total-variation-weight",
        type=float,
        default=DEFAULT_TOTAL_VARIATION_WEIGHT,
    )
    parser.add_argument(
        "--small-face-weight-reference-px",
        type=float,
        default=DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
    )
    parser.add_argument(
        "--maximum-small-face-sample-weight",
        type=float,
        default=DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
    )
    parser.add_argument("--yaw-sample-weight-strength", type=float, default=0.0)
    parser.add_argument(
        "--maximum-combined-sample-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--maximum-loss-ratio",
        type=float,
        default=DEFAULT_OVERFIT_MAXIMUM_RATIO,
    )
    parser.add_argument("--expected-corpus-summary-sha256", required=True)
    parser.add_argument("--expected-cache-row-sha256", required=True)
    parser.add_argument("--expected-base-checkpoint-sha256", required=True)
    args = parser.parse_args()
    evidence = run_experiment(args)
    if args.mode == "overfit":
        print(json.dumps(evidence["capacity_gate"], indent=2))
    else:
        print(json.dumps(evidence["decision"], indent=2))


if __name__ == "__main__":
    main()
