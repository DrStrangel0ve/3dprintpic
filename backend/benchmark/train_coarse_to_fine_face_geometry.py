"""Train a coarse-to-fine camera-aligned face-relief geometry decoder."""

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
    BLEND_ALPHAS,
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


METHOD = "dinov2_pyramid448_coarse_to_fine_normal_face_geometry"
ENCODER_PROFILE = "pyramid-448"
CHECKPOINT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
COARSE_SIZE = 40
COARSE_RESIDUAL_LIMIT = 0.25
FINE_RESIDUAL_LIMIT = 0.12
TOTAL_RESIDUAL_LIMIT = base.MAXIMUM_NORMALIZED_RESIDUAL
FINE_CONTEXT_CHANNELS = base.CONDITIONING_CHANNELS + 7
DEFAULT_EPOCHS = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_OVERFIT_ROWS = 2
DEFAULT_OVERFIT_MAXIMUM_RATIO = 0.97
DEFAULT_COARSE_LOSS_WEIGHT = 0.35
DEFAULT_FINE_LOW_FREQUENCY_WEIGHT = 0.10
PRODUCTION_INPUTS = (
    "rgb",
    "local_depth",
    "incumbent_baseline",
    "face_support",
    "fusion_weight",
    "xy_coordinates",
    "frozen_dinov2_pyramid",
)
TRAINING_ONLY_INPUTS = (
    "exact_camera_aligned_face_depth",
    "exact_six_part_face_masks",
)
RESEARCH_SOURCES = {
    "coarse_to_fine_normal_refinement": {
        "paper": "https://arxiv.org/abs/2601.01950",
        "repository": "https://github.com/AutoHDR/FNR2R",
        "repository_revision": "8d313a6bf9c082ff1d4c5c5c2e7ca01982686992",
        "license_status": "no repository license found; no code or weights imported",
        "use": "high-level coarse-to-fine design idea only",
    },
    "dinov2": {
        "repository": base.MODEL_SOURCE,
        "revision": base.MODEL_REVISION,
        "license": base.MODEL_LICENSE,
    },
}
EXECUTION_CRITICAL_PATHS = (
    "backend/benchmark/train_coarse_to_fine_face_geometry.py",
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


def surface_normal_exemplars(depth):
    """Return finite camera-space normal exemplars from one depth channel."""

    import torch
    import torch.nn.functional as functional

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("Surface-normal depth must have shape [B, 1, H, W]")
    if not torch.isfinite(depth).all():
        raise ValueError("Surface-normal depth must be finite")
    dx = depth[..., :, 1:] - depth[..., :, :-1]
    dy = depth[..., 1:, :] - depth[..., :-1, :]
    dx = functional.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = functional.pad(dy, (0, 0, 0, 1), mode="replicate")
    z = torch.full_like(dx, 2.0 / max(int(depth.shape[-1]) - 1, 1))
    normals = torch.cat((-dx, -dy, z), dim=1)
    return functional.normalize(normals, dim=1, eps=1e-6)


def build_geometry_decoder():
    """Build a two-stage decoder for broad form and local face detail."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    profile = base.resolve_encoder_profile(ENCODER_PROFILE)
    feature_channels = int(profile["feature_channels"])
    feature_size = int(profile["feature_size"])

    class CoarseToFineFaceGeometry(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = nn.Sequential(
                nn.Conv2d(feature_channels, 96, 1),
                nn.GroupNorm(8, 96),
                nn.SiLU(),
            )
            self.coarse_features = base._block(96, 64)
            self.coarse_conditioning = base._block(
                base.CONDITIONING_CHANNELS, 32
            )
            self.coarse_fusion = base._block(96, 64)
            self.coarse_output = nn.Conv2d(64, 1, 1)
            self.fine_features_80 = base._block(64, 64)
            self.fine_features_160 = base._block(64, 48)
            self.fine_conditioning = base._block(FINE_CONTEXT_CHANNELS, 48)
            self.fine_fusion = base._block(96, 48)
            self.fine_output = nn.Conv2d(48, 1, 1)
            nn.init.zeros_(self.coarse_output.weight)
            nn.init.zeros_(self.coarse_output.bias)
            nn.init.zeros_(self.fine_output.weight)
            nn.init.zeros_(self.fine_output.bias)

        def forward(self, features, conditioning, support):
            expected_feature_shape = (
                feature_channels,
                feature_size,
                feature_size,
            )
            if features.ndim != 4 or features.shape[1:] != expected_feature_shape:
                raise ValueError(
                    "Coarse-to-fine decoder received the wrong feature shape"
                )
            expected_conditioning_shape = (
                features.shape[0],
                base.CONDITIONING_CHANNELS,
                conditioning.shape[-2],
                conditioning.shape[-1],
            )
            if (
                conditioning.ndim != 4
                or conditioning.shape != expected_conditioning_shape
                or support.shape
                != (features.shape[0], 1, *conditioning.shape[-2:])
            ):
                raise ValueError("Coarse-to-fine decoder inputs disagree")
            if conditioning.shape[-2:] != (160, 160):
                raise ValueError("Coarse-to-fine decoder requires 160x160 crops")
            if not all(
                torch.isfinite(value).all()
                for value in (features, conditioning, support)
            ):
                raise ValueError("Coarse-to-fine decoder inputs must be finite")

            support = torch.clamp(support.float(), 0.0, 1.0)
            projected = self.feature_projection(features.float())
            coarse_hidden = functional.interpolate(
                projected,
                size=(COARSE_SIZE, COARSE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            coarse_hidden = self.coarse_features(coarse_hidden)
            coarse_conditioning = functional.interpolate(
                conditioning.float(),
                size=(COARSE_SIZE, COARSE_SIZE),
                mode="area",
            )
            coarse_support = functional.interpolate(
                support,
                size=(COARSE_SIZE, COARSE_SIZE),
                mode="area",
            )
            coarse_hidden = self.coarse_fusion(
                torch.cat(
                    (
                        coarse_hidden,
                        self.coarse_conditioning(coarse_conditioning),
                    ),
                    dim=1,
                )
            )
            coarse = float(COARSE_RESIDUAL_LIMIT) * torch.tanh(
                self.coarse_output(coarse_hidden)
            )
            coarse = coarse * coarse_support
            coarse_full = functional.interpolate(
                coarse,
                size=conditioning.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            coarse_full = coarse_full * support

            fine_hidden = functional.interpolate(
                coarse_hidden,
                size=(80, 80),
                mode="bilinear",
                align_corners=False,
            )
            fine_hidden = self.fine_features_80(fine_hidden)
            fine_hidden = functional.interpolate(
                fine_hidden,
                size=conditioning.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fine_hidden = self.fine_features_160(fine_hidden)
            baseline = conditioning[:, 4:5].float()
            fine_context = torch.cat(
                (
                    conditioning.float(),
                    surface_normal_exemplars(baseline),
                    coarse_full,
                    surface_normal_exemplars(baseline + coarse_full),
                ),
                dim=1,
            )
            conditioned = self.fine_conditioning(fine_context)
            fine = float(FINE_RESIDUAL_LIMIT) * torch.tanh(
                self.fine_output(
                    self.fine_fusion(
                        torch.cat((fine_hidden, conditioned), dim=1)
                    )
                )
            )
            fine = fine * support
            combined = torch.clamp(
                coarse_full + fine,
                -float(TOTAL_RESIDUAL_LIMIT),
                float(TOTAL_RESIDUAL_LIMIT),
            )
            return combined * support, coarse_full

    return CoarseToFineFaceGeometry()


def _weighted_mean(values, tensors: dict):
    import torch

    support = torch.clamp(tensors["support_face"], 0.0, 1.0)
    sample_weight = tensors["sample_weight"]
    weight = support * sample_weight
    return (values * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def geometry_loss(
    residual,
    coarse_residual,
    tensors: dict,
    *,
    coarse_loss_weight: float = DEFAULT_COARSE_LOSS_WEIGHT,
    fine_low_frequency_weight: float = DEFAULT_FINE_LOW_FREQUENCY_WEIGHT,
):
    """Supervise full geometry while separating broad form from fine detail."""

    import torch
    import torch.nn.functional as functional

    if residual.shape != coarse_residual.shape:
        raise ValueError("Coarse and final residual shapes disagree")
    total, details = base._loss(residual, tensors)
    target_residual = tensors["target"] - tensors["baseline"]
    low_target = functional.avg_pool2d(
        target_residual,
        kernel_size=9,
        stride=1,
        padding=4,
    )
    coarse_value = _weighted_mean((coarse_residual - low_target).abs(), tensors)
    fine = residual - coarse_residual
    fine_low = functional.avg_pool2d(
        fine,
        kernel_size=9,
        stride=1,
        padding=4,
    )
    fine_low_frequency = _weighted_mean(fine_low.abs(), tensors)
    total = (
        total
        + float(coarse_loss_weight) * coarse_value
        + float(fine_low_frequency_weight) * fine_low_frequency
    )
    return total, {
        **details,
        "coarse_low_frequency_value": float(coarse_value.detach()),
        "fine_low_frequency_leakage": float(fine_low_frequency.detach()),
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


def _validation_loss(
    model,
    items,
    features_by_id,
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
    totals = []
    detail_rows = []
    model.eval()
    with torch.inference_mode():
        # Per-row normalization keeps model selection independent of packing.
        for start in range(len(items)):
            stop = start + 1
            indices = [start]
            values = base._batch_values(tensors, indices, device)
            features = base._features_for(
                items[start:stop],
                features_by_id,
                encoder_profile=ENCODER_PROFILE,
            ).to(device)
            conditioning = production_conditioning(values)
            residual, coarse = model(
                features,
                conditioning,
                values["support_face"],
            )
            total, details = geometry_loss(
                residual,
                coarse,
                values,
                **loss_options,
            )
            totals.append(float(total))
            detail_rows.append(details)
    aggregate = {
        key: float(np.mean([details[key] for details in detail_rows]))
        for key in detail_rows[0]
    }
    return float(np.mean(totals)), aggregate


def train_geometry_decoder(
    train_items,
    validation_items,
    features_by_id,
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

    if not train_items or not validation_items:
        raise ValueError("Coarse-to-fine training requires nonempty splits")
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
    model = build_geometry_decoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_loss, initial_details = _validation_loss(
        model,
        validation_items,
        features_by_id,
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
            values = base._batch_values(train_tensors, indices, device)
            features = base._features_for(
                [train_items[index] for index in indices],
                features_by_id,
                encoder_profile=ENCODER_PROFILE,
            ).to(device)
            conditioning = train_conditioning[indices].to(device)
            residual, coarse = model(
                features,
                conditioning,
                values["support_face"],
            )
            total, _details = geometry_loss(
                residual,
                coarse,
                values,
                **loss_options,
            )
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite coarse-to-fine loss at epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(total.detach()))
        validation_loss, validation_details = _validation_loss(
            model,
            validation_items,
            features_by_id,
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
        "feature_batching": "streamed_from_cpu_cache",
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "loss_options": dict(loss_options),
        "curriculum": {
            **dict(curriculum_options),
            "train_sample_weights": train_weight_stats,
            "validation_sample_weights": validation_weight_stats,
        },
    }


def predict_residuals(
    model,
    items,
    features_by_id,
    *,
    device: str,
    batch_size: int,
):
    import torch

    tensors, _stats = _prepare_tensors(
        items,
        small_face_weight_reference_px=DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
        maximum_small_face_sample_weight=DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
        yaw_sample_weight_strength=base.DEFAULT_YAW_SAMPLE_WEIGHT_STRENGTH,
        maximum_combined_sample_weight=base.DEFAULT_MAXIMUM_COMBINED_SAMPLE_WEIGHT,
    )
    conditioning = production_conditioning(tensors)
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(items))
            features = base._features_for(
                items[start:stop],
                features_by_id,
                encoder_profile=ENCODER_PROFILE,
            ).to(device)
            residual, _coarse = model(
                features,
                conditioning[start:stop].to(device),
                tensors["support_face"][start:stop].to(device),
            )
            values.extend(
                residual[:, 0].detach().float().cpu().numpy().astype(np.float32)
            )
    return values


def capacity_gate(training: dict, *, maximum_ratio: float) -> dict:
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
    cache_binding: dict,
    corpus_root: Path,
    cache_root: Path,
    code_provenance: dict,
    training: dict,
    selected_alpha: float | None,
):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": METHOD,
        "model_id": base.MODEL_ID,
        "model_revision": base.MODEL_REVISION,
        "encoder_profile": base.resolve_encoder_profile(ENCODER_PROFILE),
        "selected_alpha": selected_alpha,
        "corpus_summary_sha256": _sha256(corpus_root / "summary.json"),
        "cache_manifest_sha256": _sha256(cache_root / "manifest.json"),
        "ordered_cache_row_content_sha256": cache_binding[
            "ordered_row_content_sha256"
        ],
        "model_file_sha256": dict(base.MODEL_FILE_HASHES),
        "production_inputs": list(PRODUCTION_INPUTS),
        "training_only_inputs": list(TRAINING_ONLY_INPUTS),
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


def load_geometry_checkpoint(path: str | Path, *, device: str = "cpu"):
    import torch

    try:
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load coarse-to-fine checkpoint: {error}") from error
    checks = {
        "schema": payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "method": payload.get("method") == METHOD,
        "model_id": payload.get("model_id") == base.MODEL_ID,
        "model_revision": payload.get("model_revision") == base.MODEL_REVISION,
        "encoder_profile": (
            payload.get("encoder_profile")
            == base.resolve_encoder_profile(ENCODER_PROFILE)
        ),
        "model_files": payload.get("model_file_sha256") == base.MODEL_FILE_HASHES,
        "production_inputs": payload.get("production_inputs")
        == list(PRODUCTION_INPUTS),
        "training_only_inputs": payload.get("training_only_inputs")
        == list(TRAINING_ONLY_INPUTS),
        "source_geometry_scope": (
            payload.get("source_geometry_training_and_evaluation_only") is True
        ),
        "state_dict": isinstance(payload.get("state_dict"), dict),
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError("Coarse-to-fine checkpoint validation failed: " + failed)
    model = build_geometry_decoder()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"Coarse-to-fine checkpoint state is incompatible: {error}"
        ) from error
    return model.to(device).eval(), payload


def _identity_disjoint_splits(items):
    by_split = {
        split: [item for item in items if item.row.get("split") == split]
        for split in ("train", "validation", "sealed")
    }
    if any(not rows for rows in by_split.values()):
        raise ValueError("Coarse-to-fine corpus must retain all three splits")
    identities = {}
    for item in items:
        identities.setdefault(str(item.row["identity_group"]), set()).add(
            str(item.row["split"])
        )
    if any(len(splits) != 1 for splits in identities.values()):
        raise ValueError("Coarse-to-fine identity groups cross split boundaries")
    return by_split


def _select_candidate(candidates: Sequence[dict], baseline: dict):
    selected, resolved = base._select_validation_candidate(candidates, baseline)
    return selected, resolved


def run_experiment(args: argparse.Namespace) -> dict:
    import torch

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Coarse-to-fine training requested CUDA without a GPU")
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
    all_items, summary, cache_manifest = load_cached_surfaces(
        corpus_root, cache_root
    )
    by_split = _identity_disjoint_splits(all_items)
    if args.mode == "overfit":
        train_items = base._select_overfit_items(by_split["train"], args.rows)
        validation_items = train_items
    else:
        train_items = by_split["train"]
        validation_items = by_split["validation"]
    tuning_items = [*train_items]
    seen = {str(item.row["row_id"]) for item in tuning_items}
    tuning_items.extend(
        item
        for item in validation_items
        if str(item.row["row_id"]) not in seen
    )
    features, encoder = base.extract_frozen_features(
        tuning_items,
        args.model_root,
        device=args.device,
        batch_size=args.batch_size,
        encoder_profile=ENCODER_PROFILE,
    )
    loss_options = {
        "coarse_loss_weight": float(args.coarse_loss_weight),
        "fine_low_frequency_weight": float(
            args.fine_low_frequency_weight
        ),
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
    model, training = train_geometry_decoder(
        train_items,
        validation_items,
        features,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=TRAINING_SEED,
        loss_options=loss_options,
        curriculum_options=curriculum_options,
    )
    code_provenance = _code_provenance()
    selected_alpha = None
    validation = None
    sealed = None
    sealed_encoder = None
    decision = None
    if args.mode == "split":
        zeros = [
            np.zeros_like(item.baseline, dtype=np.float32)
            for item in validation_items
        ]
        baseline_summary = evaluate_exact_surfaces(
            corpus_root,
            validation_items,
            zeros,
            alpha=0.0,
            output_dir=output / "validation_baseline",
        )
        residuals = predict_residuals(
            model,
            validation_items,
            features,
            device=args.device,
            batch_size=args.batch_size,
        )
        candidates = []
        for alpha in BLEND_ALPHAS:
            if float(alpha) == 0.0:
                continue
            candidate = evaluate_exact_surfaces(
                corpus_root,
                validation_items,
                residuals,
                alpha=float(alpha),
                output_dir=output / f"validation_alpha_{float(alpha):g}",
            )
            candidates.append({**candidate, "alpha": float(alpha)})
        selected, candidate_audits = _select_candidate(
            candidates, baseline_summary
        )
        selected_alpha = float(selected["alpha"])
        validation = {
            "baseline": baseline_summary,
            "candidates": candidate_audits,
            "selected": selected,
        }
        sealed_gate = False
        if bool(selected.get("eligible")):
            sealed_items = by_split["sealed"]
            sealed_features, sealed_encoder = base.extract_frozen_features(
                sealed_items,
                args.model_root,
                device=args.device,
                batch_size=args.batch_size,
                encoder_profile=ENCODER_PROFILE,
            )
            sealed_zeros = [
                np.zeros_like(item.baseline, dtype=np.float32)
                for item in sealed_items
            ]
            sealed_baseline = evaluate_exact_surfaces(
                corpus_root,
                sealed_items,
                sealed_zeros,
                alpha=0.0,
                output_dir=output / "sealed_baseline",
            )
            sealed_residuals = predict_residuals(
                model,
                sealed_items,
                sealed_features,
                device=args.device,
                batch_size=args.batch_size,
            )
            sealed_candidate = evaluate_exact_surfaces(
                corpus_root,
                sealed_items,
                sealed_residuals,
                alpha=selected_alpha,
                output_dir=output / "sealed_selected",
            )
            sealed_audit = base._paired_regression_audit(
                sealed_candidate, sealed_baseline
            )
            sealed_gate = bool(
                sealed_audit["regression_count"] == 0
                and _strictly_improves(sealed_candidate, sealed_baseline)
                and _per_row_non_regression(
                    sealed_candidate, sealed_baseline
                )
                >= 1.0
            )
            sealed = {
                "baseline": sealed_baseline,
                "selected": {
                    **sealed_candidate,
                    "alpha": selected_alpha,
                    "paired_regression_audit": sealed_audit,
                    "eligible": sealed_gate,
                },
            }
        decision = {
            "validation_strictly_improves": bool(selected.get("eligible")),
            "sealed_opened": sealed is not None,
            "sealed_strictly_improves": sealed_gate,
            "advance_to_exact_photo": bool(
                selected.get("eligible") and sealed_gate
            ),
        }
    checkpoint_path = output / "coarse_to_fine_face_geometry.pt"
    torch.save(
        _checkpoint_payload(
            model,
            cache_binding=cache_binding,
            corpus_root=corpus_root,
            cache_root=cache_root,
            code_provenance=code_provenance,
            training=training,
            selected_alpha=selected_alpha,
        ),
        checkpoint_path,
    )
    common = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "production_changed": False,
        "method": METHOD,
        "research_sources": RESEARCH_SOURCES,
        "production_inputs": list(PRODUCTION_INPUTS),
        "training_only_inputs": list(TRAINING_ONLY_INPUTS),
        "source_geometry_training_and_evaluation_only": True,
        "code_provenance": code_provenance,
        "encoder": encoder,
        "sealed_encoder": sealed_encoder,
        "architecture": {
            "stages": ["coarse_40", "fine_160"],
            "coarse_residual_limit": COARSE_RESIDUAL_LIMIT,
            "fine_residual_limit": FINE_RESIDUAL_LIMIT,
            "total_residual_limit": TOTAL_RESIDUAL_LIMIT,
            "fine_context_channels": FINE_CONTEXT_CHANNELS,
            "surface_normal_exemplars": [
                "incumbent_baseline",
                "incumbent_plus_predicted_coarse_geometry",
            ],
            "trainable_parameters": training["trainable_parameters"],
        },
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
        gate = capacity_gate(
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
        evidence = {
            **common,
            "status": "hold",
            "validation": validation,
            "sealed": sealed,
            "decision": decision,
        }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "split"), default="overfit")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=DEFAULT_OVERFIT_ROWS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--maximum-loss-ratio",
        type=float,
        default=DEFAULT_OVERFIT_MAXIMUM_RATIO,
    )
    parser.add_argument(
        "--coarse-loss-weight",
        type=float,
        default=DEFAULT_COARSE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--fine-low-frequency-weight",
        type=float,
        default=DEFAULT_FINE_LOW_FREQUENCY_WEIGHT,
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
    parser.add_argument(
        "--yaw-sample-weight-strength",
        type=float,
        default=base.DEFAULT_YAW_SAMPLE_WEIGHT_STRENGTH,
    )
    parser.add_argument(
        "--maximum-combined-sample-weight",
        type=float,
        default=base.DEFAULT_MAXIMUM_COMBINED_SAMPLE_WEIGHT,
    )
    parser.add_argument("--expected-corpus-summary-sha256", required=True)
    parser.add_argument("--expected-cache-row-sha256", required=True)
    args = parser.parse_args()
    evidence = run_experiment(args)
    print(
        json.dumps(
            evidence.get("capacity_gate", evidence.get("decision")),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
