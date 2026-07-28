"""Fine-tune GNM DAv2 face depth through the production residual fusion path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    GNM_FUSION_METHOD,
    GNM_PRODUCTION_FUSION_METHOD,
    GNM_TRAINING_FACE_HEIGHT_ANCHOR_PIXELS,
    _checkpoint_mode_and_alpha,
    _sha256,
)
from backend.benchmark.evaluate_gnm_fusion_residual_schedule import (
    _per_row_no_failure_regression,
    _quality,
    _summary,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _strictly_improves,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    _stack,
    apply_training_residual,
    load_cached_surfaces,
    positive_affine_fit,
    selected_image_from_exact_mask,
    surface_fusion_training_loss,
)
from backend.benchmark.train_gnm_face_depth_fusion import (
    _configure_training_scope,
    _load_state_dict,
    _prediction,
    _state_dict,
)
from backend.face_depth_refinement import fuse_face_surface_residual


TRAINING_SEED = 20260717
DEFAULT_EPOCHS = 6
DEFAULT_LEARNING_RATE = 5e-6
BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPOCH_SELECTION_ALPHA = 0.25
PART_BALANCED_VALUE_WEIGHT = 0.75
PART_BALANCED_GRADIENT_WEIGHT = 0.75
PART_BALANCED_NON_REGRESSION_WEIGHT = 1.0
WORST_PART_VALUE_NON_REGRESSION_WEIGHT = 2.0
WORST_PART_GRADIENT_NON_REGRESSION_WEIGHT = 2.0
SMALL_FACE_RESIDUAL_ZERO_STRENGTH_PIXELS = 110.0


@dataclass
class ProductionDirectFace:
    row: dict
    corpus_root: Path
    surface: object
    pixel_values: object
    patch_height: int
    patch_width: int
    crop_shape: tuple[int, int]


def _identity_groups(surfaces: list) -> set[str]:
    return {
        str(surface.row["identity_group"])
        for surface in surfaces
    }


def _validate_additional_training_contract(
    primary_surfaces: list,
    validation_surfaces: list,
    additional_surfaces: list,
    additional_summary: dict,
    additional_manifest: dict,
) -> dict:
    if not additional_surfaces:
        raise ValueError("Additional training corpus contains no rows")
    if any(
        surface.row.get("split") != "train"
        for surface in additional_surfaces
    ):
        raise ValueError(
            "Additional training corpus contains a non-training row"
        )

    row_ids = [str(surface.row["row_id"]) for surface in additional_surfaces]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Additional training corpus contains duplicate row ids")
    occupied_row_ids = {
        str(surface.row["row_id"])
        for surface in [*primary_surfaces, *validation_surfaces]
    }
    row_overlap = occupied_row_ids & set(row_ids)
    if row_overlap:
        raise ValueError(
            "Additional training corpus overlaps existing row ids: "
            + ", ".join(sorted(row_overlap))
        )

    additional_identities = _identity_groups(additional_surfaces)
    occupied_identities = _identity_groups(
        [*primary_surfaces, *validation_surfaces]
    )
    identity_overlap = occupied_identities & additional_identities
    if identity_overlap:
        raise ValueError(
            "Additional training corpus overlaps held or primary identities: "
            + ", ".join(sorted(identity_overlap))
        )

    if additional_summary.get("row_count") != len(additional_surfaces):
        raise ValueError(
            "Additional training corpus summary row count does not match"
        )
    if additional_manifest.get("row_count") != len(additional_surfaces):
        raise ValueError(
            "Additional training cache row count does not match"
        )
    detector_filter = additional_summary.get("production_detector_filter")
    if not isinstance(detector_filter, dict):
        raise ValueError(
            "Additional training corpus lacks production detector provenance"
        )
    if detector_filter.get("fallback_geometry_used") is not False:
        raise ValueError(
            "Additional training corpus used an evaluation-geometry fallback"
        )
    if detector_filter.get("passed_rows") != len(additional_surfaces):
        raise ValueError(
            "Additional training detector pass count does not match"
        )
    input_rows = detector_filter.get("input_rows")
    excluded_rows = detector_filter.get("excluded_rows")
    if (
        not isinstance(input_rows, int)
        or input_rows < len(additional_surfaces)
        or not isinstance(excluded_rows, list)
        or input_rows - len(additional_surfaces) != len(excluded_rows)
    ):
        raise ValueError(
            "Additional training detector exclusion provenance is inconsistent"
        )
    return {
        "row_count": len(additional_surfaces),
        "identity_count": len(additional_identities),
        "identity_groups": sorted(additional_identities),
        "production_detector_filter": detector_filter,
    }


def residual_application_scale(face_height_pixels: float) -> float:
    height = max(float(face_height_pixels), 1.0)
    full_strength = GNM_TRAINING_FACE_HEIGHT_ANCHOR_PIXELS
    zero_strength = SMALL_FACE_RESIDUAL_ZERO_STRENGTH_PIXELS
    if height <= full_strength:
        return 1.0
    if height >= zero_strength:
        return 0.0
    return float(
        (zero_strength - height)
        / (
            zero_strength
            - full_strength
        )
    )


def normalize_prediction_to_shape(
    prediction,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
):
    import torch

    values = torch.nn.functional.interpolate(
        prediction.float()[None, None],
        size=source_shape,
        mode="bicubic",
        align_corners=False,
    )
    low = values.amin(dim=(-2, -1), keepdim=True)
    high = values.amax(dim=(-2, -1), keepdim=True)
    values = (values - low) / (high - low).clamp_min(1e-6)
    if tuple(values.shape[-2:]) != tuple(target_shape):
        values = torch.nn.functional.interpolate(
            values,
            size=target_shape,
            mode="bicubic",
            align_corners=False,
        )
    return values[0, 0]


def worst_part_physical_non_regression(residual, tensors: dict):
    import torch
    import torch.nn.functional as functional

    prediction, _correction, _stats = apply_training_residual(
        residual,
        tensors,
    )
    _fitted_baseline, scale, shift = positive_affine_fit(
        tensors["baseline"],
        tensors["target"],
        tensors["exact_face"],
    )
    candidate = prediction * scale + shift
    baseline = tensors["baseline"] * scale + shift
    target = tensors["target"]
    parts = tensors["parts_individual"].to(candidate.dtype)

    def per_part_mean(values, masks):
        counts = masks.sum(dim=(-2, -1))
        active = counts > 0
        means = (values * masks).sum(dim=(-2, -1)) / counts.clamp_min(1.0)
        means = torch.where(active, means, means.new_full((), -1.0))
        return means

    value_excess = functional.relu(
        (candidate - target).abs() - (baseline - target).abs()
    )
    value_per_part = per_part_mean(value_excess, parts)

    candidate_x = candidate[..., :, 1:] - candidate[..., :, :-1]
    candidate_y = candidate[..., 1:, :] - candidate[..., :-1, :]
    baseline_x = baseline[..., :, 1:] - baseline[..., :, :-1]
    baseline_y = baseline[..., 1:, :] - baseline[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    masks_x = parts[..., :, 1:] * parts[..., :, :-1]
    masks_y = parts[..., 1:, :] * parts[..., :-1, :]
    gradient_x_excess = functional.relu(
        (candidate_x - target_x).abs() - (baseline_x - target_x).abs()
    )
    gradient_y_excess = functional.relu(
        (candidate_y - target_y).abs() - (baseline_y - target_y).abs()
    )
    gradient_per_part = per_part_mean(
        gradient_x_excess,
        masks_x,
    ) + per_part_mean(
        gradient_y_excess,
        masks_y,
    )
    return (
        value_per_part.max(dim=1).values.mean(),
        gradient_per_part.max(dim=1).values.mean(),
    )


def _prepare_direct_items(
    corpus_root: Path,
    surfaces: list,
    processor,
    model,
    *,
    label: str,
) -> list[ProductionDirectFace]:
    import torch

    items = []
    for index, surface in enumerate(surfaces, start=1):
        row = surface.row
        source = Image.open(corpus_root / row["source"]["path"]).convert("RGB")
        selection = Image.open(
            corpus_root / row["selection_mask"]["path"]
        ).convert("L")
        selected = selected_image_from_exact_mask(source, selection)
        crop = selected.crop(surface.bbox)
        pixel_values = processor(
            images=crop,
            return_tensors="pt",
        )["pixel_values"][0].to(dtype=torch.float16)
        items.append(
            ProductionDirectFace(
                row=row,
                corpus_root=corpus_root,
                surface=surface,
                pixel_values=pixel_values.cpu(),
                patch_height=(
                    int(pixel_values.shape[-2] // model.config.patch_size)
                ),
                patch_width=(
                    int(pixel_values.shape[-1] // model.config.patch_size)
                ),
                crop_shape=(crop.height, crop.width),
            )
        )
        if index % 10 == 0 or index == len(surfaces):
            print(
                f"prepared {label} {index}/{len(surfaces)} rows",
                flush=True,
            )
    return items


def _prepare(
    corpus_root: Path,
    cache_root: Path,
    initialization_checkpoint: Path,
    *,
    device: str,
    validation_corpus_root: Path | None = None,
    validation_cache_root: Path | None = None,
    additional_training_corpus_root: Path | None = None,
    additional_training_cache_root: Path | None = None,
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if (validation_corpus_root is None) != (validation_cache_root is None):
        raise ValueError(
            "Validation corpus and cache overrides must be supplied together"
        )
    if (
        additional_training_corpus_root is None
    ) != (
        additional_training_cache_root is None
    ):
        raise ValueError(
            "Additional training corpus and cache must be supplied together"
        )
    surfaces, summary, cache_manifest = load_cached_surfaces(
        corpus_root,
        cache_root,
        surface_support_mode="detector-face",
    )
    checkpoint = torch.load(
        initialization_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    mode, _alpha = _checkpoint_mode_and_alpha(checkpoint)
    if mode != "fusion-head" or checkpoint.get("method") not in {
        GNM_FUSION_METHOD,
        GNM_PRODUCTION_FUSION_METHOD,
    }:
        raise ValueError("Production-aware training requires a GNM fusion checkpoint")
    if checkpoint.get("model_id") != MODEL_ID:
        raise ValueError("Initialization checkpoint model id does not match")
    if checkpoint.get("model_revision") != MODEL_REVISION:
        raise ValueError("Initialization checkpoint model revision does not match")

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    processor = AutoImageProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        backend="pil",
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float16,
    ).to(device)
    _load_state_dict(
        model,
        {
            "fusion_stage": checkpoint["fusion_stage_state_dict"],
            "head": checkpoint["head_state_dict"],
        },
    )
    trainable = _configure_training_scope(model)
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    items = _prepare_direct_items(
        corpus_root,
        surfaces,
        processor,
        model,
        label="primary",
    )
    validation_override = None
    validation_provenance = None
    validation_surfaces = []
    if validation_corpus_root is not None:
        validation_surfaces, validation_summary, validation_manifest = (
            load_cached_surfaces(
                validation_corpus_root,
                validation_cache_root,
                surface_support_mode="detector-face",
            )
        )
        if any(
            surface.row.get("split") != "validation"
            for surface in validation_surfaces
        ):
            raise ValueError(
                "Validation override contains a non-validation row"
            )
        primary_training_identities = {
            surface.row["identity_group"]
            for surface in surfaces
            if surface.row["split"] == "train"
        }
        override_identities = {
            surface.row["identity_group"]
            for surface in validation_surfaces
        }
        overlap = primary_training_identities & override_identities
        if overlap:
            raise ValueError(
                "Validation override overlaps training identities: "
                + ", ".join(sorted(overlap))
            )
        validation_override = _prepare_direct_items(
            validation_corpus_root,
            validation_surfaces,
            processor,
            model,
            label="validation-override",
        )
        validation_provenance = {
            "corpus_summary_sha256": _sha256(
                validation_corpus_root / "summary.json"
            ),
            "cache_manifest_sha256": _sha256(
                validation_cache_root / "manifest.json"
            ),
            "row_count": len(validation_surfaces),
            "identity_count": len(override_identities),
            "selection_strategy": validation_summary.get(
                "selection_strategy"
            ),
            "cache_method": validation_manifest.get("method"),
        }
    additional_training_provenance = None
    if additional_training_corpus_root is not None:
        (
            additional_training_surfaces,
            additional_training_summary,
            additional_training_manifest,
        ) = load_cached_surfaces(
            additional_training_corpus_root,
            additional_training_cache_root,
            surface_support_mode="detector-face",
        )
        contract = _validate_additional_training_contract(
            surfaces,
            validation_surfaces,
            additional_training_surfaces,
            additional_training_summary,
            additional_training_manifest,
        )
        additional_items = _prepare_direct_items(
            additional_training_corpus_root,
            additional_training_surfaces,
            processor,
            model,
            label="additional-training",
        )
        items.extend(additional_items)
        additional_training_provenance = {
            "corpus_summary_sha256": _sha256(
                additional_training_corpus_root / "summary.json"
            ),
            "cache_manifest_sha256": _sha256(
                additional_training_cache_root / "manifest.json"
            ),
            "selection_strategy": additional_training_summary.get(
                "selection_strategy"
            ),
            "cache_method": additional_training_manifest.get("method"),
            **contract,
        }
    return (
        items,
        validation_override,
        model,
        trainable,
        summary,
        cache_manifest,
        checkpoint,
        {
        "snapshot": str(snapshot),
        "model_sha256": _sha256(snapshot / "model.safetensors"),
        "processor_sha256": _sha256(
            snapshot / "preprocessor_config.json"
        ),
        "preparation_peak_vram_gib": float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
        },
        validation_provenance,
        additional_training_provenance,
    )


def _production_training_loss(
    model,
    item: ProductionDirectFace,
    *,
    device: str,
):
    prediction = _prediction(model, item, device)
    tensors = {
        key: value.to(device)
        for key, value in _stack([item.surface]).items()
    }
    normalized = normalize_prediction_to_shape(
        prediction,
        source_shape=item.crop_shape,
        target_shape=tuple(tensors["local_depth"].shape[-2:]),
    )
    scale = residual_application_scale(
        item.row["render"]["face_bbox_height_pixels"]
    )
    residual = (
        normalized - tensors["local_depth"][0, 0]
    ) * scale
    loss, details = surface_fusion_training_loss(
        residual[None, None],
        tensors,
        part_balanced_physical_value_weight=(
            PART_BALANCED_VALUE_WEIGHT
        ),
        part_balanced_physical_gradient_weight=(
            PART_BALANCED_GRADIENT_WEIGHT
        ),
        part_balanced_non_regression_weight=(
            PART_BALANCED_NON_REGRESSION_WEIGHT
        ),
    )
    worst_value, worst_gradient = worst_part_physical_non_regression(
        residual[None, None],
        tensors,
    )
    loss = (
        loss
        + WORST_PART_VALUE_NON_REGRESSION_WEIGHT * worst_value
        + WORST_PART_GRADIENT_NON_REGRESSION_WEIGHT * worst_gradient
    )
    return loss, {
        **details,
        "residual_application_scale": scale,
        "worst_part_value_non_regression": float(
            worst_value.detach()
        ),
        "worst_part_gradient_non_regression": float(
            worst_gradient.detach()
        ),
    }


def _baseline_summary(
    corpus_root: Path,
    items: list[ProductionDirectFace],
    output_dir: Path,
    label: str,
) -> dict:
    rows = []
    for item in items:
        path = output_dir / label / f"{item.row['row_id']}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, item.surface.baseline_full)
        rows.append(
            {
                "row_id": item.row["row_id"],
                "face_height_pixels": item.row["render"][
                    "face_bbox_height_pixels"
                ],
                **_quality(item.corpus_root, item.row, path),
            }
        )
    return _summary(rows)


def _evaluate_model(
    corpus_root: Path,
    model,
    items: list[ProductionDirectFace],
    output_dir: Path,
    label: str,
    *,
    alpha: float,
    device: str,
) -> dict:
    import torch

    model.neck.fusion_stage.eval()
    model.head.eval()
    rows = []
    with torch.inference_mode():
        for item in items:
            prediction = _prediction(model, item, device)
            trained = normalize_prediction_to_shape(
                prediction,
                source_shape=item.crop_shape,
                target_shape=item.surface.local_native.shape,
            ).float().cpu().numpy()
            scale = residual_application_scale(
                item.row["render"]["face_bbox_height_pixels"]
            )
            residual = (
                trained - item.surface.local_native
            ) * scale * float(alpha)
            x0, y0, x1, y1 = item.surface.bbox
            baseline_crop = item.surface.baseline_full[y0:y1, x0:x1]
            refined_crop, _weight, fusion = fuse_face_surface_residual(
                baseline_crop,
                item.surface.local_native,
                residual,
                item.surface.support_native.astype(np.uint8) * 255,
                alignment_depth=baseline_crop,
            )
            candidate = item.surface.baseline_full.copy()
            candidate[y0:y1, x0:x1] = refined_crop
            path = output_dir / label / f"{item.row['row_id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, candidate)
            rows.append(
                {
                    "row_id": item.row["row_id"],
                    "face_height_pixels": item.row["render"][
                        "face_bbox_height_pixels"
                    ],
                    "residual_application_scale": scale,
                    "max_abs_correction": fusion["max_abs_correction"],
                    "correction_limit": fusion["correction_limit"],
                    **_quality(item.corpus_root, item.row, path),
                }
            )
    return _summary(rows)


def _rank(summary: dict, alpha: float = 1.0) -> tuple:
    return (
        summary["combined_part_failures"],
        summary["median_normalized_rmse"],
        -summary["median_shape_correlation"],
        -summary["median_gradient_correlation"],
        float(alpha),
    )


def _selection_record(candidate: dict, baseline: dict) -> dict:
    record = {
        **candidate,
        "strictly_improves": _strictly_improves(candidate, baseline),
        "per_row_no_failure_regression": _per_row_no_failure_regression(
            candidate,
            baseline,
        ),
    }
    record["eligible"] = bool(
        record["strictly_improves"]
        and record["per_row_no_failure_regression"]
    )
    return record


def _summaries_equivalent(candidate: dict, baseline: dict) -> bool:
    candidate_by_id = {
        row["row_id"]: row["combined_part_failures"]
        for row in candidate["rows"]
    }
    baseline_by_id = {
        row["row_id"]: row["combined_part_failures"]
        for row in baseline["rows"]
    }
    return bool(
        candidate_by_id == baseline_by_id
        and candidate["combined_part_failures"]
        == baseline["combined_part_failures"]
        and np.isclose(
            candidate["median_shape_correlation"],
            baseline["median_shape_correlation"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            candidate["median_gradient_correlation"],
            baseline["median_gradient_correlation"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            candidate["median_normalized_rmse"],
            baseline["median_normalized_rmse"],
            rtol=0.0,
            atol=1e-12,
        )
    )


def train(
    corpus_root: str | Path,
    cache_root: str | Path,
    initialization_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = "cuda",
    validation_corpus_root: str | Path | None = None,
    validation_cache_root: str | Path | None = None,
    additional_training_corpus_root: str | Path | None = None,
    additional_training_cache_root: str | Path | None = None,
) -> dict:
    import torch

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    initialization_checkpoint = Path(initialization_checkpoint)
    output_dir = Path(output_dir)
    validation_corpus_root = (
        Path(validation_corpus_root)
        if validation_corpus_root is not None
        else None
    )
    validation_cache_root = (
        Path(validation_cache_root)
        if validation_cache_root is not None
        else None
    )
    additional_training_corpus_root = (
        Path(additional_training_corpus_root)
        if additional_training_corpus_root is not None
        else None
    )
    additional_training_cache_root = (
        Path(additional_training_cache_root)
        if additional_training_cache_root is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Production-aware GNM training requires CUDA")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    (
        items,
        validation_override,
        model,
        trainable,
        summary,
        cache_manifest,
        initialization,
        model_provenance,
        validation_provenance,
        additional_training_provenance,
    ) = _prepare(
        corpus_root,
        cache_root,
        initialization_checkpoint,
        device=device,
        validation_corpus_root=validation_corpus_root,
        validation_cache_root=validation_cache_root,
        additional_training_corpus_root=additional_training_corpus_root,
        additional_training_cache_root=additional_training_cache_root,
    )
    by_split = {
        split: [item for item in items if item.row["split"] == split]
        for split in ("train", "validation", "sealed")
    }
    if validation_override is not None:
        by_split["validation"] = validation_override
    if any(not split_items for split_items in by_split.values()):
        raise ValueError("Training, validation, and sealed rows are required")
    baseline_validation = _baseline_summary(
        corpus_root,
        by_split["validation"],
        output_dir,
        "validation_baseline",
    )
    baseline_sealed = _baseline_summary(
        corpus_root,
        by_split["sealed"],
        output_dir,
        "sealed_baseline",
    )
    initial_validation = _evaluate_model(
        corpus_root,
        model,
        by_split["validation"],
        output_dir,
        "validation_initial",
        alpha=1.0,
        device=device,
    )
    initial_selection_validation = _selection_record(
        _evaluate_model(
            corpus_root,
            model,
            by_split["validation"],
            output_dir,
            "validation_initial_selection",
            alpha=EPOCH_SELECTION_ALPHA,
            device=device,
        ),
        baseline_validation,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(learning_rate),
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = random.Random(TRAINING_SEED)
    best_state = _state_dict(model)
    best_epoch = 0
    best_rank = (
        not initial_selection_validation["eligible"],
        *_rank(
            initial_selection_validation,
            EPOCH_SELECTION_ALPHA,
        ),
    )
    epochs_record = []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        model.neck.fusion_stage.train()
        model.head.train()
        order = list(by_split["train"])
        generator.shuffle(order)
        losses = []
        for item in order:
            optimizer.zero_grad(set_to_none=True)
            loss, details = _production_training_loss(
                model,
                item,
                device=device,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append({"total": float(loss.detach()), **details})
        validation = _evaluate_model(
            corpus_root,
            model,
            by_split["validation"],
            output_dir,
            f"validation_epoch_{epoch:02d}",
            alpha=1.0,
            device=device,
        )
        selection_validation = _selection_record(
            _evaluate_model(
                corpus_root,
                model,
                by_split["validation"],
                output_dir,
                f"validation_epoch_{epoch:02d}_selection",
                alpha=EPOCH_SELECTION_ALPHA,
                device=device,
            ),
            baseline_validation,
        )
        rank = (
            not selection_validation["eligible"],
            *_rank(
                selection_validation,
                EPOCH_SELECTION_ALPHA,
            ),
        )
        if rank < best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = _state_dict(model)
        epochs_record.append(
            {
                "epoch": epoch,
                "mean_total_loss": float(
                    np.mean([record["total"] for record in losses])
                ),
                "mean_part_balanced_value": float(
                    np.mean(
                        [
                            record["part_balanced_physical_value"]
                            for record in losses
                        ]
                    )
                ),
                "mean_part_balanced_gradient": float(
                    np.mean(
                        [
                            record["part_balanced_physical_gradient"]
                            for record in losses
                        ]
                    )
                ),
                "mean_part_non_regression": float(
                    np.mean(
                        [
                            record[
                                "part_balanced_value_non_regression"
                            ]
                            + record[
                                "part_balanced_gradient_non_regression"
                            ]
                            for record in losses
                        ]
                    )
                ),
                "mean_worst_part_value_non_regression": float(
                    np.mean(
                        [
                            record[
                                "worst_part_value_non_regression"
                            ]
                            for record in losses
                        ]
                    )
                ),
                "mean_worst_part_gradient_non_regression": float(
                    np.mean(
                        [
                            record[
                                "worst_part_gradient_non_regression"
                            ]
                            for record in losses
                        ]
                    )
                ),
                "validation": validation,
                "selection_validation": selection_validation,
            }
        )
        print(
            f"epoch {epoch}: validation failures "
            f"{validation['combined_part_failures']}",
            flush=True,
        )

    _load_state_dict(model, best_state)
    blend_candidates = []
    for alpha in BLEND_ALPHAS:
        candidate = (
            baseline_validation
            if alpha == 0.0
            else _evaluate_model(
                corpus_root,
                model,
                by_split["validation"],
                output_dir,
                f"validation_blend_{alpha:g}",
                alpha=alpha,
                device=device,
            )
        )
        candidate = {**candidate, "alpha": float(alpha)}
        candidate = _selection_record(
            candidate,
            baseline_validation,
        )
        blend_candidates.append(candidate)
    eligible = [
        candidate for candidate in blend_candidates if candidate["eligible"]
    ]
    selected = min(
        eligible if eligible else blend_candidates,
        key=lambda candidate: _rank(candidate, candidate["alpha"]),
    )
    sealed = (
        baseline_sealed
        if selected["alpha"] == 0.0
        else _evaluate_model(
            corpus_root,
            model,
            by_split["sealed"],
            output_dir,
            "sealed_selected",
            alpha=selected["alpha"],
            device=device,
        )
    )
    sealed = {**sealed, "alpha": selected["alpha"]}
    sealed["strictly_improves"] = _strictly_improves(
        sealed,
        baseline_sealed,
    )
    sealed["per_row_no_failure_regression"] = (
        _per_row_no_failure_regression(sealed, baseline_sealed)
    )
    sealed["outside_application_domain"] = bool(
        all(
            item.row["render"]["face_bbox_height_pixels"]
            >= SMALL_FACE_RESIDUAL_ZERO_STRENGTH_PIXELS
            for item in by_split["sealed"]
        )
    )
    sealed["equivalent_to_baseline"] = _summaries_equivalent(
        sealed,
        baseline_sealed,
    )
    decision = {
        "validation_blend_eligible": bool(selected["eligible"]),
        "sealed_per_row_no_failure_regression": bool(
            sealed["per_row_no_failure_regression"]
        ),
        "sealed_domain_contract_passes": bool(
            sealed["strictly_improves"]
            or (
                sealed["outside_application_domain"]
                and sealed["equivalent_to_baseline"]
            )
        ),
    }
    decision["advance_to_exact_production_replay"] = bool(
        all(decision.values())
    )

    checkpoint_path = output_dir / "gnm_dav2_production_fusion_head.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": GNM_PRODUCTION_FUSION_METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "training_seed": TRAINING_SEED,
            "selected_epoch": best_epoch,
            "selected_alpha": selected["alpha"],
            "corpus_summary_sha256": _sha256(
                corpus_root / "summary.json"
            ),
            "production_cache_manifest_sha256": _sha256(
                cache_root / "manifest.json"
            ),
            "initialization_checkpoint_sha256": _sha256(
                initialization_checkpoint
            ),
            "additional_training": additional_training_provenance,
            "fusion_stage_state_dict": {
                name: value.detach().cpu()
                for name, value in best_state["fusion_stage"].items()
            },
            "head_state_dict": {
                name: value.detach().cpu()
                for name, value in best_state["head"].items()
            },
        },
        checkpoint_path,
    )
    evidence = {
        "schema_version": 1,
        "method": GNM_PRODUCTION_FUSION_METHOD,
        "source_geometry_training_and_evaluation_only": True,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "trainable_scope": "neck.fusion_stage+head",
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in trainable)
            ),
            **model_provenance,
        },
        "corpus": {
            "summary_sha256": _sha256(corpus_root / "summary.json"),
            "production_cache_manifest_sha256": _sha256(
                cache_root / "manifest.json"
            ),
            "privacy": summary.get("privacy"),
            "split_counts": {
                split: len(split_items)
                for split, split_items in by_split.items()
            },
            "cache_method": cache_manifest.get("method"),
            "validation_override": validation_provenance,
            "additional_training": additional_training_provenance,
        },
        "initialization": {
            "method": initialization.get("method"),
            "checkpoint_sha256": _sha256(initialization_checkpoint),
            "selected_epoch": initialization.get("selected_epoch"),
            "selected_alpha": initialization.get("selected_alpha"),
        },
        "training": {
            "seed": TRAINING_SEED,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "production_operations": [
                "remove-affine-component-relative-to-local-depth",
                "alignment-scale",
                "correction-cap",
                "boundary-zero-fusion-weight",
                "apparent-face-size-residual-scale",
            ],
            "residual_scale_formula": (
                "1 through 90 px; linear taper; 0 at or above 110 px"
            ),
            "zero_strength_face_height_pixels": (
                SMALL_FACE_RESIDUAL_ZERO_STRENGTH_PIXELS
            ),
            "loss_weights": {
                "part_balanced_physical_value": (
                    PART_BALANCED_VALUE_WEIGHT
                ),
                "part_balanced_physical_gradient": (
                    PART_BALANCED_GRADIENT_WEIGHT
                ),
                "part_balanced_non_regression": (
                    PART_BALANCED_NON_REGRESSION_WEIGHT
                ),
                "worst_part_value_non_regression": (
                    WORST_PART_VALUE_NON_REGRESSION_WEIGHT
                ),
                "worst_part_gradient_non_regression": (
                    WORST_PART_GRADIENT_NON_REGRESSION_WEIGHT
                ),
            },
            "selected_epoch": best_epoch,
            "epoch_selection_alpha": EPOCH_SELECTION_ALPHA,
            "epochs_record": epochs_record,
            "peak_vram_gib": float(
                torch.cuda.max_memory_allocated(device) / (1024**3)
            ),
        },
        "validation": {
            "baseline": baseline_validation,
            "initial_checkpoint": initial_validation,
            "initial_selection": initial_selection_validation,
            "blend_candidates": blend_candidates,
            "selected": selected,
        },
        "sealed": {
            "baseline": baseline_sealed,
            "selected": sealed,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "device": torch.cuda.get_device_name(device),
        "decision": decision,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    torch.cuda.empty_cache()
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--initialization-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-corpus-root")
    parser.add_argument("--validation-cache-root")
    parser.add_argument("--additional-training-corpus-root")
    parser.add_argument("--additional-training-cache-root")
    args = parser.parse_args()
    evidence = train(
        args.corpus_root,
        args.cache_root,
        args.initialization_checkpoint,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        validation_corpus_root=args.validation_corpus_root,
        validation_cache_root=args.validation_cache_root,
        additional_training_corpus_root=(
            args.additional_training_corpus_root
        ),
        additional_training_cache_root=args.additional_training_cache_root,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["advance_to_exact_production_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
