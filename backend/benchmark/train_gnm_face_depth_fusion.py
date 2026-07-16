"""Fine-tune DAv2 multiscale fusion on camera-aligned GNM face depth."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import random
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import FACE_PART_NAMES
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _crop_row,
    _load_mask,
    _quality,
    _resize,
    _sha256,
    _summarize,
)


TRAINING_SEED = 20260717
DEFAULT_EPOCHS = 6
DEFAULT_LEARNING_RATE = 2e-5
BLEND_ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
FACE_VALUE_WEIGHT = 0.50
EQUAL_PART_VALUE_WEIGHT = 0.75
FACE_GRADIENT_WEIGHT = 0.50
EQUAL_PART_GRADIENT_WEIGHT = 0.75
LAPLACIAN_WEIGHT = 0.20
BACKGROUND_DISTILLATION_WEIGHT = 0.05
MINIMUM_GLOBAL_GRADIENT_RETENTION = 0.9975


@dataclass
class CachedFusionFace:
    row: dict
    pixel_values: object
    baseline: object
    target: object
    face_mask: object
    part_masks: object
    patch_height: int
    patch_width: int
    bbox: tuple[int, int, int, int]
    sample_weight: float


def _positive_affine_fit(prediction, target, mask):
    weights = mask.to(dtype=prediction.dtype)
    count = weights.sum().clamp_min(1.0)
    prediction_mean = (prediction * weights).sum() / count
    target_mean = (target * weights).sum() / count
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    variance = (centered_prediction.square() * weights).sum() / count
    covariance = (centered_prediction * centered_target * weights).sum() / count
    scale = torch_clamp(covariance / variance.clamp_min(1e-6), 1e-4, 100.0)
    shift = target_mean - scale * prediction_mean
    return scale * prediction + shift, scale, shift


def torch_clamp(values, minimum: float, maximum: float):
    import torch

    return torch.clamp(values, minimum, maximum)


def _masked_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _axis_gradient_loss(prediction, target, mask):
    valid_x = mask[:, 1:] * mask[:, :-1]
    valid_y = mask[1:, :] * mask[:-1, :]
    difference_x = (prediction[:, 1:] - prediction[:, :-1]) - (
        target[:, 1:] - target[:, :-1]
    )
    difference_y = (prediction[1:, :] - prediction[:-1, :]) - (
        target[1:, :] - target[:-1, :]
    )
    return _masked_mean(difference_x.abs(), valid_x) + _masked_mean(
        difference_y.abs(), valid_y
    )


def _equal_mask_mean(loss_function, prediction, target, masks):
    values = [
        loss_function(prediction, target, mask)
        for mask in masks
        if int(mask.count_nonzero()) >= 1
    ]
    if not values:
        raise ValueError("Face-depth training row has no usable part masks")
    return sum(values) / len(values)


def fusion_training_loss(prediction, item: CachedFusionFace):
    import torch

    prediction = prediction.float()
    target = item.target.to(device=prediction.device, dtype=torch.float32)
    face = item.face_mask.to(device=prediction.device, dtype=torch.float32)
    parts = item.part_masks.to(device=prediction.device, dtype=torch.float32)
    baseline = item.baseline.to(device=prediction.device, dtype=torch.float32)
    fitted, scale, shift = _positive_affine_fit(prediction, target, face)
    baseline_fitted, _, _ = _positive_affine_fit(baseline, target, face)

    face_value = _masked_mean((fitted - target).abs(), face)
    part_value = _equal_mask_mean(
        lambda candidate, reference, mask: _masked_mean(
            (candidate - reference).abs(), mask
        ),
        fitted,
        target,
        parts,
    )
    face_gradient = _axis_gradient_loss(fitted, target, face)
    part_gradient = _equal_mask_mean(
        _axis_gradient_loss,
        fitted,
        target,
        parts,
    )
    laplace_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=prediction.device,
        dtype=torch.float32,
    )[None, None]
    fitted_laplace = torch.nn.functional.conv2d(
        fitted[None, None], laplace_kernel, padding=1
    )[0, 0]
    target_laplace = torch.nn.functional.conv2d(
        target[None, None], laplace_kernel, padding=1
    )[0, 0]
    laplacian = _masked_mean(
        (fitted_laplace - target_laplace).abs(),
        face,
    )
    background = 1.0 - face
    background_distillation = _masked_mean(
        (prediction - item.baseline.to(prediction.device).float()).abs(),
        background,
    )
    baseline_distance = _masked_mean((fitted - baseline_fitted).abs(), face)
    total = float(item.sample_weight) * (
        FACE_VALUE_WEIGHT * face_value
        + EQUAL_PART_VALUE_WEIGHT * part_value
        + FACE_GRADIENT_WEIGHT * face_gradient
        + EQUAL_PART_GRADIENT_WEIGHT * part_gradient
        + LAPLACIAN_WEIGHT * laplacian
        + BACKGROUND_DISTILLATION_WEIGHT * background_distillation
    )
    return total, {
        "face_value": float(face_value.detach()),
        "part_value": float(part_value.detach()),
        "face_gradient": float(face_gradient.detach()),
        "part_gradient": float(part_gradient.detach()),
        "laplacian": float(laplacian.detach()),
        "background_distillation": float(background_distillation.detach()),
        "baseline_distance": float(baseline_distance.detach()),
        "fit_scale": float(scale.detach()),
        "fit_shift": float(shift.detach()),
    }


def _configure_training_scope(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.neck.fusion_stage.parameters():
        parameter.requires_grad_(True)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    model.neck.fusion_stage.float()
    model.head.float()
    return [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]


def _frozen_features(model, item: CachedFusionFace, device: str):
    import torch

    pixel_values = item.pixel_values.to(device=device, dtype=torch.float16)[None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        backbone = model.backbone.forward_with_filtered_kwargs(pixel_values)
        reassembled = model.neck.reassemble_stage(
            backbone.feature_maps,
            item.patch_height,
            item.patch_width,
        )
        return [
            model.neck.convs[index](feature)
            for index, feature in enumerate(reassembled)
        ]


def _prediction(model, item: CachedFusionFace, device: str):
    import torch

    features = _frozen_features(model, item, device)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = model.neck.fusion_stage(features)
        return model.head(
            hidden,
            item.patch_height,
            item.patch_width,
        )[0]


def _prepare_items(
    corpus_root: Path,
    rows: list[dict],
    *,
    device: str,
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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
    model.eval()
    trainable = _configure_training_scope(model)
    prepared_items = []
    torch.cuda.reset_peak_memory_stats(device)
    for index, row in enumerate(rows):
        prepared = _crop_row(corpus_root, row)
        pixel_values = processor(
            images=prepared["source_crop"],
            return_tensors="pt",
        )["pixel_values"].to(device=device, dtype=torch.float16)
        patch_height = int(pixel_values.shape[-2] // model.config.patch_size)
        patch_width = int(pixel_values.shape[-1] // model.config.patch_size)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            backbone = model.backbone.forward_with_filtered_kwargs(pixel_values)
            hidden = model.neck(
                backbone.feature_maps,
                patch_height,
                patch_width,
            )
            baseline = model.head(hidden, patch_height, patch_width)[0]
        output_shape = tuple(int(value) for value in baseline.shape)
        x0, y0, x1, y1 = prepared["bbox"]
        target = _resize(
            prepared["target_crop"],
            output_shape,
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        face = _resize(
            prepared["face_crop"].astype(np.uint8),
            output_shape,
            cv2.INTER_NEAREST,
        ).astype(bool)
        part_masks = []
        for name in FACE_PART_NAMES:
            part = _load_mask(
                corpus_root / row["exact_face_parts"][name]["path"]
            )[y0:y1, x0:x1]
            part_masks.append(
                _resize(
                    part.astype(np.uint8),
                    output_shape,
                    cv2.INTER_NEAREST,
                ).astype(bool)
            )
        if any(np.count_nonzero(part) < 4 for part in part_masks):
            raise ValueError(f"Corpus row {row['row_id']} has a tiny part mask")
        prepared_items.append(
            CachedFusionFace(
                row=row,
                pixel_values=pixel_values[0].cpu(),
                baseline=baseline.detach().cpu().to(torch.float16),
                target=torch.from_numpy(target),
                face_mask=torch.from_numpy(face),
                part_masks=torch.from_numpy(np.stack(part_masks)),
                patch_height=patch_height,
                patch_width=patch_width,
                bbox=prepared["bbox"],
                sample_weight=float(
                    np.clip(
                        90.0
                        / max(
                            float(row["render"]["face_bbox_height_pixels"]),
                            1.0,
                        ),
                        1.0,
                        1.5,
                    )
                ),
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"prepared {index + 1}/{len(rows)} rows", flush=True)
    return prepared_items, model, trainable, {
        "snapshot": str(snapshot),
        "model_sha256": _sha256(snapshot / "model.safetensors"),
        "processor_sha256": _sha256(snapshot / "preprocessor_config.json"),
        "peak_vram_gib": float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        ),
    }


def _evaluate_predictions(
    corpus_root: Path,
    items: list[CachedFusionFace],
    predictions: list[np.ndarray],
    *,
    output_dir: Path,
    label: str,
) -> dict:
    rows = [
        _quality(
            corpus_root,
            item,
            prediction,
            output_dir,
            label,
        )
        for item, prediction in zip(items, predictions, strict=True)
    ]
    return _summarize(rows)


def _baseline_predictions(items: list[CachedFusionFace]) -> list[np.ndarray]:
    return [item.baseline.float().numpy() for item in items]


def _predict_items(model, items, *, device: str) -> list[np.ndarray]:
    import torch

    model.neck.fusion_stage.eval()
    model.head.eval()
    predictions = []
    with torch.inference_mode():
        for item in items:
            predictions.append(
                _prediction(model, item, device).float().cpu().numpy()
            )
    return predictions


def _blend_predictions(
    items: list[CachedFusionFace],
    trained: list[np.ndarray],
    alpha: float,
) -> list[np.ndarray]:
    return [
        (
            (1.0 - float(alpha)) * item.baseline.float().numpy()
            + float(alpha) * candidate
        ).astype(np.float32)
        for item, candidate in zip(items, trained, strict=True)
    ]


def _per_row_non_regression(candidate: dict, baseline: dict) -> float:
    baseline_by_id = {
        row["row_id"]: (
            len(row["shape_failed_parts"]) + len(row["affine_failed_parts"])
        )
        for row in baseline["rows"]
    }
    passed = sum(
        len(row["shape_failed_parts"]) + len(row["affine_failed_parts"])
        <= baseline_by_id[row["row_id"]]
        for row in candidate["rows"]
    )
    return float(passed / max(len(candidate["rows"]), 1))


def _strictly_improves(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["combined_part_failures"]
        < baseline["combined_part_failures"]
        and candidate["median_shape_correlation"]
        >= baseline["median_shape_correlation"] - 1e-6
        and candidate["median_gradient_correlation"]
        >= (
            baseline["median_gradient_correlation"]
            * MINIMUM_GLOBAL_GRADIENT_RETENTION
        )
        and candidate["median_normalized_rmse"]
        <= baseline["median_normalized_rmse"] + 1e-6
    )


def _state_dict(model) -> dict:
    return {
        "fusion_stage": copy.deepcopy(model.neck.fusion_stage.state_dict()),
        "head": copy.deepcopy(model.head.state_dict()),
    }


def _load_state_dict(model, state: dict) -> None:
    model.neck.fusion_stage.load_state_dict(state["fusion_stage"])
    model.head.load_state_dict(state["head"])


def train(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str = "cuda",
) -> dict:
    import torch

    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["rows"]
    split_counts = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation", "sealed")
    }
    if any(count < 1 for count in split_counts.values()):
        raise ValueError("Training, validation, and sealed rows are required")
    if summary.get("selection_strategy") != "identity-stratified":
        raise ValueError("GNM fusion training requires identity-stratified rows")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GNM fusion training requires CUDA")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    items, model, trainable, model_provenance = _prepare_items(
        corpus_root,
        rows,
        device=device,
    )
    by_split = {
        name: [item for item in items if item.row["split"] == name]
        for name in split_counts
    }
    baseline_validation = _evaluate_predictions(
        corpus_root,
        by_split["validation"],
        _baseline_predictions(by_split["validation"]),
        output_dir=output_dir,
        label="validation_baseline",
    )
    baseline_sealed = _evaluate_predictions(
        corpus_root,
        by_split["sealed"],
        _baseline_predictions(by_split["sealed"]),
        output_dir=output_dir,
        label="sealed_baseline",
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
        baseline_validation["combined_part_failures"],
        baseline_validation["median_normalized_rmse"],
        -baseline_validation["median_shape_correlation"],
        -baseline_validation["median_gradient_correlation"],
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
            prediction = _prediction(model, item, device)
            loss, components = fusion_training_loss(prediction, item)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append({"total": float(loss.detach()), **components})
        validation_predictions = _predict_items(
            model,
            by_split["validation"],
            device=device,
        )
        validation = _evaluate_predictions(
            corpus_root,
            by_split["validation"],
            validation_predictions,
            output_dir=output_dir,
            label=f"validation_epoch_{epoch:02d}",
        )
        rank = (
            validation["combined_part_failures"],
            validation["median_normalized_rmse"],
            -validation["median_shape_correlation"],
            -validation["median_gradient_correlation"],
        )
        if rank < best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = _state_dict(model)
        epochs_record.append(
            {
                "epoch": epoch,
                "mean_total_loss": float(
                    np.mean([row["total"] for row in losses])
                ),
                "mean_part_value_loss": float(
                    np.mean([row["part_value"] for row in losses])
                ),
                "mean_part_gradient_loss": float(
                    np.mean([row["part_gradient"] for row in losses])
                ),
                "mean_baseline_distance": float(
                    np.mean([row["baseline_distance"] for row in losses])
                ),
                "validation": validation,
            }
        )
        print(
            f"epoch {epoch}: validation failures "
            f"{validation['combined_part_failures']}",
            flush=True,
        )

    _load_state_dict(model, best_state)
    validation_trained = _predict_items(
        model,
        by_split["validation"],
        device=device,
    )
    blend_candidates = []
    for alpha in BLEND_ALPHAS:
        candidate = _evaluate_predictions(
            corpus_root,
            by_split["validation"],
            _blend_predictions(by_split["validation"], validation_trained, alpha),
            output_dir=output_dir,
            label=f"validation_blend_{alpha:g}",
        )
        candidate["alpha"] = float(alpha)
        candidate["strictly_improves"] = _strictly_improves(
            candidate,
            baseline_validation,
        )
        candidate["per_row_non_regression_ratio"] = _per_row_non_regression(
            candidate,
            baseline_validation,
        )
        candidate["eligible"] = bool(
            candidate["strictly_improves"]
            and candidate["per_row_non_regression_ratio"] >= 1.0
        )
        blend_candidates.append(candidate)
    eligible = [candidate for candidate in blend_candidates if candidate["eligible"]]
    selected = min(
        eligible if eligible else blend_candidates,
        key=lambda candidate: (
            candidate["combined_part_failures"],
            candidate["median_normalized_rmse"],
            -candidate["median_shape_correlation"],
            -candidate["median_gradient_correlation"],
            candidate["alpha"],
        ),
    )

    sealed_trained = _predict_items(
        model,
        by_split["sealed"],
        device=device,
    )
    sealed = _evaluate_predictions(
        corpus_root,
        by_split["sealed"],
        _blend_predictions(
            by_split["sealed"],
            sealed_trained,
            selected["alpha"],
        ),
        output_dir=output_dir,
        label="sealed_selected",
    )
    sealed["alpha"] = selected["alpha"]
    sealed["strictly_improves"] = _strictly_improves(
        sealed,
        baseline_sealed,
    )
    sealed["per_row_non_regression_ratio"] = _per_row_non_regression(
        sealed,
        baseline_sealed,
    )
    decision = {
        "validation_blend_eligible": bool(selected["eligible"]),
        "sealed_strictly_improves": bool(sealed["strictly_improves"]),
        "sealed_per_row_non_regression": bool(
            sealed["per_row_non_regression_ratio"] >= 1.0
        ),
    }
    decision["advance_to_exact_production_replay"] = bool(all(decision.values()))

    checkpoint_path = output_dir / "gnm_dav2_fusion_head.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": "gnm_dav2_multiscale_fusion_head",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "training_seed": TRAINING_SEED,
            "selected_epoch": best_epoch,
            "selected_alpha": selected["alpha"],
            "corpus_summary_sha256": _sha256(summary_path),
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
        "method": "gnm_dav2_multiscale_fusion_head",
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
            "summary_sha256": _sha256(summary_path),
            "selection_strategy": summary.get("selection_strategy"),
            "split_counts": split_counts,
            "privacy": summary.get("privacy"),
        },
        "training": {
            "seed": TRAINING_SEED,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "loss_weights": {
                "face_value": FACE_VALUE_WEIGHT,
                "equal_part_value": EQUAL_PART_VALUE_WEIGHT,
                "face_gradient": FACE_GRADIENT_WEIGHT,
                "equal_part_gradient": EQUAL_PART_GRADIENT_WEIGHT,
                "laplacian": LAPLACIAN_WEIGHT,
                "background_distillation": BACKGROUND_DISTILLATION_WEIGHT,
            },
            "selected_epoch": best_epoch,
            "epochs_record": epochs_record,
            "peak_vram_gib": float(
                torch.cuda.max_memory_allocated(device) / (1024**3)
            ),
        },
        "validation": {
            "baseline": baseline_validation,
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = train(
        args.corpus_root,
        args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["advance_to_exact_production_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
