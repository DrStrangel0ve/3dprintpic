"""Fine-tune only the DAv2 depth head on exact CC0 face geometry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)


MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
MODEL_REVISION = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
FACE_PADDING_RATIO = 0.35
TRAINING_SEED = 20260716
PART_WEIGHT = 2.0
BACKGROUND_DISTILLATION_WEIGHT = 0.05
GRADIENT_LOSS_WEIGHT = 0.75
LAPLACIAN_LOSS_WEIGHT = 0.20
BLEND_ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)


@dataclass
class CachedFace:
    row: dict
    pixel_values: object
    baseline: object
    target: object
    face_mask: object
    part_weight: object
    patch_height: int
    patch_width: int
    bbox: tuple[int, int, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _padded_box(
    box: list[int] | tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float = FACE_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(value) for value in box)
    pad_x = int(round((x1 - x0) * max(float(padding_ratio), 0.0)))
    pad_y = int(round((y1 - y0) * max(float(padding_ratio), 0.0)))
    return (
        max(x0 - pad_x, 0),
        max(y0 - pad_y, 0),
        min(x1 + pad_x, width),
        min(y1 + pad_y, height),
    )


def _resize(values: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(values),
        (shape[1], shape[0]),
        interpolation=interpolation,
    )


def _near_high_target(exact_depth: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    exact = np.asarray(exact_depth, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool) & np.isfinite(exact)
    if np.count_nonzero(face) < 64:
        raise ValueError("A training crop needs at least 64 finite face pixels")
    low, high = np.percentile(exact[face], (2.0, 98.0))
    span = max(float(high - low), 1e-6)
    target = np.zeros(exact.shape, dtype=np.float32)
    target[face] = np.clip(
        (float(high) - exact[face]) / span,
        -0.5,
        1.5,
    )
    return target


def _fit_prediction(prediction, target, mask):
    import torch

    weights = mask.to(dtype=prediction.dtype)
    count = weights.sum().clamp_min(1.0)
    x_mean = (prediction * weights).sum() / count
    y_mean = (target * weights).sum() / count
    centered_x = prediction - x_mean
    centered_y = target - y_mean
    variance = (centered_x.square() * weights).sum() / count
    covariance = (centered_x * centered_y * weights).sum() / count
    scale = torch.clamp(covariance / variance.clamp_min(1e-6), 1e-4, 100.0)
    shift = y_mean - scale * x_mean
    return scale * prediction + shift, scale, shift


def _masked_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _training_loss(prediction, item: CachedFace):
    import torch

    target = item.target.to(device=prediction.device, dtype=prediction.dtype)
    face = item.face_mask.to(device=prediction.device, dtype=prediction.dtype)
    part_weight = item.part_weight.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    baseline = item.baseline.to(device=prediction.device, dtype=prediction.dtype)
    fitted, scale, shift = _fit_prediction(prediction, target, face)
    weights = face * (1.0 + PART_WEIGHT * part_weight)
    value_loss = _masked_mean(torch.abs(fitted - target), weights)

    valid_x = face[:, 1:] * face[:, :-1]
    valid_y = face[1:, :] * face[:-1, :]
    gradient_x = (fitted[:, 1:] - fitted[:, :-1]) - (
        target[:, 1:] - target[:, :-1]
    )
    gradient_y = (fitted[1:, :] - fitted[:-1, :]) - (
        target[1:, :] - target[:-1, :]
    )
    gradient_loss = _masked_mean(torch.abs(gradient_x), valid_x)
    gradient_loss += _masked_mean(torch.abs(gradient_y), valid_y)

    laplace_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=prediction.device,
        dtype=prediction.dtype,
    )[None, None]
    fitted_laplace = torch.nn.functional.conv2d(
        fitted[None, None],
        laplace_kernel,
        padding=1,
    )[0, 0]
    target_laplace = torch.nn.functional.conv2d(
        target[None, None],
        laplace_kernel,
        padding=1,
    )[0, 0]
    laplacian_loss = _masked_mean(
        torch.abs(fitted_laplace - target_laplace),
        weights,
    )
    background = 1.0 - face
    distillation_loss = _masked_mean(
        torch.abs(prediction - baseline),
        background,
    )
    total = (
        value_loss
        + GRADIENT_LOSS_WEIGHT * gradient_loss
        + LAPLACIAN_LOSS_WEIGHT * laplacian_loss
        + BACKGROUND_DISTILLATION_WEIGHT * distillation_loss
    )
    return total, {
        "value": float(value_loss.detach()),
        "gradient": float(gradient_loss.detach()),
        "laplacian": float(laplacian_loss.detach()),
        "background_distillation": float(distillation_loss.detach()),
        "fit_scale": float(scale.detach()),
        "fit_shift": float(shift.detach()),
    }


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) >= 128


def _crop_row(corpus_root: Path, row: dict) -> dict:
    source_path = corpus_root / row["source"]["path"]
    source = Image.open(source_path).convert("RGB")
    exact = np.load(corpus_root / row["exact_depth"]["path"]).astype(np.float32)
    face = _load_mask(corpus_root / row["selection_mask"]["path"])
    bbox = _padded_box(
        row["render"]["face_bbox_xyxy"],
        source.width,
        source.height,
    )
    x0, y0, x1, y1 = bbox
    crop_shape = (y1 - y0, x1 - x0)
    part = np.zeros(exact.shape, dtype=bool)
    for name in FACE_PART_NAMES:
        part |= _load_mask(corpus_root / row["exact_face_parts"][name]["path"])
    return {
        "source_crop": source.crop(bbox),
        "target_crop": _near_high_target(exact, face)[y0:y1, x0:x1],
        "face_crop": face[y0:y1, x0:x1],
        "part_crop": part[y0:y1, x0:x1],
        "crop_shape": crop_shape,
        "bbox": bbox,
    }


def _prepare_training_items(corpus_root: Path, rows: list[dict], device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    snapshot = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.neck.parameters():
        parameter.requires_grad_(False)
    cached = []
    peak_vram = 0.0
    for index, row in enumerate(rows):
        prepared = _crop_row(corpus_root, row)
        pixel_values = processor(
            images=prepared["source_crop"],
            return_tensors="pt",
        )["pixel_values"].to(device=device, dtype=dtype)
        patch_height = pixel_values.shape[-2] // model.config.patch_size
        patch_width = pixel_values.shape[-1] // model.config.patch_size
        with torch.inference_mode():
            backbone = model.backbone.forward_with_filtered_kwargs(pixel_values)
            hidden = model.neck(
                backbone.feature_maps,
                patch_height,
                patch_width,
            )
            baseline = model.head(hidden, patch_height, patch_width)
        output_shape = tuple(int(value) for value in baseline.shape[-2:])
        target = _resize(
            prepared["target_crop"],
            output_shape,
            cv2.INTER_CUBIC,
        )
        face = _resize(
            prepared["face_crop"].astype(np.uint8),
            output_shape,
            cv2.INTER_NEAREST,
        ).astype(bool)
        part = _resize(
            prepared["part_crop"].astype(np.uint8),
            output_shape,
            cv2.INTER_NEAREST,
        ).astype(bool)
        cached.append(
            CachedFace(
                row=row,
                pixel_values=pixel_values[0].detach().cpu().to(torch.float16),
                baseline=baseline[0].detach().cpu().to(torch.float16),
                target=torch.from_numpy(target.astype(np.float32)),
                face_mask=torch.from_numpy(face),
                part_weight=torch.from_numpy(part.astype(np.float32)),
                patch_height=int(patch_height),
                patch_width=int(patch_width),
                bbox=prepared["bbox"],
            )
        )
        if device.startswith("cuda"):
            peak_vram = max(
                peak_vram,
                float(torch.cuda.max_memory_allocated(device) / (1024**3)),
            )
        if (index + 1) % 40 == 0:
            print(f"prepared {index + 1}/{len(rows)} rows", flush=True)
    model.head.to(dtype=torch.float32)
    return cached, model, {
        "snapshot": str(snapshot),
        "snapshot_model_sha256": _sha256(Path(snapshot) / "model.safetensors"),
        "processor_sha256": _sha256(Path(snapshot) / "preprocessor_config.json"),
        "peak_vram_gb": peak_vram,
    }


def _head_prediction(model, item: CachedFace, device: str):
    import torch

    pixel_values = item.pixel_values.to(device=device, dtype=torch.float16)[None]
    with torch.no_grad():
        backbone = model.backbone.forward_with_filtered_kwargs(pixel_values)
        hidden = model.neck(
            backbone.feature_maps,
            item.patch_height,
            item.patch_width,
        )
    feature = hidden[-1].to(dtype=next(model.head.parameters()).dtype)
    return model.head(
        [feature],
        item.patch_height,
        item.patch_width,
    )[0]


def _quality(
    corpus_root: Path,
    item: CachedFace,
    prediction: np.ndarray,
    output_dir: Path,
    label: str,
) -> dict:
    row = item.row
    exact = np.load(corpus_root / row["exact_depth"]["path"])
    candidate = np.full(exact.shape, np.nan, dtype=np.float32)
    x0, y0, x1, y1 = item.bbox
    resized = _resize(
        prediction,
        (y1 - y0, x1 - x0),
        cv2.INTER_CUBIC,
    )
    candidate[y0:y1, x0:x1] = resized
    path = output_dir / label / f"{row['row_id']}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, candidate)
    part_paths = {
        name: corpus_root / row["exact_face_parts"][name]["path"]
        for name in FACE_PART_NAMES
    }
    metrics = _exact_face_depth_quality(
        path,
        corpus_root / row["exact_depth"]["path"],
        corpus_root / row["selection_mask"]["path"],
        expected_scale_sign=-1.0,
        part_mask_paths=part_paths,
    )
    return {
        "row_id": row["row_id"],
        "identity_group": row["identity_group"],
        "expression": row["expression"],
        "face_height_pixels": row["render"]["face_bbox_height_pixels"],
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": metrics["named_part_shape"]["failed_parts"],
        "affine_failed_parts": metrics["named_part_affine_mm"]["failed_parts"],
    }


def _summarize(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "combined_part_failures": sum(
            len(row["shape_failed_parts"]) + len(row["affine_failed_parts"])
            for row in rows
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


def _evaluate(
    corpus_root: Path,
    items: list[CachedFace],
    model,
    device: str,
    output_dir: Path,
    label: str,
) -> dict:
    import torch

    model.head.eval()
    records = []
    with torch.inference_mode():
        for item in items:
            prediction = _head_prediction(model, item, device).float().cpu().numpy()
            records.append(_quality(corpus_root, item, prediction, output_dir, label))
    return _summarize(records)


def _evaluate_baseline(
    corpus_root: Path,
    items: list[CachedFace],
    output_dir: Path,
    label: str,
) -> dict:
    records = [
        _quality(
            corpus_root,
            item,
            item.baseline.float().numpy(),
            output_dir,
            label,
        )
        for item in items
    ]
    return _summarize(records)


def _predict_items(items: list[CachedFace], model, device: str) -> list[np.ndarray]:
    import torch

    model.head.eval()
    with torch.inference_mode():
        return [
            _head_prediction(model, item, device).float().cpu().numpy()
            for item in items
        ]


def _evaluate_blend(
    corpus_root: Path,
    items: list[CachedFace],
    trained_predictions: list[np.ndarray],
    alpha: float,
    output_dir: Path,
    label: str,
) -> dict:
    records = []
    for item, trained in zip(items, trained_predictions, strict=True):
        baseline = item.baseline.float().numpy()
        prediction = (
            (1.0 - float(alpha)) * baseline + float(alpha) * trained
        ).astype(np.float32)
        records.append(
            _quality(
                corpus_root,
                item,
                prediction,
                output_dir,
                label,
            )
        )
    return _summarize(records)


def _strictly_improves(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["combined_part_failures"] < baseline["combined_part_failures"]
        and candidate["median_shape_correlation"]
        >= baseline["median_shape_correlation"]
        and candidate["median_gradient_correlation"]
        >= baseline["median_gradient_correlation"]
        and candidate["median_normalized_rmse"]
        <= baseline["median_normalized_rmse"]
    )


def train(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 5,
    learning_rate: float = 2e-4,
    device: str = "cuda",
) -> dict:
    import torch

    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((corpus_root / "summary.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    split_counts = {
        name: sum(row["split"] == name for row in rows)
        for name in ("train", "validation", "sealed")
    }
    if split_counts != {"train": 240, "validation": 40, "sealed": 40}:
        raise ValueError(
            "Training requires the complete identity-disjoint 240/40/40 corpus"
        )
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded DAv2 head training run")
    started = time.perf_counter()
    cached, model, model_provenance = _prepare_training_items(
        corpus_root,
        rows,
        device,
    )
    by_split = {
        name: [item for item in cached if item.row["split"] == name]
        for name in split_counts
    }
    baseline_validation = _evaluate_baseline(
        corpus_root,
        by_split["validation"],
        output_dir,
        "baseline_validation",
    )
    baseline_sealed = _evaluate_baseline(
        corpus_root,
        by_split["sealed"],
        output_dir,
        "baseline_sealed",
    )
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-4,
    )
    generator = random.Random(TRAINING_SEED)
    epochs_record = []
    best_state = copy.deepcopy(model.head.state_dict())
    best_rank = (
        baseline_validation["combined_part_failures"],
        baseline_validation["median_normalized_rmse"],
        -baseline_validation["median_shape_correlation"],
    )
    best_epoch = 0
    for epoch in range(1, int(epochs) + 1):
        model.head.train()
        order = list(by_split["train"])
        generator.shuffle(order)
        losses = []
        for item in order:
            prediction = _head_prediction(model, item, device)
            loss, components = _training_loss(prediction, item)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
            optimizer.step()
            losses.append({"total": float(loss.detach()), **components})
        validation = _evaluate(
            corpus_root,
            by_split["validation"],
            model,
            device,
            output_dir,
            f"validation_epoch_{epoch:02d}",
        )
        rank = (
            validation["combined_part_failures"],
            validation["median_normalized_rmse"],
            -validation["median_shape_correlation"],
        )
        if rank < best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = copy.deepcopy(model.head.state_dict())
        epochs_record.append(
            {
                "epoch": epoch,
                "mean_total_loss": float(np.mean([row["total"] for row in losses])),
                "mean_value_loss": float(np.mean([row["value"] for row in losses])),
                "mean_gradient_loss": float(
                    np.mean([row["gradient"] for row in losses])
                ),
                "mean_laplacian_loss": float(
                    np.mean([row["laplacian"] for row in losses])
                ),
                "validation": validation,
            }
        )
        print(
            f"epoch {epoch}: validation failures "
            f"{validation['combined_part_failures']}",
            flush=True,
        )
    model.head.load_state_dict(best_state)
    validation_predictions = _predict_items(
        by_split["validation"],
        model,
        device,
    )
    blend_candidates = []
    for alpha in BLEND_ALPHAS:
        candidate = _evaluate_blend(
            corpus_root,
            by_split["validation"],
            validation_predictions,
            alpha,
            output_dir,
            f"validation_blend_{alpha:g}",
        )
        blend_candidates.append(
            {
                "alpha": float(alpha),
                "eligible": _strictly_improves(candidate, baseline_validation),
                **candidate,
            }
        )
    eligible_blends = [
        candidate for candidate in blend_candidates if candidate["eligible"]
    ]
    selected_blend = (
        min(
            eligible_blends,
            key=lambda candidate: (
                candidate["combined_part_failures"],
                candidate["median_normalized_rmse"],
                -candidate["median_shape_correlation"],
            ),
        )
        if eligible_blends
        else blend_candidates[0]
    )
    sealed_predictions = _predict_items(by_split["sealed"], model, device)
    sealed = _evaluate_blend(
        corpus_root,
        by_split["sealed"],
        sealed_predictions,
        selected_blend["alpha"],
        output_dir,
        "sealed_selected_blend",
    )
    checkpoint_path = output_dir / "depth_anything_v2_face_head.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "head_state_dict": {
                name: value.detach().cpu() for name, value in best_state.items()
            },
            "training_seed": TRAINING_SEED,
            "selected_epoch": best_epoch,
            "blend_alpha": selected_blend["alpha"],
        },
        checkpoint_path,
    )
    evidence = {
        "schema_version": 1,
        "method": "frozen_dinov2_and_neck_train_depth_anything_v2_head",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            **model_provenance,
        },
        "corpus": {
            "summary_sha256": _sha256(corpus_root / "summary.json"),
            "asset_manifest_sha256": summary["asset_manifest_sha256"],
            "split_counts": split_counts,
        },
        "training": {
            "seed": TRAINING_SEED,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "part_weight": PART_WEIGHT,
            "background_distillation_weight": BACKGROUND_DISTILLATION_WEIGHT,
            "gradient_loss_weight": GRADIENT_LOSS_WEIGHT,
            "laplacian_loss_weight": LAPLACIAN_LOSS_WEIGHT,
            "selected_epoch": best_epoch,
            "epochs_record": epochs_record,
        },
        "blend_selection": {
            "selection_scope": "identity-disjoint-validation-only",
            "candidates": blend_candidates,
            "selected": selected_blend,
        },
        "baseline_validation": baseline_validation,
        "baseline_sealed": baseline_sealed,
        "selected_sealed": sealed,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "decision": {
            "validation_improved": bool(best_epoch > 0),
            "validation_blend_eligible": bool(selected_blend["eligible"]),
            "sealed_failure_delta": (
                sealed["combined_part_failures"]
                - baseline_sealed["combined_part_failures"]
            ),
            "eligible_for_exact_production_gate": bool(
                best_epoch > 0
                and selected_blend["eligible"]
                and _strictly_improves(sealed, baseline_sealed)
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name() if device.startswith("cuda") else device,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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
    if not evidence["decision"]["eligible_for_exact_production_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
