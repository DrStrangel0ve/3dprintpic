"""Train a camera-aligned face-surface residual adapter on the CC0 corpus."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
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
from backend.benchmark.train_face_depth_head import (
    FACE_PADDING_RATIO,
    MODEL_ID,
    MODEL_REVISION,
    _near_high_target,
    _padded_box,
)


CACHE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
TRAINING_SEED = 20260716
DEFAULT_NETWORK_SIZE = 160
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 24
DEFAULT_LEARNING_RATE = 1e-3
MAX_NORMALIZED_RESIDUAL = 0.75
FACE_FEATHER_RATIO = 0.12
PART_WEIGHT = 2.0
VALUE_LOSS_WEIGHT = 1.0
GRADIENT_LOSS_WEIGHT = 0.80
MULTISCALE_LOSS_WEIGHT = 0.45
NORMAL_LOSS_WEIGHT = 0.35
RESIDUAL_REGULARIZATION_WEIGHT = 0.01
BLEND_ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
MINIMUM_GLOBAL_GRADIENT_RETENTION = 0.9975
CRITICAL_PART_CHECKS = (
    "raw_gradient_correlation",
    "gradient_correlation",
    "slope_retention",
    "curvature_retention",
)


@dataclass
class CachedSurface:
    row: dict
    rgb: np.ndarray
    baseline: np.ndarray
    target: np.ndarray
    face: np.ndarray
    feather: np.ndarray
    parts: np.ndarray
    bbox: tuple[int, int, int, int]
    source_shape: tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        return np.asarray(loaded.convert("L")) >= 128


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


def _face_feather(face: np.ndarray, ratio: float = FACE_FEATHER_RATIO) -> np.ndarray:
    binary = np.asarray(face, dtype=bool).astype(np.uint8)
    if not np.any(binary):
        raise ValueError("Face mask is empty")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    rows, columns = np.nonzero(binary)
    face_size = max(
        1.0,
        float(
            min(
                columns.max() - columns.min() + 1,
                rows.max() - rows.min() + 1,
            )
        ),
    )
    feather_px = max(2.0, face_size * max(float(ratio), 0.02))
    weight = np.clip((distance - 1.0) / feather_px, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    return (weight * binary).astype(np.float32)


def _infer_variable_crop_depths(
    processor,
    model,
    images: list[Image.Image],
    *,
    device: str,
    dtype,
):
    import torch

    grouped: dict[tuple[int, ...], list[tuple[int, torch.Tensor]]] = {}
    for index, image in enumerate(images):
        pixel_values = processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise ValueError(
                "Face crop processor must emit one [1,C,H,W] tensor, "
                f"got {tuple(pixel_values.shape)}"
            )
        grouped.setdefault(tuple(pixel_values.shape[1:]), []).append(
            (index, pixel_values)
        )

    predictions: list[torch.Tensor | None] = [None] * len(images)
    with torch.inference_mode():
        for group in grouped.values():
            pixel_values = torch.cat(
                [values for _, values in group],
                dim=0,
            ).to(device=device, dtype=dtype)
            predicted = model(pixel_values=pixel_values).predicted_depth
            if predicted.ndim != 3 or predicted.shape[0] != len(group):
                raise ValueError(
                    "Face depth model must emit [B,H,W], "
                    f"got {tuple(predicted.shape)}"
                )
            for batch_index, (source_index, _values) in enumerate(group):
                predictions[source_index] = predicted[batch_index]
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("Face crop inference did not return every prediction")
    return predictions


def _cache_matches(
    cache_root: Path,
    *,
    corpus_summary_sha256: str,
    network_size: int,
    row_count: int,
) -> bool:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        manifest.get("schema_version") == CACHE_SCHEMA_VERSION
        and manifest.get("corpus_summary_sha256") == corpus_summary_sha256
        and manifest.get("model_id") == MODEL_ID
        and manifest.get("model_revision") == MODEL_REVISION
        and manifest.get("network_size") == int(network_size)
        and manifest.get("row_count") == int(row_count)
        and all(
            (cache_root / "rows" / f"{row_id}.npz").exists()
            for row_id in manifest.get("row_ids", [])
        )
        and len(manifest.get("row_ids", [])) == int(row_count)
    )


def prepare_cache(
    corpus_root: str | Path,
    cache_root: str | Path,
    *,
    device: str = "cuda",
    network_size: int = DEFAULT_NETWORK_SIZE,
    inference_batch_size: int = 4,
    force: bool = False,
) -> dict:
    """Cache pinned DAv2 crops and exact near-high supervision."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    summary_path = corpus_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["rows"]
    summary_sha = _sha256(summary_path)
    if not force and _cache_matches(
        cache_root,
        corpus_summary_sha256=summary_sha,
        network_size=network_size,
        row_count=len(rows),
    ):
        return json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))

    cache_root.mkdir(parents=True, exist_ok=True)
    row_root = cache_root / "rows"
    row_root.mkdir(parents=True, exist_ok=True)
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
        use_fast=False,
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    prepared_rows = []
    started = time.perf_counter()
    for start in range(0, len(rows), max(1, int(inference_batch_size))):
        batch_rows = rows[start : start + max(1, int(inference_batch_size))]
        batch = []
        images = []
        for row in batch_rows:
            source_path = corpus_root / row["source"]["path"]
            source = Image.open(source_path).convert("RGB")
            exact = np.load(corpus_root / row["exact_depth"]["path"]).astype(
                np.float32
            )
            face = _load_mask(corpus_root / row["selection_mask"]["path"])
            if exact.shape != face.shape or exact.shape != (
                source.height,
                source.width,
            ):
                raise ValueError(f"Corpus row {row['row_id']} has mismatched arrays")
            bbox = _padded_box(
                row["render"]["face_bbox_xyxy"],
                source.width,
                source.height,
                FACE_PADDING_RATIO,
            )
            x0, y0, x1, y1 = bbox
            crop = source.crop(bbox)
            target = _near_high_target(exact, face)[y0:y1, x0:x1]
            face_crop = face[y0:y1, x0:x1]
            parts = []
            for name in FACE_PART_NAMES:
                part = _load_mask(
                    corpus_root / row["exact_face_parts"][name]["path"]
                )
                parts.append(part[y0:y1, x0:x1])
            images.append(crop)
            batch.append(
                {
                    "row": row,
                    "crop": crop,
                    "target": target,
                    "face": face_crop,
                    "parts": np.stack(parts),
                    "bbox": bbox,
                    "source_shape": (source.height, source.width),
                }
            )

        predicted = _infer_variable_crop_depths(
            processor,
            model,
            images,
            device=device,
            dtype=dtype,
        )
        for item_index, item in enumerate(batch):
            crop = item["crop"]
            native = torch.nn.functional.interpolate(
                predicted[item_index][None, None],
                size=(crop.height, crop.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
            baseline_native = _minmax_normalize(native.float().cpu().numpy())
            network_shape = (int(network_size), int(network_size))
            rgb = _resize(
                np.asarray(crop, dtype=np.uint8),
                network_shape,
                cv2.INTER_AREA,
            )
            baseline = _resize(
                baseline_native,
                network_shape,
                cv2.INTER_CUBIC,
            ).astype(np.float32)
            target = _resize(
                item["target"],
                network_shape,
                cv2.INTER_CUBIC,
            ).astype(np.float32)
            face = _resize(
                item["face"].astype(np.uint8),
                network_shape,
                cv2.INTER_NEAREST,
            ).astype(bool)
            parts = np.stack(
                [
                    _resize(
                        part.astype(np.uint8),
                        network_shape,
                        cv2.INTER_NEAREST,
                    ).astype(bool)
                    for part in item["parts"]
                ]
            )
            if np.count_nonzero(face) < 64:
                raise ValueError(
                    f"Corpus row {item['row']['row_id']} has too little face support"
                )
            feather = _face_feather(face)
            row_path = row_root / f"{item['row']['row_id']}.npz"
            np.savez_compressed(
                row_path,
                rgb=rgb,
                baseline=baseline.astype(np.float16),
                target=target.astype(np.float16),
                face=face.astype(np.uint8),
                feather=feather.astype(np.float16),
                parts=parts.astype(np.uint8),
                bbox=np.asarray(item["bbox"], dtype=np.int32),
                source_shape=np.asarray(item["source_shape"], dtype=np.int32),
            )
            prepared_rows.append(item["row"]["row_id"])
        print(
            f"cached {min(start + len(batch_rows), len(rows))}/{len(rows)} rows",
            flush=True,
        )

    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "corpus_summary_sha256": summary_sha,
        "corpus_asset_manifest_sha256": summary.get("asset_manifest_sha256"),
        "privacy": summary.get("privacy"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": _sha256(snapshot / "model.safetensors"),
        "processor_sha256": _sha256(snapshot / "preprocessor_config.json"),
        "network_size": int(network_size),
        "row_count": len(rows),
        "row_ids": prepared_rows,
        "inference_batch_size": int(inference_batch_size),
        "peak_vram_gb": peak_vram,
        "runtime_seconds": time.perf_counter() - started,
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def load_cached_surfaces(
    corpus_root: str | Path,
    cache_root: str | Path,
) -> tuple[list[CachedSurface], dict, dict]:
    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    summary = json.loads((corpus_root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("corpus_summary_sha256") != _sha256(
        corpus_root / "summary.json"
    ):
        raise ValueError("Face-surface cache does not match the corpus summary")
    items = []
    for row in summary["rows"]:
        row_path = cache_root / "rows" / f"{row['row_id']}.npz"
        with np.load(row_path) as cached:
            items.append(
                CachedSurface(
                    row=row,
                    rgb=cached["rgb"].astype(np.uint8),
                    baseline=cached["baseline"].astype(np.float32),
                    target=cached["target"].astype(np.float32),
                    face=cached["face"].astype(bool),
                    feather=cached["feather"].astype(np.float32),
                    parts=cached["parts"].astype(bool),
                    bbox=tuple(int(value) for value in cached["bbox"]),
                    source_shape=tuple(
                        int(value) for value in cached["source_shape"]
                    ),
                )
            )
    return items, summary, manifest


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_block(input_channels: int, output_channels: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.SiLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.SiLU(),
    )


def build_surface_adapter(input_channels: int = 7):
    """Build a compact high-resolution plus global-context residual network."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class FaceSurfaceAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.high_resolution = _conv_block(input_channels, 32)
            self.encoder2 = _conv_block(32, 48)
            self.encoder3 = _conv_block(48, 64)
            self.context = nn.Sequential(
                _conv_block(64, 96),
                nn.Conv2d(96, 96, 3, padding=2, dilation=2),
                nn.GroupNorm(_group_count(96), 96),
                nn.SiLU(),
            )
            self.decoder3 = _conv_block(96 + 64, 64)
            self.decoder2 = _conv_block(64 + 48, 48)
            self.decoder1 = _conv_block(48 + 32, 32)
            self.output = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, values):
            high = self.high_resolution(values)
            level2 = self.encoder2(functional.avg_pool2d(high, 2))
            level3 = self.encoder3(functional.avg_pool2d(level2, 2))
            context = self.context(functional.avg_pool2d(level3, 2))
            decoded3 = functional.interpolate(
                context,
                size=level3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded3 = self.decoder3(torch.cat((decoded3, level3), dim=1))
            decoded2 = functional.interpolate(
                decoded3,
                size=level2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded2 = self.decoder2(torch.cat((decoded2, level2), dim=1))
            decoded1 = functional.interpolate(
                decoded2,
                size=high.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded1 = self.decoder1(torch.cat((decoded1, high), dim=1))
            return (
                torch.tanh(self.output(decoded1)) * MAX_NORMALIZED_RESIDUAL
            )

    return FaceSurfaceAdapter()


def _coordinate_channels(size: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx, yy))


def _stack(items: list[CachedSurface]) -> dict:
    import torch

    if not items:
        raise ValueError("At least one cached surface is required")
    size = items[0].baseline.shape[0]
    coordinates = _coordinate_channels(size)
    inputs = []
    for item in items:
        rgb = item.rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        inputs.append(
            np.concatenate(
                (
                    rgb,
                    item.baseline[None],
                    item.face.astype(np.float32)[None],
                    coordinates,
                ),
                axis=0,
            )
        )
    return {
        "inputs": torch.from_numpy(np.stack(inputs)).float(),
        "baseline": torch.from_numpy(
            np.stack([item.baseline for item in items])[:, None]
        ).float(),
        "target": torch.from_numpy(
            np.stack([item.target for item in items])[:, None]
        ).float(),
        "face": torch.from_numpy(
            np.stack([item.face for item in items])[:, None]
        ).float(),
        "feather": torch.from_numpy(
            np.stack([item.feather for item in items])[:, None]
        ).float(),
        "parts": torch.from_numpy(
            np.stack([np.any(item.parts, axis=0) for item in items])[:, None]
        ).float(),
        "sample_weight": torch.tensor(
            [
                float(
                    np.clip(
                        90.0
                        / max(
                            float(item.row["render"]["face_bbox_height_pixels"]),
                            1.0,
                        ),
                        1.0,
                        1.5,
                    )
                )
                for item in items
            ],
            dtype=torch.float32,
        )[:, None, None, None],
    }


def positive_affine_fit(prediction, target, mask):
    """Fit one positive affine transform independently for each batch item."""
    import torch

    weights = mask.to(dtype=prediction.dtype)
    count = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    prediction_mean = (prediction * weights).sum(
        dim=(-2, -1), keepdim=True
    ) / count
    target_mean = (target * weights).sum(dim=(-2, -1), keepdim=True) / count
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    variance = (
        centered_prediction.square() * weights
    ).sum(dim=(-2, -1), keepdim=True) / count
    covariance = (
        centered_prediction * centered_target * weights
    ).sum(dim=(-2, -1), keepdim=True) / count
    scale = torch.clamp(
        covariance / variance.clamp_min(1e-6),
        0.05,
        20.0,
    )
    shift = target_mean - scale * prediction_mean
    return scale * prediction + shift, scale, shift


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _axis_gradient_loss(prediction, target, mask):
    valid_x = mask[..., :, 1:] * mask[..., :, :-1]
    valid_y = mask[..., 1:, :] * mask[..., :-1, :]
    difference_x = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (
        target[..., :, 1:] - target[..., :, :-1]
    )
    difference_y = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    return _weighted_mean(difference_x.abs(), valid_x) + _weighted_mean(
        difference_y.abs(),
        valid_y,
    )


def _surface_normals(values):
    import torch

    gradient_x = values[..., :, 1:] - values[..., :, :-1]
    gradient_y = values[..., 1:, :] - values[..., :-1, :]
    gradient_x = gradient_x[..., :-1, :]
    gradient_y = gradient_y[..., :, :-1]
    normal = torch.cat(
        (
            -gradient_x,
            -gradient_y,
            torch.full_like(gradient_x, 0.05),
        ),
        dim=1,
    )
    return torch.nn.functional.normalize(normal, dim=1)


def surface_training_loss(
    residual,
    baseline,
    target,
    face,
    feather,
    parts,
    sample_weight,
):
    import torch.nn.functional as functional

    prediction = baseline + residual * feather
    fitted, scale, shift = positive_affine_fit(prediction, target, face)
    weights = face * (1.0 + PART_WEIGHT * parts) * sample_weight
    value_loss = _weighted_mean((fitted - target).abs(), weights)
    gradient_loss = _axis_gradient_loss(fitted, target, weights)

    multiscale_loss = fitted.new_tensor(0.0)
    for factor in (2, 4):
        pooled_prediction = functional.avg_pool2d(fitted, factor)
        pooled_target = functional.avg_pool2d(target, factor)
        pooled_weights = functional.avg_pool2d(weights, factor)
        multiscale_loss = multiscale_loss + _weighted_mean(
            (pooled_prediction - pooled_target).abs(),
            pooled_weights,
        )

    valid_normal = (
        face[..., 1:, 1:]
        * face[..., 1:, :-1]
        * face[..., :-1, 1:]
        * face[..., :-1, :-1]
        * sample_weight
    )
    predicted_normal = _surface_normals(fitted)
    target_normal = _surface_normals(target)
    normal_cosine = (predicted_normal * target_normal).sum(
        dim=1,
        keepdim=True,
    )
    normal_loss = _weighted_mean(1.0 - normal_cosine, valid_normal)
    residual_regularization = _weighted_mean(
        residual.abs(),
        face * sample_weight,
    )
    total = (
        VALUE_LOSS_WEIGHT * value_loss
        + GRADIENT_LOSS_WEIGHT * gradient_loss
        + MULTISCALE_LOSS_WEIGHT * multiscale_loss
        + NORMAL_LOSS_WEIGHT * normal_loss
        + RESIDUAL_REGULARIZATION_WEIGHT * residual_regularization
    )
    return total, {
        "value": float(value_loss.detach()),
        "gradient": float(gradient_loss.detach()),
        "multiscale": float(multiscale_loss.detach()),
        "normal": float(normal_loss.detach()),
        "residual_regularization": float(residual_regularization.detach()),
        "fit_scale_median": float(scale.detach().median()),
        "fit_shift_median": float(shift.detach().median()),
    }


def _loss_for_tensors(model, tensors: dict, device: str) -> tuple[float, dict]:
    import torch

    model.eval()
    with torch.inference_mode():
        values = {
            key: tensor.to(device)
            for key, tensor in tensors.items()
        }
        residual = model(values["inputs"])
        loss, details = surface_training_loss(
            residual,
            values["baseline"],
            values["target"],
            values["face"],
            values["feather"],
            values["parts"],
            values["sample_weight"],
        )
    return float(loss), details


def train_adapter(
    train_items: list[CachedSurface],
    validation_items: list[CachedSurface],
    *,
    device: str = "cuda",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = TRAINING_SEED,
) -> tuple[object, dict]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_tensors = _stack(train_items)
    validation_tensors = _stack(validation_items)
    model = build_surface_adapter(train_tensors["inputs"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = math.inf
    history = []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(
            len(train_items),
            generator=generator,
        ).tolist()
        batch_losses = []
        for start in range(0, len(order), max(1, int(batch_size))):
            indices = order[start : start + max(1, int(batch_size))]
            values = {
                key: tensor[indices].to(device)
                for key, tensor in train_tensors.items()
            }
            batch_input = values["inputs"].clone()
            if ((epoch + start // max(1, int(batch_size))) % 2) == 0:
                for key in (
                    "inputs",
                    "baseline",
                    "target",
                    "face",
                    "feather",
                    "parts",
                ):
                    values[key] = torch.flip(values[key], dims=[-1])
                batch_input = values["inputs"].clone()
            color_scale = (
                0.85
                + 0.30
                * torch.rand(
                    (len(indices), 3, 1, 1),
                    generator=generator,
                ).to(device)
            )
            color_shift = (
                -0.04
                + 0.08
                * torch.rand(
                    (len(indices), 3, 1, 1),
                    generator=generator,
                ).to(device)
            )
            batch_input[:, :3] = torch.clamp(
                batch_input[:, :3] * color_scale + color_shift,
                0.0,
                1.0,
            )
            residual = model(batch_input)
            loss, _ = surface_training_loss(
                residual,
                values["baseline"],
                values["target"],
                values["face"],
                values["feather"],
                values["parts"],
                values["sample_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        validation_loss, validation_details = _loss_for_tensors(
            model,
            validation_tensors,
            device,
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "validation_details": validation_details,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    return model.eval(), {
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gb": peak_vram,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def predict_residuals(
    model,
    items: list[CachedSurface],
    *,
    device: str = "cuda",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[np.ndarray]:
    import torch

    tensors = _stack(items)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            values = tensors["inputs"][
                start : start + max(1, int(batch_size))
            ].to(device)
            predictions.extend(
                model(values)[:, 0].float().cpu().numpy()
            )
    return predictions


def remove_affine_residual(
    residual: np.ndarray,
    baseline: np.ndarray,
    face: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Remove the scale/offset component that the training objective cannot identify."""
    residual = np.asarray(residual, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    valid = (
        np.asarray(face, dtype=bool)
        & np.isfinite(residual)
        & np.isfinite(baseline)
    )
    if np.count_nonzero(valid) < 24:
        raise ValueError("Affine residual removal needs at least 24 face samples")
    design = np.column_stack(
        (
            baseline[valid].astype(np.float64),
            np.ones(np.count_nonzero(valid), dtype=np.float64),
        )
    )
    scale, offset = np.linalg.lstsq(
        design,
        residual[valid].astype(np.float64),
        rcond=None,
    )[0]
    cleaned = residual - (float(scale) * baseline + float(offset))
    return cleaned.astype(np.float32), {
        "method": "remove-affine-component-relative-to-local-depth",
        "samples": int(np.count_nonzero(valid)),
        "scale": float(scale),
        "offset": float(offset),
    }


def _candidate_full_surface(
    item: CachedSurface,
    residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    network_candidate = item.baseline + (
        float(alpha) * np.asarray(residual, dtype=np.float32) * item.feather
    )
    x0, y0, x1, y1 = item.bbox
    crop_shape = (y1 - y0, x1 - x0)
    native = _resize(
        network_candidate,
        crop_shape,
        cv2.INTER_CUBIC,
    ).astype(np.float32)
    candidate = np.zeros(item.source_shape, dtype=np.float32)
    candidate[y0:y1, x0:x1] = native
    return candidate


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


def evaluate_exact_surfaces(
    corpus_root: str | Path,
    items: list[CachedSurface],
    residuals: list[np.ndarray],
    *,
    alpha: float,
    output_dir: str | Path,
) -> dict:
    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, residual in zip(items, residuals, strict=True):
        candidate = _candidate_full_surface(item, residual, alpha)
        candidate_path = output_dir / f"{item.row['row_id']}.npy"
        np.save(candidate_path, candidate)
        part_paths = {
            name: corpus_root / item.row["exact_face_parts"][name]["path"]
            for name in FACE_PART_NAMES
        }
        metrics = _exact_face_depth_quality(
            candidate_path,
            corpus_root / item.row["exact_depth"]["path"],
            corpus_root / item.row["selection_mask"]["path"],
            expected_scale_sign=-1.0,
            part_mask_paths=part_paths,
        )
        shape_failed = metrics["named_part_shape"]["failed_parts"]
        affine_failed = metrics["named_part_affine_mm"]["failed_parts"]
        check_failures = Counter()
        for part in metrics["named_part_shape"]["parts"]:
            check_failures.update(
                key
                for key, passed in part.get("checks", {}).items()
                if key != "passed" and not passed
            )
        rows.append(
            {
                "row_id": item.row["row_id"],
                "identity_group": item.row["identity_group"],
                "expression": item.row["expression"],
                "face_height_pixels": item.row["render"][
                    "face_bbox_height_pixels"
                ],
                "shape_correlation": metrics["shape_correlation"],
                "gradient_correlation": metrics["gradient_correlation"],
                "normalized_rmse": metrics["normalized_rmse"],
                "shape_failed_parts": shape_failed,
                "affine_failed_parts": affine_failed,
                "combined_part_failures": len(shape_failed)
                + len(affine_failed),
                "shape_check_failures": dict(sorted(check_failures.items())),
            }
        )
    return _summary(rows)


def _strictly_improves(candidate: dict, baseline: dict) -> bool:
    candidate_checks = candidate.get("shape_check_failure_counts", {})
    baseline_checks = baseline.get("shape_check_failure_counts", {})
    gradient_floor = (
        baseline["median_gradient_correlation"]
        * MINIMUM_GLOBAL_GRADIENT_RETENTION
    )
    return bool(
        candidate["combined_part_failures"]
        < baseline["combined_part_failures"]
        and candidate["median_shape_correlation"]
        >= baseline["median_shape_correlation"] - 1e-6
        and candidate["median_gradient_correlation"]
        >= gradient_floor
        and candidate["median_normalized_rmse"]
        <= baseline["median_normalized_rmse"] + 1e-6
        and all(
            int(candidate_checks.get(name, 0))
            <= int(baseline_checks.get(name, 0))
            for name in CRITICAL_PART_CHECKS
        )
    )


def _per_row_non_regression(candidate: dict, baseline: dict) -> float:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    passed = sum(
        row["combined_part_failures"]
        <= baseline_by_id[row["row_id"]]["combined_part_failures"]
        for row in candidate["rows"]
    )
    return float(passed / max(len(candidate["rows"]), 1))


def train(
    corpus_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    network_size: int = DEFAULT_NETWORK_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    force_cache: bool = False,
    inference_batch_size: int = 4,
) -> dict:
    import torch

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = prepare_cache(
        corpus_root,
        cache_root,
        device=device,
        network_size=network_size,
        inference_batch_size=inference_batch_size,
        force=force_cache,
    )
    items, corpus_summary, _ = load_cached_surfaces(corpus_root, cache_root)
    train_items = [item for item in items if item.row["split"] == "train"]
    validation_items = [
        item for item in items if item.row["split"] == "validation"
    ]
    sealed_items = [item for item in items if item.row["split"] == "sealed"]
    if not train_items or not validation_items or not sealed_items:
        raise ValueError("Training, validation, and sealed splits are required")

    model, training = train_adapter(
        train_items,
        validation_items,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    validation_residuals = predict_residuals(
        model,
        validation_items,
        device=device,
        batch_size=batch_size,
    )
    validation_residuals = [
        remove_affine_residual(
            residual,
            item.baseline,
            item.face,
        )[0]
        for item, residual in zip(
            validation_items,
            validation_residuals,
            strict=True,
        )
    ]
    zero_residuals = [
        np.zeros_like(item.baseline, dtype=np.float32)
        for item in validation_items
    ]
    baseline_validation = evaluate_exact_surfaces(
        corpus_root,
        validation_items,
        zero_residuals,
        alpha=0.0,
        output_dir=output_dir / "validation_baseline",
    )
    blend_candidates = []
    for alpha in BLEND_ALPHAS:
        candidate = evaluate_exact_surfaces(
            corpus_root,
            validation_items,
            validation_residuals,
            alpha=alpha,
            output_dir=output_dir / "validation_scratch",
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
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "failures": candidate["combined_part_failures"],
                    "shape": candidate["median_shape_correlation"],
                    "gradient": candidate["median_gradient_correlation"],
                    "rmse": candidate["median_normalized_rmse"],
                    "eligible": candidate["eligible"],
                }
            ),
            flush=True,
        )
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

    sealed_residuals = predict_residuals(
        model,
        sealed_items,
        device=device,
        batch_size=batch_size,
    )
    sealed_residuals = [
        remove_affine_residual(
            residual,
            item.baseline,
            item.face,
        )[0]
        for item, residual in zip(
            sealed_items,
            sealed_residuals,
            strict=True,
        )
    ]
    zero_sealed = [
        np.zeros_like(item.baseline, dtype=np.float32) for item in sealed_items
    ]
    baseline_sealed = evaluate_exact_surfaces(
        corpus_root,
        sealed_items,
        zero_sealed,
        alpha=0.0,
        output_dir=output_dir / "sealed_baseline",
    )
    sealed = evaluate_exact_surfaces(
        corpus_root,
        sealed_items,
        sealed_residuals,
        alpha=selected["alpha"],
        output_dir=output_dir / "sealed_selected",
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
    decision["eligible_for_exact_production_replay"] = bool(all(decision.values()))

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": "cc0_camera_aligned_face_surface_residual",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "network_size": int(network_size),
        "input_channels": 7,
        "max_normalized_residual": MAX_NORMALIZED_RESIDUAL,
        "face_feather_ratio": FACE_FEATHER_RATIO,
        "selected_alpha": float(selected["alpha"]),
        "selected_epoch": int(training["best_epoch"]),
        "residual_postprocess": "remove-affine-component-relative-to-local-depth",
        "corpus_summary_sha256": cache_manifest["corpus_summary_sha256"],
        "state_dict": model.state_dict(),
    }
    checkpoint_path = output_dir / "face_surface_adapter.pt"
    torch.save(checkpoint, checkpoint_path)
    evidence = {
        "schema_version": 1,
        "method": "cc0_camera_aligned_face_surface_residual",
        "research_basis": {
            "representation": "dual-scale CNN residual over normalized local depth",
            "alignment_target": "positive-affine camera-aligned near-high face surface",
            "losses": [
                "part-weighted value",
                "raw gradient",
                "multi-scale pooled shape",
                "surface-normal cosine",
                "bounded residual regularization",
            ],
            "residual_postprocess": (
                "remove the affine component relative to local depth"
            ),
            "minimum_global_gradient_retention": (
                MINIMUM_GLOBAL_GRADIENT_RETENTION
            ),
            "critical_part_checks": list(CRITICAL_PART_CHECKS),
        },
        "corpus": {
            "summary_sha256": cache_manifest["corpus_summary_sha256"],
            "asset_manifest_sha256": corpus_summary.get(
                "asset_manifest_sha256"
            ),
            "privacy": corpus_summary.get("privacy"),
            "train_rows": len(train_items),
            "validation_rows": len(validation_items),
            "sealed_rows": len(sealed_items),
            "train_identities": sorted(
                {item.row["identity_group"] for item in train_items}
            ),
            "validation_identities": sorted(
                {item.row["identity_group"] for item in validation_items}
            ),
            "sealed_identities": sorted(
                {item.row["identity_group"] for item in sealed_items}
            ),
        },
        "cache": cache_manifest,
        "training": training,
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
        },
        "device": (
            torch.cuda.get_device_name(device)
            if str(device).startswith("cuda")
            else "cpu"
        ),
        "decision": decision,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--network-size", type=int, default=DEFAULT_NETWORK_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    evidence = train(
        args.corpus_root,
        args.cache_root,
        args.output_dir,
        device=args.device,
        network_size=args.network_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_cache=args.force_cache,
        inference_batch_size=args.inference_batch_size,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_exact_production_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
