"""Train a matched DAv2-Small spatial-pyramid face-depth challenger."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
import platform
import random
from pathlib import Path
import subprocess
import time
from typing import Sequence


DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_configured_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _configured_cublas_workspace not in {
    None,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
}:
    raise RuntimeError(
        "HSRD spatial-pyramid training requires "
        f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG} before import"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.c3i_synface_corpus import verified_corpus_asset
from backend.benchmark.face_depth_rectified_flow import camera_rays_from_intrinsics
from backend.benchmark.train_face_depth_rectified_flow import (
    _attachment_boundary,
    _nearest_face_fill,
    _robust_unit_depth,
    _source_resolution_candidate,
)
from backend.benchmark.face_depth_spatial_pyramid import (
    SpatialPyramidConfig,
    SpatialPyramidLossWeights,
    build_spatial_pyramid,
    matched_deepest_control_features,
    parameter_count,
    spatial_pyramid_loss,
)
from backend.benchmark.hsrd_face_training_corpus import (
    HSRD_LICENSE,
    HSRD_REPOSITORY,
    HSRD_REVISION,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_LICENSE = "Apache-2.0"
MODEL_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
PROCESSOR_SHA256 = "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84"
METHOD = "dav2_small_frozen_spatial_feature_pyramid_camera_z"
CONTROL_METHOD = "dav2_small_frozen_deepest_resized_matched_control_camera_z"
TRAINING_SEED = 20260719
MODEL_SIZE = 192
FACE_PADDING_RATIO = 0.65
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 8e-4
BLEND_ALPHAS = (0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)
LOSS_PROFILES = {
    "balanced": SpatialPyramidLossWeights(),
    "gradient-preserving": SpatialPyramidLossWeights(
        camera_z=0.40,
        gradient=1.00,
        normal=0.25,
        laplacian=0.25,
        attachment=0.25,
        part_non_regression=0.80,
        residual=0.04,
    ),
    "correlation-preserving": SpatialPyramidLossWeights(
        camera_z=0.35,
        gradient=0.50,
        gradient_correlation=0.75,
        normal=0.15,
        laplacian=0.15,
        attachment=0.25,
        part_non_regression=0.80,
        residual=0.04,
    ),
    "correlation-strict": SpatialPyramidLossWeights(
        camera_z=0.25,
        gradient=0.25,
        gradient_correlation=1.50,
        normal=0.10,
        laplacian=0.10,
        attachment=0.25,
        part_non_regression=1.00,
        residual=0.05,
    ),
}


@dataclass
class PreparedSpatialFace:
    row: dict
    features: tuple
    conditioning: object
    coarse: object
    target: object
    support: object
    parts: tuple
    boundary: object
    rays: object
    bbox: tuple[int, int, int, int]
    source_coarse: np.ndarray
    source_support: np.ndarray


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _square_padded_box(
    bbox: Sequence[int],
    width: int,
    height: int,
    padding_ratio: float = FACE_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError("Face crop box must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Face crop box must have positive area")
    side = int(
        round(max(x1 - x0, y1 - y0) * (1.0 + 2.0 * float(padding_ratio)))
    )
    side = min(max(side, 96), int(width), int(height))
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    left = int(round(center_x - 0.5 * side))
    top = int(round(center_y - 0.5 * side))
    left = min(max(left, 0), int(width) - side)
    top = min(max(top, 0), int(height) - side)
    return left, top, left + side, top + side


def _resize(values: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(values),
        (int(size), int(size)),
        interpolation=interpolation,
    )


def _rgb_laplacian_luma(image: Image.Image, size: int = MODEL_SIZE) -> np.ndarray:
    rgb = np.asarray(
        image.convert("RGB").resize((int(size), int(size)), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    rgb /= 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    blur1 = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.0)
    blur2 = cv2.GaussianBlur(luma, (0, 0), sigmaX=2.5)
    blur3 = cv2.GaussianBlur(luma, (0, 0), sigmaX=5.0)
    levels = np.stack((luma - blur1, blur1 - blur2, blur2 - blur3), axis=0)
    levels = np.clip(levels * 4.0, -1.0, 1.0).astype(np.float32)
    if not np.isfinite(levels).all():
        raise ValueError("RGB Laplacian conditioning is non-finite")
    return levels


def _validate_corpus(summary: dict) -> dict:
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("HSRD spatial-pyramid corpus contains no rows")
    identity_splits: dict[str, set[str]] = {}
    for row in rows:
        identity = str(row.get("identity_group", ""))
        split = str(row.get("split", ""))
        if not identity or split not in {"train", "validation", "sealed"}:
            raise ValueError("HSRD spatial-pyramid row metadata is incomplete")
        identity_splits.setdefault(identity, set()).add(split)
    checks = {
        "provider": summary.get("provider") == "hsrd100-lod1-camera-depth",
        "source_revision": summary.get("source_revision") == HSRD_REVISION,
        "license": summary.get("dataset_license") == HSRD_LICENSE,
        "source_gate": summary.get("source_gate_passed") is True,
        "promotion_eligible": summary.get("promotion_eligible") is True,
        "production_training_eligible": summary.get("production_training_eligible")
        is True,
        "identity_disjoint": summary.get("identity_disjoint") is True
        and all(len(splits) == 1 for splits in identity_splits.values()),
        "small_face_heights": {
            int(row["selection_geometry"]["face_bbox_height_pixels"])
            for row in rows
        }
        == {74, 75},
        "turned_views": all(
            abs(float(row["rendering"]["camera_yaw_degrees"])) >= 30.0
            for row in rows
        ),
        "camera_z": summary.get("target_depth")
        == "floating-relative-camera-z",
        "source_geometry_restricted": summary.get(
            "source_geometry_training_and_evaluation_only"
        )
        is True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("HSRD spatial-pyramid corpus gate failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "identity_count": len(identity_splits),
        "split_identities": {
            split: sorted(
                identity
                for identity, splits in identity_splits.items()
                if split in splits
            )
            for split in ("train", "validation", "sealed")
        },
    }


def _validate_corpus_summary_hash(
    summary_path: Path, expected_sha256: str
) -> str:
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            "Expected HSRD corpus summary SHA256 must be 64 lowercase hex characters"
        )
    observed = _sha256(summary_path)
    if observed != expected:
        raise ValueError(
            "HSRD corpus summary SHA256 mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def _nvidia_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Could not record the NVIDIA driver version") from exc
    versions = {
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    }
    if len(versions) != 1:
        raise RuntimeError(
            "HSRD training requires one consistent NVIDIA driver version, observed "
            f"{sorted(versions)}"
        )
    return versions.pop()


def _select_rows(rows: Sequence[dict], limit_per_split: int | None) -> list[dict]:
    if limit_per_split is None:
        selected = list(rows)
    else:
        limit = int(limit_per_split)
        if not 1 <= limit <= 40:
            raise ValueError("Spatial-pyramid smoke limit must be in [1, 40]")
        selected = []
        for split in ("train", "validation", "sealed"):
            selected.extend([row for row in rows if row["split"] == split][:limit])
    if {row["split"] for row in selected} != {"train", "validation", "sealed"}:
        raise ValueError("Spatial-pyramid selection must retain every split")
    return selected


def _load_depth_model(device: str):
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
    observed = {
        "model": _sha256(snapshot / "model.safetensors"),
        "processor": _sha256(snapshot / "preprocessor_config.json"),
    }
    expected = {"model": MODEL_SHA256, "processor": PROCESSOR_SHA256}
    if observed != expected:
        raise ValueError(f"Pinned DAv2 Small snapshot hash mismatch: {observed}")
    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "model_sha256": observed["model"],
        "processor_sha256": observed["processor"],
        "local_files_only": True,
    }


def _prepare_one(
    corpus_root: Path,
    row: dict,
    processor,
    depth_model,
    device: str,
) -> PreparedSpatialFace:
    import torch

    source = Image.open(verified_corpus_asset(corpus_root, row["source"])).convert(
        "RGB"
    )
    exact = np.load(
        verified_corpus_asset(corpus_root, row["exact_camera_depth"])
    ).astype(np.float32)
    full_support = np.asarray(
        Image.open(verified_corpus_asset(corpus_root, row["selection_mask"])).convert(
            "L"
        )
    ) >= 128
    if exact.shape != full_support.shape or exact.shape != (source.height, source.width):
        raise ValueError("HSRD spatial-pyramid source, depth, and support disagree")
    bbox = _square_padded_box(
        row["selection_geometry"]["face_bbox_xyxy"], source.width, source.height
    )
    x0, y0, x1, y1 = bbox
    crop = source.crop(bbox)
    source_support = full_support[y0:y1, x0:x1]
    pixels = processor(images=crop, return_tensors="pt")["pixel_values"].to(
        device=device, dtype=torch.float16
    )
    patch_height = int(pixels.shape[-2] // depth_model.config.patch_size)
    patch_width = int(pixels.shape[-1] // depth_model.config.patch_size)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        backbone = depth_model.backbone.forward_with_filtered_kwargs(pixels)
        reassembled = depth_model.neck.reassemble_stage(
            backbone.feature_maps, patch_height, patch_width
        )
        features = tuple(
            depth_model.neck.convs[index](feature)
            for index, feature in enumerate(reassembled)
        )
        hidden = depth_model.neck.fusion_stage(list(features))
        raw_depth = depth_model.head(hidden, patch_height, patch_width)[0]
    raw_source = cv2.resize(
        raw_depth.float().cpu().numpy(),
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_CUBIC,
    )
    source_coarse = _robust_unit_depth(raw_source, source_support, inverse=True)
    normalized_target = _nearest_face_fill(
        _robust_unit_depth(exact, full_support, inverse=False), full_support
    )
    source_target = normalized_target[y0:y1, x0:x1]

    support = _resize(source_support.astype(np.uint8), MODEL_SIZE, cv2.INTER_NEAREST).astype(
        bool
    )
    boundary = _attachment_boundary(support, width=2)
    coarse = _resize(source_coarse, MODEL_SIZE, cv2.INTER_CUBIC).astype(np.float32)
    target = _resize(source_target, MODEL_SIZE, cv2.INTER_CUBIC).astype(np.float32)
    parts = []
    for name in FACE_PART_NAMES:
        part = np.asarray(
            Image.open(
                verified_corpus_asset(corpus_root, row["exact_face_parts"][name])
            ).convert("L")
        ) >= 128
        part = _resize(part[y0:y1, x0:x1].astype(np.uint8), MODEL_SIZE, cv2.INTER_NEAREST)
        if np.count_nonzero(part) < 4:
            raise ValueError(f"HSRD row {row['row_id']} has a tiny {name} mask")
        parts.append(torch.from_numpy(part.astype(np.float32))[None])
    rays = camera_rays_from_intrinsics(
        MODEL_SIZE,
        MODEL_SIZE,
        row["selection_geometry"]["camera"]["intrinsics"],
        bbox,
        dtype=torch.float32,
    )[0]
    conditioning = np.concatenate(
        (
            coarse[None],
            rays.numpy(),
            support.astype(np.float32)[None],
            boundary[None],
            _rgb_laplacian_luma(crop),
        ),
        axis=0,
    ).astype(np.float32)
    if conditioning.shape != (9, MODEL_SIZE, MODEL_SIZE):
        raise AssertionError("Spatial face conditioning layout changed")
    return PreparedSpatialFace(
        row=row,
        features=tuple(feature[0].detach().cpu().to(torch.float16) for feature in features),
        conditioning=torch.from_numpy(conditioning).to(torch.float16),
        coarse=torch.from_numpy(coarse)[None],
        target=torch.from_numpy(target)[None],
        support=torch.from_numpy(support.astype(np.float32))[None],
        parts=tuple(parts),
        boundary=torch.from_numpy(boundary)[None],
        rays=rays,
        bbox=bbox,
        source_coarse=source_coarse,
        source_support=source_support,
    )


def prepare_rows(corpus_root: Path, rows: Sequence[dict], device: str):
    import torch

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    processor, depth_model, provenance = _load_depth_model(device)
    prepared = []
    try:
        for index, row in enumerate(rows, start=1):
            prepared.append(
                _prepare_one(corpus_root, row, processor, depth_model, device)
            )
            if index % 10 == 0 or index == len(rows):
                print(f"prepared {index}/{len(rows)} HSRD spatial rows", flush=True)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    finally:
        depth_model.to("cpu")
        del depth_model
        torch.cuda.empty_cache()
    return prepared, {
        **provenance,
        "row_count": len(prepared),
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": peak,
        "frozen_feature_shapes": [
            list(values.shape) for values in prepared[0].features
        ],
    }


def _batch(items: Sequence[PreparedSpatialFace], device: str, *, control: bool):
    import torch

    features = tuple(
        torch.stack([item.features[level] for item in items]).to(
            device=device, dtype=torch.float32
        )
        for level in range(4)
    )
    if control:
        features = matched_deepest_control_features(features)

    def stack(name: str):
        return torch.stack([getattr(item, name) for item in items]).to(
            device=device, dtype=torch.float32
        )

    return {
        "features": features,
        "conditioning": stack("conditioning"),
        "coarse": stack("coarse"),
        "target": stack("target"),
        "support": stack("support"),
        "boundary": stack("boundary"),
        "rays": stack("rays"),
        "parts": [
            torch.stack([item.parts[index] for item in items]).to(
                device=device, dtype=torch.float32
            )
            for index in range(len(FACE_PART_NAMES))
        ],
    }


def _loss(model, batch: dict, loss_weights: SpatialPyramidLossWeights):
    residual = model(
        batch["features"],
        batch["conditioning"],
        batch["support"],
        batch["boundary"],
    )
    return spatial_pyramid_loss(
        residual,
        batch["coarse"],
        batch["target"],
        batch["support"],
        batch["parts"],
        batch["boundary"],
        batch["rays"],
        weights=loss_weights,
    )


def _validation_loss(
    model,
    items,
    device: str,
    batch_size: int,
    *,
    control: bool,
    loss_weights: SpatialPyramidLossWeights,
):
    import torch

    model.eval()
    weighted = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            batch = _batch(batch_items, device, control=control)
            loss, _, _ = _loss(model, batch, loss_weights)
            weighted.append((float(loss), len(batch_items)))
    return sum(value * count for value, count in weighted) / sum(
        count for _, count in weighted
    )


def train_decoder(
    train_items,
    validation_items,
    *,
    initial_state: dict,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    control: bool,
    loss_weights: SpatialPyramidLossWeights,
):
    import torch

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    model = build_spatial_pyramid().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = random.Random(TRAINING_SEED)
    best_state = copy.deepcopy(initial_state)
    best_epoch = 0
    best_validation = _validation_loss(
        model,
        validation_items,
        device,
        batch_size,
        control=control,
        loss_weights=loss_weights,
    )
    history = [{"epoch": 0, "validation_loss": best_validation}]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = list(train_items)
        generator.shuffle(order)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            batch = _batch(order[start : start + int(batch_size)], device, control=control)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _loss(model, batch, loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _validation_loss(
            model,
            validation_items,
            device,
            batch_size,
            control=control,
            loss_weights=loss_weights,
        )
        if validation < best_validation - 1e-9:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "mean_training_loss": float(np.mean(losses)),
                "validation_loss": validation,
            }
        )
        lane = "control" if control else "spatial"
        print(f"{lane} epoch {epoch}: validation {validation:.6f}", flush=True)
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
    }


def _predict_residuals(
    model,
    items: Sequence[PreparedSpatialFace],
    device: str,
    batch_size: int,
    *,
    control: bool,
) -> list[np.ndarray]:
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = _batch(items[start : start + batch_size], device, control=control)
            residual = model(
                batch["features"],
                batch["conditioning"],
                batch["support"],
                batch["boundary"],
            )
            predictions.extend(residual[:, 0].float().cpu().numpy())
    return predictions


def _quality(
    corpus_root: Path,
    item: PreparedSpatialFace,
    residual: np.ndarray,
    output_dir: Path,
    label: str,
) -> dict:
    row = item.row
    exact_path = verified_corpus_asset(corpus_root, row["exact_camera_depth"])
    exact = np.load(exact_path)
    full_support = np.asarray(
        Image.open(verified_corpus_asset(corpus_root, row["selection_mask"])).convert(
            "L"
        )
    ) >= 128
    low_support = item.support[0].numpy()
    candidate_crop, crop_background_exact = _source_resolution_candidate(
        item.source_coarse,
        item.source_support,
        residual,
        low_support,
    )
    x0, y0, x1, y1 = item.bbox
    candidate = np.full(exact.shape, np.nan, dtype=np.float32)
    candidate[y0:y1, x0:x1] = candidate_crop
    correction = np.zeros(exact.shape, dtype=np.float32)
    correction[y0:y1, x0:x1] = candidate_crop - item.source_coarse
    background_exact = bool(
        np.array_equal(
            correction[~full_support],
            np.zeros(np.count_nonzero(~full_support), dtype=np.float32),
        )
    )
    candidate_path = output_dir / label / f"{row['row_id']}.npy"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(candidate_path, candidate)
    metrics = _exact_face_depth_quality(
        candidate_path,
        exact_path,
        verified_corpus_asset(corpus_root, row["selection_mask"]),
        expected_scale_sign=1.0,
        part_mask_paths={
            name: verified_corpus_asset(corpus_root, row["exact_face_parts"][name])
            for name in FACE_PART_NAMES
        },
    )
    return {
        "row_id": row["row_id"],
        "identity_group": row["identity_group"],
        "split": row["split"],
        "face_height_pixels": row["selection_geometry"]["face_bbox_height_pixels"],
        "camera_yaw_degrees": row["rendering"]["camera_yaw_degrees"],
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": metrics["named_part_shape"]["failed_parts"],
        "affine_failed_parts": metrics["named_part_affine_mm"]["failed_parts"],
        "named_part_shape": metrics["named_part_shape"],
        "named_part_affine_mm": metrics["named_part_affine_mm"],
        "crop_background_bit_exact": crop_background_exact,
        "full_image_correction_background_bit_exact": background_exact,
    }


def _summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty spatial-pyramid evaluation")
    failure_counts = {name: 0 for name in FACE_PART_NAMES}
    for row in rows:
        for name in row["shape_failed_parts"] + row["affine_failed_parts"]:
            failure_counts[name] += 1
    return {
        "row_count": len(rows),
        "combined_part_failures": int(sum(failure_counts.values())),
        "failure_counts": failure_counts,
        "median_shape_correlation": float(
            np.median([row["shape_correlation"] for row in rows])
        ),
        "median_gradient_correlation": float(
            np.median([row["gradient_correlation"] for row in rows])
        ),
        "median_normalized_rmse": float(
            np.median([row["normalized_rmse"] for row in rows])
        ),
        "source_background_bit_exact": all(
            row["crop_background_bit_exact"]
            and row["full_image_correction_background_bit_exact"]
            for row in rows
        ),
        "rows": list(rows),
    }


def _evaluate(
    corpus_root: Path,
    items: Sequence[PreparedSpatialFace],
    residuals: Sequence[np.ndarray],
    output_dir: Path,
    label: str,
) -> dict:
    return _summarize(
        [
            _quality(corpus_root, item, residual, output_dir, label)
            for item, residual in zip(items, residuals, strict=True)
        ]
    )


def _tagged_part_failures(row: dict) -> set[tuple[str, str]]:
    return {
        *(("shape", str(name)) for name in row["shape_failed_parts"]),
        *(("affine", str(name)) for name in row["affine_failed_parts"]),
    }


def _per_row_non_regression(candidate: dict, reference: dict) -> bool:
    candidate_rows = {row["row_id"]: row for row in candidate["rows"]}
    reference_rows = {row["row_id"]: row for row in reference["rows"]}
    if (
        len(candidate_rows) != len(candidate["rows"])
        or len(reference_rows) != len(reference["rows"])
        or candidate_rows.keys() != reference_rows.keys()
    ):
        return False
    return all(
        _tagged_part_failures(candidate_rows[row_id])
        <= _tagged_part_failures(reference_rows[row_id])
        for row_id in reference_rows
    )


def _strictly_beats(candidate: dict, reference: dict) -> bool:
    return bool(
        candidate["combined_part_failures"] < reference["combined_part_failures"]
        and candidate["median_shape_correlation"]
        >= reference["median_shape_correlation"] - 1e-6
        and candidate["median_gradient_correlation"]
        >= reference["median_gradient_correlation"] - 1e-6
        and candidate["median_normalized_rmse"]
        <= reference["median_normalized_rmse"] + 1e-6
        and candidate["source_background_bit_exact"]
        and _per_row_non_regression(candidate, reference)
    )


def _select_blend(candidates: Sequence[dict], baseline: dict) -> dict:
    if not candidates:
        raise ValueError("Spatial-pyramid blend selection is empty")
    evaluated = []
    for candidate in candidates:
        record = dict(candidate)
        record["eligible_vs_baseline"] = _strictly_beats(record, baseline)
        evaluated.append(record)
    eligible = [candidate for candidate in evaluated if candidate["eligible_vs_baseline"]]
    selected = min(
        eligible if eligible else evaluated,
        key=lambda candidate: (
            candidate["combined_part_failures"],
            candidate["median_normalized_rmse"],
            -candidate["median_gradient_correlation"],
            -candidate["median_shape_correlation"],
            candidate["alpha"],
        ),
    )
    return selected


def _compact_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if key != "rows"}


def run(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    expected_corpus_summary_sha256: str,
    device: str = "cuda",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    limit_per_split: int | None = None,
    loss_profile: str = "balanced",
) -> dict:
    import huggingface_hub
    import PIL
    import torch
    import transformers

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before deterministic HSRD training setup; "
            "run the trainer in a fresh process"
        )
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("HSRD spatial-pyramid training requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    corpus_summary_sha256 = _validate_corpus_summary_hash(
        summary_path, expected_corpus_summary_sha256
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"Unknown spatial-pyramid loss profile: {loss_profile}")
    loss_weights = LOSS_PROFILES[loss_profile].validated()
    corpus_contract = _validate_corpus(summary)
    rows = _select_rows(summary["rows"], limit_per_split)
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "sealed")
    }
    started = time.perf_counter()
    prepared, model_provenance = prepare_rows(corpus_root, rows, device)
    by_split = {
        split: [item for item in prepared if item.row["split"] == split]
        for split in split_counts
    }
    torch.manual_seed(TRAINING_SEED)
    initial = build_spatial_pyramid()
    initial_state = copy.deepcopy(initial.state_dict())
    parameters = parameter_count(initial)
    del initial
    candidate_model, candidate_training = train_decoder(
        by_split["train"],
        by_split["validation"],
        initial_state=initial_state,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        control=False,
        loss_weights=loss_weights,
    )
    control_model, control_training = train_decoder(
        by_split["train"],
        by_split["validation"],
        initial_state=initial_state,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        control=True,
        loss_weights=loss_weights,
    )
    evaluations = {}
    residuals_by_split = {}
    for split, items in by_split.items():
        zeros = [np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32) for _ in items]
        candidate_residuals = _predict_residuals(
            candidate_model, items, device, batch_size, control=False
        )
        control_residuals = _predict_residuals(
            control_model, items, device, batch_size, control=True
        )
        residuals_by_split[split] = {
            "candidate": candidate_residuals,
            "control": control_residuals,
        }
        evaluations[split] = {
            "baseline": _evaluate(
                corpus_root, items, zeros, output_dir, f"{split}_baseline"
            ),
            "candidate": _evaluate(
                corpus_root,
                items,
                candidate_residuals,
                output_dir,
                f"{split}_candidate",
            ),
            "control": _evaluate(
                corpus_root,
                items,
                control_residuals,
                output_dir,
                f"{split}_control",
            ),
        }
    validation_sweep = {}
    for method in ("candidate", "control"):
        validation_sweep[method] = []
        for alpha in BLEND_ALPHAS:
            summary_at_alpha = _evaluate(
                corpus_root,
                by_split["validation"],
                [
                    float(alpha) * residual
                    for residual in residuals_by_split["validation"][method]
                ],
                output_dir,
                f"validation_{method}_blend_{alpha:g}",
            )
            summary_at_alpha["alpha"] = float(alpha)
            validation_sweep[method].append(summary_at_alpha)
    selected = {
        method: _select_blend(
            validation_sweep[method], evaluations["validation"]["baseline"]
        )
        for method in ("candidate", "control")
    }
    sealed_selected = {}
    for method in ("candidate", "control"):
        alpha = float(selected[method]["alpha"])
        sealed_selected[method] = _evaluate(
            corpus_root,
            by_split["sealed"],
            [
                alpha * residual
                for residual in residuals_by_split["sealed"][method]
            ],
            output_dir,
            f"sealed_{method}_selected_{alpha:g}",
        )
        sealed_selected[method]["alpha"] = alpha
    candidate_validation_eligible = bool(selected["candidate"]["eligible_vs_baseline"])
    control_validation_eligible = bool(selected["control"]["eligible_vs_baseline"])
    decision = {
        "candidate_validation_beats_baseline": candidate_validation_eligible,
        "matched_control_validation_beats_baseline": control_validation_eligible,
        "multiscale_validation_advantage": bool(
            candidate_validation_eligible
            and (
                not control_validation_eligible
                or _strictly_beats(selected["candidate"], selected["control"])
            )
        ),
        "sealed_beats_baseline": _strictly_beats(
            sealed_selected["candidate"], evaluations["sealed"]["baseline"]
        ),
        "sealed_beats_matched_control": _strictly_beats(
            sealed_selected["candidate"], sealed_selected["control"]
        ),
        "background_bit_exact": all(
            lane[method]["source_background_bit_exact"]
            for lane in evaluations.values()
            for method in ("baseline", "candidate", "control")
        ),
        "equal_parameter_count": parameter_count(candidate_model)
        == parameter_count(control_model)
        == parameters,
    }
    decision["advance_to_exact_photo_and_30mm_replay"] = bool(
        decision["candidate_validation_beats_baseline"]
        and decision["multiscale_validation_advantage"]
        and decision["sealed_beats_baseline"]
        and decision["sealed_beats_matched_control"]
        and decision["background_bit_exact"]
        and decision["equal_parameter_count"]
    )
    checkpoint_path = output_dir / "hsrd_face_spatial_pyramid.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": METHOD,
            "control_method": CONTROL_METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "config": SpatialPyramidConfig().provenance(),
            "selected_candidate_alpha": selected["candidate"]["alpha"],
            "selected_control_alpha": selected["control"]["alpha"],
            "candidate_state_dict": {
                name: value.detach().cpu()
                for name, value in candidate_model.state_dict().items()
            },
            "control_state_dict": {
                name: value.detach().cpu()
                for name, value in control_model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    evidence = {
        "schema_version": 1,
        "method": METHOD,
        "control_method": CONTROL_METHOD,
        "status": (
            "advance-to-exact-replay"
            if decision["advance_to_exact_photo_and_30mm_replay"]
            else "hold"
        ),
        "architecture": SpatialPyramidConfig().provenance(),
        "parameter_count_each": parameters,
        "training_seed": TRAINING_SEED,
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "cuda_initialized_before_setup": False,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "nvidia_driver_version": _nvidia_driver_version(),
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "transformers_version": transformers.__version__,
            "huggingface_hub_version": huggingface_hub.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "pillow_version": PIL.__version__,
        },
        "loss_profile": loss_profile,
        "loss_weights": loss_weights.__dict__,
        "corpus": {
            "provider": HSRD_REPOSITORY,
            "revision": HSRD_REVISION,
            "license": HSRD_LICENSE,
            "summary_sha256": corpus_summary_sha256,
            "expected_summary_sha256": expected_corpus_summary_sha256.lower(),
            "summary_hash_verified": True,
            "contract": corpus_contract,
            "selected_rows": len(rows),
            "split_counts": split_counts,
        },
        "foundation_model": model_provenance,
        "candidate_training": candidate_training,
        "control_training": control_training,
        "evaluations": evaluations,
        "blend_alphas": list(BLEND_ALPHAS),
        "validation_blend_sweep": validation_sweep,
        "selected_validation": selected,
        "selected_sealed": sealed_selected,
        "decision": decision,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": max(
            model_provenance["peak_vram_gib"],
            candidate_training["peak_vram_gib"],
            control_training["peak_vram_gib"],
        ),
        "production_changed": False,
    }
    evidence_path = output_dir / "summary.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "status": evidence["status"],
        "decision": decision,
        "validation": {
            name: {
                "failures": value["combined_part_failures"],
                "shape": value["median_shape_correlation"],
                "gradient": value["median_gradient_correlation"],
                "rmse": value["median_normalized_rmse"],
            }
            for name, value in evaluations["validation"].items()
        },
        "sealed": {
            name: {
                "failures": value["combined_part_failures"],
                "shape": value["median_shape_correlation"],
                "gradient": value["median_gradient_correlation"],
                "rmse": value["median_normalized_rmse"],
            }
            for name, value in evaluations["sealed"].items()
        },
        "selected_validation": {
            name: _compact_summary(value) for name, value in selected.items()
        },
        "selected_sealed": {
            name: _compact_summary(value) for name, value in sealed_selected.items()
        },
        "runtime_seconds": evidence["runtime_seconds"],
        "peak_vram_gib": evidence["peak_vram_gib"],
    }, indent=2, sort_keys=True), flush=True)
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-corpus-summary-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--loss-profile", choices=tuple(LOSS_PROFILES), default="balanced")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        args.corpus_root,
        args.output_dir,
        expected_corpus_summary_sha256=args.expected_corpus_summary_sha256,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit_per_split=args.limit_per_split,
        loss_profile=args.loss_profile,
    )


if __name__ == "__main__":
    main()
