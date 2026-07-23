"""Train a frozen-DINOv2 spatial face-relief decoder on pinned local data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from backend.benchmark.train_face_surface_adapter import (
    BLEND_ALPHAS,
    TRAINING_SEED,
    _per_row_non_regression,
    _sha256,
    _strictly_improves,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    CachedFusionSurface,
    FACE_PART_NAMES,
    CACHE_SCHEMA_VERSION as PRODUCTION_CACHE_SCHEMA_VERSION,
    METHOD as PRODUCTION_CACHE_METHOD,
    _stack,
    evaluate_exact_surfaces,
    load_cached_surfaces,
    surface_fusion_training_loss,
)
from backend.benchmark.train_gnm_production_face_depth_fusion import (
    worst_part_physical_non_regression,
)


MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_LICENSE = "Apache-2.0"
MODEL_SOURCE = "https://github.com/facebookresearch/dinov2"
MODEL_FILE_HASHES = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}
METHOD = "dinov2_small_frozen_spatial_production_conditioned_face_relief"
CHECKPOINT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 2
FEATURE_CHANNELS = 384
FEATURE_SIZE = 16
MODEL_INPUT_SIZE = 224
CONDITIONING_CHANNELS = 9
MAXIMUM_NORMALIZED_RESIDUAL = 0.35
DEFAULT_EPOCHS = 16
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_OVERFIT_ROWS = 2
DEFAULT_OVERFIT_IMPROVEMENT_RATIO = 0.97
PART_VALUE_WEIGHT = 0.50
PART_GRADIENT_WEIGHT = 0.75
PART_NON_REGRESSION_WEIGHT = 1.00
WORST_PART_VALUE_WEIGHT = 0.75
WORST_PART_GRADIENT_WEIGHT = 1.25


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_model_root(
    model_root: str | Path,
    *,
    expected_hashes: dict[str, str] = MODEL_FILE_HASHES,
) -> dict:
    root = Path(model_root)
    observed = {}
    for name, expected in expected_hashes.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned DINOv2 file is missing: {path}")
        observed[name] = _file_sha256(path)
        if observed[name] != str(expected).lower():
            raise ValueError(
                f"Pinned DINOv2 hash mismatch for {name}: "
                f"expected {expected}, observed {observed[name]}"
            )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "source": MODEL_SOURCE,
        "files": {
            name: {
                "sha256": digest,
                "size_bytes": (root / name).stat().st_size,
            }
            for name, digest in observed.items()
        },
        "local_files_only": True,
        "executing_implementation": "transformers.Dinov2Model",
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _validated_sha256(value: object, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 lowercase hex characters")
    return normalized


def _validate_corpus_artifacts(corpus_root: Path, rows: Sequence[dict]) -> dict:
    root = corpus_root.resolve()
    digest = hashlib.sha256()
    total_bytes = 0
    artifact_count = 0

    for row in rows:
        row_id = str(row.get("row_id", ""))
        if not row_id:
            raise ValueError("Corpus artifact validation requires a row ID")
        parts = row.get("exact_face_parts")
        if not isinstance(parts, dict) or set(parts) != set(FACE_PART_NAMES):
            raise ValueError(
                f"Corpus row {row_id} must bind exactly six facial-part assets"
            )
        records = [
            ("source", row.get("source")),
            ("exact_depth", row.get("exact_depth")),
            ("selection_mask", row.get("selection_mask")),
            *[
                (f"exact_face_parts/{name}", parts.get(name))
                for name in FACE_PART_NAMES
            ],
        ]
        for logical_name, record in records:
            if not isinstance(record, dict) or not record.get("path"):
                raise ValueError(
                    f"Corpus row {row_id} is missing {logical_name} provenance"
                )
            relative_path = Path(str(record["path"]))
            path = (root / relative_path).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Corpus row {row_id} {logical_name} escapes the corpus root"
                ) from error
            if not path.is_file():
                raise FileNotFoundError(
                    f"Corpus row {row_id} {logical_name} is missing: {path}"
                )
            expected_hash = _validated_sha256(
                record.get("sha256"),
                label=f"Corpus row {row_id} {logical_name} SHA256",
            )
            observed_hash = _file_sha256(path)
            if observed_hash != expected_hash:
                raise ValueError(
                    f"Corpus artifact SHA256 mismatch for {row_id} "
                    f"{logical_name}: expected {expected_hash}, "
                    f"observed {observed_hash}"
                )
            size = path.stat().st_size
            total_bytes += size
            artifact_count += 1
            digest.update(
                (
                    f"{row_id}\0{logical_name}\0{relative_path.as_posix()}\0"
                    f"{observed_hash}\0{size}\n"
                ).encode("utf-8")
            )

    return {
        "artifact_count": artifact_count,
        "artifact_bytes": total_bytes,
        "ordered_artifact_content_sha256": digest.hexdigest(),
    }


def validate_cache_binding(
    corpus_root: str | Path,
    cache_root: str | Path,
    *,
    expected_corpus_summary_sha256: str,
    expected_ordered_row_content_sha256: str,
) -> dict:
    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    summary_path = corpus_root / "summary.json"
    manifest_path = cache_root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_summary_hash = _file_sha256(summary_path)
    expected_summary_hash = _validated_sha256(
        expected_corpus_summary_sha256,
        label="Expected corpus-summary SHA256",
    )
    if observed_summary_hash != expected_summary_hash:
        raise ValueError(
            "Corpus summary SHA256 mismatch: "
            f"expected {expected_summary_hash}, observed {observed_summary_hash}"
        )
    expected_ids = [str(row.get("row_id", "")) for row in summary.get("rows", [])]
    manifest_ids = [str(row_id) for row_id in manifest.get("row_ids", [])]
    if (
        not expected_ids
        or any(not row_id for row_id in expected_ids)
        or len(set(expected_ids)) != len(expected_ids)
        or manifest_ids != expected_ids
    ):
        raise ValueError("Production cache row IDs do not exactly match the corpus")
    checks = {
        "schema": manifest.get("schema_version") == PRODUCTION_CACHE_SCHEMA_VERSION,
        "method": manifest.get("method") == PRODUCTION_CACHE_METHOD,
        "summary_hash": manifest.get("corpus_summary_sha256") == observed_summary_hash,
        "row_count": manifest.get("row_count") == len(expected_ids),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("Production cache contract failed: " + ", ".join(failed))
    artifact_binding = _validate_corpus_artifacts(corpus_root, summary["rows"])
    digest = hashlib.sha256()
    total_bytes = 0
    for row_id in expected_ids:
        path = cache_root / "rows" / f"{row_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Production cache row is missing: {path}")
        row_hash = _file_sha256(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{row_id}\0{row_hash}\0{size}\n".encode("ascii"))
    observed_content_hash = digest.hexdigest()
    expected_content_hash = _validated_sha256(
        expected_ordered_row_content_sha256,
        label="Expected cache-row SHA256",
    )
    if observed_content_hash != expected_content_hash:
        raise ValueError(
            "Production cache row-content SHA256 mismatch: "
            f"expected {expected_content_hash}, observed {observed_content_hash}"
        )
    return {
        "schema_version": manifest["schema_version"],
        "method": manifest["method"],
        "manifest_sha256": _file_sha256(manifest_path),
        "summary_sha256": _file_sha256(summary_path),
        "expected_summary_sha256": expected_summary_hash,
        "row_count": len(expected_ids),
        "row_bytes": total_bytes,
        "ordered_row_content_sha256": observed_content_hash,
        "expected_ordered_row_content_sha256": expected_content_hash,
        **artifact_binding,
    }


def conditioning_from_tensors(tensors: dict):
    """Pack RGB, incumbent geometry, support, feather, and pixel coordinates."""

    import torch

    inputs = tensors["inputs"]
    required = {
        "inputs",
        "baseline",
        "support_face",
        "fusion_weight",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError("Missing spatial conditioning tensors: " + ", ".join(missing))
    if inputs.ndim != 4 or inputs.shape[1] != 7:
        raise ValueError("Production face inputs must have shape [B, 7, H, W]")
    packed = torch.cat(
        (
            inputs[:, :3],
            inputs[:, 3:4],
            tensors["baseline"],
            tensors["support_face"],
            tensors["fusion_weight"],
            inputs[:, -2:],
        ),
        dim=1,
    )
    if packed.shape[1] != CONDITIONING_CHANNELS or not torch.isfinite(packed).all():
        raise ValueError("DINOv2 face conditioning is invalid")
    return packed


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _block(input_channels: int, output_channels: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.SiLU(),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.GroupNorm(_group_count(output_channels), output_channels),
        nn.SiLU(),
    )


def build_spatial_decoder():
    """Build a high-resolution decoder around one frozen DINOv2 token grid."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class Dinov2FaceSpatialDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_projection = nn.Sequential(
                nn.Conv2d(FEATURE_CHANNELS, 96, 1),
                nn.GroupNorm(8, 96),
                nn.SiLU(),
            )
            self.decode_40 = _block(96, 64)
            self.decode_80 = _block(64, 48)
            self.decode_160 = _block(48, 32)
            self.conditioning = _block(CONDITIONING_CHANNELS, 32)
            self.fusion = _block(64, 32)
            self.output = nn.Conv2d(32, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features, conditioning, support):
            if features.ndim != 4 or features.shape[1:] != (
                FEATURE_CHANNELS,
                FEATURE_SIZE,
                FEATURE_SIZE,
            ):
                raise ValueError("DINOv2 feature grid must have shape [B, 384, 16, 16]")
            if (
                conditioning.ndim != 4
                or conditioning.shape[1] != CONDITIONING_CHANNELS
                or support.shape != (conditioning.shape[0], 1, *conditioning.shape[-2:])
                or features.shape[0] != conditioning.shape[0]
            ):
                raise ValueError("DINOv2 decoder inputs disagree")
            if not all(torch.isfinite(value).all() for value in (features, conditioning, support)):
                raise ValueError("DINOv2 decoder inputs must be finite")
            hidden = self.feature_projection(features.float())
            for target_size, layer in (
                (40, self.decode_40),
                (80, self.decode_80),
                (conditioning.shape[-1], self.decode_160),
            ):
                hidden = functional.interpolate(
                    hidden,
                    size=(int(target_size), int(target_size)),
                    mode="bilinear",
                    align_corners=False,
                )
                hidden = layer(hidden)
            conditioned = self.conditioning(conditioning.float())
            residual = float(MAXIMUM_NORMALIZED_RESIDUAL) * torch.tanh(
                self.output(self.fusion(torch.cat((hidden, conditioned), dim=1)))
            )
            return residual * torch.clamp(support, 0.0, 1.0)

    return Dinov2FaceSpatialDecoder()


def _load_encoder(model_root: Path, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    provenance = validate_pinned_model_root(model_root)
    processor = AutoImageProcessor.from_pretrained(model_root, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_root,
        local_files_only=True,
        dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, provenance


def aligned_processor_pixels(processor, images: Sequence[Image.Image]):
    """Resize the full square crop to 224 without the default center crop."""

    prepared = [
        image.convert("RGB").resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BICUBIC
        )
        for image in images
    ]
    pixels = processor(
        images=prepared,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )["pixel_values"]
    if tuple(pixels.shape[-2:]) != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        raise ValueError("Aligned DINOv2 preprocessing must preserve the full 224 crop")
    return pixels


def _tokens_to_grid(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (257, FEATURE_CHANNELS):
        raise ValueError(
            "Pinned DINOv2 Small must emit one CLS token and a 16x16 patch grid"
        )
    return tokens[:, 1:].transpose(1, 2).reshape(
        tokens.shape[0], FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE
    )


def extract_frozen_features(
    items: Sequence[CachedFusionSurface],
    model_root: str | Path,
    *,
    device: str,
    batch_size: int = 8,
) -> tuple[dict[str, object], dict]:
    import torch

    if not items:
        raise ValueError("DINOv2 feature extraction requires at least one face")
    processor, encoder, provenance = _load_encoder(Path(model_root), device)
    features = {}
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            batch = items[start : start + max(1, int(batch_size))]
            images = [Image.fromarray(item.rgb, mode="RGB") for item in batch]
            pixels = aligned_processor_pixels(processor, images).to(
                device=device,
                dtype=(torch.float16 if str(device).startswith("cuda") else torch.float32),
            )
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if str(device).startswith("cuda")
                else torch.autocast("cpu", enabled=False)
            )
            with context:
                output = encoder(pixel_values=pixels)
            grids = _tokens_to_grid(output.last_hidden_state).float().cpu()
            for item, grid in zip(batch, grids, strict=True):
                row_id = str(item.row["row_id"])
                if not row_id or row_id in features:
                    raise ValueError("DINOv2 feature row IDs must be unique and nonempty")
                features[row_id] = grid
    runtime = time.perf_counter() - started
    peak = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return features, {
        **provenance,
        "feature_shape": [FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE],
        "preprocessing": {
            "source_crop_resized_to": [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            "preserves_full_source_crop": True,
            "processor_resize": False,
            "processor_center_crop": False,
        },
        "row_count": len(features),
        "runtime_seconds": float(runtime),
        "peak_vram_gib": peak,
    }


def _features_for(items: Sequence[CachedFusionSurface], features_by_id: dict):
    import torch

    grids = []
    for item in items:
        row_id = str(item.row["row_id"])
        if row_id not in features_by_id:
            raise ValueError(f"Missing DINOv2 feature grid for {row_id}")
        grids.append(features_by_id[row_id])
    values = torch.stack(grids).float()
    if values.shape[1:] != (FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError("Cached DINOv2 feature grid has the wrong shape")
    return values


def _loss(residual, tensors: dict):
    total, details = surface_fusion_training_loss(
        residual,
        tensors,
        part_balanced_physical_value_weight=PART_VALUE_WEIGHT,
        part_balanced_physical_gradient_weight=PART_GRADIENT_WEIGHT,
        part_balanced_non_regression_weight=PART_NON_REGRESSION_WEIGHT,
    )
    worst_value, worst_gradient = worst_part_physical_non_regression(
        residual, tensors
    )
    total = (
        total
        + WORST_PART_VALUE_WEIGHT * worst_value
        + WORST_PART_GRADIENT_WEIGHT * worst_gradient
    )
    return total, {
        **details,
        "worst_part_value_non_regression": float(worst_value.detach()),
        "worst_part_gradient_non_regression": float(worst_gradient.detach()),
    }


def _batch_values(tensors: dict, indices: list[int], device: str) -> dict:
    return {key: value[indices].to(device) for key, value in tensors.items()}


def _validation_loss(model, items, features_by_id, device: str) -> tuple[float, dict]:
    import torch

    tensors = _stack(list(items))
    conditioning = conditioning_from_tensors(tensors)
    features = _features_for(items, features_by_id)
    model.eval()
    with torch.inference_mode():
        values = {key: value.to(device) for key, value in tensors.items()}
        residual = model(
            features.to(device), conditioning.to(device), values["support_face"]
        )
        loss, details = _loss(residual, values)
    return float(loss), details


def train_decoder(
    train_items: Sequence[CachedFusionSurface],
    validation_items: Sequence[CachedFusionSurface],
    features_by_id: dict,
    *,
    device: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = TRAINING_SEED,
    horizontal_flip_augmentation: bool = True,
) -> tuple[object, dict]:
    import torch

    if not train_items or not validation_items:
        raise ValueError("DINOv2 training requires nonempty train and validation rows")
    if horizontal_flip_augmentation:
        raise ValueError(
            "DINOv2 token grids cannot be flipped after positional self-attention; "
            "encode flipped RGB separately before enabling augmentation"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    train_tensors = _stack(list(train_items))
    validation_tensors = _stack(list(validation_items))
    train_conditioning = conditioning_from_tensors(train_tensors)
    validation_conditioning = conditioning_from_tensors(validation_tensors)
    train_features = _features_for(train_items, features_by_id)
    validation_features = _features_for(validation_items, features_by_id)
    model = build_spatial_decoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_loss, initial_details = _validation_loss(
        model, validation_items, features_by_id, device
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = initial_loss
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(len(train_items), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), max(1, int(batch_size))):
            indices = order[start : start + max(1, int(batch_size))]
            values = _batch_values(train_tensors, indices, device)
            features = train_features[indices].to(device)
            conditioning = train_conditioning[indices].to(device)
            residual = model(features, conditioning, values["support_face"])
            loss, _details = _loss(residual, values)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite DINOv2 face loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss, validation_details = _validation_loss(
            model, validation_items, features_by_id, device
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
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
    peak = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    return model.eval(), {
        "initial_validation_loss": initial_loss,
        "initial_validation_details": initial_details,
        "best_validation_loss": float(best_validation),
        "best_epoch": best_epoch,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "horizontal_flip_augmentation": bool(horizontal_flip_augmentation),
        "history": history,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_vram_gib": peak,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def predict_residuals(model, items, features_by_id, *, device: str, batch_size: int):
    import torch

    tensors = _stack(list(items))
    conditioning = conditioning_from_tensors(tensors)
    features = _features_for(items, features_by_id)
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            stop = start + max(1, int(batch_size))
            support = tensors["support_face"][start:stop].to(device)
            predicted = model(
                features[start:stop].to(device),
                conditioning[start:stop].to(device),
                support,
            )
            output.extend(predicted[:, 0].float().cpu().numpy())
    return output


def _select_overfit_items(items: Sequence[CachedFusionSurface], limit: int):
    selected = sorted(
        (item for item in items if item.row.get("split") == "train"),
        key=lambda item: (
            float(item.row["render"]["face_bbox_height_pixels"]),
            str(item.row["row_id"]),
        ),
    )[: int(limit)]
    if len(selected) != int(limit):
        raise ValueError("Overfit smoke could not select the requested train rows")
    identities = [str(item.row["identity_group"]) for item in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("Overfit smoke rows must use distinct identities")
    return selected


def overfit_gate(training: dict, *, maximum_ratio: float) -> dict:
    initial = float(training["initial_validation_loss"])
    final = float(training["best_validation_loss"])
    ratio = final / max(initial, 1e-12)
    checks = {
        "finite": math.isfinite(initial) and math.isfinite(final),
        "positive_initial": initial > 0.0,
        "selected_trained_epoch": int(training["best_epoch"]) > 0,
        "loss_ratio": ratio <= float(maximum_ratio),
    }
    return {
        "checks": checks,
        "initial_loss": initial,
        "best_loss": final,
        "best_to_initial_ratio": ratio,
        "maximum_ratio": float(maximum_ratio),
        "passed": all(checks.values()),
    }


def _part_records(row: dict, field: str) -> dict[str, dict]:
    metric = row.get(field)
    if not isinstance(metric, dict):
        raise ValueError(f"Face row is missing {field} telemetry")
    parts = metric.get("parts")
    if not isinstance(parts, list) or len(parts) != 6:
        raise ValueError(f"Face row must contain six {field} part records")
    records = {str(part.get("name", "")): part for part in parts}
    if len(records) != 6 or "" in records:
        raise ValueError(f"Face row contains duplicate or unnamed {field} parts")
    return records


def _finite_metric(record: dict, name: str) -> float:
    value = float(record.get(name, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"Face part metric {name} is missing or non-finite")
    return value


def _summary_row_map(summary: dict) -> dict[str, dict]:
    rows = summary.get("rows", [])
    ids = [str(row.get("row_id", "")) for row in rows]
    if not ids or any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Face summaries require unique nonempty row IDs")
    return {str(row["row_id"]): row for row in rows}


def _append_numeric_regression(
    regressions: list[dict],
    *,
    scope: str,
    row_id: str,
    metric: str,
    candidate_record: dict,
    baseline_record: dict,
    higher_is_better: bool,
    part: str | None = None,
    comparison=lambda value: value,
) -> None:
    candidate_value = _finite_metric(candidate_record, metric)
    baseline_value = _finite_metric(baseline_record, metric)
    candidate_comparison = float(comparison(candidate_value))
    baseline_comparison = float(comparison(baseline_value))
    regressed = (
        candidate_comparison < baseline_comparison - 1e-6
        if higher_is_better
        else candidate_comparison > baseline_comparison + 1e-6
    )
    if not regressed:
        return
    event = {
        "scope": scope,
        "row_id": row_id,
        "metric": metric,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "baseline_comparison": baseline_comparison,
        "candidate_comparison": candidate_comparison,
        "higher_is_better": higher_is_better,
    }
    if part is not None:
        event["part"] = part
    regressions.append(event)


def _paired_regression_audit(candidate: dict, baseline: dict) -> dict:
    candidate_rows = _summary_row_map(candidate)
    baseline_rows = _summary_row_map(baseline)
    if set(candidate_rows) != set(baseline_rows):
        raise ValueError("Paired face summaries require identical row IDs")
    regressions: list[dict] = []
    for row_id, row in candidate_rows.items():
        reference = baseline_rows[row_id]
        for metric, higher_is_better in (
            ("shape_correlation", True),
            ("gradient_correlation", True),
            ("normalized_rmse", False),
        ):
            _append_numeric_regression(
                regressions,
                scope="whole_face",
                row_id=row_id,
                metric=metric,
                candidate_record=row,
                baseline_record=reference,
                higher_is_better=higher_is_better,
            )
        for field in ("shape_failed_parts", "affine_failed_parts"):
            added = sorted(set(row.get(field, [])) - set(reference.get(field, [])))
            for part in added:
                regressions.append(
                    {
                        "scope": "named_part",
                        "row_id": row_id,
                        "part": str(part),
                        "metric": field,
                        "kind": "new_failed_part",
                    }
                )
        candidate_shape = _part_records(row, "named_part_shape")
        baseline_shape = _part_records(reference, "named_part_shape")
        candidate_affine = _part_records(row, "named_part_affine_mm")
        baseline_affine = _part_records(reference, "named_part_affine_mm")
        if set(candidate_shape) != set(baseline_shape) or set(candidate_affine) != set(
            baseline_affine
        ):
            raise ValueError("Paired face rows contain different named parts")
        for name in candidate_shape:
            current_shape = candidate_shape[name]
            prior_shape = baseline_shape[name]
            current_affine = candidate_affine[name]
            prior_affine = baseline_affine[name]
            if not (current_shape.get("available") and current_affine.get("available")):
                regressions.append(
                    {
                        "scope": "named_part",
                        "row_id": row_id,
                        "part": name,
                        "metric": "available",
                        "kind": "missing_candidate_telemetry",
                        "baseline_value": bool(
                            prior_shape.get("available")
                            and prior_affine.get("available")
                        ),
                        "candidate_value": False,
                    }
                )
                continue
            for metric, higher_is_better in (
                ("shape_correlation", True),
                ("minimum_raw_gradient_correlation", True),
                ("face_normalized_shape_rmse", False),
            ):
                _append_numeric_regression(
                    regressions,
                    scope="named_part",
                    row_id=row_id,
                    part=name,
                    metric=metric,
                    candidate_record=current_shape,
                    baseline_record=prior_shape,
                    higher_is_better=higher_is_better,
                )
            for metric in ("rmse_mm", "p95_absolute_error_mm"):
                _append_numeric_regression(
                    regressions,
                    scope="named_part",
                    row_id=row_id,
                    part=name,
                    metric=metric,
                    candidate_record=current_affine,
                    baseline_record=prior_affine,
                    higher_is_better=False,
                )
            _append_numeric_regression(
                regressions,
                scope="named_part",
                row_id=row_id,
                part=name,
                metric="bias_mm",
                candidate_record=current_affine,
                baseline_record=prior_affine,
                higher_is_better=False,
                comparison=abs,
            )
            _append_numeric_regression(
                regressions,
                scope="named_part",
                row_id=row_id,
                part=name,
                metric="span_retention",
                candidate_record=current_affine,
                baseline_record=prior_affine,
                higher_is_better=False,
                comparison=lambda value: abs(value - 1.0),
            )
    whole_face_count = sum(
        event["scope"] == "whole_face" for event in regressions
    )
    named_part_count = len(regressions) - whole_face_count
    return {
        "schema_version": 1,
        "passed": not regressions,
        "regression_count": len(regressions),
        "whole_face_regression_count": whole_face_count,
        "named_part_regression_count": named_part_count,
        "regressions": regressions,
    }


def _tagged_part_non_regression(candidate: dict, baseline: dict) -> bool:
    return _paired_regression_audit(candidate, baseline)[
        "named_part_regression_count"
    ] == 0


def _whole_face_non_regression(candidate: dict, baseline: dict) -> bool:
    return _paired_regression_audit(candidate, baseline)[
        "whole_face_regression_count"
    ] == 0


def _select_validation_candidate(
    candidates: Sequence[dict], baseline: dict
) -> tuple[dict, list[dict]]:
    resolved = []
    for candidate in candidates:
        alpha = float(candidate.get("alpha", 0.0))
        audit = _paired_regression_audit(candidate, baseline)
        tagged = audit["named_part_regression_count"] == 0
        whole_face = audit["whole_face_regression_count"] == 0
        eligible = bool(
            alpha > 0.0
            and tagged
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
    eligible = [candidate for candidate in resolved if candidate["eligible"]]
    if not eligible:
        return (
            {
                **baseline,
                "alpha": 0.0,
                "tagged_part_non_regression": True,
                "whole_face_non_regression": True,
                "paired_regression_audit": _paired_regression_audit(
                    baseline, baseline
                ),
                "eligible": False,
            },
            resolved,
        )
    return min(
        eligible,
        key=lambda candidate: (
            int(candidate["combined_part_failures"]),
            -float(candidate["median_gradient_correlation"]),
            -float(candidate["median_shape_correlation"]),
            float(candidate["median_normalized_rmse"]),
            float(candidate["alpha"]),
        ),
    ), resolved


def run_split_training(
    corpus_root: str | Path,
    cache_root: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    expected_corpus_summary_sha256: str,
    expected_cache_row_sha256: str,
) -> dict:
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DINOv2 split training requested CUDA without a GPU")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_binding = validate_cache_binding(
        corpus_root,
        cache_root,
        expected_corpus_summary_sha256=expected_corpus_summary_sha256,
        expected_ordered_row_content_sha256=expected_cache_row_sha256,
    )
    all_items, summary, cache_manifest = load_cached_surfaces(corpus_root, cache_root)
    by_split = {
        split: [item for item in all_items if item.row.get("split") == split]
        for split in ("train", "validation", "sealed")
    }
    if any(not items for items in by_split.values()):
        raise ValueError("DINOv2 corpus must retain train, validation, and sealed rows")
    identity_splits = {}
    for item in all_items:
        identity_splits.setdefault(str(item.row["identity_group"]), set()).add(
            str(item.row["split"])
        )
    if any(len(splits) != 1 for splits in identity_splits.values()):
        raise ValueError("DINOv2 corpus identity groups cross split boundaries")

    tuning_items = by_split["train"] + by_split["validation"]
    features, encoder = extract_frozen_features(
        tuning_items, model_root, device=device, batch_size=batch_size
    )
    model, training = train_decoder(
        by_split["train"],
        by_split["validation"],
        features,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        horizontal_flip_augmentation=False,
    )
    zeros = [np.zeros_like(item.baseline, dtype=np.float32) for item in by_split["validation"]]
    baseline = evaluate_exact_surfaces(
        corpus_root,
        by_split["validation"],
        zeros,
        alpha=0.0,
        output_dir=output / "validation_baseline",
    )
    residuals = predict_residuals(
        model,
        by_split["validation"],
        features,
        device=device,
        batch_size=batch_size,
    )
    candidates = []
    for alpha in BLEND_ALPHAS:
        if float(alpha) == 0.0:
            continue
        candidate = evaluate_exact_surfaces(
            corpus_root,
            by_split["validation"],
            residuals,
            alpha=float(alpha),
            output_dir=output / f"validation_alpha_{float(alpha):g}",
        )
        candidates.append({**candidate, "alpha": float(alpha)})
    selected, candidate_audits = _select_validation_candidate(
        candidates, baseline
    )

    sealed_record = None
    sealed_gate = False
    sealed_encoder = None
    if float(selected["alpha"]) > 0.0:
        sealed_features, sealed_encoder = extract_frozen_features(
            by_split["sealed"], model_root, device=device, batch_size=batch_size
        )
        sealed_zeros = [
            np.zeros_like(item.baseline, dtype=np.float32)
            for item in by_split["sealed"]
        ]
        sealed_baseline = evaluate_exact_surfaces(
            corpus_root,
            by_split["sealed"],
            sealed_zeros,
            alpha=0.0,
            output_dir=output / "sealed_baseline",
        )
        sealed_residuals = predict_residuals(
            model,
            by_split["sealed"],
            sealed_features,
            device=device,
            batch_size=batch_size,
        )
        sealed_candidate = evaluate_exact_surfaces(
            corpus_root,
            by_split["sealed"],
            sealed_residuals,
            alpha=float(selected["alpha"]),
            output_dir=output / "sealed_selected",
        )
        sealed_audit = _paired_regression_audit(sealed_candidate, sealed_baseline)
        sealed_tagged = sealed_audit["named_part_regression_count"] == 0
        sealed_whole_face = sealed_audit["whole_face_regression_count"] == 0
        sealed_gate = bool(
            sealed_tagged
            and sealed_whole_face
            and _strictly_improves(sealed_candidate, sealed_baseline)
            and _per_row_non_regression(sealed_candidate, sealed_baseline) >= 1.0
        )
        sealed_record = {
            "baseline": sealed_baseline,
            "selected": {
                **sealed_candidate,
                "alpha": float(selected["alpha"]),
                "tagged_part_non_regression": sealed_tagged,
                "whole_face_non_regression": sealed_whole_face,
                "paired_regression_audit": sealed_audit,
                "eligible": sealed_gate,
            },
        }

    checkpoint_path = output / "dinov2_face_spatial_decoder.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "method": METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "selected_alpha": float(selected["alpha"]),
            "selected_epoch": int(training["best_epoch"]),
            "corpus_summary_sha256": _sha256(Path(corpus_root) / "summary.json"),
            "cache_manifest_sha256": _sha256(Path(cache_root) / "manifest.json"),
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    decision = {
        "validation_strictly_improves": bool(selected.get("eligible")),
        "sealed_opened": sealed_record is not None,
        "sealed_strictly_improves": bool(sealed_gate),
    }
    decision["advance_to_exact_photo"] = bool(all(decision.values()))
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "advance" if decision["advance_to_exact_photo"] else "hold",
        "production_changed": False,
        "method": METHOD,
        "source_geometry_training_and_evaluation_only": True,
        "encoder": encoder,
        "sealed_encoder": sealed_encoder,
        "corpus": {
            "privacy": summary.get("privacy"),
            "summary_sha256": _sha256(Path(corpus_root) / "summary.json"),
            "cache_manifest_sha256": _sha256(Path(cache_root) / "manifest.json"),
            "cache_method": cache_manifest.get("method"),
            "identity_disjoint": True,
            "split_counts": {split: len(items) for split, items in by_split.items()},
            "cache_binding": cache_binding,
        },
        "training": training,
        "validation": {
            "baseline": baseline,
            "candidates": candidate_audits,
            "selected": selected,
        },
        "sealed": sealed_record,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "decision": decision,
        "device": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else "cpu",
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def run_overfit_smoke(
    corpus_root: str | Path,
    cache_root: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    rows: int = DEFAULT_OVERFIT_ROWS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    maximum_loss_ratio: float = DEFAULT_OVERFIT_IMPROVEMENT_RATIO,
    expected_corpus_summary_sha256: str,
    expected_cache_row_sha256: str,
) -> dict:
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DINOv2 face smoke requested CUDA but no GPU is available")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_binding = validate_cache_binding(
        corpus_root,
        cache_root,
        expected_corpus_summary_sha256=expected_corpus_summary_sha256,
        expected_ordered_row_content_sha256=expected_cache_row_sha256,
    )
    all_items, summary, cache_manifest = load_cached_surfaces(corpus_root, cache_root)
    selected = _select_overfit_items(all_items, rows)
    features, encoder = extract_frozen_features(
        selected, model_root, device=device, batch_size=batch_size
    )
    model, training = train_decoder(
        selected,
        selected,
        features,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        horizontal_flip_augmentation=False,
    )
    gate = overfit_gate(training, maximum_ratio=maximum_loss_ratio)
    checkpoint_path = output / "dinov2_face_spatial_decoder_smoke.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "method": METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "row_ids": [item.row["row_id"] for item in selected],
            "state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "smoke-pass" if gate["passed"] else "smoke-hold",
        "production_changed": False,
        "advance_to_split_training": bool(gate["passed"]),
        "method": METHOD,
        "source_geometry_training_and_evaluation_only": True,
        "encoder": encoder,
        "architecture": {
            "feature_shape": [FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE],
            "conditioning_channels": CONDITIONING_CHANNELS,
            "conditioning_layout": (
                "rgb,local_depth,incumbent_baseline,face_support,"
                "fusion_weight,xy_coordinates"
            ),
            "maximum_normalized_residual": MAXIMUM_NORMALIZED_RESIDUAL,
            "trainable_parameters": training["trainable_parameters"],
        },
        "loss_weights": {
            "part_value": PART_VALUE_WEIGHT,
            "part_gradient": PART_GRADIENT_WEIGHT,
            "part_non_regression": PART_NON_REGRESSION_WEIGHT,
            "worst_part_value_non_regression": WORST_PART_VALUE_WEIGHT,
            "worst_part_gradient_non_regression": WORST_PART_GRADIENT_WEIGHT,
        },
        "corpus": {
            "privacy": summary.get("privacy"),
            "summary_sha256": _sha256(Path(corpus_root) / "summary.json"),
            "cache_manifest_sha256": _sha256(Path(cache_root) / "manifest.json"),
            "cache_method": cache_manifest.get("method"),
            "cache_binding": cache_binding,
            "selected_rows": [
                {
                    "row_id": item.row["row_id"],
                    "identity_group": item.row["identity_group"],
                    "face_height_pixels": item.row["render"][
                        "face_bbox_height_pixels"
                    ],
                }
                for item in selected
            ],
        },
        "training": training,
        "gate": gate,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "device": (
            torch.cuda.get_device_name(device)
            if str(device).startswith("cuda")
            else "cpu"
        ),
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
    parser.add_argument("--expected-corpus-summary-sha256", required=True)
    parser.add_argument("--expected-cache-row-sha256", required=True)
    parser.add_argument(
        "--maximum-loss-ratio",
        type=float,
        default=DEFAULT_OVERFIT_IMPROVEMENT_RATIO,
    )
    args = parser.parse_args()
    if args.mode == "overfit":
        evidence = run_overfit_smoke(
            args.corpus_root,
            args.cache_root,
            args.model_root,
            args.output_dir,
            device=args.device,
            rows=args.rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            maximum_loss_ratio=args.maximum_loss_ratio,
            expected_corpus_summary_sha256=(
                args.expected_corpus_summary_sha256
            ),
            expected_cache_row_sha256=args.expected_cache_row_sha256,
        )
        print(json.dumps(evidence["gate"], indent=2))
    else:
        evidence = run_split_training(
            args.corpus_root,
            args.cache_root,
            args.model_root,
            args.output_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            expected_corpus_summary_sha256=(
                args.expected_corpus_summary_sha256
            ),
            expected_cache_row_sha256=args.expected_cache_row_sha256,
        )
        print(json.dumps(evidence["decision"], indent=2))


if __name__ == "__main__":
    main()
