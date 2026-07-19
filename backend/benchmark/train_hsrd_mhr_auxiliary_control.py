"""Compare matched HSRD-only and HSRD-plus-MHR face-depth controls."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import time
from typing import Sequence


DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_configured_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _configured_cublas_workspace not in {
    None,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
}:
    raise RuntimeError(
        "HSRD/MHR auxiliary training requires "
        f"CUBLAS_WORKSPACE_CONFIG={DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG} "
        "before import"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG

import cv2
import numpy as np
from PIL import __version__ as PILLOW_VERSION

from backend.benchmark.face_depth_spatial_pyramid import (
    SpatialPyramidConfig,
    build_spatial_pyramid,
    parameter_count,
)
from backend.benchmark.hsrd_face_training_corpus import (
    HSRD_LICENSE,
    HSRD_REPOSITORY,
    HSRD_REVISION,
)
from backend.benchmark.mhr_face_training_corpus import (
    MHR_LICENSE,
    MHR_PROVIDER,
    MHR_RELEASE_ARCHIVE_SHA256,
    MHR_RELEASE_URL,
    MHR_SOURCE_REVISION,
    MHR_SOURCE_URL,
)
from backend.benchmark.train_face_depth_head import (
    _validate_training_corpus_summary,
)
from backend.benchmark.train_hsrd_face_spatial_pyramid import (
    BLEND_ALPHAS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    LOSS_PROFILES,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SIZE,
    TRAINING_SEED,
    _batch,
    _compact_summary,
    _evaluate,
    _load_depth_model,
    _loss,
    _nvidia_driver_version,
    _predict_residuals,
    _prepare_one,
    _select_blend,
    _select_rows,
    _sha256,
    _strictly_beats,
    _validate_corpus,
)


METHOD = "dav2_small_deepest_hsrd_mhr_auxiliary_camera_z"
CONTROL_METHOD = "dav2_small_deepest_hsrd_only_camera_z"
DEFAULT_AUXILIARY_WEIGHT = 0.25
AUXILIARY_SCHEDULE_SEED = TRAINING_SEED + 1


def _resolve_auxiliary_epochs(epochs: int, auxiliary_epochs: int | None) -> int:
    total = int(epochs)
    if total < 1:
        raise ValueError("Training epochs must be positive")
    active = total if auxiliary_epochs is None else int(auxiliary_epochs)
    if not 1 <= active <= total:
        raise ValueError(f"MHR auxiliary epochs must be in [1, {total}]")
    return active


def _validate_summary_hash(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> str:
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"Expected {label} summary SHA256 must be 64 lowercase hex")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} summary SHA256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _validate_mhr_auxiliary_corpus(summary: dict) -> dict:
    contract = _validate_training_corpus_summary(summary)
    depth_provenance = summary.get("depth_target_provenance") or {}
    geometry_contract = summary.get("geometry_target_contract") or {}
    supervision_manifest = geometry_contract.get("supervision_manifest") or {}
    supervision_sha256 = str(supervision_manifest.get("sha256", ""))
    checks = {
        "provider": summary.get("provider") == MHR_PROVIDER,
        "source_revision": summary.get("source_revision") == MHR_SOURCE_REVISION,
        "license": (summary.get("source") or {}).get("license") == MHR_LICENSE,
        "privacy_safe_synthetic": summary.get("privacy_safe_synthetic") is True,
        "training_matrix": summary.get("matrix_kind") == "training",
        "training_eligible": summary.get("training_eligible") is True,
        "not_production_evidence": summary.get("production_training_eligible") is False,
        "not_promotion_evidence": summary.get("promotion_eligible") is False,
        "source_geometry_restricted": summary.get(
            "source_geometry_training_and_evaluation_only"
        )
        is True,
        "camera_aligned": contract.get("camera_aligned") is True,
        "target_depth": contract.get("target_depth") == "floating-normalized-camera-z",
        "camera_convention": depth_provenance.get("camera_convention")
        == "OpenCV +X right, +Y down, +Z forward",
        "no_metric_scale_claim": depth_provenance.get("metric_scale_claimed") is False,
        "training_rows_only": geometry_contract.get("training_rows_only") is True,
        "raw_targets_excluded_from_evidence": geometry_contract.get(
            "compact_evidence_must_exclude_raw_targets"
        )
        is True,
        "supervision_manifest": (
            supervision_manifest.get("path") == "training_supervision.json"
            and supervision_manifest.get("excluded_from_compact_evidence") is True
            and supervision_manifest.get("row_count") == 240
            and len(supervision_sha256) == 64
            and all(character in "0123456789abcdef" for character in supervision_sha256)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("MHR auxiliary corpus gate failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "training_contract": contract,
        "supervision_manifest": supervision_manifest,
    }


def _select_mhr_training_rows(
    rows: Sequence[dict],
    limit: int | None,
) -> list[dict]:
    training = [row for row in rows if row.get("split") == "train"]
    if not training:
        raise ValueError("MHR auxiliary corpus contains no training rows")
    if any(row.get("split") != "train" for row in training):
        raise AssertionError("MHR auxiliary selection leaked a held-out row")
    if limit is not None:
        count = int(limit)
        if not 1 <= count <= len(training):
            raise ValueError(f"MHR auxiliary row limit must be in [1, {len(training)}]")
        training = training[:count]
    row_ids = [str(row.get("row_id", "")) for row in training]
    if not all(row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("MHR auxiliary training row IDs must be non-empty and unique")
    return training


def _partition_hsrd_rows(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if not all(row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("HSRD row IDs must be non-empty and unique")
    tuning = [row for row in rows if row.get("split") in {"train", "validation"}]
    sealed = [row for row in rows if row.get("split") == "sealed"]
    if (
        not tuning
        or not sealed
        or {row.get("split") for row in tuning} != {"train", "validation"}
        or len(tuning) + len(sealed) != len(rows)
    ):
        raise ValueError("HSRD rows must partition into train, validation, and sealed")
    if {row["row_id"] for row in tuning} & {row["row_id"] for row in sealed}:
        raise AssertionError("HSRD tuning and sealed row IDs overlap")
    return tuning, sealed


def _adapt_mhr_row(row: dict) -> dict:
    render = row.get("render") or {}
    camera = render.get("camera") or {}
    spec = row.get("spec") or {}
    bbox = render.get("face_bbox_xyxy")
    intrinsics = camera.get("intrinsics")
    yaw = spec.get("camera_yaw_deg")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and isinstance(intrinsics, list)
        and len(intrinsics) == 3
        and yaw is not None
    ):
        raise ValueError(f"MHR row {row.get('row_id')} lacks camera-aligned geometry")
    adapted = dict(row)
    adapted["selection_geometry"] = {
        "face_bbox_xyxy": list(bbox),
        "face_bbox_height_pixels": int(render["face_bbox_height_pixels"]),
        "camera": {"intrinsics": intrinsics},
    }
    adapted["rendering"] = {"camera_yaw_degrees": float(yaw)}
    adapted["auxiliary_source_provider"] = MHR_PROVIDER
    return adapted


def _prepare_sources(
    hsrd_root: Path,
    hsrd_rows: Sequence[dict],
    mhr_root: Path,
    mhr_rows: Sequence[dict],
    device: str,
):
    import torch

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    processor, depth_model, provenance = _load_depth_model(device)
    hsrd_prepared = []
    mhr_prepared = []
    try:
        total = len(hsrd_rows) + len(mhr_rows)
        for index, row in enumerate(hsrd_rows, start=1):
            hsrd_prepared.append(
                _prepare_one(hsrd_root, row, processor, depth_model, device)
            )
            if index % 10 == 0 or index == len(hsrd_rows):
                print(f"prepared {index}/{total} HSRD/MHR rows", flush=True)
        offset = len(hsrd_rows)
        for index, row in enumerate(mhr_rows, start=1):
            mhr_prepared.append(
                _prepare_one(
                    mhr_root,
                    _adapt_mhr_row(row),
                    processor,
                    depth_model,
                    device,
                )
            )
            done = offset + index
            if index % 20 == 0 or index == len(mhr_rows):
                print(f"prepared {done}/{total} HSRD/MHR rows", flush=True)
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    finally:
        depth_model.to("cpu")
        del depth_model
        torch.cuda.empty_cache()
    return (
        hsrd_prepared,
        mhr_prepared,
        {
            **provenance,
            "hsrd_row_count": len(hsrd_prepared),
            "mhr_auxiliary_row_count": len(mhr_prepared),
            "runtime_seconds": time.perf_counter() - started,
            "peak_vram_gib": peak,
            "frozen_feature_shapes": [
                list(values.shape) for values in hsrd_prepared[0].features
            ],
        },
    )


def _state_dict_sha256(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _schedule_sha256(row_ids: Sequence[str]) -> str:
    payload = json.dumps(list(row_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _next_auxiliary_batch(
    order: list,
    cursor: int,
    count: int,
    generator: random.Random,
) -> tuple[list, int]:
    if not order:
        raise ValueError("Auxiliary batching requires at least one row")
    selected = []
    while len(selected) < count:
        if cursor >= len(order):
            generator.shuffle(order)
            cursor = 0
        take = min(count - len(selected), len(order) - cursor)
        selected.extend(order[cursor : cursor + take])
        cursor += take
    return selected, cursor


def train_matched_control(
    hsrd_train_items,
    validation_items,
    *,
    auxiliary_items,
    auxiliary_weight: float,
    auxiliary_epochs: int,
    initial_state: dict,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_weights,
):
    import torch

    weight = float(auxiliary_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("MHR auxiliary loss weight must be in [0, 1]")
    if bool(auxiliary_items) != bool(weight > 0.0):
        raise ValueError("Auxiliary rows and a positive auxiliary weight must agree")
    if auxiliary_items and not 1 <= int(auxiliary_epochs) <= int(epochs):
        raise ValueError("Active auxiliary epochs must fit inside training epochs")
    if not auxiliary_items and int(auxiliary_epochs) != 0:
        raise ValueError("HSRD-only control must use zero auxiliary epochs")
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.cuda.manual_seed_all(TRAINING_SEED)
    model = build_spatial_pyramid().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    hsrd_generator = random.Random(TRAINING_SEED)
    auxiliary_generator = random.Random(AUXILIARY_SCHEDULE_SEED)
    best_epoch = 0
    from backend.benchmark.train_hsrd_face_spatial_pyramid import _validation_loss

    best_validation = _validation_loss(
        model,
        validation_items,
        device,
        batch_size,
        control=True,
        loss_weights=loss_weights,
    )
    history = [{"epoch": 0, "validation_loss": best_validation}]
    hsrd_schedule = []
    auxiliary_schedule = []
    optimizer_steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    auxiliary_order = list(auxiliary_items or [])
    auxiliary_cursor = len(auxiliary_order)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        hsrd_order = list(hsrd_train_items)
        hsrd_generator.shuffle(hsrd_order)
        total_losses = []
        hsrd_losses = []
        auxiliary_losses = []
        for start in range(0, len(hsrd_order), int(batch_size)):
            hsrd_items = hsrd_order[start : start + int(batch_size)]
            hsrd_schedule.extend(str(item.row["row_id"]) for item in hsrd_items)
            hsrd_batch = _batch(hsrd_items, device, control=True)
            optimizer.zero_grad(set_to_none=True)
            hsrd_loss, _, _ = _loss(model, hsrd_batch, loss_weights)
            loss = hsrd_loss
            if auxiliary_order and epoch <= int(auxiliary_epochs):
                auxiliary_batch_items, auxiliary_cursor = _next_auxiliary_batch(
                    auxiliary_order,
                    auxiliary_cursor,
                    len(hsrd_items),
                    auxiliary_generator,
                )
                auxiliary_schedule.extend(
                    str(item.row["row_id"]) for item in auxiliary_batch_items
                )
                auxiliary_batch = _batch(
                    auxiliary_batch_items,
                    device,
                    control=True,
                )
                auxiliary_loss, _, _ = _loss(model, auxiliary_batch, loss_weights)
                loss = hsrd_loss + weight * auxiliary_loss
                auxiliary_losses.append(float(auxiliary_loss.detach()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            hsrd_losses.append(float(hsrd_loss.detach()))
            total_losses.append(float(loss.detach()))
        validation = _validation_loss(
            model,
            validation_items,
            device,
            batch_size,
            control=True,
            loss_weights=loss_weights,
        )
        if validation < best_validation - 1e-9:
            best_validation = validation
            best_epoch = epoch
        history.append(
            {
                "epoch": epoch,
                "mean_hsrd_training_loss": float(np.mean(hsrd_losses)),
                "mean_auxiliary_training_loss": (
                    float(np.mean(auxiliary_losses)) if auxiliary_losses else None
                ),
                "mean_total_training_loss": float(np.mean(total_losses)),
                "validation_loss": validation,
            }
        )
        if not auxiliary_order:
            lane = "hsrd-only"
        elif epoch <= int(auxiliary_epochs):
            lane = "mixed"
        else:
            lane = "hsrd-finetune"
        print(f"{lane} epoch {epoch}: validation {validation:.6f}", flush=True)
    final_state = copy.deepcopy(model.state_dict())
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "selected_epoch": int(epochs),
        "checkpoint_selection": "fixed-final-step",
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": 1e-4,
            "gradient_clip_norm": 1.0,
        },
        "optimizer_steps": optimizer_steps,
        "hsrd_exposure_count": len(hsrd_schedule),
        "hsrd_schedule_sha256": _schedule_sha256(hsrd_schedule),
        "auxiliary_weight": weight,
        "auxiliary_epochs": int(auxiliary_epochs),
        "hsrd_only_finetune_epochs": int(epochs) - int(auxiliary_epochs),
        "auxiliary_exposure_count": len(auxiliary_schedule),
        "auxiliary_schedule_seed": (
            AUXILIARY_SCHEDULE_SEED if auxiliary_schedule else None
        ),
        "auxiliary_schedule_sha256": (
            _schedule_sha256(auxiliary_schedule) if auxiliary_schedule else None
        ),
        "initial_state_sha256": _state_dict_sha256(initial_state),
        "selected_state_sha256": _state_dict_sha256(final_state),
    }


def _matched_training_contract(control: dict, candidate: dict) -> dict:
    def valid_sha256(value) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    control_optimizer = control.get("optimizer")
    candidate_optimizer = candidate.get("optimizer")
    optimizer_valid = all(
        isinstance(optimizer, dict)
        and optimizer.get("name") == "AdamW"
        and all(
            isinstance(optimizer.get(name), (int, float))
            and np.isfinite(float(optimizer[name]))
            for name in ("learning_rate", "weight_decay", "gradient_clip_norm")
        )
        for optimizer in (control_optimizer, candidate_optimizer)
    )
    control_epoch = control.get("selected_epoch")
    candidate_epoch = candidate.get("selected_epoch")
    checks = {
        "valid_initial_state_hashes": valid_sha256(control.get("initial_state_sha256"))
        and valid_sha256(candidate.get("initial_state_sha256")),
        "same_initial_state": valid_sha256(control.get("initial_state_sha256"))
        and control["initial_state_sha256"] == candidate.get("initial_state_sha256"),
        "valid_hsrd_schedule_hashes": valid_sha256(control.get("hsrd_schedule_sha256"))
        and valid_sha256(candidate.get("hsrd_schedule_sha256")),
        "same_hsrd_schedule": valid_sha256(control.get("hsrd_schedule_sha256"))
        and control["hsrd_schedule_sha256"] == candidate.get("hsrd_schedule_sha256"),
        "same_positive_hsrd_exposure": positive_int(control.get("hsrd_exposure_count"))
        and control["hsrd_exposure_count"] == candidate.get("hsrd_exposure_count"),
        "same_positive_optimizer_steps": positive_int(control.get("optimizer_steps"))
        and control["optimizer_steps"] == candidate.get("optimizer_steps"),
        "valid_same_optimizer": optimizer_valid
        and control_optimizer == candidate_optimizer,
        "same_fixed_final_epoch": positive_int(control_epoch)
        and control_epoch == candidate_epoch
        and control.get("checkpoint_selection") == "fixed-final-step"
        and candidate.get("checkpoint_selection") == "fixed-final-step",
        "control_has_no_auxiliary": control.get("auxiliary_exposure_count") == 0
        and control.get("auxiliary_epochs") == 0
        and control.get("auxiliary_weight") == 0.0
        and control.get("auxiliary_schedule_seed") is None
        and control.get("auxiliary_schedule_sha256") is None,
        "candidate_has_valid_auxiliary": positive_int(
            candidate.get("auxiliary_exposure_count")
        )
        and positive_int(candidate.get("auxiliary_epochs"))
        and positive_int(candidate_epoch)
        and candidate["auxiliary_epochs"] <= candidate_epoch
        and isinstance(candidate.get("auxiliary_weight"), (int, float))
        and np.isfinite(float(candidate["auxiliary_weight"]))
        and 0.0 < float(candidate["auxiliary_weight"]) <= 1.0
        and candidate.get("auxiliary_schedule_seed") == AUXILIARY_SCHEDULE_SEED
        and valid_sha256(candidate.get("auxiliary_schedule_sha256")),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _validation_gate(selected_validation: dict, foundation: dict) -> dict:
    required = {"hsrd_control", "mixed_auxiliary"}
    if set(selected_validation) != required:
        raise ValueError("Validation gate requires exact control and candidate methods")
    checks = {
        "mixed_beats_foundation": _strictly_beats(
            selected_validation["mixed_auxiliary"],
            foundation,
        ),
        "mixed_beats_hsrd_control": _strictly_beats(
            selected_validation["mixed_auxiliary"],
            selected_validation["hsrd_control"],
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _prepare_sealed_after_validation(
    validation_gate: dict,
    hsrd_root: Path,
    sealed_rows: Sequence[dict],
    mhr_root: Path,
    device: str,
):
    if validation_gate.get("passed") is not True:
        return None
    return _prepare_sources(hsrd_root, sealed_rows, mhr_root, [], device)


def _evaluate_models(
    corpus_root: Path,
    by_split: dict,
    control_model,
    candidate_model,
    device: str,
    batch_size: int,
    output_dir: Path,
):
    import torch

    residuals_by_split = {}
    candidate_model.to(device)
    for split, items in by_split.items():
        residuals_by_split.setdefault(split, {})["mixed_auxiliary"] = (
            _predict_residuals(
                candidate_model,
                items,
                device,
                batch_size,
                control=True,
            )
        )
    candidate_model.to("cpu")
    torch.cuda.empty_cache()
    control_model.to(device)
    for split, items in by_split.items():
        residuals_by_split.setdefault(split, {})["hsrd_control"] = _predict_residuals(
            control_model,
            items,
            device,
            batch_size,
            control=True,
        )
    control_model.to("cpu")
    torch.cuda.empty_cache()

    evaluations = {}
    for split, items in by_split.items():
        zeros = [np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32) for _ in items]
        evaluations[split] = {
            "foundation": _evaluate(
                corpus_root,
                items,
                zeros,
                output_dir,
                f"{split}_foundation",
            ),
            "hsrd_control": _evaluate(
                corpus_root,
                items,
                residuals_by_split[split]["hsrd_control"],
                output_dir,
                f"{split}_hsrd_control",
            ),
            "mixed_auxiliary": _evaluate(
                corpus_root,
                items,
                residuals_by_split[split]["mixed_auxiliary"],
                output_dir,
                f"{split}_mixed_auxiliary",
            ),
        }
    return evaluations, residuals_by_split


def run(
    hsrd_corpus_root: str | Path,
    mhr_corpus_root: str | Path,
    output_dir: str | Path,
    *,
    expected_hsrd_summary_sha256: str,
    expected_mhr_summary_sha256: str,
    device: str = "cuda",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    auxiliary_weight: float = DEFAULT_AUXILIARY_WEIGHT,
    auxiliary_epochs: int | None = None,
    hsrd_limit_per_split: int | None = None,
    mhr_training_limit: int | None = None,
    loss_profile: str = "balanced",
) -> dict:
    import huggingface_hub
    import torch
    import transformers

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA initialized before deterministic HSRD/MHR setup; use a fresh process"
        )
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("HSRD/MHR auxiliary training requires CUDA")
    if not 0.0 < float(auxiliary_weight) <= 1.0:
        raise ValueError("MHR auxiliary loss weight must be in (0, 1]")
    resolved_auxiliary_epochs = _resolve_auxiliary_epochs(epochs, auxiliary_epochs)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    hsrd_root = Path(hsrd_corpus_root)
    mhr_root = Path(mhr_corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hsrd_summary_path = hsrd_root / "summary.json"
    mhr_summary_path = mhr_root / "summary.json"
    hsrd_hash = _validate_summary_hash(
        hsrd_summary_path,
        expected_hsrd_summary_sha256,
        label="HSRD",
    )
    mhr_hash = _validate_summary_hash(
        mhr_summary_path,
        expected_mhr_summary_sha256,
        label="MHR",
    )
    hsrd_summary = json.loads(hsrd_summary_path.read_text(encoding="utf-8"))
    mhr_summary = json.loads(mhr_summary_path.read_text(encoding="utf-8"))
    hsrd_contract = _validate_corpus(hsrd_summary)
    mhr_contract = _validate_mhr_auxiliary_corpus(mhr_summary)
    if loss_profile not in LOSS_PROFILES:
        raise ValueError(f"Unknown spatial-pyramid loss profile: {loss_profile}")
    loss_weights = LOSS_PROFILES[loss_profile].validated()
    hsrd_rows = _select_rows(hsrd_summary["rows"], hsrd_limit_per_split)
    mhr_rows = _select_mhr_training_rows(
        mhr_summary["rows"],
        mhr_training_limit,
    )
    hsrd_split_counts = {
        split: sum(row["split"] == split for row in hsrd_rows)
        for split in ("train", "validation", "sealed")
    }
    tuning_rows, sealed_rows = _partition_hsrd_rows(hsrd_rows)

    started = time.perf_counter()
    hsrd_prepared, mhr_prepared, tuning_model_provenance = _prepare_sources(
        hsrd_root,
        tuning_rows,
        mhr_root,
        mhr_rows,
        device,
    )
    by_split = {
        split: [item for item in hsrd_prepared if item.row["split"] == split]
        for split in ("train", "validation")
    }
    torch.manual_seed(TRAINING_SEED)
    initial = build_spatial_pyramid()
    initial_state = copy.deepcopy(initial.state_dict())
    initial_state_sha256 = _state_dict_sha256(initial_state)
    parameters = parameter_count(initial)
    del initial

    control_model, control_training = train_matched_control(
        by_split["train"],
        by_split["validation"],
        auxiliary_items=None,
        auxiliary_weight=0.0,
        auxiliary_epochs=0,
        initial_state=initial_state,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_weights=loss_weights,
    )
    control_model.to("cpu")
    torch.cuda.empty_cache()
    candidate_model, candidate_training = train_matched_control(
        by_split["train"],
        by_split["validation"],
        auxiliary_items=mhr_prepared,
        auxiliary_weight=auxiliary_weight,
        auxiliary_epochs=resolved_auxiliary_epochs,
        initial_state=initial_state,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_weights=loss_weights,
    )
    candidate_model.to("cpu")
    torch.cuda.empty_cache()
    matched_training = _matched_training_contract(
        control_training,
        candidate_training,
    )
    if not matched_training["passed"]:
        raise RuntimeError("Matched HSRD training contract failed")

    evaluations, residuals_by_split = _evaluate_models(
        hsrd_root,
        by_split,
        control_model,
        candidate_model,
        device,
        batch_size,
        output_dir,
    )
    methods = ("hsrd_control", "mixed_auxiliary")
    validation_sweep = {}
    for method in methods:
        validation_sweep[method] = []
        for alpha in BLEND_ALPHAS:
            summary_at_alpha = _evaluate(
                hsrd_root,
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
    selected_validation = {
        method: _select_blend(
            validation_sweep[method],
            evaluations["validation"]["foundation"],
        )
        for method in methods
    }
    validation_gate = _validation_gate(
        selected_validation,
        evaluations["validation"]["foundation"],
    )
    sealed_bundle = _prepare_sealed_after_validation(
        validation_gate,
        hsrd_root,
        sealed_rows,
        mhr_root,
        device,
    )
    sealed_model_provenance = None
    selected_sealed = None
    if sealed_bundle is not None:
        sealed_prepared, unused_mhr, sealed_model_provenance = sealed_bundle
        if unused_mhr:
            raise AssertionError("Sealed HSRD preparation unexpectedly loaded MHR rows")
        sealed_evaluations, sealed_residuals = _evaluate_models(
            hsrd_root,
            {"sealed": sealed_prepared},
            control_model,
            candidate_model,
            device,
            batch_size,
            output_dir,
        )
        evaluations.update(sealed_evaluations)
        residuals_by_split.update(sealed_residuals)
        selected_sealed = {}
        for method in methods:
            alpha = float(selected_validation[method]["alpha"])
            selected_sealed[method] = _evaluate(
                hsrd_root,
                sealed_prepared,
                [alpha * residual for residual in residuals_by_split["sealed"][method]],
                output_dir,
                f"sealed_{method}_selected_{alpha:g}",
            )
            selected_sealed[method]["alpha"] = alpha

    decision = {
        "mixed_validation_beats_foundation": validation_gate["checks"][
            "mixed_beats_foundation"
        ],
        "mixed_validation_beats_hsrd_control": validation_gate["checks"][
            "mixed_beats_hsrd_control"
        ],
        "validation_gate_passed_before_sealed_access": validation_gate["passed"],
        "sealed_evaluated": selected_sealed is not None,
        "mixed_sealed_beats_foundation": (
            _strictly_beats(
                selected_sealed["mixed_auxiliary"],
                evaluations["sealed"]["foundation"],
            )
            if selected_sealed is not None
            else None
        ),
        "mixed_sealed_beats_hsrd_control": (
            _strictly_beats(
                selected_sealed["mixed_auxiliary"],
                selected_sealed["hsrd_control"],
            )
            if selected_sealed is not None
            else None
        ),
        "background_bit_exact": all(
            lane[method]["source_background_bit_exact"]
            for lane in evaluations.values()
            for method in ("foundation", *methods)
        ),
        "matched_training_contract": matched_training["passed"],
        "equal_parameter_count": parameter_count(control_model)
        == parameter_count(candidate_model)
        == parameters,
        "mhr_used_for_training_only": all(
            item.row.get("split") == "train" for item in mhr_prepared
        ),
    }
    decision["advance_to_exact_photo_and_30mm_replay"] = bool(
        validation_gate["passed"]
        and selected_sealed is not None
        and decision["mixed_sealed_beats_foundation"]
        and decision["mixed_sealed_beats_hsrd_control"]
        and decision["background_bit_exact"]
        and decision["matched_training_contract"]
        and decision["equal_parameter_count"]
        and decision["mhr_used_for_training_only"]
    )

    checkpoint_path = output_dir / "hsrd_mhr_auxiliary_control.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": METHOD,
            "control_method": CONTROL_METHOD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "architecture": SpatialPyramidConfig().provenance(),
            "initial_state_sha256": initial_state_sha256,
            "auxiliary_weight": float(auxiliary_weight),
            "auxiliary_epochs": resolved_auxiliary_epochs,
            "selected_candidate_alpha": selected_validation["mixed_auxiliary"]["alpha"],
            "selected_control_alpha": selected_validation["hsrd_control"]["alpha"],
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
        "auxiliary_schedule_seed": AUXILIARY_SCHEDULE_SEED,
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
            "pillow_version": PILLOW_VERSION,
        },
        "loss_profile": loss_profile,
        "loss_weights": loss_weights.__dict__,
        "auxiliary_weight": float(auxiliary_weight),
        "auxiliary_epochs": resolved_auxiliary_epochs,
        "hsrd_only_finetune_epochs": int(epochs) - resolved_auxiliary_epochs,
        "hsrd_corpus": {
            "provider": HSRD_REPOSITORY,
            "revision": HSRD_REVISION,
            "license": HSRD_LICENSE,
            "summary_sha256": hsrd_hash,
            "expected_summary_sha256": expected_hsrd_summary_sha256.lower(),
            "summary_hash_verified": True,
            "contract": hsrd_contract,
            "selected_rows": len(hsrd_rows),
            "split_counts": hsrd_split_counts,
            "sole_validation_and_sealed_decision_source": True,
            "sealed_rows_accessed": selected_sealed is not None,
        },
        "mhr_auxiliary_corpus": {
            "provider": MHR_PROVIDER,
            "repository": MHR_SOURCE_URL,
            "release": MHR_RELEASE_URL,
            "revision": MHR_SOURCE_REVISION,
            "license": MHR_LICENSE,
            "release_archive_sha256": MHR_RELEASE_ARCHIVE_SHA256,
            "summary_sha256": mhr_hash,
            "expected_summary_sha256": expected_mhr_summary_sha256.lower(),
            "summary_hash_verified": True,
            "contract": mhr_contract,
            "selected_training_rows": len(mhr_rows),
            "validation_rows_used": 0,
            "sealed_rows_used": 0,
            "promotion_evidence": False,
        },
        "foundation_model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "tuning_preparation": tuning_model_provenance,
            "sealed_preparation_after_passing_validation_gate": (
                sealed_model_provenance
            ),
        },
        "initial_state_sha256": initial_state_sha256,
        "control_training": control_training,
        "candidate_training": candidate_training,
        "matched_training": matched_training,
        "evaluations": evaluations,
        "blend_alphas": list(BLEND_ALPHAS),
        "validation_blend_sweep": validation_sweep,
        "selected_validation": selected_validation,
        "selected_sealed": selected_sealed,
        "validation_gate": validation_gate,
        "decision": decision,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gib": max(
            value
            for value in (
                tuning_model_provenance["peak_vram_gib"],
                (
                    sealed_model_provenance["peak_vram_gib"]
                    if sealed_model_provenance is not None
                    else None
                ),
                control_training["peak_vram_gib"],
                candidate_training["peak_vram_gib"],
            )
            if value is not None
        ),
        "production_changed": False,
    }
    evidence_path = output_dir / "summary.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
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
                "sealed": (
                    {
                        name: {
                            "failures": value["combined_part_failures"],
                            "shape": value["median_shape_correlation"],
                            "gradient": value["median_gradient_correlation"],
                            "rmse": value["median_normalized_rmse"],
                        }
                        for name, value in evaluations["sealed"].items()
                    }
                    if "sealed" in evaluations
                    else None
                ),
                "selected_validation": {
                    name: _compact_summary(value)
                    for name, value in selected_validation.items()
                },
                "selected_sealed": (
                    {
                        name: _compact_summary(value)
                        for name, value in selected_sealed.items()
                    }
                    if selected_sealed is not None
                    else None
                ),
                "runtime_seconds": evidence["runtime_seconds"],
                "peak_vram_gib": evidence["peak_vram_gib"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsrd-corpus-root", type=Path, required=True)
    parser.add_argument("--mhr-corpus-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-hsrd-summary-sha256", required=True)
    parser.add_argument("--expected-mhr-summary-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--auxiliary-weight",
        type=float,
        default=DEFAULT_AUXILIARY_WEIGHT,
    )
    parser.add_argument("--auxiliary-epochs", type=int)
    parser.add_argument("--hsrd-limit-per-split", type=int)
    parser.add_argument("--mhr-training-limit", type=int)
    parser.add_argument(
        "--loss-profile", choices=tuple(LOSS_PROFILES), default="balanced"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        args.hsrd_corpus_root,
        args.mhr_corpus_root,
        args.output_dir,
        expected_hsrd_summary_sha256=args.expected_hsrd_summary_sha256,
        expected_mhr_summary_sha256=args.expected_mhr_summary_sha256,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        auxiliary_weight=args.auxiliary_weight,
        auxiliary_epochs=args.auxiliary_epochs,
        hsrd_limit_per_split=args.hsrd_limit_per_split,
        mhr_training_limit=args.mhr_training_limit,
        loss_profile=args.loss_profile,
    )


if __name__ == "__main__":
    main()
