"""Train a low-rank GNM geometry predictor on identity-disjoint face rows."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from backend.benchmark.gnm_face_training_corpus import (
    GNM_EXPRESSION_DIMENSION,
    GNM_IDENTITY_DIMENSION,
    GNM_LICENSE,
    GNM_SOURCE_REVISION,
    GNMTrainingSceneSpec,
    _asset_paths,
    _load_gnm_model,
    decode_geometry_targets,
    preflight_gnm_root,
)
from backend.benchmark.mediapipe_expression_features import (
    blendshape_provenance,
    validated_blendshape_features,
)
from backend.benchmark.train_face_depth_head import _padded_box
from backend.benchmark.train_face_surface_fusion_adapter import (
    _detect_production_region,
    selected_image_from_exact_mask,
)
from backend.gnm_face_foundation import MEDIAPIPE_TO_DLIB68


ENCODER_ID = "depth-anything/Depth-Anything-V2-Small-hf"
ENCODER_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
ENCODER_MODEL_SHA256 = (
    "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
)
ENCODER_LICENSE = "Apache-2.0"
ENCODER_FEATURE_DIMENSION = 768
FEATURE_ENCODER_DAV2_SMALL = "dav2-small"
FEATURE_ENCODER_NONE = "none"
FEATURE_ENCODER_KINDS = (
    FEATURE_ENCODER_DAV2_SMALL,
    FEATURE_ENCODER_NONE,
)
CHECKPOINT_SCHEMA_VERSION = 4
IDENTITY_SKIN_DIMENSION = 170
EXPRESSION_SKIN_DIMENSION = 350
LANDMARK_FEATURE_DLIB68 = "mediapipe-to-dlib68-v1"
LANDMARK_FEATURE_FULL468 = "mediapipe-full468-v1"
LANDMARK_FEATURE_DLIB68_BLENDSHAPES52 = (
    "mediapipe-to-dlib68-plus-blendshapes52-v1"
)
LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9 = (
    "mediapipe-to-dlib68-plus-blendshapes52-plus-pose9-v1"
)
LANDMARK_FEATURE_KINDS = (
    LANDMARK_FEATURE_DLIB68,
    LANDMARK_FEATURE_FULL468,
    LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
    LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
)
SMALL_FACE_MINIMUM_PIXELS = 55
SMALL_FACE_MAXIMUM_PIXELS = 100
DEFAULT_IDENTITY_RANKS = (0, 4, 8, 12, 16)
DEFAULT_EXPRESSION_RANKS = (0, 8, 16, 24)
DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
PART_GROUPS = {
    "left_eye": ("left_eye", "left_orbital_region"),
    "right_eye": ("right_eye", "right_orbital_region"),
    "left_eyebrow": ("left_brow_region", "left_orbital_region"),
    "right_eyebrow": ("right_brow_region", "right_orbital_region"),
    "nose": ("nose_region",),
    "mouth": ("upper_lip_region", "lower_lip_region", "mouth_sock"),
}


def landmark_feature_dimension(feature_kind: str) -> int:
    if feature_kind == LANDMARK_FEATURE_DLIB68:
        return 68 * 3
    if feature_kind == LANDMARK_FEATURE_FULL468:
        return 468 * 3
    if feature_kind == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52:
        return 68 * 3 + 52
    if feature_kind == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9:
        return 68 * 3 + 52 + 9
    raise ValueError(
        f"Unsupported structured GNM landmark feature kind: {feature_kind}"
    )


@dataclass
class PreparedRow:
    row_id: str
    split: str
    identity_group: str
    expression_name: str
    face_height_pixels: int
    crop: Image.Image
    landmark_features: np.ndarray
    identity_target: np.ndarray
    expression_target: np.ndarray
    detector_scope: str
    detector_name: str


@dataclass
class RidgeHead:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    components: np.ndarray
    score_mean: np.ndarray
    weights: np.ndarray
    score_minimum: np.ndarray
    score_maximum: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        count = int(len(features))
        if self.rank == 0:
            return np.zeros((count, 0), dtype=np.float64)
        normalized = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        scores = normalized @ self.weights + self.score_mean
        return np.clip(scores, self.score_minimum, self.score_maximum)

    def predict_targets(self, features: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(features)
        return self.target_mean + scores @ self.components


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_landmark_features(
    landmarks_xyz: np.ndarray,
    *,
    feature_kind: str = LANDMARK_FEATURE_DLIB68,
) -> np.ndarray:
    landmarks = np.asarray(landmarks_xyz, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] < 3 or len(landmarks) < 468:
        raise ValueError("Structured GNM training requires 468 MediaPipe landmarks")
    if feature_kind in (
        LANDMARK_FEATURE_DLIB68,
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
    ):
        mapped = np.asarray(
            [
                np.mean(landmarks[list(indices), :3], axis=0)
                for indices in MEDIAPIPE_TO_DLIB68
            ],
            dtype=np.float64,
        )
        expected_count = 68
    elif feature_kind == LANDMARK_FEATURE_FULL468:
        mapped = landmarks[:468, :3].copy()
        expected_count = 468
    else:
        raise ValueError(
            f"Unsupported structured GNM landmark feature kind: {feature_kind}"
        )
    if mapped.shape != (expected_count, 3) or not np.all(np.isfinite(mapped)):
        raise ValueError("Structured GNM landmark mapping produced invalid values")
    xy = mapped[:, :2]
    xy_center = np.mean(xy, axis=0)
    xy_scale = float(np.max(np.ptp(xy, axis=0)))
    if not math.isfinite(xy_scale) or xy_scale <= 1e-8:
        raise ValueError("Structured GNM landmarks have no usable image scale")
    normalized_xy = (xy - xy_center) / xy_scale
    z = mapped[:, 2] - np.median(mapped[:, 2])
    z_low, z_high = np.percentile(z, (5.0, 95.0))
    z_scale = float(z_high - z_low)
    if not math.isfinite(z_scale) or z_scale <= 1e-8:
        z_scale = float(np.max(np.abs(z)))
    if not math.isfinite(z_scale) or z_scale <= 1e-8:
        z_scale = 1.0
    normalized_z = z / z_scale
    features = np.column_stack((normalized_xy, normalized_z)).reshape(-1)
    if not np.all(np.isfinite(features)):
        raise ValueError("Structured GNM landmark features must be finite")
    return features.astype(np.float32)


def structured_face_features(
    landmarks_xyz: np.ndarray,
    *,
    feature_kind: str,
    blendshape_names=None,
    blendshape_scores=None,
    facial_transformation_matrix=None,
) -> tuple[np.ndarray, dict]:
    landmarks = normalized_landmark_features(
        landmarks_xyz,
        feature_kind=feature_kind,
    )
    if feature_kind not in (
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
    ):
        return landmarks, {"blendshapes_enabled": False}
    blendshapes = validated_blendshape_features(
        blendshape_names,
        blendshape_scores,
    )
    components = [landmarks, blendshapes]
    stats = {
        "blendshapes_enabled": True,
        "blendshape": blendshape_provenance(),
    }
    if feature_kind == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9:
        matrix = np.asarray(facial_transformation_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Structured GNM pose features require a finite 4x4 transform"
            )
        rotation = matrix[:3, :3]
        gram_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if (
            not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-3)
            or gram_error > 1e-2
            or not 0.99 <= determinant <= 1.01
        ):
            raise ValueError("Structured GNM pose transform is not rigid")
        components.append(rotation.reshape(-1).astype(np.float32))
        stats["pose_transform_enabled"] = True
        stats["pose_rotation_determinant"] = determinant
        stats["pose_rotation_gram_error"] = gram_error
    else:
        stats["pose_transform_enabled"] = False
    combined = np.concatenate(components).astype(np.float32)
    return combined, stats


def audit_identity_splits(rows: list[dict]) -> dict:
    by_identity: dict[str, set[str]] = {}
    for row in rows:
        by_identity.setdefault(str(row["identity_group"]), set()).add(
            str(row["split"])
        )
    conflicting = {
        identity: sorted(splits)
        for identity, splits in by_identity.items()
        if len(splits) != 1
    }
    groups = {
        split: {
            identity
            for identity, splits in by_identity.items()
            if splits == {split}
        }
        for split in ("train", "validation", "sealed")
    }
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_sealed": sorted(groups["train"] & groups["sealed"]),
        "validation_sealed": sorted(
            groups["validation"] & groups["sealed"]
        ),
    }
    missing_splits = [split for split, values in groups.items() if not values]
    passed = not conflicting and not missing_splits and not any(overlaps.values())
    return {
        "passed": passed,
        "identity_counts": {
            split: len(values) for split, values in groups.items()
        },
        "conflicting_identities": conflicting,
        "overlaps": overlaps,
        "missing_splits": missing_splits,
    }


def _row_signature(row: dict) -> str:
    payload = {
        "row_id": row.get("row_id"),
        "split": row.get("split"),
        "identity_group": row.get("identity_group"),
        "expression": row.get("expression"),
        "spec": row.get("spec"),
        "source_sha256": (row.get("source") or {}).get("sha256"),
        "selection_sha256": (row.get("selection_mask") or {}).get("sha256"),
        "exact_depth_sha256": (row.get("exact_depth") or {}).get("sha256"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_corpus_rows(corpus_roots: list[Path]) -> tuple[list[tuple[Path, dict]], list[dict]]:
    unique: dict[str, tuple[Path, dict, str]] = {}
    sources = []
    for root in corpus_roots:
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("provider") != "google-gnm-head-v3":
            raise ValueError(f"{root} is not a GNM v3 corpus")
        if summary.get("source_revision") != GNM_SOURCE_REVISION:
            raise ValueError(f"{root} uses an unpinned GNM source revision")
        if summary.get("source_geometry_training_and_evaluation_only") is not True:
            raise ValueError(f"{root} lacks the source-geometry usage contract")
        sources.append(
            {
                "path": str(root),
                "summary_sha256": _sha256(summary_path),
                "row_count": int(summary.get("row_count", -1)),
                "schema_version": summary.get("schema_version"),
                "selection_strategy": summary.get("selection_strategy"),
            }
        )
        for row in summary.get("rows") or []:
            row_id = str(row["row_id"])
            signature = _row_signature(row)
            existing = unique.get(row_id)
            if existing is not None and existing[2] != signature:
                raise ValueError(f"Conflicting duplicate GNM row: {row_id}")
            if existing is None:
                unique[row_id] = (root, row, signature)
    rows = [(root, row) for root, row, _signature in unique.values()]
    rows.sort(key=lambda item: str(item[1]["row_id"]))
    split_audit = audit_identity_splits([row for _root, row in rows])
    if not split_audit["passed"]:
        raise ValueError(
            "GNM structured corpus identity split audit failed: "
            + json.dumps(split_audit, sort_keys=True)
        )
    return rows, sources


def _prepare_rows(
    corpus_rows: list[tuple[Path, dict]],
    *,
    model,
    asset_paths: dict[str, Path],
    landmark_feature_kind: str,
) -> tuple[list[PreparedRow], list[dict]]:
    prepared = []
    excluded = []
    request_blendshapes = (
        landmark_feature_kind
        in (
            LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
            LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
        )
    )
    request_pose_transform = bool(
        landmark_feature_kind == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9
    )
    for root, row in corpus_rows:
        row_id = str(row["row_id"])
        try:
            source = Image.open(root / row["source"]["path"]).convert("RGB")
            selection = Image.open(
                root / row["selection_mask"]["path"]
            ).convert("L")
            selected = selected_image_from_exact_mask(source, selection)
            region, detector = _detect_production_region(
                np.asarray(selected),
                np.asarray(selection),
                output_face_blendshapes=request_blendshapes,
                output_facial_transformation_matrixes=request_pose_transform,
            )
            landmarks = np.asarray(region.get("landmarks_xyz"), dtype=np.float32)
            crop_box = _padded_box(
                region["bbox"],
                selected.width,
                selected.height,
                0.35,
            )
            crop = selected.crop(crop_box).convert("RGB")
            landmark_features, _feature_stats = structured_face_features(
                landmarks,
                feature_kind=landmark_feature_kind,
                blendshape_names=region.get("blendshape_names"),
                blendshape_scores=region.get("blendshape_scores"),
                facial_transformation_matrix=region.get(
                    "facial_transformation_matrix"
                ),
            )
            spec = GNMTrainingSceneSpec(**row["spec"])
            identity, expression = decode_geometry_targets(
                spec,
                model=model,
                asset_paths=asset_paths,
            )
            prepared.append(
                PreparedRow(
                    row_id=row_id,
                    split=str(row["split"]),
                    identity_group=str(row["identity_group"]),
                    expression_name=str(row["expression"]),
                    face_height_pixels=int(
                        row["render"]["face_bbox_height_pixels"]
                    ),
                    crop=crop,
                    landmark_features=landmark_features,
                    identity_target=identity,
                    expression_target=expression,
                    detector_scope=str(detector.get("scope") or "unknown"),
                    detector_name=str(region.get("detector") or "unknown"),
                )
            )
        except Exception as exc:
            excluded.append(
                {
                    "row_id": row_id,
                    "split": str(row.get("split")),
                    "error_type": type(exc).__name__,
                }
            )
    if not prepared:
        raise RuntimeError("Structured GNM preprocessing produced no rows")
    return prepared, excluded


def _extract_embeddings(
    rows: list[PreparedRow],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    snapshot = Path(
        snapshot_download(
            ENCODER_ID,
            revision=ENCODER_REVISION,
            local_files_only=True,
        )
    )
    model_path = snapshot / "model.safetensors"
    if _sha256(model_path) != ENCODER_MODEL_SHA256:
        raise ValueError("Depth Anything V2 Small checkpoint checksum mismatch")
    processor = AutoImageProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=False,
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    encoder = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    encoder.eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    processed: dict[tuple[int, ...], list[tuple[int, object]]] = {}
    for index, row in enumerate(rows):
        pixel_values = processor(
            images=row.crop,
            return_tensors="pt",
        )["pixel_values"]
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise ValueError(
                "Structured GNM crop processor must emit [1,C,H,W], "
                f"got {tuple(pixel_values.shape)}"
            )
        processed.setdefault(tuple(pixel_values.shape[1:]), []).append(
            (index, pixel_values)
        )
    embeddings: list[np.ndarray | None] = [None] * len(rows)
    started = time.perf_counter()
    for group in processed.values():
        for start in range(0, len(group), max(1, int(batch_size))):
            batch = group[start : start + max(1, int(batch_size))]
            pixel_values = torch.cat(
                [values for _index, values in batch],
                dim=0,
            ).to(device=device, dtype=dtype)
            with torch.inference_mode():
                outputs = encoder(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[-1].float()
            cls = hidden[:, 0]
            patch_mean = hidden[:, 1:].mean(dim=1)
            combined = torch.cat((cls, patch_mean), dim=1)
            combined = torch.nn.functional.normalize(combined, dim=1)
            values = combined.cpu().numpy().astype(np.float32)
            for batch_index, (source_index, _pixel_values) in enumerate(batch):
                embeddings[source_index] = values[batch_index]
    peak_vram = (
        float(torch.cuda.max_memory_reserved(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    if any(value is None for value in embeddings):
        raise RuntimeError("Structured GNM embedding inference lost a crop")
    values = np.stack(embeddings)
    if values.shape[1] != ENCODER_FEATURE_DIMENSION:
        raise ValueError(
            "Structured GNM encoder feature dimension changed: "
            f"{values.shape[1]} != {ENCODER_FEATURE_DIMENSION}"
        )
    return values, {
        "kind": FEATURE_ENCODER_DAV2_SMALL,
        "enabled": True,
        "model_id": ENCODER_ID,
        "revision": ENCODER_REVISION,
        "model_sha256": ENCODER_MODEL_SHA256,
        "license": ENCODER_LICENSE,
        "feature": "final-cls-plus-mean-patch-l2-normalized",
        "dimension": int(values.shape[1]),
        "batch_size": int(batch_size),
        "device": str(device),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_vram_gib": peak_vram,
    }


def _pca(values: np.ndarray, maximum_rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    centered = values - mean
    _u, _singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(maximum_rank), int(vh.shape[0]))
    return mean, vh[:rank]


def fit_ridge_head(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    target_mean: np.ndarray,
    components: np.ndarray,
    rank: int,
    alpha: float,
    sample_weights: np.ndarray,
) -> RidgeHead:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if len(features) != len(targets) or weights.shape != (len(features),):
        raise ValueError("Ridge head arrays have inconsistent row counts")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Ridge head sample weights must be positive and finite")
    weights = weights / np.sum(weights)
    feature_mean = np.sum(features * weights[:, None], axis=0)
    feature_scale = np.sqrt(
        np.sum(np.square(features - feature_mean) * weights[:, None], axis=0)
    )
    feature_scale = np.maximum(feature_scale, 1e-6)
    selected_components = np.asarray(components[: int(rank)], dtype=np.float64)
    if int(rank) == 0:
        return RidgeHead(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=np.asarray(target_mean, dtype=np.float64),
            components=selected_components,
            score_mean=np.zeros(0, dtype=np.float64),
            weights=np.zeros((features.shape[1], 0), dtype=np.float64),
            score_minimum=np.zeros(0, dtype=np.float64),
            score_maximum=np.zeros(0, dtype=np.float64),
        )
    scores = (targets - target_mean) @ selected_components.T
    score_mean = np.sum(scores * weights[:, None], axis=0)
    normalized = (features - feature_mean) / feature_scale
    centered_scores = scores - score_mean
    square_root = np.sqrt(weights * len(weights))
    weighted_features = normalized * square_root[:, None]
    weighted_scores = centered_scores * square_root[:, None]
    kernel = weighted_features @ weighted_features.T
    kernel.flat[:: len(kernel) + 1] += float(alpha)
    coefficients = np.linalg.solve(kernel, weighted_scores)
    regression = weighted_features.T @ coefficients
    score_low = np.min(scores, axis=0)
    score_high = np.max(scores, axis=0)
    score_margin = np.maximum((score_high - score_low) * 0.05, 1e-6)
    return RidgeHead(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=np.asarray(target_mean, dtype=np.float64),
        components=selected_components,
        score_mean=score_mean,
        weights=regression,
        score_minimum=score_low - score_margin,
        score_maximum=score_high + score_margin,
    )


def _balanced_weights(keys: list[str]) -> np.ndarray:
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)


def _part_masks(model, skin: np.ndarray) -> dict[str, np.ndarray]:
    masks = {}
    for part, groups in PART_GROUPS.items():
        weight = np.maximum.reduce(
            [np.asarray(model.vertex_group(group)) for group in groups]
        )
        mask = skin & (weight >= 0.10)
        if np.count_nonzero(mask) < 8:
            raise ValueError(f"GNM part {part!r} has insufficient skin vertices")
        masks[part] = mask[skin]
    return masks


def _geometry_summary(
    predicted: np.ndarray,
    target: np.ndarray,
    row_indices: np.ndarray,
    rows: list[PreparedRow],
    part_masks: dict[str, np.ndarray],
    *,
    control: np.ndarray | None = None,
) -> dict:
    selected_predicted = predicted[row_indices]
    selected_target = target[row_indices]
    selected_control = control[row_indices] if control is not None else None
    row_records = []
    for local_index, global_index in enumerate(row_indices):
        expected = selected_target[local_index]
        candidate = selected_predicted[local_index]
        diagonal = float(np.linalg.norm(np.ptp(expected, axis=0)))
        diagonal = max(diagonal, 1e-8)
        difference = candidate - expected
        normalized_rmse = float(
            np.sqrt(np.mean(np.sum(np.square(difference), axis=1))) / diagonal
        )
        expected_z = expected[:, 2] - np.mean(expected[:, 2])
        candidate_z = candidate[:, 2] - np.mean(candidate[:, 2])
        denominator = float(np.linalg.norm(expected_z) * np.linalg.norm(candidate_z))
        z_correlation = (
            float(np.dot(expected_z, candidate_z) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        parts = {}
        for part, mask in part_masks.items():
            part_difference = difference[mask]
            part_expected_z = expected[mask, 2] - np.mean(expected[mask, 2])
            part_candidate_z = candidate[mask, 2] - np.mean(candidate[mask, 2])
            part_denominator = float(
                np.linalg.norm(part_expected_z) * np.linalg.norm(part_candidate_z)
            )
            parts[part] = {
                "normalized_rmse": float(
                    np.sqrt(np.mean(np.sum(np.square(part_difference), axis=1)))
                    / diagonal
                ),
                "z_correlation": (
                    float(
                        np.dot(part_expected_z, part_candidate_z)
                        / part_denominator
                    )
                    if part_denominator > 1e-12
                    else 0.0
                ),
            }
        record = {
            "row_id": rows[int(global_index)].row_id,
            "identity_group": rows[int(global_index)].identity_group,
            "face_height_pixels": rows[int(global_index)].face_height_pixels,
            "normalized_vertex_rmse": normalized_rmse,
            "z_shape_correlation": z_correlation,
            "parts": parts,
        }
        if selected_control is not None:
            control_difference = selected_control[local_index] - expected
            control_rmse = float(
                np.sqrt(
                    np.mean(np.sum(np.square(control_difference), axis=1))
                )
                / diagonal
            )
            record["control_normalized_vertex_rmse"] = control_rmse
            record["beats_control_rmse"] = normalized_rmse < control_rmse
        row_records.append(record)
    part_summary = {
        part: {
            "median_normalized_rmse": float(
                np.median([row["parts"][part]["normalized_rmse"] for row in row_records])
            ),
            "median_z_correlation": float(
                np.median([row["parts"][part]["z_correlation"] for row in row_records])
            ),
        }
        for part in PART_GROUPS
    }
    summary = {
        "row_count": len(row_records),
        "median_normalized_vertex_rmse": float(
            np.median([row["normalized_vertex_rmse"] for row in row_records])
        ),
        "median_z_shape_correlation": float(
            np.median([row["z_shape_correlation"] for row in row_records])
        ),
        "parts": part_summary,
        "rows": row_records,
    }
    if selected_control is not None:
        summary["paired_rmse_win_rate"] = float(
            np.mean([row["beats_control_rmse"] for row in row_records])
        )
    return summary


def _indices_for(
    rows: list[PreparedRow],
    split: str,
    *,
    small_only: bool,
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if row.split == split
            and (
                not small_only
                or SMALL_FACE_MINIMUM_PIXELS
                <= row.face_height_pixels
                <= SMALL_FACE_MAXIMUM_PIXELS
            )
        ],
        dtype=np.int64,
    )


def _candidate_vertices(
    identity_head: RidgeHead,
    expression_head: RidgeHead,
    features: np.ndarray,
    *,
    mean_vertices: np.ndarray,
    identity_vertex_modes: np.ndarray,
    expression_vertex_modes: np.ndarray,
) -> np.ndarray:
    result = np.broadcast_to(
        mean_vertices,
        (len(features), *mean_vertices.shape),
    ).copy()
    identity_scores = identity_head.predict_scores(features)
    expression_scores = expression_head.predict_scores(features)
    if identity_head.rank:
        result += np.einsum(
            "br,rvk->bvk",
            identity_scores,
            identity_vertex_modes[: identity_head.rank],
        )
    if expression_head.rank:
        result += np.einsum(
            "br,rvk->bvk",
            expression_scores,
            expression_vertex_modes[: expression_head.rank],
        )
    return result


def _save_checkpoint(
    output_dir: Path,
    identity_head: RidgeHead,
    expression_head: RidgeHead,
    *,
    selection: dict,
    landmark_feature_kind: str,
    feature_encoder_kind: str,
    feature_encoder_dimension: int,
) -> dict:
    path = output_dir / "structured_geometry_checkpoint.npz"
    arrays = {}
    for prefix, head in (
        ("identity", identity_head),
        ("expression", expression_head),
    ):
        arrays.update(
            {
                f"{prefix}_feature_mean": head.feature_mean.astype(np.float32),
                f"{prefix}_feature_scale": head.feature_scale.astype(np.float32),
                f"{prefix}_target_mean": head.target_mean.astype(np.float32),
                f"{prefix}_components": head.components.astype(np.float32),
                f"{prefix}_score_mean": head.score_mean.astype(np.float32),
                f"{prefix}_weights": head.weights.astype(np.float32),
                f"{prefix}_score_minimum": head.score_minimum.astype(np.float32),
                f"{prefix}_score_maximum": head.score_maximum.astype(np.float32),
            }
        )
    arrays["metadata_schema_version"] = np.asarray(
        CHECKPOINT_SCHEMA_VERSION,
        dtype=np.int32,
    )
    arrays["metadata_landmark_feature_kind"] = np.asarray(
        landmark_feature_kind
    )
    arrays["metadata_feature_dimension"] = np.asarray(
        identity_head.feature_mean.shape[0],
        dtype=np.int32,
    )
    arrays["metadata_feature_encoder_kind"] = np.asarray(
        feature_encoder_kind
    )
    arrays["metadata_feature_encoder_dimension"] = np.asarray(
        feature_encoder_dimension,
        dtype=np.int32,
    )
    encoder_metadata = (
        {
            "model_id": ENCODER_ID,
            "revision": ENCODER_REVISION,
            "model_sha256": ENCODER_MODEL_SHA256,
            "license": ENCODER_LICENSE,
        }
        if feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
        else {
            "model_id": "",
            "revision": "",
            "model_sha256": "",
            "license": "",
        }
    )
    arrays["metadata_feature_encoder_model_id"] = np.asarray(
        encoder_metadata["model_id"]
    )
    arrays["metadata_feature_encoder_revision"] = np.asarray(
        encoder_metadata["revision"]
    )
    arrays["metadata_feature_encoder_model_sha256"] = np.asarray(
        encoder_metadata["model_sha256"]
    )
    arrays["metadata_feature_encoder_license"] = np.asarray(
        encoder_metadata["license"]
    )
    arrays["metadata_mediapipe_version"] = np.asarray(
        package_version("mediapipe")
    )
    if landmark_feature_kind in (
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
        LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
    ):
        provenance = blendshape_provenance()
        arrays["metadata_face_landmarker_sha256"] = np.asarray(
            provenance["model_sha256"]
        )
        arrays["metadata_blendshape_names_sha256"] = np.asarray(
            provenance["names_sha256"]
        )
    np.savez(path, **arrays)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
        "identity_rank": int(identity_head.rank),
        "expression_rank": int(expression_head.rank),
        "ridge_alpha": float(selection["ridge_alpha"]),
        "landmark_feature_kind": landmark_feature_kind,
        "feature_encoder_kind": feature_encoder_kind,
        "feature_encoder_dimension": int(feature_encoder_dimension),
        "training_only": True,
    }


def train_and_evaluate(
    gnm_root: str | Path,
    corpus_roots: list[str | Path],
    output_dir: str | Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    landmark_feature_kind: str = LANDMARK_FEATURE_DLIB68,
    feature_encoder_kind: str = FEATURE_ENCODER_DAV2_SMALL,
    expression_only: bool = False,
) -> dict:
    started = time.perf_counter()
    gnm_root = Path(gnm_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if landmark_feature_kind not in LANDMARK_FEATURE_KINDS:
        raise ValueError(
            f"Unsupported structured GNM landmark feature kind: "
            f"{landmark_feature_kind}"
        )
    if feature_encoder_kind not in FEATURE_ENCODER_KINDS:
        raise ValueError(
            f"Unsupported structured GNM feature encoder kind: "
            f"{feature_encoder_kind}"
        )
    preflight = preflight_gnm_root(gnm_root)
    if not preflight["runnable"]:
        raise ValueError("Pinned GNM preflight failed")
    model = _load_gnm_model(gnm_root)
    assets = _asset_paths(gnm_root)
    loaded_rows, corpus_sources = _load_corpus_rows(
        [Path(root) for root in corpus_roots]
    )
    split_audit = audit_identity_splits([row for _root, row in loaded_rows])
    rows, excluded = _prepare_rows(
        loaded_rows,
        model=model,
        asset_paths=assets,
        landmark_feature_kind=landmark_feature_kind,
    )
    prepared_split_audit = audit_identity_splits(
        [
            {"identity_group": row.identity_group, "split": row.split}
            for row in rows
        ]
    )
    if not prepared_split_audit["passed"]:
        raise ValueError("Detector-clean rows broke the identity split contract")
    if feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL:
        embeddings, encoder = _extract_embeddings(
            rows,
            device=device,
            batch_size=batch_size,
        )
    else:
        embeddings = np.empty((len(rows), 0), dtype=np.float32)
        encoder = {
            "kind": FEATURE_ENCODER_NONE,
            "enabled": False,
            "feature": "none",
            "dimension": 0,
            "batch_size": 0,
            "device": "cpu",
            "runtime_seconds": 0.0,
            "peak_vram_gib": 0.0,
        }
    landmark_features = np.stack([row.landmark_features for row in rows])
    features = np.concatenate((embeddings, landmark_features), axis=1)
    identity_targets = np.stack(
        [row.identity_target[:IDENTITY_SKIN_DIMENSION] for row in rows]
    ).astype(np.float64)
    expression_targets = np.stack(
        [row.expression_target[:EXPRESSION_SKIN_DIMENSION] for row in rows]
    ).astype(np.float64)
    train_indices = _indices_for(rows, "train", small_only=False)
    validation_indices = _indices_for(rows, "validation", small_only=False)
    validation_small_indices = _indices_for(
        rows,
        "validation",
        small_only=True,
    )
    sealed_indices = _indices_for(rows, "sealed", small_only=False)
    sealed_small_indices = _indices_for(rows, "sealed", small_only=True)
    if min(
        len(train_indices),
        len(validation_indices),
        len(validation_small_indices),
        len(sealed_indices),
        len(sealed_small_indices),
    ) == 0:
        raise ValueError("Structured GNM proof lacks a required held-out slice")

    train_identity_first = {}
    for index in train_indices:
        train_identity_first.setdefault(rows[int(index)].identity_group, int(index))
    unique_identity_indices = np.asarray(
        list(train_identity_first.values()),
        dtype=np.int64,
    )
    expression_first = {}
    for index in train_indices:
        key = (rows[int(index)].identity_group, rows[int(index)].expression_name)
        expression_first.setdefault(key, int(index))
    unique_expression_indices = np.asarray(
        list(expression_first.values()),
        dtype=np.int64,
    )
    identity_mean, identity_components = _pca(
        identity_targets[unique_identity_indices],
        max(DEFAULT_IDENTITY_RANKS),
    )
    expression_mean, expression_components = _pca(
        expression_targets[unique_expression_indices],
        max(DEFAULT_EXPRESSION_RANKS),
    )

    skin = np.asarray(model.vertex_group("skin")) > 0.5
    template = np.asarray(model.template_vertex_positions, dtype=np.float64)[skin]
    identity_basis = np.asarray(
        model.vertex_identity_basis[:IDENTITY_SKIN_DIMENSION, skin],
        dtype=np.float64,
    )
    expression_basis = np.asarray(
        model.expression_basis[:EXPRESSION_SKIN_DIMENSION, skin],
        dtype=np.float64,
    )
    target_vertices = (
        template[None]
        + np.einsum("bi,ivk->bvk", identity_targets, identity_basis)
        + np.einsum("bi,ivk->bvk", expression_targets, expression_basis)
    )
    zero_vertices = np.broadcast_to(template, target_vertices.shape).copy()
    mean_vertices = (
        template
        + np.einsum("i,ivk->vk", identity_mean, identity_basis)
        + np.einsum("i,ivk->vk", expression_mean, expression_basis)
    )
    mean_control_vertices = np.broadcast_to(
        mean_vertices,
        target_vertices.shape,
    ).copy()
    identity_vertex_modes = np.einsum(
        "ri,ivk->rvk",
        identity_components,
        identity_basis,
    )
    expression_vertex_modes = np.einsum(
        "ri,ivk->rvk",
        expression_components,
        expression_basis,
    )
    part_masks = _part_masks(model, skin)
    train_features = features[train_indices]
    train_identity_targets = identity_targets[train_indices]
    train_expression_targets = expression_targets[train_indices]
    identity_weights = _balanced_weights(
        [rows[int(index)].identity_group for index in train_indices]
    )
    expression_weights = _balanced_weights(
        [
            f"{rows[int(index)].identity_group}:{rows[int(index)].expression_name}"
            for index in train_indices
        ]
    )

    mean_validation_small = _geometry_summary(
        mean_control_vertices,
        target_vertices,
        validation_small_indices,
        rows,
        part_masks,
        control=zero_vertices,
    )
    candidates = []
    fitted: dict[tuple[int, int, float], tuple[RidgeHead, RidgeHead]] = {}
    identity_ranks = (0,) if expression_only else DEFAULT_IDENTITY_RANKS
    for identity_rank in identity_ranks:
        if identity_rank > len(identity_components):
            continue
        for expression_rank in DEFAULT_EXPRESSION_RANKS:
            if expression_rank > len(expression_components):
                continue
            for alpha in DEFAULT_RIDGE_ALPHAS:
                identity_head = fit_ridge_head(
                    train_features,
                    train_identity_targets,
                    target_mean=identity_mean,
                    components=identity_components,
                    rank=identity_rank,
                    alpha=alpha,
                    sample_weights=identity_weights,
                )
                expression_head = fit_ridge_head(
                    train_features,
                    train_expression_targets,
                    target_mean=expression_mean,
                    components=expression_components,
                    rank=expression_rank,
                    alpha=alpha,
                    sample_weights=expression_weights,
                )
                predicted = _candidate_vertices(
                    identity_head,
                    expression_head,
                    features,
                    mean_vertices=mean_vertices,
                    identity_vertex_modes=identity_vertex_modes,
                    expression_vertex_modes=expression_vertex_modes,
                )
                summary = _geometry_summary(
                    predicted,
                    target_vertices,
                    validation_small_indices,
                    rows,
                    part_masks,
                    control=mean_control_vertices,
                )
                part_regressions = sum(
                    summary["parts"][part]["median_normalized_rmse"]
                    > mean_validation_small["parts"][part][
                        "median_normalized_rmse"
                    ]
                    * 1.02
                    for part in PART_GROUPS
                )
                record = {
                    "identity_rank": int(identity_rank),
                    "expression_rank": int(expression_rank),
                    "ridge_alpha": float(alpha),
                    "part_rmse_regressions": int(part_regressions),
                    "median_normalized_vertex_rmse": summary[
                        "median_normalized_vertex_rmse"
                    ],
                    "median_z_shape_correlation": summary[
                        "median_z_shape_correlation"
                    ],
                    "paired_rmse_win_rate": summary["paired_rmse_win_rate"],
                }
                record["selection_key"] = [
                    record["part_rmse_regressions"],
                    -record["paired_rmse_win_rate"],
                    record["median_normalized_vertex_rmse"],
                    -record["median_z_shape_correlation"],
                    identity_rank + expression_rank,
                    float(alpha),
                ]
                candidates.append(record)
                fitted[(identity_rank, expression_rank, float(alpha))] = (
                    identity_head,
                    expression_head,
                )
    selected = min(candidates, key=lambda item: tuple(item["selection_key"]))
    selected_heads = fitted[
        (
            selected["identity_rank"],
            selected["expression_rank"],
            selected["ridge_alpha"],
        )
    ]
    selected_vertices = _candidate_vertices(
        selected_heads[0],
        selected_heads[1],
        features,
        mean_vertices=mean_vertices,
        identity_vertex_modes=identity_vertex_modes,
        expression_vertex_modes=expression_vertex_modes,
    )

    evaluations = {}
    for split, indices in (
        ("validation_all", validation_indices),
        ("validation_small", validation_small_indices),
        ("sealed_all", sealed_indices),
        ("sealed_small", sealed_small_indices),
    ):
        evaluations[split] = {
            "zero_mean_control": _geometry_summary(
                zero_vertices,
                target_vertices,
                indices,
                rows,
                part_masks,
            ),
            "training_mean_control": _geometry_summary(
                mean_control_vertices,
                target_vertices,
                indices,
                rows,
                part_masks,
                control=zero_vertices,
            ),
            "structured_candidate": _geometry_summary(
                selected_vertices,
                target_vertices,
                indices,
                rows,
                part_masks,
                control=mean_control_vertices,
            ),
        }

    validation = evaluations["validation_small"]
    sealed = evaluations["sealed_small"]

    def no_part_regression(evaluation: dict) -> bool:
        control = evaluation["training_mean_control"]["parts"]
        candidate = evaluation["structured_candidate"]["parts"]
        return all(
            candidate[part]["median_normalized_rmse"]
            <= control[part]["median_normalized_rmse"] * 1.02
            for part in PART_GROUPS
        )

    excluded_counts = Counter(record["split"] for record in excluded)
    source_counts = Counter(str(row["split"]) for _root, row in loaded_rows)
    prepared_counts = Counter(row.split for row in rows)
    detector_coverage = {
        split: float(prepared_counts[split] / max(source_counts[split], 1))
        for split in ("train", "validation", "sealed")
    }
    checks = {
        "identity_splits_disjoint": bool(
            split_audit["passed"] and prepared_split_audit["passed"]
        ),
        "sealed_not_used_for_selection": True,
        "selected_model_is_image_conditioned": bool(
            selected["identity_rank"] + selected["expression_rank"] > 0
        ),
        "validation_small_rmse_improves_mean_by_2pct": bool(
            validation["structured_candidate"]["median_normalized_vertex_rmse"]
            <= validation["training_mean_control"][
                "median_normalized_vertex_rmse"
            ]
            * 0.98
        ),
        "validation_small_z_no_regression": bool(
            validation["structured_candidate"]["median_z_shape_correlation"]
            >= validation["training_mean_control"][
                "median_z_shape_correlation"
            ]
        ),
        "validation_small_parts_no_regression": no_part_regression(validation),
        "sealed_small_rmse_improves_mean_by_2pct": bool(
            sealed["structured_candidate"]["median_normalized_vertex_rmse"]
            <= sealed["training_mean_control"]["median_normalized_vertex_rmse"]
            * 0.98
        ),
        "sealed_small_z_no_regression": bool(
            sealed["structured_candidate"]["median_z_shape_correlation"]
            >= sealed["training_mean_control"]["median_z_shape_correlation"]
        ),
        "sealed_small_parts_no_regression": no_part_regression(sealed),
        "sealed_small_paired_rmse_win_rate_at_least_60pct": bool(
            sealed["structured_candidate"]["paired_rmse_win_rate"] >= 0.60
        ),
        "held_out_detector_coverage_at_least_90pct": bool(
            detector_coverage["validation"] >= 0.90
            and detector_coverage["sealed"] >= 0.90
        ),
    }
    checks["eligible_for_production_depth_gate"] = bool(all(checks.values()))
    checkpoint = _save_checkpoint(
        output_dir,
        selected_heads[0],
        selected_heads[1],
        selection=selected,
        landmark_feature_kind=landmark_feature_kind,
        feature_encoder_kind=feature_encoder_kind,
        feature_encoder_dimension=int(embeddings.shape[1]),
    )
    evidence = {
        "schema_version": 1,
        "method": "gnm-low-rank-identity-expression-geometry",
        "status": (
            "advance-to-production-depth-gate"
            if checks["eligible_for_production_depth_gate"]
            else "hold"
        ),
        "privacy": (
            "scan-derived public GNM coefficients and procedural scenes; "
            "no private images or raw targets in compact evidence"
        ),
        "source_geometry_training_and_evaluation_only": True,
        "gnm": {
            "revision": GNM_SOURCE_REVISION,
            "license": GNM_LICENSE,
            "preflight": preflight,
            "identity_target_dimension": GNM_IDENTITY_DIMENSION,
            "expression_target_dimension": GNM_EXPRESSION_DIMENSION,
            "trained_identity_skin_dimension": IDENTITY_SKIN_DIMENSION,
            "trained_expression_skin_dimension": EXPRESSION_SKIN_DIMENSION,
        },
        "encoder": encoder,
        "corpora": corpus_sources,
        "split_audit": split_audit,
        "prepared_split_audit": prepared_split_audit,
        "preprocessing": {
            "source_row_counts": dict(source_counts),
            "prepared_row_counts": dict(prepared_counts),
            "excluded_row_counts": dict(excluded_counts),
            "excluded_rows": excluded,
            "detector_coverage": detector_coverage,
            "detector_scope_counts": dict(Counter(row.detector_scope for row in rows)),
            "detector_name_counts": dict(Counter(row.detector_name for row in rows)),
            "feature_dimension": int(features.shape[1]),
            "feature_encoder_kind": feature_encoder_kind,
            "feature_encoder_dimension": int(embeddings.shape[1]),
            "landmark_feature_dimension": int(landmark_features.shape[1]),
            "landmark_feature_kind": landmark_feature_kind,
            "blendshape_provenance": (
                blendshape_provenance()
                if landmark_feature_kind
                == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52
                or landmark_feature_kind
                == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9
                else {"enabled": False}
            ),
        },
        "model_selection": {
            "selection_split": "validation 55-100 px only",
            "sealed_split_used": False,
            "expression_only": bool(expression_only),
            "identity_rank_candidates": list(identity_ranks),
            "candidate_count": len(candidates),
            "selected": selected,
            "top_candidates": sorted(
                candidates,
                key=lambda item: tuple(item["selection_key"]),
            )[:10],
        },
        "checkpoint": checkpoint,
        "evaluations": evaluations,
        "decision": checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnm-root", required=True)
    parser.add_argument("--corpus-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--landmark-feature-kind",
        choices=LANDMARK_FEATURE_KINDS,
        default=LANDMARK_FEATURE_DLIB68,
    )
    parser.add_argument(
        "--feature-encoder-kind",
        choices=FEATURE_ENCODER_KINDS,
        default=FEATURE_ENCODER_DAV2_SMALL,
        help="Optional image encoder prepended to the structured face features",
    )
    parser.add_argument(
        "--expression-only",
        action="store_true",
        help="Fit only the GNM expression head; keep identity at training mean",
    )
    args = parser.parse_args()
    evidence = train_and_evaluate(
        args.gnm_root,
        args.corpus_root,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        landmark_feature_kind=args.landmark_feature_kind,
        feature_encoder_kind=args.feature_encoder_kind,
        expression_only=args.expression_only,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "decision": evidence["decision"],
                "selected": evidence["model_selection"]["selected"],
                "runtime_seconds": evidence["runtime_seconds"],
            },
            indent=2,
        )
    )
    if not evidence["decision"]["eligible_for_production_depth_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
