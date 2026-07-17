"""Candidate GNM provider driven by a learned identity/expression checkpoint."""

from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.train_face_surface_fusion_adapter import (
    selected_image_from_exact_mask,
)
from backend.benchmark.train_gnm_structured_geometry import (
    ENCODER_ID,
    ENCODER_FEATURE_DIMENSION,
    ENCODER_LICENSE,
    ENCODER_MODEL_SHA256,
    ENCODER_REVISION,
    CHECKPOINT_SCHEMA_VERSION,
    EXPRESSION_SKIN_DIMENSION,
    FEATURE_ENCODER_DAV2_SMALL,
    FEATURE_ENCODER_KINDS,
    FEATURE_ENCODER_NONE,
    IDENTITY_SKIN_DIMENSION,
    LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
    LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
    LANDMARK_FEATURE_DLIB68,
    LANDMARK_FEATURE_KINDS,
    landmark_feature_dimension,
    structured_face_features,
)
from backend.benchmark.mediapipe_expression_features import (
    blendshape_provenance,
)
from backend.gnm_face_foundation import (
    GNM_EXPRESSION_DIMENSION,
    GNM_IDENTITY_DIMENSION,
    GNMMeanFaceFoundation,
    GNMSurfaceCandidateSet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_arrays(checkpoint, prefix: str) -> dict[str, np.ndarray]:
    names = (
        "feature_mean",
        "feature_scale",
        "target_mean",
        "components",
        "score_mean",
        "weights",
        "score_minimum",
        "score_maximum",
    )
    arrays = {
        name: np.asarray(checkpoint[f"{prefix}_{name}"], dtype=np.float64)
        for name in names
    }
    feature_dimension = int(arrays["feature_mean"].shape[0])
    target_dimension = int(arrays["target_mean"].shape[0])
    rank = int(arrays["components"].shape[0])
    expected = {
        "feature_mean": (feature_dimension,),
        "feature_scale": (feature_dimension,),
        "target_mean": (target_dimension,),
        "components": (rank, target_dimension),
        "score_mean": (rank,),
        "weights": (feature_dimension, rank),
        "score_minimum": (rank,),
        "score_maximum": (rank,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"Structured GNM {prefix} checkpoint {name} has shape "
                f"{arrays[name].shape}, expected {shape}"
            )
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(
                f"Structured GNM {prefix} checkpoint {name} is non-finite"
            )
    if np.any(arrays["feature_scale"] <= 0.0):
        raise ValueError(
            f"Structured GNM {prefix} feature scale must be positive"
        )
    if np.any(arrays["score_minimum"] > arrays["score_maximum"]):
        raise ValueError(
            f"Structured GNM {prefix} score bounds are inverted"
        )
    return arrays


def predict_checkpoint_head(
    features: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.shape != arrays["feature_mean"].shape:
        raise ValueError("Structured GNM feature vector has an invalid shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("Structured GNM feature vector must be finite")
    normalized = (
        features - arrays["feature_mean"]
    ) / arrays["feature_scale"]
    scores = normalized @ arrays["weights"] + arrays["score_mean"]
    scores = np.clip(
        scores,
        arrays["score_minimum"],
        arrays["score_maximum"],
    )
    target = arrays["target_mean"] + scores @ arrays["components"]
    if not np.all(np.isfinite(target)):
        raise ValueError("Structured GNM predicted coefficients are non-finite")
    return target.astype(np.float32), scores.astype(np.float32)


class StructuredGNMFaceFoundation:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        device: str = "cuda",
        coefficient_strength: float = 1.0,
        identity_coefficient_strength: float | None = None,
        expression_coefficient_strength: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.identity_coefficient_strength = float(
            coefficient_strength
            if identity_coefficient_strength is None
            else identity_coefficient_strength
        )
        self.expression_coefficient_strength = float(
            coefficient_strength
            if expression_coefficient_strength is None
            else expression_coefficient_strength
        )
        for name, value in (
            ("identity", self.identity_coefficient_strength),
            ("expression", self.expression_coefficient_strength),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Structured GNM {name} coefficient strength must be "
                    "within [0, 1]"
                )
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        if self.checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("Structured GNM checkpoint checksum mismatch")
        with np.load(self.checkpoint_path, allow_pickle=False) as checkpoint:
            self.identity_head = _head_arrays(checkpoint, "identity")
            self.expression_head = _head_arrays(checkpoint, "expression")
            self.landmark_feature_kind = (
                str(checkpoint["metadata_landmark_feature_kind"].item())
                if "metadata_landmark_feature_kind" in checkpoint.files
                else LANDMARK_FEATURE_DLIB68
            )
            checkpoint_schema_version = (
                int(checkpoint["metadata_schema_version"].item())
                if "metadata_schema_version" in checkpoint.files
                else 1
            )
            if checkpoint_schema_version not in (
                1,
                2,
                CHECKPOINT_SCHEMA_VERSION,
            ):
                raise ValueError(
                    "Structured GNM checkpoint uses an unsupported schema: "
                    f"{checkpoint_schema_version}"
                )
            required_schema_fields = {
                "metadata_landmark_feature_kind",
                "metadata_feature_dimension",
                "metadata_mediapipe_version",
                "metadata_feature_encoder_kind",
                "metadata_feature_encoder_dimension",
                "metadata_feature_encoder_model_id",
                "metadata_feature_encoder_revision",
                "metadata_feature_encoder_model_sha256",
                "metadata_feature_encoder_license",
            }
            if checkpoint_schema_version == CHECKPOINT_SCHEMA_VERSION:
                missing_schema_fields = sorted(
                    required_schema_fields - set(checkpoint.files)
                )
                if missing_schema_fields:
                    raise ValueError(
                        "Structured GNM checkpoint lacks schema-v4 metadata: "
                        + ", ".join(missing_schema_fields)
                    )
            self.feature_encoder_kind = (
                str(checkpoint["metadata_feature_encoder_kind"].item())
                if "metadata_feature_encoder_kind" in checkpoint.files
                else FEATURE_ENCODER_DAV2_SMALL
            )
            checkpoint_feature_encoder_dimension = (
                int(
                    checkpoint[
                        "metadata_feature_encoder_dimension"
                    ].item()
                )
                if "metadata_feature_encoder_dimension" in checkpoint.files
                else ENCODER_FEATURE_DIMENSION
            )
            checkpoint_feature_encoder_model_id = (
                str(checkpoint["metadata_feature_encoder_model_id"].item())
                if "metadata_feature_encoder_model_id" in checkpoint.files
                else ENCODER_ID
            )
            checkpoint_feature_encoder_revision = (
                str(checkpoint["metadata_feature_encoder_revision"].item())
                if "metadata_feature_encoder_revision" in checkpoint.files
                else ENCODER_REVISION
            )
            checkpoint_feature_encoder_model_sha256 = (
                str(
                    checkpoint[
                        "metadata_feature_encoder_model_sha256"
                    ].item()
                )
                if "metadata_feature_encoder_model_sha256" in checkpoint.files
                else ENCODER_MODEL_SHA256
            )
            checkpoint_feature_encoder_license = (
                str(checkpoint["metadata_feature_encoder_license"].item())
                if "metadata_feature_encoder_license" in checkpoint.files
                else ENCODER_LICENSE
            )
            checkpoint_mediapipe_version = (
                str(checkpoint["metadata_mediapipe_version"].item())
                if "metadata_mediapipe_version" in checkpoint.files
                else None
            )
            checkpoint_feature_dimension = (
                int(checkpoint["metadata_feature_dimension"].item())
                if "metadata_feature_dimension" in checkpoint.files
                else int(self.identity_head["feature_mean"].shape[0])
            )
            checkpoint_face_landmarker_sha256 = (
                str(checkpoint["metadata_face_landmarker_sha256"].item())
                if "metadata_face_landmarker_sha256" in checkpoint.files
                else None
            )
            checkpoint_blendshape_names_sha256 = (
                str(checkpoint["metadata_blendshape_names_sha256"].item())
                if "metadata_blendshape_names_sha256" in checkpoint.files
                else None
            )
        if checkpoint_schema_version == CHECKPOINT_SCHEMA_VERSION and (
            self.feature_encoder_kind not in FEATURE_ENCODER_KINDS
        ):
            raise ValueError(
                "Structured GNM checkpoint uses an unsupported feature "
                f"encoder: {self.feature_encoder_kind}"
            )
        if checkpoint_schema_version in (1, 2):
            self.feature_encoder_kind = FEATURE_ENCODER_DAV2_SMALL
            checkpoint_feature_encoder_dimension = ENCODER_FEATURE_DIMENSION
            checkpoint_feature_encoder_model_id = ENCODER_ID
            checkpoint_feature_encoder_revision = ENCODER_REVISION
            checkpoint_feature_encoder_model_sha256 = ENCODER_MODEL_SHA256
            checkpoint_feature_encoder_license = ENCODER_LICENSE
        expected_encoder_dimension = (
            ENCODER_FEATURE_DIMENSION
            if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
            else 0
        )
        if checkpoint_feature_encoder_dimension != expected_encoder_dimension:
            raise ValueError(
                "Structured GNM feature encoder dimension changed: "
                f"{checkpoint_feature_encoder_dimension} != "
                f"{expected_encoder_dimension}"
            )
        expected_encoder_metadata = (
            (ENCODER_ID, ENCODER_REVISION, ENCODER_MODEL_SHA256, ENCODER_LICENSE)
            if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
            else ("", "", "", "")
        )
        actual_encoder_metadata = (
            checkpoint_feature_encoder_model_id,
            checkpoint_feature_encoder_revision,
            checkpoint_feature_encoder_model_sha256,
            checkpoint_feature_encoder_license,
        )
        if actual_encoder_metadata != expected_encoder_metadata:
            raise ValueError(
                "Structured GNM feature encoder provenance mismatch"
            )
        if (
            checkpoint_schema_version == CHECKPOINT_SCHEMA_VERSION
            and checkpoint_mediapipe_version != package_version("mediapipe")
        ):
            raise ValueError("Structured GNM MediaPipe version mismatch")
        if self.landmark_feature_kind not in LANDMARK_FEATURE_KINDS:
            raise ValueError(
                "Structured GNM checkpoint uses an unsupported landmark "
                f"feature kind: {self.landmark_feature_kind}"
            )
        if checkpoint_feature_dimension != int(
            self.identity_head["feature_mean"].shape[0]
        ):
            raise ValueError("Structured GNM checkpoint feature dimension changed")
        expected_feature_dimension = (
            checkpoint_feature_encoder_dimension
            + landmark_feature_dimension(self.landmark_feature_kind)
        )
        if checkpoint_feature_dimension != expected_feature_dimension:
            raise ValueError(
                "Structured GNM checkpoint feature composition changed: "
                f"{checkpoint_feature_dimension} != "
                f"{expected_feature_dimension}"
            )
        self.requires_face_blendshapes = bool(
            self.landmark_feature_kind
            in (
                LANDMARK_FEATURE_DLIB68_BLENDSHAPES52,
                LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9,
            )
        )
        self.requires_facial_transformation_matrix = bool(
            self.landmark_feature_kind
            == LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9
        )
        if self.requires_face_blendshapes:
            provenance = blendshape_provenance()
            if (
                checkpoint_face_landmarker_sha256
                != provenance["model_sha256"]
                or checkpoint_blendshape_names_sha256
                != provenance["names_sha256"]
            ):
                raise ValueError(
                    "Structured GNM blendshape checkpoint provenance mismatch"
                )
        if self.identity_head["target_mean"].shape != (
            IDENTITY_SKIN_DIMENSION,
        ):
            raise ValueError("Structured GNM identity target dimension changed")
        if self.expression_head["target_mean"].shape != (
            EXPRESSION_SKIN_DIMENSION,
        ):
            raise ValueError("Structured GNM expression target dimension changed")
        if self.identity_head["feature_mean"].shape != self.expression_head[
            "feature_mean"
        ].shape:
            raise ValueError("Structured GNM heads disagree on feature dimension")

        self.device = str(device)
        self.processor = None
        self.encoder = None
        self.dtype = None
        if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )

            snapshot = Path(
                snapshot_download(
                    ENCODER_ID,
                    revision=ENCODER_REVISION,
                    local_files_only=True,
                )
            )
            if _sha256(snapshot / "model.safetensors") != ENCODER_MODEL_SHA256:
                raise ValueError("Structured GNM encoder checksum mismatch")
            self.processor = AutoImageProcessor.from_pretrained(
                snapshot,
                local_files_only=True,
                use_fast=False,
            )
            self.dtype = (
                torch.float16
                if self.device.startswith("cuda")
                else torch.float32
            )
            self.encoder = AutoModelForDepthEstimation.from_pretrained(
                snapshot,
                local_files_only=True,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.encoder.eval()
        self.geometry = GNMMeanFaceFoundation(load_geometry_bases=True)

    def _embedding(self, face_image_rgb: np.ndarray, face_mask: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.feature_encoder_kind == FEATURE_ENCODER_NONE:
            return np.empty((0,), dtype=np.float32), {
                "runtime_seconds": 0.0,
                "peak_vram_gib": 0.0,
                "processor_shape": None,
            }
        import torch

        image = np.asarray(face_image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Structured GNM face image must be RGB")
        mask = cv2.resize(
            (np.asarray(face_mask) > 0).astype(np.uint8) * 255,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        selected = selected_image_from_exact_mask(
            Image.fromarray(image[:, :, :3]),
            Image.fromarray(mask),
        )
        pixel_values = self.processor(
            images=selected,
            return_tensors="pt",
        )["pixel_values"].to(device=self.device, dtype=self.dtype)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.encoder(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1].float()
        combined = torch.cat(
            (hidden[:, 0], hidden[:, 1:].mean(dim=1)),
            dim=1,
        )
        combined = torch.nn.functional.normalize(combined, dim=1)
        values = combined[0].cpu().numpy().astype(np.float32)
        peak_vram = (
            float(torch.cuda.max_memory_reserved(self.device) / (1024**3))
            if self.device.startswith("cuda")
            else 0.0
        )
        return values, {
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_vram_gib": peak_vram,
            "processor_shape": [int(value) for value in pixel_values.shape],
        }

    def fit_and_render_conditioned(
        self,
        media_pipe_landmarks_xy: np.ndarray,
        face_mask: np.ndarray,
        *,
        media_pipe_landmarks_xyz: np.ndarray,
        face_image_rgb: np.ndarray,
        media_pipe_blendshape_names=None,
        media_pipe_blendshape_scores=None,
        media_pipe_facial_transformation_matrix=None,
    ) -> tuple[np.ndarray, dict]:
        embedding, inference = self._embedding(face_image_rgb, face_mask)
        landmark_features, expression_features = structured_face_features(
            media_pipe_landmarks_xyz,
            feature_kind=self.landmark_feature_kind,
            blendshape_names=media_pipe_blendshape_names,
            blendshape_scores=media_pipe_blendshape_scores,
            facial_transformation_matrix=(
                media_pipe_facial_transformation_matrix
            ),
        )
        features = np.concatenate((embedding, landmark_features)).astype(
            np.float32
        )
        identity_skin, identity_scores = predict_checkpoint_head(
            features,
            self.identity_head,
        )
        expression_skin, expression_scores = predict_checkpoint_head(
            features,
            self.expression_head,
        )
        identity = np.zeros(GNM_IDENTITY_DIMENSION, dtype=np.float32)
        expression = np.zeros(GNM_EXPRESSION_DIMENSION, dtype=np.float32)
        identity[:IDENTITY_SKIN_DIMENSION] = (
            identity_skin * self.identity_coefficient_strength
        )
        expression[:EXPRESSION_SKIN_DIMENSION] = (
            expression_skin * self.expression_coefficient_strength
        )
        candidate_surface, candidate_stats = self.geometry.fit_and_render(
            media_pipe_landmarks_xy,
            face_mask,
            identity=identity,
            expression=expression,
        )
        fallback_surface, fallback_stats = self.geometry.fit_and_render(
            media_pipe_landmarks_xy,
            face_mask,
        )
        structured_stats = {
                "enabled": True,
                "method": (
                    "dav2-small-embedding-plus-mediapipe-low-rank-gnm"
                    if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
                    else "mediapipe-structured-only-low-rank-gnm"
                ),
                "checkpoint_sha256": self.checkpoint_sha256,
                "identity_rank": int(len(identity_scores)),
                "expression_rank": int(len(expression_scores)),
                "identity_coefficient_strength": (
                    self.identity_coefficient_strength
                ),
                "expression_coefficient_strength": (
                    self.expression_coefficient_strength
                ),
                "identity_score_l2": float(np.linalg.norm(identity_scores)),
                "expression_score_l2": float(np.linalg.norm(expression_scores)),
                "encoder_id": (
                    ENCODER_ID
                    if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
                    else None
                ),
                "encoder_revision": (
                    ENCODER_REVISION
                    if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
                    else None
                ),
                "encoder_sha256": (
                    ENCODER_MODEL_SHA256
                    if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
                    else None
                ),
                "encoder_license": (
                    ENCODER_LICENSE
                    if self.feature_encoder_kind == FEATURE_ENCODER_DAV2_SMALL
                    else None
                ),
                "feature_encoder_kind": self.feature_encoder_kind,
                "feature_dimension": int(len(features)),
                "landmark_feature_dimension": int(len(landmark_features)),
                "landmark_feature_kind": self.landmark_feature_kind,
                "expression_features": expression_features,
                **inference,
        }
        return GNMSurfaceCandidateSet(
            candidate_surface=candidate_surface,
            candidate_stats=candidate_stats,
            fallback_surface=fallback_surface,
            fallback_stats=fallback_stats,
        ), {"structured_geometry": structured_stats}
