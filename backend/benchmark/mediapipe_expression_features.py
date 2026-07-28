"""Validation helpers for one-pass MediaPipe expression features."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from backend.face_depth_refinement import (
    FACE_BLENDSHAPE_NAMES,
    FACE_LANDMARKER_LICENSE,
    FACE_LANDMARKER_MODEL_SHA256,
)


BLENDSHAPE_NAMES = FACE_BLENDSHAPE_NAMES
BLENDSHAPE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(BLENDSHAPE_NAMES, separators=(",", ":")).encode("utf-8")
).hexdigest()


def validated_blendshape_features(
    names,
    scores,
) -> np.ndarray:
    ordered_names = tuple(str(name) for name in names or ())
    if ordered_names != BLENDSHAPE_NAMES:
        raise ValueError("MediaPipe blendshape category schema changed")
    features = np.asarray(scores, dtype=np.float32)
    if features.shape != (len(BLENDSHAPE_NAMES),):
        raise ValueError("MediaPipe blendshape feature count changed")
    if not np.all(np.isfinite(features)):
        raise ValueError("MediaPipe blendshape features must be finite")
    if np.any(features < 0.0) or np.any(features > 1.0):
        raise ValueError("MediaPipe blendshape scores must be within [0, 1]")
    return features


def blendshape_provenance() -> dict:
    return {
        "feature_count": len(BLENDSHAPE_NAMES),
        "names_sha256": BLENDSHAPE_SCHEMA_SHA256,
        "model_sha256": FACE_LANDMARKER_MODEL_SHA256,
        "license": FACE_LANDMARKER_LICENSE,
        "extraction": "same-pass-same-face-index-as-landmarks",
    }
