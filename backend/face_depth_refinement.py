from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Callable
import urllib.request

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator


FACE_REFINEMENT_MODES = ("off", "auto", "on")
DEFAULT_FACE_PADDING_RATIO = 0.35
DEFAULT_FACE_FEATHER_RATIO = 0.20
DEFAULT_FACE_DETAIL_STRENGTH = 1.0
DEFAULT_FACE_MAX_CORRECTION_RATIO = 0.08
FACE_FEATURE_WEIGHT_FLOOR = 0.08
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
FACE_LANDMARKER_LICENSE = "Apache-2.0"
FACE_BLENDSHAPE_NAMES = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)
FACE_LANDMARKER_MODEL_MAX_BYTES = 16 * 1024 * 1024
YUNET_MODEL_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{YUNET_MODEL_REVISION}/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_MODEL_MAX_BYTES = 1024 * 1024
YUNET_SCORE_THRESHOLD = 0.90
YUNET_SELECTION_ROI_SCORE_THRESHOLD = 0.65
YUNET_MINIMUM_EYE_SEPARATION_RATIO = 0.18
YUNET_PROFILE_MINIMUM_EYE_SEPARATION_RATIO = 0.08
YUNET_MINIMUM_MOUTH_SEPARATION_RATIO = 0.12
YUNET_PROFILE_MINIMUM_MOUTH_SEPARATION_RATIO = 0.04
YUNET_MAX_INPUT_DIMENSION = 1024
DETECTOR_MODEL_PATH_ENVIRONMENT_NAMES = (
    "FACE_LANDMARKER_MODEL_PATH",
    "YUNET_FACE_DETECTOR_MODEL_PATH",
)
DEFAULT_MIN_FACE_PIXELS = 96
MIN_FACE_PIXELS_FLOOR = 48
MIN_FACE_IMAGE_RATIO = 0.25
SELECTION_FACE_COMPLETION_MINIMUM_RATIO = 5.0 / 6.0
SELECTION_FACE_MINIMUM_OVERLAP_RATIO = 0.50
SELECTION_ROI_DETECTION_DIMENSION = 384
SELECTION_ROI_PADDING_RATIO = 0.45
SELECTION_ROI_MINIMUM_COVERAGE = 0.001
SELECTION_ROI_MIN_FACE_COMPONENT_RATIO = 0.35
SELECTION_ROI_MAPPED_FACE_PIXELS_FLOOR = 24
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]
FACE_PART_INDEX_GROUPS = {
    "left_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "right_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    ],
    "nose": [168, 6, 197, 195, 5, 4, 1, 2, 98, 327],
}
FACE_PART_NAMES = tuple(FACE_PART_INDEX_GROUPS)
FACE_FEATURE_INDEX_GROUPS = list(FACE_PART_INDEX_GROUPS.values())
EYEWEAR_LANDMARK_INDICES = sorted(
    set(
        FACE_FEATURE_INDEX_GROUPS[0]
        + FACE_FEATURE_INDEX_GROUPS[1]
        + FACE_FEATURE_INDEX_GROUPS[2]
        + FACE_FEATURE_INDEX_GROUPS[3]
    )
)
EYEWEAR_MINIMUM_DARK_COVERAGE = 0.40
EYEWEAR_MINIMUM_COMPONENT_COVERAGE = 0.45
EYEWEAR_MINIMUM_COMPONENT_WIDTH_RATIO = 0.60
EYEWEAR_ACCESSORY_RESIDUAL_RATIO = 0.02
EYEWEAR_MAXIMUM_CORRECTION_RATIO = 0.15
EYEWEAR_SHAPE_PRIOR_MIN_CONFIDENCE = 0.70
EYEWEAR_SHAPE_PRIOR_CORRECTION_LIMIT_SCALE = 0.75


def _clamp_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(float(value))) for value in box)
    x0 = min(max(x0, 0), max(width - 1, 0))
    y0 = min(max(y0, 0), max(height - 1, 0))
    x1 = min(max(x1, x0 + 1), width)
    y1 = min(max(y1, y0 + 1), height)
    return x0, y0, x1, y1


def _padded_box(box, width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_box(box, width, height)
    pad_x = int(round((x1 - x0) * max(0.0, float(padding_ratio))))
    pad_y = int(round((y1 - y0) * max(0.0, float(padding_ratio))))
    return _clamp_box((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), width, height)


def face_masks_from_box(
    image_shape: tuple[int, int] | tuple[int, int, int],
    box,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a conservative face oval and feature mask for detector fallbacks."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = _clamp_box(box, width, height)
    face_width = max(1, x1 - x0)
    face_height = max(1, y1 - y0)
    center = (int(round(x0 + face_width * 0.5)), int(round(y0 + face_height * 0.52)))
    axes = (max(1, int(round(face_width * 0.47))), max(1, int(round(face_height * 0.51))))

    face_mask = np.zeros((height, width), dtype=np.uint8)
    feature_mask = np.zeros_like(face_mask)
    cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)

    def ellipse(cx, cy, ax, ay) -> None:
        cv2.ellipse(
            feature_mask,
            (int(round(x0 + face_width * cx)), int(round(y0 + face_height * cy))),
            (max(1, int(round(face_width * ax))), max(1, int(round(face_height * ay)))),
            0,
            0,
            360,
            255,
            -1,
        )

    ellipse(0.31, 0.40, 0.18, 0.10)
    ellipse(0.69, 0.40, 0.18, 0.10)
    ellipse(0.50, 0.58, 0.13, 0.21)
    ellipse(0.50, 0.77, 0.25, 0.10)
    _add_face_accessory_support(face_mask, feature_mask, (x0, y0, x1, y1))
    feature_mask = cv2.bitwise_and(feature_mask, face_mask)
    return face_mask, feature_mask


def _add_face_accessory_support(
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    box,
) -> None:
    """Extend a face oval to ears, temples, and common eyewear coverage."""
    height, width = face_mask.shape
    x0, y0, x1, y1 = _clamp_box(box, width, height)
    face_width = max(1, x1 - x0)
    face_height = max(1, y1 - y0)

    ear_axes = (
        max(2, int(round(face_width * 0.10))),
        max(2, int(round(face_height * 0.19))),
    )
    ear_y = int(round(y0 + face_height * 0.55))
    for ear_x in (
        int(round(x0 - face_width * 0.015)),
        int(round(x1 + face_width * 0.015)),
    ):
        center = (int(np.clip(ear_x, 0, width - 1)), int(np.clip(ear_y, 0, height - 1)))
        cv2.ellipse(face_mask, center, ear_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(feature_mask, center, ear_axes, 0, 0, 360, 255, -1)

    lens_axes = (
        max(2, int(round(face_width * 0.25))),
        max(2, int(round(face_height * 0.13))),
    )
    eyewear_y = int(round(y0 + face_height * 0.36))
    lens_centers = []
    for ratio in (0.27, 0.73):
        center = (int(round(x0 + face_width * ratio)), eyewear_y)
        lens_centers.append(center)
        cv2.ellipse(face_mask, center, lens_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(feature_mask, center, lens_axes, 0, 0, 360, 255, -1)
    bridge_half_height = max(1, int(round(face_height * 0.035)))
    cv2.rectangle(
        feature_mask,
        (lens_centers[0][0], eyewear_y - bridge_half_height),
        (lens_centers[1][0], eyewear_y + bridge_half_height),
        255,
        -1,
    )


def _connection_indices(connections) -> list[int]:
    indices = set()
    for connection in connections or ():
        try:
            start, end = connection
        except (TypeError, ValueError):
            start = getattr(connection, "start", None)
            end = getattr(connection, "end", None)
        if start is not None:
            indices.add(int(start))
        if end is not None:
            indices.add(int(end))
    return sorted(indices)


def _fill_landmark_region(mask: np.ndarray, points: np.ndarray, indices: list[int]) -> None:
    selected = points[indices] if indices and max(indices) < len(points) else np.empty((0, 2), dtype=np.int32)
    if len(selected) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(selected.astype(np.int32)), 255)


def _landmark_part_masks(
    points: np.ndarray,
    image_shape,
    index_groups: dict[str, list[int]] | None = None,
) -> dict[str, np.ndarray]:
    height, width = int(image_shape[0]), int(image_shape[1])
    groups = index_groups or FACE_PART_INDEX_GROUPS
    masks = {}
    for name in FACE_PART_NAMES:
        mask = np.zeros((height, width), dtype=np.uint8)
        _fill_landmark_region(mask, points, list(groups.get(name, ())))
        masks[name] = mask
    return masks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_verified_model(
    *,
    environment_name: str,
    cache_name: str,
    url: str,
    expected_sha256: str,
    maximum_bytes: int,
) -> Path:
    configured = str(os.getenv(environment_name) or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_file():
            raise FileNotFoundError(
                f"Configured face model is unavailable: {environment_name}"
            )
        try:
            configured_size = configured_path.stat().st_size
            configured_sha256 = _sha256_file(configured_path)
        except OSError as exc:
            raise RuntimeError(
                f"Configured face model could not be verified: {environment_name}"
            ) from exc
        if configured_size > int(maximum_bytes):
            raise RuntimeError(
                f"Configured face model exceeds the size limit: {environment_name}"
            )
        if configured_sha256 != expected_sha256:
            raise RuntimeError(
                f"Configured face model checksum mismatch: {environment_name}"
            )
        return configured_path

    cache_path = Path.home() / ".cache" / "3dprintpic" / cache_name
    if cache_path.is_file() and _sha256_file(cache_path) == expected_sha256:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f"{cache_path.name}.",
            suffix=".part",
            delete=False,
        ) as partial_file:
            partial_path = Path(partial_file.name)
            total_bytes = 0
            with urllib.request.urlopen(url, timeout=30) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total_bytes += len(block)
                    if total_bytes > int(maximum_bytes):
                        raise RuntimeError(
                            f"Model download exceeded {int(maximum_bytes)} bytes: {url}"
                        )
                    partial_file.write(block)
        actual_sha256 = _sha256_file(partial_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Face model checksum mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        os.replace(partial_path, cache_path)
        partial_path = None
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
    return cache_path


def _resolve_face_landmarker_model() -> Path:
    return _resolve_verified_model(
        environment_name="FACE_LANDMARKER_MODEL_PATH",
        cache_name="face_landmarker.task",
        url=FACE_LANDMARKER_MODEL_URL,
        expected_sha256=FACE_LANDMARKER_MODEL_SHA256,
        maximum_bytes=FACE_LANDMARKER_MODEL_MAX_BYTES,
    )


def _resolve_yunet_model() -> Path:
    return _resolve_verified_model(
        environment_name="YUNET_FACE_DETECTOR_MODEL_PATH",
        cache_name="face_detection_yunet_2023mar.onnx",
        url=YUNET_MODEL_URL,
        expected_sha256=YUNET_MODEL_SHA256,
        maximum_bytes=YUNET_MODEL_MAX_BYTES,
    )


def _detector_error_record(detector: str, exc: Exception) -> str:
    """Keep detector diagnostics useful without exposing configured model paths."""
    message = str(exc)
    for environment_name in DETECTOR_MODEL_PATH_ENVIRONMENT_NAMES:
        configured = str(os.getenv(environment_name) or "").strip()
        if not configured:
            continue
        configured_path = Path(configured).expanduser()
        variants = {
            configured,
            str(configured_path),
            str(configured_path.absolute()),
        }
        variants.update(value.replace("\\", "/") for value in tuple(variants))
        for value in sorted(variants, key=len, reverse=True):
            if value:
                message = re.sub(
                    re.escape(value),
                    "<configured-model-path>",
                    message,
                    flags=re.IGNORECASE,
                )
    suffix = f":{message}" if message else ""
    return f"{detector}:{type(exc).__name__}{suffix}"


def _landmark_region(
    points: np.ndarray,
    image_shape,
    *,
    detector_name: str,
    oval_indices: list[int] | None = None,
    feature_index_groups: list[list[int]] | None = None,
    part_index_groups: dict[str, list[int]] | None = None,
) -> dict | None:
    height, width = int(image_shape[0]), int(image_shape[1])
    if not len(points):
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0) + 1
    face_mask = np.zeros((height, width), dtype=np.uint8)
    feature_mask = np.zeros_like(face_mask)
    _fill_landmark_region(face_mask, points, oval_indices or FACE_OVAL_INDICES)
    for indices in feature_index_groups or FACE_FEATURE_INDEX_GROUPS:
        _fill_landmark_region(feature_mask, points, indices)
    part_masks = _landmark_part_masks(points, image_shape, part_index_groups)
    if not np.any(face_mask):
        face_mask, fallback_features = face_masks_from_box(image_shape, (x0, y0, x1, y1))
        feature_mask = np.maximum(feature_mask, fallback_features)
    elif not np.any(feature_mask):
        _, feature_mask = face_masks_from_box(image_shape, (x0, y0, x1, y1))
    _add_face_accessory_support(face_mask, feature_mask, (x0, y0, x1, y1))
    return {
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "face_mask": face_mask,
        "feature_mask": cv2.bitwise_and(feature_mask, face_mask),
        "part_masks": {
            name: cv2.bitwise_and(mask, face_mask)
            for name, mask in part_masks.items()
        },
        "detector": detector_name,
        "landmark_count": int(len(points)),
    }


def _validated_face_transformation_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("MediaPipe face transformation matrix is invalid")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-3):
        raise ValueError("MediaPipe face transformation matrix is not affine")
    rotation = matrix[:3, :3]
    gram_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    determinant = float(np.linalg.det(rotation))
    if gram_error > 1e-2 or not 0.99 <= determinant <= 1.01:
        raise ValueError("MediaPipe face transformation rotation is invalid")
    return matrix.astype(np.float32)


def _detect_faces_mediapipe_tasks(
    image_rgb: np.ndarray,
    max_faces: int,
    min_face_pixels: int,
    *,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
) -> list[dict]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_resolve_face_landmarker_model())),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=bool(output_face_blendshapes),
        output_facial_transformation_matrixes=bool(
            output_facial_transformation_matrixes
        ),
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))

    height, width = image_rgb.shape[:2]
    result_landmarks = list(result.face_landmarks or ())
    result_blendshapes = list(result.face_blendshapes or ())
    result_transforms = list(
        getattr(result, "facial_transformation_matrixes", None) or ()
    )
    if output_face_blendshapes and len(result_blendshapes) != len(result_landmarks):
        raise ValueError(
            "MediaPipe face landmark/blendshape result counts disagree"
        )
    if output_facial_transformation_matrixes and len(result_transforms) != len(
        result_landmarks
    ):
        raise ValueError(
            "MediaPipe face landmark/transformation result counts disagree"
        )
    regions = []
    for face_index, landmarks in enumerate(result_landmarks):
        landmarks_xyz = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks],
            dtype=np.float32,
        )
        points = np.asarray(
            [
                [
                    np.clip(round(landmark.x * (width - 1)), 0, width - 1),
                    np.clip(round(landmark.y * (height - 1)), 0, height - 1),
                ]
                for landmark in landmarks
            ],
            dtype=np.int32,
        )
        region = _landmark_region(points, image_rgb.shape, detector_name="mediapipe-face-landmarker")
        if region is None:
            continue
        region["landmarks_xyz"] = landmarks_xyz
        if output_facial_transformation_matrixes:
            region["facial_transformation_matrix"] = (
                _validated_face_transformation_matrix(
                    result_transforms[face_index]
                )
            )
        if output_face_blendshapes:
            categories = {
                str(category.category_name): float(category.score)
                for category in result_blendshapes[face_index]
            }
            if tuple(sorted(categories)) != FACE_BLENDSHAPE_NAMES:
                raise ValueError("MediaPipe blendshape category schema changed")
            scores = [categories[name] for name in FACE_BLENDSHAPE_NAMES]
            if not np.all(np.isfinite(scores)):
                raise ValueError("MediaPipe blendshape scores are non-finite")
            region["blendshape_names"] = list(FACE_BLENDSHAPE_NAMES)
            region["blendshape_scores"] = scores
        x0, y0, x1, y1 = region["bbox"]
        if min(x1 - x0, y1 - y0) >= min_face_pixels:
            regions.append(region)
    return regions


def _detect_faces_mediapipe(
    image_rgb: np.ndarray,
    max_faces: int,
    min_face_pixels: int,
    *,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
) -> list[dict]:
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    face_mesh_api = getattr(solutions, "face_mesh", None) if solutions is not None else None
    if (
        face_mesh_api is None
        or output_face_blendshapes
        or output_facial_transformation_matrixes
    ):
        task_kwargs = {}
        if output_face_blendshapes:
            task_kwargs["output_face_blendshapes"] = True
        if output_facial_transformation_matrixes:
            task_kwargs["output_facial_transformation_matrixes"] = True
        return _detect_faces_mediapipe_tasks(
            image_rgb,
            max_faces,
            min_face_pixels,
            **task_kwargs,
        )

    height, width = image_rgb.shape[:2]
    regions = []
    with face_mesh_api.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as detector:
        result = detector.process(image_rgb)

    oval_indices = _connection_indices(getattr(face_mesh_api, "FACEMESH_FACE_OVAL", ()))
    part_connections = {
        "left_eye": getattr(face_mesh_api, "FACEMESH_LEFT_EYE", ()),
        "right_eye": getattr(face_mesh_api, "FACEMESH_RIGHT_EYE", ()),
        "left_eyebrow": getattr(face_mesh_api, "FACEMESH_LEFT_EYEBROW", ()),
        "right_eyebrow": getattr(face_mesh_api, "FACEMESH_RIGHT_EYEBROW", ()),
        "mouth": getattr(face_mesh_api, "FACEMESH_LIPS", ()),
        "nose": getattr(face_mesh_api, "FACEMESH_NOSE", ()),
    }
    part_index_groups = {
        name: _connection_indices(connections)
        for name, connections in part_connections.items()
    }
    for landmarks in result.multi_face_landmarks or ():
        landmarks_xyz = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark],
            dtype=np.float32,
        )
        points = np.asarray(
            [
                [
                    np.clip(round(landmark.x * (width - 1)), 0, width - 1),
                    np.clip(round(landmark.y * (height - 1)), 0, height - 1),
                ]
                for landmark in landmarks.landmark
            ],
            dtype=np.int32,
        )
        if not len(points):
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0) + 1
        if min(x1 - x0, y1 - y0) < min_face_pixels:
            continue
        region = _landmark_region(
            points,
            image_rgb.shape,
            detector_name="mediapipe-face-mesh",
            oval_indices=oval_indices or list(range(len(points))),
            feature_index_groups=list(part_index_groups.values()),
            part_index_groups=part_index_groups,
        )
        if region is not None:
            region["landmarks_xyz"] = landmarks_xyz
            regions.append(region)
    return regions


def _effective_min_face_pixels(image_shape, requested: int) -> int:
    height, width = int(image_shape[0]), int(image_shape[1])
    short_edge = max(1, min(height, width))
    adaptive_cap = max(
        MIN_FACE_PIXELS_FLOOR,
        int(round(short_edge * MIN_FACE_IMAGE_RATIO)),
    )
    return max(1, min(int(requested), adaptive_cap))


def _face_region_selection_overlap(region: dict, roi_mask: np.ndarray) -> float:
    face_mask = np.asarray(region.get("face_mask"), dtype=np.uint8)
    selected = np.asarray(roi_mask, dtype=bool)
    if face_mask.shape != selected.shape:
        return 0.0
    face = face_mask > 0
    face_pixels = int(np.count_nonzero(face))
    if face_pixels <= 0:
        return 0.0
    return float(np.count_nonzero(face & selected) / face_pixels)


def _face_region_iou(left: dict, right: dict) -> float:
    lx0, ly0, lx1, ly1 = (float(value) for value in left["bbox"])
    rx0, ry0, rx1, ry1 = (float(value) for value in right["bbox"])
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0,
        min(ly1, ry1) - max(ly0, ry0),
    )
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _merge_selected_face_regions(
    regions: list[dict],
    candidates: list[dict],
    roi_mask: np.ndarray,
    *,
    max_faces: int,
) -> tuple[list[dict], int]:
    merged = list(regions)
    accepted = 0
    for candidate in candidates:
        overlap = _face_region_selection_overlap(candidate, roi_mask)
        if overlap < SELECTION_FACE_MINIMUM_OVERLAP_RATIO:
            continue
        candidate = dict(candidate)
        candidate["selection_overlap_ratio"] = overlap
        if any(_face_region_iou(candidate, existing) >= 0.50 for existing in merged):
            continue
        merged.append(candidate)
        accepted += 1
        if len(merged) >= max(1, int(max_faces)):
            break
    return merged, accepted


def _yunet_keypoints_are_face_like(
    keypoints: np.ndarray,
    box,
    *,
    minimum_eye_separation_ratio: float = YUNET_MINIMUM_EYE_SEPARATION_RATIO,
    minimum_mouth_separation_ratio: float = (
        YUNET_MINIMUM_MOUTH_SEPARATION_RATIO
    ),
) -> bool:
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        return False
    x0, y0, x1, y1 = (float(value) for value in box)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    margin_x = width * 0.15
    margin_y = height * 0.15
    if np.any(points[:, 0] < x0 - margin_x) or np.any(points[:, 0] > x1 + margin_x):
        return False
    if np.any(points[:, 1] < y0 - margin_y) or np.any(points[:, 1] > y1 + margin_y):
        return False

    eyes = points[:2]
    nose = points[2]
    mouth = points[3:]
    if abs(float(eyes[1, 0] - eyes[0, 0])) < (
        width * float(minimum_eye_separation_ratio)
    ):
        return False
    if abs(float(mouth[1, 0] - mouth[0, 0])) < (
        width * float(minimum_mouth_separation_ratio)
    ):
        return False
    eye_y = float(np.mean(eyes[:, 1]))
    mouth_y = float(np.mean(mouth[:, 1]))
    if mouth_y <= eye_y + height * 0.12:
        return False
    if float(nose[1]) < eye_y - height * 0.10:
        return False
    if float(nose[1]) > mouth_y + height * 0.10:
        return False
    return True


def _detect_faces_yunet(
    image_rgb: np.ndarray,
    max_faces: int,
    min_face_pixels: int,
    *,
    score_threshold: float = YUNET_SCORE_THRESHOLD,
    minimum_eye_separation_ratio: float = YUNET_MINIMUM_EYE_SEPARATION_RATIO,
    minimum_mouth_separation_ratio: float = (
        YUNET_MINIMUM_MOUTH_SEPARATION_RATIO
    ),
) -> list[dict]:
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV FaceDetectorYN is unavailable")
    score_threshold = float(score_threshold)
    minimum_eye_separation_ratio = float(minimum_eye_separation_ratio)
    minimum_mouth_separation_ratio = float(minimum_mouth_separation_ratio)
    if not 0.0 < score_threshold <= 1.0:
        raise ValueError("YuNet score threshold must be in (0, 1]")
    if not 0.0 < minimum_eye_separation_ratio <= 0.5:
        raise ValueError("YuNet eye-separation ratio must be in (0, 0.5]")
    if not 0.0 < minimum_mouth_separation_ratio <= 0.5:
        raise ValueError("YuNet mouth-separation ratio must be in (0, 0.5]")
    height, width = image_rgb.shape[:2]
    scale = min(1.0, float(YUNET_MAX_INPUT_DIMENSION) / max(height, width))
    if scale < 1.0:
        probe_width = max(1, int(round(width * scale)))
        probe_height = max(1, int(round(height * scale)))
        probe_rgb = cv2.resize(
            image_rgb,
            (probe_width, probe_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        probe_rgb = image_rgb
        probe_height, probe_width = height, width

    detector = cv2.FaceDetectorYN.create(
        str(_resolve_yunet_model()),
        "",
        (probe_width, probe_height),
        score_threshold,
        0.3,
        5000,
    )
    detector.setInputSize((probe_width, probe_height))
    _, detections = detector.detect(cv2.cvtColor(probe_rgb, cv2.COLOR_RGB2BGR))
    if detections is None:
        return []

    inverse_scale = 1.0 / scale
    regions = []
    for detection in np.asarray(detections, dtype=np.float64):
        if detection.size < 15 or not np.all(np.isfinite(detection[:15])):
            continue
        confidence = float(detection[14])
        if confidence < score_threshold:
            continue
        x, y, box_width, box_height = detection[:4] * inverse_scale
        box = _clamp_box((x, y, x + box_width, y + box_height), width, height)
        x0, y0, x1, y1 = box
        if min(x1 - x0, y1 - y0) < int(min_face_pixels):
            continue
        keypoints = detection[4:14].reshape(5, 2) * inverse_scale
        if not _yunet_keypoints_are_face_like(
            keypoints,
            box,
            minimum_eye_separation_ratio=minimum_eye_separation_ratio,
            minimum_mouth_separation_ratio=minimum_mouth_separation_ratio,
        ):
            continue
        face_mask, feature_mask = face_masks_from_box(image_rgb.shape, box)
        regions.append(
            {
                "bbox": list(box),
                "face_mask": face_mask,
                "feature_mask": feature_mask,
                "detector": "opencv-yunet-2023mar",
                "landmark_count": 5,
                "confidence": confidence,
                "keypoints": keypoints.round(3).tolist(),
                "score_threshold": score_threshold,
                "minimum_eye_separation_ratio": minimum_eye_separation_ratio,
                "minimum_mouth_separation_ratio": (
                    minimum_mouth_separation_ratio
                ),
                "model_revision": YUNET_MODEL_REVISION,
                "model_sha256": YUNET_MODEL_SHA256,
            }
        )
    regions.sort(
        key=lambda region: (
            float(region["confidence"]),
            (region["bbox"][2] - region["bbox"][0])
            * (region["bbox"][3] - region["bbox"][1]),
        ),
        reverse=True,
    )
    return regions[: max(1, int(max_faces))]


def _detect_faces_opencv(image_rgb: np.ndarray, max_faces: int, min_face_pixels: int) -> list[dict]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("OpenCV face detector cascade is unavailable")
    detections = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_face_pixels, min_face_pixels),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    ordered = sorted(detections, key=lambda item: int(item[2]) * int(item[3]), reverse=True)[:max_faces]
    regions = []
    for x, y, width, height in ordered:
        box = (int(x), int(y), int(x + width), int(y + height))
        face_mask, feature_mask = face_masks_from_box(image_rgb.shape, box)
        regions.append(
            {
                "bbox": list(box),
                "face_mask": face_mask,
                "feature_mask": feature_mask,
                "detector": "opencv-haar",
                "landmark_count": 0,
            }
        )
    return regions


def detect_face_regions(
    image_rgb: np.ndarray,
    *,
    max_faces: int = 3,
    min_face_pixels: int = DEFAULT_MIN_FACE_PIXELS,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
    yunet_score_threshold: float = YUNET_SCORE_THRESHOLD,
    yunet_minimum_eye_separation_ratio: float = (
        YUNET_MINIMUM_EYE_SEPARATION_RATIO
    ),
    yunet_minimum_mouth_separation_ratio: float = (
        YUNET_MINIMUM_MOUTH_SEPARATION_RATIO
    ),
) -> tuple[list[dict], list[str]]:
    min_face_pixels = _effective_min_face_pixels(image_rgb.shape, min_face_pixels)
    errors = []
    try:
        mediapipe_kwargs = {}
        if output_face_blendshapes:
            mediapipe_kwargs["output_face_blendshapes"] = True
        if output_facial_transformation_matrixes:
            mediapipe_kwargs["output_facial_transformation_matrixes"] = True
        regions = _detect_faces_mediapipe(
            image_rgb,
            max_faces,
            min_face_pixels,
            **mediapipe_kwargs,
        )
        if regions:
            return regions, errors
    except Exception as exc:
        errors.append(_detector_error_record("mediapipe", exc))
    try:
        regions = _detect_faces_yunet(
            image_rgb,
            max_faces,
            min_face_pixels,
            score_threshold=yunet_score_threshold,
            minimum_eye_separation_ratio=(
                yunet_minimum_eye_separation_ratio
            ),
            minimum_mouth_separation_ratio=(
                yunet_minimum_mouth_separation_ratio
            ),
        )
        if regions:
            upgraded = _upgrade_yunet_regions_with_mediapipe(
                image_rgb,
                regions,
                max_faces=max_faces,
                output_face_blendshapes=output_face_blendshapes,
                output_facial_transformation_matrixes=(
                    output_facial_transformation_matrixes
                ),
            )
            return upgraded or regions, errors
    except Exception as exc:
        errors.append(_detector_error_record("yunet", exc))
    try:
        return _detect_faces_opencv(image_rgb, max_faces, min_face_pixels), errors
    except Exception as exc:
        errors.append(_detector_error_record("opencv-haar", exc))
        return [], errors


def _square_padded_component_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_box(box, width, height)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * SELECTION_ROI_PADDING_RATIO)
    return _clamp_box(
        (
            center_x - 0.5 * side,
            center_y - 0.5 * side,
            center_x + 0.5 * side,
            center_y + 0.5 * side,
        ),
        width,
        height,
    )


def _map_roi_face_region(
    region: dict,
    *,
    roi_box: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
    image_shape: tuple[int, int, int],
    scale_x: float,
    scale_y: float,
    component_mask: np.ndarray,
) -> dict | None:
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = roi_box
    roi_height, roi_width = int(roi_shape[0]), int(roi_shape[1])
    bx0, by0, bx1, by1 = region["bbox"]
    mapped_box = _clamp_box(
        (
            x0 + float(bx0) / scale_x,
            y0 + float(by0) / scale_y,
            x0 + float(bx1) / scale_x,
            y0 + float(by1) / scale_y,
        ),
        image_width,
        image_height,
    )
    if (
        min(mapped_box[2] - mapped_box[0], mapped_box[3] - mapped_box[1])
        < SELECTION_ROI_MAPPED_FACE_PIXELS_FLOOR
    ):
        return None

    mapped = dict(region)
    mapped["bbox"] = list(mapped_box)
    mapped["detector"] = f"selection-roi:{region.get('detector') or 'unknown'}"
    mapped["detection_scope"] = "selection-component-upscaled"
    mapped["detection_roi_bbox"] = list(roi_box)
    mapped["detection_roi_scale"] = [float(scale_x), float(scale_y)]
    for mask_name in ("face_mask", "feature_mask"):
        source_mask = np.asarray(region[mask_name], dtype=np.uint8)
        resized = cv2.resize(source_mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((image_height, image_width), dtype=np.uint8)
        full[y0:y1, x0:x1] = resized
        mapped[mask_name] = full
    selected = np.asarray(component_mask, dtype=bool)
    mapped_face = mapped["face_mask"] > 0
    selected_overlap_ratio = float(
        np.count_nonzero(mapped_face & selected)
        / max(np.count_nonzero(mapped_face), 1)
    )
    if selected_overlap_ratio < 0.50:
        return None
    mapped["selection_overlap_ratio"] = selected_overlap_ratio
    mapped_parts = {}
    for name, source_mask in (region.get("part_masks") or {}).items():
        resized = cv2.resize(
            np.asarray(source_mask, dtype=np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        )
        full = np.zeros((image_height, image_width), dtype=np.uint8)
        full[y0:y1, x0:x1] = resized
        mapped_parts[name] = full
    mapped["part_masks"] = mapped_parts

    if region.get("landmarks_xyz") is not None:
        landmarks = np.asarray(region["landmarks_xyz"], dtype=np.float32).copy()
        landmarks[:, 0] = (
            x0 + landmarks[:, 0] * max(roi_width - 1, 1)
        ) / max(image_width - 1, 1)
        landmarks[:, 1] = (
            y0 + landmarks[:, 1] * max(roi_height - 1, 1)
        ) / max(image_height - 1, 1)
        mapped["landmarks_xyz"] = landmarks
    if region.get("keypoints") is not None:
        keypoints = np.asarray(region["keypoints"], dtype=np.float32).reshape(-1, 2)
        keypoints[:, 0] = x0 + keypoints[:, 0] / scale_x
        keypoints[:, 1] = y0 + keypoints[:, 1] / scale_y
        mapped["keypoints"] = keypoints.round(3).tolist()
    return mapped


def _upgrade_yunet_regions_with_mediapipe(
    image_rgb: np.ndarray,
    regions: list[dict],
    *,
    max_faces: int,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
) -> list[dict]:
    image_height, image_width = image_rgb.shape[:2]
    upgraded = []
    for guide in regions[: max(1, int(max_faces))]:
        if guide.get("face_mask") is None:
            continue
        roi_box = _square_padded_component_box(
            guide["bbox"], image_width, image_height
        )
        x0, y0, x1, y1 = roi_box
        crop = image_rgb[y0:y1, x0:x1]
        if not crop.size:
            continue
        scale = SELECTION_ROI_DETECTION_DIMENSION / float(max(crop.shape[:2]))
        target_width = max(1, int(round(crop.shape[1] * scale)))
        target_height = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(
            crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC
        )
        try:
            local_kwargs = {}
            if output_face_blendshapes:
                local_kwargs["output_face_blendshapes"] = True
            if output_facial_transformation_matrixes:
                local_kwargs[
                    "output_facial_transformation_matrixes"
                ] = True
            local_regions = _detect_faces_mediapipe(
                resized,
                1,
                MIN_FACE_PIXELS_FLOOR,
                **local_kwargs,
            )
        except Exception:
            continue
        guide_mask = np.asarray(guide.get("face_mask"), dtype=np.uint8) > 0
        for local in local_regions[:1]:
            mapped = _map_roi_face_region(
                local,
                roi_box=roi_box,
                roi_shape=crop.shape[:2],
                image_shape=image_rgb.shape,
                scale_x=target_width / float(crop.shape[1]),
                scale_y=target_height / float(crop.shape[0]),
                component_mask=guide_mask,
            )
            if mapped is None:
                continue
            mapped["detector"] = (
                "yunet-guided:" + str(local.get("detector") or "mediapipe")
            )
            mapped["detection_scope"] = "yunet-face-upscaled"
            mapped["guide_detector"] = guide.get("detector")
            mapped["guide_confidence"] = guide.get("confidence")
            upgraded.append(mapped)
    return upgraded[: max(1, int(max_faces))]


def detect_face_regions_in_roi(
    image_rgb: np.ndarray,
    roi_mask: np.ndarray,
    *,
    max_faces: int = 3,
    min_face_pixels: int = DEFAULT_MIN_FACE_PIXELS,
    detector: Callable[[np.ndarray, int, int], tuple[list[dict], list[str]] | list[dict]] | None = None,
    allow_selection_detail_fallback: bool = False,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
) -> tuple[list[dict], list[str], dict]:
    image_height, image_width = image_rgb.shape[:2]
    mask = np.asarray(roi_mask, dtype=np.uint8)
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(
        64,
        int(round(image_height * image_width * SELECTION_ROI_MINIMUM_COVERAGE)),
    )
    components = []
    for component in range(1, component_count):
        x, y, width, height, area = (int(value) for value in stats[component])
        if area < minimum_area or min(width, height) < MIN_FACE_PIXELS_FLOOR:
            continue
        components.append((area, component, (x, y, x + width, y + height)))
    components.sort(reverse=True)

    regions = []
    errors = []
    attempts = []
    for _area, _component, component_box in components[: max(1, int(max_faces))]:
        roi_box = _square_padded_component_box(
            component_box,
            image_width,
            image_height,
        )
        x0, y0, x1, y1 = roi_box
        crop = image_rgb[y0:y1, x0:x1]
        if not crop.size:
            continue
        scale = SELECTION_ROI_DETECTION_DIMENSION / float(max(crop.shape[:2]))
        target_width = max(1, int(round(crop.shape[1] * scale)))
        target_height = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        component_width = component_box[2] - component_box[0]
        component_height = component_box[3] - component_box[1]
        component_minimum = min(
            component_width * target_width / float(crop.shape[1]),
            component_height * target_height / float(crop.shape[0]),
        )
        roi_component_cap = max(
            MIN_FACE_PIXELS_FLOOR,
            int(round(component_minimum * SELECTION_ROI_MIN_FACE_COMPONENT_RATIO)),
        )
        effective_minimum = min(
            _effective_min_face_pixels(resized.shape, min_face_pixels),
            roi_component_cap,
        )
        if detector is not None:
            detection_result = detector(resized, max_faces, effective_minimum)
        else:
            detection_kwargs = {
                "max_faces": max_faces,
                "min_face_pixels": effective_minimum,
                "yunet_score_threshold": (
                    YUNET_SELECTION_ROI_SCORE_THRESHOLD
                ),
                "yunet_minimum_eye_separation_ratio": (
                    YUNET_PROFILE_MINIMUM_EYE_SEPARATION_RATIO
                ),
                "yunet_minimum_mouth_separation_ratio": (
                    YUNET_PROFILE_MINIMUM_MOUTH_SEPARATION_RATIO
                ),
            }
            if output_face_blendshapes:
                detection_kwargs["output_face_blendshapes"] = True
            if output_facial_transformation_matrixes:
                detection_kwargs[
                    "output_facial_transformation_matrixes"
                ] = True
            detection_result = detect_face_regions(
                resized,
                **detection_kwargs,
            )
        if isinstance(detection_result, tuple):
            local_regions, local_errors = detection_result
        else:
            local_regions, local_errors = detection_result, []
        errors.extend(f"selection-roi:{error}" for error in local_errors)
        mapped_count = 0
        for region in local_regions:
            mapped = _map_roi_face_region(
                region,
                roi_box=roi_box,
                roi_shape=crop.shape[:2],
                image_shape=image_rgb.shape,
                scale_x=target_width / float(crop.shape[1]),
                scale_y=target_height / float(crop.shape[0]),
                component_mask=labels == _component,
            )
            if mapped is not None:
                mx0, my0, mx1, my1 = mapped["bbox"]
                duplicate = False
                for existing in regions:
                    ex0, ey0, ex1, ey1 = existing["bbox"]
                    intersection = max(0, min(mx1, ex1) - max(mx0, ex0)) * max(
                        0,
                        min(my1, ey1) - max(my0, ey0),
                    )
                    union = (
                        max(0, mx1 - mx0) * max(0, my1 - my0)
                        + max(0, ex1 - ex0) * max(0, ey1 - ey0)
                        - intersection
                    )
                    if union > 0 and intersection / union >= 0.50:
                        duplicate = True
                        break
                if not duplicate:
                    regions.append(mapped)
                    mapped_count += 1
        attempts.append(
            {
                "component_bbox": list(component_box),
                "roi_bbox": list(roi_box),
                "input_shape": [target_height, target_width],
                "effective_minimum_face_pixels": int(effective_minimum),
                "detected_faces": int(mapped_count),
            }
        )
        if len(regions) >= max_faces:
            break
    fallback_regions = 0
    fallback_errors_are_clean = all(
        str(error).endswith(
            "opencv-haar:RuntimeError:OpenCV face detector cascade is unavailable"
        )
        for error in errors
    )
    if not regions and allow_selection_detail_fallback and fallback_errors_are_clean:
        for _area, component, component_box in components[: max(1, int(max_faces))]:
            x0, y0, x1, y1 = component_box
            if min(x1 - x0, y1 - y0) < max(MIN_FACE_PIXELS_FLOOR, 64):
                continue
            component_mask = (labels == component).astype(np.uint8) * 255
            regions.append(
                {
                    "bbox": list(component_box),
                    "face_mask": component_mask,
                    "feature_mask": component_mask.copy(),
                    "part_masks": {},
                    "detector": "selection-detail-fallback",
                    "landmark_count": 0,
                    "semantic_scope": "selected-component-detail",
                    "fallback_reason": "no_validated_face_in_selection_roi",
                }
            )
            fallback_regions += 1
            if len(regions) >= max_faces:
                break
    return regions[:max_faces], errors, {
        "enabled": True,
        "eligible_components": int(len(components)),
        "attempts": attempts,
        "detected_faces": int(max(0, len(regions) - fallback_regions)),
        "validated_face_regions": int(max(0, len(regions) - fallback_regions)),
        "selection_detail_fallback_regions": int(fallback_regions),
        "fallback_errors_clean": bool(fallback_errors_are_clean),
        "selection_roi_yunet_policy": {
            "score_threshold": YUNET_SELECTION_ROI_SCORE_THRESHOLD,
            "minimum_eye_separation_ratio": (
                YUNET_PROFILE_MINIMUM_EYE_SEPARATION_RATIO
            ),
            "minimum_mouth_separation_ratio": (
                YUNET_PROFILE_MINIMUM_MOUTH_SEPARATION_RATIO
            ),
            "minimum_mapped_selection_overlap": 0.50,
        },
    }


def _resize_float(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    return cv2.resize(np.asarray(values, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def _resize_mask(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    return cv2.resize(np.asarray(values, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def _smooth_nan_aware(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(np.float32, copy=True)
    valid = np.isfinite(values)
    if not np.any(valid):
        return values.astype(np.float32, copy=True)
    weighted = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(weighted, weights, out=result, where=weights > 1e-6)
    return result


def _robust_span(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    finite = np.asarray(values[valid], dtype=np.float64)
    if not finite.size:
        return 0.0
    low, high = np.percentile(finite, (5.0, 95.0))
    return max(0.0, float(high - low))


def _landmark_depth_prior(
    landmark_points_xy: np.ndarray,
    landmark_relative_z: np.ndarray,
    shape: tuple[int, int],
    face_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Rasterize MediaPipe's weak-perspective z signal into a smooth prior."""
    points = np.asarray(landmark_points_xy, dtype=np.float64)[:468]
    relative_z = np.asarray(landmark_relative_z, dtype=np.float64).reshape(-1)[: len(points)]
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(relative_z)
    height, width = int(shape[0]), int(shape[1])
    valid &= (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height - 1)
    )
    points = points[valid]
    relative_z = relative_z[valid]
    if len(points) < 100:
        raise ValueError("landmark prior requires at least 100 visible face landmarks")

    z_low, z_high = np.percentile(relative_z, (5.0, 95.0))
    z_span = float(z_high - z_low)
    if not np.isfinite(z_span) or z_span <= 1e-5:
        raise ValueError("landmark prior has no usable relative-depth span")
    # MediaPipe uses smaller relative z for points closer to the camera.
    normalized_nearness = np.clip((z_high - relative_z) / z_span, 0.0, 1.0)
    rows, cols = np.indices((height, width), dtype=np.float64)
    interpolator = LinearNDInterpolator(points, normalized_nearness, fill_value=np.nan)
    prior = np.asarray(interpolator(cols, rows), dtype=np.float32)
    prior = np.where(np.asarray(face_mask) > 0, prior, np.nan)
    face_rows, face_cols = np.where(np.asarray(face_mask) > 0)
    face_size = max(
        1.0,
        float(min(face_cols.max() - face_cols.min() + 1, face_rows.max() - face_rows.min() + 1)),
    )
    smoothing_sigma_px = float(np.clip(face_size * 0.035, 1.5, 8.0))
    prior = _smooth_nan_aware(prior, smoothing_sigma_px)
    return prior, {
        "landmarks_used": int(len(points)),
        "relative_z_p05": float(z_low),
        "relative_z_p95": float(z_high),
        "relative_z_span": z_span,
        "smoothing_sigma_px": smoothing_sigma_px,
    }


def _odd_kernel_size(value: float) -> int:
    size = max(3, int(round(float(value))))
    return size if size % 2 else size + 1


def _detect_eyewear_occlusion_weight(
    image_rgb: np.ndarray,
    face_mask: np.ndarray,
    landmark_points_xy: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Find one broad, dark eyewear component across the landmark eye band."""
    face_binary = np.asarray(face_mask) > 0
    empty = np.zeros(face_binary.shape, dtype=np.float32)
    image = np.asarray(image_rgb)
    points = np.asarray(landmark_points_xy, dtype=np.float64)
    if face_binary.ndim != 2 or image.ndim != 3 or image.shape[2] < 3:
        return empty, {"enabled": False, "reason": "invalid_input"}
    if image.shape[:2] != face_binary.shape:
        image = cv2.resize(
            image[:, :, :3],
            (face_binary.shape[1], face_binary.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image = image[:, :, :3]
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) <= max(EYEWEAR_LANDMARK_INDICES)
        or not np.all(np.isfinite(points[EYEWEAR_LANDMARK_INDICES]))
        or not np.any(face_binary)
    ):
        return empty, {"enabled": False, "reason": "landmarks_unavailable"}

    visible_points = points[: min(468, len(points))]
    face_size = max(
        1.0,
        float(min(np.ptp(visible_points[:, 0]), np.ptp(visible_points[:, 1]))),
    )
    support = np.zeros(face_binary.shape, dtype=np.uint8)
    eyewear_points = np.rint(points[EYEWEAR_LANDMARK_INDICES]).astype(np.int32)
    cv2.fillConvexPoly(support, cv2.convexHull(eyewear_points), 1)
    support_kernel = _odd_kernel_size(face_size * 0.10)
    support = cv2.dilate(
        support,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (support_kernel, support_kernel),
        ),
    ).astype(bool)
    support &= face_binary
    support_pixels = int(np.count_nonzero(support))
    face_pixels = int(np.count_nonzero(face_binary))
    if support_pixels < 32:
        return empty, {"enabled": False, "reason": "empty_eyewear_support"}

    luminance = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    rows = np.indices(face_binary.shape)[0]
    support_row_max = int(np.max(np.where(support)[0]))
    skin_reference = face_binary & ~support & (rows > support_row_max)
    minimum_reference_pixels = max(64, int(round(face_pixels * 0.02)))
    if np.count_nonzero(skin_reference) < minimum_reference_pixels:
        skin_reference = face_binary & ~support
    if np.count_nonzero(skin_reference) < minimum_reference_pixels:
        return empty, {"enabled": False, "reason": "skin_reference_unavailable"}
    skin_luminance_median = float(np.median(luminance[skin_reference]))
    dark_threshold = max(18.0, skin_luminance_median * 0.66)
    dark = support & (luminance < dark_threshold)
    close_width = _odd_kernel_size(face_size * 0.08)
    connected = cv2.morphologyEx(
        dark.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, close_width), dtype=np.uint8),
    )
    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        connected,
        connectivity=8,
    )
    if component_count <= 1:
        return empty, {
            "enabled": False,
            "reason": "no_dark_component",
            "skin_luminance_median": skin_luminance_median,
            "dark_luminance_threshold": float(dark_threshold),
        }
    component_index = max(
        range(1, component_count),
        key=lambda index: int(component_stats[index, cv2.CC_STAT_AREA]),
    )
    component_area = int(component_stats[component_index, cv2.CC_STAT_AREA])
    component_width = int(component_stats[component_index, cv2.CC_STAT_WIDTH])
    dark_coverage = float(np.count_nonzero(dark) / support_pixels)
    component_coverage = float(component_area / support_pixels)
    component_width_ratio = float(component_width / face_size)
    support_face_ratio = float(support_pixels / max(face_pixels, 1))
    failures = []
    if dark_coverage < EYEWEAR_MINIMUM_DARK_COVERAGE:
        failures.append("dark_coverage")
    if component_coverage < EYEWEAR_MINIMUM_COMPONENT_COVERAGE:
        failures.append("component_coverage")
    if component_width_ratio < EYEWEAR_MINIMUM_COMPONENT_WIDTH_RATIO:
        failures.append("component_width")
    detection_stats = {
        "method": "landmark-guided-dark-eyewear-component",
        "face_size_px": float(face_size),
        "support_pixels": support_pixels,
        "support_face_ratio": support_face_ratio,
        "skin_luminance_median": skin_luminance_median,
        "dark_luminance_threshold": float(dark_threshold),
        "dark_coverage_ratio": dark_coverage,
        "component_coverage_ratio": component_coverage,
        "component_width_ratio": component_width_ratio,
        "minimum_dark_coverage_ratio": EYEWEAR_MINIMUM_DARK_COVERAGE,
        "minimum_component_coverage_ratio": EYEWEAR_MINIMUM_COMPONENT_COVERAGE,
        "minimum_component_width_ratio": EYEWEAR_MINIMUM_COMPONENT_WIDTH_RATIO,
    }
    if failures:
        return empty, {
            "enabled": False,
            "reason": "eyewear_detection_gate",
            "failures": failures,
            **detection_stats,
        }

    hard_component = labels == component_index
    dilation_size = _odd_kernel_size(face_size * 0.055)
    hard_component = cv2.dilate(
        hard_component.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size),
        ),
    ).astype(bool)
    hard_component &= support
    feather_sigma = max(1.0, face_size * 0.025)
    weight = gaussian_filter(
        hard_component.astype(np.float32),
        sigma=feather_sigma,
        mode="nearest",
    )
    weight_max = float(np.max(weight))
    if weight_max <= 1e-6:
        return empty, {
            "enabled": False,
            "reason": "empty_eyewear_weight",
            **detection_stats,
        }
    weight = np.clip((weight / weight_max - 0.025) / 0.975, 0.0, 1.0)
    weight *= support.astype(np.float32)
    core_pixels = int(np.count_nonzero(weight >= 0.5))
    if core_pixels < max(32, int(round(face_pixels * 0.02))):
        return empty, {
            "enabled": False,
            "reason": "insufficient_eyewear_core",
            "core_pixels": core_pixels,
            **detection_stats,
        }
    return weight.astype(np.float32), {
        "enabled": True,
        "core_pixels": core_pixels,
        "feather_sigma_px": float(feather_sigma),
        **detection_stats,
    }


def _fuse_face_landmark_shape_prior_with_context(
    global_depth: np.ndarray,
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    landmark_points_xy: np.ndarray,
    landmark_relative_z: np.ndarray,
    *,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
    max_correction_ratio: float = DEFAULT_FACE_MAX_CORRECTION_RATIO,
    minimum_abs_correlation: float = 0.15,
    maximum_yaw_proxy: float = 0.32,
    minimum_confidence_weight: float = 0.0,
    correction_limit_scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, dict, dict | None]:
    """Fuse a gated coarse face prior while preserving the generic depth map."""
    global_depth = np.asarray(global_depth, dtype=np.float32)
    face_binary = np.asarray(face_mask) > 0
    if global_depth.ndim != 2 or face_binary.shape != global_depth.shape:
        raise ValueError("Landmark shape prior expects matching two-dimensional arrays")
    points = np.asarray(landmark_points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Landmark points must have shape (N, 2)")
    visible_points = points[:468]
    visible_width = max(float(np.ptp(visible_points[:, 0])), 1e-6)
    visible_height = max(float(np.ptp(visible_points[:, 1])), 1e-6)
    contour_margin = min(
        float(np.min(visible_points[:, 0])),
        float(np.min(visible_points[:, 1])),
        float(global_depth.shape[1] - 1 - np.max(visible_points[:, 0])),
        float(global_depth.shape[0] - 1 - np.max(visible_points[:, 1])),
    )
    minimum_contour_margin = 0.01 * min(visible_width, visible_height)
    if contour_margin < minimum_contour_margin:
        return global_depth, np.zeros_like(global_depth), {
            "enabled": False,
            "reason": "clipped_contour_gate",
            "contour_margin_px": float(contour_margin),
            "minimum_contour_margin_px": float(minimum_contour_margin),
        }, None
    if len(points) > 454:
        face_width = visible_width
        face_center_x = float(np.min(visible_points[:, 0]) + np.max(visible_points[:, 0])) * 0.5
        yaw_proxy = abs(float(points[1, 0]) - face_center_x) / face_width
    else:
        yaw_proxy = 0.0
    if yaw_proxy > float(maximum_yaw_proxy):
        return global_depth, np.zeros_like(global_depth), {
            "enabled": False,
            "reason": "yaw_gate",
            "yaw_proxy": float(yaw_proxy),
            "maximum_yaw_proxy": float(maximum_yaw_proxy),
        }, None

    prior, prior_stats = _landmark_depth_prior(
        points,
        landmark_relative_z,
        global_depth.shape,
        face_binary,
    )
    distance = cv2.distanceTransform(face_binary.astype(np.uint8), cv2.DIST_L2, 5)
    face_rows, face_cols = np.where(face_binary)
    face_size = max(
        1.0,
        float(min(face_cols.max() - face_cols.min() + 1, face_rows.max() - face_rows.min() + 1)),
    )
    feather_px = max(2.0, face_size * max(0.05, float(feather_ratio)))
    weight = np.clip((distance - 2.0) / feather_px, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    valid = face_binary & np.isfinite(prior) & np.isfinite(global_depth)
    stable = valid & (weight >= 0.35) & (np.asarray(feature_mask) == 0)
    if np.count_nonzero(stable) < 100:
        stable = valid & (weight >= 0.35)
    if np.count_nonzero(stable) < 100:
        raise ValueError("landmark prior has insufficient stable overlap")

    coarse_sigma_px = float(np.clip(face_size * 0.055, 2.0, 10.0))
    global_coarse = _smooth_nan_aware(global_depth, coarse_sigma_px)
    prior_values = np.asarray(prior[stable], dtype=np.float64)
    target_values = np.asarray(global_coarse[stable], dtype=np.float64)
    prior_centered = prior_values - np.median(prior_values)
    target_centered = target_values - np.median(target_values)
    denominator = float(np.dot(prior_centered, prior_centered))
    target_norm = float(np.dot(target_centered, target_centered))
    if denominator <= 1e-12 or target_norm <= 1e-12:
        return global_depth, np.zeros_like(global_depth), {
            "enabled": False,
            "reason": "flat_alignment_signal",
            "yaw_proxy": float(yaw_proxy),
            **prior_stats,
        }, None
    correlation = float(
        np.dot(prior_centered, target_centered)
        / np.sqrt(denominator * target_norm)
    )
    if abs(correlation) < float(minimum_abs_correlation):
        return global_depth, np.zeros_like(global_depth), {
            "enabled": False,
            "reason": "correlation_gate",
            "correlation": correlation,
            "minimum_abs_correlation": float(minimum_abs_correlation),
            "yaw_proxy": float(yaw_proxy),
            **prior_stats,
        }, None

    scale = float(np.dot(prior_centered, target_centered) / denominator)
    reference_span = _robust_span(global_depth, face_binary)
    scale_limit = max(reference_span * 2.0, 1e-6)
    scale = float(np.clip(scale, -scale_limit, scale_limit))
    offset = float(np.median(target_values - scale * prior_values))
    aligned_prior = prior * scale + offset
    correction = aligned_prior - global_coarse
    correction -= float(np.median(correction[stable]))
    resolved_minimum_confidence = float(
        np.clip(minimum_confidence_weight, 0.0, 1.0)
    )
    resolved_correction_limit_scale = float(
        np.clip(correction_limit_scale, 0.0, 1.0)
    )
    correction_limit = (
        reference_span
        * max(0.0, float(max_correction_ratio))
        * resolved_correction_limit_scale
    )
    correction = np.clip(correction, -correction_limit, correction_limit)
    unfloored_confidence_weight = float(
        np.clip(
            (abs(correlation) - float(minimum_abs_correlation))
            / max(0.60 - float(minimum_abs_correlation), 1e-6),
            0.0,
            1.0,
        )
    )
    confidence_weight = max(
        unfloored_confidence_weight,
        resolved_minimum_confidence,
    )
    correction = (
        np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
        * weight
        * confidence_weight
    )

    refined = global_depth.copy()
    refined[valid] = global_depth[valid] + correction[valid]
    boundary = (distance > 0) & (distance <= 2.0)
    stats = {
        "enabled": True,
        "method": "mediapipe-relative-z-coarse-prior",
        "correlation": correlation,
        "confidence_weight": confidence_weight,
        "confidence_weight_unfloored": unfloored_confidence_weight,
        "minimum_confidence_weight": resolved_minimum_confidence,
        "confidence_floor_applied": bool(
            confidence_weight > unfloored_confidence_weight + 1e-8
        ),
        "minimum_abs_correlation": float(minimum_abs_correlation),
        "yaw_proxy": float(yaw_proxy),
        "maximum_yaw_proxy": float(maximum_yaw_proxy),
        "scale": scale,
        "offset": offset,
        "coarse_sigma_px": coarse_sigma_px,
        "feather_px": float(feather_px),
        "correction_limit": float(correction_limit),
        "correction_limit_scale": resolved_correction_limit_scale,
        "max_abs_correction": float(np.max(np.abs(correction))),
        "mean_abs_correction": float(np.mean(np.abs(correction[face_binary]))),
        "boundary_max_abs_correction": float(np.max(np.abs(correction[boundary]))) if np.any(boundary) else 0.0,
        **prior_stats,
    }
    context = {
        "aligned_prior": aligned_prior.astype(np.float32),
        "reference_span": float(reference_span),
    }
    return refined, weight.astype(np.float32), stats, context


def fuse_face_landmark_shape_prior(
    global_depth: np.ndarray,
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    landmark_points_xy: np.ndarray,
    landmark_relative_z: np.ndarray,
    *,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
    max_correction_ratio: float = DEFAULT_FACE_MAX_CORRECTION_RATIO,
    minimum_abs_correlation: float = 0.15,
    maximum_yaw_proxy: float = 0.32,
    minimum_confidence_weight: float = 0.0,
    correction_limit_scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    refined, weight, stats, _ = _fuse_face_landmark_shape_prior_with_context(
        global_depth,
        face_mask,
        feature_mask,
        landmark_points_xy,
        landmark_relative_z,
        feather_ratio=feather_ratio,
        max_correction_ratio=max_correction_ratio,
        minimum_abs_correlation=minimum_abs_correlation,
        maximum_yaw_proxy=maximum_yaw_proxy,
        minimum_confidence_weight=minimum_confidence_weight,
        correction_limit_scale=correction_limit_scale,
    )
    return refined, weight, stats


def _reconstruct_eyewear_occlusion(
    refined_depth: np.ndarray,
    source_depth: np.ndarray,
    aligned_prior: np.ndarray,
    occlusion_weight: np.ndarray,
    *,
    reference_span: float,
    eye_masks: dict[str, np.ndarray] | None = None,
    accessory_residual_ratio: float = EYEWEAR_ACCESSORY_RESIDUAL_RATIO,
    maximum_correction_ratio: float = EYEWEAR_MAXIMUM_CORRECTION_RATIO,
    minimum_input_residual_ratio: float = 0.05,
    maximum_output_residual_ratio: float = 0.05,
    minimum_residual_reduction_ratio: float = 0.35,
    maximum_saturated_core_ratio: float = 0.05,
    minimum_eye_detail_retention: float = 0.65,
    maximum_eye_detail_retention: float = 1.60,
) -> tuple[np.ndarray, dict]:
    """Replace an eyewear sheet with a bounded face-prior surface."""
    refined = np.asarray(refined_depth, dtype=np.float32)
    source = np.asarray(source_depth, dtype=np.float32)
    prior = np.asarray(aligned_prior, dtype=np.float32)
    weight = np.clip(np.asarray(occlusion_weight, dtype=np.float32), 0.0, 1.0)
    if not (refined.shape == source.shape == prior.shape == weight.shape) or refined.ndim != 2:
        raise ValueError("Eyewear reconstruction expects matching two-dimensional arrays")
    span = float(reference_span)
    if not np.isfinite(span) or span <= 1e-8:
        return refined, {"enabled": False, "reason": "invalid_reference_span"}
    finite = np.isfinite(refined) & np.isfinite(source) & np.isfinite(prior)
    core = finite & (weight >= 0.5)
    core_pixels = int(np.count_nonzero(core))
    if core_pixels < 32:
        return refined, {
            "enabled": False,
            "reason": "insufficient_occlusion_core",
            "core_pixels": core_pixels,
        }

    input_residual_p95 = float(np.percentile(np.abs(source[core] - prior[core]), 95.0))
    input_residual_ratio = input_residual_p95 / span
    if input_residual_ratio < float(minimum_input_residual_ratio):
        return refined, {
            "enabled": False,
            "reason": "source_depth_already_consistent",
            "core_pixels": core_pixels,
            "input_residual_p95_ratio": input_residual_ratio,
            "minimum_input_residual_p95_ratio": float(minimum_input_residual_ratio),
        }

    accessory_limit = span * max(0.0, float(accessory_residual_ratio))
    target = prior + np.clip(source - prior, -accessory_limit, accessory_limit)
    requested_correction = np.zeros_like(refined)
    requested_correction[finite] = weight[finite] * (target[finite] - refined[finite])
    maximum_correction = span * max(0.0, float(maximum_correction_ratio))
    saturated_core = core & (np.abs(requested_correction) >= maximum_correction * (1.0 - 1e-6))
    saturated_core_ratio = float(np.count_nonzero(saturated_core) / core_pixels)
    correction = np.clip(requested_correction, -maximum_correction, maximum_correction)
    candidate = refined.copy()
    candidate[finite] = refined[finite] + correction[finite]

    # Removing the low-frequency eyewear sheet can also erase real frame and
    # eye-region relief from the face crop. Restore only the high-frequency
    # residual needed to satisfy the bilateral detail gate; the broad sheet
    # remains governed by the landmark prior and the correction cap.
    eye_detail_restoration: dict[str, dict] = {}
    if eye_masks is not None:
        detail_sigma_px = 1.0
        refined_detail = refined - _smooth_nan_aware(refined, detail_sigma_px)
        refined_detail_gradient = np.hypot(*np.gradient(refined_detail))
        retention_target = min(
            float(maximum_eye_detail_retention) - 0.05,
            max(
                float(minimum_eye_detail_retention) + 0.05,
                float(minimum_eye_detail_retention) * 1.10,
            ),
        )

        for name in ("left_eye", "right_eye"):
            raw_mask = eye_masks.get(name)
            if raw_mask is None:
                continue
            eye = np.asarray(raw_mask) > 0
            if eye.shape != refined.shape:
                eye = _resize_mask(eye.astype(np.uint8) * 255, refined.shape) > 0
            metric_eye = cv2.erode(
                eye.astype(np.uint8), np.ones((3, 3), np.uint8)
            ) > 0
            metric_support = metric_eye & finite & (weight > 0.05)
            samples = int(np.count_nonzero(metric_support))
            if samples < 12:
                continue

            before_q95 = float(
                np.percentile(refined_detail_gradient[metric_support], 95.0)
            )
            denominator = max(before_q95, span * 1e-6)

            feathered_eye = cv2.GaussianBlur(
                eye.astype(np.float32),
                (0, 0),
                sigmaX=1.2,
                sigmaY=1.2,
            )
            restoration_support = np.clip(feathered_eye, 0.0, 1.0) * weight

            def restored_candidate(strength: float) -> tuple[np.ndarray, float]:
                trial = candidate + (
                    float(strength) * refined_detail * restoration_support
                )
                trial = refined + np.clip(
                    trial - refined,
                    -maximum_correction,
                    maximum_correction,
                )
                trial_detail = trial - _smooth_nan_aware(trial, detail_sigma_px)
                trial_gradient = np.hypot(*np.gradient(trial_detail))
                after_q95 = float(
                    np.percentile(trial_gradient[metric_support], 95.0)
                )
                return trial, after_q95 / denominator

            _, initial_retention = restored_candidate(0.0)
            selected_strength = 0.0
            selected_retention = initial_retention
            if initial_retention < float(minimum_eye_detail_retention):
                upper_strength = 2.0
                upper_candidate, upper_retention = restored_candidate(upper_strength)
                if upper_retention >= retention_target:
                    lower_strength = 0.0
                    for _ in range(18):
                        middle_strength = (lower_strength + upper_strength) * 0.5
                        _, middle_retention = restored_candidate(middle_strength)
                        if middle_retention < retention_target:
                            lower_strength = middle_strength
                        else:
                            upper_strength = middle_strength
                    selected_strength = upper_strength
                    candidate, selected_retention = restored_candidate(
                        selected_strength
                    )
                else:
                    selected_strength = upper_strength
                    candidate = upper_candidate
                    selected_retention = upper_retention

            eye_detail_restoration[name] = {
                "samples": samples,
                "target_gradient_q95_retention": retention_target,
                "initial_gradient_q95_retention": float(initial_retention),
                "restoration_strength": float(selected_strength),
                "final_gradient_q95_retention": float(selected_retention),
                "applied": bool(selected_strength > 0.0),
            }

    correction = np.zeros_like(refined)
    correction[finite] = candidate[finite] - refined[finite]
    output_residual_p95 = float(np.percentile(np.abs(candidate[core] - prior[core]), 95.0))
    output_residual_ratio = output_residual_p95 / span
    residual_reduction = float(1.0 - output_residual_ratio / max(input_residual_ratio, 1e-8))
    correction_values = np.abs(correction[finite])
    maximum_correction_observed_ratio = (
        float(np.max(correction_values) / span) if correction_values.size else 0.0
    )
    correction_p95_ratio = (
        float(np.percentile(correction_values, 95.0) / span) if correction_values.size else 0.0
    )
    transition = finite & (weight > 0.0) & (weight < 0.5)
    transition_correction_p95_ratio = (
        float(np.percentile(np.abs(correction[transition]), 95.0) / span)
        if np.any(transition)
        else 0.0
    )
    eye_detail_retention = []
    if eye_masks is not None:
        detail_sigma_px = 1.0
        refined_detail = refined - _smooth_nan_aware(refined, detail_sigma_px)
        candidate_detail = candidate - _smooth_nan_aware(candidate, detail_sigma_px)
        refined_detail_gradient = np.hypot(*np.gradient(refined_detail))
        candidate_detail_gradient = np.hypot(*np.gradient(candidate_detail))
        for name in ("left_eye", "right_eye"):
            raw_mask = eye_masks.get(name)
            if raw_mask is None:
                eye_detail_retention.append(
                    {"name": name, "available": False, "passed": False}
                )
                continue
            eye = np.asarray(raw_mask) > 0
            if eye.shape != refined.shape:
                eye = _resize_mask(eye.astype(np.uint8) * 255, refined.shape) > 0
            eye = cv2.erode(eye.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            support = eye & finite & (weight > 0.05)
            samples = int(np.count_nonzero(support))
            record = {"name": name, "available": False, "passed": False, "samples": samples}
            if samples < 12:
                eye_detail_retention.append(record)
                continue
            before_q95 = float(np.percentile(refined_detail_gradient[support], 95.0))
            after_q95 = float(np.percentile(candidate_detail_gradient[support], 95.0))
            retention = after_q95 / max(before_q95, span * 1e-6)
            passed = bool(
                np.isfinite(retention)
                and float(minimum_eye_detail_retention)
                <= retention
                <= float(maximum_eye_detail_retention)
            )
            record.update(
                {
                    "available": True,
                    "detail_sigma_px": detail_sigma_px,
                    "before_gradient_q95": before_q95,
                    "after_gradient_q95": after_q95,
                    "gradient_q95_retention": float(retention),
                    "passed": passed,
                    "restoration": eye_detail_restoration.get(name),
                }
            )
            eye_detail_retention.append(record)
    failures = []
    if not np.all(np.isfinite(candidate[finite])):
        failures.append("non_finite_candidate")
    if output_residual_ratio > float(maximum_output_residual_ratio):
        failures.append("output_residual")
    if residual_reduction < float(minimum_residual_reduction_ratio):
        failures.append("residual_reduction")
    if maximum_correction_observed_ratio > float(maximum_correction_ratio) + 1e-6:
        failures.append("maximum_correction")
    if saturated_core_ratio > float(maximum_saturated_core_ratio):
        failures.append("correction_saturation")
    if eye_masks is not None and (
        len(eye_detail_retention) != 2
        or any(not record["passed"] for record in eye_detail_retention)
    ):
        failures.append("bilateral_eye_detail_retention")
    stats = {
        "method": "landmark-prior-eyewear-deocclusion",
        "core_pixels": core_pixels,
        "accessory_residual_ratio": float(accessory_residual_ratio),
        "maximum_correction_ratio": float(maximum_correction_ratio),
        "input_residual_p95_ratio": input_residual_ratio,
        "output_residual_p95_ratio": output_residual_ratio,
        "residual_reduction_ratio": residual_reduction,
        "correction_p95_ratio": correction_p95_ratio,
        "maximum_correction_observed_ratio": maximum_correction_observed_ratio,
        "saturated_core_ratio": saturated_core_ratio,
        "transition_correction_p95_ratio": transition_correction_p95_ratio,
        "bilateral_eye_detail_retention": {
            "parts": eye_detail_retention,
            "minimum_gradient_q95_retention": float(minimum_eye_detail_retention),
            "maximum_gradient_q95_retention": float(maximum_eye_detail_retention),
            "passed": bool(
                len(eye_detail_retention) == 2
                and all(record["passed"] for record in eye_detail_retention)
            ),
        },
        "eye_detail_restoration": {
            "method": "bounded-high-frequency-eye-residual",
            "parts": eye_detail_restoration,
            "applied": any(
                record.get("applied", False)
                for record in eye_detail_restoration.values()
            ),
        },
        "quality_gates": {
            "passed": not failures,
            "failures": failures,
            "minimum_input_residual_p95_ratio": float(minimum_input_residual_ratio),
            "maximum_output_residual_p95_ratio": float(maximum_output_residual_ratio),
            "minimum_residual_reduction_ratio": float(minimum_residual_reduction_ratio),
            "maximum_correction_ratio": float(maximum_correction_ratio),
            "maximum_saturated_core_ratio": float(maximum_saturated_core_ratio),
        },
    }
    if failures:
        return refined, {
            "enabled": False,
            "reason": "quality_gate_failed",
            **stats,
        }
    return candidate, {"enabled": True, **stats}


def _fit_face_depth(local_depth: np.ndarray, global_depth: np.ndarray, anchor_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    valid = np.asarray(anchor_mask, dtype=bool) & np.isfinite(local_depth) & np.isfinite(global_depth)
    if np.count_nonzero(valid) < 24:
        valid = np.isfinite(local_depth) & np.isfinite(global_depth)
    local = np.asarray(local_depth[valid], dtype=np.float64)
    target = np.asarray(global_depth[valid], dtype=np.float64)
    if not local.size:
        raise ValueError("Face depth alignment has no finite overlap")

    local_low, local_high = np.percentile(local, (5.0, 95.0))
    target_low, target_high = np.percentile(target, (5.0, 95.0))
    trimmed = (
        (local >= local_low)
        & (local <= local_high)
        & (target >= target_low)
        & (target <= target_high)
    )
    if np.count_nonzero(trimmed) >= 24:
        local = local[trimmed]
        target = target[trimmed]

    local_median = float(np.median(local))
    target_median = float(np.median(target))
    local_centered = local - local_median
    denominator = float(np.dot(local_centered, local_centered))
    if denominator > 1e-12:
        scale = float(np.dot(local_centered, target - target_median) / denominator)
        scale = float(np.clip(scale, 0.25, 4.0))
    else:
        scale = 1.0
    offset = float(np.median(target - scale * local))
    aligned = local_depth.astype(np.float32) * scale + offset
    return aligned, {
        "scale": scale,
        "offset": offset,
        "anchor_pixels": int(local.size),
    }


def face_blend_weight(
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    *,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
) -> tuple[np.ndarray, np.ndarray, float]:
    binary = (np.asarray(face_mask) > 0).astype(np.uint8)
    if not np.any(binary):
        raise ValueError("Face mask is empty")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(binary)
    face_size = max(1.0, float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    feather_px = max(2.0, face_size * max(0.05, float(feather_ratio)))
    transition = np.clip((distance - 1.0) / feather_px, 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)

    feature = (np.asarray(feature_mask) > 0).astype(np.float32)
    feature = gaussian_filter(feature, sigma=max(1.0, feather_px * 0.16), mode="nearest")
    if float(feature.max()) > 0:
        feature /= float(feature.max())
    # The face oval is only a safety envelope. Giving every pixel inside it a
    # base weight transfers crop-specific depth into cheeks, jaw, and hair and
    # can turn the oval into a visible relief plateau. Keep a softly expanded
    # support around actual facial features and make the rest exactly zero.
    feature = np.clip(
        (feature - FACE_FEATURE_WEIGHT_FLOOR) / (1.0 - FACE_FEATURE_WEIGHT_FLOOR),
        0.0,
        1.0,
    )
    weight = transition * feature * binary
    return weight.astype(np.float32), distance.astype(np.float32), feather_px


def face_surface_support_mask(
    face_mask: np.ndarray,
    selection_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """Select the connected subject component that contains the detected face."""
    face = (np.asarray(face_mask) > 0).astype(np.uint8)
    face_pixels = int(np.count_nonzero(face))
    if face_pixels == 0:
        raise ValueError("Face surface support requires a non-empty face mask")
    if selection_mask is None:
        return face * 255, {
            "mode": "detector-face",
            "reason": "selection_mask_unavailable",
            "face_pixels": face_pixels,
            "support_pixels": face_pixels,
            "support_expansion_ratio": 1.0,
        }

    selection = np.asarray(selection_mask)
    if selection.shape != face.shape:
        selection = _resize_mask(selection, face.shape)
    selection = (selection > 0).astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        selection,
        8,
    )
    best_component = None
    best_overlap = 0
    for component in range(1, component_count):
        overlap = int(np.count_nonzero((labels == component) & (face > 0)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_component = component
    minimum_overlap = max(24, int(round(face_pixels * 0.25)))
    if best_component is None or best_overlap < minimum_overlap:
        return face * 255, {
            "mode": "detector-face",
            "reason": "selection_component_overlap_gate",
            "face_pixels": face_pixels,
            "selection_components": max(0, int(component_count - 1)),
            "best_overlap_pixels": best_overlap,
            "minimum_overlap_pixels": minimum_overlap,
            "support_pixels": face_pixels,
            "support_expansion_ratio": 1.0,
        }

    subject = (labels == best_component).astype(np.uint8)
    support = np.maximum(subject, face)
    support_pixels = int(np.count_nonzero(support))
    return support * 255, {
        "mode": "selection-subject",
        "reason": "connected_selection_component",
        "face_pixels": face_pixels,
        "selection_components": max(0, int(component_count - 1)),
        "selected_component_pixels": int(
            stats[best_component, cv2.CC_STAT_AREA]
        ),
        "face_overlap_pixels": best_overlap,
        "support_pixels": support_pixels,
        "support_expansion_ratio": float(support_pixels / face_pixels),
    }


def fuse_face_depth(
    global_depth: np.ndarray,
    local_face_depth: np.ndarray,
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    *,
    detail_strength: float = DEFAULT_FACE_DETAIL_STRENGTH,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
    max_correction_ratio: float = DEFAULT_FACE_MAX_CORRECTION_RATIO,
) -> tuple[np.ndarray, np.ndarray, dict]:
    global_depth = np.asarray(global_depth, dtype=np.float32)
    local_face_depth = np.asarray(local_face_depth, dtype=np.float32)
    if global_depth.ndim != 2 or local_face_depth.ndim != 2:
        raise ValueError("Face depth fusion expects two-dimensional depth arrays")
    if local_face_depth.shape != global_depth.shape:
        local_face_depth = _resize_float(local_face_depth, global_depth.shape)
    if face_mask.shape != global_depth.shape:
        face_mask = _resize_mask(face_mask, global_depth.shape)
    if feature_mask.shape != global_depth.shape:
        feature_mask = _resize_mask(feature_mask, global_depth.shape)

    weight, distance, feather_px = face_blend_weight(
        face_mask,
        feature_mask,
        feather_ratio=feather_ratio,
    )
    anchor_mask = (distance > 0) & (distance <= feather_px * 1.5) & (feature_mask == 0)
    aligned, alignment = _fit_face_depth(local_face_depth, global_depth, anchor_mask)

    ys, xs = np.nonzero(face_mask > 0)
    face_size = max(1.0, float(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    detail_sigma_px = float(np.clip(face_size * 0.025, 1.0, 8.0))
    local_detail = aligned - _smooth_nan_aware(aligned, detail_sigma_px)
    global_detail = global_depth - _smooth_nan_aware(global_depth, detail_sigma_px)

    face_binary = face_mask > 0
    reference_span = max(
        _robust_span(global_depth, face_binary),
        _robust_span(aligned, face_binary),
    )
    # Never replace a stronger global feature with a smoother crop estimate.
    # The crop is allowed to add only excess high-frequency magnitude with a
    # compatible direction, or detail where the global estimate is flat.
    global_magnitude = np.abs(global_detail)
    local_magnitude = np.abs(local_detail)
    excess_magnitude = np.maximum(local_magnitude - global_magnitude, 0.0)
    global_flat_threshold = reference_span * 0.002
    compatible = (local_detail * global_detail >= 0.0) | (global_magnitude <= global_flat_threshold)
    correction = (
        np.sign(local_detail)
        * excess_magnitude
        * compatible
        * max(0.0, float(detail_strength))
    )
    correction_limit = reference_span * max(0.0, float(max_correction_ratio))
    if correction_limit > 0 and math.isfinite(correction_limit):
        correction = np.clip(correction, -correction_limit, correction_limit)
    else:
        correction = np.zeros_like(correction)
    correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0) * weight

    refined = global_depth.copy()
    valid = np.isfinite(global_depth)
    refined[valid] = global_depth[valid] + correction[valid]
    boundary = (distance > 0) & (distance <= 2.0)
    stats = {
        **alignment,
        "detail_sigma_px": detail_sigma_px,
        "detail_fusion": "monotonic-excess",
        "blend_strategy": "feature-supported-shape-preserving",
        "active_blend_ratio": float(np.count_nonzero(weight > 1e-4) / max(np.count_nonzero(face_binary), 1)),
        "feather_px": feather_px,
        "correction_limit": float(correction_limit),
        "max_abs_correction": float(np.max(np.abs(correction))),
        "mean_abs_correction": float(np.mean(np.abs(correction[face_binary]))),
        "boundary_max_abs_correction": float(np.max(np.abs(correction[boundary]))) if np.any(boundary) else 0.0,
    }
    return refined, weight, stats


def fuse_face_surface_residual(
    global_depth: np.ndarray,
    local_face_depth: np.ndarray,
    surface_residual: np.ndarray,
    face_mask: np.ndarray,
    *,
    alignment_depth: np.ndarray | None = None,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
    max_correction_ratio: float = DEFAULT_FACE_MAX_CORRECTION_RATIO,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a bounded non-affine local-depth residual without moving the boundary."""
    global_depth = np.asarray(global_depth, dtype=np.float32)
    local_face_depth = np.asarray(local_face_depth, dtype=np.float32)
    surface_residual = np.asarray(surface_residual, dtype=np.float32)
    if global_depth.ndim != 2:
        raise ValueError("Face surface residual fusion expects a 2D depth map")
    if local_face_depth.shape != global_depth.shape:
        local_face_depth = _resize_float(local_face_depth, global_depth.shape)
    if surface_residual.shape != global_depth.shape:
        surface_residual = _resize_float(surface_residual, global_depth.shape)
    if face_mask.shape != global_depth.shape:
        face_mask = _resize_mask(face_mask, global_depth.shape)
    reference = (
        global_depth
        if alignment_depth is None
        else np.asarray(alignment_depth, dtype=np.float32)
    )
    if reference.shape != global_depth.shape:
        reference = _resize_float(reference, global_depth.shape)
    if not np.all(np.isfinite(surface_residual)):
        raise ValueError("Face surface residual contains non-finite values")

    binary = (face_mask > 0).astype(np.uint8)
    if not np.any(binary):
        raise ValueError("Face surface residual mask is empty")
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
    feather_px = max(2.0, face_size * max(0.05, float(feather_ratio)))
    weight = np.clip((distance - 1.0) / feather_px, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight) * binary
    boundary = (distance > 0) & (distance <= 2.0)
    weight[boundary] = 0.0

    outer_radius = max(2, int(round(feather_px * 0.75)))
    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outer_radius * 2 + 1, outer_radius * 2 + 1),
    )
    outer_ring = (cv2.dilate(binary, outer_kernel) > 0) & (binary == 0)
    outer_valid = (
        outer_ring
        & np.isfinite(local_face_depth)
        & np.isfinite(reference)
    )
    if np.count_nonzero(outer_valid) >= 24:
        anchor_mask = outer_valid
        anchor_strategy = "outer-face-ring"
    else:
        anchor_mask = (
            (distance > 0)
            & (distance <= feather_px * 1.5)
            & np.isfinite(local_face_depth)
            & np.isfinite(reference)
        )
        anchor_strategy = "inner-boundary-fallback"
    _aligned, alignment = _fit_face_depth(
        local_face_depth,
        reference,
        anchor_mask,
    )

    face_valid = (
        (binary > 0)
        & np.isfinite(local_face_depth)
        & np.isfinite(surface_residual)
    )
    if np.count_nonzero(face_valid) < 24:
        raise ValueError("Face surface residual has insufficient finite support")
    affine_design = np.column_stack(
        (
            local_face_depth[face_valid].astype(np.float64),
            np.ones(np.count_nonzero(face_valid), dtype=np.float64),
        )
    )
    residual_scale, residual_offset = np.linalg.lstsq(
        affine_design,
        surface_residual[face_valid].astype(np.float64),
        rcond=None,
    )[0]
    non_affine_residual = surface_residual - (
        float(residual_scale) * local_face_depth + float(residual_offset)
    )
    correction = non_affine_residual * float(alignment["scale"])
    reference_span = max(
        _robust_span(reference, face_valid),
        _robust_span(global_depth, face_valid),
    )
    correction_limit = reference_span * max(
        0.0,
        float(max_correction_ratio),
    )
    if correction_limit > 0 and math.isfinite(correction_limit):
        correction = np.clip(
            correction,
            -correction_limit,
            correction_limit,
        )
    else:
        correction = np.zeros_like(correction)
    correction = np.nan_to_num(
        correction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ) * weight

    refined = global_depth.copy()
    valid = np.isfinite(global_depth)
    refined[valid] = global_depth[valid] + correction[valid]
    stats = {
        **alignment,
        "enabled": True,
        "method": "bounded-non-affine-face-surface-residual",
        "alignment_anchor": anchor_strategy,
        "outer_anchor_pixels": int(np.count_nonzero(outer_valid)),
        "residual_affine_scale": float(residual_scale),
        "residual_affine_offset": float(residual_offset),
        "face_pixels": int(np.count_nonzero(face_valid)),
        "feather_px": float(feather_px),
        "correction_limit": float(correction_limit),
        "max_abs_correction": float(np.max(np.abs(correction))),
        "mean_abs_correction": float(np.mean(np.abs(correction[face_valid]))),
        "boundary_max_abs_correction": (
            float(np.max(np.abs(correction[boundary])))
            if np.any(boundary)
            else 0.0
        ),
    }
    return refined, weight.astype(np.float32), stats


def _save_preview(values: np.ndarray, path: Path) -> None:
    finite = values[np.isfinite(values)]
    if not finite.size:
        preview = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            high = low + 1.0
        preview = np.clip((values - low) / (high - low), 0.0, 1.0)
        preview = (np.nan_to_num(preview) * 255).astype(np.uint8)
    Image.fromarray(preview).save(path)


def refine_depth_for_faces(
    image_path,
    depth_path,
    output_dir,
    *,
    infer_depth: Callable[[Path, Path], str | Path],
    mode: str = "auto",
    detail_strength: float = DEFAULT_FACE_DETAIL_STRENGTH,
    feather_ratio: float = DEFAULT_FACE_FEATHER_RATIO,
    max_correction_ratio: float = DEFAULT_FACE_MAX_CORRECTION_RATIO,
    max_faces: int = 3,
    min_face_pixels: int = DEFAULT_MIN_FACE_PIXELS,
    detector: Callable[[np.ndarray], tuple[list[dict], list[str]] | list[dict]] | None = None,
    detection_roi_mask: str | Path | np.ndarray | None = None,
    detection_image_path: str | Path | None = None,
    detection_image_mode: str | None = None,
    infer_surface_residual: Callable | None = None,
    enable_gnm_foundation: bool = True,
) -> tuple[str, dict]:
    mode = str(mode or "auto").strip().lower()
    if mode not in FACE_REFINEMENT_MODES:
        raise ValueError(f"Unsupported face refinement mode {mode!r}; expected one of {FACE_REFINEMENT_MODES}")
    metadata = {
        "mode": mode,
        "applied": False,
        "detected_faces": 0,
        "refined_faces": 0,
        "eyewear_deoccluded_faces": 0,
        "faces": [],
        "detector_errors": [],
        "part_mask_schema_version": 1,
        "part_names": list(FACE_PART_NAMES),
        "part_mask_faces": 0,
        "selection_detail_fallback_regions": 0,
        "surface_residual_faces": 0,
        "parametric_foundation_faces": 0,
        "refined_selection_detail_regions": 0,
        "refined_regions_total": 0,
        "minimum_face_pixels": {
            "requested": int(min_face_pixels),
            "effective": None,
        },
        "selection_roi_detection": {"enabled": False, "reason": "not_provided"},
        "detection_image": {"mode": "refinement-image"},
    }
    if mode == "off":
        metadata["reason"] = "disabled"
        return str(depth_path), metadata

    try:
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        global_depth = np.squeeze(np.load(depth_path)).astype(np.float32)
        if global_depth.ndim != 2:
            raise ValueError(f"Expected a 2D global depth map, got {global_depth.shape}")
    except Exception as exc:
        if mode == "on":
            raise
        metadata["reason"] = f"input_unavailable:{type(exc).__name__}:{exc}"
        return str(depth_path), metadata

    detection_rgb = image_rgb
    if detection_image_path is not None:
        try:
            candidate_detection_rgb = np.asarray(
                Image.open(detection_image_path).convert("RGB")
            )
            if candidate_detection_rgb.shape != image_rgb.shape:
                raise ValueError("detection image does not align with refinement image")
            detection_rgb = candidate_detection_rgb
            metadata["detection_image"] = {
                "mode": str(detection_image_mode or "separate-detection-image")
            }
        except Exception as exc:
            metadata["detection_image"] = {
                "mode": "refinement-image-fallback",
                "error": f"{type(exc).__name__}:detection_image_unavailable",
            }

    roi_mask = None
    if detection_roi_mask is not None:
        try:
            if isinstance(detection_roi_mask, (str, Path)):
                roi_mask = np.asarray(
                    Image.open(detection_roi_mask).convert("L")
                )
            else:
                roi_mask = np.asarray(detection_roi_mask)
            if roi_mask.shape != image_rgb.shape[:2]:
                roi_mask = _resize_mask(roi_mask, image_rgb.shape[:2])
            metadata["selection_roi_detection"] = {
                "enabled": True,
                "reason": "mask_loaded",
            }
        except Exception as exc:
            metadata["selection_roi_detection"] = {
                "enabled": True,
                "reason": "roi_mask_load_error",
                "error": f"{type(exc).__name__}:selection_roi_mask_unavailable",
            }

    effective_min_face_pixels = _effective_min_face_pixels(detection_rgb.shape, min_face_pixels)
    metadata["minimum_face_pixels"]["effective"] = int(effective_min_face_pixels)
    request_face_blendshapes = False
    request_facial_transformation_matrixes = False
    if enable_gnm_foundation:
        try:
            try:
                from backend.gnm_face_foundation import (
                    active_gnm_conditioned_feature_requirements,
                )
            except ImportError:
                from gnm_face_foundation import (
                    active_gnm_conditioned_feature_requirements,
                )
            conditioned_requirements = (
                active_gnm_conditioned_feature_requirements()
            )
            request_face_blendshapes = bool(
                conditioned_requirements.get(
                    "face_blendshapes",
                    False,
                )
            )
            request_facial_transformation_matrixes = bool(
                conditioned_requirements.get(
                    "facial_transformation_matrixes",
                    False,
                )
            )
        except Exception:
            request_face_blendshapes = False
            request_facial_transformation_matrixes = False
    metadata["conditioned_feature_requirements"] = {
        "face_blendshapes": request_face_blendshapes,
        "facial_transformation_matrixes": (
            request_facial_transformation_matrixes
        ),
        "same_pass_as_landmarks": True,
    }
    def run_detector(values: np.ndarray, requested_faces: int, requested_minimum: int):
        if detector is not None:
            return detector(values)
        kwargs = {
            "max_faces": requested_faces,
            "min_face_pixels": requested_minimum,
        }
        if request_face_blendshapes:
            kwargs["output_face_blendshapes"] = True
        if request_facial_transformation_matrixes:
            kwargs["output_facial_transformation_matrixes"] = True
        return detect_face_regions(values, **kwargs)

    detection_result = run_detector(detection_rgb, max_faces, effective_min_face_pixels)
    if isinstance(detection_result, tuple):
        regions, detector_errors = detection_result
    else:
        regions, detector_errors = detection_result, []
    metadata["detector_errors"] = [str(error) for error in detector_errors]
    if roi_mask is not None:
        initial_candidates = list(regions)
        regions, initial_selected_faces = _merge_selected_face_regions(
            [],
            initial_candidates,
            roi_mask,
            max_faces=max_faces,
        )
        metadata["selection_roi_detection"].update(
            {
                "full_frame_candidates": int(len(initial_candidates)),
                "full_frame_selected_faces": int(initial_selected_faces),
            }
        )

    if (
        roi_mask is not None
        and detector is None
        and len(regions) < max(1, int(max_faces))
        and effective_min_face_pixels > MIN_FACE_PIXELS_FLOOR
    ):
        relaxed_minimum = max(
            MIN_FACE_PIXELS_FLOOR,
            int(
                round(
                    effective_min_face_pixels
                    * SELECTION_FACE_COMPLETION_MINIMUM_RATIO
                )
            ),
        )
        relaxed_result = run_detector(detection_rgb, max_faces, relaxed_minimum)
        if isinstance(relaxed_result, tuple):
            relaxed_regions, relaxed_errors = relaxed_result
        else:
            relaxed_regions, relaxed_errors = relaxed_result, []
        metadata["detector_errors"].extend(
            f"selection-completion:{error}" for error in relaxed_errors
        )
        regions, relaxed_accepted = _merge_selected_face_regions(
            regions,
            list(relaxed_regions),
            roi_mask,
            max_faces=max_faces,
        )
        metadata["selection_roi_detection"]["full_frame_relaxed_completion"] = {
            "enabled": True,
            "requested_minimum_face_pixels": int(effective_min_face_pixels),
            "effective_minimum_face_pixels": int(relaxed_minimum),
            "candidate_faces": int(len(relaxed_regions)),
            "accepted_missing_faces": int(relaxed_accepted),
        }

    if len(regions) < max(1, int(max_faces)) and roi_mask is not None:
        try:
            roi_kwargs = {
                "max_faces": max_faces,
                "min_face_pixels": min_face_pixels,
                "detector": run_detector,
                "allow_selection_detail_fallback": mode == "auto" and not regions,
            }
            if request_face_blendshapes:
                roi_kwargs["output_face_blendshapes"] = True
            if request_facial_transformation_matrixes:
                roi_kwargs["output_facial_transformation_matrixes"] = True
            roi_regions, roi_errors, roi_stats = detect_face_regions_in_roi(
                detection_rgb,
                roi_mask,
                **roi_kwargs,
            )
            metadata["detector_errors"].extend(str(error) for error in roi_errors)
            fallback_regions = int(
                roi_stats.get("selection_detail_fallback_regions", 0)
            )
            regions, roi_accepted = _merge_selected_face_regions(
                regions,
                list(roi_regions),
                roi_mask,
                max_faces=max_faces,
            )
            metadata["selection_roi_detection"].update(
                {
                    "component_pass": roi_stats,
                    "component_pass_accepted_faces": int(roi_accepted),
                }
            )
            metadata["selection_detail_fallback_regions"] = int(
                fallback_regions
            )
        except Exception as exc:
            metadata["selection_roi_detection"] = {
                "enabled": True,
                "reason": "roi_detection_error",
                "error": f"{type(exc).__name__}:selection_roi_detection_failed",
            }
    metadata["detected_faces"] = int(
        len(regions) - metadata["selection_detail_fallback_regions"]
    )
    if not regions:
        if mode == "on" and detector_errors:
            raise RuntimeError(
                "Face detection failed in required mode: "
                + "; ".join(str(error) for error in detector_errors)
            )
        metadata["reason"] = "no_face_detected"
        return str(depth_path), metadata

    output_dir = Path(output_dir)
    artifact_dir = output_dir / "face_refinement"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    combined_weight = np.zeros(global_depth.shape, dtype=np.float32)
    combined_region = np.zeros(global_depth.shape, dtype=np.uint8)
    combined_occlusion = np.zeros(global_depth.shape, dtype=np.float32)
    image_height, image_width = image_rgb.shape[:2]
    depth_height, depth_width = global_depth.shape
    refined = global_depth.copy()

    for index, region in enumerate(regions[: max(1, int(max_faces))]):
        face_record = {
            "index": index,
            "detector": str(region.get("detector") or "custom"),
            "landmark_count": int(region.get("landmark_count") or 0),
            "bbox": [int(value) for value in region["bbox"]],
            "status": "pending",
        }
        is_selection_detail = (
            region.get("semantic_scope") == "selected-component-detail"
        )
        for audit_key in (
            "confidence",
            "keypoints",
            "model_revision",
            "model_sha256",
            "semantic_scope",
            "fallback_reason",
            "detection_scope",
            "detection_roi_bbox",
            "detection_roi_scale",
        ):
            if audit_key in region:
                face_record[audit_key] = region[audit_key]
        try:
            crop_box = _padded_box(
                region["bbox"],
                image_width,
                image_height,
                DEFAULT_FACE_PADDING_RATIO,
            )
            x0, y0, x1, y1 = crop_box
            face_record["crop_bbox"] = list(crop_box)
            crop_path = artifact_dir / f"face_{index:02d}_input.png"
            Image.fromarray(image_rgb[y0:y1, x0:x1]).save(crop_path)
            face_output_dir = artifact_dir / f"face_{index:02d}_depth"
            face_output_dir.mkdir(parents=True, exist_ok=True)
            local_depth_path = Path(infer_depth(crop_path, face_output_dir))
            local_depth = np.squeeze(np.load(local_depth_path)).astype(np.float32)

            dx0 = int(round(x0 * depth_width / image_width))
            dy0 = int(round(y0 * depth_height / image_height))
            dx1 = int(round(x1 * depth_width / image_width))
            dy1 = int(round(y1 * depth_height / image_height))
            dx0, dy0, dx1, dy1 = _clamp_box((dx0, dy0, dx1, dy1), depth_width, depth_height)
            target_shape = (dy1 - dy0, dx1 - dx0)
            face_mask = _resize_mask(region["face_mask"][y0:y1, x0:x1], target_shape)
            feature_mask = _resize_mask(region["feature_mask"][y0:y1, x0:x1], target_shape)
            selection_crop = (
                _resize_mask(roi_mask[y0:y1, x0:x1], target_shape)
                if roi_mask is not None
                else None
            )
            surface_support_mask, surface_support_stats = (
                face_surface_support_mask(
                    face_mask,
                    selection_crop,
                )
            )
            region_part_masks = region.get("part_masks") or {}
            local_part_masks = {
                name: _resize_mask(
                    region_part_masks[name][y0:y1, x0:x1],
                    target_shape,
                )
                for name in FACE_PART_NAMES
                if region_part_masks.get(name) is not None
            }
            local_depth = _resize_float(local_depth, target_shape)
            surface_residual = None
            surface_residual_inference = {
                "enabled": False,
                "reason": (
                    "selection_detail_fallback"
                    if is_selection_detail
                    else "not_requested"
                ),
            }
            if infer_surface_residual is not None and not is_selection_detail:
                surface_output_dir = artifact_dir / f"face_{index:02d}_surface"
                surface_output_dir.mkdir(parents=True, exist_ok=True)
                inferred = infer_surface_residual(
                    crop_path,
                    local_depth.copy(),
                    face_mask.copy(),
                    feature_mask.copy(),
                    surface_support_mask.copy(),
                    surface_output_dir,
                )
                provider_stats = {}
                if (
                    isinstance(inferred, tuple)
                    and len(inferred) == 2
                    and isinstance(inferred[1], dict)
                ):
                    inferred, provider_stats = inferred
                if isinstance(inferred, (str, Path)):
                    inferred = np.squeeze(np.load(inferred)).astype(np.float32)
                surface_residual = np.squeeze(
                    np.asarray(inferred, dtype=np.float32)
                )
                if surface_residual.ndim != 2:
                    raise ValueError(
                        "Face surface residual inference must return a 2D array"
                    )
                surface_residual = _resize_float(
                    surface_residual,
                    target_shape,
                )
                if not np.all(np.isfinite(surface_residual)):
                    raise ValueError(
                        "Face surface residual inference returned non-finite values"
                    )
                surface_residual_inference = {
                    "enabled": True,
                    **provider_stats,
                }
            source_shape_input = refined[dy0:dy1, dx0:dx1].copy()
            shape_input = source_shape_input
            shape_weight = np.zeros(target_shape, dtype=np.float32)
            shape_prior_stats = {"enabled": False, "reason": "no_relative_z_landmarks"}
            shape_context = None
            landmark_points = None
            eyewear_weight = np.zeros(target_shape, dtype=np.float32)
            eyewear_detection_stats = {
                "enabled": False,
                "reason": "no_relative_z_landmarks",
            }
            landmarks_xyz = region.get("landmarks_xyz")
            if landmarks_xyz is not None:
                try:
                    landmarks_xyz = np.asarray(landmarks_xyz, dtype=np.float64)
                    landmark_x = (
                        (landmarks_xyz[:, 0] * max(image_width - 1, 1) - x0)
                        / max(x1 - x0 - 1, 1)
                        * max(target_shape[1] - 1, 1)
                    )
                    landmark_y = (
                        (landmarks_xyz[:, 1] * max(image_height - 1, 1) - y0)
                        / max(y1 - y0 - 1, 1)
                        * max(target_shape[0] - 1, 1)
                    )
                    landmark_points = np.column_stack((landmark_x, landmark_y))
                    complete_local_parts = bool(
                        len(local_part_masks) == len(FACE_PART_NAMES)
                        and all(
                            np.any(local_part_masks.get(name, 0))
                            for name in FACE_PART_NAMES
                        )
                    )
                    if not complete_local_parts:
                        local_part_masks = _landmark_part_masks(
                            landmark_points,
                            target_shape,
                        )
                    local_part_masks = {
                        name: cv2.bitwise_and(mask, face_mask)
                        for name, mask in local_part_masks.items()
                    }
                    face_image = cv2.resize(
                        image_rgb[y0:y1, x0:x1],
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                    eyewear_weight, eyewear_detection_stats = _detect_eyewear_occlusion_weight(
                        face_image,
                        face_mask,
                        landmark_points,
                    )
                    (
                        shape_input,
                        shape_weight,
                        shape_prior_stats,
                        shape_context,
                    ) = _fuse_face_landmark_shape_prior_with_context(
                        shape_input,
                        face_mask,
                        feature_mask,
                        landmark_points,
                        landmarks_xyz[:, 2],
                        feather_ratio=feather_ratio,
                        max_correction_ratio=max_correction_ratio,
                        minimum_confidence_weight=(
                            EYEWEAR_SHAPE_PRIOR_MIN_CONFIDENCE
                            if eyewear_detection_stats.get("enabled")
                            else 0.0
                        ),
                        correction_limit_scale=(
                            EYEWEAR_SHAPE_PRIOR_CORRECTION_LIMIT_SCALE
                            if eyewear_detection_stats.get("enabled")
                            else 0.50
                        ),
                    )
                except Exception as exc:
                    shape_prior_stats = {
                        "enabled": False,
                        "reason": "prior_error",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                    shape_context = None
                    eyewear_detection_stats = {
                        "enabled": False,
                        "reason": "landmark_processing_error",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
            refined_crop, weight, stats = fuse_face_depth(
                shape_input,
                local_depth,
                face_mask,
                feature_mask,
                detail_strength=detail_strength,
                feather_ratio=feather_ratio,
                max_correction_ratio=max_correction_ratio,
            )
            parametric_weight = np.zeros(target_shape, dtype=np.float32)
            parametric_foundation_stats = {
                "enabled": False,
                "reason": "not_eligible",
            }
            detector_name = str(region.get("detector") or "")
            if not enable_gnm_foundation:
                parametric_foundation_stats["reason"] = "disabled"
            elif is_selection_detail:
                parametric_foundation_stats["reason"] = "selection_detail_fallback"
            elif "mediapipe" not in detector_name.lower():
                parametric_foundation_stats["reason"] = "non_mediapipe_landmarks"
            elif landmark_points is None or len(landmark_points) < 468:
                parametric_foundation_stats["reason"] = "insufficient_landmarks"
            else:
                face_rows, face_columns = np.nonzero(face_mask > 0)
                face_support = (
                    max(
                        int(face_rows.max() - face_rows.min() + 1),
                        int(face_columns.max() - face_columns.min() + 1),
                    )
                    if len(face_rows)
                    else 0
                )
                try:
                    from .gnm_face_foundation import (
                        GNM_MAX_FACE_SUPPORT_PIXELS,
                        GNMSurfaceCandidateSet,
                        fuse_gnm_face_foundation,
                        get_gnm_mean_face_foundation,
                        select_gnm_surface_candidate,
                    )
                except ImportError:
                    from gnm_face_foundation import (
                        GNM_MAX_FACE_SUPPORT_PIXELS,
                        GNMSurfaceCandidateSet,
                        fuse_gnm_face_foundation,
                        get_gnm_mean_face_foundation,
                        select_gnm_surface_candidate,
                    )

                if face_support > GNM_MAX_FACE_SUPPORT_PIXELS:
                    parametric_foundation_stats = {
                        "enabled": False,
                        "reason": "face_support_above_small_face_gate",
                        "support_pixels": face_support,
                        "maximum_support_pixels": GNM_MAX_FACE_SUPPORT_PIXELS,
                    }
                else:
                    try:
                        provider = get_gnm_mean_face_foundation()
                        conditioned = getattr(
                            provider,
                            "fit_and_render_conditioned",
                            None,
                        )
                        if callable(conditioned):
                            conditioned_kwargs = {
                                "media_pipe_landmarks_xyz": landmarks_xyz,
                                "face_image_rgb": image_rgb[y0:y1, x0:x1],
                                "media_pipe_blendshape_names": region.get(
                                    "blendshape_names"
                                ),
                                "media_pipe_blendshape_scores": region.get(
                                    "blendshape_scores"
                                ),
                            }
                            if getattr(
                                provider,
                                "requires_facial_transformation_matrix",
                                False,
                            ) is True:
                                conditioned_kwargs[
                                    "media_pipe_facial_transformation_matrix"
                                ] = region.get(
                                    "facial_transformation_matrix"
                                )
                            gnm_surface, provider_stats = conditioned(
                                landmark_points,
                                face_mask,
                                **conditioned_kwargs,
                            )
                            if isinstance(
                                gnm_surface,
                                GNMSurfaceCandidateSet,
                            ):
                                (
                                    gnm_surface,
                                    selected_provider_stats,
                                    surface_selection,
                                ) = select_gnm_surface_candidate(
                                    refined_crop,
                                    gnm_surface,
                                    face_mask,
                                )
                                structured_stats = dict(
                                    provider_stats.get(
                                        "structured_geometry",
                                        {},
                                    )
                                )
                                structured_stats["surface_selection"] = (
                                    surface_selection
                                )
                                provider_stats = {
                                    **selected_provider_stats,
                                    **provider_stats,
                                    "structured_geometry": structured_stats,
                                }
                        else:
                            gnm_surface, provider_stats = provider.fit_and_render(
                                landmark_points,
                                face_mask,
                            )
                        (
                            refined_crop,
                            parametric_weight,
                            fusion_stats,
                        ) = fuse_gnm_face_foundation(
                            refined_crop,
                            gnm_surface,
                            face_mask,
                            correction_region_mask=np.maximum.reduce(
                                [
                                    local_part_masks[name]
                                    for name in (
                                        "nose",
                                        "left_eye",
                                        "right_eye",
                                    )
                                ]
                            ),
                            preserve_detail_mask=local_part_masks.get("mouth"),
                        )
                        parametric_foundation_stats = {
                            **provider_stats,
                            **fusion_stats,
                        }
                        if parametric_foundation_stats.get("enabled"):
                            metadata["parametric_foundation_faces"] += 1
                    except Exception as exc:
                        parametric_foundation_stats = {
                            "enabled": False,
                            "reason": "provider_error",
                            "error_type": type(exc).__name__,
                        }
            surface_weight = np.zeros(target_shape, dtype=np.float32)
            surface_residual_stats = surface_residual_inference
            if surface_residual is not None:
                requested_support_mode = str(
                    surface_residual_inference.get(
                        "surface_support_mode",
                        "detector-face",
                    )
                )
                if requested_support_mode == "selection-subject":
                    residual_support_mask = surface_support_mask
                    residual_support_stats = surface_support_stats
                elif requested_support_mode == "detector-face":
                    residual_support_mask = face_mask
                    residual_support_stats = {
                        "mode": "detector-face",
                        "reason": "checkpoint_contract",
                        "face_pixels": int(np.count_nonzero(face_mask)),
                        "support_pixels": int(np.count_nonzero(face_mask)),
                        "support_expansion_ratio": 1.0,
                    }
                else:
                    raise ValueError(
                        "Unsupported face surface support mode "
                        f"{requested_support_mode!r}"
                    )
                (
                    refined_crop,
                    surface_weight,
                    residual_fusion_stats,
                ) = fuse_face_surface_residual(
                    refined_crop,
                    local_depth,
                    surface_residual,
                    residual_support_mask,
                    alignment_depth=shape_input,
                    feather_ratio=feather_ratio,
                    max_correction_ratio=max_correction_ratio,
                )
                surface_residual_stats = {
                    **surface_residual_inference,
                    **residual_fusion_stats,
                    "surface_support": residual_support_stats,
                    "surface_support_mode": requested_support_mode,
                }
                metadata["surface_residual_faces"] += 1
            eyewear_deocclusion_stats = {
                "enabled": False,
                "reason": "eyewear_not_detected",
                "detection": eyewear_detection_stats,
            }
            if eyewear_detection_stats.get("enabled"):
                if shape_context is None:
                    eyewear_deocclusion_stats["reason"] = "landmark_prior_unavailable"
                else:
                    refined_crop, reconstruction_stats = _reconstruct_eyewear_occlusion(
                        refined_crop,
                        source_shape_input,
                        shape_context["aligned_prior"],
                        eyewear_weight,
                        reference_span=shape_context["reference_span"],
                        eye_masks={
                            name: local_part_masks[name]
                            for name in ("left_eye", "right_eye")
                            if name in local_part_masks
                        },
                    )
                    eyewear_deocclusion_stats = {
                        **reconstruction_stats,
                        "detection": eyewear_detection_stats,
                    }
            detail_weight = np.maximum(
                np.maximum(
                    np.maximum(weight, shape_weight),
                    parametric_weight,
                ),
                surface_weight,
            )
            if eyewear_deocclusion_stats.get("enabled"):
                detail_weight *= 1.0 - eyewear_weight
                combined_occlusion[dy0:dy1, dx0:dx1] = np.maximum(
                    combined_occlusion[dy0:dy1, dx0:dx1],
                    eyewear_weight,
                )
                metadata["eyewear_deoccluded_faces"] += 1
            refined[dy0:dy1, dx0:dx1] = refined_crop
            combined_weight[dy0:dy1, dx0:dx1] = np.maximum(
                combined_weight[dy0:dy1, dx0:dx1],
                detail_weight,
            )
            if not is_selection_detail:
                combined_region[dy0:dy1, dx0:dx1] = np.maximum(
                    combined_region[dy0:dy1, dx0:dx1],
                    (face_mask > 0).astype(np.uint8) * 255,
                )
            part_mask_files = {}
            if not is_selection_detail and all(
                np.any(local_part_masks.get(name, 0)) for name in FACE_PART_NAMES
            ):
                part_dir = artifact_dir / f"face_{index:02d}_parts"
                part_dir.mkdir(parents=True, exist_ok=True)
                full_face_mask = np.zeros(global_depth.shape, dtype=np.uint8)
                full_face_mask[dy0:dy1, dx0:dx1] = face_mask
                face_mask_path = part_dir / "face.png"
                Image.fromarray(full_face_mask).save(face_mask_path)
                for name in FACE_PART_NAMES:
                    part_path = part_dir / f"{name}.png"
                    full_part_mask = np.zeros(global_depth.shape, dtype=np.uint8)
                    full_part_mask[dy0:dy1, dx0:dx1] = local_part_masks[name]
                    Image.fromarray(full_part_mask).save(part_path)
                    part_mask_files[name] = part_path.relative_to(output_dir).as_posix()
                metadata["part_mask_faces"] += 1
            face_record.update(
                {
                    "status": "refined",
                    "depth_bbox": [dx0, dy0, dx1, dy1],
                    "landmark_shape_prior": shape_prior_stats,
                    "parametric_face_foundation": parametric_foundation_stats,
                    "surface_residual": surface_residual_stats,
                    "eyewear_deocclusion": eyewear_deocclusion_stats,
                    "part_masks": {
                        "schema_version": 1,
                        "coordinate_space": "depth",
                        "face_file": (
                            face_mask_path.relative_to(output_dir).as_posix()
                            if part_mask_files
                            else None
                        ),
                        "files": part_mask_files,
                        "complete": len(part_mask_files) == len(FACE_PART_NAMES),
                    },
                    **stats,
                }
            )
            if is_selection_detail:
                metadata["refined_selection_detail_regions"] += 1
            else:
                metadata["refined_faces"] += 1
            metadata["refined_regions_total"] += 1
        except Exception as exc:
            face_record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        metadata["faces"].append(face_record)

    if not metadata["refined_regions_total"]:
        metadata["reason"] = "detail_depth_inference_failed"
        if mode == "on":
            errors = "; ".join(face.get("error", "unknown") for face in metadata["faces"])
            raise RuntimeError(f"Face refinement was required but no face could be refined: {errors}")
        return str(depth_path), metadata

    refined_path = output_dir / "output_depth_data_face_refined.npy"
    preview_path = output_dir / "output_depth_face_refined_preview.png"
    weight_path = output_dir / "output_face_refinement_weight.png"
    region_path = output_dir / "output_face_refinement_region.png"
    occlusion_path = output_dir / "output_face_refinement_occlusion.png"
    metadata_path = output_dir / "output_face_refinement_metadata.json"
    np.save(refined_path, refined)
    _save_preview(refined, preview_path)
    Image.fromarray((np.clip(combined_weight, 0.0, 1.0) * 255).astype(np.uint8)).save(weight_path)
    Image.fromarray(combined_region).save(region_path)
    Image.fromarray((np.clip(combined_occlusion, 0.0, 1.0) * 255).astype(np.uint8)).save(
        occlusion_path
    )
    metadata.update(
        {
            "applied": True,
            "detail_strength": float(detail_strength),
            "feather_ratio": float(feather_ratio),
            "max_correction_ratio": float(max_correction_ratio),
            "depth_file": refined_path.name,
            "preview_file": preview_path.name,
            "weight_file": weight_path.name,
            "region_file": region_path.name,
            "occlusion_file": occlusion_path.name,
            "metadata_file": metadata_path.name,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return str(refined_path), metadata
