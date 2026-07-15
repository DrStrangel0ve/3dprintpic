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
YUNET_MAX_INPUT_DIMENSION = 1024
DETECTOR_MODEL_PATH_ENVIRONMENT_NAMES = (
    "FACE_LANDMARKER_MODEL_PATH",
    "YUNET_FACE_DETECTOR_MODEL_PATH",
)
DEFAULT_MIN_FACE_PIXELS = 96
MIN_FACE_PIXELS_FLOOR = 48
MIN_FACE_IMAGE_RATIO = 0.25
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
EYEWEAR_MINIMUM_DARK_COVERAGE = 0.50
EYEWEAR_MINIMUM_COMPONENT_COVERAGE = 0.45
EYEWEAR_MINIMUM_COMPONENT_WIDTH_RATIO = 0.60
EYEWEAR_ACCESSORY_RESIDUAL_RATIO = 0.02
EYEWEAR_MAXIMUM_CORRECTION_RATIO = 0.15


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


def _detect_faces_mediapipe_tasks(image_rgb: np.ndarray, max_faces: int, min_face_pixels: int) -> list[dict]:
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
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))

    height, width = image_rgb.shape[:2]
    regions = []
    for landmarks in result.face_landmarks or ():
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
        x0, y0, x1, y1 = region["bbox"]
        if min(x1 - x0, y1 - y0) >= min_face_pixels:
            regions.append(region)
    return regions


def _detect_faces_mediapipe(image_rgb: np.ndarray, max_faces: int, min_face_pixels: int) -> list[dict]:
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    face_mesh_api = getattr(solutions, "face_mesh", None) if solutions is not None else None
    if face_mesh_api is None:
        return _detect_faces_mediapipe_tasks(image_rgb, max_faces, min_face_pixels)

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


def _yunet_keypoints_are_face_like(keypoints: np.ndarray, box) -> bool:
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
    if abs(float(eyes[1, 0] - eyes[0, 0])) < width * 0.18:
        return False
    if abs(float(mouth[1, 0] - mouth[0, 0])) < width * 0.12:
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


def _detect_faces_yunet(image_rgb: np.ndarray, max_faces: int, min_face_pixels: int) -> list[dict]:
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("OpenCV FaceDetectorYN is unavailable")
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
        YUNET_SCORE_THRESHOLD,
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
        if confidence < YUNET_SCORE_THRESHOLD:
            continue
        x, y, box_width, box_height = detection[:4] * inverse_scale
        box = _clamp_box((x, y, x + box_width, y + box_height), width, height)
        x0, y0, x1, y1 = box
        if min(x1 - x0, y1 - y0) < int(min_face_pixels):
            continue
        keypoints = detection[4:14].reshape(5, 2) * inverse_scale
        if not _yunet_keypoints_are_face_like(keypoints, box):
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
        raise RuntimeError(f"OpenCV face detector could not load {cascade_path}")
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
) -> tuple[list[dict], list[str]]:
    min_face_pixels = _effective_min_face_pixels(image_rgb.shape, min_face_pixels)
    errors = []
    try:
        regions = _detect_faces_mediapipe(image_rgb, max_faces, min_face_pixels)
        if regions:
            return regions, errors
    except Exception as exc:
        errors.append(_detector_error_record("mediapipe", exc))
    try:
        regions = _detect_faces_yunet(image_rgb, max_faces, min_face_pixels)
        if regions:
            return regions, errors
    except Exception as exc:
        errors.append(_detector_error_record("yunet", exc))
    try:
        return _detect_faces_opencv(image_rgb, max_faces, min_face_pixels), errors
    except Exception as exc:
        errors.append(_detector_error_record("opencv-haar", exc))
        return [], errors


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
    correction_limit = reference_span * max(0.0, float(max_correction_ratio)) * 0.50
    correction = np.clip(correction, -correction_limit, correction_limit)
    confidence_weight = float(
        np.clip(
            (abs(correlation) - float(minimum_abs_correlation))
            / max(0.60 - float(minimum_abs_correlation), 1e-6),
            0.0,
            1.0,
        )
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
        "minimum_abs_correlation": float(minimum_abs_correlation),
        "yaw_proxy": float(yaw_proxy),
        "maximum_yaw_proxy": float(maximum_yaw_proxy),
        "scale": scale,
        "offset": offset,
        "coarse_sigma_px": coarse_sigma_px,
        "feather_px": float(feather_px),
        "correction_limit": float(correction_limit),
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
    )
    return refined, weight, stats


def _reconstruct_eyewear_occlusion(
    refined_depth: np.ndarray,
    source_depth: np.ndarray,
    aligned_prior: np.ndarray,
    occlusion_weight: np.ndarray,
    *,
    reference_span: float,
    accessory_residual_ratio: float = EYEWEAR_ACCESSORY_RESIDUAL_RATIO,
    maximum_correction_ratio: float = EYEWEAR_MAXIMUM_CORRECTION_RATIO,
    minimum_input_residual_ratio: float = 0.05,
    maximum_output_residual_ratio: float = 0.05,
    minimum_residual_reduction_ratio: float = 0.35,
    maximum_saturated_core_ratio: float = 0.05,
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
        "minimum_face_pixels": {
            "requested": int(min_face_pixels),
            "effective": None,
        },
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

    effective_min_face_pixels = _effective_min_face_pixels(image_rgb.shape, min_face_pixels)
    metadata["minimum_face_pixels"]["effective"] = int(effective_min_face_pixels)
    detection_result = detector(image_rgb) if detector else detect_face_regions(
        image_rgb,
        max_faces=max_faces,
        min_face_pixels=effective_min_face_pixels,
    )
    if isinstance(detection_result, tuple):
        regions, detector_errors = detection_result
    else:
        regions, detector_errors = detection_result, []
    metadata["detector_errors"] = [str(error) for error in detector_errors]
    metadata["detected_faces"] = int(len(regions))
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
        for audit_key in ("confidence", "keypoints", "model_revision", "model_sha256"):
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
            source_shape_input = refined[dy0:dy1, dx0:dx1].copy()
            shape_input = source_shape_input
            shape_weight = np.zeros(target_shape, dtype=np.float32)
            shape_prior_stats = {"enabled": False, "reason": "no_relative_z_landmarks"}
            shape_context = None
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
                    )
                    eyewear_deocclusion_stats = {
                        **reconstruction_stats,
                        "detection": eyewear_detection_stats,
                    }
            detail_weight = np.maximum(weight, shape_weight)
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
            combined_region[dy0:dy1, dx0:dx1] = np.maximum(
                combined_region[dy0:dy1, dx0:dx1],
                (face_mask > 0).astype(np.uint8) * 255,
            )
            part_mask_files = {}
            if all(np.any(local_part_masks.get(name, 0)) for name in FACE_PART_NAMES):
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
            metadata["refined_faces"] += 1
        except Exception as exc:
            face_record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        metadata["faces"].append(face_record)

    if not metadata["refined_faces"]:
        metadata["reason"] = "face_depth_inference_failed"
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
