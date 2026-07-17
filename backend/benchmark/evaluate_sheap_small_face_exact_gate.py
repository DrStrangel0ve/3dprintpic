"""Gate pinned SHeaP geometry on the hardest exact CC0 face row."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    SUBJECT_BOUNDARY_PIXELS,
    build_small_face_candidate,
)
from backend.benchmark.face_part_metrics import (
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    _exact_face_depth_quality,
)
from backend.benchmark.run_makehuman_face_depth_smoke import (
    _correlation,
    _resize_nan_aware,
)
from backend.benchmark.sheap_depth_provider import (
    SHEAP_FACE_CROP_MARGIN,
    SHEAP_FACE_CROP_SHIFT_UP,
    SHeaPProvider,
    apply_sheap_image_similarity,
    fit_sheap_landmark_similarity,
    project_sheap_vertices,
    rasterize_sheap_camera_depth,
    sheap_head_crop_bbox_from_landmarks,
)
from backend.benchmark.vggheads_depth_provider import (
    fill_depth_nearest,
    normalize_front_depth,
    subject_interior_taper,
)
from backend.face_depth_refinement import (
    DEFAULT_FACE_FEATHER_RATIO,
    DEFAULT_FACE_MAX_CORRECTION_RATIO,
    _fit_face_depth,
    _resize_float,
    _resize_mask,
    _robust_span,
    fuse_face_surface_residual,
)


METHOD = "sheap-camera-depth-low-frequency-fusion"
ROW_ID = "small_side_lit_shelves_256"
EXPECTED_SOURCE_SHA256 = (
    "2aca3775162f1b2320a47dbfead27e49e2d7b29654d80151d8ebc17315a8c074"
)
EXPECTED_SELECTION_SHA256 = (
    "853d871ac859dc0904cd866bc8e552a55efbc3af8ba59d2da3ee4aac3583d0ef"
)
EXPECTED_EXACT_DEPTH_SHA256 = (
    "a21f6396c57abd850786e00d40cf2776e048f7b8d0664170e86cdca8fa27a21b"
)
EXPECTED_BASELINE_SHA256 = (
    "417202239d046203027c01d8a9c2f8a6b8a5e057b95385174b144575f5df4fe6"
)
EXPECTED_METADATA_SHA256 = (
    "2a5c858b75542ec05494daeeff79cb19439fcc0da6695fec95f70a87656744d2"
)
EXPECTED_LOCAL_FACE_DEPTH_SHA256 = (
    "e1f4d129185ddd56ebf5cf0541952fa8470e87da1caf8f550aa493ad266dbab6"
)
EXPECTED_FACE_MASK_SHA256 = (
    "929245e9e1192e09cfcf7ea246e04f8392bfb092bff346bbbd74f8d85d165cbc"
)
EXPECTED_PART_MASK_SHA256 = {
    "left_eye": "76ffb7dc7df0fb93f94278276260ee9d97bfc09b31128593dafddfc06c4ee85f",
    "left_eyebrow": "2b0f98d2eb9d0b0b56e3ec5fcf0aad58b99ff0cbd391589d253c035eda9ee1a3",
    "mouth": "25ef20b733f902fd5fee88249a59a10b9fb48e563c402822bd3b0decf7c52c99",
    "nose": "fc8de05a6f93302a375f8822dff6f9e81196a65fbcfcb840c15c09ebd72bda8e",
    "right_eye": "3bb1b841ecccd9fdfc2779dfefd4898343be1c1a9b85a95a911716578bfc10d1",
    "right_eyebrow": "f60d4c00e0234472047f16e71c0929ba2029fd733c605def5bf6f1230adc726d",
}
EXPECTED_SELECTION_BBOX_XYXY = [130, 91, 211, 167]
EXPECTED_FACE_BBOX_XYXY = [162, 102, 193, 139]
PROVIDER_ALPHAS = (0.125, 0.25, 0.5)
MINIMUM_PROVIDER_FACE_COVERAGE = 0.50
COMPARISON_EPSILON = 1e-9
SHEAP_MEDIAPIPE_EMBEDDING_SIZE_BYTES = 4_518
SHEAP_MEDIAPIPE_EMBEDDING_SHA256 = (
    "8863363013fc6fe3752d4318ea02a1784970200616a54b785920cdd0568816fc"
)
FACE_LANDMARKER_MODEL_SIZE_BYTES = 3_758_596
FACE_LANDMARKER_MODEL_SHA256 = (
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
)
EXPECTED_CROP_LANDMARKS_SHA256 = (
    "2a7f0ca0376626b97b46e37827e92af6729bbe824f5e3ce776f3ef6cb3288a0c"
)
EXPECTED_CROP_LANDMARK_COUNT = 478
MINIMUM_REGISTRATION_SCALE = 0.90
MAXIMUM_REGISTRATION_SCALE = 1.10
MAXIMUM_REGISTRATION_ROTATION_DEGREES = 5.0
MAXIMUM_REGISTRATION_MEDIAN_ERROR_PIXELS = 1.0
MAXIMUM_REGISTRATION_P95_ERROR_PIXELS = 2.0

EXPECTED_INPUT_HASH_KEYS = frozenset(
    {
        "source",
        "selection_mask",
        "exact_depth",
        "baseline",
        "metadata",
        "local_face_depth",
        "face_mask",
        "mediapipe_embedding",
        "face_landmarker_model",
        *(f"part_mask:{name}" for name in EXPECTED_PART_MASK_SHA256),
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(
        json.dumps(
            list(contiguous.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _expected_input_hashes() -> dict[str, str]:
    expected = {
        "source": EXPECTED_SOURCE_SHA256,
        "selection_mask": EXPECTED_SELECTION_SHA256,
        "exact_depth": EXPECTED_EXACT_DEPTH_SHA256,
        "baseline": EXPECTED_BASELINE_SHA256,
        "metadata": EXPECTED_METADATA_SHA256,
        "local_face_depth": EXPECTED_LOCAL_FACE_DEPTH_SHA256,
        "face_mask": EXPECTED_FACE_MASK_SHA256,
        "mediapipe_embedding": SHEAP_MEDIAPIPE_EMBEDDING_SHA256,
        "face_landmarker_model": FACE_LANDMARKER_MODEL_SHA256,
    }
    expected.update(
        {
            f"part_mask:{name}": sha256
            for name, sha256 in EXPECTED_PART_MASK_SHA256.items()
        }
    )
    if set(expected) != EXPECTED_INPUT_HASH_KEYS:
        raise RuntimeError("SHeaP exact input pin schema is incomplete")
    return expected


def _verify_input_hashes(input_paths: dict[str, Path]) -> dict[str, str]:
    expected = _expected_input_hashes()
    if set(input_paths) != set(expected):
        missing = sorted(set(expected) - set(input_paths))
        extra = sorted(set(input_paths) - set(expected))
        raise ValueError(
            "SHeaP exact input path schema changed: "
            f"missing={missing}, extra={extra}"
        )
    actual = {name: _sha256(path) for name, path in input_paths.items()}
    mismatches = sorted(
        name for name in expected if actual[name] != expected[name]
    )
    if mismatches:
        raise ValueError(
            "SHeaP exact gate input hashes changed: "
            + ", ".join(mismatches)
        )
    return actual


def _measurement_source_manifest() -> dict:
    file_callables = (
        build_small_face_candidate,
        fill_depth_nearest,
        normalize_front_depth,
        subject_interior_taper,
        _exact_face_depth_quality,
        face_part_cross_height_metrics,
        face_part_affine_surface_error_metrics,
        _correlation,
        SHeaPProvider,
    )
    function_callables = (
        *file_callables,
        fuse_face_surface_residual,
        _fit_face_depth,
        _resize_float,
        _resize_mask,
        _robust_span,
        _resize_nan_aware,
    )
    repo_root = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve()}
    for item in file_callables:
        source_path = inspect.getsourcefile(item)
        if source_path is None:
            raise RuntimeError(
                f"Cannot locate measurement source for {item!r}"
            )
        paths.add(Path(source_path).resolve())
    files = {}
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = str(path)
        files[relative] = _sha256(path)
    functions = {}
    for item in function_callables:
        name = f"{item.__module__}.{item.__qualname__}"
        functions[name] = hashlib.sha256(
            inspect.getsource(item).encode("utf-8")
        ).hexdigest()
    return {
        "schema_version": 1,
        "files": files,
        "functions": functions,
        "captured_defaults": {
            "face_feather_ratio": DEFAULT_FACE_FEATHER_RATIO,
            "face_max_correction_ratio": (
                DEFAULT_FACE_MAX_CORRECTION_RATIO
            ),
        },
    }


def _runtime_attestation() -> dict:
    return {
        "python": platform.python_version(),
        "packages": {
            "mediapipe": _distribution_version("mediapipe"),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": _distribution_version("Pillow"),
            "scipy": _distribution_version("scipy"),
            "torch": _distribution_version("torch"),
            "torchvision": _distribution_version("torchvision"),
            "roma": _distribution_version("roma"),
        },
    }


def _detect_pinned_face_landmarks(
    image_rgb: np.ndarray,
    model_path: Path,
    *,
    max_faces: int,
    min_face_pixels: int,
) -> tuple[list[dict], dict]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {model_path}")
    size_bytes = int(model_path.stat().st_size)
    model_sha256 = _sha256(model_path)
    if size_bytes != FACE_LANDMARKER_MODEL_SIZE_BYTES:
        raise ValueError("MediaPipe face-landmarker model size changed")
    if model_sha256 != FACE_LANDMARKER_MODEL_SHA256:
        raise ValueError("MediaPipe face-landmarker model hash changed")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=int(max_faces),
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        )

    height, width = image_rgb.shape[:2]
    detections = []
    for landmarks in list(result.face_landmarks or ()):
        landmarks_xyz = np.asarray(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in landmarks
            ],
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
        if not len(points):
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0) + 1
        if min(int(x1 - x0), int(y1 - y0)) < int(min_face_pixels):
            continue
        detections.append(
            {
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "detector": "mediapipe-face-landmarker",
                "landmark_count": int(len(landmarks_xyz)),
                "landmarks_xyz": landmarks_xyz,
                "landmarks_sha256": _array_sha256(landmarks_xyz),
            }
        )
    return detections, {
        "provider": "mediapipe-face-landmarker",
        "package_version": getattr(mp, "__version__", None),
        "model": {
            "path": str(model_path),
            "size_bytes": size_bytes,
            "sha256": model_sha256,
            "size_pinned": True,
            "hash_pinned": True,
        },
        "settings": {
            "running_mode": "IMAGE",
            "num_faces": int(max_faces),
            "min_face_detection_confidence": 0.5,
            "min_face_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
    }


def _selection_bbox(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(np.asarray(mask) > 0)
    if not len(rows):
        raise ValueError("Exact SHeaP gate has an empty selection mask")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]


def _bbox_iou(first: list[int], second: list[int]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in first)
    bx0, by0, bx1, by1 = (float(value) for value in second)
    overlap_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    overlap = overlap_width * overlap_height
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - overlap
    return float(overlap / union) if union > 0 else 0.0


def _registration_checks(fit: dict, detection_iou: float) -> dict[str, bool]:
    return {
        "embedding_pinned": True,
        "correspondence_count": fit["correspondences"] == 105,
        "detection_matches_baseline": detection_iou >= 0.80,
        "scale_bounded": (
            MINIMUM_REGISTRATION_SCALE
            <= fit["scale"]
            <= MAXIMUM_REGISTRATION_SCALE
        ),
        "rotation_bounded": (
            abs(fit["rotation_degrees"])
            <= MAXIMUM_REGISTRATION_ROTATION_DEGREES
        ),
        "median_residual_bounded": (
            fit["residual_median_pixels"]
            <= MAXIMUM_REGISTRATION_MEDIAN_ERROR_PIXELS
        ),
        "p95_residual_bounded": (
            fit["residual_p95_pixels"]
            <= MAXIMUM_REGISTRATION_P95_ERROR_PIXELS
        ),
    }


def _landmark_similarity_registration(
    inference: dict,
    image_rgb: np.ndarray,
    expected_face_bbox: list[int],
    embedding_path: Path,
    detection: dict,
) -> tuple[np.ndarray, dict]:
    if embedding_path.stat().st_size != SHEAP_MEDIAPIPE_EMBEDDING_SIZE_BYTES:
        raise ValueError("SHeaP MediaPipe embedding size changed")
    embedding_sha256 = _sha256(embedding_path)
    if embedding_sha256 != SHEAP_MEDIAPIPE_EMBEDDING_SHA256:
        raise ValueError("SHeaP MediaPipe embedding hash changed")
    with np.load(embedding_path) as embedding:
        required = {"lmk_face_idx", "lmk_b_coords", "landmark_indices"}
        if set(embedding.files) != required:
            raise ValueError("SHeaP MediaPipe embedding schema changed")
        face_indices = embedding["lmk_face_idx"].astype(np.int64)
        barycentric = embedding["lmk_b_coords"].astype(np.float64)
        landmark_indices = embedding["landmark_indices"].astype(np.int64)
    if (
        face_indices.shape != (105,)
        or barycentric.shape != (105, 3)
        or landmark_indices.shape != (105,)
        or not np.all(np.isfinite(barycentric))
        or not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("SHeaP MediaPipe embedding contents changed")
    faces = np.asarray(inference["faces"], dtype=np.int64)
    world_vertices = np.asarray(inference["world_vertices"], dtype=np.float64)
    if np.any(face_indices < 0) or np.any(face_indices >= len(faces)):
        raise ValueError("SHeaP MediaPipe embedding references invalid faces")
    landmark_world = np.sum(
        world_vertices[faces[face_indices]] * barycentric[:, :, None],
        axis=1,
    )
    projected_landmarks = project_sheap_vertices(
        landmark_world,
        inference["metadata"]["crop"],
    )

    detection_iou = _bbox_iou(detection["bbox"], expected_face_bbox)
    observed = np.asarray(detection["landmarks_xyz"], dtype=np.float64)
    if observed.shape[0] <= int(np.max(landmark_indices)):
        raise ValueError("SHeaP registration landmark count changed")
    height, width = image_rgb.shape[:2]
    observed_xy = observed[landmark_indices, :2] * np.array(
        [width - 1, height - 1],
        dtype=np.float64,
    )
    matrix, fit = fit_sheap_landmark_similarity(
        projected_landmarks[:, :2],
        observed_xy,
    )
    checks = _registration_checks(fit, detection_iou)
    record = {
        "embedding": {
            "path": str(embedding_path),
            "size_bytes": SHEAP_MEDIAPIPE_EMBEDDING_SIZE_BYTES,
            "sha256": embedding_sha256,
            "source": "pinned SMIRK FLAME MediaPipe embedding",
        },
        "detector": detection["detector"],
        "detector_landmarks_sha256": detection["landmarks_sha256"],
        "detected_bbox_xyxy": [int(value) for value in detection["bbox"]],
        "baseline_bbox_iou": detection_iou,
        "matrix": matrix.tolist(),
        "fit": fit,
        "checks": checks,
        "accepted": bool(all(checks.values())),
    }
    return matrix, record


def _quality(
    candidate_path: Path,
    exact_depth_path: Path,
    selection_mask_path: Path,
    part_mask_paths: dict[str, Path],
) -> dict:
    return _exact_face_depth_quality(
        candidate_path,
        exact_depth_path,
        selection_mask_path,
        expected_scale_sign=-1.0,
        part_mask_paths=part_mask_paths,
    )


def _compact_quality(metrics: dict) -> dict:
    shape = metrics["named_part_shape"]
    affine = metrics["named_part_affine_mm"]
    return {
        "shape_correlation": float(metrics["shape_correlation"]),
        "gradient_correlation": float(metrics["gradient_correlation"]),
        "normalized_rmse": float(metrics["normalized_rmse"]),
        "checks": metrics["checks"],
        "shape_failed_parts": list(shape["failed_parts"]),
        "affine_failed_parts": list(affine["failed_parts"]),
        "combined_part_failures": int(
            len(shape["failed_parts"]) + len(affine["failed_parts"])
        ),
        "parts": {
            part["name"]: {
                "shape_correlation": float(part["shape_correlation"]),
                "face_normalized_shape_rmse": float(
                    part["face_normalized_shape_rmse"]
                ),
                "minimum_raw_gradient_correlation": float(
                    part["minimum_raw_gradient_correlation"]
                ),
                "shape_passed": bool(part["passed"]),
            }
            for part in shape["parts"]
        },
        "affine_parts": {
            part["name"]: {
                "rmse_mm": float(part["rmse_mm"]),
                "p95_absolute_error_mm": float(
                    part["p95_absolute_error_mm"]
                ),
                "bias_mm": float(part["bias_mm"]),
                "span_retention": float(part["span_retention"]),
                "affine_passed": bool(part["passed"]),
            }
            for part in affine["parts"]
        },
    }


def _compare_parts(baseline: dict, candidate: dict) -> dict:
    baseline_shape = baseline["parts"]
    candidate_shape = candidate["parts"]
    baseline_affine = baseline["affine_parts"]
    candidate_affine = candidate["affine_parts"]
    if (
        set(baseline_shape) != set(candidate_shape)
        or set(baseline_affine) != set(candidate_affine)
        or set(baseline_shape) != set(baseline_affine)
    ):
        raise ValueError("SHeaP candidate has mismatched exact face parts")

    rows = []
    for name in sorted(baseline_shape):
        before_shape = baseline_shape[name]
        after_shape = candidate_shape[name]
        before_affine = baseline_affine[name]
        after_affine = candidate_affine[name]
        checks = {
            "shape_correlation_non_regression": (
                after_shape["shape_correlation"] + COMPARISON_EPSILON
                >= before_shape["shape_correlation"]
            ),
            "shape_rmse_non_regression": (
                after_shape["face_normalized_shape_rmse"]
                <= before_shape["face_normalized_shape_rmse"]
                + COMPARISON_EPSILON
            ),
            "raw_gradient_non_regression": (
                after_shape["minimum_raw_gradient_correlation"]
                + COMPARISON_EPSILON
                >= before_shape["minimum_raw_gradient_correlation"]
            ),
            "affine_rmse_non_regression": (
                after_affine["rmse_mm"]
                <= before_affine["rmse_mm"] + COMPARISON_EPSILON
            ),
            "affine_p95_non_regression": (
                after_affine["p95_absolute_error_mm"]
                <= before_affine["p95_absolute_error_mm"]
                + COMPARISON_EPSILON
            ),
            "affine_bias_non_regression": (
                abs(after_affine["bias_mm"])
                <= abs(before_affine["bias_mm"]) + COMPARISON_EPSILON
            ),
            "affine_span_non_regression": (
                abs(after_affine["span_retention"] - 1.0)
                <= abs(before_affine["span_retention"] - 1.0)
                + COMPARISON_EPSILON
            ),
        }
        rows.append(
            {
                "name": name,
                "checks": checks,
                "passed": bool(all(checks.values())),
                "baseline": {
                    **before_shape,
                    **before_affine,
                },
                "candidate": {
                    **after_shape,
                    **after_affine,
                },
            }
        )
    passed_checks = sum(
        int(passed)
        for row in rows
        for passed in row["checks"].values()
    )
    total_checks = sum(len(row["checks"]) for row in rows)
    strict_improvements = sum(
        int(
            (
                row["candidate"]["shape_correlation"]
                > row["baseline"]["shape_correlation"]
                + COMPARISON_EPSILON
            )
            and (
                row["candidate"]["minimum_raw_gradient_correlation"]
                > row["baseline"]["minimum_raw_gradient_correlation"]
                + COMPARISON_EPSILON
            )
            and (
                row["candidate"]["rmse_mm"]
                < row["baseline"]["rmse_mm"] - COMPARISON_EPSILON
            )
        )
        for row in rows
    )
    return {
        "parts": rows,
        "passed_checks": int(passed_checks),
        "total_checks": int(total_checks),
        "all_part_metrics_non_regressing": bool(
            passed_checks == total_checks
        ),
        "parts_with_strict_shape_gradient_affine_improvement": int(
            strict_improvements
        ),
    }


def _select_variant_and_decision(
    variants: list[dict],
) -> tuple[list[dict], dict]:
    if not variants:
        raise ValueError("SHeaP exact gate produced no variants")
    ordered = sorted(
        variants,
        key=lambda item: (
            item["eligible"],
            item["comparison"]["passed_checks"],
            -item["quality"]["combined_part_failures"],
            item["quality"]["shape_correlation"],
            item["quality"]["gradient_correlation"],
            -item["quality"]["normalized_rmse"],
        ),
        reverse=True,
    )
    selected = ordered[0]
    eligible = bool(selected["eligible"])
    decision = {
        "status": "eligible-for-30mm-stl-replay" if eligible else "hold",
        "eligible_for_30mm_stl_replay": eligible,
        "selected_variant": selected["variant_id"],
        "selected_passed_part_metric_checks": int(
            selected["comparison"]["passed_checks"]
        ),
        "selected_total_part_metric_checks": int(
            selected["comparison"]["total_checks"]
        ),
        "failed_checks": [
            name
            for name, passed in selected["checks"].items()
            if not passed
        ],
        "production_changed": False,
        "next_action": (
            "run-bounded-30mm-stl-replay"
            if eligible
            else "retain-current-production-face-depth"
        ),
    }
    return ordered, decision


def _write_depth_preview(path: Path, depth: np.ndarray) -> None:
    finite = np.isfinite(depth)
    if not np.any(finite):
        raise ValueError("Cannot preview depth without finite pixels")
    low, high = np.percentile(depth[finite], (1.0, 99.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("Cannot preview non-finite depth percentiles")
    span = max(float(high - low), 1e-8)
    normalized = np.clip((depth - low) / span, 0.0, 1.0)
    normalized[~finite] = 0.0
    Image.fromarray(
        np.rint(normalized * 255.0).astype(np.uint8),
        mode="L",
    ).save(path)


def evaluate(
    baseline_row_dir: str | Path,
    exact_row_dir: str | Path,
    provider_root: str | Path,
    checkpoint_path: str | Path,
    flame_tensor_path: str | Path,
    flame_source_path: str | Path,
    eyelids_path: str | Path,
    mediapipe_embedding_path: str | Path,
    face_landmarker_model_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    model_type: str = "expressive",
) -> dict:
    started = time.perf_counter()
    baseline_row_dir = Path(baseline_row_dir).resolve()
    exact_row_dir = Path(exact_row_dir).resolve()
    mediapipe_embedding_path = Path(mediapipe_embedding_path).resolve()
    face_landmarker_model_path = Path(face_landmarker_model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type = str(model_type)

    source_path = exact_row_dir / "source.png"
    selection_mask_path = exact_row_dir / "selection_mask.png"
    exact_depth_path = exact_row_dir / "exact_depth.npy"
    part_dir = exact_row_dir / "exact_face_parts"
    baseline_path = baseline_row_dir / "output_depth_data_face_refined.npy"
    metadata_path = baseline_row_dir / "output_face_refinement_metadata.json"
    local_depth_path = (
        baseline_row_dir
        / "face_refinement"
        / "face_00_depth"
        / "output_depth_data.npy"
    )
    face_mask_path = (
        baseline_row_dir
        / "face_refinement"
        / "face_00_parts"
        / "face.png"
    )
    part_mask_paths = {
        name: part_dir / f"{name}.png"
        for name in (
            "left_eye",
            "left_eyebrow",
            "mouth",
            "nose",
            "right_eye",
            "right_eyebrow",
        )
    }
    input_paths = {
        "source": source_path,
        "selection_mask": selection_mask_path,
        "exact_depth": exact_depth_path,
        "baseline": baseline_path,
        "metadata": metadata_path,
        "local_face_depth": local_depth_path,
        "face_mask": face_mask_path,
        "mediapipe_embedding": mediapipe_embedding_path,
        "face_landmarker_model": face_landmarker_model_path,
        **{
            f"part_mask:{name}": path
            for name, path in part_mask_paths.items()
        },
    }
    missing = [
        f"{name}={path}"
        for name, path in input_paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "SHeaP exact gate artifacts are unavailable: "
            + ", ".join(missing)
        )
    input_hashes = _verify_input_hashes(input_paths)
    input_hashes_pinned = True

    with Image.open(selection_mask_path) as loaded:
        selection_mask = np.asarray(loaded.convert("L")) > 0
    with Image.open(source_path) as loaded:
        image_rgb = np.asarray(loaded.convert("RGB"))
    with Image.open(face_mask_path) as loaded:
        face_mask = np.asarray(loaded.convert("L"))
    selection_bbox = _selection_bbox(selection_mask)
    if selection_bbox != EXPECTED_SELECTION_BBOX_XYXY:
        raise ValueError(
            f"SHeaP exact selection bbox changed: {selection_bbox}"
        )
    baseline_depth = np.load(baseline_path).astype(np.float32)
    local_face_depth = np.load(local_depth_path).astype(np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    face_record = metadata["faces"][0]
    crop_bbox = face_record["crop_bbox"]
    face_bbox = [int(value) for value in face_record["bbox"]]
    if face_bbox != EXPECTED_FACE_BBOX_XYXY:
        raise ValueError(f"SHeaP exact face bbox changed: {face_bbox}")
    detections, detector_provenance = _detect_pinned_face_landmarks(
        image_rgb,
        face_landmarker_model_path,
        max_faces=3,
        min_face_pixels=8,
    )
    if not detections:
        raise RuntimeError("SHeaP exact crop found no MediaPipe face")
    detections.sort(
        key=lambda record: _bbox_iou(record["bbox"], face_bbox),
        reverse=True,
    )
    crop_detection = detections[0]
    crop_detection_iou = _bbox_iou(crop_detection["bbox"], face_bbox)
    if crop_detection_iou < 0.80:
        raise RuntimeError("SHeaP exact crop face disagrees with baseline")
    if crop_detection["landmark_count"] != EXPECTED_CROP_LANDMARK_COUNT:
        raise RuntimeError("SHeaP exact crop landmark count changed")
    if (
        crop_detection["landmarks_sha256"]
        != EXPECTED_CROP_LANDMARKS_SHA256
    ):
        raise RuntimeError("SHeaP exact crop landmarks changed")
    image_height, image_width = image_rgb.shape[:2]
    crop_landmarks_xy = np.asarray(
        crop_detection["landmarks_xyz"],
        dtype=np.float64,
    )[:, :2] * np.array([image_width, image_height], dtype=np.float64)
    target_bbox = sheap_head_crop_bbox_from_landmarks(
        crop_landmarks_xy,
        image_width=image_width,
        image_height=image_height,
    )

    provider = SHeaPProvider(
        provider_root,
        checkpoint_path,
        flame_tensor_path,
        flame_source_path,
        eyelids_path,
        device=device,
        model_type=model_type,
    )
    inference = provider.infer(
        source_path,
        target_bbox_xyxy=target_bbox,
    )
    if inference["metadata"]["crop"]["crop_clamped"]:
        raise RuntimeError("SHeaP exact crop unexpectedly reached an image edge")
    camera_depth, camera_raster = rasterize_sheap_camera_depth(
        inference["vertices"],
        inference["faces"],
        height=baseline_depth.shape[0],
        width=baseline_depth.shape[1],
    )
    provider_paths = {
        "camera_depth": output_dir / "provider_depth_camera.npy",
        "projected_vertices": output_dir / "provider_projected_vertices.npy",
        "world_vertices": output_dir / "provider_world_vertices.npy",
        "faces": output_dir / "provider_faces.npy",
    }
    np.save(provider_paths["camera_depth"], camera_depth)
    np.save(provider_paths["projected_vertices"], inference["vertices"])
    np.save(provider_paths["world_vertices"], inference["world_vertices"])
    np.save(provider_paths["faces"], inference["faces"])
    _write_depth_preview(output_dir / "provider_depth_camera.png", camera_depth)
    provider_artifacts = {
        name: {
            "file": path.name,
            "sha256": _sha256(path),
        }
        for name, path in provider_paths.items()
    }

    registration_matrix, registration = _landmark_similarity_registration(
        inference,
        image_rgb,
        face_bbox,
        mediapipe_embedding_path,
        crop_detection,
    )
    geometry_variants = [
        {
            "geometry_variant": "camera",
            "depth": camera_depth,
            "raster": camera_raster,
            "registration": None,
        }
    ]
    if registration["accepted"]:
        aligned_vertices = apply_sheap_image_similarity(
            inference["vertices"],
            registration_matrix,
        )
        aligned_depth, aligned_raster = rasterize_sheap_camera_depth(
            aligned_vertices,
            inference["faces"],
            height=baseline_depth.shape[0],
            width=baseline_depth.shape[1],
        )
        aligned_vertices_path = (
            output_dir / "provider_projected_vertices_landmark_similarity.npy"
        )
        aligned_depth_path = (
            output_dir / "provider_depth_landmark_similarity.npy"
        )
        np.save(aligned_vertices_path, aligned_vertices)
        np.save(aligned_depth_path, aligned_depth)
        provider_artifacts["landmark_similarity_vertices"] = {
            "file": aligned_vertices_path.name,
            "sha256": _sha256(aligned_vertices_path),
        }
        provider_artifacts["landmark_similarity_depth"] = {
            "file": aligned_depth_path.name,
            "sha256": _sha256(aligned_depth_path),
        }
        _write_depth_preview(
            output_dir / "provider_depth_landmark_similarity.png",
            aligned_depth,
        )
        geometry_variants.append(
            {
                "geometry_variant": "landmark_similarity",
                "depth": aligned_depth,
                "raster": aligned_raster,
                "registration": registration,
            }
        )

    x0, y0, x1, y1 = (int(value) for value in crop_bbox)
    expected_face_pixels = int(
        np.count_nonzero(face_mask[y0:y1, x0:x1] > 0)
    )
    for geometry in geometry_variants:
        provider_crop = geometry["depth"][y0:y1, x0:x1]
        local_mask = face_mask[y0:y1, x0:x1] > 0
        face_pixels = int(
            np.count_nonzero(np.isfinite(provider_crop) & local_mask)
        )
        geometry["face_finite_pixels"] = face_pixels
        geometry["face_expected_pixels"] = expected_face_pixels
        geometry["face_coverage"] = (
            face_pixels / expected_face_pixels
            if expected_face_pixels
            else 0.0
        )

    baseline_metrics = _quality(
        baseline_path,
        exact_depth_path,
        selection_mask_path,
        part_mask_paths,
    )
    baseline_compact = _compact_quality(baseline_metrics)
    baseline_metrics_path = output_dir / "baseline_metrics.json"
    baseline_metrics_path.write_text(
        json.dumps(baseline_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    variants = []
    selection_distance = cv2.distanceTransform(
        selection_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    boundary = (
        (selection_distance > 0)
        & (selection_distance <= SUBJECT_BOUNDARY_PIXELS)
    )
    for geometry in geometry_variants:
        for alpha in PROVIDER_ALPHAS:
            candidate_depth, fusion = build_small_face_candidate(
                baseline_depth,
                local_face_depth,
                geometry["depth"],
                face_mask,
                selection_mask,
                crop_bbox,
                provider_alpha=alpha,
            )
            fusion["method"] = f"{METHOD}:{model_type}"
            variant_id = (
                f"{geometry['geometry_variant']}_a{alpha:.3f}"
            )
            candidate_path = output_dir / f"candidate_{variant_id}.npy"
            np.save(candidate_path, candidate_depth)
            _write_depth_preview(
                output_dir / f"candidate_{variant_id}.png",
                candidate_depth,
            )
            metrics = _quality(
                candidate_path,
                exact_depth_path,
                selection_mask_path,
                part_mask_paths,
            )
            compact = _compact_quality(metrics)
            comparison = _compare_parts(baseline_compact, compact)
            metrics_name = f"candidate_{variant_id}_metrics.json"
            (output_dir / metrics_name).write_text(
                json.dumps(metrics, indent=2) + "\n",
                encoding="utf-8",
            )
            background_exact = bool(
                np.array_equal(
                    candidate_depth[~selection_mask],
                    baseline_depth[~selection_mask],
                )
            )
            boundary_exact = bool(
                not np.any(boundary)
                or np.array_equal(
                    candidate_depth[boundary],
                    baseline_depth[boundary],
                )
            )
            checks = {
                "provider_face_coverage": bool(
                    geometry["face_coverage"]
                    >= MINIMUM_PROVIDER_FACE_COVERAGE
                ),
                "provider_depth_orientation_positive_scale": bool(
                    math.isfinite(float(fusion["provider_scale"]))
                    and float(fusion["provider_scale"]) > 0
                ),
                "all_six_part_metrics_non_regressing": bool(
                    comparison["all_part_metrics_non_regressing"]
                ),
                "all_six_parts_strict_shape_gradient_affine_improvement": bool(
                    comparison[
                        "parts_with_strict_shape_gradient_affine_improvement"
                    ]
                    == 6
                ),
                "combined_part_failures_non_regressing": bool(
                    compact["combined_part_failures"]
                    <= baseline_compact["combined_part_failures"]
                ),
                "overall_shape_non_regressing": bool(
                    compact["shape_correlation"] + COMPARISON_EPSILON
                    >= baseline_compact["shape_correlation"]
                ),
                "overall_gradient_non_regressing": bool(
                    compact["gradient_correlation"] + COMPARISON_EPSILON
                    >= baseline_compact["gradient_correlation"]
                ),
                "overall_rmse_non_regressing": bool(
                    compact["normalized_rmse"]
                    <= baseline_compact["normalized_rmse"]
                    + COMPARISON_EPSILON
                ),
                "background_value_exact": background_exact,
                "attachment_boundary_value_exact": boundary_exact,
            }
            variants.append(
                {
                    "variant_id": variant_id,
                    "geometry_variant": geometry["geometry_variant"],
                    "provider_alpha": float(alpha),
                    "provider_face_coverage": float(
                        geometry["face_coverage"]
                    ),
                    "candidate_sha256": _sha256(candidate_path),
                    "metrics_file": metrics_name,
                    "metrics_sha256": _sha256(output_dir / metrics_name),
                    "quality": compact,
                    "comparison": comparison,
                    "fusion": fusion,
                    "checks": checks,
                    "eligible": bool(all(checks.values())),
                }
            )

    variants, decision = _select_variant_and_decision(variants)
    evidence = {
        "schema_version": 2,
        "method": f"{METHOD}:{model_type}",
        "row_id": ROW_ID,
        "privacy": "CC0 MakeHuman synthetic identity and procedural scene only",
        "source_geometry": "evaluation-only",
        "harness": _measurement_source_manifest(),
        "runtime": _runtime_attestation(),
        "production_changed": False,
        "input_hashes": input_hashes,
        "input_hashes_pinned": input_hashes_pinned,
        "selection_bbox_xyxy": selection_bbox,
        "detected_face_bbox_xyxy": face_bbox,
        "crop_detector_bbox_xyxy": [
            int(value) for value in crop_detection["bbox"]
        ],
        "crop_detector_baseline_iou": float(crop_detection_iou),
        "crop_detector_landmark_count": int(
            crop_detection["landmark_count"]
        ),
        "crop_detector_landmarks_sha256": crop_detection[
            "landmarks_sha256"
        ],
        "detector": detector_provenance,
        "provider_head_crop_bbox_xyxy": target_bbox,
        "configuration": {
            "provider_alphas": list(PROVIDER_ALPHAS),
            "minimum_provider_face_coverage": MINIMUM_PROVIDER_FACE_COVERAGE,
            "front_surface": "minimum-positive-camera-depth",
            "depth_interpolation": "perspective-correct-reciprocal-depth",
            "head_crop_policy": {
                "source": (
                    "official SHeaP video_demo.py formula with MediaPipe "
                    "landmarks replacing face_alignment landmarks"
                ),
                "detector": crop_detection["detector"],
                "face_landmark_margin": SHEAP_FACE_CROP_MARGIN,
                "face_landmark_shift_up": SHEAP_FACE_CROP_SHIFT_UP,
                "integer_slice": True,
                "resize": "torchvision bilinear antialias",
            },
            "landmark_registration_guards": {
                "minimum_scale": MINIMUM_REGISTRATION_SCALE,
                "maximum_scale": MAXIMUM_REGISTRATION_SCALE,
                "maximum_absolute_rotation_degrees": (
                    MAXIMUM_REGISTRATION_ROTATION_DEGREES
                ),
                "maximum_median_error_pixels": (
                    MAXIMUM_REGISTRATION_MEDIAN_ERROR_PIXELS
                ),
                "maximum_p95_error_pixels": (
                    MAXIMUM_REGISTRATION_P95_ERROR_PIXELS
                ),
            },
            "strict_part_metrics": [
                "shape_correlation",
                "face_normalized_shape_rmse",
                "minimum_raw_gradient_correlation",
                "affine_rmse_mm",
                "affine_p95_absolute_error_mm",
                "affine_absolute_bias_mm",
                "affine_span_distance_from_one",
            ],
        },
        "provider": {
            **provider.provenance(),
            "inference": inference["metadata"],
            "artifacts": provider_artifacts,
            "registration": registration,
            "geometry_variants": [
                {
                    "geometry_variant": geometry["geometry_variant"],
                    "raster": geometry["raster"],
                    "face_coverage": float(geometry["face_coverage"]),
                    "face_finite_pixels": int(
                        geometry["face_finite_pixels"]
                    ),
                    "face_expected_pixels": int(
                        geometry["face_expected_pixels"]
                    ),
                }
                for geometry in geometry_variants
            ],
        },
        "baseline": {
            "sha256": input_hashes["baseline"],
            "metrics_file": "baseline_metrics.json",
            "metrics_sha256": _sha256(baseline_metrics_path),
            "quality": baseline_compact,
        },
        "variants": variants,
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-row-dir", required=True)
    parser.add_argument("--exact-row-dir", required=True)
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--flame-tensor-path", required=True)
    parser.add_argument("--flame-source-path", required=True)
    parser.add_argument("--eyelids-path", required=True)
    parser.add_argument("--mediapipe-embedding-path", required=True)
    parser.add_argument("--face-landmarker-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-type",
        choices=("expressive", "paper"),
        default="expressive",
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.baseline_row_dir,
        args.exact_row_dir,
        args.provider_root,
        args.checkpoint_path,
        args.flame_tensor_path,
        args.flame_source_path,
        args.eyelids_path,
        args.mediapipe_embedding_path,
        args.face_landmarker_model_path,
        args.output_dir,
        device=args.device,
        model_type=args.model_type,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_30mm_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
