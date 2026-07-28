from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .face_depth_refinement import refine_depth_for_faces
from .pic_to_3d import (
    METRIC_FAR_HIGH_DEPTH_MODELS,
    depth_data_to_3d_model,
    process_image_get_depth_data,
    relief_value_transform_for_model,
)
from .stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
from .video_pipeline import PreparedVideo, VideoSegmentationError, prepare_turntable_video


STL_HARD_CHECKS = (
    "stl_exists",
    "stl_is_watertight",
    "stl_is_volume",
    "stl_is_manifold",
    "stl_winding_consistent",
    "stl_positive_volume",
    "stl_single_component",
)


@dataclass(frozen=True)
class VideoSubjectReliefResult:
    stl_path: Path
    diagnostics_path: Path
    report_path: Path
    preview_path: Path
    selected_frame_path: Path
    selected_mask_path: Path
    prepared: PreparedVideo
    report: dict


def _cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for tracked-subject video relief") from exc
    return cv2


def _mask_row(mask: np.ndarray, source_index: int, position: int) -> dict:
    cv2 = _cv2()
    binary = mask.astype(bool)
    height, width = binary.shape
    pixels = int(np.count_nonzero(binary))
    coverage = pixels / float(max(1, width * height))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)
    ]
    if pixels:
        rows, cols = np.nonzero(binary)
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        centroid_x = float(np.mean(cols) / max(1, width - 1))
        centroid_y = float(np.mean(rows) / max(1, height - 1))
    else:
        x0 = x1 = y0 = y1 = 0
        centroid_x = centroid_y = 0.0
    return {
        "position": int(position),
        "source_index": int(source_index),
        "coverage": float(coverage),
        "largest_component_ratio": max(component_areas, default=0) / float(max(1, pixels)),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "frame_width": int(width),
        "frame_height": int(height),
    }


def analyze_subject_motion(mask_paths: tuple[Path, ...], source_indices: list[int]) -> dict:
    if len(mask_paths) != len(source_indices):
        raise ValueError("Mask paths and source indices must have the same length")
    rows = [
        _mask_row(np.asarray(Image.open(path).convert("L")) > 127, source_index, position)
        for position, (path, source_index) in enumerate(zip(mask_paths, source_indices, strict=True))
    ]
    valid = [row for row in rows if row["coverage"] > 0]
    if len(valid) < 3:
        return {
            "classification": "insufficient-tracking",
            "confidence": 0.0,
            "rows": rows,
            "valid_frame_count": len(valid),
        }

    indices = np.asarray([row["source_index"] for row in valid], dtype=np.float64)
    progress = (indices - indices[0]) / max(1.0, float(indices[-1] - indices[0]))
    coverages = np.asarray([row["coverage"] for row in valid], dtype=np.float64)
    log_coverage = np.log(np.maximum(coverages, 1e-9))
    log_slope = float(np.polyfit(progress, log_coverage, 1)[0])
    fitted_change = float(math.exp(np.clip(log_slope, -20.0, 20.0)))
    start_end_ratio = float(coverages[-1] / max(coverages[0], 1e-9))
    max_min_ratio = float(np.max(coverages) / max(np.min(coverages), 1e-9))
    deltas = np.diff(log_coverage)
    increasing_fraction = float(np.mean(deltas > 0)) if deltas.size else 0.0
    decreasing_fraction = float(np.mean(deltas < 0)) if deltas.size else 0.0
    centroid_jumps = [
        math.hypot(
            valid[index]["centroid_x"] - valid[index - 1]["centroid_x"],
            valid[index]["centroid_y"] - valid[index - 1]["centroid_y"],
        )
        for index in range(1, len(valid))
    ]
    max_centroid_jump = float(max(centroid_jumps, default=0.0))

    if start_end_ratio >= 1.8 and fitted_change >= 1.8 and increasing_fraction >= 0.6:
        classification = "approach"
        confidence = min(1.0, math.log(max(start_end_ratio, 1.0)) / math.log(8.0))
    elif start_end_ratio <= 1.0 / 1.8 and fitted_change <= 1.0 / 1.8 and decreasing_fraction >= 0.6:
        classification = "recede"
        confidence = min(1.0, math.log(max(1.0 / start_end_ratio, 1.0)) / math.log(8.0))
    elif max_min_ratio <= 2.5:
        classification = "constant-scale"
        confidence = float(np.clip(1.0 - (max_min_ratio - 1.0) / 1.5, 0.0, 1.0))
    else:
        classification = "ambiguous-scale-change"
        confidence = 0.0

    return {
        "classification": classification,
        "confidence": float(confidence),
        "valid_frame_count": len(valid),
        "start_end_coverage_ratio": start_end_ratio,
        "fitted_coverage_change": fitted_change,
        "coverage_max_min_ratio": max_min_ratio,
        "increasing_step_fraction": increasing_fraction,
        "decreasing_step_fraction": decreasing_fraction,
        "max_centroid_jump": max_centroid_jump,
        "rows": rows,
    }


def select_best_subject_view(prepared: PreparedVideo, motion: dict) -> dict:
    cv2 = _cv2()
    source_indices = prepared.report["sampling"]["selected_source_indices"]
    rows = motion["rows"]
    raw_sharpness = []
    candidates = []
    for position, (frame_path, mask_path, source_index, row) in enumerate(
        zip(prepared.frame_paths, prepared.mask_paths, source_indices, rows, strict=True)
    ):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        interior = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian[interior])) if np.count_nonzero(interior) >= 32 else 0.0
        raw_sharpness.append(sharpness)

        bbox = row["bbox"]
        width = int(row["frame_width"])
        height = int(row["frame_height"])
        height_ratio = bbox["height"] / float(max(1, height))
        margins = (
            bbox["x"] / float(max(1, width)),
            (width - bbox["x"] - bbox["width"]) / float(max(1, width)),
            bbox["y"] / float(max(1, height)),
            (height - bbox["y"] - bbox["height"]) / float(max(1, height)),
        )
        candidates.append(
            {
                "position": int(position),
                "source_index": int(source_index),
                "frame_path": str(frame_path),
                "mask_path": str(mask_path),
                "height_ratio": float(height_ratio),
                "minimum_border_margin_ratio": float(min(margins)),
                "coverage": float(row["coverage"]),
                "largest_component_ratio": float(row["largest_component_ratio"]),
                "object_sharpness": sharpness,
            }
        )

    sharpness_values = np.asarray(raw_sharpness, dtype=np.float64)
    sharp_low, sharp_high = np.percentile(sharpness_values, [10.0, 90.0])
    sharp_span = max(float(sharp_high - sharp_low), 1e-9)
    for candidate in candidates:
        sharpness_score = float(
            np.clip((candidate["object_sharpness"] - sharp_low) / sharp_span, 0.0, 1.0)
        )
        size_score = float(np.clip(candidate["height_ratio"] / 0.58, 0.0, 1.0))
        visibility_score = float(
            np.clip(candidate["minimum_border_margin_ratio"] / 0.08, 0.0, 1.0)
        )
        component_score = float(np.clip(candidate["largest_component_ratio"], 0.0, 1.0))
        candidate["score_components"] = {
            "subject_resolution": size_score,
            "full_visibility": visibility_score,
            "object_sharpness": sharpness_score,
            "mask_coherence": component_score,
        }
        candidate["selection_score"] = float(
            0.45 * size_score
            + 0.30 * visibility_score
            + 0.20 * sharpness_score
            + 0.05 * component_score
        )
        if candidate["coverage"] < 0.005 or candidate["largest_component_ratio"] < 0.75:
            candidate["selection_score"] = -1.0

    selected = max(candidates, key=lambda row: (row["selection_score"], row["source_index"]))
    if selected["selection_score"] < 0:
        raise VideoSegmentationError("No tracked frame is suitable for subject-relief generation")
    return {"selected": selected, "candidates": candidates}


def _crop_subject_view(frame_path: Path, mask_path: Path, output_dir: Path) -> tuple[Path, Path, dict]:
    frame = np.asarray(Image.open(frame_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    rows, cols = np.nonzero(mask)
    if not rows.size:
        raise VideoSegmentationError("The selected subject frame has an empty mask")
    height, width = mask.shape
    bbox_width = int(cols.max() - cols.min() + 1)
    bbox_height = int(rows.max() - rows.min() + 1)
    pad_x = max(16, int(round(bbox_width * 0.35)))
    pad_y = max(16, int(round(bbox_height * 0.10)))
    x0 = max(0, int(cols.min()) - pad_x)
    x1 = min(width, int(cols.max()) + pad_x + 1)
    y0 = max(0, int(rows.min()) - pad_y)
    y1 = min(height, int(rows.max()) + pad_y + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / "selected_subject.png"
    crop_mask_path = output_dir / "selected_subject_mask.png"
    Image.fromarray(frame[y0:y1, x0:x1]).save(crop_path)
    Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L").save(crop_mask_path)
    return crop_path, crop_mask_path, {
        "source_bounds": [x0, y0, x1, y1],
        "source_size": [int(width), int(height)],
        "crop_size": [int(x1 - x0), int(y1 - y0)],
        "padding_pixels": {"x": int(pad_x), "y": int(pad_y)},
    }


def generate_video_subject_relief(
    video_path: Path,
    output_dir: Path,
    *,
    selected_frame_count: int = 12,
    frame_selection: str = "sharpness-motion-selector",
    segmentation_provider: str = "sam2.1-hiera-tiny-video",
    segmentation_device: str = "auto",
    object_point: tuple[float, float] = (0.5, 0.4),
    strict_segmentation: bool = True,
    frame_max_side: int = 960,
    depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf",
    depth_device: str = "auto",
    target_dimension: int = 360,
    max_xy_size_mm: float = 120.0,
    relief_height_mm: float = 8.0,
    minimum_feature_mm: float = 0.4,
) -> VideoSubjectReliefResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}

    stage = time.perf_counter()
    prepared = prepare_turntable_video(
        Path(video_path),
        output_dir / "tracking",
        selected_frame_count=selected_frame_count,
        frame_selection=frame_selection,
        segmentation_provider=segmentation_provider,
        segmentation_device=segmentation_device,
        object_point=object_point,
        strict_segmentation=False,
        frame_max_side=frame_max_side,
    )
    timings["tracking_seconds"] = round(time.perf_counter() - stage, 3)

    source_indices = prepared.report["sampling"]["selected_source_indices"]
    motion = analyze_subject_motion(prepared.mask_paths, source_indices)
    mask_failures = list(prepared.report["segmentation"]["quality"]["failed_checks"])
    accepted_failures = []
    if motion["classification"] in {"approach", "recede"} and "mask_area_stability" in mask_failures:
        mask_failures.remove("mask_area_stability")
        accepted_failures.append("mask_area_stability:expected_perspective_scale_change")
    if strict_segmentation and mask_failures:
        raise VideoSegmentationError(
            "Tracked subject masks failed quality checks: " + ", ".join(mask_failures),
            report={"motion": motion, "mask_quality": prepared.report["segmentation"]["quality"]},
        )

    selection = select_best_subject_view(prepared, motion)
    selected = selection["selected"]
    selected_frame_path = prepared.frame_paths[int(selected["position"])]
    selected_mask_path = prepared.mask_paths[int(selected["position"])]
    crop_path, crop_mask_path, crop = _crop_subject_view(
        selected_frame_path, selected_mask_path, output_dir / "subject"
    )

    stage = time.perf_counter()
    depth_dir = output_dir / "depth"
    depth_path = Path(
        process_image_get_depth_data(
            crop_path,
            output_dir=depth_dir,
            provider="transformers",
            model_name=depth_model,
            device=depth_device,
        )
    )

    def infer_face_depth(face_crop: Path, face_output_dir: Path):
        return process_image_get_depth_data(
            face_crop,
            output_dir=face_output_dir,
            provider="transformers",
            model_name=depth_model,
            device=depth_device,
        )

    depth_path, face_refinement = refine_depth_for_faces(
        crop_path,
        depth_path,
        depth_dir,
        infer_depth=infer_face_depth,
        mode="auto",
    )
    depth_path = Path(depth_path)
    depth = np.squeeze(np.load(depth_path)).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D subject depth map, got {depth.shape}")
    cv2 = _cv2()
    mask = np.asarray(Image.open(crop_mask_path).convert("L")) > 127
    resized_mask = cv2.resize(
        mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
    ) > 0
    masked_depth_path = depth_dir / "subject_depth_masked.npy"
    np.save(masked_depth_path, np.where(resized_mask, depth, np.nan).astype(np.float32))
    timings["depth_seconds"] = round(time.perf_counter() - stage, 3)

    def refinement_artifact(name: str) -> Path | None:
        value = face_refinement.get(name)
        return depth_dir / value if face_refinement.get("applied") and value else None

    stage = time.perf_counter()
    stl_path = output_dir / "output_model.stl"
    relief_postprocess = depth_data_to_3d_model(
        masked_depth_path,
        output_stl_path=str(stl_path),
        target_dimension=int(target_dimension),
        z_scale=float(relief_height_mm),
        max_xy_size=float(max_xy_size_mm),
        invert=depth_model in METRIC_FAR_HIGH_DEPTH_MODELS,
        sigma=0.45,
        relief_gamma=0.8,
        detail_boost=0.8,
        background_detail_boost=1.0,
        source_image=crop_path,
        background_photo_detail_mm=0.08,
        trim_top_background=False,
        feature_weight_mask=refinement_artifact("weight_file"),
        feature_exclusion_mask=refinement_artifact("occlusion_file"),
        printable_feature_depth_mm=0.35,
        feature_bridge_depth_mm=0.6,
        low_percentile=1.0,
        high_percentile=99.0,
        base_border_px=0,
        value_transform=relief_value_transform_for_model(depth_model),
        minimum_feature_mm=float(minimum_feature_mm),
        max_relief_slope=2.0,
        face_region_mask=refinement_artifact("region_file"),
    )
    timings["stl_seconds"] = round(time.perf_counter() - stage, 3)
    diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
    failed_stl_checks = [key for key in STL_HARD_CHECKS if not bool(diagnostics.get(key))]
    diagnostics["stl_passes_hard_checks"] = not failed_stl_checks
    diagnostics["stl_failed_checks"] = failed_stl_checks
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    timings["total_seconds"] = round(time.perf_counter() - started, 3)

    report = {
        "status": "printable" if not failed_stl_checks else "stl-emitted",
        "source_video": str(Path(video_path).resolve()),
        "runner": "tracked-subject-video-relief",
        "geometry_contract": "best visible tracked frame -> full-context depth -> masked 2.5D relief STL",
        "motion": motion,
        "accepted_mask_failures": accepted_failures,
        "remaining_mask_failures": mask_failures,
        "selection": selection,
        "crop": crop,
        "depth_model": depth_model,
        "depth_device": depth_device,
        "face_refinement": face_refinement,
        "relief": {
            "height_mm": float(relief_height_mm),
            "max_xy_size_mm": float(max_xy_size_mm),
            "target_dimension": int(target_dimension),
            "postprocess": relief_postprocess,
        },
        "stl_diagnostics": diagnostics,
        "timings": timings,
    }
    report_path = output_dir / "video_subject_relief.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return VideoSubjectReliefResult(
        stl_path=stl_path,
        diagnostics_path=diagnostics_path,
        report_path=report_path,
        preview_path=crop_path,
        selected_frame_path=selected_frame_path,
        selected_mask_path=selected_mask_path,
        prepared=prepared,
        report=report,
    )
