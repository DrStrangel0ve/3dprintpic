"""Train a face residual through the production detector and fusion contract."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _near_high_target,
    _padded_box,
)
from backend.benchmark.train_face_surface_adapter import (
    BLEND_ALPHAS,
    CRITICAL_PART_CHECKS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NETWORK_SIZE,
    GRADIENT_LOSS_WEIGHT,
    MAX_NORMALIZED_RESIDUAL,
    MINIMUM_GLOBAL_GRADIENT_RETENTION,
    MULTISCALE_LOSS_WEIGHT,
    NORMAL_LOSS_WEIGHT,
    PART_WEIGHT,
    RESIDUAL_REGULARIZATION_WEIGHT,
    TRAINING_SEED,
    VALUE_LOSS_WEIGHT,
    _axis_gradient_loss,
    _coordinate_channels,
    _minmax_normalize,
    _per_row_non_regression,
    _resize,
    _sha256,
    _strictly_improves,
    _surface_normals,
    _weighted_mean,
    build_surface_adapter,
    positive_affine_fit,
)
from backend.face_depth_refinement import (
    DEFAULT_FACE_MAX_CORRECTION_RATIO,
    MIN_FACE_PIXELS_FLOOR,
    SELECTION_ROI_DETECTION_DIMENSION,
    _clamp_box,
    _map_roi_face_region,
    detect_face_regions,
    detect_face_regions_in_roi,
    face_surface_support_mask,
    fuse_face_surface_residual,
    refine_depth_for_faces,
)


CACHE_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
METHOD = "cc0_production_native_face_surface_residual"
SUBJECT_SUPPORTED_METHOD = (
    "cc0_production_native_subject_supported_face_surface_residual"
)
SURFACE_SUPPORT_MODES = ("detector-face", "selection-subject")
NEUTRAL_BACKGROUND = (245, 245, 245)
SELECTION_FEATHER_RADIUS = 1.5
VALUE_NON_REGRESSION_WEIGHT = 0.35
GRADIENT_NON_REGRESSION_WEIGHT = 0.75
CURVATURE_NON_REGRESSION_WEIGHT = 0.50
DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX = 90.0
DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT = 1.5
DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT = 0.0
DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT = 0.0
DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT = 0.0
DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT = 0.0
DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT = 0.0


@dataclass
class CachedFusionSurface:
    row: dict
    rgb: np.ndarray
    local_depth: np.ndarray
    baseline: np.ndarray
    target: np.ndarray
    exact_face: np.ndarray
    support_face: np.ndarray
    fusion_weight: np.ndarray
    parts: np.ndarray
    alignment_scale: float
    correction_limit: float
    bbox: tuple[int, int, int, int]
    baseline_full: np.ndarray
    local_native: np.ndarray
    support_native: np.ndarray
    surface_support_mode: str
    surface_support_stats: dict


def selected_image_from_exact_mask(
    source: Image.Image,
    selection_mask: Image.Image,
) -> Image.Image:
    soft_mask = selection_mask.convert("L").filter(
        ImageFilter.GaussianBlur(radius=SELECTION_FEATHER_RADIUS)
    )
    background = Image.new("RGB", source.size, NEUTRAL_BACKGROUND)
    return Image.composite(source.convert("RGB"), background, soft_mask)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        return np.asarray(loaded.convert("L")) >= 128


def _detect_production_region(
    image_rgb: np.ndarray,
    selection_mask: np.ndarray,
    *,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
) -> tuple[dict, dict]:
    detection_kwargs = {"max_faces": 1, "min_face_pixels": 96}
    if output_face_blendshapes:
        detection_kwargs["output_face_blendshapes"] = True
    if output_facial_transformation_matrixes:
        detection_kwargs["output_facial_transformation_matrixes"] = True
    regions, errors = detect_face_regions(image_rgb, **detection_kwargs)
    detection_scope = "full-image"
    roi_stats = {"enabled": False, "reason": "full_image_face_detected"}
    if not regions:
        roi_kwargs = {
            "max_faces": 1,
            "min_face_pixels": 96,
            "allow_selection_detail_fallback": False,
        }
        if output_face_blendshapes:
            roi_kwargs["output_face_blendshapes"] = True
        if output_facial_transformation_matrixes:
            roi_kwargs["output_facial_transformation_matrixes"] = True
        regions, roi_errors, roi_stats = detect_face_regions_in_roi(
            image_rgb,
            selection_mask,
            **roi_kwargs,
        )
        errors.extend(roi_errors)
        detection_scope = "selection-roi"
    if not regions:
        rows, columns = np.nonzero(np.asarray(selection_mask) > 0)
        if rows.size and columns.size:
            image_height, image_width = image_rgb.shape[:2]
            component_box = (
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            )
            component_width = component_box[2] - component_box[0]
            component_height = component_box[3] - component_box[1]
            center_x = 0.5 * (component_box[0] + component_box[2])
            center_y = 0.5 * (component_box[1] + component_box[3])
            for padding_ratio in (0.25, 0.10, 0.0):
                side = max(component_width, component_height) * (
                    1.0 + 2.0 * padding_ratio
                )
                roi_box = _clamp_box(
                    (
                        center_x - 0.5 * side,
                        center_y - 0.5 * side,
                        center_x + 0.5 * side,
                        center_y + 0.5 * side,
                    ),
                    image_width,
                    image_height,
                )
                x0, y0, x1, y1 = roi_box
                crop = image_rgb[y0:y1, x0:x1]
                if not crop.size:
                    continue
                scale = SELECTION_ROI_DETECTION_DIMENSION / float(
                    max(crop.shape[:2])
                )
                target_width = max(1, int(round(crop.shape[1] * scale)))
                target_height = max(1, int(round(crop.shape[0] * scale)))
                resized = cv2.resize(
                    crop,
                    (target_width, target_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                local_kwargs = {
                    "max_faces": 1,
                    "min_face_pixels": MIN_FACE_PIXELS_FLOOR,
                }
                if output_face_blendshapes:
                    local_kwargs["output_face_blendshapes"] = True
                if output_facial_transformation_matrixes:
                    local_kwargs[
                        "output_facial_transformation_matrixes"
                    ] = True
                local_regions, local_errors = detect_face_regions(
                    resized,
                    **local_kwargs,
                )
                errors.extend(
                    f"tight-roi-{padding_ratio:.2f}:{error}"
                    for error in local_errors
                )
                for local in local_regions:
                    mapped = _map_roi_face_region(
                        local,
                        roi_box=roi_box,
                        roi_shape=crop.shape[:2],
                        image_shape=image_rgb.shape,
                        scale_x=target_width / float(crop.shape[1]),
                        scale_y=target_height / float(crop.shape[0]),
                        component_mask=np.asarray(selection_mask) > 0,
                    )
                    if mapped is not None:
                        regions = [mapped]
                        detection_scope = (
                            f"selection-roi-tight-{padding_ratio:.2f}"
                        )
                        roi_stats = {
                            "enabled": True,
                            "training_only_tight_retry": True,
                            "padding_ratio": padding_ratio,
                            "roi_bbox": list(roi_box),
                            "input_shape": [target_height, target_width],
                        }
                        break
                if regions:
                    break
    if not regions:
        raise RuntimeError(
            "Production face detector found no face: " + "; ".join(errors)
        )
    region = regions[0]
    if region.get("semantic_scope") == "selected-component-detail":
        raise RuntimeError("Selection-detail fallback is not a face training target")
    return region, {
        "scope": detection_scope,
        "errors": errors,
        "roi": roi_stats,
    }


def _infer_depth_pair(
    processor,
    model,
    full_image: Image.Image,
    crop: Image.Image,
    *,
    device: str,
    dtype,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    outputs = []
    for image in (full_image, crop):
        pixel_values = processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(device=device, dtype=dtype)
        with torch.inference_mode():
            predicted = model(pixel_values=pixel_values).predicted_depth
        native = torch.nn.functional.interpolate(
            predicted[:, None],
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        outputs.append(_minmax_normalize(native.float().cpu().numpy()))
    return outputs[0], outputs[1]


def _production_baseline(
    selected: Image.Image,
    global_depth: np.ndarray,
    local_depth: np.ndarray,
    region: dict,
) -> tuple[np.ndarray, dict]:
    expected_crop = None

    def cached_local(image_path, output_dir):
        actual = np.asarray(Image.open(image_path).convert("RGB"))
        if expected_crop is None or not np.array_equal(actual, expected_crop):
            raise ValueError("Production baseline crop differs from cached crop")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output_depth_data.npy"
        np.save(path, local_depth)
        return str(path)

    width, height = selected.size
    crop_box = _padded_box(
        region["bbox"],
        width,
        height,
        0.35,
    )
    expected_crop = np.asarray(selected.crop(crop_box).convert("RGB"))
    with tempfile.TemporaryDirectory(prefix="face-fusion-cache-") as temporary:
        temporary = Path(temporary)
        selected_path = temporary / "selected.png"
        depth_path = temporary / "global.npy"
        selected.save(selected_path)
        np.save(depth_path, global_depth)
        refined_path, metadata = refine_depth_for_faces(
            selected_path,
            depth_path,
            temporary / "refined",
            infer_depth=cached_local,
            detector=lambda _image: [copy.deepcopy(region)],
            mode="on",
        )
        refined = np.load(refined_path).astype(np.float32)
    if metadata.get("refined_faces") != 1:
        raise RuntimeError("Production-native cache did not refine exactly one face")
    return refined, metadata


def _cache_matches(
    cache_root: Path,
    *,
    corpus_summary_sha256: str,
    network_size: int,
    row_count: int,
) -> bool:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        manifest.get("schema_version") == CACHE_SCHEMA_VERSION
        and manifest.get("method") == METHOD
        and manifest.get("corpus_summary_sha256") == corpus_summary_sha256
        and manifest.get("model_id") == MODEL_ID
        and manifest.get("model_revision") == MODEL_REVISION
        and manifest.get("network_size") == int(network_size)
        and manifest.get("row_count") == int(row_count)
        and len(manifest.get("row_ids", [])) == int(row_count)
        and all(
            (cache_root / "rows" / f"{row_id}.npz").is_file()
            for row_id in manifest.get("row_ids", [])
        )
    )


def prepare_cache(
    corpus_root: str | Path,
    cache_root: str | Path,
    *,
    device: str = "cuda",
    network_size: int = DEFAULT_NETWORK_SIZE,
    force: bool = False,
) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    summary_path = corpus_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary["rows"]
    summary_sha = _sha256(summary_path)
    if not force and _cache_matches(
        cache_root,
        corpus_summary_sha256=summary_sha,
        network_size=network_size,
        row_count=len(rows),
    ):
        return json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))

    cache_root.mkdir(parents=True, exist_ok=True)
    row_root = cache_root / "rows"
    row_root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    processor = AutoImageProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=False,
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    prepared_rows = []
    detector_coverages = []
    detector_precisions = []
    detector_scopes = {}
    started = time.perf_counter()
    network_shape = (int(network_size), int(network_size))
    for index, row in enumerate(rows, start=1):
        source = Image.open(corpus_root / row["source"]["path"]).convert("RGB")
        selection_image = Image.open(
            corpus_root / row["selection_mask"]["path"]
        ).convert("L")
        selection = np.asarray(selection_image) >= 128
        selected = selected_image_from_exact_mask(source, selection_image)
        selected_rgb = np.asarray(selected)
        region, detection = _detect_production_region(
            selected_rgb,
            selection.astype(np.uint8) * 255,
        )
        detector_scopes[detection["scope"]] = (
            detector_scopes.get(detection["scope"], 0) + 1
        )
        support_full = np.asarray(region["face_mask"]) > 0
        intersection = np.count_nonzero(support_full & selection)
        detector_coverages.append(
            float(intersection / max(np.count_nonzero(selection), 1))
        )
        detector_precisions.append(
            float(intersection / max(np.count_nonzero(support_full), 1))
        )

        crop_box = _padded_box(
            region["bbox"],
            selected.width,
            selected.height,
            0.35,
        )
        x0, y0, x1, y1 = crop_box
        crop = selected.crop(crop_box)
        global_depth, local_depth = _infer_depth_pair(
            processor,
            model,
            selected,
            crop,
            device=device,
            dtype=dtype,
        )
        baseline_full, baseline_metadata = _production_baseline(
            selected,
            global_depth,
            local_depth,
            region,
        )
        exact = np.load(corpus_root / row["exact_depth"]["path"]).astype(
            np.float32
        )
        target_full = _near_high_target(exact, selection)
        baseline_crop = baseline_full[y0:y1, x0:x1]
        support_native = support_full[y0:y1, x0:x1]
        exact_native = selection[y0:y1, x0:x1]
        if np.count_nonzero(support_native) < 64:
            raise ValueError(
                f"Production detector support is too small for {row['row_id']}"
            )
        _unchanged, fusion_weight_native, fusion_stats = (
            fuse_face_surface_residual(
                baseline_crop,
                local_depth,
                np.zeros_like(local_depth, dtype=np.float32),
                support_native.astype(np.uint8) * 255,
                alignment_depth=baseline_crop,
                max_correction_ratio=DEFAULT_FACE_MAX_CORRECTION_RATIO,
            )
        )
        part_native = []
        for name in FACE_PART_NAMES:
            part = _load_mask(
                corpus_root / row["exact_face_parts"][name]["path"]
            )
            part_native.append(part[y0:y1, x0:x1])

        rgb = _resize(
            np.asarray(crop),
            network_shape,
            cv2.INTER_AREA,
        ).astype(np.uint8)
        local_network = _resize(
            local_depth,
            network_shape,
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        baseline_network = _resize(
            baseline_crop,
            network_shape,
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        target_network = _resize(
            target_full[y0:y1, x0:x1],
            network_shape,
            cv2.INTER_CUBIC,
        ).astype(np.float32)
        exact_network = _resize(
            exact_native.astype(np.uint8),
            network_shape,
            cv2.INTER_NEAREST,
        ).astype(bool)
        support_network = _resize(
            support_native.astype(np.uint8),
            network_shape,
            cv2.INTER_NEAREST,
        ).astype(bool)
        fusion_weight = _resize(
            fusion_weight_native,
            network_shape,
            cv2.INTER_LINEAR,
        ).astype(np.float32)
        parts = np.stack(
            [
                _resize(
                    part.astype(np.uint8),
                    network_shape,
                    cv2.INTER_NEAREST,
                ).astype(bool)
                for part in part_native
            ]
        )
        row_path = row_root / f"{row['row_id']}.npz"
        np.savez_compressed(
            row_path,
            rgb=rgb,
            local_depth=local_network.astype(np.float16),
            baseline=baseline_network.astype(np.float16),
            target=target_network.astype(np.float16),
            exact_face=exact_network.astype(np.uint8),
            support_face=support_network.astype(np.uint8),
            fusion_weight=fusion_weight.astype(np.float16),
            parts=parts.astype(np.uint8),
            alignment_scale=np.asarray(
                fusion_stats["scale"],
                dtype=np.float32,
            ),
            correction_limit=np.asarray(
                fusion_stats["correction_limit"],
                dtype=np.float32,
            ),
            bbox=np.asarray(crop_box, dtype=np.int32),
            baseline_full=baseline_full.astype(np.float16),
            local_native=local_depth.astype(np.float16),
            support_native=support_native.astype(np.uint8),
        )
        prepared_rows.append(row["row_id"])
        print(
            json.dumps(
                {
                    "cached": index,
                    "total": len(rows),
                    "row_id": row["row_id"],
                    "detector_scope": detection["scope"],
                    "exact_support_coverage": detector_coverages[-1],
                    "support_precision": detector_precisions[-1],
                    "baseline_refined_faces": baseline_metadata[
                        "refined_faces"
                    ],
                }
            ),
            flush=True,
        )

    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": METHOD,
        "corpus_summary_sha256": summary_sha,
        "corpus_asset_manifest_sha256": summary.get("asset_manifest_sha256"),
        "privacy": summary.get("privacy"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": _sha256(snapshot / "model.safetensors"),
        "processor_sha256": _sha256(snapshot / "preprocessor_config.json"),
        "network_size": int(network_size),
        "row_count": len(rows),
        "row_ids": prepared_rows,
        "selected_image_contract": {
            "background_rgb": list(NEUTRAL_BACKGROUND),
            "mask_gaussian_blur_radius": SELECTION_FEATHER_RADIUS,
        },
        "detector": {
            "scope_counts": detector_scopes,
            "median_exact_support_coverage": float(
                np.median(detector_coverages)
            ),
            "minimum_exact_support_coverage": float(
                np.min(detector_coverages)
            ),
            "median_support_precision": float(
                np.median(detector_precisions)
            ),
            "minimum_support_precision": float(
                np.min(detector_precisions)
            ),
        },
        "fusion_contract": {
            "method": "bounded-non-affine-face-surface-residual",
            "maximum_correction_ratio": DEFAULT_FACE_MAX_CORRECTION_RATIO,
            "boundary_pixels_held_zero": 2.0,
        },
        "implementation_sha256": {
            "face_depth_refinement": _sha256(
                Path(__file__).parents[1] / "face_depth_refinement.py"
            ),
            "training_script": _sha256(Path(__file__)),
        },
        "peak_vram_gb": peak_vram,
        "runtime_seconds": time.perf_counter() - started,
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def load_cached_surfaces(
    corpus_root: str | Path,
    cache_root: str | Path,
    *,
    surface_support_mode: str = "detector-face",
) -> tuple[list[CachedFusionSurface], dict, dict]:
    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    surface_support_mode = str(surface_support_mode)
    if surface_support_mode not in SURFACE_SUPPORT_MODES:
        raise ValueError(
            "Unsupported face surface support mode "
            f"{surface_support_mode!r}"
        )
    summary = json.loads((corpus_root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("corpus_summary_sha256") != _sha256(
        corpus_root / "summary.json"
    ):
        raise ValueError("Production-native cache does not match its corpus")
    items = []
    for row in summary["rows"]:
        with np.load(cache_root / "rows" / f"{row['row_id']}.npz") as cached:
            bbox = tuple(int(value) for value in cached["bbox"])
            x0, y0, x1, y1 = bbox
            local_native = cached["local_native"].astype(np.float32)
            baseline_full = cached["baseline_full"].astype(np.float32)
            detector_support_native = cached["support_native"].astype(bool)
            if surface_support_mode == "selection-subject":
                selection = _load_mask(
                    corpus_root / row["selection_mask"]["path"]
                )
                support_native, support_stats = face_surface_support_mask(
                    detector_support_native.astype(np.uint8) * 255,
                    selection[y0:y1, x0:x1].astype(np.uint8) * 255,
                )
                support_native = support_native > 0
                baseline_crop = baseline_full[y0:y1, x0:x1]
                _unchanged, fusion_weight_native, fusion_stats = (
                    fuse_face_surface_residual(
                        baseline_crop,
                        local_native,
                        np.zeros_like(local_native, dtype=np.float32),
                        support_native.astype(np.uint8) * 255,
                        alignment_depth=baseline_crop,
                        max_correction_ratio=(
                            DEFAULT_FACE_MAX_CORRECTION_RATIO
                        ),
                    )
                )
                network_shape = cached["support_face"].shape
                support_face = _resize(
                    support_native.astype(np.uint8),
                    network_shape,
                    cv2.INTER_NEAREST,
                ).astype(bool)
                fusion_weight = _resize(
                    fusion_weight_native,
                    network_shape,
                    cv2.INTER_LINEAR,
                ).astype(np.float32)
                alignment_scale = float(fusion_stats["scale"])
                correction_limit = float(fusion_stats["correction_limit"])
            else:
                support_native = detector_support_native
                support_face = cached["support_face"].astype(bool)
                fusion_weight = cached["fusion_weight"].astype(np.float32)
                alignment_scale = float(cached["alignment_scale"])
                correction_limit = float(cached["correction_limit"])
                support_pixels = int(np.count_nonzero(support_native))
                support_stats = {
                    "mode": "detector-face",
                    "reason": "cached_detector_support",
                    "face_pixels": support_pixels,
                    "support_pixels": support_pixels,
                    "support_expansion_ratio": 1.0,
                }
            items.append(
                CachedFusionSurface(
                    row=row,
                    rgb=cached["rgb"].astype(np.uint8),
                    local_depth=cached["local_depth"].astype(np.float32),
                    baseline=cached["baseline"].astype(np.float32),
                    target=cached["target"].astype(np.float32),
                    exact_face=cached["exact_face"].astype(bool),
                    support_face=support_face,
                    fusion_weight=fusion_weight,
                    parts=cached["parts"].astype(bool),
                    alignment_scale=alignment_scale,
                    correction_limit=correction_limit,
                    bbox=bbox,
                    baseline_full=baseline_full,
                    local_native=local_native,
                    support_native=support_native,
                    surface_support_mode=surface_support_mode,
                    surface_support_stats=support_stats,
                )
            )
    return items, summary, manifest


def small_face_sample_weight(
    face_height_pixels: float,
    *,
    reference_pixels: float = DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
    maximum_weight: float = DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
) -> float:
    reference_pixels = max(float(reference_pixels), 1.0)
    maximum_weight = max(float(maximum_weight), 1.0)
    return float(
        np.clip(
            reference_pixels / max(float(face_height_pixels), 1.0),
            1.0,
            maximum_weight,
        )
    )


def _stack(
    items: list[CachedFusionSurface],
    *,
    small_face_weight_reference_px: float = (
        DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX
    ),
    maximum_small_face_sample_weight: float = (
        DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT
    ),
) -> dict:
    import torch

    size = items[0].baseline.shape[0]
    coordinates = _coordinate_channels(size)
    inputs = []
    for item in items:
        rgb = item.rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        inputs.append(
            np.concatenate(
                (
                    rgb,
                    item.local_depth[None],
                    item.support_face.astype(np.float32)[None],
                    coordinates,
                ),
                axis=0,
            )
        )
    return {
        "inputs": torch.from_numpy(np.stack(inputs)).float(),
        "local_depth": torch.from_numpy(
            np.stack([item.local_depth for item in items])[:, None]
        ).float(),
        "baseline": torch.from_numpy(
            np.stack([item.baseline for item in items])[:, None]
        ).float(),
        "target": torch.from_numpy(
            np.stack([item.target for item in items])[:, None]
        ).float(),
        "exact_face": torch.from_numpy(
            np.stack([item.exact_face for item in items])[:, None]
        ).float(),
        "support_face": torch.from_numpy(
            np.stack([item.support_face for item in items])[:, None]
        ).float(),
        "fusion_weight": torch.from_numpy(
            np.stack([item.fusion_weight for item in items])[:, None]
        ).float(),
        "parts": torch.from_numpy(
            np.stack([np.any(item.parts, axis=0) for item in items])[:, None]
        ).float(),
        "parts_individual": torch.from_numpy(
            np.stack([item.parts for item in items])
        ).float(),
        "alignment_scale": torch.tensor(
            [item.alignment_scale for item in items],
            dtype=torch.float32,
        )[:, None, None, None],
        "correction_limit": torch.tensor(
            [item.correction_limit for item in items],
            dtype=torch.float32,
        )[:, None, None, None],
        "sample_weight": torch.tensor(
            [
                small_face_sample_weight(
                    item.row["render"]["face_bbox_height_pixels"],
                    reference_pixels=small_face_weight_reference_px,
                    maximum_weight=maximum_small_face_sample_weight,
                )
                for item in items
            ],
            dtype=torch.float32,
        )[:, None, None, None],
    }


def remove_affine_component(
    residual,
    local_depth,
    support_face,
):
    weights = support_face.to(dtype=residual.dtype)
    count = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    local_mean = (local_depth * weights).sum(
        dim=(-2, -1),
        keepdim=True,
    ) / count
    residual_mean = (residual * weights).sum(
        dim=(-2, -1),
        keepdim=True,
    ) / count
    local_centered = local_depth - local_mean
    residual_centered = residual - residual_mean
    denominator = (
        local_centered.square() * weights
    ).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    scale = (
        local_centered * residual_centered * weights
    ).sum(dim=(-2, -1), keepdim=True) / denominator
    offset = residual_mean - scale * local_mean
    return residual - (scale * local_depth + offset), scale, offset


def apply_training_residual(residual, tensors: dict):
    non_affine, residual_scale, residual_offset = remove_affine_component(
        residual,
        tensors["local_depth"],
        tensors["support_face"],
    )
    correction = non_affine * tensors["alignment_scale"]
    correction = correction.clamp(
        min=-tensors["correction_limit"],
        max=tensors["correction_limit"],
    )
    correction = correction * tensors["fusion_weight"]
    return tensors["baseline"] + correction, correction, {
        "residual_affine_scale_median": float(
            residual_scale.detach().median()
        ),
        "residual_affine_offset_median": float(
            residual_offset.detach().median()
        ),
    }


def _gradient_non_regression_loss(
    candidate,
    baseline,
    target,
    weights,
):
    import torch.nn.functional as functional

    valid_x = weights[..., :, 1:] * weights[..., :, :-1]
    valid_y = weights[..., 1:, :] * weights[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    candidate_x = candidate[..., :, 1:] - candidate[..., :, :-1]
    candidate_y = candidate[..., 1:, :] - candidate[..., :-1, :]
    baseline_x = baseline[..., :, 1:] - baseline[..., :, :-1]
    baseline_y = baseline[..., 1:, :] - baseline[..., :-1, :]
    excess_x = functional.relu(
        (candidate_x - target_x).abs() - (baseline_x - target_x).abs()
    )
    excess_y = functional.relu(
        (candidate_y - target_y).abs() - (baseline_y - target_y).abs()
    )
    return _weighted_mean(excess_x, valid_x) + _weighted_mean(
        excess_y,
        valid_y,
    )


def _curvature_non_regression_loss(
    candidate,
    baseline,
    target,
    weights,
):
    import torch.nn.functional as functional

    def laplacian(values):
        return (
            4.0 * values[..., 1:-1, 1:-1]
            - values[..., :-2, 1:-1]
            - values[..., 2:, 1:-1]
            - values[..., 1:-1, :-2]
            - values[..., 1:-1, 2:]
        )

    valid = (
        weights[..., 1:-1, 1:-1]
        * weights[..., :-2, 1:-1]
        * weights[..., 2:, 1:-1]
        * weights[..., 1:-1, :-2]
        * weights[..., 1:-1, 2:]
    )
    target_curvature = laplacian(target)
    candidate_error = (laplacian(candidate) - target_curvature).abs()
    baseline_error = (laplacian(baseline) - target_curvature).abs()
    return _weighted_mean(
        functional.relu(candidate_error - baseline_error),
        valid,
    )


def physical_amplitude_losses(
    prediction,
    baseline,
    target,
    fit_mask,
    weights,
):
    fitted_baseline, baseline_scale, baseline_shift = positive_affine_fit(
        baseline,
        target,
        fit_mask,
    )
    fixed_prediction = prediction * baseline_scale + baseline_shift
    return (
        _weighted_mean((fixed_prediction - target).abs(), weights),
        _axis_gradient_loss(fixed_prediction, target, weights),
        baseline_scale,
        baseline_shift,
    )


def _part_balanced_mean(values, part_masks, sample_weight):
    masks = part_masks.to(dtype=values.dtype)
    counts = masks.sum(dim=(-2, -1))
    active = counts > 0
    per_part = (values * masks).sum(dim=(-2, -1)) / counts.clamp_min(1.0)
    row_weights = sample_weight[:, 0, 0, 0][:, None].to(values.dtype)
    weights = active.to(values.dtype) * row_weights
    return (per_part * weights).sum() / weights.sum().clamp_min(1.0)


def part_balanced_physical_losses(
    prediction,
    baseline,
    target,
    part_masks,
    sample_weight,
):
    import torch.nn.functional as functional

    prediction_error = (prediction - target).abs()
    baseline_error = (baseline - target).abs()
    value_loss = _part_balanced_mean(
        prediction_error,
        part_masks,
        sample_weight,
    )
    value_non_regression = _part_balanced_mean(
        functional.relu(prediction_error - baseline_error),
        part_masks,
        sample_weight,
    )

    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    baseline_x = baseline[..., :, 1:] - baseline[..., :, :-1]
    baseline_y = baseline[..., 1:, :] - baseline[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    valid_x = part_masks[..., :, 1:] * part_masks[..., :, :-1]
    valid_y = part_masks[..., 1:, :] * part_masks[..., :-1, :]
    prediction_error_x = (prediction_x - target_x).abs()
    prediction_error_y = (prediction_y - target_y).abs()
    baseline_error_x = (baseline_x - target_x).abs()
    baseline_error_y = (baseline_y - target_y).abs()
    gradient_loss = _part_balanced_mean(
        prediction_error_x,
        valid_x,
        sample_weight,
    ) + _part_balanced_mean(
        prediction_error_y,
        valid_y,
        sample_weight,
    )
    gradient_non_regression = _part_balanced_mean(
        functional.relu(prediction_error_x - baseline_error_x),
        valid_x,
        sample_weight,
    ) + _part_balanced_mean(
        functional.relu(prediction_error_y - baseline_error_y),
        valid_y,
        sample_weight,
    )
    return (
        value_loss,
        gradient_loss,
        value_non_regression,
        gradient_non_regression,
    )


def surface_fusion_training_loss(
    residual,
    tensors: dict,
    *,
    physical_amplitude_value_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT
    ),
    physical_amplitude_gradient_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT
    ),
    part_balanced_physical_value_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT
    ),
    part_balanced_physical_gradient_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT
    ),
    part_balanced_non_regression_weight: float = (
        DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT
    ),
):
    import torch.nn.functional as functional

    prediction, correction, fusion_stats = apply_training_residual(
        residual,
        tensors,
    )
    fitted, scale, shift = positive_affine_fit(
        prediction,
        tensors["target"],
        tensors["exact_face"],
    )
    weights = (
        tensors["exact_face"]
        * (1.0 + PART_WEIGHT * tensors["parts"])
        * tensors["sample_weight"]
    )
    (
        physical_value_loss,
        physical_gradient_loss,
        baseline_scale,
        baseline_shift,
    ) = physical_amplitude_losses(
        prediction,
        tensors["baseline"],
        tensors["target"],
        tensors["exact_face"],
        weights,
    )
    fitted_baseline = (
        tensors["baseline"] * baseline_scale + baseline_shift
    )
    physically_aligned_prediction = (
        prediction * baseline_scale + baseline_shift
    )
    part_value_loss = prediction.new_tensor(0.0)
    part_gradient_loss = prediction.new_tensor(0.0)
    part_value_non_regression = prediction.new_tensor(0.0)
    part_gradient_non_regression = prediction.new_tensor(0.0)
    if any(
        float(weight) > 0.0
        for weight in (
            part_balanced_physical_value_weight,
            part_balanced_physical_gradient_weight,
            part_balanced_non_regression_weight,
        )
    ):
        (
            part_value_loss,
            part_gradient_loss,
            part_value_non_regression,
            part_gradient_non_regression,
        ) = part_balanced_physical_losses(
            physically_aligned_prediction,
            fitted_baseline,
            tensors["target"],
            tensors["parts_individual"],
            tensors["sample_weight"],
        )
    value_loss = _weighted_mean(
        (fitted - tensors["target"]).abs(),
        weights,
    )
    gradient_loss = _axis_gradient_loss(
        fitted,
        tensors["target"],
        weights,
    )
    multiscale_loss = fitted.new_tensor(0.0)
    for factor in (2, 4):
        pooled_prediction = functional.avg_pool2d(fitted, factor)
        pooled_target = functional.avg_pool2d(tensors["target"], factor)
        pooled_weights = functional.avg_pool2d(weights, factor)
        multiscale_loss = multiscale_loss + _weighted_mean(
            (pooled_prediction - pooled_target).abs(),
            pooled_weights,
        )
    valid_normal = (
        tensors["exact_face"][..., 1:, 1:]
        * tensors["exact_face"][..., 1:, :-1]
        * tensors["exact_face"][..., :-1, 1:]
        * tensors["exact_face"][..., :-1, :-1]
        * tensors["sample_weight"]
    )
    normal_cosine = (
        _surface_normals(fitted)
        * _surface_normals(tensors["target"])
    ).sum(dim=1, keepdim=True)
    normal_loss = _weighted_mean(1.0 - normal_cosine, valid_normal)
    residual_regularization = _weighted_mean(
        correction.abs(),
        tensors["support_face"] * tensors["sample_weight"],
    )
    value_non_regression = _weighted_mean(
        functional.relu(
            (fitted - tensors["target"]).abs()
            - (fitted_baseline - tensors["target"]).abs()
        ),
        weights,
    )
    gradient_non_regression = _gradient_non_regression_loss(
        fitted,
        fitted_baseline,
        tensors["target"],
        weights,
    )
    curvature_non_regression = _curvature_non_regression_loss(
        fitted,
        fitted_baseline,
        tensors["target"],
        weights,
    )
    total = (
        VALUE_LOSS_WEIGHT * value_loss
        + GRADIENT_LOSS_WEIGHT * gradient_loss
        + MULTISCALE_LOSS_WEIGHT * multiscale_loss
        + NORMAL_LOSS_WEIGHT * normal_loss
        + RESIDUAL_REGULARIZATION_WEIGHT * residual_regularization
        + VALUE_NON_REGRESSION_WEIGHT * value_non_regression
        + GRADIENT_NON_REGRESSION_WEIGHT * gradient_non_regression
        + CURVATURE_NON_REGRESSION_WEIGHT * curvature_non_regression
        + float(physical_amplitude_value_weight) * physical_value_loss
        + float(physical_amplitude_gradient_weight)
        * physical_gradient_loss
        + float(part_balanced_physical_value_weight) * part_value_loss
        + float(part_balanced_physical_gradient_weight)
        * part_gradient_loss
        + float(part_balanced_non_regression_weight)
        * (
            part_value_non_regression
            + part_gradient_non_regression
        )
    )
    return total, {
        "value": float(value_loss.detach()),
        "gradient": float(gradient_loss.detach()),
        "multiscale": float(multiscale_loss.detach()),
        "normal": float(normal_loss.detach()),
        "residual_regularization": float(residual_regularization.detach()),
        "value_non_regression": float(value_non_regression.detach()),
        "gradient_non_regression": float(gradient_non_regression.detach()),
        "curvature_non_regression": float(
            curvature_non_regression.detach()
        ),
        "physical_amplitude_value": float(
            physical_value_loss.detach()
        ),
        "physical_amplitude_gradient": float(
            physical_gradient_loss.detach()
        ),
        "part_balanced_physical_value": float(
            part_value_loss.detach()
        ),
        "part_balanced_physical_gradient": float(
            part_gradient_loss.detach()
        ),
        "part_balanced_value_non_regression": float(
            part_value_non_regression.detach()
        ),
        "part_balanced_gradient_non_regression": float(
            part_gradient_non_regression.detach()
        ),
        "baseline_fit_scale_median": float(
            baseline_scale.detach().median()
        ),
        "baseline_fit_shift_median": float(
            baseline_shift.detach().median()
        ),
        "fit_scale_median": float(scale.detach().median()),
        "fit_shift_median": float(shift.detach().median()),
        **fusion_stats,
    }


def _loss_for_tensors(
    model,
    tensors: dict,
    device: str,
    *,
    physical_amplitude_value_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT
    ),
    physical_amplitude_gradient_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT
    ),
    part_balanced_physical_value_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT
    ),
    part_balanced_physical_gradient_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT
    ),
    part_balanced_non_regression_weight: float = (
        DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT
    ),
) -> tuple[float, dict]:
    import torch

    model.eval()
    with torch.inference_mode():
        values = {key: tensor.to(device) for key, tensor in tensors.items()}
        loss, details = surface_fusion_training_loss(
            model(values["inputs"]),
            values,
            physical_amplitude_value_weight=(
                physical_amplitude_value_weight
            ),
            physical_amplitude_gradient_weight=(
                physical_amplitude_gradient_weight
            ),
            part_balanced_physical_value_weight=(
                part_balanced_physical_value_weight
            ),
            part_balanced_physical_gradient_weight=(
                part_balanced_physical_gradient_weight
            ),
            part_balanced_non_regression_weight=(
                part_balanced_non_regression_weight
            ),
        )
    return float(loss), details


def train_adapter(
    train_items: list[CachedFusionSurface],
    validation_items: list[CachedFusionSurface],
    *,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = TRAINING_SEED,
    small_face_weight_reference_px: float = (
        DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX
    ),
    maximum_small_face_sample_weight: float = (
        DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT
    ),
    physical_amplitude_value_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT
    ),
    physical_amplitude_gradient_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT
    ),
    part_balanced_physical_value_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT
    ),
    part_balanced_physical_gradient_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT
    ),
    part_balanced_non_regression_weight: float = (
        DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT
    ),
) -> tuple[object, dict]:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_tensors = _stack(
        train_items,
        small_face_weight_reference_px=small_face_weight_reference_px,
        maximum_small_face_sample_weight=maximum_small_face_sample_weight,
    )
    validation_tensors = _stack(
        validation_items,
        small_face_weight_reference_px=small_face_weight_reference_px,
        maximum_small_face_sample_weight=maximum_small_face_sample_weight,
    )
    model = build_surface_adapter(7).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=1e-4,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = math.inf
    history = []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    spatial_keys = (
        "inputs",
        "local_depth",
        "baseline",
        "target",
        "exact_face",
        "support_face",
        "fusion_weight",
        "parts",
        "parts_individual",
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(
            len(train_items),
            generator=generator,
        ).tolist()
        batch_losses = []
        for start in range(0, len(order), max(1, int(batch_size))):
            indices = order[start : start + max(1, int(batch_size))]
            values = {
                key: tensor[indices].to(device)
                for key, tensor in train_tensors.items()
            }
            if ((epoch + start // max(1, int(batch_size))) % 2) == 0:
                for key in spatial_keys:
                    values[key] = torch.flip(values[key], dims=[-1])
            batch_input = values["inputs"].clone()
            color_scale = (
                0.85
                + 0.30
                * torch.rand(
                    (len(indices), 3, 1, 1),
                    generator=generator,
                ).to(device)
            )
            color_shift = (
                -0.04
                + 0.08
                * torch.rand(
                    (len(indices), 3, 1, 1),
                    generator=generator,
                ).to(device)
            )
            batch_input[:, :3] = torch.clamp(
                batch_input[:, :3] * color_scale + color_shift,
                0.0,
                1.0,
            )
            loss, _details = surface_fusion_training_loss(
                model(batch_input),
                values,
                physical_amplitude_value_weight=(
                    physical_amplitude_value_weight
                ),
                physical_amplitude_gradient_weight=(
                    physical_amplitude_gradient_weight
                ),
                part_balanced_physical_value_weight=(
                    part_balanced_physical_value_weight
                ),
                part_balanced_physical_gradient_weight=(
                    part_balanced_physical_gradient_weight
                ),
                part_balanced_non_regression_weight=(
                    part_balanced_non_regression_weight
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        validation_loss, validation_details = _loss_for_tensors(
            model,
            validation_tensors,
            device,
            physical_amplitude_value_weight=(
                physical_amplitude_value_weight
            ),
            physical_amplitude_gradient_weight=(
                physical_amplitude_gradient_weight
            ),
            part_balanced_physical_value_weight=(
                part_balanced_physical_value_weight
            ),
            part_balanced_physical_gradient_weight=(
                part_balanced_physical_gradient_weight
            ),
            part_balanced_non_regression_weight=(
                part_balanced_non_regression_weight
            ),
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(batch_losses)),
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
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    return model.eval(), {
        "seed": seed,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "small_face_weight_reference_px": float(
            small_face_weight_reference_px
        ),
        "maximum_small_face_sample_weight": float(
            maximum_small_face_sample_weight
        ),
        "physical_amplitude_value_weight": float(
            physical_amplitude_value_weight
        ),
        "physical_amplitude_gradient_weight": float(
            physical_amplitude_gradient_weight
        ),
        "part_balanced_physical_value_weight": float(
            part_balanced_physical_value_weight
        ),
        "part_balanced_physical_gradient_weight": float(
            part_balanced_physical_gradient_weight
        ),
        "part_balanced_non_regression_weight": float(
            part_balanced_non_regression_weight
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation),
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "peak_vram_gb": peak_vram,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }


def predict_residuals(
    model,
    items: list[CachedFusionSurface],
    *,
    device: str,
    batch_size: int,
) -> list[np.ndarray]:
    import torch

    inputs = _stack(items)["inputs"]
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            predictions.extend(
                model(
                    inputs[
                        start : start + max(1, int(batch_size))
                    ].to(device)
                )[:, 0]
                .float()
                .cpu()
                .numpy()
            )
    return predictions


def _candidate_full_surface(
    item: CachedFusionSurface,
    residual: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    x0, y0, x1, y1 = item.bbox
    native_residual = _resize(
        residual,
        item.local_native.shape,
        cv2.INTER_CUBIC,
    ).astype(np.float32)
    baseline_crop = item.baseline_full[y0:y1, x0:x1]
    refined_crop, _weight, stats = fuse_face_surface_residual(
        baseline_crop,
        item.local_native,
        float(alpha) * native_residual,
        item.support_native.astype(np.uint8) * 255,
        alignment_depth=baseline_crop,
        max_correction_ratio=DEFAULT_FACE_MAX_CORRECTION_RATIO,
    )
    candidate = item.baseline_full.copy()
    candidate[y0:y1, x0:x1] = refined_crop
    return candidate, stats


def _summary(rows: list[dict]) -> dict:
    from collections import Counter

    check_failures = Counter()
    for row in rows:
        check_failures.update(row.get("shape_check_failures", {}))
    return {
        "row_count": len(rows),
        "combined_part_failures": int(
            sum(row["combined_part_failures"] for row in rows)
        ),
        "median_shape_correlation": float(
            np.median([row["shape_correlation"] for row in rows])
        ),
        "median_gradient_correlation": float(
            np.median([row["gradient_correlation"] for row in rows])
        ),
        "median_normalized_rmse": float(
            np.median([row["normalized_rmse"] for row in rows])
        ),
        "shape_check_failure_counts": dict(sorted(check_failures.items())),
        "rows": rows,
    }


def evaluate_exact_surfaces(
    corpus_root: str | Path,
    items: list[CachedFusionSurface],
    residuals: list[np.ndarray],
    *,
    alpha: float,
    output_dir: str | Path,
) -> dict:
    from collections import Counter

    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item, residual in zip(items, residuals, strict=True):
        candidate, fusion_stats = _candidate_full_surface(
            item,
            residual,
            alpha,
        )
        candidate_path = output_dir / f"{item.row['row_id']}.npy"
        np.save(candidate_path, candidate)
        part_paths = {
            name: corpus_root / item.row["exact_face_parts"][name]["path"]
            for name in FACE_PART_NAMES
        }
        metrics = _exact_face_depth_quality(
            candidate_path,
            corpus_root / item.row["exact_depth"]["path"],
            corpus_root / item.row["selection_mask"]["path"],
            expected_scale_sign=-1.0,
            part_mask_paths=part_paths,
        )
        shape_failed = metrics["named_part_shape"]["failed_parts"]
        affine_failed = metrics["named_part_affine_mm"]["failed_parts"]
        check_failures = Counter()
        for part in metrics["named_part_shape"]["parts"]:
            check_failures.update(
                key
                for key, passed in part.get("checks", {}).items()
                if key != "passed" and not passed
            )
        rows.append(
            {
                "row_id": item.row["row_id"],
                "identity_group": item.row["identity_group"],
                "expression": item.row["expression"],
                "face_height_pixels": item.row["render"][
                    "face_bbox_height_pixels"
                ],
                "shape_correlation": metrics["shape_correlation"],
                "gradient_correlation": metrics["gradient_correlation"],
                "normalized_rmse": metrics["normalized_rmse"],
                "shape_failed_parts": shape_failed,
                "affine_failed_parts": affine_failed,
                "named_part_shape": metrics.get("named_part_shape"),
                "named_part_affine_mm": metrics.get("named_part_affine_mm"),
                "combined_part_failures": len(shape_failed)
                + len(affine_failed),
                "shape_check_failures": dict(sorted(check_failures.items())),
                "fusion": {
                    "alignment_anchor": fusion_stats["alignment_anchor"],
                    "boundary_max_abs_correction": fusion_stats[
                        "boundary_max_abs_correction"
                    ],
                    "max_abs_correction": fusion_stats["max_abs_correction"],
                    "surface_support_mode": item.surface_support_mode,
                    "support_expansion_ratio": item.surface_support_stats[
                        "support_expansion_ratio"
                    ],
                },
            }
        )
    return _summary(rows)


def train(
    corpus_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    network_size: int = DEFAULT_NETWORK_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    force_cache: bool = False,
    small_face_weight_reference_px: float = (
        DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX
    ),
    maximum_small_face_sample_weight: float = (
        DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT
    ),
    surface_support_mode: str = "detector-face",
    physical_amplitude_value_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT
    ),
    physical_amplitude_gradient_weight: float = (
        DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT
    ),
    part_balanced_physical_value_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT
    ),
    part_balanced_physical_gradient_weight: float = (
        DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT
    ),
    part_balanced_non_regression_weight: float = (
        DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT
    ),
) -> dict:
    import torch

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = prepare_cache(
        corpus_root,
        cache_root,
        device=device,
        network_size=network_size,
        force=force_cache,
    )
    items, corpus_summary, _manifest = load_cached_surfaces(
        corpus_root,
        cache_root,
        surface_support_mode=surface_support_mode,
    )
    train_items = [item for item in items if item.row["split"] == "train"]
    validation_items = [
        item for item in items if item.row["split"] == "validation"
    ]
    sealed_items = [item for item in items if item.row["split"] == "sealed"]
    model, training = train_adapter(
        train_items,
        validation_items,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        small_face_weight_reference_px=small_face_weight_reference_px,
        maximum_small_face_sample_weight=maximum_small_face_sample_weight,
        physical_amplitude_value_weight=physical_amplitude_value_weight,
        physical_amplitude_gradient_weight=(
            physical_amplitude_gradient_weight
        ),
        part_balanced_physical_value_weight=(
            part_balanced_physical_value_weight
        ),
        part_balanced_physical_gradient_weight=(
            part_balanced_physical_gradient_weight
        ),
        part_balanced_non_regression_weight=(
            part_balanced_non_regression_weight
        ),
    )
    validation_residuals = predict_residuals(
        model,
        validation_items,
        device=device,
        batch_size=batch_size,
    )
    zero_validation = [
        np.zeros_like(residual) for residual in validation_residuals
    ]
    baseline_validation = evaluate_exact_surfaces(
        corpus_root,
        validation_items,
        zero_validation,
        alpha=0.0,
        output_dir=output_dir / "validation_baseline",
    )
    blend_candidates = []
    for alpha in BLEND_ALPHAS:
        candidate = evaluate_exact_surfaces(
            corpus_root,
            validation_items,
            validation_residuals,
            alpha=alpha,
            output_dir=output_dir / f"validation_alpha_{alpha:.2f}",
        )
        candidate["alpha"] = float(alpha)
        candidate["per_row_non_regression_ratio"] = _per_row_non_regression(
            candidate,
            baseline_validation,
        )
        candidate["eligible"] = bool(
            alpha > 0
            and _strictly_improves(candidate, baseline_validation)
            and candidate["per_row_non_regression_ratio"] >= 1.0
            and all(
                row["fusion"]["boundary_max_abs_correction"] <= 1e-7
                for row in candidate["rows"]
            )
        )
        blend_candidates.append(candidate)
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "failures": candidate["combined_part_failures"],
                    "shape": candidate["median_shape_correlation"],
                    "gradient": candidate["median_gradient_correlation"],
                    "rmse": candidate["median_normalized_rmse"],
                    "eligible": candidate["eligible"],
                }
            ),
            flush=True,
        )
    eligible = [
        candidate for candidate in blend_candidates if candidate["eligible"]
    ]
    selected = min(
        eligible if eligible else blend_candidates,
        key=lambda candidate: (
            candidate["combined_part_failures"],
            candidate["median_normalized_rmse"],
            -candidate["median_shape_correlation"],
            -candidate["median_gradient_correlation"],
            candidate["alpha"],
        ),
    )

    sealed_residuals = predict_residuals(
        model,
        sealed_items,
        device=device,
        batch_size=batch_size,
    )
    zero_sealed = [np.zeros_like(residual) for residual in sealed_residuals]
    baseline_sealed = evaluate_exact_surfaces(
        corpus_root,
        sealed_items,
        zero_sealed,
        alpha=0.0,
        output_dir=output_dir / "sealed_baseline",
    )
    sealed = evaluate_exact_surfaces(
        corpus_root,
        sealed_items,
        sealed_residuals,
        alpha=selected["alpha"],
        output_dir=output_dir / "sealed_selected",
    )
    sealed["alpha"] = selected["alpha"]
    sealed["strictly_improves"] = _strictly_improves(
        sealed,
        baseline_sealed,
    )
    sealed["per_row_non_regression_ratio"] = _per_row_non_regression(
        sealed,
        baseline_sealed,
    )
    decision = {
        "validation_blend_eligible": bool(selected["eligible"]),
        "sealed_strictly_improves": bool(sealed["strictly_improves"]),
        "sealed_per_row_non_regression": bool(
            sealed["per_row_non_regression_ratio"] >= 1.0
        ),
        "all_boundaries_exact_zero": bool(
            all(
                row["fusion"]["boundary_max_abs_correction"] <= 1e-7
                for row in sealed["rows"]
            )
        ),
    }
    decision["eligible_for_exact_production_replay"] = bool(all(decision.values()))

    checkpoint_method = (
        SUBJECT_SUPPORTED_METHOD
        if surface_support_mode == "selection-subject"
        else METHOD
    )
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": checkpoint_method,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "network_size": int(network_size),
        "input_channels": 7,
        "max_normalized_residual": MAX_NORMALIZED_RESIDUAL,
        "selected_alpha": float(selected["alpha"]),
        "selected_epoch": int(training["best_epoch"]),
        "residual_postprocess": (
            "remove-affine-component-relative-to-local-depth"
        ),
        "fusion_contract": cache_manifest["fusion_contract"],
        "surface_support_mode": surface_support_mode,
        "corpus_summary_sha256": cache_manifest["corpus_summary_sha256"],
        "state_dict": model.state_dict(),
    }
    checkpoint_path = output_dir / "face_surface_fusion_adapter.pt"
    torch.save(checkpoint, checkpoint_path)
    evidence = {
        "schema_version": 1,
        "method": checkpoint_method,
        "research_basis": {
            "input_domain": "production neutral selected-image face crops",
            "support_domain": (
                "connected selected-subject component"
                if surface_support_mode == "selection-subject"
                else "production detector face masks"
            ),
            "surface_support_mode": surface_support_mode,
            "surface_support_summary": {
                "median_expansion_ratio": float(
                    np.median(
                        [
                            item.surface_support_stats[
                                "support_expansion_ratio"
                            ]
                            for item in items
                        ]
                    )
                ),
                "minimum_expansion_ratio": float(
                    np.min(
                        [
                            item.surface_support_stats[
                                "support_expansion_ratio"
                            ]
                            for item in items
                        ]
                    )
                ),
                "maximum_expansion_ratio": float(
                    np.max(
                        [
                            item.surface_support_stats[
                                "support_expansion_ratio"
                            ]
                            for item in items
                        ]
                    )
                ),
            },
            "optimization_domain": (
                "current baseline refinement plus bounded non-affine "
                "surface-residual fusion"
            ),
            "non_regression_losses": {
                "value_weight": VALUE_NON_REGRESSION_WEIGHT,
                "gradient_weight": GRADIENT_NON_REGRESSION_WEIGHT,
                "curvature_weight": CURVATURE_NON_REGRESSION_WEIGHT,
            },
            "physical_amplitude_losses": {
                "baseline_affine_anchor": True,
                "value_weight": float(
                    physical_amplitude_value_weight
                ),
                "gradient_weight": float(
                    physical_amplitude_gradient_weight
                ),
            },
            "part_balanced_losses": {
                "equal_named_part_weighting": True,
                "named_parts": list(FACE_PART_NAMES),
                "baseline_affine_anchor": True,
                "physical_value_weight": float(
                    part_balanced_physical_value_weight
                ),
                "physical_gradient_weight": float(
                    part_balanced_physical_gradient_weight
                ),
                "non_regression_weight": float(
                    part_balanced_non_regression_weight
                ),
            },
            "minimum_global_gradient_retention": (
                MINIMUM_GLOBAL_GRADIENT_RETENTION
            ),
            "critical_part_checks": list(CRITICAL_PART_CHECKS),
            "small_face_curriculum": {
                "weight_reference_pixels": float(
                    small_face_weight_reference_px
                ),
                "maximum_sample_weight": float(
                    maximum_small_face_sample_weight
                ),
                "weight_formula": (
                    "clip(reference_pixels / face_height_pixels, 1, cap)"
                ),
            },
        },
        "corpus": {
            "summary_sha256": cache_manifest["corpus_summary_sha256"],
            "asset_manifest_sha256": corpus_summary.get(
                "asset_manifest_sha256"
            ),
            "privacy": corpus_summary.get("privacy"),
            "train_rows": len(train_items),
            "validation_rows": len(validation_items),
            "sealed_rows": len(sealed_items),
            "train_identities": sorted(
                {item.row["identity_group"] for item in train_items}
            ),
            "validation_identities": sorted(
                {item.row["identity_group"] for item in validation_items}
            ),
            "sealed_identities": sorted(
                {item.row["identity_group"] for item in sealed_items}
            ),
        },
        "cache": cache_manifest,
        "training": training,
        "validation": {
            "baseline": baseline_validation,
            "blend_candidates": blend_candidates,
            "selected": selected,
        },
        "sealed": {
            "baseline": baseline_sealed,
            "selected": sealed,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
        },
        "device": (
            torch.cuda.get_device_name(device)
            if str(device).startswith("cuda")
            else "cpu"
        ),
        "decision": decision,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--network-size", type=int, default=DEFAULT_NETWORK_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--small-face-weight-reference-px",
        type=float,
        default=DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
    )
    parser.add_argument(
        "--maximum-small-face-sample-weight",
        type=float,
        default=DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
    )
    parser.add_argument(
        "--surface-support-mode",
        choices=SURFACE_SUPPORT_MODES,
        default="detector-face",
    )
    parser.add_argument(
        "--physical-amplitude-value-weight",
        type=float,
        default=DEFAULT_PHYSICAL_AMPLITUDE_VALUE_WEIGHT,
    )
    parser.add_argument(
        "--physical-amplitude-gradient-weight",
        type=float,
        default=DEFAULT_PHYSICAL_AMPLITUDE_GRADIENT_WEIGHT,
    )
    parser.add_argument(
        "--part-balanced-physical-value-weight",
        type=float,
        default=DEFAULT_PART_BALANCED_PHYSICAL_VALUE_WEIGHT,
    )
    parser.add_argument(
        "--part-balanced-physical-gradient-weight",
        type=float,
        default=DEFAULT_PART_BALANCED_PHYSICAL_GRADIENT_WEIGHT,
    )
    parser.add_argument(
        "--part-balanced-non-regression-weight",
        type=float,
        default=DEFAULT_PART_BALANCED_NON_REGRESSION_WEIGHT,
    )
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    evidence = train(
        args.corpus_root,
        args.cache_root,
        args.output_dir,
        device=args.device,
        network_size=args.network_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_cache=args.force_cache,
        small_face_weight_reference_px=args.small_face_weight_reference_px,
        maximum_small_face_sample_weight=(
            args.maximum_small_face_sample_weight
        ),
        surface_support_mode=args.surface_support_mode,
        physical_amplitude_value_weight=(
            args.physical_amplitude_value_weight
        ),
        physical_amplitude_gradient_weight=(
            args.physical_amplitude_gradient_weight
        ),
        part_balanced_physical_value_weight=(
            args.part_balanced_physical_value_weight
        ),
        part_balanced_physical_gradient_weight=(
            args.part_balanced_physical_gradient_weight
        ),
        part_balanced_non_regression_weight=(
            args.part_balanced_non_regression_weight
        ),
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_exact_production_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
