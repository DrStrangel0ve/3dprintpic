from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
FRAME_SELECTION_MODES = {"uniform-frame-sampler", "sharpness-motion-selector"}
VIDEO_SEGMENTATION_PROVIDERS = {
    "turntable-grabcut",
    "sam2.1-hiera-tiny-video",
    "sam2.1-hiera-base-plus-video",
}
SAM2_VIDEO_MODELS = {
    "sam2.1-hiera-tiny-video": "facebook/sam2.1-hiera-tiny",
    "sam2.1-hiera-base-plus-video": "facebook/sam2.1-hiera-base-plus",
}
SAM2_VIDEO_REVISIONS = {
    "sam2.1-hiera-tiny-video": "de431c4043854a71d8101e17995dfe596bf101a5",
    "sam2.1-hiera-base-plus-video": "b7320756a13354e7530a63935656d35b2f91a290",
}


class VideoPipelineError(ValueError):
    pass


class VideoDecodeError(VideoPipelineError):
    pass


class VideoSegmentationError(VideoPipelineError):
    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report or {}


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    declared_frame_count: int
    declared_duration_seconds: float
    codec: str


@dataclass
class FrameCandidate:
    source_index: int
    bgr: np.ndarray
    sharpness: float
    exposure_score: float
    fingerprint: np.ndarray
    selection_score: float = 0.0


@dataclass(frozen=True)
class PreparedVideo:
    bundle_path: Path
    report_path: Path
    frame_paths: tuple[Path, ...]
    mask_paths: tuple[Path, ...]
    bundle: dict
    report: dict


def _cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError("OpenCV video support is unavailable; install opencv-python-headless") from exc
    return cv2


def opencv_video_preflight() -> dict:
    try:
        cv2 = _cv2()
        return {"available": True, "version": str(cv2.__version__), "error": None}
    except Exception as exc:
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _transformers_sam2_video_preflight() -> tuple[bool, str | None, str | None]:
    try:
        import transformers
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        del Sam2VideoModel, Sam2VideoProcessor
        return True, str(transformers.__version__), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def video_model_preflight() -> dict:
    opencv = opencv_video_preflight()
    sam2_available, transformers_version, sam2_error = _transformers_sam2_video_preflight()
    sam3_dir = Path(os.environ["SAM3_DIR"]).expanduser() if os.environ.get("SAM3_DIR") else None
    sam3_source = bool(sam3_dir and (sam3_dir / "sam3").is_dir()) or _module_available("sam3")
    sam3_checkpoint = bool(os.environ.get("SAM3_CHECKPOINT"))
    vggt_omega_dir = Path(os.environ["VGGT_OMEGA_DIR"]).expanduser() if os.environ.get("VGGT_OMEGA_DIR") else None
    vggt_omega_source = bool(vggt_omega_dir and (vggt_omega_dir / "vggt_omega").is_dir()) or _module_available(
        "vggt_omega"
    )
    vggt_omega_checkpoint = bool(os.environ.get("VGGT_OMEGA_CHECKPOINT"))
    return {
        "frame_decoding": {
            "id": "opencv-videoio",
            "runnable": bool(opencv["available"]),
            "status": "available" if opencv["available"] else "missing",
            "checks": opencv,
        },
        "segmentation": [
            {
                "id": "turntable-grabcut",
                "runnable": bool(opencv["available"]),
                "status": "available" if opencv["available"] else "missing",
                "checks": {"opencv_available": bool(opencv["available"])},
                "setup_errors": [] if opencv["available"] else [str(opencv["error"])],
            },
            {
                "id": "sam2.1-hiera-tiny-video",
                "runnable": sam2_available,
                "status": "available" if sam2_available else "missing",
                "checks": {
                    "transformers_sam2_video_available": sam2_available,
                    "transformers_version": transformers_version,
                    "model_revision": SAM2_VIDEO_REVISIONS["sam2.1-hiera-tiny-video"],
                },
                "setup_errors": [] if sam2_available else [sam2_error or "SAM2 video classes are unavailable"],
            },
            {
                "id": "sam2.1-hiera-base-plus-video",
                "runnable": sam2_available,
                "status": "available" if sam2_available else "missing",
                "checks": {
                    "transformers_sam2_video_available": sam2_available,
                    "transformers_version": transformers_version,
                    "model_revision": SAM2_VIDEO_REVISIONS["sam2.1-hiera-base-plus-video"],
                },
                "setup_errors": [] if sam2_available else [sam2_error or "SAM2 video classes are unavailable"],
            },
            {
                "id": "sam3.1-video",
                "runnable": sam3_source and sam3_checkpoint,
                "status": "available" if sam3_source and sam3_checkpoint else "setup-required",
                "checks": {
                    "sam3_source_available": sam3_source,
                    "checkpoint_configured": sam3_checkpoint,
                },
                "setup_errors": [
                    message
                    for ok, message in (
                        (sam3_source, "SAM3_DIR or an installed sam3 package is required"),
                        (sam3_checkpoint, "SAM3_CHECKPOINT is required after Hugging Face access is approved"),
                    )
                    if not ok
                ],
            },
        ],
        "geometry": [
            {
                "id": "multiview-visual-hull",
                "runnable": bool(opencv["available"]),
                "status": "available" if opencv["available"] else "missing",
                "checks": {"builtin_provider": True},
                "setup_errors": [] if opencv["available"] else [str(opencv["error"])],
            },
            {
                "id": "vggt-omega",
                "runnable": vggt_omega_source and vggt_omega_checkpoint,
                "status": "available" if vggt_omega_source and vggt_omega_checkpoint else "setup-required",
                "checks": {
                    "vggt_omega_source_available": vggt_omega_source,
                    "checkpoint_configured": vggt_omega_checkpoint,
                    "stl_extractor_attached": False,
                },
                "setup_errors": [
                    message
                    for ok, message in (
                        (vggt_omega_source, "VGGT_OMEGA_DIR or an installed vggt_omega package is required"),
                        (vggt_omega_checkpoint, "VGGT_OMEGA_CHECKPOINT is required after model access is approved"),
                        (False, "VGGT-Omega currently emits cameras/depth/points; STL mesh extraction is not attached"),
                    )
                    if not ok
                ],
            },
        ],
    }


def probe_video(path: Path, *, max_duration_seconds: float = 180.0, max_pixels: int = 4096 * 4096) -> VideoProbe:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoDecodeError("The uploaded video could not be opened by OpenCV/FFmpeg")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()

    if width < 2 or height < 2:
        raise VideoDecodeError(f"Video dimensions are invalid: {width}x{height}")
    if width * height > max_pixels:
        raise VideoDecodeError(f"Video dimensions {width}x{height} exceed the configured pixel limit")
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    duration = declared_frame_count / fps if declared_frame_count > 0 else 0.0
    if max_duration_seconds > 0 and duration > max_duration_seconds:
        raise VideoDecodeError(
            f"Video is {duration:.1f}s; the current safety limit is {float(max_duration_seconds):.1f}s"
        )
    codec = "".join(chr((codec_value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")
    return VideoProbe(
        width=width,
        height=height,
        fps=fps,
        declared_frame_count=declared_frame_count,
        declared_duration_seconds=duration,
        codec=codec,
    )


def _resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame
    scale = float(max_side) / float(longest)
    target = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
    return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)


def _candidate_metrics(frame: np.ndarray) -> tuple[float, float, np.ndarray]:
    cv2 = _cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    mean_luma = float(np.mean(gray)) / 255.0
    exposure_score = max(0.0, 1.0 - clipped * 3.0) * max(0.0, 1.0 - abs(mean_luma - 0.5) * 1.25)
    fingerprint = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return sharpness, exposure_score, fingerprint


def decode_video_candidates(
    path: Path,
    probe: VideoProbe,
    *,
    selected_frame_count: int,
    max_decode_frames: int = 12000,
    frame_max_side: int = 960,
    candidate_multiplier: int = 8,
) -> tuple[list[FrameCandidate], int, list[str]]:
    cv2 = _cv2()
    candidate_limit = max(selected_frame_count, selected_frame_count * max(2, candidate_multiplier))
    if probe.declared_frame_count > 0:
        stride = max(1, int(math.floor(probe.declared_frame_count / candidate_limit)))
    else:
        stride = 1

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError("The video opened during probing but failed when frame decoding started")
    candidates: list[FrameCandidate] = []
    decoded_frames = 0
    warnings: list[str] = []
    try:
        while decoded_frames < max_decode_frames:
            ok, frame = capture.read()
            if not ok:
                break
            source_index = decoded_frames
            decoded_frames += 1
            if source_index % stride != 0:
                continue
            frame = _resize_frame(frame, frame_max_side)
            sharpness, exposure_score, fingerprint = _candidate_metrics(frame)
            candidates.append(
                FrameCandidate(
                    source_index=source_index,
                    bgr=frame,
                    sharpness=sharpness,
                    exposure_score=exposure_score,
                    fingerprint=fingerprint,
                )
            )
            if probe.declared_frame_count <= 0 and len(candidates) > candidate_limit * 2:
                candidates = candidates[::2]
                stride *= 2
    finally:
        capture.release()

    if decoded_frames >= max_decode_frames:
        raise VideoDecodeError(f"Video reached the configured decode limit of {max_decode_frames} frames")
    if decoded_frames < 3:
        raise VideoDecodeError(f"Video yielded only {decoded_frames} decodable frame(s); at least 3 are required")
    if not candidates:
        raise VideoDecodeError("Video decoding produced no frame candidates")
    if candidates[-1].source_index < decoded_frames - 1:
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, decoded_frames - 1)
            ok, frame = capture.read()
            if ok:
                frame = _resize_frame(frame, frame_max_side)
                sharpness, exposure_score, fingerprint = _candidate_metrics(frame)
                candidates.append(
                    FrameCandidate(
                        source_index=decoded_frames - 1,
                        bgr=frame,
                        sharpness=sharpness,
                        exposure_score=exposure_score,
                        fingerprint=fingerprint,
                    )
                )
        finally:
            capture.release()

    if probe.declared_frame_count > 0:
        decode_ratio = decoded_frames / float(probe.declared_frame_count)
        if decode_ratio < 0.8:
            raise VideoDecodeError(
                f"Video decoding stopped at {decoded_frames}/{probe.declared_frame_count} declared frames"
            )
        if decode_ratio < 0.98:
            warnings.append(
                f"Decoded {decoded_frames}/{probe.declared_frame_count} declared frames; the container count may be imprecise"
            )
    return candidates, decoded_frames, warnings


def _nearest_unused_candidate(candidates: list[FrameCandidate], target_index: float, used: set[int]) -> FrameCandidate:
    available = [candidate for candidate in candidates if candidate.source_index not in used]
    if not available:
        raise VideoDecodeError("Frame selection exhausted the decoded candidates")
    return min(available, key=lambda candidate: (abs(candidate.source_index - target_index), candidate.source_index))


def _normalized(values: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        return data
    low, high = np.percentile(data, [10, 90])
    if not math.isfinite(float(high - low)) or high <= low:
        return np.ones_like(data)
    return np.clip((data - low) / (high - low), 0.0, 1.0)


def select_video_frames(
    candidates: list[FrameCandidate],
    *,
    decoded_frames: int,
    selected_frame_count: int,
    mode: str,
) -> list[FrameCandidate]:
    if mode not in FRAME_SELECTION_MODES:
        valid = ", ".join(sorted(FRAME_SELECTION_MODES))
        raise VideoPipelineError(f"Unsupported frame selection mode '{mode}'. Valid: {valid}")
    count = min(max(3, int(selected_frame_count)), len(candidates))
    if count < 3:
        raise VideoDecodeError("At least three distinct decoded frame candidates are required")

    desired = np.linspace(0.0, float(decoded_frames), count, endpoint=False)
    if mode == "uniform-frame-sampler":
        selected: list[FrameCandidate] = []
        used: set[int] = set()
        for target in desired:
            candidate = _nearest_unused_candidate(candidates, float(target), used)
            candidate.selection_score = 1.0 - min(1.0, abs(candidate.source_index - target) / max(1.0, decoded_frames))
            selected.append(candidate)
            used.add(candidate.source_index)
        return sorted(selected, key=lambda candidate: candidate.source_index)

    sharpness_scores = _normalized(candidate.sharpness for candidate in candidates)
    exposure_scores = _normalized(candidate.exposure_score for candidate in candidates)
    for index, candidate in enumerate(candidates):
        candidate.selection_score = float(0.75 * sharpness_scores[index] + 0.25 * exposure_scores[index])

    nominal_spacing = float(decoded_frames) / float(count)
    selected = []
    used: set[int] = set()
    previous: FrameCandidate | None = None
    for target_index, target in enumerate(desired):
        lower = max(0.0, float(target) - nominal_spacing * 0.5)
        upper = min(float(decoded_frames), float(target) + nominal_spacing * 0.5)
        bucket = [
            candidate
            for candidate in candidates
            if candidate.source_index not in used
            and candidate.source_index >= lower
            and (candidate.source_index < upper or target_index == count - 1)
            and (previous is None or candidate.source_index - previous.source_index >= nominal_spacing * 0.5)
        ]
        if not bucket:
            candidate = _nearest_unused_candidate(candidates, float(target), used)
        else:
            def score(item: FrameCandidate) -> tuple[float, int]:
                diversity = 0.0 if previous is None else float(np.mean(np.abs(item.fingerprint - previous.fingerprint)))
                center_distance = abs(item.source_index - target) / max(1.0, nominal_spacing)
                return item.selection_score + 0.25 * diversity - 0.30 * center_distance, -item.source_index

            candidate = max(bucket, key=score)
        selected.append(candidate)
        used.add(candidate.source_index)
        previous = candidate
    return sorted(selected, key=lambda candidate: candidate.source_index)


def _component_mask(binary: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    cv2 = _cv2()
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    height, width = binary.shape
    px = int(round(np.clip(point[0], 0.0, 1.0) * (width - 1)))
    py = int(round(np.clip(point[1], 0.0, 1.0) * (height - 1)))
    selected_label = int(labels[py, px])
    if selected_label <= 0:
        diagonal = max(1.0, math.hypot(width, height))
        ranked = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            cx, cy = centroids[label]
            distance = math.hypot(float(cx) - px, float(cy) - py) / diagonal
            ranked.append((area * max(0.05, 1.0 - distance * 2.0), area, -label, label))
        selected_label = max(ranked)[-1]
    return (labels == selected_label).astype(np.uint8)


def _fill_mask_holes(binary: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    padded = cv2.copyMakeBorder(binary.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return np.maximum(padded, holes)[1:-1, 1:-1]


def segment_turntable_frame(
    frame: np.ndarray,
    *,
    point: tuple[float, float],
    previous_mask: np.ndarray | None = None,
) -> np.ndarray:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    if min(height, width) < 16:
        raise VideoSegmentationError(f"Frame is too small for segmentation: {width}x{height}")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(2, min(height, width) // 24)
    border_pixels = np.concatenate(
        [
            lab[:border].reshape(-1, 3),
            lab[-border:].reshape(-1, 3),
            lab[:, :border].reshape(-1, 3),
            lab[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.concatenate(
        [distance[:border].ravel(), distance[-border:].ravel(), distance[:, :border].ravel(), distance[:, -border:].ravel()]
    )
    robust_threshold = max(10.0, float(np.percentile(border_distance, 99.0)) + 6.0)
    scaled_distance = np.clip(distance * (255.0 / max(1.0, float(np.percentile(distance, 99.0)))), 0, 255).astype(
        np.uint8
    )
    otsu_threshold, _ = cv2.threshold(scaled_distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_distance = float(otsu_threshold) * max(1.0, float(np.percentile(distance, 99.0))) / 255.0
    rough = distance > max(robust_threshold, otsu_distance * 0.75)

    gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[rough] = cv2.GC_PR_FGD
    gc_mask[:border] = cv2.GC_BGD
    gc_mask[-border:] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    px = int(round(np.clip(point[0], 0.0, 1.0) * (width - 1)))
    py = int(round(np.clip(point[1], 0.0, 1.0) * (height - 1)))
    seed_radius = max(2, min(height, width) // 80)
    cv2.circle(gc_mask, (px, py), seed_radius * 2, cv2.GC_FGD, thickness=-1)
    if previous_mask is not None:
        previous = previous_mask.astype(np.uint8)
        if previous.shape != (height, width):
            previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((max(3, seed_radius * 2 + 1),) * 2, dtype=np.uint8)
        gc_mask[previous > 0] = cv2.GC_PR_FGD
        confirmed_previous = (cv2.erode(previous, kernel, iterations=1) > 0) & rough
        gc_mask[confirmed_previous] = cv2.GC_FGD
        gc_mask[:border] = cv2.GC_BGD
        gc_mask[-border:] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame, gc_mask, None, background_model, foreground_model, 4, cv2.GC_INIT_WITH_MASK)
        binary = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        binary = rough.astype(np.uint8)

    binary = _component_mask(binary, point)
    kernel_size = max(3, int(round(min(height, width) / 160.0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = _fill_mask_holes(binary)
    return binary.astype(np.uint8)


def segment_frames_sam2_video(
    frames: list[np.ndarray],
    *,
    provider: str,
    point: tuple[float, float],
    device: str = "auto",
) -> list[np.ndarray]:
    if provider not in SAM2_VIDEO_MODELS:
        raise VideoSegmentationError(f"No SAM2 model mapping exists for '{provider}'")
    try:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except Exception as exc:
        raise VideoSegmentationError(f"SAM2 video dependencies are unavailable: {type(exc).__name__}: {exc}") from exc

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        raise VideoSegmentationError("SAM2 video segmentation requested CUDA, but torch.cuda.is_available() is false")

    model_id = SAM2_VIDEO_MODELS[provider]
    model_revision = SAM2_VIDEO_REVISIONS[provider]
    try:
        model = Sam2VideoModel.from_pretrained(model_id, revision=model_revision).to(selected_device).eval()
        processor = Sam2VideoProcessor.from_pretrained(model_id, revision=model_revision)
        pil_frames = [Image.fromarray(frame[:, :, ::-1].copy()) for frame in frames]
        session = processor.init_video_session(video=pil_frames, inference_device=selected_device)
        height, width = frames[0].shape[:2]
        px = float(np.clip(point[0], 0.0, 1.0) * (width - 1))
        py = float(np.clip(point[1], 0.0, 1.0) * (height - 1))
        margin = max(3.0, min(width, height) * 0.02)
        points = [[[[px, py], [margin, margin], [width - margin, margin], [margin, height - margin], [width - margin, height - margin]]]]
        labels = [[[1, 0, 0, 0, 0]]]
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_points=points,
            input_labels=labels,
        )
        with torch.inference_mode():
            model(inference_session=session, frame_idx=0)
            masks_by_index: dict[int, np.ndarray] = {}
            for output in model.propagate_in_video_iterator(session, show_progress_bar=False):
                resized = processor.post_process_masks(
                    [output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=False,
                )[0]
                mask = resized[0, 0].detach().float().cpu().numpy() > 0.0
                masks_by_index[int(output.frame_idx)] = mask.astype(np.uint8)
    except Exception as exc:
        raise VideoSegmentationError(f"SAM2 video inference failed: {type(exc).__name__}: {exc}") from exc
    missing = [index for index in range(len(frames)) if index not in masks_by_index]
    if missing:
        raise VideoSegmentationError(f"SAM2 did not return masks for frame indices: {missing}")
    return [masks_by_index[index] for index in range(len(frames))]


def _mask_metrics(mask: np.ndarray) -> dict:
    cv2 = _cv2()
    binary = mask.astype(bool)
    height, width = binary.shape
    pixels = int(np.count_nonzero(binary))
    coverage = pixels / float(max(1, width * height))
    border = max(1, min(height, width) // 40)
    border_pixels = np.concatenate(
        [binary[:border].ravel(), binary[-border:].ravel(), binary[:, :border].ravel(), binary[:, -border:].ravel()]
    )
    border_fraction = float(np.mean(border_pixels))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    component_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)]
    largest_component_ratio = max(component_areas, default=0) / float(max(1, pixels))
    if pixels:
        ys, xs = np.nonzero(binary)
        centroid_x = float(np.mean(xs) / max(1, width - 1))
        centroid_y = float(np.mean(ys) / max(1, height - 1))
        bbox = {
            "x": int(xs.min()),
            "y": int(ys.min()),
            "width": int(xs.max() - xs.min() + 1),
            "height": int(ys.max() - ys.min() + 1),
        }
    else:
        centroid_x = centroid_y = 0.0
        bbox = {"x": 0, "y": 0, "width": 0, "height": 0}
    return {
        "mask_pixels": pixels,
        "coverage": coverage,
        "border_fraction": border_fraction,
        "component_count": max(0, component_count - 1),
        "largest_component_ratio": largest_component_ratio,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox": bbox,
    }


def evaluate_mask_sequence(masks: list[np.ndarray], source_indices: list[int] | None = None) -> dict:
    if not masks:
        return {
            "status": "fail",
            "passes_hard_checks": False,
            "failed_checks": ["mask_sequence_nonempty"],
            "warnings": [],
            "per_frame": [],
        }
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise VideoSegmentationError(f"Mask dimensions are inconsistent: {sorted(shapes)}")
    source_indices = source_indices or list(range(len(masks)))
    metrics = []
    for position, mask in enumerate(masks):
        row = _mask_metrics(mask)
        row["position"] = position
        row["source_index"] = int(source_indices[position])
        metrics.append(row)

    failed: list[str] = []
    warnings: list[str] = []
    for row in metrics:
        suffix = f"frame_{row['source_index']}"
        if row["coverage"] < 0.008:
            failed.append(f"mask_not_empty:{suffix}")
        if row["coverage"] > 0.85:
            failed.append(f"mask_not_full_frame:{suffix}")
        if row["largest_component_ratio"] < 0.75:
            failed.append(f"mask_single_dominant_component:{suffix}")
        if row["border_fraction"] > 0.35:
            failed.append(f"mask_no_border_leak:{suffix}")

    coverages = np.asarray([row["coverage"] for row in metrics], dtype=np.float64)
    median_coverage = float(np.median(coverages))
    valid_coverage = coverages[coverages > 0]
    area_ratio = float(np.max(valid_coverage) / max(1e-9, np.min(valid_coverage))) if len(valid_coverage) else math.inf
    centroid_jumps = []
    consecutive_ious = []
    for index in range(1, len(masks)):
        previous, current = masks[index - 1].astype(bool), masks[index].astype(bool)
        intersection = int(np.count_nonzero(previous & current))
        union = int(np.count_nonzero(previous | current))
        consecutive_ious.append(intersection / float(max(1, union)))
        centroid_jumps.append(
            math.hypot(
                metrics[index]["centroid_x"] - metrics[index - 1]["centroid_x"],
                metrics[index]["centroid_y"] - metrics[index - 1]["centroid_y"],
            )
        )
    max_centroid_jump = max(centroid_jumps, default=0.0)
    min_consecutive_iou = min(consecutive_ious, default=1.0)
    median_consecutive_iou = float(np.median(consecutive_ious)) if consecutive_ious else 1.0
    if area_ratio > 4.0:
        failed.append("mask_area_stability")
    elif area_ratio > 2.5:
        warnings.append("Mask area varies by more than 2.5x across selected frames")
    if max_centroid_jump > 0.32:
        failed.append("mask_centroid_stability")
    elif max_centroid_jump > 0.18:
        warnings.append("Object-mask centroid moves substantially between selected frames")
    if min_consecutive_iou < 0.02:
        failed.append("mask_temporal_overlap")
    elif median_consecutive_iou < 0.25:
        warnings.append("Median consecutive mask overlap is low; inspect for identity drift")

    failed = list(dict.fromkeys(failed))
    return {
        "status": "fail" if failed else ("warn" if warnings else "pass"),
        "passes_hard_checks": not failed,
        "failed_checks": failed,
        "warnings": warnings,
        "aggregate": {
            "frame_count": len(masks),
            "coverage_median": median_coverage,
            "coverage_min": float(np.min(coverages)),
            "coverage_max": float(np.max(coverages)),
            "coverage_max_min_ratio": area_ratio,
            "max_centroid_jump": max_centroid_jump,
            "minimum_consecutive_iou": min_consecutive_iou,
            "median_consecutive_iou": median_consecutive_iou,
        },
        "per_frame": metrics,
    }


def _save_frames_and_masks(
    selected: list[FrameCandidate],
    masks: list[np.ndarray],
    output_dir: Path,
) -> tuple[list[Path], list[Path]]:
    cv2 = _cv2()
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    mask_paths: list[Path] = []
    for position, (candidate, mask) in enumerate(zip(selected, masks, strict=True)):
        stem = f"view_{position:03d}_src_{candidate.source_index:06d}"
        frame_path = frames_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}_mask.png"
        if not cv2.imwrite(str(frame_path), candidate.bgr):
            raise VideoPipelineError(f"Could not write decoded frame: {frame_path}")
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        frame_paths.append(frame_path)
        mask_paths.append(mask_path)
    return frame_paths, mask_paths


def prepare_turntable_video(
    video_path: Path,
    output_dir: Path,
    *,
    selected_frame_count: int = 12,
    frame_selection: str = "uniform-frame-sampler",
    segmentation_provider: str = "turntable-grabcut",
    object_point: tuple[float, float] = (0.5, 0.5),
    segmentation_device: str = "auto",
    rotation_degrees: float = 360.0,
    rotation_direction: str = "counter-clockwise",
    start_azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
    strict_segmentation: bool = True,
    max_duration_seconds: float = 180.0,
    max_decode_frames: int = 12000,
    frame_max_side: int = 960,
) -> PreparedVideo:
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise VideoDecodeError(f"Unsupported video extension '{video_path.suffix}'. Valid: {supported}")
    if not (0.0 <= float(object_point[0]) <= 1.0 and 0.0 <= float(object_point[1]) <= 1.0):
        raise VideoSegmentationError("Object point coordinates must be normalized values between 0 and 1")
    if segmentation_provider not in VIDEO_SEGMENTATION_PROVIDERS:
        valid = ", ".join(sorted(VIDEO_SEGMENTATION_PROVIDERS))
        raise VideoSegmentationError(f"Unsupported video segmentation provider '{segmentation_provider}'. Valid: {valid}")
    if rotation_direction not in {"clockwise", "counter-clockwise"}:
        raise VideoPipelineError("rotation_direction must be 'clockwise' or 'counter-clockwise'")
    if not math.isfinite(float(rotation_degrees)) or abs(float(rotation_degrees)) < 30.0:
        raise VideoPipelineError("rotation_degrees must describe at least 30 degrees of object coverage")

    output_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_video(video_path, max_duration_seconds=max_duration_seconds)
    candidates, decoded_frames, decode_warnings = decode_video_candidates(
        video_path,
        probe,
        selected_frame_count=selected_frame_count,
        max_decode_frames=max_decode_frames,
        frame_max_side=frame_max_side,
    )
    selected = select_video_frames(
        candidates,
        decoded_frames=decoded_frames,
        selected_frame_count=selected_frame_count,
        mode=frame_selection,
    )
    if len(selected) < min(3, int(selected_frame_count)):
        raise VideoDecodeError(f"Only {len(selected)} distinct frames were selected")

    frames = [candidate.bgr for candidate in selected]
    if segmentation_provider == "turntable-grabcut":
        masks = []
        previous_mask = None
        for frame in frames:
            mask = segment_turntable_frame(frame, point=object_point, previous_mask=previous_mask)
            masks.append(mask)
            previous_mask = mask
    else:
        masks = segment_frames_sam2_video(
            frames,
            provider=segmentation_provider,
            point=object_point,
            device=segmentation_device,
        )

    source_indices = [candidate.source_index for candidate in selected]
    mask_quality = evaluate_mask_sequence(masks, source_indices=source_indices)
    frame_paths, mask_paths = _save_frames_and_masks(selected, masks, output_dir)
    sampling = {
        "mode": frame_selection,
        "requested_frame_count": int(selected_frame_count),
        "selected_frame_count": len(selected),
        "candidate_count": len(candidates),
        "decoded_frame_count": decoded_frames,
        "selected_source_indices": source_indices,
        "selected_timestamps_seconds": [candidate.source_index / probe.fps for candidate in selected],
        "sharpness": [candidate.sharpness for candidate in selected],
        "selection_scores": [candidate.selection_score for candidate in selected],
        "warnings": decode_warnings,
    }
    direction = -1.0 if rotation_direction == "clockwise" else 1.0
    views = []
    for position, (candidate, frame_path, mask_path) in enumerate(zip(selected, frame_paths, mask_paths, strict=True)):
        progress = candidate.source_index / float(max(1, decoded_frames))
        azimuth = float(start_azimuth_deg) + direction * float(rotation_degrees) * progress
        views.append(
            {
                "index": position,
                "sample_id": f"video_view_{position:03d}",
                "source_frame_index": candidate.source_index,
                "source_time_seconds": candidate.source_index / probe.fps,
                "image": str(frame_path.resolve()),
                "mask": str(mask_path.resolve()),
                "camera": {
                    "azimuth_deg": azimuth,
                    "elevation_deg": float(elevation_deg),
                    "roll_deg": 0.0,
                },
            }
        )
    bundle = {
        "sample_id": video_path.stem,
        "source_kind": "turntable-video",
        "video_path": str(video_path.resolve()),
        "primary_image": str(frame_paths[0].resolve()),
        "projection_assumption": "orthographic turntable orbit",
        "rotation_degrees": float(rotation_degrees),
        "rotation_direction": rotation_direction,
        "views": views,
    }
    report = {
        "status": "blocked" if strict_segmentation and not mask_quality["passes_hard_checks"] else "prepared",
        "video": asdict(probe),
        "sampling": sampling,
        "segmentation": {
            "provider": segmentation_provider,
            "model_id": SAM2_VIDEO_MODELS.get(segmentation_provider),
            "model_revision": SAM2_VIDEO_REVISIONS.get(segmentation_provider),
            "device": segmentation_device,
            "object_point": {"x": float(object_point[0]), "y": float(object_point[1])},
            "strict": bool(strict_segmentation),
            "quality": mask_quality,
        },
        "camera_assumption": {
            "model": "turntable-orbit",
            "rotation_degrees": float(rotation_degrees),
            "rotation_direction": rotation_direction,
            "start_azimuth_deg": float(start_azimuth_deg),
            "elevation_deg": float(elevation_deg),
        },
    }
    bundle_path = output_dir / "multiview_input.json"
    report_path = output_dir / "video_preparation.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if strict_segmentation and not mask_quality["passes_hard_checks"]:
        checks = ", ".join(mask_quality["failed_checks"])
        raise VideoSegmentationError(f"Video masks failed quality checks: {checks}", report=report)
    return PreparedVideo(
        bundle_path=bundle_path,
        report_path=report_path,
        frame_paths=tuple(frame_paths),
        mask_paths=tuple(mask_paths),
        bundle=bundle,
        report=report,
    )
