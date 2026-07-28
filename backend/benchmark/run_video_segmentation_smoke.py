from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from backend.video_pipeline import (
    SAM2_VIDEO_MODELS,
    SAM2_VIDEO_REVISIONS,
    decode_video_candidates,
    evaluate_mask_sequence,
    probe_video,
    segment_frames_sam2_video,
    segment_turntable_frame,
    select_video_frames,
)


def write_fixture(path: Path, *, frame_count: int = 24, size: int = 256) -> np.ndarray:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer is unavailable")
    try:
        for _ in range(frame_count):
            frame = np.full((size, size, 3), 238, dtype=np.uint8)
            cv2.rectangle(frame, (68, 82), (187, 173), (170, 105, 45), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    reference = np.zeros((size, size), dtype=bool)
    reference[82:174, 68:188] = True
    return reference


def mask_ious(masks: list[np.ndarray], references: list[np.ndarray]) -> list[float]:
    scores = []
    for predicted, reference in zip(masks, references, strict=True):
        predicted = predicted.astype(bool)
        reference = reference.astype(bool)
        intersection = int(np.count_nonzero(predicted & reference))
        union = int(np.count_nonzero(predicted | reference))
        scores.append(intersection / float(max(1, union)))
    return scores


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="3dprintpic-sam2-video-smoke-") as temp_dir:
        video_path = Path(temp_dir) / "sam2_fixture.avi"
        reference_mask = write_fixture(video_path)
        probe = probe_video(video_path)
        candidates, decoded_frames, decode_warnings = decode_video_candidates(
            video_path,
            probe,
            selected_frame_count=args.selected_frames,
            frame_max_side=args.frame_size,
        )
        selected = select_video_frames(
            candidates,
            decoded_frames=decoded_frames,
            selected_frame_count=args.selected_frames,
            mode="uniform-frame-sampler",
        )
        frames = [candidate.bgr for candidate in selected]
        learned_masks = segment_frames_sam2_video(
            frames,
            provider=args.provider,
            point=(0.5, 0.5),
            device=args.device,
        )
        baseline_masks = []
        previous_mask = None
        for frame in frames:
            previous_mask = segment_turntable_frame(frame, point=(0.5, 0.5), previous_mask=previous_mask)
            baseline_masks.append(previous_mask)

    references = [reference_mask] * len(learned_masks)
    learned_ious = mask_ious(learned_masks, references)
    baseline_ious = mask_ious(baseline_masks, references)
    agreement_ious = mask_ious(learned_masks, baseline_masks)
    source_indices = [candidate.source_index for candidate in selected]
    quality = evaluate_mask_sequence(learned_masks, source_indices)

    import huggingface_hub
    import torch
    import transformers

    payload = {
        "schema_version": 1,
        "status": "pass" if quality["passes_hard_checks"] and min(learned_ious) >= args.minimum_iou else "fail",
        "provider": args.provider,
        "model_id": SAM2_VIDEO_MODELS[args.provider],
        "model_revision": SAM2_VIDEO_REVISIONS[args.provider],
        "device": args.device,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "fixture": {
            "kind": "static high-contrast rectangle turntable smoke",
            "decoded_frames": decoded_frames,
            "selected_source_indices": source_indices,
            "decode_warnings": decode_warnings,
        },
        "mask_quality": quality,
        "iou_vs_ground_truth": learned_ious,
        "iou_vs_ground_truth_median": float(np.median(learned_ious)),
        "iou_vs_ground_truth_minimum": float(np.min(learned_ious)),
        "turntable_grabcut_iou_vs_ground_truth": baseline_ious,
        "turntable_grabcut_iou_vs_ground_truth_median": float(np.median(baseline_ious)),
        "turntable_grabcut_iou_vs_ground_truth_minimum": float(np.min(baseline_ious)),
        "iou_vs_turntable_grabcut": agreement_ious,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pinned SAM2 video segmentation smoke test.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=sorted(SAM2_VIDEO_MODELS), default="sam2.1-hiera-tiny-video")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selected-frames", type=int, default=3)
    parser.add_argument("--frame-size", type=int, default=256)
    parser.add_argument("--minimum-iou", type=float, default=0.95)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
