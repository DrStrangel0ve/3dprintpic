from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.video_subject_relief import generate_video_subject_relief


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track a subject in a non-turntable video and emit a printable best-frame relief STL."
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-frames", type=int, default=12)
    parser.add_argument("--frame-selection", default="sharpness-motion-selector")
    parser.add_argument("--segmentation-provider", default="sam2.1-hiera-tiny-video")
    parser.add_argument("--segmentation-device", default="auto")
    parser.add_argument("--object-point-x", type=float, default=0.5)
    parser.add_argument("--object-point-y", type=float, default=0.4)
    parser.add_argument("--frame-max-side", type=int, default=960)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--depth-device", default="auto")
    parser.add_argument("--target-dimension", type=int, default=360)
    parser.add_argument("--max-xy-size-mm", type=float, default=120.0)
    parser.add_argument("--relief-height-mm", type=float, default=8.0)
    parser.add_argument("--minimum-feature-mm", type=float, default=0.4)
    parser.add_argument("--allow-mask-warnings", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = generate_video_subject_relief(
        args.input_video,
        args.output_dir,
        selected_frame_count=args.selected_frames,
        frame_selection=args.frame_selection,
        segmentation_provider=args.segmentation_provider,
        segmentation_device=args.segmentation_device,
        object_point=(args.object_point_x, args.object_point_y),
        strict_segmentation=not args.allow_mask_warnings,
        frame_max_side=args.frame_max_side,
        depth_model=args.depth_model,
        depth_device=args.depth_device,
        target_dimension=args.target_dimension,
        max_xy_size_mm=args.max_xy_size_mm,
        relief_height_mm=args.relief_height_mm,
        minimum_feature_mm=args.minimum_feature_mm,
    )
    summary = {
        "status": result.report["status"],
        "motion": result.report["motion"]["classification"],
        "selected_source_index": result.report["selection"]["selected"]["source_index"],
        "stl_path": str(result.stl_path),
        "report_path": str(result.report_path),
        "stl_passes_hard_checks": result.report["stl_diagnostics"]["stl_passes_hard_checks"],
        "stl_failed_checks": result.report["stl_diagnostics"]["stl_failed_checks"],
        "timings": result.report["timings"],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["stl_passes_hard_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
