from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, make_procedural_mesh, render_mesh
from backend.benchmark.metrics import mesh_surface_distance_metrics
from backend.benchmark.run_image_to_mesh_provider import run_provider
from backend.stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
from backend.video_pipeline import prepare_turntable_video, video_model_preflight


def write_turntable_video(path: Path, mesh, *, frame_count: int, size: int) -> dict[int, np.ndarray]:
    import cv2

    config = RenderConfig(size=size, background_rgb=(0.91, 0.93, 0.96))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer is unavailable")
    silhouettes = {}
    try:
        for index in range(frame_count):
            camera = CameraSpec(azimuth_deg=index * 360.0 / frame_count, elevation_deg=8.0)
            result = render_mesh(mesh, camera=camera, config=config, base_color=(80, 145, 205))
            rgb = np.clip(result.rgb * 255.0, 0, 255).astype(np.uint8)
            writer.write(rgb[:, :, ::-1])
            silhouettes[index] = result.silhouette.astype(bool)
    finally:
        writer.release()
    return silhouettes


def visual_hull_args(prepared, output_dir: Path, resolution: int) -> SimpleNamespace:
    return SimpleNamespace(
        provider="multiview-visual-hull",
        input_image=prepared.frame_paths[0],
        input_bundle=prepared.bundle_path,
        output_mesh=output_dir / "output_mesh.ply",
        output_stl=output_dir / "output_model.stl",
        raw_output_mesh=output_dir / "output_mesh_raw.ply",
        mesh_repair="printable",
        mesh_target_max_dimension=96.0,
        mesh_min_bbox_dimension=12.0,
        mesh_max_bbox_aspect_ratio=0.0,
        mesh_target_bbox_extents=None,
        mesh_target_faces=40000,
        mesh_max_normalized_face_density_log1p=0.0,
        visual_hull_resolution=resolution,
        visual_hull_grid_extent=1.9,
        visual_hull_ortho_scale=2.0,
        visual_hull_mask_dilate=1,
    )


def mask_ious(prepared, ground_truth: dict[int, np.ndarray]) -> list[float]:
    scores = []
    indices = prepared.report["sampling"]["selected_source_indices"]
    for source_index, mask_path in zip(indices, prepared.mask_paths, strict=True):
        predicted = np.asarray(Image.open(mask_path).convert("L")) > 127
        reference = ground_truth[int(source_index)]
        intersection = np.count_nonzero(predicted & reference)
        union = np.count_nonzero(predicted | reference)
        scores.append(float(intersection / max(1, union)))
    return scores


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for method in sorted({row["frame_selection"] for row in rows}):
        selected = [row for row in rows if row["frame_selection"] == method]
        summaries.append(
            {
                "frame_selection": method,
                "sample_count": len(selected),
                "all_runs_succeeded": all(row["run_succeeded"] for row in selected),
                "all_mask_quality_gates_passed": all(row["mask_quality_passed"] for row in selected),
                "all_stl_hard_checks_passed": all(row["stl_hard_checks_passed"] for row in selected),
                "mask_iou_median": statistics.median(row["mask_iou_median"] for row in selected),
                "mask_iou_minimum": min(row["mask_iou_minimum"] for row in selected),
                "mesh_chamfer_l1_median": statistics.median(row["mesh_chamfer_l1"] for row in selected),
                "mesh_hausdorff95_median": statistics.median(row["mesh_hausdorff95"] for row in selected),
                "runtime_seconds_median": statistics.median(row["runtime_seconds"] for row in selected),
            }
        )
    return summaries


def markdown_report(payload: dict) -> str:
    lines = [
        "# Video Pipeline Smoke Evidence",
        "",
        "This deterministic synthetic turntable slice checks video decoding, frame selection, temporal object masks,",
        "turntable-camera assignment, visual-hull reconstruction, printable STL repair, and ground-truth surface distance.",
        "",
        "| Frame selector | Samples | Mask IoU median | Mask IoU minimum | Chamfer L1 median | H95 median | STL gates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {frame_selection} | {sample_count} | {mask_iou_median:.4f} | {mask_iou_minimum:.4f} | "
            "{mesh_chamfer_l1_median:.4f} | {mesh_hausdorff95_median:.4f} | {stl} |".format(
                **row,
                stl="pass" if row["all_stl_hard_checks_passed"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            f"Overall gate: **{'pass' if payload['overall_passed'] else 'fail'}**.",
            f"Recommended frame selector: **`{payload['recommended_frame_selection']}`**, chosen by median final-mesh Chamfer among passing lanes.",
            "",
            "The built-in turntable lane is a controlled-capture baseline. SAM 2 video, SAM 3.1, and VGGT-Omega",
            "readiness is recorded in `results.json`; unavailable dependencies or gated checkpoints are reported as",
            "setup blockers and are not counted as successful model runs.",
            "",
            "A separate pinned learned-segmentation result can be generated with",
            "`backend.benchmark.run_video_segmentation_smoke` and stored as `sam2_tiny_cpu_smoke.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="3dprintpic-video-smoke-") as temp_dir:
        root = Path(temp_dir)
        for sample_index in range(args.samples):
            mesh = make_procedural_mesh(sample_index)
            source_mesh = root / f"sample_{sample_index}_source.ply"
            mesh.export(source_mesh)
            video_path = root / f"sample_{sample_index}.avi"
            ground_truth = write_turntable_video(
                video_path,
                mesh,
                frame_count=args.video_frames,
                size=args.frame_size,
            )
            for frame_selection in ("uniform-frame-sampler", "sharpness-motion-selector"):
                run_started = time.perf_counter()
                run_dir = root / f"sample_{sample_index}_{frame_selection}"
                prepared = prepare_turntable_video(
                    video_path,
                    run_dir / "prepared",
                    selected_frame_count=args.selected_frames,
                    frame_selection=frame_selection,
                    segmentation_provider="turntable-grabcut",
                    object_point=(0.5, 0.5),
                    frame_max_side=args.frame_size,
                )
                ious = mask_ious(prepared, ground_truth)
                mesh_path, stl_path = run_provider(visual_hull_args(prepared, run_dir, args.visual_hull_resolution))
                diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
                surface = mesh_surface_distance_metrics(source_mesh, mesh_path, max_points=4096)
                stl_passed = all(
                    diagnostics.get(key)
                    for key in (
                        "stl_exists",
                        "stl_is_watertight",
                        "stl_is_volume",
                        "stl_is_manifold",
                        "stl_winding_consistent",
                        "stl_positive_volume",
                        "stl_single_component",
                    )
                )
                rows.append(
                    {
                        "sample_index": sample_index,
                        "frame_selection": frame_selection,
                        "segmentation_provider": "turntable-grabcut",
                        "reconstruction_provider": "multiview-visual-hull",
                        "run_succeeded": bool(mesh_path.exists() and stl_path and stl_path.exists()),
                        "selected_source_indices": prepared.report["sampling"]["selected_source_indices"],
                        "mask_quality_passed": prepared.report["segmentation"]["quality"]["passes_hard_checks"],
                        "mask_quality": prepared.report["segmentation"]["quality"]["aggregate"],
                        "mask_ious": ious,
                        "mask_iou_median": float(np.median(ious)),
                        "mask_iou_minimum": float(np.min(ious)),
                        "mesh_chamfer_l1": surface["chamfer_l1"],
                        "mesh_chamfer_rmse": surface["chamfer_rmse"],
                        "mesh_hausdorff95": surface["hausdorff95"],
                        "stl_hard_checks_passed": stl_passed,
                        "stl_diagnostics": diagnostics,
                        "runtime_seconds": round(time.perf_counter() - run_started, 3),
                    }
                )

    summary = summarize(rows)
    overall_passed = all(
        row["all_runs_succeeded"]
        and row["all_mask_quality_gates_passed"]
        and row["all_stl_hard_checks_passed"]
        and row["mask_iou_median"] >= args.minimum_mask_iou_median
        and row["mask_iou_minimum"] >= args.minimum_mask_iou
        for row in summary
    )
    passing_summaries = [
        row
        for row in summary
        if row["all_runs_succeeded"]
        and row["all_mask_quality_gates_passed"]
        and row["all_stl_hard_checks_passed"]
    ]
    recommended_frame_selection = min(
        passing_summaries,
        key=lambda row: (row["mesh_chamfer_l1_median"], row["mesh_hausdorff95_median"]),
    )["frame_selection"] if passing_summaries else None
    payload = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": vars(args),
        "model_preflight": video_model_preflight(),
        "rows": rows,
        "summary": summary,
        "recommended_frame_selection": recommended_frame_selection,
        "overall_passed": overall_passed,
        "total_runtime_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic end-to-end video-to-STL smoke benchmark.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--video-frames", type=int, default=24)
    parser.add_argument("--selected-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--visual-hull-resolution", type=int, default=24)
    parser.add_argument("--minimum-mask-iou-median", type=float, default=0.75)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.60)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"overall_passed": payload["overall_passed"], "summary": payload["summary"]}, indent=2))
    if not payload["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
