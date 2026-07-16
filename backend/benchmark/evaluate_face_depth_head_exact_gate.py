"""Replay a trained DAv2 face head through the exact production face pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _strictly_improves,
)
from backend.face_depth_refinement import refine_depth_for_faces


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Depth prediction contains no finite values")
    low = float(np.min(values[finite]))
    high = float(np.max(values[finite]))
    if high <= low:
        return values.copy()
    return ((values - low) / (high - low)).astype(np.float32)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


class BlendedFaceDepth:
    def __init__(self, checkpoint_path: Path, device: str):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_id") != MODEL_ID:
            raise ValueError("Face-head checkpoint model id does not match DAv2 Large")
        if checkpoint.get("model_revision") != MODEL_REVISION:
            raise ValueError("Face-head checkpoint revision does not match the pinned model")
        alpha = float(checkpoint.get("blend_alpha", 0.0))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Face-head checkpoint blend alpha must be in [0, 1]")
        snapshot = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
        )
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=self.dtype,
        ).to(device)
        self.model.eval()
        self.baseline_head = copy.deepcopy(self.model.head).to(
            device=device,
            dtype=torch.float32,
        )
        self.trained_head = copy.deepcopy(self.model.head).to(
            device=device,
            dtype=torch.float32,
        )
        self.trained_head.load_state_dict(checkpoint["head_state_dict"])
        self.baseline_head.eval()
        self.trained_head.eval()
        self.alpha = alpha
        self.snapshot = Path(snapshot)
        self.checkpoint = checkpoint
        self.timings = []
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

    def infer(self, image_path: Path, alpha: float | None = None) -> np.ndarray:
        import torch

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(device=self.device, dtype=self.dtype)
        patch_height = pixel_values.shape[-2] // self.model.config.patch_size
        patch_width = pixel_values.shape[-1] // self.model.config.patch_size
        started = time.perf_counter()
        with torch.inference_mode():
            backbone = self.model.backbone.forward_with_filtered_kwargs(pixel_values)
            hidden = self.model.neck(
                backbone.feature_maps,
                patch_height,
                patch_width,
            )
            feature = hidden[-1].to(dtype=torch.float32)
            baseline = self.baseline_head(
                [feature],
                patch_height,
                patch_width,
            )
            trained = self.trained_head(
                [feature],
                patch_height,
                patch_width,
            )
            blend = self.alpha if alpha is None else float(alpha)
            prediction = (1.0 - blend) * baseline + blend * trained
            prediction = torch.nn.functional.interpolate(
                prediction[:, None],
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]
        self.timings.append(time.perf_counter() - started)
        return _minmax_normalize(prediction.float().cpu().numpy())

    def callback(self, image_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "output_depth_data.npy"
        np.save(path, self.infer(Path(image_path)))
        return str(path)

    def provenance(self) -> dict:
        import torch

        peak_vram = (
            float(torch.cuda.max_memory_allocated(self.device) / (1024**3))
            if self.device.startswith("cuda")
            else 0.0
        )
        return {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": _sha256(self.snapshot / "model.safetensors"),
            "checkpoint_selected_epoch": int(
                self.checkpoint.get("selected_epoch", 0)
            ),
            "blend_alpha": self.alpha,
            "inference_calls": len(self.timings),
            "mean_inference_seconds": (
                float(np.mean(self.timings)) if self.timings else None
            ),
            "peak_vram_gb": peak_vram,
        }


def _quality(run_root: Path, row: dict, candidate_path: Path) -> dict:
    part_paths = {
        name: run_root / row["exact_face_part_masks"]["files"][name]["path"]
        for name in FACE_PART_NAMES
    }
    metrics = _exact_face_depth_quality(
        candidate_path,
        run_root / row["exact_depth"]["path"],
        run_root / row["selection_mask"]["path"],
        expected_scale_sign=-1.0,
        part_mask_paths=part_paths,
    )
    return {
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": metrics["named_part_shape"]["failed_parts"],
        "affine_failed_parts": metrics["named_part_affine_mm"]["failed_parts"],
        "combined_part_failures": len(metrics["named_part_shape"]["failed_parts"])
        + len(metrics["named_part_affine_mm"]["failed_parts"]),
    }


def _summary(rows: list[dict]) -> dict:
    return {
        "row_count": len(rows),
        "combined_part_failures": sum(
            row["combined_part_failures"] for row in rows
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
        "rows": rows,
    }


def evaluate(
    run_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict:
    run_root = Path(run_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    provider = BlendedFaceDepth(checkpoint_path, device)
    baseline_rows = []
    candidate_rows = []
    equivalence = []
    started = time.perf_counter()
    for row in run_summary["rows"]:
        if row["render"].get("scene_kind") != "face":
            continue
        job_dir = run_root.parent / row["variants"]["candidate"]["job_id"]
        cached_local = np.load(
            job_dir
            / "face_refinement"
            / "face_00_depth"
            / "output_depth_data.npy"
        ).astype(np.float32)
        crop_path = job_dir / "face_refinement" / "face_00_input.png"
        reproduced = provider.infer(crop_path, alpha=0.0)
        equivalence.append(
            {
                "row_id": row["row_id"],
                "cached_shape": list(cached_local.shape),
                "reproduced_shape": list(reproduced.shape),
                "correlation": _correlation(cached_local, reproduced),
                "maximum_absolute_difference": float(
                    np.max(np.abs(cached_local - reproduced))
                ),
            }
        )
        baseline_path = job_dir / "output_depth_data_face_refined.npy"
        baseline_rows.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                **_quality(run_root, row, baseline_path),
            }
        )
        row_output = output_dir / row["row_id"]
        refined_path, metadata = refine_depth_for_faces(
            run_root / row["source"]["path"],
            job_dir / "output_depth_data.npy",
            row_output,
            infer_depth=provider.callback,
            mode="on",
            detection_roi_mask=run_root / row["selection_mask"]["path"],
        )
        candidate_rows.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "refined_faces": metadata["refined_faces"],
                "crop_bbox": metadata["faces"][0].get("crop_bbox"),
                "max_abs_correction": metadata["faces"][0].get(
                    "max_abs_correction"
                ),
                **_quality(run_root, row, Path(refined_path)),
            }
        )
    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    by_id = {row["row_id"]: row for row in candidate_rows}
    baseline_by_id = {row["row_id"]: row for row in baseline_rows}
    per_row_no_regression = all(
        by_id[row_id]["combined_part_failures"]
        <= baseline_by_id[row_id]["combined_part_failures"]
        for row_id in by_id
    )
    hard_row = "small_side_lit_shelves_256"
    decision = {
        "local_baseline_equivalent": bool(
            all(
                row["cached_shape"] == row["reproduced_shape"]
                and row["correlation"] >= 0.999
                and row["maximum_absolute_difference"] <= 0.02
                for row in equivalence
            )
        ),
        "aggregate_strictly_improves": _strictly_improves(candidate, baseline),
        "per_row_no_failure_regression": per_row_no_regression,
        "hard_small_face_improves": bool(
            by_id[hard_row]["combined_part_failures"]
            < baseline_by_id[hard_row]["combined_part_failures"]
        ),
    }
    decision["eligible_for_full_stl_replay"] = bool(all(decision.values()))
    evidence = {
        "schema_version": 1,
        "method": "trained_dav2_head_blend_through_unchanged_face_refinement",
        "source_revision": run_summary["server_provenance"]["revision"],
        "source_summary_sha256": _sha256(run_root / "summary.json"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "provider": provider.provenance(),
        "local_baseline_equivalence": equivalence,
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.run_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_full_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
