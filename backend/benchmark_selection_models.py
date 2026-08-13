from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from . import main
except ImportError:
    import main


DEFAULT_POINTS = {"center": (0.5, 0.5)}
SAM3_TRACKER_MODEL_ID = "facebook/sam3"
SAM3_TRACKER_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_PERSON_MODEL_ID = "sam3-concept-person"
SAM3_TAXONOMY_MODEL_ID = "sam3-concept-taxonomy"
SAM3_SELECTION_CONCEPTS = tuple(main.SAM3_SELECTION_CONCEPTS)
SAM3_PERSON_MASK_THRESHOLD = 0.2
SAM3_PERSON_MAX_HOLE_IMAGE_RATIO = 0.001


def _load_benchmark_model(model_id: str):
    if model_id != "sam3-tracker":
        processor, model, resolved_model_id = main.load_sam2_selection_model(model_id, "cuda")
        revision = main.SELECTION_MODEL_REVISIONS.get(resolved_model_id)
        return processor, model, resolved_model_id, revision

    from huggingface_hub import hf_hub_download
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    snapshot = Path(
        hf_hub_download(
            SAM3_TRACKER_MODEL_ID,
            filename="config.json",
            revision=SAM3_TRACKER_REVISION,
            local_files_only=True,
        )
    ).parent
    processor = Sam3TrackerProcessor.from_pretrained(snapshot, local_files_only=True)
    model = Sam3TrackerModel.from_pretrained(
        snapshot,
        dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    return processor, model, SAM3_TRACKER_MODEL_ID, SAM3_TRACKER_REVISION


def _candidate_bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.where(mask)
    if not rows.size:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def benchmark_sam3_person(
    image: Image.Image,
    points: dict[str, tuple[float, float]],
    output_dir: Path,
) -> dict:
    from huggingface_hub import hf_hub_download
    from transformers import Sam3Model, Sam3Processor

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshot = Path(
        hf_hub_download(
            SAM3_TRACKER_MODEL_ID,
            filename="config.json",
            revision=SAM3_TRACKER_REVISION,
            local_files_only=True,
        )
    ).parent
    load_started = time.perf_counter()
    processor = Sam3Processor.from_pretrained(snapshot, local_files_only=True)
    model = Sam3Model.from_pretrained(
        snapshot,
        dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    load_peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    inputs = processor(images=image, text="person", return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    elapsed_seconds = time.perf_counter() - started
    inference_peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=SAM3_PERSON_MASK_THRESHOLD,
        target_sizes=inputs["original_sizes"].detach().cpu().tolist(),
    )[0]
    raw_candidates = processed["masks"].detach().cpu().numpy().astype(bool)
    scores = processed["scores"].detach().cpu().numpy().reshape(-1)
    candidates = np.asarray(
        [
            main.fill_bounded_selection_holes(
                candidate,
                SAM3_PERSON_MAX_HOLE_IMAGE_RATIO,
            )
            for candidate in raw_candidates
        ],
        dtype=bool,
    )

    candidate_rows = [
        {
            "index": index,
            "score": float(scores[index]),
            "raw_coverage": float(raw_candidates[index].mean()),
            "coverage": float(candidate.mean()),
            "filled_pixels": int(candidate.sum() - raw_candidates[index].sum()),
            "bbox_xyxy": _candidate_bbox(candidate),
        }
        for index, candidate in enumerate(candidates)
    ]
    rows = {}
    for name, (normalized_x, normalized_y) in points.items():
        pixel_points = main.selection_points_to_pixels(
            [{"x": normalized_x, "y": normalized_y}],
            *image.size,
        )
        px = int(np.clip(round(pixel_points[0][0]), 0, image.width - 1))
        py = int(np.clip(round(pixel_points[0][1]), 0, image.height - 1))
        matching = [index for index, candidate in enumerate(candidates) if candidate[py, px]]
        if not matching:
            rows[name] = {"selected_index": None, "selected_coverage": 0.0}
            continue
        selected_index = max(matching, key=lambda index: float(scores[index]))
        selected = main.seeded_sam2_component(candidates[selected_index], pixel_points)
        Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
            output_dir / f"{SAM3_PERSON_MODEL_ID}_{name}_selected.png"
        )
        rows[name] = {
            "selected_index": int(selected_index),
            "selected_coverage": float(selected.mean()),
        }

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "resolved_model_id": SAM3_TRACKER_MODEL_ID,
        "resolved_revision": SAM3_TRACKER_REVISION,
        "concept": "person",
        "detection_threshold": 0.3,
        "mask_threshold": SAM3_PERSON_MASK_THRESHOLD,
        "max_hole_image_ratio": SAM3_PERSON_MAX_HOLE_IMAGE_RATIO,
        "parameter_count": parameter_count,
        "load_seconds": load_seconds,
        "load_peak_allocated_gib": load_peak_allocated_gib,
        "inference_seconds": elapsed_seconds,
        "inference_peak_allocated_gib": inference_peak_allocated_gib,
        "candidate_count": len(candidates),
        "candidates": candidate_rows,
        "points": rows,
    }


def benchmark_sam3_taxonomy(
    image: Image.Image,
    points: dict[str, tuple[float, float]],
    output_dir: Path,
) -> dict:
    from huggingface_hub import hf_hub_download
    from transformers import Sam3Model, Sam3Processor

    snapshot = Path(
        hf_hub_download(
            SAM3_TRACKER_MODEL_ID,
            filename="config.json",
            revision=SAM3_TRACKER_REVISION,
            local_files_only=True,
        )
    ).parent
    processor = Sam3Processor.from_pretrained(snapshot, local_files_only=True)
    model = Sam3Model.from_pretrained(
        snapshot,
        dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    concept_rows = {}
    point_rows = {name: [] for name in points}
    for concept in SAM3_SELECTION_CONCEPTS:
        inputs = processor(images=image, text=concept, return_tensors="pt").to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        inference_seconds = time.perf_counter() - started
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,
            mask_threshold=SAM3_PERSON_MASK_THRESHOLD,
            target_sizes=inputs["original_sizes"].detach().cpu().tolist(),
        )[0]
        masks = processed["masks"].detach().cpu().numpy().astype(bool)
        scores = processed["scores"].detach().cpu().numpy().reshape(-1)
        kept = []
        for index, (mask, score) in enumerate(zip(masks, scores)):
            cleaned = main.fill_bounded_selection_holes(
                mask,
                SAM3_PERSON_MAX_HOLE_IMAGE_RATIO,
            )
            coverage = float(cleaned.mean())
            if coverage < 0.0005 or coverage > main.SAM2_SELECTION_MAX_COVERAGE:
                continue
            kept.append((index, cleaned, float(score)))
            for point_name, coordinates in points.items():
                pixel = main.selection_points_to_pixels(
                    [{"x": coordinates[0], "y": coordinates[1]}],
                    *image.size,
                )[0]
                px = int(np.clip(round(pixel[0]), 0, image.width - 1))
                py = int(np.clip(round(pixel[1]), 0, image.height - 1))
                if cleaned[py, px]:
                    point_rows[point_name].append(
                        {
                            "concept": concept,
                            "index": index,
                            "score": float(score),
                            "coverage": coverage,
                            "bbox_xyxy": _candidate_bbox(cleaned),
                        }
                    )
                    selected = main.seeded_sam2_component(cleaned, [pixel])
                    Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
                        output_dir / f"{point_name}_{concept}_{index}.png"
                    )
        concept_rows[concept] = {
            "inference_seconds": inference_seconds,
            "raw_candidate_count": len(masks),
            "kept_candidate_count": len(kept),
        }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "resolved_model_id": SAM3_TRACKER_MODEL_ID,
        "resolved_revision": SAM3_TRACKER_REVISION,
        "concepts": concept_rows,
        "points": point_rows,
    }


def benchmark_model(
    image: Image.Image,
    model_id: str,
    points: dict[str, tuple[float, float]],
    output_dir: Path,
) -> dict:
    if model_id == SAM3_TAXONOMY_MODEL_ID:
        return benchmark_sam3_taxonomy(image, points, output_dir)
    if model_id == SAM3_PERSON_MODEL_ID:
        return benchmark_sam3_person(image, points, output_dir)

    main.SELECTION_MODEL_CACHE.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    processor, model, resolved_model_id, resolved_revision = _load_benchmark_model(model_id)
    load_seconds = time.perf_counter() - load_started
    load_peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    rows = {}
    for name, (normalized_x, normalized_y) in points.items():
        pixel_points = main.selection_points_to_pixels(
            [{"x": normalized_x, "y": normalized_y}],
            *image.size,
        )
        inputs = processor(
            images=image,
            input_points=[[pixel_points]],
            input_labels=[[[1]]],
            return_tensors="pt",
        ).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        elapsed_seconds = time.perf_counter() - started
        post_masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            mask_threshold=0.0,
            binarize=True,
            max_hole_area=256.0,
            max_sprinkle_area=128.0,
        )[0]
        candidates = post_masks.detach().cpu().numpy().astype(bool)
        candidates = candidates.reshape((-1, image.height, image.width))
        scores = outputs.iou_scores.detach().cpu().reshape(-1).numpy()
        selected, selected_index = main.select_sam2_object_candidate(
            candidates,
            scores,
            pixel_points,
        )
        candidate_rows = []
        for index, candidate in enumerate(candidates):
            cleaned = main.seeded_sam2_component(candidate, pixel_points)
            candidate_rows.append(
                {
                    "index": index,
                    "score": float(scores[index]),
                    "coverage": float(cleaned.mean()),
                    "bbox_xyxy": _candidate_bbox(cleaned),
                }
            )
        Image.fromarray(selected.astype(np.uint8) * 255, mode="L").save(
            output_dir / f"{model_id}_{name}_selected.png"
        )
        rows[name] = {
            "selected_index": int(selected_index),
            "selected_coverage": float(selected.mean()),
            "candidates": candidate_rows,
            "seconds": elapsed_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    del model, processor
    main.SELECTION_MODEL_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "resolved_model_id": resolved_model_id,
        "resolved_revision": resolved_revision,
        "parameter_count": parameter_count,
        "load_seconds": load_seconds,
        "load_peak_allocated_gib": load_peak_allocated_gib,
        "points": rows,
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sam2.1-hiera-tiny", "sam2.1-hiera-large"],
    )
    parser.add_argument(
        "--points-json",
        type=Path,
        help='JSON object mapping prompt names to normalized [x, y] coordinates',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = DEFAULT_POINTS
    if args.points_json:
        raw_points = json.loads(args.points_json.read_text(encoding="utf-8"))
        points = {
            str(name): (float(coordinates[0]), float(coordinates[1]))
            for name, coordinates in raw_points.items()
        }
    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")
    results = {
        model_id: benchmark_model(image, model_id, points, args.output_dir)
        for model_id in args.models
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main_cli()
