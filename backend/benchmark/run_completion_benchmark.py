import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from skimage.restoration import inpaint_biharmonic

from backend.benchmark.metrics import (
    align_depth,
    depth_metrics,
    load_bool_mask,
    load_mask,
    load_rgb,
    mae_rgb,
    psnr_rgb,
    seam_mae,
    silhouette_iou,
    silhouette_iou_from_reference_mask,
    surface_distance_metrics,
    ssim_rgb,
)
from backend.pic_to_3d import (
    MODERN_INPAINT_MODELS,
    complete_image,
    depth_data_to_3d_model,
    process_image_get_depth_data,
)


METADATA_FIELDS = {
    "sample_id",
    "method",
    "raw_completed_image",
    "completed_image",
    "full_image",
    "masked_image",
    "mask",
    "gt_depth",
    "gt_silhouette",
    "depth_data",
    "depth_preview",
    "source",
    "asset_id",
    "asset_path",
    "asset_category",
    "asset_source_split",
    "asset_key",
    "view_index",
    "stl_model",
    "prompt",
    "steps",
    "guidance",
    "seed",
    "inpaint_max_dimension",
    "edit_mask_fill",
    "model_name",
    "lora_weights",
    "lora_scale",
}

DEFAULT_PROMPT_TEMPLATE = (
    "Complete the missing half of the same {category} with matching geometry, "
    "lighting, viewpoint, and background. The masked half must contain the missing "
    "{category}, not an empty background."
)


def infer_category(sample):
    asset_category = str(sample.get("asset_category") or "")
    if asset_category:
        return asset_category.replace("_", " ").replace("-", " ")
    asset_id = str(sample.get("asset_id") or "")
    if asset_id:
        return re.sub(r"[_-]\d+$", "", asset_id).replace("_", " ").replace("-", " ")
    source = str(sample.get("source") or "")
    if source == "procedural":
        return "object"
    return "object"


def render_prompt(sample, prompt_template):
    values = {
        "sample_id": sample.get("id", ""),
        "asset_id": sample.get("asset_id", ""),
        "category": infer_category(sample),
        "source": sample.get("source", ""),
        "completion_mode": sample.get("completion_mode", ""),
    }
    try:
        return prompt_template.format(**values)
    except (KeyError, ValueError):
        return prompt_template


def experiment_key(sample, method, args):
    is_modern = method in MODERN_INPAINT_MODELS
    edit_mask_fill_arg = getattr(args, "edit_mask_fill", "input")
    edit_mask_fill = "" if edit_mask_fill_arg in (None, "", "input") else edit_mask_fill_arg
    return {
        "sample_id": sample["id"],
        "method": method,
        "prompt": render_prompt(sample, args.prompt) if is_modern else "",
        "steps": args.steps if is_modern else "",
        "guidance": args.guidance if is_modern else "",
        "seed": args.seed if is_modern else "",
        "inpaint_max_dimension": args.inpaint_max_dimension if is_modern else "",
        "edit_mask_fill": edit_mask_fill if is_modern else "",
        "model_name": (args.model_name or "") if is_modern else "",
        "lora_weights": (args.lora_weights or "") if is_modern else "",
        "lora_scale": (args.lora_scale if args.lora_scale is not None else "") if is_modern else "",
    }


def key_tuple(row):
    def text(value):
        return "" if value is None else str(value)

    return (
        text(row.get("sample_id", "")),
        text(row.get("method", "")),
        text(row.get("prompt", "")),
        text(row.get("steps", "")),
        text(row.get("guidance", "")),
        text(row.get("seed", "")),
        text(row.get("inpaint_max_dimension", "")),
        text(row.get("edit_mask_fill", "")),
        text(row.get("model_name", "")),
        text(row.get("lora_weights", "")),
        text(row.get("lora_scale", "")),
    )


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


def sample_asset_key(sample):
    return str(sample.get("asset_key") or sample.get("asset_path") or sample.get("asset_id") or sample.get("id") or "")


def source_split_from_key(value):
    parts = str(value or "").replace("\\", "/").split("/")
    for part in parts:
        split = part.lower()
        if split == "validation":
            return "val"
        if split in {"train", "test", "val"}:
            return split
    return "unknown"


def sample_category(sample):
    return str(sample.get("asset_category") or infer_category(sample) or "unknown")


def sample_source_split(sample):
    explicit = str(sample.get("asset_source_split") or "")
    if explicit:
        return explicit
    return source_split_from_key(sample_asset_key(sample))


def category_counts(rows):
    counts = {}
    for row in rows:
        category = sample_category(row)
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def source_split_counts(rows):
    counts = {}
    for row in rows:
        source_split = sample_source_split(row)
        counts[source_split] = counts.get(source_split, 0) + 1
    return dict(sorted(counts.items()))


def source_split_counts_from_asset_keys(asset_keys):
    counts = {}
    for key in asset_keys:
        source_split = source_split_from_key(key)
        counts[source_split] = counts.get(source_split, 0) + 1
    return dict(sorted(counts.items()))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_report_path(lora_weights):
    if not lora_weights:
        return None
    path = Path(lora_weights)
    candidates = []
    if path.is_dir():
        candidates.append(path / "training_report.json")
    else:
        candidates.append(path.with_name("training_report.json"))
        candidates.append(path.parent / "training_report.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prompt_uses_template_fields(prompt):
    return bool(re.search(r"\{(category|sample_id|asset_id|source|completion_mode)\}", str(prompt or "")))


def eval_prompt_family(prompt):
    return "templated" if prompt_uses_template_fields(prompt) else "literal"


def training_provenance_fields(train_report, report_path, eval_prompt):
    if not train_report:
        return {}
    train_template = str(train_report.get("prompt_template") or train_report.get("args", {}).get("prompt") or "")
    train_family = str(train_report.get("prompt_family") or "")
    eval_family = eval_prompt_family(eval_prompt)
    prompt_matches = True
    mismatch_detail = ""
    if train_template and train_template != str(eval_prompt or ""):
        prompt_matches = False
        mismatch_detail = "train_prompt_template differs from eval_prompt_template"
    elif train_family == "literal" and eval_family == "templated":
        prompt_matches = False
        mismatch_detail = "literal training prompts evaluated with templated prompts"

    return {
        "training_report_sha256": file_sha256(report_path) if report_path else "",
        "train_loss_recipe": train_report.get("loss_recipe", ""),
        "train_metadata_sha256": train_report.get("metadata_sha256", ""),
        "train_pair_export_report": train_report.get("pair_export_report", ""),
        "train_pair_export_sha256": train_report.get("pair_export_sha256", ""),
        "train_prompt_family": train_family,
        "train_prompt_template": train_template,
        "eval_prompt_family": eval_family,
        "eval_prompt_template": eval_prompt or "",
        "prompt_family_matches_eval": prompt_matches,
        "prompt_family_mismatch_detail": mismatch_detail,
        "train_rows": train_report.get("rows", 0),
        "train_asset_count": train_report.get("asset_count", 0),
        "train_max_steps": train_report.get("max_train_steps", ""),
        "train_final_step": train_report.get("final_step", ""),
        "train_resolution": train_report.get("resolution", ""),
        "train_rank": train_report.get("rank", ""),
        "train_mask_loss_weight": train_report.get("mask_loss_weight", ""),
        "train_seam_loss_weight": train_report.get("seam_loss_weight", ""),
        "train_object_loss_weight": train_report.get("object_loss_weight", ""),
        "train_final_loss": train_report.get("final_loss", ""),
        "train_best_loss": train_report.get("best_loss", ""),
        "train_base_model": train_report.get("base_model", ""),
        "adapter_file": train_report.get("adapter_file", ""),
        "adapter_sha256": train_report.get("adapter_sha256", ""),
    }


def write_split_audit(output_dir, samples, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_asset_keys = sorted({key for key in (sample_asset_key(sample) for sample in samples) if key})
    report_path = training_report_path(args.lora_weights)
    train_report = None
    if report_path:
        with report_path.open(encoding="utf-8") as report_file:
            train_report = json.load(report_file)
    train_asset_keys = sorted(set(train_report.get("asset_keys", []))) if train_report else []
    train_source_split_counts = {}
    if train_report:
        train_source_split_counts = train_report.get("source_split_counts", {}) or source_split_counts_from_asset_keys(train_asset_keys)
    overlap = sorted(set(train_asset_keys) & set(eval_asset_keys))
    audit = {
        "manifest": args.manifest,
        "start_index": args.start_index,
        "limit": args.limit,
        "eval_n": len(samples),
        "eval_category_counts": category_counts(samples),
        "eval_source_split_counts": source_split_counts(samples),
        "eval_asset_count": len(eval_asset_keys),
        "eval_asset_keys": eval_asset_keys,
        "lora_weights": args.lora_weights or "",
        "training_report": str(report_path) if report_path else "",
        "train_category_counts": train_report.get("category_counts", {}) if train_report else {},
        "train_source_split_counts": train_source_split_counts,
        "train_asset_count": train_report.get("asset_count", 0) if train_report else 0,
        "train_eval_asset_overlap_count": len(overlap),
        "train_eval_asset_overlap_keys": overlap,
        **training_provenance_fields(train_report, report_path, args.prompt),
    }
    (output_dir / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def run_mirror(sample, output_dir):
    completed_path, _ = complete_image(
        sample["masked_image"],
        output_dir=str(output_dir),
        mode=sample["completion_mode"],
        provider="mirror",
    )
    return completed_path


def run_mirror_seam_repair(sample, output_dir):
    completed_path, _ = complete_image(
        sample["masked_image"],
        output_dir=str(output_dir),
        mode=sample["completion_mode"],
        provider="mirror-seam-repair",
    )
    return completed_path


def run_masked(sample, output_dir):
    out = output_dir / "completed_input.png"
    Image.open(sample["masked_image"]).convert("RGB").save(out)
    return str(out)


def run_biharmonic(sample, output_dir):
    image = load_rgb(sample["masked_image"])
    mask = load_mask(sample["mask"])
    result = inpaint_biharmonic(image, mask, channel_axis=-1)
    out = output_dir / "completed_input.png"
    Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8)).save(out)
    return str(out)


def run_modern(sample, output_dir, method, prompt, steps, seed, guidance, inpaint_max_dimension, edit_mask_fill, model_name, lora_weights, lora_scale):
    rendered_prompt = render_prompt(sample, prompt)
    completed_path, _ = complete_image(
        sample["masked_image"],
        output_dir=str(output_dir),
        mode=sample["completion_mode"],
        provider=method,
        prompt=rendered_prompt,
        model_name=model_name,
        num_inference_steps=steps,
        seed=seed,
        guidance_scale=guidance,
        inpaint_max_dimension=inpaint_max_dimension,
        edit_mask_fill=edit_mask_fill,
        lora_weights=lora_weights,
        lora_scale=lora_scale,
    )
    raw_completed_path = output_dir / "raw_completed_input.png"
    if raw_completed_path.exists():
        return str(raw_completed_path)
    return completed_path


def preserve_visible_pixels(completed_path, sample, output_dir):
    completed = Image.open(completed_path).convert("RGB")
    masked = Image.open(sample["masked_image"]).convert("RGB")
    mask = Image.open(sample["mask"]).convert("L")
    preserved = Image.composite(completed, masked, mask)
    out = output_dir / "completed_visible_preserved.png"
    preserved.save(out)
    return str(out)


def resize_completed_to_sample(completed_path, sample, output_dir):
    completed = Image.open(completed_path).convert("RGB")
    reference = Image.open(sample["full_image"]).convert("RGB")
    if completed.size == reference.size:
        return completed_path
    resized_path = output_dir / "completed_resized_to_sample.png"
    completed.resize(reference.size, resample=Image.Resampling.BICUBIC).save(resized_path)
    return str(resized_path)


def _resize_mask(mask, shape):
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) > 127


def _prefixed(prefix, values):
    return {f"{prefix}{key}": value for key, value in values.items()}


def stl_diagnostics(stl_path):
    import trimesh

    path = Path(stl_path)
    diagnostics = {
        "stl_model": str(path),
        "stl_exists": path.exists(),
        "stl_file_size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists():
        return diagnostics

    loaded = trimesh.load_mesh(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    extents = np.asarray(mesh.extents, dtype=np.float64)
    has_3d_extents = extents.shape == (3,) and np.all(np.isfinite(extents))
    finite_extents = extents[np.isfinite(extents)]
    bbox_has_volume = bool(has_3d_extents and np.all(extents > 0))
    bbox_volume = float(np.prod(extents)) if has_3d_extents else math.nan
    z_range = float(extents[2]) if extents.shape == (3,) and np.isfinite(extents[2]) else math.nan
    min_extent = float(np.min(finite_extents)) if len(finite_extents) else math.nan
    max_extent = float(np.max(finite_extents)) if len(finite_extents) else math.nan
    if not len(finite_extents):
        aspect_ratio = math.nan
    elif min_extent <= 0:
        aspect_ratio = math.inf
    else:
        aspect_ratio = float(max_extent / min_extent)
    signed_volume = float(mesh.volume) if np.isfinite(mesh.volume) else math.nan
    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = math.nan
    component_excess = abs(component_count - 1) if np.isfinite(component_count) else math.nan
    faces_per_bbox_volume = (
        float(len(mesh.faces) / bbox_volume) if np.isfinite(bbox_volume) and bbox_volume > 0 else math.nan
    )
    diagnostics.update(
        {
            "stl_vertices": int(len(mesh.vertices)),
            "stl_faces": int(len(mesh.faces)),
            "stl_is_watertight": bool(mesh.is_watertight),
            "stl_is_volume": bool(mesh.is_volume),
            "stl_winding_consistent": bool(mesh.is_winding_consistent),
            "stl_component_count": component_count,
            "stl_single_component": bool(component_count == 1) if np.isfinite(component_count) else False,
            "stl_component_excess": component_excess,
            "stl_euler_number": int(mesh.euler_number) if mesh.euler_number is not None else math.nan,
            "stl_surface_area": float(mesh.area) if np.isfinite(mesh.area) else math.nan,
            "stl_volume": signed_volume,
            "stl_volume_abs": abs(signed_volume) if np.isfinite(signed_volume) else math.nan,
            "stl_positive_volume": bool(np.isfinite(signed_volume) and signed_volume > 0),
            "stl_z_range": z_range,
            "stl_bbox_x": float(extents[0]) if extents.shape == (3,) and np.isfinite(extents[0]) else math.nan,
            "stl_bbox_y": float(extents[1]) if extents.shape == (3,) and np.isfinite(extents[1]) else math.nan,
            "stl_bbox_z": z_range,
            "stl_bbox_min_dimension": min_extent,
            "stl_bbox_max_dimension": max_extent,
            "stl_bbox_has_volume": bbox_has_volume,
            "stl_bbox_aspect_ratio": aspect_ratio,
            "stl_bbox_volume": bbox_volume,
            "stl_faces_per_bbox_volume": faces_per_bbox_volume,
            "stl_faces_per_bbox_volume_log1p": (
                float(math.log1p(faces_per_bbox_volume)) if np.isfinite(faces_per_bbox_volume) else math.nan
            ),
        }
    )
    return diagnostics


def evaluate_sample(sample, method, raw_completed_path, completed_path, output_dir, args):
    full = load_rgb(sample["full_image"])
    raw_completed = load_rgb(raw_completed_path)
    completed = load_rgb(completed_path)
    mask = load_mask(sample["mask"])
    visible = ~mask
    if full.shape != completed.shape or full.shape != raw_completed.shape:
        raise ValueError(
            f"Image shape mismatch for {sample['id']}: "
            f"full={full.shape}, raw={raw_completed.shape}, completed={completed.shape}"
        )
    row = {
        **experiment_key(sample, method, args),
        "sample_id": sample["id"],
        "method": method,
        "raw_completed_image": raw_completed_path,
        "completed_image": completed_path,
        "full_image": sample.get("full_image", ""),
        "masked_image": sample.get("masked_image", ""),
        "mask": sample.get("mask", ""),
        "gt_depth": sample.get("gt_depth", ""),
        "gt_silhouette": sample.get("gt_silhouette") or sample.get("silhouette", ""),
        "source": sample.get("source", ""),
        "asset_id": sample.get("asset_id", ""),
        "asset_path": sample.get("asset_path", ""),
        "asset_category": sample.get("asset_category", ""),
        "asset_source_split": sample.get("asset_source_split", ""),
        "asset_key": sample.get("asset_key", ""),
        "view_index": sample.get("view_index", ""),
        "masked_psnr": psnr_rgb(full, completed, mask),
        "masked_ssim": ssim_rgb(full, completed, mask),
        "masked_mae": mae_rgb(full, completed, mask),
        "visible_mae_pre_preserve": mae_rgb(full, raw_completed, visible),
        "visible_mae": mae_rgb(full, completed, visible),
        "seam_mae_pre_preserve": seam_mae(full, raw_completed, mask),
        "seam_mae": seam_mae(full, completed, mask),
    }

    silhouette_path = sample.get("gt_silhouette") or sample.get("silhouette")
    reference_silhouette = None
    if silhouette_path:
        reference_silhouette = _resize_mask(load_bool_mask(silhouette_path), mask.shape)
        object_eval_mask = mask & reference_silhouette
        if np.any(object_eval_mask):
            row.update(
                {
                    "object_masked_psnr": psnr_rgb(full, completed, object_eval_mask),
                    "object_masked_ssim": ssim_rgb(full, completed, object_eval_mask),
                    "object_masked_mae": mae_rgb(full, completed, object_eval_mask),
                    "object_seam_mae": seam_mae(full, completed, object_eval_mask),
                }
            )

    if not args.skip_depth and "gt_depth" in sample:
        depth_path = process_image_get_depth_data(
            completed_path,
            output_dir=str(output_dir),
            provider=args.depth_provider,
            model_name=args.depth_model,
            device=args.device,
        )
        row["depth_data"] = depth_path
        for preview_name in ("output_depth_preview.png", "output_image.webp"):
            preview_path = output_dir / preview_name
            if preview_path.exists():
                row["depth_preview"] = str(preview_path)
                break
        pred_depth = np.load(depth_path)
        gt_depth = np.load(sample["gt_depth"])
        if pred_depth.shape != gt_depth.shape:
            pred_depth = np.asarray(Image.fromarray(pred_depth).resize(gt_depth.shape[::-1], resample=Image.Resampling.BILINEAR))
        gt_depth = gt_depth.astype(np.float32)
        pred_depth = pred_depth.astype(np.float32)
        row.update(depth_metrics(gt_depth, pred_depth, mask, fit_mask=visible))
        aligned_depth, _, _ = align_depth(pred_depth, gt_depth, visible)
        row.update(surface_distance_metrics(gt_depth, aligned_depth, mask))
        if silhouette_path:
            reference_silhouette = _resize_mask(load_bool_mask(silhouette_path), gt_depth.shape)
            object_fit = visible & reference_silhouette
            object_eval = mask & reference_silhouette
            if np.count_nonzero(object_fit) >= 2 and np.count_nonzero(object_eval) > 0:
                object_depth = depth_metrics(
                    gt_depth,
                    pred_depth,
                    object_eval,
                    fit_mask=object_fit,
                )
                object_aligned_depth, _, _ = align_depth(pred_depth, gt_depth, object_fit)
                object_surface = surface_distance_metrics(gt_depth, object_aligned_depth, object_eval)
            else:
                object_depth = {
                    "depth_scale": math.nan,
                    "depth_shift": math.nan,
                    "depth_mae": math.nan,
                    "depth_rmse": math.nan,
                    "depth_corr": math.nan,
                }
                object_surface = {
                    "surface_chamfer_l1": math.nan,
                    "surface_chamfer_rmse": math.nan,
                    "surface_rmse": math.nan,
                    "surface_hausdorff95": math.nan,
                    "surface_point_count": 0,
                }
            row.update(_prefixed("object_", object_depth))
            row.update(_prefixed("object_", object_surface))

            fit_for_silhouette = object_fit if np.count_nonzero(object_fit) >= 2 else visible
            aligned_depth, _, _ = align_depth(pred_depth, gt_depth, fit_for_silhouette)
            row["silhouette_iou_masked"] = silhouette_iou_from_reference_mask(reference_silhouette, aligned_depth, mask)
        else:
            row["silhouette_iou_masked"] = silhouette_iou(gt_depth, aligned_depth, mask)

        if args.emit_stl:
            stl_path = output_dir / "output_model.stl"
            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=args.stl_target_dimension,
                z_scale=args.stl_z_scale,
                invert=not args.stl_no_invert,
                sigma=args.stl_sigma,
            )
            row.update(stl_diagnostics(stl_path))

    return row


def run_one(sample, method, output_dir, args):
    method_dir = output_dir / sample["id"] / method
    method_dir.mkdir(parents=True, exist_ok=True)
    if method == "masked":
        completed_path = run_masked(sample, method_dir)
    elif method == "mirror":
        completed_path = run_mirror(sample, method_dir)
    elif method == "mirror-seam-repair":
        completed_path = run_mirror_seam_repair(sample, method_dir)
    elif method == "biharmonic":
        completed_path = run_biharmonic(sample, method_dir)
    elif method in MODERN_INPAINT_MODELS:
        completed_path = run_modern(
            sample,
            method_dir,
            method,
            args.prompt,
            args.steps,
            args.seed,
            args.guidance,
            args.inpaint_max_dimension,
            getattr(args, "edit_mask_fill", "input"),
            args.model_name,
            args.lora_weights,
            args.lora_scale,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    raw_completed_path = resize_completed_to_sample(completed_path, sample, method_dir)
    completed_path = preserve_visible_pixels(raw_completed_path, sample, method_dir)
    return evaluate_sample(sample, method, raw_completed_path, completed_path, method_dir, args)


def summarize(rows, methods=None, attempted_n=None, failures=None):
    methods = list(methods or sorted({row["method"] for row in rows}))
    failures = failures or []
    summary = []
    metric_names = []
    for row in rows:
        for key, value in row.items():
            if key in METADATA_FIELDS or key in metric_names:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                metric_names.append(key)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_attempted_n = attempted_n if attempted_n is not None else len(method_rows)
        logged_failure_count = sum(1 for row in failures if row.get("method") == method)
        error_count = max(0, method_attempted_n - len(method_rows))
        item = {
            "method": method,
            "n": len(method_rows),
            "attempted_n": method_attempted_n,
            "error_count": error_count,
            "logged_failure_count": logged_failure_count,
            "success_rate": (len(method_rows) / method_attempted_n) if method_attempted_n else 0.0,
        }
        for metric in metric_names:
            values = []
            for row in method_rows:
                try:
                    values.append(float(row.get(metric, np.nan)))
                except (TypeError, ValueError):
                    values.append(np.nan)
            values = np.asarray(values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            item[f"{metric}_median"] = float(np.median(finite)) if len(finite) else np.nan
            item[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
        summary.append(item)
    return summary


def summarize_failures(failures, methods=None, attempted_n=None):
    methods = list(methods or sorted({row["method"] for row in failures}))
    return [
        {
            "method": method,
            "n": 0,
            "attempted_n": attempted_n if attempted_n is not None else sum(1 for row in failures if row["method"] == method),
            "error_count": attempted_n if attempted_n is not None else sum(1 for row in failures if row["method"] == method),
            "logged_failure_count": sum(1 for row in failures if row["method"] == method),
            "success_rate": 0.0,
            "last_error_type": next((row.get("error_type", "") for row in reversed(failures) if row["method"] == method), ""),
            "last_error": next((row.get("error", "") for row in reversed(failures) if row["method"] == method), ""),
        }
        for method in methods
    ]


def failure_row(sample, method, args, error_type: str, error: str, *, traceback_text: str = "", skipped: bool = False):
    return {
        **experiment_key(sample, method, args),
        "source": sample.get("source", ""),
        "asset_id": sample.get("asset_id", ""),
        "asset_path": sample.get("asset_path", ""),
        "asset_category": sample.get("asset_category", ""),
        "asset_source_split": sample.get("asset_source_split", ""),
        "asset_key": sample.get("asset_key", ""),
        "skipped": bool(skipped),
        "error_type": error_type,
        "error": error,
        "traceback": traceback_text,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate completion methods before depth/STL generation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/runs/default")
    parser.add_argument("--methods", default="masked,mirror,mirror-seam-repair,biharmonic")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based manifest row offset for held-out slices.")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--emit-stl", action="store_true", help="Convert each predicted depth map to an STL and report mesh diagnostics.")
    parser.add_argument("--stl-target-dimension", type=int, default=160)
    parser.add_argument("--stl-z-scale", type=float, default=50.0)
    parser.add_argument("--stl-sigma", type=float, default=4.0)
    parser.add_argument("--stl-no-invert", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inpaint-max-dimension", type=int, default=768)
    parser.add_argument(
        "--edit-mask-fill",
        choices=("input", "white", "gray", "checker", "mirror", "biharmonic"),
        default="input",
        help="For edit-only providers, optionally replace the masked region before generation.",
    )
    parser.add_argument("--model-name", default=None, help="Optional base model override for modern completion providers.")
    parser.add_argument("--lora-weights", default=None, help="Optional Diffusers LoRA adapter directory or safetensors file.")
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse rows from per_sample_metrics.jsonl and skip completed sample/method/config pairs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write failures.jsonl and keep going after a sample/method error.")
    parser.add_argument(
        "--max-method-failures",
        type=int,
        default=0,
        help="With --continue-on-error, skip remaining samples for a method/config after this many logged failures. 0 disables the cap.",
    )
    args = parser.parse_args()
    if args.emit_stl and args.skip_depth:
        raise ValueError("--emit-stl requires depth generation; remove --skip-depth")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_method_failures < 0:
        raise ValueError("--max-method-failures must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_sample_metrics.jsonl"
    failures_path = output_dir / "failures.jsonl"
    if not args.resume:
        for path in (jsonl_path, failures_path):
            if path.exists():
                path.unlink()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    samples = []
    with open(args.manifest, encoding="utf-8") as manifest_file:
        for manifest_index, line in enumerate(manifest_file):
            if manifest_index < args.start_index:
                continue
            samples.append(json.loads(line))
            if len(samples) >= args.limit:
                break
    write_split_audit(output_dir, samples, args)

    rows = load_jsonl(jsonl_path) if args.resume else []
    failures = load_jsonl(failures_path) if args.resume else []
    completed_keys = {key_tuple(row) for row in rows}
    failure_keys = {key_tuple(row) for row in failures}
    method_failure_counts = {
        method: sum(1 for row in failures if row.get("method") == method and not row.get("skipped")) for method in methods
    }
    for sample in samples:
        for method in methods:
            key = key_tuple(experiment_key(sample, method, args))
            if args.resume and key in completed_keys:
                continue
            if args.continue_on_error and args.max_method_failures and method_failure_counts.get(method, 0) >= args.max_method_failures:
                skip_failure = failure_row(
                    sample,
                    method,
                    args,
                    "MethodFailureLimitExceeded",
                    (
                        f"Skipped after {method_failure_counts.get(method, 0)} non-skipped failures for method "
                        f"{method}; --max-method-failures={args.max_method_failures}."
                    ),
                    skipped=True,
                )
                if key_tuple(skip_failure) not in failure_keys:
                    append_jsonl(failures_path, skip_failure)
                    failures.append(skip_failure)
                    failure_keys.add(key_tuple(skip_failure))
                    print(json.dumps(skip_failure))
                continue
            try:
                row = run_one(sample, method, output_dir, args)
                rows.append(row)
                completed_keys.add(key_tuple(row))
                append_jsonl(jsonl_path, row)
                print(json.dumps(row))
            except Exception as exc:
                failure = failure_row(sample, method, args, type(exc).__name__, str(exc), traceback_text=traceback.format_exc())
                append_jsonl(failures_path, failure)
                failures.append(failure)
                failure_keys.add(key_tuple(failure))
                method_failure_counts[method] = method_failure_counts.get(method, 0) + 1
                print(json.dumps(failure))
                if not args.continue_on_error:
                    raise

    per_sample_path = output_dir / "per_sample_metrics.csv"
    if rows:
        with per_sample_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = []
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        failures = load_jsonl(failures_path)
        if not args.continue_on_error or not failures:
            raise ValueError("No benchmark rows produced")
        with per_sample_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "method"])
            writer.writeheader()

    failures = load_jsonl(failures_path)
    summary = (
        summarize(rows, methods=methods, attempted_n=len(samples), failures=failures)
        if rows
        else summarize_failures(failures, methods=methods, attempted_n=len(samples))
    )
    summary_path = output_dir / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = []
        for row in summary:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(per_sample_path)
    print(summary_path)


if __name__ == "__main__":
    main()
