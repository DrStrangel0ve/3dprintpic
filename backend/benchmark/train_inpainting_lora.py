from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_metadata(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def resolve_record_path(value: str, metadata_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = metadata_dir / path
    if candidate.exists():
        return candidate
    return path


def resize_image(image: Image.Image, resolution: int, resample) -> Image.Image:
    if image.size == (resolution, resolution):
        return image
    return image.resize((resolution, resolution), resample=resample)


def image_to_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = resize_image(image, resolution, Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = resize_image(image, resolution, Image.Resampling.NEAREST)
    array = (np.asarray(image, dtype=np.float32) >= 127).astype(np.float32)
    return torch.from_numpy(array)[None, :, :]


class InpaintPairDataset(Dataset):
    def __init__(
        self,
        metadata_path: Path,
        rows: list[dict],
        resolution: int,
        tokenizer=None,
        max_length: int | None = None,
        require_object_mask: bool = False,
        sample_weight_field: str = "sample_weight",
    ):
        self.metadata_dir = metadata_path.parent
        self.rows = rows
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.require_object_mask = require_object_mask
        self.sample_weight_field = sample_weight_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = resolve_record_path(row["image"], self.metadata_dir)
        mask_path = resolve_record_path(row["mask"], self.metadata_dir)
        target_path = resolve_record_path(row["target"], self.metadata_dir)
        object_mask_value = row.get("object_mask") or row.get("gt_silhouette") or row.get("silhouette")
        if self.require_object_mask and not object_mask_value:
            raise ValueError(f"row {row.get('id', index)} is missing object_mask metadata")
        if object_mask_value:
            object_mask_values = mask_to_tensor(resolve_record_path(object_mask_value, self.metadata_dir), self.resolution)
            has_object_mask = True
        else:
            object_mask_values = torch.zeros((1, self.resolution, self.resolution), dtype=torch.float32)
            has_object_mask = False
        try:
            sample_weight = float(row.get(self.sample_weight_field, 1.0) or 1.0)
        except (TypeError, ValueError):
            raise ValueError(f"row {row.get('id', index)} has non-numeric {self.sample_weight_field!r}")
        if not math.isfinite(sample_weight) or sample_weight < 0:
            raise ValueError(f"row {row.get('id', index)} has invalid {self.sample_weight_field!r}: {sample_weight}")

        item = {
            "id": row.get("id", str(index)),
            "pixel_values": image_to_tensor(target_path, self.resolution),
            "masked_pixel_values": image_to_tensor(image_path, self.resolution),
            "mask_values": mask_to_tensor(mask_path, self.resolution),
            "object_mask_values": object_mask_values,
            "has_object_mask": has_object_mask,
            "sample_weight": sample_weight,
            "prompt": row.get("prompt", ""),
        }
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                item["prompt"],
                max_length=self.max_length or self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = tokens.input_ids[0]
        return item


def collate(batch: list[dict]) -> dict:
    output = {
        "ids": [item["id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "masked_pixel_values": torch.stack([item["masked_pixel_values"] for item in batch]),
        "mask_values": torch.stack([item["mask_values"] for item in batch]),
        "object_mask_values": torch.stack([item["object_mask_values"] for item in batch]),
        "has_object_mask": torch.tensor([item["has_object_mask"] for item in batch], dtype=torch.bool),
        "sample_weights": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        "prompts": [item["prompt"] for item in batch],
    }
    if "input_ids" in batch[0]:
        output["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pair_export_report_path(metadata_path: Path) -> Path | None:
    return first_existing(
        [
            metadata_path.with_name("pair_export_report.json"),
            metadata_path.parent / "export_report.json",
            metadata_path.parent / "training_pairs_report.json",
        ]
    )


def prompt_family_from_rows(rows: list[dict]) -> str:
    templates = {str(row.get("prompt_template") or "") for row in rows if row.get("prompt_template")}
    prompts = {str(row.get("prompt") or "") for row in rows if row.get("prompt")}
    categories = {str(row.get("category") or row.get("asset_category") or "") for row in rows}
    if not prompts:
        return "empty"
    if len(prompts) == 1 and (not templates or prompts == templates):
        return "literal"
    if len(prompts) <= 1:
        return "single"
    if len(categories) > 1:
        return "category-templated"
    return "templated"


def args_dict(args) -> dict:
    return {key: value for key, value in sorted(vars(args).items())}


def format_recipe_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p")


def loss_recipe_label(mask_weight: float, seam_weight: float, object_weight: float) -> str:
    parts = [
        f"mask{format_recipe_number(mask_weight)}",
        f"seam{format_recipe_number(seam_weight)}",
        f"object{format_recipe_number(object_weight)}",
    ]
    return "_".join(parts)


def component_load_kwargs(args) -> dict:
    kwargs = {"use_safetensors": True}
    if args.variant:
        kwargs["variant"] = args.variant
    if args.mixed_precision == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif args.mixed_precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    return kwargs


def seam_mask_from_latent_mask(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def weighted_mse_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_weight: float,
    seam_weight: float,
    object_mask: torch.Tensor | None = None,
    object_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
):
    per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    if mask_weight <= 0 and seam_weight <= 0 and object_weight <= 0 and sample_weight is None:
        return per_element.mean(), per_element.mean().detach()

    weights = torch.ones_like(mask, dtype=per_element.dtype, device=per_element.device)
    if mask_weight > 0:
        weights = weights + float(mask_weight) * mask.to(dtype=per_element.dtype)
    if seam_weight > 0:
        seam = seam_mask_from_latent_mask(mask).to(dtype=per_element.dtype)
        weights = weights + float(seam_weight) * seam
    if object_weight > 0:
        if object_mask is None:
            raise ValueError("object_weight requires an object_mask tensor")
        weights = weights + float(object_weight) * object_mask.to(dtype=per_element.dtype)
    if weights.shape[1] == 1 and per_element.shape[1] != 1:
        weights = weights.expand(-1, per_element.shape[1], -1, -1)
    weighted = per_element * weights
    if sample_weight is not None:
        sample_weights = sample_weight.to(dtype=per_element.dtype, device=per_element.device).flatten()
        sample_weights = torch.clamp(sample_weights, min=0)
        if torch.all(torch.abs(sample_weights - 1.0) <= 1e-12):
            return weighted.sum() / weights.sum().clamp_min(1.0), per_element.mean().detach()
        weighted_per_sample = weighted.flatten(1).sum(dim=1) / weights.flatten(1).sum(dim=1).clamp_min(1.0)
        loss = (weighted_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        return loss, per_element.mean().detach()
    return weighted.sum() / weights.sum().clamp_min(1.0), per_element.mean().detach()


def row_has_object_mask(row: dict) -> bool:
    return bool(row.get("object_mask") or row.get("gt_silhouette") or row.get("silhouette"))


def row_category(row: dict) -> str:
    return str(row.get("asset_category") or row.get("category") or row.get("source") or "unknown")


def row_asset_key(row: dict) -> str:
    return str(row.get("asset_key") or row.get("asset_path") or row.get("asset_id") or row.get("id") or "")


def sample_weight_value(row: dict, field: str = "sample_weight") -> float:
    try:
        value = float(row.get(field, 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return value if math.isfinite(value) and value >= 0 else 1.0


def sample_weight_summary(rows: list[dict], field: str = "sample_weight") -> dict:
    values = np.asarray([sample_weight_value(row, field=field) for row in rows], dtype=np.float64)
    non_default = int(np.count_nonzero(np.abs(values - 1.0) > 1e-12))
    return {
        "sample_weight_field": field,
        "sample_weight_rows": len(values),
        "sample_weight_non_default_rows": non_default,
        "sample_weight_min": float(np.min(values)) if len(values) else 1.0,
        "sample_weight_mean": float(np.mean(values)) if len(values) else 1.0,
        "sample_weight_max": float(np.max(values)) if len(values) else 1.0,
    }


def args_sample_weight_field(args) -> str:
    return getattr(args, "sample_weight_field", "sample_weight")


def source_split_from_key(value: str) -> str:
    parts = str(value or "").replace("\\", "/").split("/")
    for part in parts:
        split = part.lower()
        if split == "validation":
            return "val"
        if split in {"train", "test", "val"}:
            return split
    return "unknown"


def row_source_split(row: dict) -> str:
    explicit = str(row.get("asset_source_split") or "")
    if explicit:
        return explicit
    return source_split_from_key(row_asset_key(row))


def dataset_identity_summary(rows: list[dict]) -> dict:
    category_counts = {}
    source_split_counts = {}
    asset_keys = []
    for row in rows:
        category = row_category(row)
        category_counts[category] = category_counts.get(category, 0) + 1
        source_split = row_source_split(row)
        source_split_counts[source_split] = source_split_counts.get(source_split, 0) + 1
        key = row_asset_key(row)
        if key:
            asset_keys.append(key)
    unique_asset_keys = sorted(set(asset_keys))
    return {
        "category_counts": dict(sorted(category_counts.items())),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "asset_count": len(unique_asset_keys),
        "asset_keys": unique_asset_keys,
    }


def dry_run(args, rows: list[dict]) -> None:
    metadata_path = Path(args.metadata)
    export_report_path = pair_export_report_path(metadata_path)
    export_report = optional_json(export_report_path) if export_report_path else {}
    dataset = InpaintPairDataset(
        metadata_path,
        rows,
        args.resolution,
        require_object_mask=args.object_loss_weight > 0,
        sample_weight_field=args_sample_weight_field(args),
    )
    sample = dataset[0]
    batch = collate([dataset[index] for index in range(min(len(dataset), args.train_batch_size))])
    object_masked_region = sample["object_mask_values"] * sample["mask_values"]
    mask_fractions = []
    object_mask_fractions = []
    object_inpaint_fractions = []
    for index in range(len(dataset)):
        item = dataset[index]
        mask_fractions.append(float(item["mask_values"].mean().item()))
        object_mask_fractions.append(float(item["object_mask_values"].mean().item()))
        object_inpaint_fractions.append(float((item["object_mask_values"] * item["mask_values"]).mean().item()))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": args.metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "pair_export_report": str(export_report_path) if export_report_path else "",
        "pair_export_sha256": export_report.get("metadata_sha256", ""),
        "command": sys.argv,
        "args": args_dict(args),
        "rows": len(rows),
        "object_mask_rows": sum(1 for row in rows if row_has_object_mask(row)),
        "prompt_family": prompt_family_from_rows(rows),
        "prompt_template": rows[0].get("prompt_template", "") if rows else "",
        "resolution": args.resolution,
        "first_id": sample["id"],
        "first_prompt": sample["prompt"],
        "pixel_shape": list(sample["pixel_values"].shape),
        "masked_pixel_shape": list(sample["masked_pixel_values"].shape),
        "mask_shape": list(sample["mask_values"].shape),
        "mask_fraction": float(sample["mask_values"].mean().item()),
        "has_object_mask": bool(sample["has_object_mask"]),
        "object_mask_shape": list(sample["object_mask_values"].shape),
        "object_mask_fraction": float(sample["object_mask_values"].mean().item()),
        "object_inpaint_fraction": float(object_masked_region.mean().item()),
        "mask_fraction_mean": float(np.mean(mask_fractions)),
        "object_mask_fraction_mean": float(np.mean(object_mask_fractions)),
        "object_inpaint_fraction_mean": float(np.mean(object_inpaint_fractions)),
        "object_inpaint_fraction_min": float(np.min(object_inpaint_fractions)),
        "object_inpaint_nonzero_rows": sum(fraction > 0 for fraction in object_inpaint_fractions),
        "mask_loss_weight": args.mask_loss_weight,
        "seam_loss_weight": args.seam_loss_weight,
        "object_loss_weight": args.object_loss_weight,
        "loss_recipe": loss_recipe_label(args.mask_loss_weight, args.seam_loss_weight, args.object_loss_weight),
        "batch_pixel_shape": list(batch["pixel_values"].shape),
        "batch_sample_weight_mean": float(batch["sample_weights"].mean().item()),
        **sample_weight_summary(rows, field=args_sample_weight_field(args)),
        **dataset_identity_summary(rows),
    }
    write_json(Path(args.output_dir) / "training_plan.json", report)
    print(json.dumps(report, indent=2))


def train(args, rows: list[dict]) -> None:
    from accelerate import Accelerator
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata)
    export_report_path = pair_export_report_path(metadata_path)
    export_report = optional_json(export_report_path) if export_report_path else {}
    set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        project_dir=str(output_dir),
    )

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    load_kwargs = component_load_kwargs(args)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", **load_kwargs)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", **load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", **load_kwargs)
    if int(unet.config.in_channels) != 9:
        raise ValueError(f"Expected a 9-channel inpainting UNet, got in_channels={unet.config.in_channels}")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    unet.add_adapter(lora_config)
    cast_training_params(unet, dtype=torch.float32)

    dataset = InpaintPairDataset(
        Path(args.metadata),
        rows,
        args.resolution,
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        require_object_mask=args.object_loss_weight > 0,
        sample_weight_field=args_sample_weight_field(args),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in unet.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    log_path = output_dir / "train_metrics.csv"
    best_loss = None
    best_unweighted_loss = None
    last_metrics = {}
    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["step", "loss", "unweighted_loss", "lr"])
        writer.writeheader()

        global_step = 0
        while global_step < args.max_train_steps:
            for batch in dataloader:
                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    masked_pixel_values = batch["masked_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    mask_values = batch["mask_values"].to(accelerator.device, dtype=weight_dtype)
                    object_mask_values = batch["object_mask_values"].to(accelerator.device, dtype=weight_dtype)
                    sample_weights = batch["sample_weights"].to(accelerator.device, dtype=torch.float32)
                    input_ids = batch["input_ids"].to(accelerator.device)

                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor
                    masks = F.interpolate(mask_values, size=latents.shape[-2:], mode="nearest")
                    object_masks = F.interpolate(object_mask_values, size=latents.shape[-2:], mode="nearest") * masks

                    noise = torch.randn_like(latents)
                    batch_size = latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (batch_size,),
                        device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    latent_model_input = torch.cat([noisy_latents, masks, masked_latents], dim=1)

                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]
                    model_pred = unet(latent_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                    loss, unweighted_loss = weighted_mse_loss(
                        model_pred,
                        target,
                        masks,
                        mask_weight=args.mask_loss_weight,
                        seam_weight=args.seam_loss_weight,
                        object_mask=object_masks,
                        object_weight=args.object_loss_weight,
                        sample_weight=sample_weights,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    if accelerator.is_main_process:
                        current_loss = float(loss.detach().item())
                        current_unweighted_loss = float(unweighted_loss.item())
                        best_loss = current_loss if best_loss is None else min(best_loss, current_loss)
                        best_unweighted_loss = (
                            current_unweighted_loss
                            if best_unweighted_loss is None
                            else min(best_unweighted_loss, current_unweighted_loss)
                        )
                        last_metrics = {
                            "step": global_step,
                            "loss": current_loss,
                            "unweighted_loss": current_unweighted_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        writer.writerow(
                            last_metrics
                        )
                        csv_file.flush()
                    if global_step >= args.max_train_steps:
                        break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(output_dir),
            unet_lora_layers=state_dict,
            safe_serialization=True,
        )
        adapter_file = first_existing(
            [
                output_dir / "pytorch_lora_weights.safetensors",
                output_dir / "adapter_model.safetensors",
                output_dir / "pytorch_lora_weights.bin",
            ]
        )
        write_json(
            output_dir / "training_report.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "command": sys.argv,
                "args": args_dict(args),
                "metadata": args.metadata,
                "metadata_sha256": file_sha256(metadata_path),
                "pair_export_report": str(export_report_path) if export_report_path else "",
                "pair_export_sha256": export_report.get("metadata_sha256", ""),
                "rows": len(rows),
                "base_model": args.base_model,
                "variant": args.variant,
                "resolution": args.resolution,
                "max_train_steps": args.max_train_steps,
                "rank": args.rank,
                "lora_alpha": args.lora_alpha,
                "mask_loss_weight": args.mask_loss_weight,
                "seam_loss_weight": args.seam_loss_weight,
                "object_loss_weight": args.object_loss_weight,
                "loss_recipe": loss_recipe_label(args.mask_loss_weight, args.seam_loss_weight, args.object_loss_weight),
                "prompt_family": prompt_family_from_rows(rows),
                "prompt_template": rows[0].get("prompt_template", "") if rows else "",
                "train_metrics": str(log_path),
                "final_step": last_metrics.get("step", 0),
                "final_loss": last_metrics.get("loss"),
                "final_unweighted_loss": last_metrics.get("unweighted_loss"),
                "best_loss": best_loss,
                "best_unweighted_loss": best_unweighted_loss,
                "object_mask_rows": sum(1 for row in rows if row_has_object_mask(row)),
                **sample_weight_summary(rows, field=args_sample_weight_field(args)),
                **dataset_identity_summary(rows),
                "target_modules": target_modules,
                "adapter_path": str(output_dir),
                "adapter_file": str(adapter_file) if adapter_file else "",
                "adapter_sha256": file_sha256(adapter_file) if adapter_file else "",
                "benchmark_hint": (
                    "Use --model-name "
                    f"{args.base_model} --lora-weights {output_dir} "
                    "when running backend.benchmark.run_completion_benchmark or optimize_completion."
                ),
            },
        )
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a small LoRA adapter for SD1.5-style inpainting from exported benchmark pairs."
    )
    parser.add_argument("--metadata", required=True, help="metadata.jsonl from export_training_pairs.")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/lora/default")
    parser.add_argument("--base-model", default="Lykon/dreamshaper-8-inpainting")
    parser.add_argument("--variant", default="fp16", help="Model weight variant to load. Use an empty string for no variant.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate the pair dataset without loading model weights.")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-loss-weight", type=float, default=0.0, help="Extra latent MSE weight inside the inpaint mask.")
    parser.add_argument("--seam-loss-weight", type=float, default=0.0, help="Extra latent MSE weight on mask boundary latents.")
    parser.add_argument(
        "--object-loss-weight",
        type=float,
        default=0.0,
        help="Extra latent MSE weight for pixels that are both inside the inpaint mask and the object silhouette.",
    )
    parser.add_argument(
        "--sample-weight-field",
        default="sample_weight",
        help="Metadata JSONL field used for per-example loss weights. Missing rows default to 1.0.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--target-modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")
    if args.max_train_steps < 1 and not args.dry_run:
        raise ValueError("--max-train-steps must be positive")
    if args.mask_loss_weight < 0 or args.seam_loss_weight < 0 or args.object_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    return args


def main():
    args = parse_args()
    rows = read_metadata(Path(args.metadata), limit=args.limit)
    if args.object_loss_weight > 0:
        missing = [row.get("id", str(index)) for index, row in enumerate(rows) if not row_has_object_mask(row)]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"--object-loss-weight requires object masks for every row; missing: {preview}")
    if args.dry_run:
        dry_run(args, rows)
    else:
        train(args, rows)


if __name__ == "__main__":
    main()
