import os
import shutil
import time
import cv2
import numpy as np
from stl import mesh
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    grey_closing,
    label,
    laplace,
    maximum_filter,
    minimum_filter,
    zoom,
)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
import argparse

try:
    from .face_relief_geometry import align_face_to_scene_gradient_domain
    from .da3_depth_provider import (
        DA3_LARGE_MODEL_ID,
        infer_da3_depth,
        is_da3_model,
        release_da3_models,
    )
except ImportError:  # pragma: no cover - supports running from backend/
    if __package__:
        raise
    from face_relief_geometry import align_face_to_scene_gradient_domain
    from da3_depth_provider import (
        DA3_LARGE_MODEL_ID,
        infer_da3_depth,
        is_da3_model,
        release_da3_models,
    )

_DEPTH_PIPELINE_CACHE = {}
_INPAINT_PIPELINE_CACHE = {}
DEPTHPRO_MODEL_ID = "apple/DepthPro-hf"
DEFAULT_DEPTH_FALLBACK_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
RELIEF_VALUE_TRANSFORM_LINEAR = "linear"
RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH = "inverse-depth"
RELIEF_VALUE_TRANSFORMS = {
    RELIEF_VALUE_TRANSFORM_LINEAR,
    RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
}
HIGH_RELIEF_FACE_SCREENING_WEIGHT = 0.25
HIGH_RELIEF_FACE_CARDINAL_EDGE_RETRY_WEIGHT = 0.10
HIGH_RELIEF_FACE_DETAIL_RETRY_WEIGHT = 8.0
HIGH_RELIEF_FACE_MIN_DETAIL_CORRELATION = 0.85
HIGH_RELIEF_SELECTION_SCREENING_WEIGHT = 2.0
HIGH_RELIEF_SELECTION_DETAIL_GRADIENT_RETENTION = 0.9
DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO = 0.65
NORMALIZATION_REFERENCE_BOUNDARY_PX = 1.0
NORMALIZATION_REFERENCE_TAPER_PX = 4.0
METRIC_FAR_HIGH_DEPTH_MODELS = frozenset(
    {
        DEPTHPRO_MODEL_ID,
        DA3_LARGE_MODEL_ID,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    }
)


def relief_value_transform_for_model(model_name):
    if model_name in METRIC_FAR_HIGH_DEPTH_MODELS:
        return RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH
    return RELIEF_VALUE_TRANSFORM_LINEAR


def _load_depth_input_image(input_image_path):
    from PIL import Image, ImageOps

    with Image.open(input_image_path) as source_image:
        return ImageOps.exif_transpose(source_image).convert("RGB")

MODERN_INPAINT_MODELS = {
    "sdxl-inpaint": {
        "label": "SDXL Inpaint",
        "model": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "pipeline_class": "StableDiffusionXLInpaintPipeline",
    },
    "dreamshaper-inpaint": {
        "label": "DreamShaper 8 Inpaint",
        "model": "Lykon/dreamshaper-8-inpainting",
        "pipeline_class": "StableDiffusionInpaintPipeline",
    },
    "amused-inpaint": {
        "label": "AMUSED 512 Inpaint",
        "model": "amused/amused-512",
        "pipeline_class": "AmusedInpaintPipeline",
    },
    "flux-fill": {
        "label": "FLUX.1 Fill",
        "model": "black-forest-labs/FLUX.1-Fill-dev",
        "pipeline_class": "FluxFillPipeline",
    },
    "flux2-klein-inpaint": {
        "label": "FLUX.2 Klein 4B Inpaint",
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline_class": "Flux2KleinInpaintPipeline",
    },
    "qwen-image-inpaint": {
        "label": "Qwen Image Inpaint",
        "model": "Qwen/Qwen-Image-Edit",
        "pipeline_class": "QwenImageInpaintPipeline",
    },
    "qwen-image-edit": {
        "label": "Qwen Image Edit",
        "model": "Qwen/Qwen-Image-Edit",
        "pipeline_class": "QwenImageEditPipeline",
    },
}


def complete_image(
    input_image_path,
    output_dir="./output",
    mode="none",
    provider="mirror",
    prompt=None,
    model_name=None,
    device="auto",
    num_inference_steps=24,
    guidance_scale=None,
    seed=None,
    inpaint_max_dimension=768,
    lora_weights=None,
    lora_scale=1.0,
    edit_mask_fill=None,
):
    if mode in (None, "", "none"):
        return input_image_path, None

    if provider in (None, "", "mirror"):
        return complete_image_by_symmetry(input_image_path, output_dir=output_dir, mode=mode)

    if provider == "mirror-seam-repair":
        return complete_image_by_symmetry_with_seam_repair(input_image_path, output_dir=output_dir, mode=mode)

    if provider in MODERN_INPAINT_MODELS:
        return complete_image_with_modern_inpaint(
            input_image_path,
            output_dir=output_dir,
            mode=mode,
            provider=provider,
            prompt=prompt,
            model_name=model_name,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            inpaint_max_dimension=inpaint_max_dimension,
            lora_weights=lora_weights,
            lora_scale=lora_scale,
            edit_mask_fill=edit_mask_fill,
        )

    raise ValueError(f"Unsupported completion provider: {provider}")


def complete_image_by_symmetry(input_image_path, output_dir="./output", mode="none", feather_px=24):
    if mode in (None, "", "none"):
        return input_image_path, None

    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(input_image_path).convert("RGB")
    width, height = image.size
    if width < 4:
        raise ValueError("Image is too narrow for symmetry completion")

    if mode == "mirror-auto":
        mode = _choose_symmetry_direction(image)

    if mode not in ("mirror-left-to-right", "mirror-right-to-left"):
        raise ValueError(f"Unsupported completion mode: {mode}")

    midpoint = width // 2
    output = image.copy()

    if mode == "mirror-left-to-right":
        source_box = (0, 0, midpoint, height)
        source_half = image.crop(source_box)
        mirrored_half = source_half.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target_width = width - midpoint
        replacement = mirrored_half.resize((target_width, height))
        output.paste(replacement, (midpoint, 0))
        _blend_vertical_seam(output, midpoint, feather_px, mode)
    else:
        source_box = (midpoint, 0, width, height)
        source_half = image.crop(source_box)
        mirrored_half = source_half.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target_width = midpoint
        replacement = mirrored_half.resize((target_width, height))
        output.paste(replacement, (0, 0))
        _blend_vertical_seam(output, midpoint, feather_px, mode)

    completed_path = os.path.join(output_dir, "completed_input.png")
    output.save(completed_path)
    return completed_path, mode


def complete_image_by_symmetry_with_seam_repair(input_image_path, output_dir="./output", mode="none", feather_px=24, repair_px=2):
    if mode in (None, "", "none"):
        return input_image_path, None

    from PIL import Image
    from skimage.restoration import inpaint_biharmonic

    os.makedirs(output_dir, exist_ok=True)
    mirrored_path, applied_mode = complete_image_by_symmetry(
        input_image_path,
        output_dir=output_dir,
        mode=mode,
        feather_px=feather_px,
    )
    if applied_mode is None:
        return mirrored_path, applied_mode

    output = Image.open(mirrored_path).convert("RGB")
    width, height = output.size
    midpoint = width // 2
    if applied_mode == "mirror-left-to-right":
        left = midpoint
        right = min(width, midpoint + repair_px)
    elif applied_mode == "mirror-right-to-left":
        left = max(0, midpoint - repair_px)
        right = midpoint
    else:
        raise ValueError(f"Unsupported completion mode: {applied_mode}")
    if right <= left:
        return mirrored_path, applied_mode

    data = np.asarray(output, dtype=np.float32) / 255.0
    repair_mask = np.zeros((height, width), dtype=bool)
    repair_mask[:, left:right] = True
    repaired = inpaint_biharmonic(data, repair_mask, channel_axis=-1)
    repaired_image = Image.fromarray(np.clip(repaired * 255, 0, 255).astype(np.uint8))
    completed_path = os.path.join(output_dir, "completed_input.png")
    repaired_image.save(completed_path)
    return completed_path, f"{applied_mode}:seam-repair"


def complete_image_with_modern_inpaint(
    input_image_path,
    output_dir="./output",
    mode="mirror-auto",
    provider="flux-fill",
    prompt=None,
    model_name=None,
    device="auto",
    num_inference_steps=24,
    guidance_scale=None,
    seed=None,
    inpaint_max_dimension=768,
    lora_weights=None,
    lora_scale=1.0,
    edit_mask_fill=None,
):
    import torch
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(input_image_path).convert("RGB")
    min_dimension = 512 if provider == "amused-inpaint" else None
    image = _resize_for_inpaint(image, max_dimension=inpaint_max_dimension, min_dimension=min_dimension)
    mask, applied_mode = create_half_completion_mask(image, mode=mode)
    hard_mask, _ = create_half_completion_mask(image, mode=applied_mode, blur=False)

    mask_path = os.path.join(output_dir, "completion_mask.png")
    masked_input_path = os.path.join(output_dir, "masked_input.png")
    edit_input_path = os.path.join(output_dir, "edit_input.png")
    completed_path = os.path.join(output_dir, "completed_input.png")
    mask.save(mask_path)
    _save_masked_preview(image, hard_mask, masked_input_path)
    edit_image = _masked_edit_image(image, hard_mask, edit_mask_fill)
    if edit_image is not image:
        edit_image.save(edit_input_path)

    prompt = prompt or (
        "Complete the missing half of the subject naturally. Keep the original identity, "
        "shape, lighting, perspective, and background consistent. Do not leave the masked area empty."
    )
    negative_prompt = "blank, empty, white void, missing object, blurry, distorted, inconsistent lighting, extra limbs, duplicate object"
    model_name = model_name or MODERN_INPAINT_MODELS[provider]["model"]
    pipeline_device = _resolve_torch_device(device)
    generator = None
    if seed is not None:
        generator_device = "cuda" if pipeline_device == "cuda" and torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    pipe = _load_inpaint_pipeline(provider, model_name, pipeline_device, lora_weights=lora_weights, lora_scale=lora_scale)

    if provider == "flux-fill":
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else 30.0,
            max_sequence_length=512,
            generator=generator,
        )
    elif provider == "sdxl-inpaint":
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else 7.5,
            strength=0.99,
            generator=generator,
        )
    elif provider == "dreamshaper-inpaint":
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else 7.5,
            strength=0.99,
            generator=generator,
        )
    elif provider == "amused-inpaint":
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else 10.0,
            strength=1.0,
            generator=generator,
        )
    elif provider == "qwen-image-inpaint":
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            height=image.height,
            width=image.width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale if guidance_scale is not None else 4.0,
            generator=generator,
        )
    elif provider == "qwen-image-edit":
        edit_prompt = _qwen_edit_prompt(prompt, edit_mask_fill)
        result = pipe(
            prompt=edit_prompt,
            negative_prompt=negative_prompt,
            image=edit_image,
            height=edit_image.height,
            width=edit_image.width,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=guidance_scale if guidance_scale is not None else 4.0,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported modern inpaint provider: {provider}")

    raw_completed_path = os.path.join(output_dir, "raw_completed_input.png")
    generated = result.images[0].convert("RGB")
    if generated.size != image.size:
        generated = generated.resize(image.size)
    generated.save(raw_completed_path)
    preserved = Image.composite(generated, image, hard_mask)
    preserved.save(completed_path)
    return completed_path, f"{provider}:{applied_mode}"


def complete_selection_context_with_modern_inpaint(
    image,
    keep_mask,
    *,
    provider="flux2-klein-inpaint",
    model_name=None,
    prompt=None,
    device="auto",
    num_inference_steps=4,
    guidance_scale=1.0,
    seed=0,
    inpaint_max_dimension=512,
    mask_overlap_px=4,
    boundary_feather_px=0,
):
    """Generate removed selection context without conditioning on removed pixels."""
    import torch
    from PIL import Image

    if provider != "flux2-klein-inpaint":
        raise ValueError(f"Unsupported selection-context provider: {provider}")

    source = image.convert("RGB")
    hard_keep_mask = keep_mask.convert("L").point(lambda value: 255 if value > 127 else 0)
    working_image = _resize_for_inpaint(
        source,
        max_dimension=inpaint_max_dimension,
        multiple=16,
    )
    working_keep_mask = hard_keep_mask.resize(
        working_image.size,
        resample=Image.Resampling.NEAREST,
    )
    working_keep = np.asarray(working_keep_mask, dtype=np.uint8) > 0

    row_weight = np.linspace(0.0, 1.0, working_image.height, dtype=np.float32)[:, None, None]
    canvas_top = np.asarray((238, 240, 242), dtype=np.float32)[None, None, :]
    canvas_bottom = np.asarray((214, 218, 222), dtype=np.float32)[None, None, :]
    canvas_values = canvas_top * (1.0 - row_weight) + canvas_bottom * row_weight
    canvas_values = np.broadcast_to(
        canvas_values,
        (working_image.height, working_image.width, 3),
    ).copy()
    condition_values = canvas_values.astype(np.uint8)
    working_source_values = np.asarray(working_image, dtype=np.uint8)
    condition_values[working_keep] = working_source_values[working_keep]
    condition_image = Image.fromarray(condition_values, mode="RGB")
    overlap_kernel_size = max(1, int(mask_overlap_px) * 2 + 1)
    generation_keep = cv2.erode(
        working_keep.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (overlap_kernel_size, overlap_kernel_size),
        ),
        iterations=1,
    ) > 0
    if not np.any(generation_keep):
        generation_keep = working_keep
    fill_mask = Image.fromarray((~generation_keep).astype(np.uint8) * 255, mode="L")

    model_name = model_name or MODERN_INPAINT_MODELS[provider]["model"]
    pipeline_device = _resolve_torch_device(device)
    generator_device = (
        "cuda"
        if pipeline_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    started = time.perf_counter()
    if pipeline_device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pipe = _load_inpaint_pipeline(provider, model_name, pipeline_device)
    prompt = prompt or (
        "Reconstruct a clean, coherent background around the preserved foreground subjects. "
        "Continue the scene with natural perspective, lighting, and large readable surfaces. "
        "Remove all masked people, objects, eyewear, hands, text, and occluders. "
        "Do not duplicate or alter the preserved subjects and do not place objects over them."
    )
    result = pipe(
        prompt=prompt,
        image=condition_image,
        mask_image=fill_mask,
        height=working_image.height,
        width=working_image.width,
        strength=1.0,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    generated = result.images[0].convert("RGB")
    if generated.size != source.size:
        generated = generated.resize(source.size, Image.Resampling.LANCZOS)
    effective_boundary_feather_px = max(0, int(round(float(boundary_feather_px))))
    preserved = _composite_generated_context_outward(
        source,
        generated,
        hard_keep_mask,
        feather_px=effective_boundary_feather_px,
    )
    peak_vram_gib = None
    if pipeline_device == "cuda" and torch.cuda.is_available():
        peak_vram_gib = float(torch.cuda.max_memory_allocated() / (1024**3))
    return preserved, {
        "provider": provider,
        "model": model_name,
        "method": "mask-native-generative-inpaint",
        "source_free_removed_context": True,
        "removed_source_pixels_conditioned": False,
        "working_dimensions": {
            "width": int(working_image.width),
            "height": int(working_image.height),
        },
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
        "mask_overlap_px": int(mask_overlap_px),
        "boundary_feather_px": int(boundary_feather_px),
        "effective_output_boundary_feather_px": int(effective_boundary_feather_px),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_vram_gib": peak_vram_gib,
        "selected_pixels_exact": bool(
            np.array_equal(
                np.asarray(preserved)[np.asarray(hard_keep_mask) > 0],
                np.asarray(source)[np.asarray(hard_keep_mask) > 0],
            )
        ),
    }


def _composite_generated_context_outward(source, generated, keep_mask, *, feather_px):
    from PIL import Image

    source_values = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_values = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    keep = np.asarray(keep_mask.convert("L"), dtype=np.uint8) > 0
    composed = generated_values.astype(np.float32)
    if feather_px > 0 and np.any(keep) and np.any(~keep):
        distance, nearest = distance_transform_edt(~keep, return_indices=True)
        ring = (~keep) & (distance <= float(feather_px))
        nearest_values = source_values[nearest[0], nearest[1]].astype(np.float32)
        blend = np.clip(distance / float(feather_px), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        composed[ring] = (
            nearest_values[ring] * (1.0 - blend[ring, None])
            + composed[ring] * blend[ring, None]
        )
    composed[keep] = source_values[keep]
    return Image.fromarray(np.rint(np.clip(composed, 0.0, 255.0)).astype(np.uint8), mode="RGB")


def create_half_completion_mask(image, mode="mirror-auto", blur=True):
    from PIL import Image, ImageFilter

    if mode == "mirror-auto":
        mode = _choose_symmetry_direction(image)
    if mode not in ("mirror-left-to-right", "mirror-right-to-left"):
        raise ValueError(f"Unsupported completion mask mode: {mode}")

    width, height = image.size
    midpoint = width // 2
    mask = Image.new("L", (width, height), 0)
    mask_data = np.zeros((height, width), dtype=np.uint8)
    if mode == "mirror-left-to-right":
        mask_data[:, midpoint:] = 255
    else:
        mask_data[:, :midpoint] = 255
    mask = Image.fromarray(mask_data, mode="L")
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, width // 96)))
    return mask, mode


def _save_masked_preview(image, mask, output_path):
    from PIL import Image

    preview = image.copy()
    fill = Image.new("RGB", image.size, (255, 255, 255))
    preview = Image.composite(fill, preview, mask)
    preview.save(output_path)


def _masked_edit_image(image, mask, fill_mode):
    if not fill_mode or fill_mode == "input":
        return image

    from PIL import Image

    if fill_mode == "white":
        fill = Image.new("RGB", image.size, (255, 255, 255))
    elif fill_mode == "gray":
        fill = Image.new("RGB", image.size, (192, 192, 192))
    elif fill_mode == "checker":
        tile = 16
        yy, xx = np.indices((image.height, image.width))
        checker = ((xx // tile + yy // tile) % 2).astype(np.uint8)
        values = np.where(checker[..., None] == 0, 216, 152).astype(np.uint8)
        fill = Image.fromarray(np.repeat(values, 3, axis=2), mode="RGB")
    elif fill_mode == "mirror":
        fill = _mirror_prefill_image(image, mask)
    elif fill_mode == "biharmonic":
        fill = _biharmonic_prefill_image(image, mask)
    else:
        raise ValueError(f"Unsupported edit mask fill mode: {fill_mode}")
    return Image.composite(fill, image, mask)


def _qwen_edit_prompt(prompt, edit_mask_fill):
    suffixes = {
        "checker": "Treat the checkerboard region as the area to fill. Do not alter the visible half.",
        "gray": "Treat the gray region as the area to fill. Do not alter the visible half.",
        "mirror": "Treat the mirrored prefilled half as the only area to refine. Do not alter the visible original half.",
        "biharmonic": "Treat the smooth prefilled half as the only area to refine. Do not alter the visible original half.",
        "white": "Treat the white blank region as the area to fill. Do not alter the visible half.",
        "input": "Treat the white blank region as the area to fill. Do not alter the visible half.",
    }
    fill_mode = edit_mask_fill or "input"
    suffix = suffixes.get(fill_mode, suffixes["input"])
    return " ".join(part for part in (prompt.strip(), suffix) if part)


def _masked_half(mask):
    data = np.asarray(mask.convert("L")) > 127
    if not np.any(data):
        return None
    midpoint = data.shape[1] // 2
    left_count = int(np.count_nonzero(data[:, :midpoint]))
    right_count = int(np.count_nonzero(data[:, midpoint:]))
    return "left" if left_count > right_count else "right"


def _mirror_prefill_image(image, mask):
    from PIL import Image

    half = _masked_half(mask)
    if half is None:
        return image.copy()
    width, height = image.size
    midpoint = width // 2
    output = image.copy()
    if half == "right":
        source = image.crop((0, 0, midpoint, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        output.paste(source.resize((width - midpoint, height)), (midpoint, 0))
    else:
        source = image.crop((midpoint, 0, width, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        output.paste(source.resize((midpoint, height)), (0, 0))
    return output


def _biharmonic_prefill_image(image, mask):
    from PIL import Image
    from skimage.restoration import inpaint_biharmonic

    data = np.asarray(image, dtype=np.float32) / 255.0
    mask_data = np.asarray(mask.convert("L")) > 127
    if not np.any(mask_data):
        return image.copy()
    filled = inpaint_biharmonic(data, mask_data, channel_axis=-1)
    return Image.fromarray(np.clip(filled * 255, 0, 255).astype(np.uint8), mode="RGB")


def _resize_for_inpaint(image, max_dimension=768, multiple=16, min_dimension=None):
    width, height = image.size
    effective_max_dimension = max_dimension
    if min_dimension is not None:
        effective_max_dimension = max(max_dimension, min_dimension)

    scale = min(1.0, effective_max_dimension / max(width, height))
    if min_dimension is not None and max(width, height) < min_dimension:
        scale = min_dimension / max(width, height)
    new_width = max(multiple, int(width * scale) // multiple * multiple)
    new_height = max(multiple, int(height * scale) // multiple * multiple)
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height))


def _resolve_torch_device(device):
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in ("cuda", "cpu"):
        return device
    if isinstance(device, str) and device.isdigit():
        return "cuda"
    return device


def _inpaint_torch_dtype(provider, device):
    import torch

    if device != "cuda":
        return torch.float32
    if provider in (
        "flux-fill",
        "flux2-klein-inpaint",
        "qwen-image-inpaint",
        "qwen-image-edit",
    ):
        return torch.bfloat16
    return torch.float16


def _enable_inpaint_memory_savers(pipe):
    for method_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        method = getattr(pipe, method_name, None)
        if callable(method):
            method()


def _load_inpaint_pipeline(provider, model_name, device, lora_weights=None, lora_scale=1.0):
    normalized_lora_weights = os.path.abspath(lora_weights) if lora_weights else None
    cache_key = (provider, model_name, device, normalized_lora_weights, float(lora_scale or 1.0))
    if cache_key in _INPAINT_PIPELINE_CACHE:
        return _INPAINT_PIPELINE_CACHE[cache_key]

    dtype = _inpaint_torch_dtype(provider, device)
    if provider == "flux-fill":
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(model_name, torch_dtype=dtype)
    elif provider == "flux2-klein-inpaint":
        from diffusers import Flux2KleinInpaintPipeline

        pipe = Flux2KleinInpaintPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
    elif provider == "sdxl-inpaint":
        from diffusers import StableDiffusionXLInpaintPipeline

        kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        if device == "cuda":
            kwargs["variant"] = "fp16"
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_name, **kwargs)
    elif provider == "dreamshaper-inpaint":
        from diffusers import StableDiffusionInpaintPipeline

        kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        if device == "cuda":
            kwargs["variant"] = "fp16"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, **kwargs)
    elif provider == "amused-inpaint":
        from diffusers import AmusedInpaintPipeline

        kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            kwargs["variant"] = "fp16"
        pipe = AmusedInpaintPipeline.from_pretrained(model_name, **kwargs)
    elif provider == "qwen-image-inpaint":
        from diffusers import QwenImageInpaintPipeline

        pipe = QwenImageInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
    elif provider == "qwen-image-edit":
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(model_name, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported modern inpaint provider: {provider}")

    if normalized_lora_weights:
        if not hasattr(pipe, "load_lora_weights"):
            raise ValueError(f"{provider} does not support LoRA adapter loading")
        if not os.path.exists(normalized_lora_weights):
            raise FileNotFoundError(f"LoRA adapter path does not exist: {normalized_lora_weights}")
        adapter_name = "metric_lora"
        lora_load_kwargs = _local_lora_load_kwargs(normalized_lora_weights)
        try:
            pipe.load_lora_weights(normalized_lora_weights, adapter_name=adapter_name, **lora_load_kwargs)
        except TypeError:
            pipe.load_lora_weights(normalized_lora_weights, **lora_load_kwargs)
        if hasattr(pipe, "set_adapters"):
            try:
                pipe.set_adapters([adapter_name], adapter_weights=[float(lora_scale or 1.0)])
            except TypeError:
                pipe.set_adapters(adapter_name, adapter_weights=float(lora_scale or 1.0))

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if device == "cuda":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _enable_inpaint_memory_savers(pipe)
    else:
        pipe.to("cpu")

    _INPAINT_PIPELINE_CACHE[cache_key] = pipe
    return pipe


def release_inpaint_pipelines(provider=None):
    import gc

    matching_keys = [
        key
        for key in _INPAINT_PIPELINE_CACHE
        if provider is None or key[0] == provider
    ]
    for key in matching_keys:
        pipe = _INPAINT_PIPELINE_CACHE.pop(key)
        free_hooks = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(free_hooks):
            free_hooks()
        del pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def release_depth_pipelines():
    import gc

    _DEPTH_PIPELINE_CACHE.clear()
    release_da3_models()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _local_lora_load_kwargs(lora_weights):
    if os.path.isdir(lora_weights):
        for filename in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors"):
            if os.path.exists(os.path.join(lora_weights, filename)):
                return {"weight_name": filename}
    return {}


def _choose_symmetry_direction(image):
    data = np.asarray(image.convert("L"), dtype=np.float32)
    midpoint = data.shape[1] // 2
    left = data[:, :midpoint]
    right = data[:, midpoint:]

    left_score = _half_detail_score(left)
    right_score = _half_detail_score(right)
    return "mirror-left-to-right" if left_score >= right_score else "mirror-right-to-left"


def _half_detail_score(values):
    if values.size == 0:
        return 0.0
    vertical_edges = np.abs(np.diff(values, axis=0)).mean() if values.shape[0] > 1 else 0.0
    horizontal_edges = np.abs(np.diff(values, axis=1)).mean() if values.shape[1] > 1 else 0.0
    contrast = values.std()
    return float(vertical_edges + horizontal_edges + contrast)


def _blend_vertical_seam(output, midpoint, feather_px, mode):
    if feather_px <= 0:
        return

    from PIL import Image

    width, height = output.size
    if mode == "mirror-left-to-right":
        left = midpoint
        right = min(width, midpoint + feather_px)
        if midpoint <= 0:
            return
        anchor = output.crop((midpoint - 1, 0, midpoint, height))
    else:
        left = max(0, midpoint - feather_px)
        right = midpoint
        if midpoint >= width:
            return
        anchor = output.crop((midpoint, 0, midpoint + 1, height))
    if right <= left:
        return

    output_region = output.crop((left, 0, right, height))
    mask_width = right - left
    anchor_region = anchor.resize((mask_width, height))
    mask = Image.new("L", (mask_width, height), 0)
    mask_data = np.zeros((height, mask_width), dtype=np.uint8)
    if mode == "mirror-left-to-right":
        alpha = np.linspace(0.0, 1.0, mask_width, dtype=np.float32)
    else:
        alpha = np.linspace(1.0, 0.0, mask_width, dtype=np.float32)
    mask_data[:] = (alpha * 255).astype(np.uint8)
    mask.putdata(mask_data.ravel())
    blended = Image.composite(output_region, anchor_region, mask)
    output.paste(blended, (left, 0))


# Image processing and getting depth data
def process_image_get_depth_data(
    input_image_path,
    output_dir="./output",
    provider="depth-anything-v2",
    model_name=None,
    device="auto",
):
    if provider in ("depth-anything-v2", "transformers"):
        return process_image_get_depth_data_transformers(
            input_image_path,
            output_dir=output_dir,
            model_name=model_name or "depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

    if provider == "sapiens":
        return process_image_get_depth_data_sapiens(input_image_path, output_dir=output_dir)

    raise ValueError(f"Unsupported depth provider: {provider}")


def process_image_get_depth_data_sapiens(input_image_path, output_dir="./output"):
    from gradio_client import Client, handle_file

    # Initialize the Gradio Client
    client = Client("facebook/sapiens_depth")

    # Perform the prediction and get the result
    result = client.predict(
        image=handle_file(input_image_path),
        depth_model_name="1b",
        seg_model_name="fg-bg-1b (recommended)",
        api_name="/process_image"
    )

    # Unpack the result (since it's a tuple of two paths)
    image_path, npy_path = result

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Move the image and the .npy file to the output directory
    shutil.copy(image_path, os.path.join(output_dir, "output_image.webp"))
    shutil.copy(npy_path, os.path.join(output_dir, "output_depth_data.npy"))

    print(f"Files saved successfully to {output_dir}")

    return os.path.join(output_dir, "output_depth_data.npy")


def process_image_get_depth_data_transformers(
    input_image_path,
    output_dir="./output",
    model_name="depth-anything/Depth-Anything-V2-Small-hf",
    device="auto",
    requested_model_name=None,
    fallback_reason=None,
):
    model_name = model_name or "depth-anything/Depth-Anything-V2-Small-hf"
    if is_da3_model(model_name):
        try:
            image = _load_depth_input_image(input_image_path)
            depth_data, da3_metadata = infer_da3_depth(
                image,
                model_id=model_name,
                device=device,
            )
            return _save_depth_outputs(
                depth_data,
                output_dir,
                metadata={
                    **da3_metadata,
                    "requested_model": requested_model_name or model_name,
                    "effective_model": model_name,
                    "fallback_model": None,
                    "fallback_reason": fallback_reason,
                    "relief_value_transform": RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
                },
                normalize_depth=False,
                preview_value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
            )
        except Exception as exc:
            return _run_depth_fallback(
                input_image_path,
                output_dir,
                requested_model_name or model_name,
                device,
                (
                    "Depth Anything V3 Large failed locally "
                    f"({type(exc).__name__}: {exc}); used verified local fallback instead."
                ),
            )
    if model_name == DEPTHPRO_MODEL_ID:
        return process_image_get_depth_data_depthpro(
            input_image_path,
            output_dir=output_dir,
            model_name=model_name,
            device=device,
            requested_model_name=requested_model_name,
            fallback_reason=fallback_reason,
        )

    try:
        import torch
        import transformers
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Local depth generation requires torch, transformers, and pillow. "
            "Install backend requirements before using depth-anything-v2."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    if device == "auto":
        pipeline_device = 0 if torch.cuda.is_available() else -1
    elif device == "cpu":
        pipeline_device = -1
    elif device == "cuda":
        pipeline_device = 0
    elif isinstance(device, str) and device.isdigit():
        pipeline_device = int(device)
    else:
        pipeline_device = device

    cache_key = (model_name, pipeline_device)
    if cache_key not in _DEPTH_PIPELINE_CACHE:
        pipe_kwargs = {"model": model_name, "device": pipeline_device}
        if pipeline_device != -1:
            transformers_major = int(transformers.__version__.split(".", 1)[0])
            dtype_arg = "dtype" if transformers_major >= 5 else "torch_dtype"
            pipe_kwargs[dtype_arg] = torch.float16
        _DEPTH_PIPELINE_CACHE[cache_key] = pipeline("depth-estimation", **pipe_kwargs)

    depth_pipe = _DEPTH_PIPELINE_CACHE[cache_key]
    image = _load_depth_input_image(input_image_path)
    result = depth_pipe(image)

    predicted_depth = result.get("predicted_depth")
    if predicted_depth is not None:
        depth_data = predicted_depth.detach().float().cpu().numpy()
    else:
        depth_image = result["depth"]
        depth_data = np.asarray(depth_image, dtype=np.float32)

    relief_value_transform = relief_value_transform_for_model(model_name)
    return _save_depth_outputs(
        depth_data,
        output_dir,
        metadata={
            "provider": "transformers",
            "requested_model": requested_model_name or model_name,
            "effective_model": model_name,
            "fallback_model": model_name if fallback_reason else None,
            "fallback_reason": fallback_reason,
            "relief_value_transform": relief_value_transform,
        },
        normalize_depth=relief_value_transform == RELIEF_VALUE_TRANSFORM_LINEAR,
        preview_value_transform=relief_value_transform,
    )


def process_image_get_depth_data_depthpro(
    input_image_path,
    output_dir="./output",
    model_name=DEPTHPRO_MODEL_ID,
    device="auto",
    requested_model_name=None,
    fallback_reason=None,
):
    try:
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
    except ImportError as exc:
        return _run_depth_fallback(
            input_image_path,
            output_dir,
            requested_model_name or model_name,
            device,
            f"Apple Depth Pro dependencies are unavailable: {exc}",
        )

    if not _hf_model_has_local_weights(model_name):
        return _run_depth_fallback(
            input_image_path,
            output_dir,
            requested_model_name or model_name,
            device,
            "Apple Depth Pro weights are not fully cached yet; used verified local fallback instead.",
        )

    try:
        os.makedirs(output_dir, exist_ok=True)
        resolved_device = _resolve_torch_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        cache_key = ("depthpro", model_name, resolved_device, str(dtype))
        if cache_key not in _DEPTH_PIPELINE_CACHE:
            processor = DepthProImageProcessor.from_pretrained(model_name, local_files_only=True)
            model_kwargs = {
                "local_files_only": True,
                "use_fov_model": False,
                "attn_implementation": "sdpa",
            }
            try:
                model_kwargs["dtype"] = dtype
                model = DepthProForDepthEstimation.from_pretrained(model_name, **model_kwargs)
            except TypeError:
                model_kwargs.pop("dtype", None)
                model_kwargs["torch_dtype"] = dtype
                model = DepthProForDepthEstimation.from_pretrained(model_name, **model_kwargs)
            model.to(resolved_device)
            model.eval()
            _DEPTH_PIPELINE_CACHE[cache_key] = (processor, model, resolved_device, dtype)

        processor, model, resolved_device, dtype = _DEPTH_PIPELINE_CACHE[cache_key]
        image = _load_depth_input_image(input_image_path)
        inputs = processor(images=image, return_tensors="pt")
        prepared_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if torch.is_floating_point(value):
                    prepared_inputs[key] = value.to(device=resolved_device, dtype=dtype)
                else:
                    prepared_inputs[key] = value.to(device=resolved_device)
            else:
                prepared_inputs[key] = value

        with torch.inference_mode():
            outputs = model(**prepared_inputs)

        post_processed_output = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        depth_data = post_processed_output[0]["predicted_depth"].detach().float().cpu().numpy()
        return _save_depth_outputs(
            depth_data,
            output_dir,
            metadata={
                "provider": "depthpro",
                "requested_model": requested_model_name or model_name,
                "effective_model": model_name,
                "fallback_model": None,
                "fallback_reason": fallback_reason,
                "depth_value_semantics": "distance_far_high",
                "relief_value_transform": RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
                "fov_model_enabled": bool(model.use_fov_model),
            },
            normalize_depth=False,
            preview_value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
        )
    except Exception as exc:
        return _run_depth_fallback(
            input_image_path,
            output_dir,
            requested_model_name or model_name,
            device,
            f"Apple Depth Pro failed locally ({type(exc).__name__}: {exc}); used verified local fallback instead.",
        )


def _run_depth_fallback(input_image_path, output_dir, requested_model_name, device, reason):
    fallback_model = os.getenv("DEPTH_FALLBACK_MODEL", DEFAULT_DEPTH_FALLBACK_MODEL)
    if fallback_model == requested_model_name:
        raise RuntimeError(reason)
    print(reason)
    return process_image_get_depth_data_transformers(
        input_image_path,
        output_dir=output_dir,
        model_name=fallback_model,
        device=device,
        requested_model_name=requested_model_name,
        fallback_reason=reason,
    )


def _hf_model_has_local_weights(model_name):
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    for filename in ("model.safetensors", "pytorch_model.bin"):
        cached = try_to_load_from_cache(model_name, filename)
        if isinstance(cached, str) and os.path.exists(cached) and not cached.endswith(".incomplete"):
            return True
    return False


def _transform_relief_values(values, value_transform=RELIEF_VALUE_TRANSFORM_LINEAR):
    if value_transform not in RELIEF_VALUE_TRANSFORMS:
        raise ValueError(
            f"Unsupported relief value transform: {value_transform}. "
            f"Expected one of {sorted(RELIEF_VALUE_TRANSFORMS)}."
        )

    transformed = values.astype(np.float32, copy=True)
    if value_transform == RELIEF_VALUE_TRANSFORM_LINEAR:
        return transformed, False

    finite_positive = transformed[np.isfinite(transformed) & (transformed > 0)]
    if finite_positive.size == 0:
        raise ValueError("Inverse-depth relief requires at least one finite positive depth value")

    # A tiny number of invalid near-zero estimates should not become unbounded
    # relief spikes. The 0.1th percentile is effectively a robust nearest plane.
    depth_floor = float(np.nanpercentile(finite_positive, 0.1))
    depth_floor = max(depth_floor, float(np.finfo(np.float32).tiny))
    positive = np.isfinite(transformed) & (transformed > 0)
    transformed[positive] = 1.0 / np.maximum(transformed[positive], depth_floor)
    transformed[np.isfinite(transformed) & ~positive] = np.nan
    return transformed, True


def _save_depth_outputs(
    depth_data,
    output_dir,
    metadata=None,
    normalize_depth=True,
    preview_value_transform=RELIEF_VALUE_TRANSFORM_LINEAR,
):
    from PIL import Image

    depth_data = np.squeeze(depth_data).astype(np.float32)
    finite_mask = np.isfinite(depth_data)
    if not np.any(finite_mask):
        raise ValueError("Depth model returned no finite depth values")

    finite_values = depth_data[finite_mask]
    depth_min = float(np.min(finite_values))
    depth_max = float(np.max(finite_values))
    if normalize_depth and depth_max > depth_min:
        depth_data = (depth_data - depth_min) / (depth_max - depth_min)

    npy_path = os.path.join(output_dir, "output_depth_data.npy")
    preview_path = os.path.join(output_dir, "output_depth_preview.png")
    metadata_path = os.path.join(output_dir, "output_depth_metadata.json")
    np.save(npy_path, depth_data)

    def normalized_preview(values):
        finite_values = values[np.isfinite(values)]
        preview_low = float(np.nanpercentile(finite_values, 1.0))
        preview_high = float(np.nanpercentile(finite_values, 99.0))
        if preview_high <= preview_low:
            preview_low = float(np.nanmin(finite_values))
            preview_high = float(np.nanmax(finite_values))
        if preview_high > preview_low:
            values = np.clip((values - preview_low) / (preview_high - preview_low), 0.0, 1.0)
        else:
            values = np.zeros_like(values)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)

    preview_data = normalized_preview(depth_data)
    Image.fromarray((preview_data * 255).astype(np.uint8)).save(preview_path)

    relief_preview_name = None
    if preview_value_transform != RELIEF_VALUE_TRANSFORM_LINEAR:
        relief_preview_name = "output_relief_preview.png"
        relief_preview_path = os.path.join(output_dir, relief_preview_name)
        relief_preview_data, _ = _transform_relief_values(depth_data, preview_value_transform)
        relief_preview_data = normalized_preview(relief_preview_data)
        Image.fromarray((relief_preview_data * 255).astype(np.uint8)).save(relief_preview_path)

    if metadata:
        import json

        metadata = dict(metadata)
        metadata.setdefault("stored_depth_normalized", bool(normalize_depth))
        metadata.setdefault("relief_value_transform", preview_value_transform)
        if relief_preview_name:
            metadata.setdefault("relief_preview", relief_preview_name)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    print(f"Depth data saved successfully to {npy_path}")
    return npy_path


def _smooth_nan_aware(values, sigma):
    if sigma <= 0:
        return values

    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    weights = valid.astype(np.float32)
    smoothed_values = gaussian_filter(filled, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)

    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed_values / smoothed_weights

    smoothed[smoothed_weights <= 1e-6] = np.nan
    return smoothed


def _resize_nan_aware(values, target_shape):
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_shape must contain positive dimensions")
    if values.shape == (target_height, target_width):
        return values

    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    weights = valid.astype(np.float32)
    factors = (target_height / values.shape[0], target_width / values.shape[1])
    resized_values = zoom(filled, factors, order=1)
    resized_weights = zoom(weights, factors, order=1)

    if resized_values.shape != (target_height, target_width):
        resized_values = resized_values[:target_height, :target_width]
        resized_weights = resized_weights[:target_height, :target_width]
        pad_height = target_height - resized_values.shape[0]
        pad_width = target_width - resized_values.shape[1]
        if pad_height > 0 or pad_width > 0:
            resized_values = np.pad(resized_values, ((0, max(0, pad_height)), (0, max(0, pad_width))), mode="edge")
            resized_weights = np.pad(resized_weights, ((0, max(0, pad_height)), (0, max(0, pad_width))), mode="edge")

    with np.errstate(invalid="ignore", divide="ignore"):
        resized = resized_values / resized_weights
    resized[resized_weights <= 1e-6] = np.nan
    return resized.astype(np.float32, copy=False)


def _resize_binary_mask(values, target_shape):
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    binary = np.asarray(values) > 0
    if binary.shape == (target_height, target_width):
        return binary
    factors = (target_height / binary.shape[0], target_width / binary.shape[1])
    resized = zoom(binary.astype(np.uint8), factors, order=0) > 0
    resized = resized[:target_height, :target_width]
    pad_height = target_height - resized.shape[0]
    pad_width = target_width - resized.shape[1]
    if pad_height > 0 or pad_width > 0:
        resized = np.pad(
            resized,
            ((0, max(0, pad_height)), (0, max(0, pad_width))),
            mode="edge",
        )
    return resized


def compose_selection_depth_with_context(
    depth_values,
    selection_mask,
    *,
    value_transform=RELIEF_VALUE_TRANSFORM_LINEAR,
    relief_height_mm=10.0,
    sample_pitch_mm=1.0,
    max_slope_mm_per_mm=2.0,
    base_margin_ratio=0.03,
    background_depth_ratio=DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    background_feather_mm=1.5,
    background_smoothing_mm=0.6,
):
    """Keep selected depth while retaining bounded, printable scene context."""
    source = np.asarray(depth_values, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Selection depth composition requires a 2D depth array")
    raw_mask = np.asarray(selection_mask)
    if raw_mask.ndim > 2:
        raw_mask = raw_mask[..., 0]
    if raw_mask.ndim != 2 or min(raw_mask.shape) < 2:
        raise ValueError("Selection mask must be a non-empty 2D image")
    depth_aspect = source.shape[1] / float(source.shape[0])
    mask_aspect = raw_mask.shape[1] / float(raw_mask.shape[0])
    if abs(depth_aspect - mask_aspect) / max(depth_aspect, mask_aspect, 1e-8) > 0.02:
        raise ValueError("Selection mask aspect ratio does not match the depth source")
    selected = _resize_binary_mask(selection_mask, source.shape)
    valid = np.isfinite(source)
    selected &= valid

    try:
        height_mm = float(relief_height_mm)
        pitch_mm = float(sample_pitch_mm)
        max_slope = float(max_slope_mm_per_mm)
        margin_ratio = float(base_margin_ratio)
        background_ratio = float(background_depth_ratio)
        background_feather = float(background_feather_mm)
        background_smoothing = float(background_smoothing_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid selection depth composition controls") from exc
    if (
        not np.isfinite(height_mm)
        or height_mm <= 0
        or not np.isfinite(pitch_mm)
        or pitch_mm <= 0
        or not np.isfinite(max_slope)
        or max_slope <= 0
        or not np.isfinite(margin_ratio)
        or margin_ratio < 0
        or not np.isfinite(background_ratio)
        or not 0 <= background_ratio <= 1
        or not np.isfinite(background_feather)
        or background_feather < 0
        or not np.isfinite(background_smoothing)
        or background_smoothing < 0
    ):
        raise ValueError("Selection depth composition controls are outside their supported range")

    canonical, reverses_order = _transform_relief_values(source, value_transform)
    canonical_valid = np.isfinite(canonical)
    valid = canonical_valid
    selected &= canonical_valid
    if np.count_nonzero(selected) < 4:
        raise ValueError(
            "Selection mask does not overlap enough valid relief-depth samples"
        )
    selected_values = canonical[selected]
    low = float(np.percentile(selected_values, 1.0))
    high = float(np.percentile(selected_values, 99.0))
    span = max(high - low, float(np.finfo(np.float32).eps))
    base_value = low - margin_ratio * span
    if reverses_order:
        base_value = max(base_value, low * 0.25, float(np.finfo(np.float32).tiny))

    # A one-pixel canonical change corresponds to this much physical relief.
    # Building the ramp before normalization keeps the selected surface intact
    # and prevents the structural-edge detector from preserving a vertical wall.
    canonical_step = span * max_slope * pitch_mm / height_mm
    canonical_step = max(canonical_step, span / max(source.shape) * 0.25)
    outside_distance, nearest = distance_transform_edt(
        ~selected,
        return_distances=True,
        return_indices=True,
    )
    nearest_selected_values = canonical[tuple(nearest)]
    ramp = np.maximum(
        base_value,
        nearest_selected_values - outside_distance * canonical_step,
    )
    composed_background = ramp
    background = (~selected) & valid
    background_context_enabled = False
    background_context_pixels = 0
    background_context_correlation = None
    background_context_residual_correlation = None
    background_context_residual_rms_retention = None
    background_context_normalized_correlation = None
    background_context_normalized_rms_retention = None
    background_context_recoverable_coverage_ratio = 0.0
    background_context_measured_pixels = 0
    background_context_measured_coverage_ratio = 0.0
    background_context_recoverable_measured_ratio = 0.0
    background_input_low = None
    background_input_high = None
    background_output_low = None
    background_output_high = None
    background_output_span_ratio = 0.0
    background_context_slope_guard = {
        "enabled": False,
        "reason": "background_context_disabled",
    }
    smoothing_sigma_px = 0.0
    feather_distance_px = 0.0
    if background_ratio > 0 and np.count_nonzero(background) >= 4:
        smoothing_sigma_px = (
            max(0.8, min(24.0, background_smoothing / pitch_mm))
            if background_smoothing > 0
            else 0.0
        )
        context_source = (
            _smooth_nan_aware(canonical, sigma=smoothing_sigma_px)
            if smoothing_sigma_px > 0
            else canonical
        )
        background_values = context_source[background]
        background_input_low, background_input_high = (
            float(value) for value in np.percentile(background_values, [2.0, 98.0])
        )
        background_span = background_input_high - background_input_low
        if np.isfinite(background_span) and background_span > float(np.finfo(np.float32).eps):
            context_unit = np.clip(
                (context_source - background_input_low) / background_span,
                0.0,
                1.0,
            )
            feather_distance_px = max(1.0, background_feather / pitch_mm)
            feather = np.clip(outside_distance / feather_distance_px, 0.0, 1.0)
            feather = feather * feather * (3.0 - 2.0 * feather)
            support_upper = nearest_selected_values + outside_distance * canonical_step
            context_ceiling = np.minimum(
                support_upper,
                np.maximum(base_value + background_ratio * span, ramp),
            )
            context_capacity = np.maximum(context_ceiling - ramp, 0.0)
            desired_context_residual = feather * context_capacity * context_unit
            context_candidate = ramp + desired_context_residual
            context_candidate = np.where(
                np.isfinite(context_candidate),
                context_candidate,
                ramp,
            )
            guarded_candidate, background_context_slope_guard = (
                _guard_weighted_feature_updates(
                    np.where(selected, canonical, ramp),
                    np.where(selected, canonical, context_candidate),
                    max_neighbor_step_mm=canonical_step,
                    max_ratio=2.5,
                )
            )
            composed_background = np.where(selected, ramp, guarded_candidate)
            context_lift = background & (composed_background > ramp + span * 1e-6)
            background_context_pixels = int(np.count_nonzero(context_lift))
            background_context_enabled = background_context_pixels > 0

            measured_background = background & (outside_distance >= feather_distance_px)
            background_context_measured_pixels = int(np.count_nonzero(measured_background))
            background_context_measured_coverage_ratio = float(
                background_context_measured_pixels
                / max(np.count_nonzero(background), 1)
            )
            if np.count_nonzero(measured_background) >= 4:
                source_measure = context_source[measured_background].astype(np.float64)
                output_measure = composed_background[measured_background].astype(np.float64)
                source_measure -= float(np.mean(source_measure))
                output_measure -= float(np.mean(output_measure))
                denominator = float(
                    np.sqrt(np.sum(source_measure * source_measure) * np.sum(output_measure * output_measure))
                )
                if denominator > 1e-12:
                    background_context_correlation = float(
                        np.sum(source_measure * output_measure) / denominator
                    )
                desired_residual = desired_context_residual[measured_background].astype(
                    np.float64
                )
                output_residual = (
                    composed_background[measured_background]
                    - ramp[measured_background]
                ).astype(np.float64)
                desired_residual -= float(np.mean(desired_residual))
                output_residual -= float(np.mean(output_residual))
                desired_norm = float(np.linalg.norm(desired_residual))
                output_norm = float(np.linalg.norm(output_residual))
                background_context_residual_correlation = (
                    float(
                        np.dot(desired_residual, output_residual)
                        / (desired_norm * output_norm)
                    )
                    if desired_norm > 1e-12 and output_norm > 1e-12
                    else (1.0 if desired_norm <= 1e-12 and output_norm <= 1e-8 else 0.0)
                )
                desired_rms = float(np.sqrt(np.mean(np.square(desired_residual))))
                output_rms = float(np.sqrt(np.mean(np.square(output_residual))))
                background_context_residual_rms_retention = (
                    output_rms / desired_rms
                    if desired_rms > 1e-12
                    else (1.0 if output_rms <= 1e-8 else None)
                )
                recoverable_background = (
                    measured_background
                    & (context_capacity > span * 1e-4)
                )
                background_context_recoverable_coverage_ratio = float(
                    np.count_nonzero(recoverable_background)
                    / max(np.count_nonzero(background), 1)
                )
                background_context_recoverable_measured_ratio = float(
                    np.count_nonzero(recoverable_background)
                    / max(background_context_measured_pixels, 1)
                )
                if np.count_nonzero(recoverable_background) >= 4:
                    normalized_source = context_unit[recoverable_background].astype(
                        np.float64
                    )
                    normalized_output = np.divide(
                        composed_background[recoverable_background]
                        - ramp[recoverable_background],
                        context_capacity[recoverable_background],
                        out=np.zeros(
                            np.count_nonzero(recoverable_background),
                            dtype=np.float64,
                        ),
                        where=context_capacity[recoverable_background] > span * 1e-4,
                    )
                    normalized_source -= float(np.mean(normalized_source))
                    normalized_output -= float(np.mean(normalized_output))
                    normalized_source_norm = float(np.linalg.norm(normalized_source))
                    normalized_output_norm = float(np.linalg.norm(normalized_output))
                    background_context_normalized_correlation = (
                        float(
                            np.dot(normalized_source, normalized_output)
                            / (normalized_source_norm * normalized_output_norm)
                        )
                        if normalized_source_norm > 1e-12
                        and normalized_output_norm > 1e-12
                        else (
                            1.0
                            if normalized_source_norm <= 1e-12
                            and normalized_output_norm <= 1e-8
                            else 0.0
                        )
                    )
                    normalized_source_rms = float(
                        np.sqrt(np.mean(np.square(normalized_source)))
                    )
                    normalized_output_rms = float(
                        np.sqrt(np.mean(np.square(normalized_output)))
                    )
                    background_context_normalized_rms_retention = (
                        normalized_output_rms / normalized_source_rms
                        if normalized_source_rms > 1e-12
                        else (1.0 if normalized_output_rms <= 1e-8 else None)
                    )
            background_output_low, background_output_high = (
                float(value) for value in np.percentile(composed_background[background], [2.0, 98.0])
            )
            background_output_span_ratio = (
                background_output_high - background_output_low
            ) / span

    composed_canonical = np.where(selected, canonical, composed_background).astype(
        np.float32,
        copy=False,
    )

    if reverses_order:
        composed = np.reciprocal(
            np.maximum(composed_canonical, float(np.finfo(np.float32).tiny)),
            dtype=np.float32,
        )
    else:
        composed = composed_canonical
    composed[selected] = source[selected]

    halo = (~selected) & (ramp > base_value + span * 1e-6)
    halo_distances = outside_distance[halo]
    return composed.astype(np.float32, copy=False), {
        "enabled": True,
        "method": "full_scene_depth_with_bounded_background_context_v3",
        "value_transform": value_transform,
        "mask_pixels": int(np.count_nonzero(selected)),
        "mask_coverage_ratio": float(np.mean(selected)),
        "support_halo_pixels": int(np.count_nonzero(halo)),
        "support_halo_coverage_ratio": float(np.mean(halo)),
        "support_distance_p95_px": (
            float(np.percentile(halo_distances, 95.0)) if halo_distances.size else 0.0
        ),
        "support_distance_max_px": (
            float(np.max(halo_distances)) if halo_distances.size else 0.0
        ),
        "sample_pitch_mm": pitch_mm,
        "max_slope_mm_per_mm": max_slope,
        "canonical_step_per_pixel": float(canonical_step),
        "selected_canonical_p01": low,
        "selected_canonical_p99": high,
        "base_canonical_value": float(base_value),
        "base_margin_ratio": margin_ratio,
        "background_depth_ratio": background_ratio,
        "background_feather_mm": background_feather,
        "background_feather_px": float(feather_distance_px),
        "background_smoothing_mm": background_smoothing,
        "background_smoothing_sigma_px": float(smoothing_sigma_px),
        "background_context_enabled": background_context_enabled,
        "background_context_pixels": background_context_pixels,
        "background_context_coverage_ratio": float(background_context_pixels / source.size),
        "background_context_correlation": background_context_correlation,
        "background_context_residual_correlation": (
            background_context_residual_correlation
        ),
        "background_context_residual_rms_retention": (
            background_context_residual_rms_retention
        ),
        "background_context_normalized_correlation": (
            background_context_normalized_correlation
        ),
        "background_context_normalized_rms_retention": (
            background_context_normalized_rms_retention
        ),
        "background_context_recoverable_coverage_ratio": (
            background_context_recoverable_coverage_ratio
        ),
        "background_context_measured_pixels": background_context_measured_pixels,
        "background_context_measured_coverage_ratio": (
            background_context_measured_coverage_ratio
        ),
        "background_context_recoverable_measured_ratio": (
            background_context_recoverable_measured_ratio
        ),
        "background_context_slope_guard": background_context_slope_guard,
        "background_input_canonical_p02": background_input_low,
        "background_input_canonical_p98": background_input_high,
        "background_output_canonical_p02": background_output_low,
        "background_output_canonical_p98": background_output_high,
        "background_output_span_ratio": float(background_output_span_ratio),
    }


def _target_shape_for_max_dimension(shape, target_dimension):
    target_dimension = int(round(float(target_dimension)))
    if target_dimension < 2:
        raise ValueError("target_dimension must be at least 2 or -1")
    height, width = shape
    max_dimension = max(height, width)
    if max_dimension <= target_dimension:
        return height, width
    scale = target_dimension / float(max_dimension)
    return max(2, int(round(height * scale))), max(2, int(round(width * scale)))


def _normalize_relief_values(
    values,
    low_percentile=1.0,
    high_percentile=99.0,
    reference_mask=None,
    reference_values=None,
):
    normalized = values.astype(np.float32, copy=True)
    finite_mask = np.isfinite(normalized)
    if reference_values is None:
        normalization_reference = normalized
    else:
        normalization_reference = np.asarray(
            reference_values,
            dtype=np.float32,
        )
        if normalization_reference.shape != normalized.shape:
            raise ValueError(
                "Relief normalization reference must match the depth grid"
            )
    reference = finite_mask & np.isfinite(normalization_reference)
    if reference_mask is not None:
        reference_mask = np.asarray(reference_mask) > 0
        if reference_mask.shape != normalized.shape:
            raise ValueError("Relief normalization mask must match the depth grid")
        masked_reference = reference & reference_mask
        if np.count_nonzero(masked_reference) >= 4:
            reference = masked_reference
    finite = normalization_reference[reference]
    if finite.size == 0:
        return normalized

    low = float(np.nanpercentile(finite, low_percentile))
    high = float(np.nanpercentile(finite, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        normalized[np.isfinite(normalized)] = 0.0
        return normalized

    normalized = (normalized - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def _normalization_reference_subject_weight(reference_mask):
    selected = np.asarray(reference_mask) > 0
    if selected.ndim != 2 or not np.any(selected):
        raise ValueError(
            "Relief normalization blend expects a non-empty 2D mask"
        )
    distance = distance_transform_edt(selected)
    weight = np.clip(
        (
            distance - NORMALIZATION_REFERENCE_BOUNDARY_PX
        )
        / NORMALIZATION_REFERENCE_TAPER_PX,
        0.0,
        1.0,
    )
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight *= selected
    weight[
        distance <= NORMALIZATION_REFERENCE_BOUNDARY_PX
    ] = 0.0
    return weight.astype(np.float32, copy=False)


def _detail_edge_window_size(detail_radius):
    try:
        radius = float(detail_radius)
    except (TypeError, ValueError):
        radius = 0.0
    support_radius = max(1, int(np.ceil(max(0.0, radius) * 2.0)))
    return min(31, support_radius * 2 + 1)


def _shape_relief_values(
    values,
    invert=False,
    gamma=0.75,
    detail_boost=1.4,
    detail_radius=2.0,
    detail_edge_threshold=0.12,
    low_percentile=1.0,
    high_percentile=99.0,
    value_transform=RELIEF_VALUE_TRANSFORM_LINEAR,
    detail_protection_mask=None,
    background_detail_boost=1.0,
    normalization_mask=None,
    normalization_reference_values=None,
):
    transformed, reverses_order = _transform_relief_values(values, value_transform)
    transformed_reference = None
    if normalization_reference_values is not None:
        transformed_reference, reference_reverses_order = (
            _transform_relief_values(
                normalization_reference_values,
                value_transform,
            )
        )
        if reference_reverses_order != reverses_order:
            raise ValueError(
                "Relief normalization reference has incompatible semantics"
            )
    reference_relief = _normalize_relief_values(
        transformed,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        reference_mask=normalization_mask,
        reference_values=transformed_reference,
    )
    relief = reference_relief
    if (
        transformed_reference is not None
        and normalization_mask is not None
    ):
        candidate_relief = _normalize_relief_values(
            transformed,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            reference_mask=normalization_mask,
        )
        subject_weight = _normalization_reference_subject_weight(
            normalization_mask
        )
        relief = reference_relief + subject_weight * (
            candidate_relief - reference_relief
        )
    if bool(invert) ^ reverses_order:
        relief = 1.0 - relief

    if detail_boost > 0 and detail_radius > 0:
        local_base = _smooth_nan_aware(relief, sigma=detail_radius)
        detail = relief - local_base
        finite = np.isfinite(relief)
        if np.any(finite) and detail_edge_threshold > 0:
            fill_value = float(np.nanmedian(relief[finite]))
            filled = np.where(finite, relief, fill_value)
            # Cover the same transition support as the Gaussian detail filter.
            # A 3x3 gate misses soft depth shoulders and lets unsharp masking
            # turn face silhouettes into raised rims.
            edge_window_size = _detail_edge_window_size(detail_radius)
            local_range = maximum_filter(filled, size=edge_window_size, mode="nearest") - minimum_filter(
                filled,
                size=edge_window_size,
                mode="nearest",
            )
            edge_ratio = local_range / float(detail_edge_threshold)
            edge_gate = 1.0 / (1.0 + np.power(edge_ratio, 4))
            detail = np.clip(detail, -detail_edge_threshold, detail_edge_threshold)
        else:
            edge_gate = 1.0
        detail_multiplier = 1.0
        if detail_protection_mask is not None and background_detail_boost > 1.0:
            protection = np.asarray(detail_protection_mask, dtype=np.float32)
            if protection.shape != relief.shape:
                protection = _resize_binary_mask(protection > 0, relief.shape).astype(np.float32)
            # Feather the semantic boundary so boosted architecture does not
            # produce a visible ridge where the face-safe region begins.
            protection = gaussian_filter(protection, sigma=max(1.0, float(detail_radius) * 1.5))
            protection = np.clip(protection, 0.0, 1.0)
            detail_multiplier = 1.0 + (float(background_detail_boost) - 1.0) * (1.0 - protection)
        elif background_detail_boost > 1.0:
            detail_multiplier = float(background_detail_boost)
        relief = relief + detail_boost * detail * edge_gate * detail_multiplier

    relief = np.clip(relief, 0.0, 1.0)
    if gamma > 0 and gamma != 1:
        relief = np.power(relief, gamma)
    return np.clip(relief, 0.0, 1.0)


def _has_background_photo_detail_protection(protection_mask):
    if protection_mask is not None:
        try:
            if bool(np.any(np.asarray(protection_mask) > 0)):
                return True
        except (TypeError, ValueError):
            pass
    return False


def _effective_background_photo_detail_mm(requested_detail_mm, protection_mask):
    requested = float(requested_detail_mm)
    if not np.isfinite(requested):
        raise ValueError("background_photo_detail_mm must be finite")
    requested = float(np.clip(requested, 0.0, 0.60))
    return (
        requested
        if _has_background_photo_detail_protection(protection_mask)
        else min(requested, 0.12)
    )


BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM = 5.0
BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM = 2.0


def _photo_detail_sampling(
    max_xy_size,
    shape,
    protection_halo_mm=BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
):
    longest_dimension = max((int(value) for value in shape), default=1)
    sample_pitch_mm = (
        float(max_xy_size) / max(longest_dimension - 1, 1)
        if max_xy_size is not None and float(max_xy_size) > 0
        else 1.0
    )
    halo_px = max(0.0, float(protection_halo_mm)) / max(sample_pitch_mm, 1e-6)
    return sample_pitch_mm, halo_px


def _photo_detail_background_gate(
    protection_mask,
    protection_halo_px,
    zero_guard_px=0.0,
):
    """Feather from a protected boundary to full detail over a physical halo."""
    protected = np.asarray(protection_mask, dtype=bool)
    try:
        radius_px = float(protection_halo_px)
    except (TypeError, ValueError) as exc:
        raise ValueError("Photo-detail protection halo must be finite") from exc
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Photo-detail protection halo must be positive")
    try:
        inner_px = float(zero_guard_px)
    except (TypeError, ValueError) as exc:
        raise ValueError("Photo-detail zero guard must be finite") from exc
    if not np.isfinite(inner_px) or inner_px < 0 or inner_px >= radius_px:
        raise ValueError("Photo-detail zero guard must be within the halo")
    distance_px = distance_transform_edt(~protected)
    normalized = np.clip(
        (distance_px - inner_px) / (radius_px - inner_px),
        0.0,
        1.0,
    )
    gate = normalized * normalized * (3.0 - 2.0 * normalized)
    gate[protected] = 0.0
    return gate.astype(np.float32, copy=False)


def _inject_photo_relief_detail(
    relief,
    source_image,
    max_detail_ratio=0.0,
    protection_mask=None,
    protection_halo_px=8.0,
):
    """Recover printable texture that a monocular depth model flattened away."""
    if source_image is None or max_detail_ratio <= 0:
        return relief, {"enabled": False, "max_detail_ratio": float(max(0.0, max_detail_ratio))}

    from PIL import Image

    try:
        if isinstance(source_image, (str, os.PathLike)):
            image = Image.open(source_image).convert("RGB")
        elif isinstance(source_image, Image.Image):
            image = source_image.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(source_image)).convert("RGB")
    except (OSError, ValueError, TypeError) as exc:
        return relief, {
            "enabled": False,
            "reason": "source_image_unreadable",
            "error": str(exc),
        }
    image = image.resize((relief.shape[1], relief.shape[0]), Image.Resampling.LANCZOS)
    rgb = np.flip(np.asarray(image, dtype=np.float32) / 255.0, axis=1)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    fine_detail = gray - gaussian_filter(gray, sigma=1.1)
    medium_detail = gray - gaussian_filter(gray, sigma=3.5)
    photo_detail = 0.72 * fine_detail + 0.28 * medium_detail

    finite = np.isfinite(relief)
    if not np.any(finite):
        return relief, {"enabled": False, "reason": "no_finite_relief"}
    low, high = np.percentile(relief[finite], [48.0, 76.0])
    span = max(float(high - low), 1e-6)
    foreground_weight = np.clip((relief - low) / span, 0.0, 1.0)
    foreground_weight = foreground_weight * foreground_weight * (3.0 - 2.0 * foreground_weight)
    background_gate = 1.0 - foreground_weight

    protected = None
    if protection_mask is not None:
        candidate_protection = np.asarray(protection_mask) > 0
        if candidate_protection.shape != relief.shape:
            candidate_protection = _resize_binary_mask(
                candidate_protection, relief.shape
            )
        if np.any(candidate_protection):
            protected = candidate_protection
    if protected is not None:
        # Use a physical Euclidean feather rather than square dilation followed
        # by a Gaussian. The old composition extended the nominal 5 mm halo to
        # about 7.1 mm diagonally and attenuated legitimate scenery beyond it.
        # Protected pixels stay exact while background detail reaches full gain
        # at the declared physical distance in every direction.
        background_gate = _photo_detail_background_gate(
            protected,
            max(1.0, float(protection_halo_px)),
            zero_guard_px=(
                max(1.0, float(protection_halo_px))
                * BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM
                / BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM
            ),
        )
        # A semantic mask tells us which pixels are truly background. Preserve
        # raised architecture and scenery instead of suppressing it merely
        # because its monocular depth happens to resemble the foreground.

    usable = finite & (background_gate > 0.5)
    if not np.any(usable):
        return relief, {"enabled": False, "reason": "no_background_pixels"}
    detail_scale_percentile = 96.0
    detail_scale = float(
        np.percentile(np.abs(photo_detail[usable]), detail_scale_percentile)
    )
    if not np.isfinite(detail_scale) or detail_scale <= 1e-6:
        return relief, {"enabled": False, "reason": "no_photo_detail"}

    detail_contrast_gamma = 0.75
    normalized_detail = np.clip(photo_detail / detail_scale, -1.0, 1.0)
    normalized_detail = np.sign(normalized_detail) * np.power(
        np.abs(normalized_detail), detail_contrast_gamma
    )
    fused = relief + float(max_detail_ratio) * normalized_detail * background_gate
    return np.clip(fused, 0.0, 1.0), {
        "enabled": True,
        "max_detail_ratio": float(max_detail_ratio),
        "detail_scale": detail_scale,
        "detail_scale_percentile": detail_scale_percentile,
        "detail_contrast_gamma": detail_contrast_gamma,
        "background_coverage_ratio": float(np.mean(usable)),
        "face_protected": protected is not None,
        "protection_halo_px": (
            float(max(1.0, protection_halo_px)) if protected is not None else 0.0
        ),
        "protection_method": (
            "euclidean_inner_guard_smoothstep_to_full_gain"
            if protected is not None
            else "none"
        ),
        "protection_zero_guard_mm": (
            BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM if protected is not None else 0.0
        ),
    }


def _top_silhouette_mask(source_image, target_shape, padding_px=1):
    """Keep everything below the first non-background pixel in each column."""
    full_mask = np.ones((int(target_shape[0]), int(target_shape[1])), dtype=bool)
    if source_image is None:
        return full_mask, {"enabled": False, "reason": "no_source_image"}

    from PIL import Image

    try:
        if isinstance(source_image, (str, os.PathLike)):
            image = Image.open(source_image).convert("RGBA")
        elif isinstance(source_image, Image.Image):
            image = source_image.convert("RGBA")
        else:
            image = Image.fromarray(np.asarray(source_image)).convert("RGBA")
    except (OSError, ValueError, TypeError) as exc:
        return full_mask, {
            "enabled": False,
            "reason": "source_image_unreadable",
            "error": str(exc),
        }

    image = image.resize((full_mask.shape[1], full_mask.shape[0]), Image.Resampling.LANCZOS)
    rgba = np.asarray(image, dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    top_band = rgb[: max(2, min(6, rgb.shape[0])), :, :].reshape(-1, 3)
    background_color = np.median(top_band, axis=0)
    color_distance = np.max(np.abs(rgb - background_color), axis=2)
    background = (alpha < 0.05) | ((color_distance < 0.08) & (alpha > 0.95))
    content = maximum_filter((~background).astype(np.uint8), size=3) > 0

    has_content = np.any(content, axis=0)
    first_content = np.argmax(content, axis=0)
    first_content = np.where(has_content, first_content, full_mask.shape[0])
    first_content = np.maximum(0, first_content - max(0, int(padding_px)))
    rows = np.arange(full_mask.shape[0])[:, None]
    silhouette = rows >= first_content[None, :]
    silhouette = np.flip(silhouette, axis=1)
    removed = ~silhouette
    return silhouette, {
        "enabled": bool(np.any(removed)),
        "background_color": [float(value) for value in background_color],
        "removed_area_ratio": float(np.mean(removed)),
        "skyline_min_row": int(np.min(first_content)),
        "skyline_max_row": int(np.max(first_content)),
        "padding_px": int(max(0, padding_px)),
    }


def _load_flipped_feature_exclusion_mask(feature_exclusion_mask, target_shape):
    if feature_exclusion_mask is None:
        return None

    from PIL import Image

    if isinstance(feature_exclusion_mask, (str, os.PathLike)):
        exclusion = np.asarray(Image.open(feature_exclusion_mask).convert("L")) > 0
    elif isinstance(feature_exclusion_mask, Image.Image):
        exclusion = np.asarray(feature_exclusion_mask.convert("L")) > 0
    else:
        exclusion = np.asarray(feature_exclusion_mask) > 0
    exclusion = _resize_binary_mask(exclusion, target_shape)
    return np.flip(exclusion, axis=1)


def _enhance_weighted_relief_features(
    values,
    feature_weight_mask,
    max_feature_depth_mm=0.0,
    detail_radius_px=1.4,
    edge_limit_mm=4.0,
    feature_exclusion_mask=None,
):
    """Add bounded signed depth to printable face features in physical units."""
    if feature_weight_mask is None or max_feature_depth_mm <= 0:
        return values, {
            "enabled": False,
            "max_feature_depth_mm": float(max(0.0, max_feature_depth_mm)),
        }

    from PIL import Image

    try:
        if isinstance(feature_weight_mask, (str, os.PathLike)):
            weight_image = Image.open(feature_weight_mask).convert("L")
        elif isinstance(feature_weight_mask, Image.Image):
            weight_image = feature_weight_mask.convert("L")
        else:
            weight_array = np.asarray(feature_weight_mask)
            if np.issubdtype(weight_array.dtype, np.floating):
                weight_array = np.uint8(np.clip(weight_array, 0.0, 1.0) * 255.0)
            weight_image = Image.fromarray(weight_array).convert("L")
    except (OSError, ValueError, TypeError) as exc:
        return values, {
            "enabled": False,
            "reason": "feature_weight_unreadable",
            "error": str(exc),
        }
    try:
        exclusion = _load_flipped_feature_exclusion_mask(
            feature_exclusion_mask,
            values.shape,
        )
    except (OSError, ValueError, TypeError) as exc:
        return values, {
            "enabled": False,
            "reason": "feature_exclusion_unreadable",
            "error": str(exc),
        }

    weight_image = weight_image.resize((values.shape[1], values.shape[0]), Image.Resampling.BILINEAR)
    weight = np.flip(np.asarray(weight_image, dtype=np.float32) / 255.0, axis=1)
    if exclusion is not None:
        weight = np.where(exclusion, 0.0, weight)
    valid = np.isfinite(values)
    support = valid & (weight > 0.08)
    if not np.any(support):
        return values, {"enabled": False, "reason": "empty_feature_weight"}

    local_base = _smooth_nan_aware(values, sigma=detail_radius_px)
    detail = values - local_base
    detail_scale = float(np.percentile(np.abs(detail[support]), 88.0))
    if not np.isfinite(detail_scale) or detail_scale <= 1e-6:
        return values, {"enabled": False, "reason": "no_weighted_depth_detail"}
    normalized_detail = np.clip(detail / detail_scale, -1.0, 1.0)

    fill_value = float(np.nanmedian(values[valid]))
    filled = np.where(valid, values, fill_value)
    local_range = maximum_filter(filled, size=7, mode="nearest") - minimum_filter(
        filled,
        size=7,
        mode="nearest",
    )
    edge_gate = 1.0 / (1.0 + np.power(local_range / max(float(edge_limit_mm), 1e-6), 4))
    addition = float(max_feature_depth_mm) * normalized_detail * weight * edge_gate
    if exclusion is not None:
        addition = np.where(exclusion, 0.0, addition)
    enhanced = np.where(valid, values + addition, np.nan)
    applied = np.abs(addition[support])
    return enhanced, {
        "enabled": True,
        "max_feature_depth_mm": float(max_feature_depth_mm),
        "detail_scale_mm": detail_scale,
        "support_ratio": float(np.mean(support)),
        "excluded_ratio": float(np.mean(exclusion)) if exclusion is not None else 0.0,
        "applied_depth_p95_mm": float(np.percentile(applied, 95.0)),
        "applied_depth_max_mm": float(np.max(applied)),
    }


def _bridge_weighted_face_features(
    values,
    face_region_mask,
    feature_weight_mask,
    max_bridge_depth_mm=0.0,
    feature_exclusion_mask=None,
):
    """Fill narrow peripheral valleys so eyewear remains attached to a face."""
    if face_region_mask is None or feature_weight_mask is None or max_bridge_depth_mm <= 0:
        return values, {
            "enabled": False,
            "max_bridge_depth_mm": float(max(0.0, max_bridge_depth_mm)),
        }

    from PIL import Image

    try:
        if isinstance(feature_weight_mask, (str, os.PathLike)):
            weight_image = Image.open(feature_weight_mask).convert("L")
        elif isinstance(feature_weight_mask, Image.Image):
            weight_image = feature_weight_mask.convert("L")
        else:
            weight_array = np.asarray(feature_weight_mask)
            if np.issubdtype(weight_array.dtype, np.floating):
                weight_array = np.uint8(np.clip(weight_array, 0.0, 1.0) * 255.0)
            weight_image = Image.fromarray(weight_array).convert("L")
    except (OSError, ValueError, TypeError) as exc:
        return values, {
            "enabled": False,
            "reason": "feature_weight_unreadable",
            "error": str(exc),
        }
    try:
        exclusion = _load_flipped_feature_exclusion_mask(
            feature_exclusion_mask,
            values.shape,
        )
    except (OSError, ValueError, TypeError) as exc:
        return values, {
            "enabled": False,
            "reason": "feature_exclusion_unreadable",
            "error": str(exc),
        }

    region = _resize_binary_mask(face_region_mask, values.shape)
    weight_image = weight_image.resize((values.shape[1], values.shape[0]), Image.Resampling.BILINEAR)
    weight = np.flip(np.asarray(weight_image, dtype=np.float32) / 255.0, axis=1)
    valid = np.isfinite(values)
    inner_region = binary_erosion(
        region,
        structure=np.ones((13, 13), dtype=bool),
        border_value=0,
    )
    peripheral_band = region & ~inner_region
    feature_support = maximum_filter((weight > 0.08).astype(np.uint8), size=9) > 0
    bridge_zone = valid & peripheral_band & feature_support
    if exclusion is not None:
        bridge_zone &= ~exclusion
    if not np.any(bridge_zone):
        return values, {"enabled": False, "reason": "no_peripheral_feature_gaps"}

    fill_value = float(np.nanmedian(values[valid]))
    filled = np.where(valid, values, fill_value)
    closed = grey_closing(filled, size=(5, 5))
    potential_raise = np.maximum(closed - values, 0.0)
    bridge_weight = maximum_filter(weight, size=9)
    if exclusion is not None:
        bridge_weight = np.where(exclusion, 0.0, bridge_weight)
    addition = np.minimum(potential_raise, float(max_bridge_depth_mm)) * bridge_weight
    addition = np.where(bridge_zone, addition, 0.0)
    if exclusion is not None:
        addition = np.where(exclusion, 0.0, addition)
    bridged = np.where(valid, values + addition, np.nan)
    applied = addition[bridge_zone]
    active = applied > 0.02
    return bridged, {
        "enabled": True,
        "max_bridge_depth_mm": float(max_bridge_depth_mm),
        "bridge_zone_ratio": float(np.mean(bridge_zone)),
        "excluded_ratio": float(np.mean(exclusion)) if exclusion is not None else 0.0,
        "bridged_pixels": int(np.count_nonzero(active)),
        "applied_raise_p95_mm": float(np.percentile(applied, 95.0)),
        "applied_raise_max_mm": float(np.max(applied)),
    }


def _surface_edge_audit(values, baseline_values, max_neighbor_step_mm):
    """Measure new grid-edge steepness, including the STL diagonals."""
    try:
        max_step = float(max_neighbor_step_mm)
    except (TypeError, ValueError):
        return {"enabled": False, "reason": "invalid_max_neighbor_step"}
    if not np.isfinite(max_step) or max_step <= 0:
        return {"enabled": False, "reason": "invalid_max_neighbor_step"}

    surface = np.asarray(values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if surface.shape != baseline.shape:
        raise ValueError("Surface audit inputs must have identical shapes")
    edge_views = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), 1.0),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), 1.0),
        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), np.sqrt(2.0)),
        ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), np.sqrt(2.0)),
    )
    ratios = []
    physical_ratios = []
    exempt_edges = 0
    physical_violations = 0
    for first, second, edge_length in edge_views:
        valid = (
            np.isfinite(surface[first])
            & np.isfinite(surface[second])
            & np.isfinite(baseline[first])
            & np.isfinite(baseline[second])
        )
        if not np.any(valid):
            continue
        steps = np.abs(surface[second] - surface[first])[valid]
        baseline_steps = np.abs(baseline[second] - baseline[first])[valid]
        physical_limit = max_step * float(edge_length)
        allowed = np.maximum(physical_limit, baseline_steps)
        ratios.append(steps / np.maximum(allowed, 1e-8))
        physical_ratios.append(steps / max(physical_limit, 1e-8))
        exempt_edges += int(np.count_nonzero(baseline_steps > physical_limit + 1e-8))
        physical_violations += int(np.count_nonzero(steps > physical_limit + 1e-8))
    if not ratios:
        return {"enabled": False, "reason": "no_finite_edges"}

    ratio_values = np.concatenate(ratios)
    physical_ratio_values = np.concatenate(physical_ratios)
    return {
        "enabled": True,
        "edge_count": int(ratio_values.size),
        "max_neighbor_step_mm": float(max_step),
        "baseline_exempt_edge_count": int(exempt_edges),
        "physical_violation_edge_count": int(physical_violations),
        "accepted_surface_ratio_p99": float(np.percentile(ratio_values, 99.0)),
        "accepted_surface_ratio_max": float(np.max(ratio_values)),
        "physical_slope_ratio_p99": float(np.percentile(physical_ratio_values, 99.0)),
        "physical_slope_ratio_max": float(np.max(physical_ratio_values)),
    }


def _guard_weighted_feature_updates(
    baseline_values,
    candidate_values,
    max_neighbor_step_mm,
    max_ratio=1.025,
    iterations=16,
):
    """Attenuate feature additions that make an accepted surface steeper."""
    baseline = np.asarray(baseline_values)
    candidate = np.asarray(candidate_values)
    if baseline.shape != candidate.shape:
        raise ValueError("Feature guard inputs must have identical shapes")
    delta = np.where(
        np.isfinite(baseline) & np.isfinite(candidate),
        candidate - baseline,
        0.0,
    )
    if float(np.max(np.abs(delta), initial=0.0)) <= 1e-8:
        return candidate_values, {
            "enabled": False,
            "reason": "no_feature_update",
            "applied_scale": 1.0,
        }

    candidate_audit = _surface_edge_audit(candidate, baseline, max_neighbor_step_mm)
    if not candidate_audit.get("enabled", False):
        return candidate_values, {
            "enabled": False,
            "reason": candidate_audit.get("reason", "audit_unavailable"),
            "applied_scale": 1.0,
            "candidate_audit": candidate_audit,
        }

    ratio_limit = max(float(max_ratio), 1.0)

    def accepted(audit):
        return (
            audit.get("enabled", False)
            and float(audit["accepted_surface_ratio_max"]) <= ratio_limit + 1e-6
        )

    if accepted(candidate_audit):
        return candidate_values, {
            "enabled": True,
            "attenuated": False,
            "applied_scale": 1.0,
            "max_ratio": ratio_limit,
            "candidate_audit": candidate_audit,
            "final_audit": candidate_audit,
        }

    baseline64 = baseline.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    output = candidate64.copy()
    movable = np.isfinite(baseline64) & np.isfinite(candidate64) & (np.abs(delta) > 1e-10)
    lower = np.minimum(baseline64, candidate64)
    upper = np.maximum(baseline64, candidate64)
    edge_views = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), 1.0),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), 1.0),
        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), np.sqrt(2.0)),
        ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), np.sqrt(2.0)),
    )
    projection_iterations = 0
    final_audit = candidate_audit
    for projection_iterations in range(1, max(8, int(iterations) * 4) + 1):
        for first, second, edge_length in edge_views:
            first_values = output[first]
            second_values = output[second]
            valid_edges = (
                np.isfinite(first_values)
                & np.isfinite(second_values)
                & np.isfinite(baseline64[first])
                & np.isfinite(baseline64[second])
            )
            baseline_steps = np.abs(baseline64[second] - baseline64[first])
            allowed = np.maximum(
                float(max_neighbor_step_mm) * float(edge_length),
                baseline_steps,
            ) * ratio_limit
            difference = second_values - first_values
            excess = np.abs(difference) - allowed
            first_movable = movable[first]
            second_movable = movable[second]
            adjustable_count = first_movable.astype(np.int8) + second_movable.astype(np.int8)
            violations = valid_edges & (excess > 1e-8) & (adjustable_count > 0)
            if not np.any(violations):
                continue
            direction = np.sign(difference[violations])
            correction = excess[violations] / adjustable_count[violations]
            first_change = direction * correction * first_movable[violations]
            second_change = direction * correction * second_movable[violations]
            first_values[violations] += first_change
            second_values[violations] -= second_change
        finite_output = np.isfinite(output)
        output[finite_output] = np.clip(
            output[finite_output],
            lower[finite_output],
            upper[finite_output],
        )
        final_audit = _surface_edge_audit(output, baseline64, max_neighbor_step_mm)
        if accepted(final_audit):
            break

    if not accepted(final_audit):
        # Projection should converge because the baseline is feasible. Keep a
        # scalar backstop so numerical edge cases still fail toward that safe
        # surface rather than emitting an unchecked update.
        projection_final_audit = final_audit
        projected_delta = np.where(movable, output - baseline64, 0.0)
        low_scale = 0.0
        high_scale = 1.0
        output = baseline64.copy()
        final_audit = _surface_edge_audit(output, baseline64, max_neighbor_step_mm)
        for _ in range(max(24, int(iterations))):
            scale = (low_scale + high_scale) * 0.5
            trial = np.where(
                np.isfinite(baseline64),
                baseline64 + scale * projected_delta,
                np.nan,
            )
            trial_audit = _surface_edge_audit(trial, baseline64, max_neighbor_step_mm)
            if accepted(trial_audit):
                low_scale = scale
                output = trial
                final_audit = trial_audit
            else:
                high_scale = scale
    else:
        projection_final_audit = final_audit
        low_scale = 1.0

    fell_back_to_baseline = False
    if not accepted(final_audit):
        output = baseline64.copy()
        final_audit = _surface_edge_audit(output, baseline64, max_neighbor_step_mm)
        fell_back_to_baseline = True

    original_rms = float(np.sqrt(np.mean(np.square(delta[movable]))))
    retained_rms = float(np.sqrt(np.mean(np.square((output - baseline64)[movable]))))
    retained_ratio = retained_rms / max(original_rms, 1e-12)
    attenuated_pixels = int(
        np.count_nonzero(
            movable
            & (np.abs(output - candidate64) > max(1e-6, original_rms * 1e-4))
        )
    )
    return output.astype(candidate.dtype, copy=False), {
        "enabled": True,
        "attenuated": True,
        "attenuation_strategy": "local_edge_projection",
        "applied_scale": float(retained_ratio),
        "retained_update_rms_ratio": float(retained_ratio),
        "attenuated_pixels": int(attenuated_pixels),
        "updated_pixels": int(np.count_nonzero(movable)),
        "projection_iterations": int(projection_iterations),
        "projection_final_audit": projection_final_audit,
        "binary_backstop_scale": float(low_scale),
        "fell_back_to_baseline": fell_back_to_baseline,
        "max_ratio": ratio_limit,
        "candidate_audit": candidate_audit,
        "final_audit": final_audit,
    }


def _detail_component_passes(
    record,
    minimum_correlation,
    minimum_rms_retention,
    maximum_rms_retention,
):
    if not record.get("available", True):
        return False
    try:
        correlation = float(record["correlation"])
        retention = float(record["rms_retention"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite(correlation)
        and np.isfinite(retention)
        and float(minimum_correlation) <= correlation
        and float(minimum_rms_retention) <= retention <= float(maximum_rms_retention)
    )


def _guard_face_detail_updates(
    baseline_values,
    candidate_values,
    reference_values,
    face_region_mask,
    *,
    minimum_correlation=0.8,
    minimum_rms_retention=0.6,
    maximum_rms_retention=2.0,
    iterations=12,
):
    """Retain as much feature enhancement as per-face curvature gates allow."""
    baseline = np.asarray(baseline_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    reference = np.asarray(reference_values, dtype=np.float32)
    if baseline.shape != candidate.shape or baseline.shape != reference.shape:
        raise ValueError("Face detail guard surfaces must have identical shapes")

    def measure(values):
        return _face_detail_preservation_metrics(reference, values, face_region_mask)

    def passes(metrics):
        if not metrics.get("available", False):
            return False
        components = metrics.get("components", [])
        return bool(components) and all(
            _detail_component_passes(
                record,
                minimum_correlation,
                minimum_rms_retention,
                maximum_rms_retention,
            )
            for record in components
        )

    baseline_metrics = measure(baseline)
    candidate_metrics = measure(candidate)
    if passes(candidate_metrics):
        return candidate_values, {
            "enabled": True,
            "attenuated": False,
            "applied_scale": 1.0,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "final": candidate_metrics,
        }
    if not passes(baseline_metrics):
        return baseline_values, {
            "enabled": False,
            "reason": "baseline_quality_gate",
            "attenuated": True,
            "applied_scale": 0.0,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "final": baseline_metrics,
        }

    delta = np.where(
        np.isfinite(baseline) & np.isfinite(candidate),
        candidate - baseline,
        0.0,
    )
    low = 0.0
    high = 1.0
    best = baseline.copy()
    best_metrics = baseline_metrics
    for _ in range(max(1, int(iterations))):
        scale = (low + high) * 0.5
        trial = np.where(
            np.isfinite(baseline),
            baseline + scale * delta,
            np.nan,
        ).astype(np.float32, copy=False)
        trial_metrics = measure(trial)
        if passes(trial_metrics):
            low = scale
            best = trial
            best_metrics = trial_metrics
        else:
            high = scale
    return best.astype(candidate.dtype, copy=False), {
        "enabled": True,
        "attenuated": True,
        "applied_scale": float(low),
        "minimum_correlation": float(minimum_correlation),
        "minimum_rms_retention": float(minimum_rms_retention),
        "maximum_rms_retention": float(maximum_rms_retention),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "final": best_metrics,
    }


def _restore_face_laplacian_detail(
    processed_values,
    reference_values,
    face_region_mask,
    max_neighbor_step_mm,
    max_correction_mm=2.0,
):
    """Restore facial detail on the attached scene without restoring a face plate."""
    if face_region_mask is None:
        return processed_values, {"enabled": False, "reason": "no_face_region"}
    processed = np.asarray(processed_values, dtype=np.float32)
    reference = np.asarray(reference_values, dtype=np.float32)
    if processed.shape != reference.shape:
        raise ValueError("Face detail surfaces must have identical shapes")
    face = _resize_binary_mask(face_region_mask, processed.shape)
    valid = face & np.isfinite(processed) & np.isfinite(reference)
    components, component_count = label(valid, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count == 0:
        return processed_values, {"enabled": False, "reason": "empty_face_region"}

    candidate = processed.copy()
    total_weight = np.zeros(processed.shape, dtype=np.float32)
    correction_limits = []
    smoothing_sigmas = []
    applied_components = 0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        rows, cols = np.where(component)
        if rows.size < 64:
            continue
        width = int(cols.max() - cols.min() + 1)
        height = int(rows.max() - rows.min() + 1)
        face_dimension = max(1, min(width, height))
        smoothing_sigma = float(np.clip(face_dimension * 0.075, 2.0, 14.0))
        feather_px = float(np.clip(round(face_dimension * 0.08), 4, 16))
        reference_base = _smooth_nan_aware(reference, sigma=smoothing_sigma)
        processed_base = _smooth_nan_aware(processed, sigma=smoothing_sigma)
        reference_detail = reference - reference_base
        processed_detail = processed - processed_base
        distance = distance_transform_edt(component)
        weight = np.clip(
            (distance - 1.0) / max(feather_px - 1.0, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        weight = weight * weight * (3.0 - 2.0 * weight)
        inner = component & (weight >= 0.5)
        detail_samples = np.abs(reference_detail[inner if np.any(inner) else component])
        robust_detail = float(np.percentile(detail_samples, 99.0))
        correction_limit = min(
            float(max_correction_mm),
            max(0.4, robust_detail * 1.75),
        )
        correction = np.clip(
            reference_detail - processed_detail,
            -correction_limit,
            correction_limit,
        )
        candidate[component] += correction[component] * weight[component]
        total_weight = np.maximum(total_weight, weight)
        correction_limits.append(correction_limit)
        smoothing_sigmas.append(smoothing_sigma)
        applied_components += 1

    if not applied_components:
        return processed_values, {"enabled": False, "reason": "no_usable_face_component"}
    guarded, guard_stats = _guard_weighted_feature_updates(
        processed,
        candidate,
        max_neighbor_step_mm,
        max_ratio=1.25,
    )
    boundary = valid & ~binary_erosion(
        valid,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    guarded = np.asarray(guarded).copy()
    guarded[boundary] = processed[boundary]
    guarded, boundary_guard_stats = _guard_weighted_feature_updates(
        processed,
        guarded,
        max_neighbor_step_mm,
        max_ratio=1.25,
    )
    guarded = np.asarray(guarded).copy()
    guarded[boundary] = processed[boundary]
    applied = np.abs(np.asarray(guarded, dtype=np.float64) - processed)
    weighted = total_weight > 1e-4
    final_audit = _surface_edge_audit(guarded, processed, max_neighbor_step_mm)
    return guarded, {
        "enabled": True,
        "method": "laplacian_face_detail_restoration",
        "components": int(applied_components),
        "weighted_pixels": int(np.count_nonzero(weighted)),
        "smoothing_sigma_px": smoothing_sigmas,
        "correction_limits_mm": correction_limits,
        "applied_correction_p95_mm": float(np.percentile(applied[weighted], 95.0)),
        "applied_correction_max_mm": float(np.max(applied[weighted], initial=0.0)),
        "boundary_correction_max_mm": float(np.max(applied[boundary], initial=0.0)),
        "detail_guard": guard_stats,
        "boundary_guard": boundary_guard_stats,
        "attachment_slope_ratio_p99": final_audit.get("accepted_surface_ratio_p99"),
        "attachment_slope_ratio_max": final_audit.get("accepted_surface_ratio_max"),
    }


def _attach_face_boundary_to_local_surface(
    values,
    face_region_mask,
    max_separation_mm=0.0,
    sample_pitch_mm=0.4,
    feather_width_mm=12.0,
):
    """Blend large face-edge cliffs inward so eyewear stays on the face."""
    if face_region_mask is None or max_separation_mm <= 0:
        return values, {
            "enabled": False,
            "max_separation_mm": float(max(0.0, max_separation_mm)),
        }

    region = _resize_binary_mask(face_region_mask, values.shape)
    valid = np.isfinite(values)
    region &= valid
    if not np.any(region) or not np.any(valid & ~region):
        return values, {"enabled": False, "reason": "no_face_boundary"}

    attached = values.copy()
    components, component_count = label(region)
    pitch = max(float(sample_pitch_mm), 1e-6)
    feather_px = max(3, int(round(float(feather_width_mm) / pitch)))
    smoothing_sigma = max(1.0, min(4.0, feather_px / 6.0))
    adjusted = np.zeros(values.shape, dtype=bool)
    adjustment_magnitudes = []
    attached_components = 0
    boundary_pixels = 0

    for component_index in range(1, component_count + 1):
        component = components == component_index
        if np.count_nonzero(component) < 16:
            continue

        boundary = component & ~binary_erosion(
            component,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        if not np.any(boundary):
            continue

        # For each face pixel, locate the nearest finite sample immediately
        # outside that face. This follows the local scene surface instead of
        # lifting architecture behind the subject into a broad support halo.
        _, outside_indices = distance_transform_edt(
            component | ~valid,
            return_indices=True,
        )
        outside_values = attached[tuple(outside_indices)]
        separation = attached - outside_values
        boundary_shift = np.sign(separation) * np.maximum(
            np.abs(separation) - float(max_separation_mm),
            0.0,
        )
        active_boundary = boundary & (np.abs(boundary_shift) > 0.02)
        if not np.any(active_boundary):
            continue

        boundary_seeds = np.where(boundary, boundary_shift, 0.0)
        _, boundary_indices = distance_transform_edt(~boundary, return_indices=True)
        shift_field = boundary_seeds[tuple(boundary_indices)]
        numerator = gaussian_filter(
            np.where(component, shift_field, 0.0),
            sigma=smoothing_sigma,
        )
        denominator = gaussian_filter(component.astype(np.float32), sigma=smoothing_sigma)
        smooth_shift = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-5,
        )
        smooth_shift[boundary] = boundary_shift[boundary]
        inward_distance = distance_transform_edt(component)
        inward_weight = np.clip(
            1.0 - np.maximum(inward_distance - 1.0, 0.0) / float(feather_px),
            0.0,
            1.0,
        )
        applied_shift = smooth_shift * inward_weight
        component_adjusted = component & (np.abs(applied_shift) > 0.02)
        attached[component] -= applied_shift[component]

        adjusted |= component_adjusted
        adjustment_magnitudes.append(np.abs(applied_shift[component_adjusted]))
        boundary_pixels += int(np.count_nonzero(active_boundary))
        attached_components += 1

    if not adjustment_magnitudes:
        return values, {
            "enabled": False,
            "reason": "face_boundary_within_limit",
            "max_separation_mm": float(max_separation_mm),
        }

    magnitudes = np.concatenate(adjustment_magnitudes)
    return np.where(valid, attached, np.nan), {
        "enabled": True,
        "max_separation_mm": float(max_separation_mm),
        "feather_width_mm": float(feather_width_mm),
        "feather_width_px": int(feather_px),
        "attached_components": int(attached_components),
        "adjusted_boundary_pixels": int(boundary_pixels),
        "adjusted_pixels": int(np.count_nonzero(adjusted)),
        "applied_shift_p95_mm": float(np.percentile(magnitudes, 95.0)),
        "applied_shift_max_mm": float(np.max(magnitudes)),
    }


def _stabilize_face_relief_height(
    values,
    face_region_mask,
    relief_height_mm,
    reference_face_height_mm=10.0,
):
    """Keep facial shape in a stable physical range as scene relief grows."""
    requested_height = max(float(relief_height_mm), 0.0)
    reference_height = max(float(reference_face_height_mm), 1e-6)
    if face_region_mask is None or requested_height <= reference_height:
        return values, {
            "enabled": False,
            "relief_height_mm": requested_height,
            "reference_face_height_mm": reference_height,
            "shape_scale": 1.0,
        }

    region = _resize_binary_mask(face_region_mask, values.shape) & np.isfinite(values)
    if not np.any(region):
        return values, {"enabled": False, "reason": "empty_face_region"}

    stabilized = values.copy()
    components, component_count = label(region)
    shape_scale = min(1.0, reference_height / requested_height)
    applied = []
    stabilized_components = 0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        if np.count_nonzero(component) < 16:
            continue
        component_center = float(np.median(stabilized[component]))
        original = stabilized[component].copy()
        stabilized[component] = component_center + (original - component_center) * shape_scale
        applied.append(np.abs(stabilized[component] - original))
        stabilized_components += 1

    if not applied:
        return values, {"enabled": False, "reason": "no_face_components"}

    magnitudes = np.concatenate(applied)
    return np.where(np.isfinite(values), stabilized, np.nan), {
        "enabled": True,
        "relief_height_mm": requested_height,
        "reference_face_height_mm": reference_height,
        "shape_scale": float(shape_scale),
        "stabilized_components": int(stabilized_components),
        "adjusted_pixels": int(np.count_nonzero(magnitudes > 0.02)),
        "applied_shift_p95_mm": float(np.percentile(magnitudes, 95.0)),
        "applied_shift_max_mm": float(np.max(magnitudes)),
    }


def _expand_face_region_to_depth_connected_head(
    face_region_mask,
    reference_values,
    depth_tolerance_mm=4.0,
):
    """Extend a facial oval over its hair and ears without taking nearby scenery."""
    if face_region_mask is None:
        return None, {"enabled": False, "reason": "no_face_region"}

    face_region = _resize_binary_mask(face_region_mask, reference_values.shape)
    valid = np.isfinite(reference_values)
    face_region &= valid
    components, component_count = label(face_region)
    if component_count == 0:
        return face_region, {"enabled": False, "reason": "empty_face_region"}

    rows, cols = np.indices(face_region.shape, dtype=np.float32)
    head_region = face_region.copy()
    expanded_components = 0
    neck_added_pixels = 0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        component_rows, component_cols = np.where(component)
        if component_rows.size < 16:
            continue

        width = float(component_cols.max() - component_cols.min() + 1)
        height = float(component_rows.max() - component_rows.min() + 1)
        center_col = float(component_cols.min() + component_cols.max()) * 0.5
        center_row = float(component_rows.min() + component_rows.max()) * 0.5 - 0.12 * height
        radius_col = max(2.0, 0.60 * width)
        radius_row = max(2.0, 0.67 * height)
        ellipse = (
            np.square((cols - center_col) / radius_col)
            + np.square((rows - center_row) / radius_row)
            <= 1.0
        )

        # Carry the protected surface through the jaw and into a narrow neck
        # corridor. The geometric corridor prevents a similar-depth backdrop
        # from joining the region, while overlap with the lower face ensures
        # the candidate remains connected to the subject.
        neck_start_row = float(component_rows.min()) + 0.76 * height
        neck_end_row = min(
            float(reference_values.shape[0] - 1),
            float(component_rows.max()) + 0.30 * height,
        )
        neck_progress = np.clip(
            (rows - neck_start_row) / max(neck_end_row - neck_start_row, 1.0),
            0.0,
            1.0,
        )
        neck_radius = width * (0.24 - 0.08 * neck_progress)
        neck_corridor = (
            (rows >= neck_start_row)
            & (rows <= neck_end_row)
            & (np.abs(cols - center_col) <= neck_radius)
        )

        component_level = float(np.median(reference_values[component]))
        head_candidate = (
            ellipse
            & valid
            & (np.abs(reference_values - component_level) <= float(depth_tolerance_mm))
        )
        neck_depth_tolerance_mm = min(float(depth_tolerance_mm), 2.0)
        neck_candidate = (
            neck_corridor
            & valid
            & (np.abs(reference_values - component_level) <= neck_depth_tolerance_mm)
        )
        candidate = head_candidate | neck_candidate
        candidate |= component
        candidate_labels, _ = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
        connected_ids = np.unique(candidate_labels[component])
        connected_ids = connected_ids[connected_ids > 0]
        if connected_ids.size:
            connected_region = np.isin(candidate_labels, connected_ids)
            neck_added_pixels += int(
                np.count_nonzero(connected_region & neck_corridor & ~head_region)
            )
            head_region |= connected_region
            expanded_components += 1

    added = head_region & ~face_region
    return head_region, {
        "enabled": bool(np.any(added)),
        "expanded_components": int(expanded_components),
        "face_pixels": int(np.count_nonzero(face_region)),
        "head_pixels": int(np.count_nonzero(head_region)),
        "added_pixels": int(np.count_nonzero(added)),
        "neck_added_pixels": int(neck_added_pixels),
        "neck_depth_tolerance_mm": float(min(float(depth_tolerance_mm), 2.0)),
        "depth_tolerance_mm": float(depth_tolerance_mm),
    }


def _face_translation_core_mask(face_region_mask, target_shape, margin_ratio=0.06):
    """Keep inner facial form rigid while allowing the oval and jaw to attach."""
    if face_region_mask is None:
        return None, {"enabled": False, "reason": "no_face_region"}
    region = _resize_binary_mask(face_region_mask, target_shape)
    if np.count_nonzero(region) < 16:
        return region, {"enabled": False, "reason": "empty_face_region"}
    components, component_count = label(region, structure=np.ones((3, 3), dtype=np.uint8))
    core = np.zeros(region.shape, dtype=bool)
    component_margins = []
    components_with_core = 0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        rows, cols = np.where(component)
        if rows.size < 16:
            continue
        face_width = int(cols.max() - cols.min() + 1)
        face_height = int(rows.max() - rows.min() + 1)
        margin_px = int(
            np.clip(
                round(min(face_width, face_height) * max(float(margin_ratio), 0.0)),
                3,
                12,
            )
        )
        distance = distance_transform_edt(component)
        component_core = component & (distance >= float(margin_px))
        minimum_core_pixels = min(64, max(8, int(round(rows.size * 0.08))))
        if np.count_nonzero(component_core) < minimum_core_pixels:
            component_core = component & (distance >= 2.0)
            margin_px = 2
        if np.any(component_core):
            core |= component_core
            components_with_core += 1
            component_margins.append(int(margin_px))
    return core, {
        "enabled": bool(components_with_core == component_count),
        "face_pixels": int(np.count_nonzero(region)),
        "core_pixels": int(np.count_nonzero(core)),
        "component_count": int(component_count),
        "components_with_core": int(components_with_core),
        "component_margins_px": component_margins,
        "margin_px": int(max(component_margins, default=0)),
        "margin_ratio": float(margin_ratio),
    }


def _align_stabilized_head_to_reference_boundary(
    stabilized_values,
    reference_values,
    processed_values,
    head_region_mask,
    transition_width_px=10.0,
    protected_core_mask=None,
    protected_face_mask=None,
    soft_core_screening_weight=0.25,
    max_neighbor_step_mm=None,
):
    """Attach a low-height face to the scene without warping its gradients."""
    if head_region_mask is None:
        return processed_values, {"enabled": False, "reason": "no_head_region"}
    head_region = _resize_binary_mask(head_region_mask, processed_values.shape)
    protected_core = (
        _resize_binary_mask(protected_core_mask, processed_values.shape)
        if protected_core_mask is not None
        else None
    )
    protected_face = (
        _resize_binary_mask(protected_face_mask, processed_values.shape)
        if protected_face_mask is not None
        else None
    )
    aligned, stats = align_face_to_scene_gradient_domain(
        stabilized_values,
        reference_values,
        processed_values,
        head_region,
        screening_length_px=max(50.0, float(transition_width_px) * 10.0),
        protected_core_mask=protected_core,
        protected_face_mask=protected_face,
        soft_core_screening_weight=soft_core_screening_weight,
        max_neighbor_step_mm=max_neighbor_step_mm,
    )
    if stats.get("enabled", False):
        stats["transition_width_px"] = float(transition_width_px)
    return aligned, stats


def _restore_stabilized_face_surface(values, protected_values, face_region_mask, enabled=False):
    """Restore a scale-stabilized face after global scene slope limiting."""
    if not enabled or face_region_mask is None:
        return values, {"enabled": False}

    region = _resize_binary_mask(face_region_mask, values.shape)
    restorable = region & np.isfinite(values) & np.isfinite(protected_values)
    if not np.any(restorable):
        return values, {"enabled": False, "reason": "empty_face_region"}

    restored = values.copy()
    change = np.abs(protected_values[restorable] - values[restorable])
    restored[restorable] = protected_values[restorable]
    return restored, {
        "enabled": True,
        "restored_pixels": int(np.count_nonzero(change > 0.02)),
        "restored_shift_p95_mm": float(np.percentile(change, 95.0)),
        "restored_shift_max_mm": float(np.max(change)),
    }


def _resolve_relief_value_transform(npy_file, value_transform):
    if value_transform != "auto":
        if value_transform not in RELIEF_VALUE_TRANSFORMS:
            raise ValueError(
                f"Unsupported relief value transform: {value_transform}. "
                f"Expected 'auto' or one of {sorted(RELIEF_VALUE_TRANSFORMS)}."
            )
        return value_transform

    import json

    metadata_path = os.path.join(os.path.dirname(os.fspath(npy_file)), "output_depth_metadata.json")
    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        detected = metadata.get("relief_value_transform")
        if detected in RELIEF_VALUE_TRANSFORMS:
            return detected
        return relief_value_transform_for_model(metadata.get("effective_model"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass
    return RELIEF_VALUE_TRANSFORM_LINEAR


def _flatten_border(values, border_px, preserve_mask=None):
    border_px = int(border_px or 0)
    if border_px <= 0:
        return values
    border_px = min(border_px, values.shape[0] // 2, values.shape[1] // 2)
    if border_px <= 0:
        return values
    values = values.copy()
    border = np.zeros(values.shape, dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    if preserve_mask is not None:
        preserve = _resize_binary_mask(preserve_mask, values.shape)
        border &= ~preserve
    values[border] = 0.0
    return values


def _physical_sample_pitch_mm(shape, max_xy_size):
    try:
        physical_size = float(max_xy_size)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(physical_size) or physical_size <= 0:
        return None
    coordinate_max = max(int(shape[0]) - 1, int(shape[1]) - 1)
    if coordinate_max <= 0:
        return None
    return physical_size / float(coordinate_max)


def _prepare_relief_for_printing(values, max_xy_size, minimum_feature_mm):
    input_shape = tuple(int(value) for value in values.shape)
    input_pitch_mm = _physical_sample_pitch_mm(input_shape, max_xy_size)
    stats = {
        "enabled": False,
        "input_shape": list(input_shape),
        "mesh_grid_shape": list(input_shape),
        "input_sample_pitch_mm": input_pitch_mm,
        "mesh_sample_pitch_mm": input_pitch_mm,
        "minimum_feature_mm": None,
        "filter_sigma_px": 0.0,
        "resampled": False,
    }

    try:
        feature_mm = float(minimum_feature_mm)
    except (TypeError, ValueError):
        return values, stats
    if input_pitch_mm is None or not np.isfinite(feature_mm) or feature_mm <= 0:
        return values, stats

    stats["enabled"] = True
    stats["minimum_feature_mm"] = feature_mm
    feature_width_px = feature_mm / input_pitch_mm
    filter_sigma_px = max(0.0, feature_width_px / 4.0)
    filtered = values
    if filter_sigma_px >= 0.35:
        filtered = _smooth_nan_aware(filtered, sigma=filter_sigma_px)
        stats["filter_sigma_px"] = float(filter_sigma_px)

    # Two samples across the smallest printable feature retain its shape while
    # avoiding triangles far denser than the nozzle can reproduce.
    target_pitch_mm = feature_mm / 2.0
    max_dimension = max(input_shape)
    printable_dimension = max(2, int(np.floor(float(max_xy_size) / target_pitch_mm)) + 1)
    output_dimension = min(max_dimension, printable_dimension)
    target_shape = _target_shape_for_max_dimension(input_shape, output_dimension)
    if target_shape != input_shape:
        filtered = _resize_nan_aware(filtered, target_shape)
        stats["resampled"] = True

    stats["mesh_grid_shape"] = [int(filtered.shape[0]), int(filtered.shape[1])]
    stats["mesh_sample_pitch_mm"] = _physical_sample_pitch_mm(filtered.shape, max_xy_size)
    return filtered, stats


def _structural_relief_edge_barriers(values, max_step, structural_region_mask=None):
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    finite = values[valid]
    empty_horizontal = np.zeros((values.shape[0], max(values.shape[1] - 1, 0)), dtype=bool)
    empty_vertical = np.zeros((max(values.shape[0] - 1, 0), values.shape[1]), dtype=bool)
    if not finite.size or min(values.shape) < 3:
        return empty_horizontal, empty_vertical, {
            "edge_preservation_threshold_mm": None,
            "minimum_structural_edge_pixels": 0,
            "preserved_structural_edge_pairs": 0,
            "suppressed_internal_region_edge_pairs": 0,
            "preserved_region_boundary_pairs": 0,
            "suppressed_excessive_structural_pairs": 0,
            "suppressed_excessive_region_boundary_pairs": 0,
            "structural_jump_limit_mm": None,
            "region_boundary_jump_limit_mm": None,
        }

    height_range = float(np.max(finite) - np.min(finite))
    edge_threshold = max(float(max_step) * 4.0, height_range * 0.08)
    strong_horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (np.abs(values[:, 1:] - values[:, :-1]) >= edge_threshold)
    )
    strong_vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (np.abs(values[1:, :] - values[:-1, :]) >= edge_threshold)
    )
    edge_pixels = np.zeros(values.shape, dtype=bool)
    edge_pixels[:, :-1] |= strong_horizontal
    edge_pixels[:, 1:] |= strong_horizontal
    edge_pixels[:-1, :] |= strong_vertical
    edge_pixels[1:, :] |= strong_vertical

    minimum_edge_pixels = max(64, int(round(min(values.shape) * 0.10)))
    components, component_count = label(edge_pixels, structure=np.ones((3, 3), dtype=np.uint8))
    if component_count:
        counts = np.bincount(components.ravel())
        keep = counts >= minimum_edge_pixels
        keep[0] = False
        structural_pixels = keep[components]
    else:
        structural_pixels = np.zeros(values.shape, dtype=bool)

    horizontal_barrier = strong_horizontal & structural_pixels[:, :-1] & structural_pixels[:, 1:]
    vertical_barrier = strong_vertical & structural_pixels[:-1, :] & structural_pixels[1:, :]
    horizontal_jump = np.abs(values[:, 1:] - values[:, :-1])
    vertical_jump = np.abs(values[1:, :] - values[:-1, :])
    structural_jump_limit = max(float(max_step) * 8.0, 6.0)
    excessive_horizontal = horizontal_jump > structural_jump_limit
    excessive_vertical = vertical_jump > structural_jump_limit
    suppressed_excessive_pairs = int(
        np.count_nonzero(horizontal_barrier & excessive_horizontal)
        + np.count_nonzero(vertical_barrier & excessive_vertical)
    )
    horizontal_barrier &= ~excessive_horizontal
    vertical_barrier &= ~excessive_vertical
    suppressed_internal_pairs = 0
    preserved_region_boundary_pairs = 0
    suppressed_excessive_region_pairs = 0
    region_boundary_jump_limit = max(float(max_step) * 4.0, 3.0)
    if structural_region_mask is not None:
        region = _resize_binary_mask(structural_region_mask, values.shape)
        internal_horizontal = region[:, :-1] & region[:, 1:]
        internal_vertical = region[:-1, :] & region[1:, :]
        suppressed_internal_pairs = int(
            np.count_nonzero(horizontal_barrier & internal_horizontal)
            + np.count_nonzero(vertical_barrier & internal_vertical)
        )
        horizontal_barrier &= ~internal_horizontal
        vertical_barrier &= ~internal_vertical

        region_boundary_horizontal = region[:, :-1] ^ region[:, 1:]
        region_boundary_vertical = region[:-1, :] ^ region[1:, :]
        # Region masks identify faces and their immediate features. Remove the
        # generic structural barrier there, then retain only physically modest
        # boundary jumps. This keeps a crisp outline at normal relief while
        # preventing tall settings from turning faces into vertical cliffs.
        horizontal_barrier &= ~region_boundary_horizontal
        vertical_barrier &= ~region_boundary_vertical
        region_candidate_horizontal = strong_horizontal & region_boundary_horizontal
        region_candidate_vertical = strong_vertical & region_boundary_vertical
        preserve_horizontal = region_candidate_horizontal & (horizontal_jump <= region_boundary_jump_limit)
        preserve_vertical = region_candidate_vertical & (vertical_jump <= region_boundary_jump_limit)
        suppressed_excessive_region_pairs = int(
            np.count_nonzero(region_candidate_horizontal & ~preserve_horizontal)
            + np.count_nonzero(region_candidate_vertical & ~preserve_vertical)
        )
        preserved_region_boundary_pairs = int(
            np.count_nonzero(preserve_horizontal) + np.count_nonzero(preserve_vertical)
        )
        horizontal_barrier |= preserve_horizontal
        vertical_barrier |= preserve_vertical
    return horizontal_barrier, vertical_barrier, {
        "edge_preservation_threshold_mm": edge_threshold,
        "minimum_structural_edge_pixels": minimum_edge_pixels,
        "preserved_structural_edge_pairs": int(
            np.count_nonzero(horizontal_barrier) + np.count_nonzero(vertical_barrier)
        ),
        "suppressed_internal_region_edge_pairs": suppressed_internal_pairs,
        "preserved_region_boundary_pairs": preserved_region_boundary_pairs,
        "suppressed_excessive_structural_pairs": suppressed_excessive_pairs,
        "suppressed_excessive_region_boundary_pairs": suppressed_excessive_region_pairs,
        "structural_jump_limit_mm": structural_jump_limit,
        "region_boundary_jump_limit_mm": region_boundary_jump_limit,
    }


def _limit_positive_relief_slope(
    values,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    structural_region_mask=None,
):
    stats = {
        "enabled": False,
        "max_slope_mm_per_mm": None,
        "max_neighbor_step_mm": None,
        "iterations": 0,
    }
    try:
        pitch_mm = float(sample_pitch_mm)
        max_slope = float(max_slope_mm_per_mm)
    except (TypeError, ValueError):
        return values, stats
    if not np.isfinite(pitch_mm) or pitch_mm <= 0 or not np.isfinite(max_slope) or max_slope <= 0:
        return values, stats

    valid = np.isfinite(values)
    if not np.any(valid):
        return values, stats

    max_step = pitch_mm * max_slope
    stats.update(
        {
            "enabled": True,
            "max_slope_mm_per_mm": max_slope,
            "max_neighbor_step_mm": max_step,
        }
    )
    horizontal_barrier, vertical_barrier, edge_stats = _structural_relief_edge_barriers(
        values,
        max_step,
        structural_region_mask=structural_region_mask,
    )
    stats.update(edge_stats)
    work = np.where(valid, values, np.inf).astype(np.float32, copy=True)
    finite_values = work[valid]
    height_range = float(np.max(finite_values) - np.min(finite_values))
    max_iterations = min(512, max(1, int(np.ceil(height_range / max_step)) + 1))
    cross = np.array([[False, True, False], [True, True, True], [False, True, False]])

    for iteration in range(max_iterations):
        if stats["preserved_structural_edge_pairs"]:
            left = np.full_like(work, np.inf)
            right = np.full_like(work, np.inf)
            up = np.full_like(work, np.inf)
            down = np.full_like(work, np.inf)
            left[:, 1:] = np.where(horizontal_barrier, np.inf, work[:, :-1])
            right[:, :-1] = np.where(horizontal_barrier, np.inf, work[:, 1:])
            up[1:, :] = np.where(vertical_barrier, np.inf, work[:-1, :])
            down[:-1, :] = np.where(vertical_barrier, np.inf, work[1:, :])
            neighborhood_min = np.minimum.reduce((work, left, right, up, down))
        else:
            neighborhood_min = minimum_filter(work, footprint=cross, mode="constant", cval=np.inf)
        updated = np.minimum(work, neighborhood_min + max_step)
        max_change = float(np.max(work[valid] - updated[valid]))
        work = updated
        stats["iterations"] = iteration + 1
        if max_change <= 1e-5:
            break

    work[~valid] = np.nan
    return work, stats


def _face_detail_preservation_metrics(reference_values, candidate_values, face_region_mask):
    """Measure curvature retention globally and per disconnected face."""
    stats = {
        "available": False,
        "correlation": None,
        "rms_retention": None,
        "samples": 0,
        "components": [],
        "component_count": 0,
        "measured_component_count": 0,
        "unavailable_component_count": 0,
        "flat_reference_violation": False,
    }
    if face_region_mask is None:
        return stats

    reference = np.asarray(reference_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError("Face detail metric surfaces must have identical shapes")
    valid = np.isfinite(reference) & np.isfinite(candidate)
    region = _resize_binary_mask(face_region_mask, reference.shape) & valid
    components, component_count = label(
        region,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    reference_curvature = laplace(np.where(valid, reference, 0.0)).astype(np.float64)
    candidate_curvature = laplace(np.where(valid, candidate, 0.0)).astype(np.float64)

    component_records = []
    measured_records = []
    aggregate_mask = np.zeros(reference.shape, dtype=bool)
    for component_index in range(1, component_count + 1):
        component = components == component_index
        metric_region = binary_erosion(
            component,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        metric_region &= binary_erosion(
            valid,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        samples = int(np.count_nonzero(metric_region))
        if samples < 16:
            component_records.append(
                {
                    "component": int(component_index),
                    "samples": samples,
                    "available": False,
                    "reason": "insufficient_interior_samples",
                    "correlation": None,
                    "rms_retention": None,
                    "flat_reference_violation": False,
                }
            )
            continue
        aggregate_mask |= metric_region
        source_samples = reference_curvature[metric_region].copy()
        output_samples = candidate_curvature[metric_region].copy()
        source_samples -= np.mean(source_samples)
        output_samples -= np.mean(output_samples)
        source_rms = float(np.sqrt(np.mean(np.square(source_samples))))
        output_rms = float(np.sqrt(np.mean(np.square(output_samples))))
        flat_reference_violation = False
        if source_rms <= 1e-10:
            correlation = 1.0 if output_rms <= 1e-8 else 0.0
            retention = 1.0 if output_rms <= 1e-8 else None
            flat_reference_violation = output_rms > 1e-8
        else:
            correlation = float(
                np.dot(source_samples, output_samples)
                / max(
                    np.linalg.norm(source_samples) * np.linalg.norm(output_samples),
                    1e-12,
                )
            )
            retention = output_rms / source_rms
        record = {
            "component": int(component_index),
            "samples": samples,
            "available": True,
            "correlation": correlation,
            "rms_retention": float(retention) if retention is not None else None,
            "flat_reference_violation": bool(flat_reference_violation),
        }
        component_records.append(record)
        measured_records.append(record)

    if not measured_records:
        stats.update(
            {
                "components": component_records,
                "component_count": int(component_count),
                "unavailable_component_count": int(len(component_records)),
            }
        )
        return stats

    source_samples = reference_curvature[aggregate_mask].copy()
    output_samples = candidate_curvature[aggregate_mask].copy()
    source_samples -= np.mean(source_samples)
    output_samples -= np.mean(output_samples)
    source_rms = float(np.sqrt(np.mean(np.square(source_samples))))
    output_rms = float(np.sqrt(np.mean(np.square(output_samples))))
    aggregate_flat_reference_violation = False
    if source_rms <= 1e-10:
        correlation = 1.0 if output_rms <= 1e-8 else 0.0
        retention = 1.0 if output_rms <= 1e-8 else None
        aggregate_flat_reference_violation = output_rms > 1e-8
    else:
        correlation = float(
            np.dot(source_samples, output_samples)
            / max(
                np.linalg.norm(source_samples) * np.linalg.norm(output_samples),
                1e-12,
            )
        )
        retention = output_rms / source_rms
    finite_retentions = [
        float(record["rms_retention"])
        for record in measured_records
        if record["rms_retention"] is not None
        and np.isfinite(float(record["rms_retention"]))
    ]
    unavailable_count = len(component_records) - len(measured_records)
    return {
        "available": unavailable_count == 0,
        "correlation": correlation,
        "rms_retention": float(retention) if retention is not None else None,
        "samples": int(np.count_nonzero(aggregate_mask)),
        "components": component_records,
        "component_count": int(component_count),
        "measured_component_count": int(len(measured_records)),
        "unavailable_component_count": int(unavailable_count),
        "flat_reference_violation": bool(
            aggregate_flat_reference_violation
            or any(record["flat_reference_violation"] for record in measured_records)
        ),
        "minimum_component_correlation": float(
            min(record["correlation"] for record in measured_records)
        ),
        "minimum_component_rms_retention": (
            float(min(finite_retentions)) if finite_retentions else None
        ),
        "maximum_component_rms_retention": (
            float(max(finite_retentions)) if finite_retentions else None
        ),
    }


def _cap_selection_background_relief(
    values,
    selection_mask,
    *,
    relief_height_mm,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    background_depth_ratio,
):
    """Enforce a physical far-background budget without clipping the support ramp."""
    stats = {"enabled": False, "affected_pixels": 0}
    if selection_mask is None:
        stats["reason"] = "no_selection_mask"
        return values, stats
    try:
        height_mm = float(relief_height_mm)
        pitch_mm = float(sample_pitch_mm)
        max_slope = float(max_slope_mm_per_mm)
        background_ratio = float(background_depth_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid physical background cap controls") from exc
    if (
        not np.isfinite(height_mm)
        or height_mm <= 0
        or not np.isfinite(pitch_mm)
        or pitch_mm <= 0
        or not np.isfinite(max_slope)
        or max_slope <= 0
        or not np.isfinite(background_ratio)
        or not 0 <= background_ratio <= 1
    ):
        raise ValueError("Physical background cap controls are outside their supported range")

    source = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(source)
    selected = _resize_binary_mask(selection_mask, source.shape) & valid
    background = valid & ~selected
    if np.count_nonzero(selected) < 4 or np.count_nonzero(background) < 4:
        stats["reason"] = "insufficient_selection_or_background"
        return values, stats

    base_height = float(np.min(source[valid]))
    far_background_ceiling = base_height + background_ratio * height_mm
    outside_distance, nearest = distance_transform_edt(
        ~selected,
        return_distances=True,
        return_indices=True,
    )
    nearest_selected_height = source[tuple(nearest)]
    support_delta = outside_distance * pitch_mm * max_slope
    support_floor = nearest_selected_height - support_delta
    support_upper = nearest_selected_height + support_delta
    physical_ceiling = np.minimum(
        support_upper,
        np.maximum(far_background_ceiling, support_floor),
    )
    capped = source.copy()
    before = source[background]
    capped[background] = np.clip(
        before,
        support_floor[background],
        physical_ceiling[background],
    )
    boundary_floor = np.full(source.shape, -np.inf, dtype=np.float32)
    boundary_ceiling = np.full(source.shape, np.inf, dtype=np.float32)
    boundary_touched = np.zeros(source.shape, dtype=bool)
    max_step = pitch_mm * max_slope
    for background_slice, selected_slice in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        touches_selected = background[background_slice] & selected[selected_slice]
        local_floor = boundary_floor[background_slice]
        local_ceiling = boundary_ceiling[background_slice]
        local_touched = boundary_touched[background_slice]
        selected_values = source[selected_slice]
        local_floor[touches_selected] = np.maximum(
            local_floor[touches_selected],
            selected_values[touches_selected] - max_step,
        )
        local_ceiling[touches_selected] = np.minimum(
            local_ceiling[touches_selected],
            selected_values[touches_selected] + max_step,
        )
        local_touched[touches_selected] = True
    feasible_boundary = boundary_touched & (boundary_floor <= boundary_ceiling)
    capped[feasible_boundary] = np.clip(
        capped[feasible_boundary],
        boundary_floor[feasible_boundary],
        boundary_ceiling[feasible_boundary],
    )
    conflicting_boundary = boundary_touched & ~feasible_boundary
    capped[conflicting_boundary] = (
        boundary_floor[conflicting_boundary] + boundary_ceiling[conflicting_boundary]
    ) * 0.5
    signed_correction = capped[background] - before
    absolute_correction = np.abs(signed_correction)
    far_background = (
        background
        & (support_floor <= far_background_ceiling)
        & (support_upper >= far_background_ceiling)
        & ~boundary_touched
    )
    far_values = capped[far_background]
    far_violation = (
        float(max(0.0, np.max(far_values) - far_background_ceiling))
        if far_values.size
        else 0.0
    )
    horizontal_boundary = (
        (selected[:, :-1] ^ selected[:, 1:])
        & valid[:, :-1]
        & valid[:, 1:]
    )
    vertical_boundary = (
        (selected[:-1, :] ^ selected[1:, :])
        & valid[:-1, :]
        & valid[1:, :]
    )
    attachment_jumps = np.concatenate(
        (
            np.abs(capped[:, 1:] - capped[:, :-1])[horizontal_boundary],
            np.abs(capped[1:, :] - capped[:-1, :])[vertical_boundary],
        )
    ).astype(np.float64)
    attachment_step_limit = pitch_mm * max_slope
    attachment_jump_p99 = (
        float(np.percentile(attachment_jumps, 99.0))
        if attachment_jumps.size
        else 0.0
    )
    attachment_jump_max = (
        float(np.max(attachment_jumps)) if attachment_jumps.size else 0.0
    )
    attachment_violation = max(0.0, attachment_jump_max - attachment_step_limit)
    feasible_horizontal_boundary = horizontal_boundary & ~(
        conflicting_boundary[:, :-1] | conflicting_boundary[:, 1:]
    )
    feasible_vertical_boundary = vertical_boundary & ~(
        conflicting_boundary[:-1, :] | conflicting_boundary[1:, :]
    )
    feasible_attachment_jumps = np.concatenate(
        (
            np.abs(capped[:, 1:] - capped[:, :-1])[feasible_horizontal_boundary],
            np.abs(capped[1:, :] - capped[:-1, :])[feasible_vertical_boundary],
        )
    ).astype(np.float64)
    feasible_attachment_jump_max = (
        float(np.max(feasible_attachment_jumps))
        if feasible_attachment_jumps.size
        else 0.0
    )
    feasible_attachment_violation = max(
        0.0,
        feasible_attachment_jump_max - attachment_step_limit,
    )
    has_attachment_conflicts = bool(np.any(conflicting_boundary))
    far_background_cap_passed = far_violation <= 1e-5
    feasible_attachment_passed = feasible_attachment_violation <= 1e-5
    attachment_constraints_passed = (
        not has_attachment_conflicts and attachment_violation <= 1e-5
    )
    stats.update(
        {
            "enabled": True,
            "background_depth_ratio": background_ratio,
            "base_height_mm": base_height,
            "far_background_ceiling_mm": float(far_background_ceiling),
            "support_slope_mm_per_mm": max_slope,
            "sample_pitch_mm": pitch_mm,
            "background_pixels": int(np.count_nonzero(background)),
            "far_background_pixels": int(np.count_nonzero(far_background)),
            "affected_pixels": int(np.count_nonzero(absolute_correction > 1e-5)),
            "correction_p95_mm": (
                float(np.percentile(absolute_correction[absolute_correction > 1e-5], 95.0))
                if np.any(absolute_correction > 1e-5)
                else 0.0
            ),
            "correction_max_mm": (
                float(np.max(absolute_correction)) if absolute_correction.size else 0.0
            ),
            "upward_correction_pixels": int(np.count_nonzero(signed_correction > 1e-5)),
            "downward_correction_pixels": int(np.count_nonzero(signed_correction < -1e-5)),
            "far_background_max_mm": float(np.max(far_values)) if far_values.size else None,
            "far_background_cap_violation_mm": far_violation,
            "attachment_step_limit_mm": float(attachment_step_limit),
            "attachment_jump_p99_mm": attachment_jump_p99,
            "attachment_jump_max_mm": attachment_jump_max,
            "attachment_slope_violation_mm": float(attachment_violation),
            "feasible_attachment_jump_max_mm": feasible_attachment_jump_max,
            "feasible_attachment_slope_violation_mm": float(
                feasible_attachment_violation
            ),
            "attachment_constraint_conflicts": int(np.count_nonzero(conflicting_boundary)),
            "attachment_constraint_conflict_max_mm": (
                float(np.max(boundary_floor[conflicting_boundary] - boundary_ceiling[conflicting_boundary]))
                if np.any(conflicting_boundary)
                else 0.0
            ),
            "far_background_cap_passed": far_background_cap_passed,
            "feasible_attachment_constraints_passed": feasible_attachment_passed,
            "attachment_constraints_passed": attachment_constraints_passed,
            "emission_passed": far_background_cap_passed and feasible_attachment_passed,
            "passed": far_background_cap_passed and attachment_constraints_passed,
        }
    )
    return capped, stats


def _relax_selection_attachment_conflicts(
    values,
    selection_mask,
    *,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    feather_pixels=1,
):
    """Resolve concave one-pixel attachment conflicts without broad smoothing."""
    source = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(source)
    selected = _resize_binary_mask(selection_mask, source.shape) & valid
    background = valid & ~selected
    try:
        max_step = float(sample_pitch_mm) * float(max_slope_mm_per_mm)
    except (TypeError, ValueError):
        max_step = 0.0
    stats = {
        "enabled": False,
        "method": "localized_concave_attachment_projection",
        "feather_pixels": int(max(0, feather_pixels)),
        "max_step_mm": float(max_step),
    }
    if (
        not np.isfinite(max_step)
        or max_step <= 0.0
        or np.count_nonzero(selected) < 4
        or np.count_nonzero(background) < 4
    ):
        stats["reason"] = "insufficient_geometry_or_invalid_step"
        return values, stats

    neighbor_min = np.full(source.shape, np.inf, dtype=np.float32)
    neighbor_max = np.full(source.shape, -np.inf, dtype=np.float32)
    neighbor_count = np.zeros(source.shape, dtype=np.int16)
    boundary_views = (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    )
    for background_slice, selected_slice in boundary_views:
        touches = background[background_slice] & selected[selected_slice]
        local_min = neighbor_min[background_slice]
        local_max = neighbor_max[background_slice]
        local_count = neighbor_count[background_slice]
        selected_values = source[selected_slice]
        local_min[touches] = np.minimum(
            local_min[touches],
            selected_values[touches],
        )
        local_max[touches] = np.maximum(
            local_max[touches],
            selected_values[touches],
        )
        local_count[touches] += 1

    conflicting = (
        background
        & (neighbor_count >= 2)
        & ((neighbor_max - neighbor_min) > 2.0 * max_step + 1e-6)
    )
    conflict_count = int(np.count_nonzero(conflicting))
    stats["input_conflict_pixels"] = conflict_count
    if not conflict_count:
        stats["reason"] = "no_attachment_conflicts"
        return values, stats

    correction_sum = np.zeros(source.shape, dtype=np.float64)
    correction_count = np.zeros(source.shape, dtype=np.int16)
    for background_slice, selected_slice in boundary_views:
        touches = conflicting[background_slice] & selected[selected_slice]
        if not np.any(touches):
            continue
        background_values = source[background_slice]
        selected_values = source[selected_slice]
        target = background_values + np.clip(
            selected_values - background_values,
            -max_step,
            max_step,
        )
        local_sum = correction_sum[selected_slice]
        local_count = correction_count[selected_slice]
        local_sum[touches] += (target - selected_values)[touches]
        local_count[touches] += 1

    seeds = correction_count > 0
    seed_correction = np.zeros(source.shape, dtype=np.float64)
    seed_correction[seeds] = (
        correction_sum[seeds] / correction_count[seeds]
    )
    feather = int(max(0, feather_pixels))
    if feather:
        distance, nearest = distance_transform_edt(~seeds, return_indices=True)
        correction_field = seed_correction[tuple(nearest)]
        weight = np.clip(1.0 - distance / float(feather + 1), 0.0, 1.0)
        correction = correction_field * weight * selected
    else:
        correction = seed_correction

    relaxed = source.astype(np.float64, copy=True)
    relaxed[selected] += correction[selected]
    absolute = np.abs(correction[selected])
    changed = selected & (np.abs(correction) > 1e-6)
    stats.update(
        {
            "enabled": True,
            "seed_pixels": int(np.count_nonzero(seeds)),
            "adjusted_pixels": int(np.count_nonzero(changed)),
            "correction_p95_mm": float(np.percentile(absolute, 95.0)),
            "correction_max_mm": float(np.max(absolute, initial=0.0)),
        }
    )
    return relaxed.astype(source.dtype, copy=False), stats


def _restore_background_from_reference(
    candidate_values,
    reference_values,
    foreground_mask,
    *,
    sample_pitch_mm,
    feather_mm=1.5,
    max_neighbor_step_mm=None,
    reference_is_accepted=False,
):
    """Restore rejected background geometry while feathering its subject attachment."""
    candidate = np.asarray(candidate_values, dtype=np.float32)
    reference = np.asarray(reference_values, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError("Background fallback surfaces must have identical shapes")
    foreground = _resize_binary_mask(foreground_mask, candidate.shape)
    valid = np.isfinite(candidate) & np.isfinite(reference)
    background = valid & ~foreground
    try:
        pitch_mm = float(sample_pitch_mm)
        feather_distance_mm = float(feather_mm)
    except (TypeError, ValueError):
        pitch_mm = 0.0
        feather_distance_mm = 0.0
    feather_px = (
        max(1.0, feather_distance_mm / pitch_mm)
        if np.isfinite(pitch_mm) and pitch_mm > 0 and feather_distance_mm > 0
        else 1.0
    )
    distance = distance_transform_edt(~foreground)
    weight = np.clip(distance / feather_px, 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)
    restored = candidate.copy()
    restored[background] = (
        candidate[background] * (1.0 - weight[background])
        + reference[background] * weight[background]
    )
    slope_guard_stats = {
        "enabled": False,
        "reason": "not_requested",
    }
    try:
        max_step = float(max_neighbor_step_mm)
    except (TypeError, ValueError):
        max_step = 0.0
    if np.isfinite(max_step) and max_step > 0:
        guard_baseline = (
            np.where(background, reference, candidate)
            if bool(reference_is_accepted)
            else candidate
        )
        restored, slope_guard_stats = _guard_weighted_feature_updates(
            guard_baseline,
            restored,
            max_neighbor_step_mm=max_step,
            max_ratio=1.0,
        )
    return restored, {
        "enabled": True,
        "background_pixels": int(np.count_nonzero(background)),
        "feather_mm": float(max(feather_distance_mm, 0.0)),
        "feather_px": float(feather_px),
        "restored_pixels": int(
            np.count_nonzero(background & (np.abs(restored - candidate) > 1e-5))
        ),
        "correction_max_mm": (
            float(np.max(np.abs(restored[background] - candidate[background])))
            if np.any(background)
            else 0.0
        ),
        "slope_guard_baseline": (
            "accepted_reference" if bool(reference_is_accepted) else "candidate"
        ),
        "slope_guard": slope_guard_stats,
    }


def _localized_background_structure_metrics(
    reference,
    candidate,
    context,
    *,
    sample_pitch_mm,
    window_mm=6.0,
    minimum_correlation=0.35,
    maximum_shape_error_ratio=0.95,
):
    """Detect spatially localized context loss hidden by aggregate scores."""
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    context = np.asarray(context, dtype=bool)
    try:
        pitch_mm = float(sample_pitch_mm)
    except (TypeError, ValueError):
        pitch_mm = 0.0
    window_px = (
        int(np.clip(round(float(window_mm) / pitch_mm), 9, 31))
        if np.isfinite(pitch_mm) and pitch_mm > 0
        else 15
    )
    if window_px % 2 == 0:
        window_px += 1
    stride_px = max(4, window_px // 2)
    border_exclusion_px = max(2, window_px // 2)
    minimum_window_coverage_ratio = 0.5
    minimum_samples = max(
        16,
        int(np.ceil(window_px * window_px * minimum_window_coverage_ratio)),
    )
    source_values = reference[context].astype(np.float64)
    source_rms = (
        float(np.std(source_values)) if source_values.size else 0.0
    )
    minimum_structure_rms = max(0.05, source_rms * 0.08)

    def _starts(length):
        first = min(border_exclusion_px, max(0, (int(length) - window_px) // 2))
        last = max(first, int(length) - border_exclusion_px - window_px)
        starts = list(range(first, last + 1, stride_px))
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    records = []
    measured_windows = 0
    for row_start in _starts(reference.shape[0]):
        row_slice = slice(row_start, min(row_start + window_px, reference.shape[0]))
        for col_start in _starts(reference.shape[1]):
            col_slice = slice(col_start, min(col_start + window_px, reference.shape[1]))
            local_context = context[row_slice, col_slice]
            samples = int(np.count_nonzero(local_context))
            if samples < minimum_samples:
                continue
            measured_windows += 1
            source = reference[row_slice, col_slice][local_context].astype(np.float64)
            output = candidate[row_slice, col_slice][local_context].astype(np.float64)
            source_centered = source - float(np.mean(source))
            output_centered = output - float(np.mean(output))
            local_source_rms = float(np.sqrt(np.mean(np.square(source_centered))))
            if local_source_rms < minimum_structure_rms:
                continue
            source_norm = float(np.linalg.norm(source_centered))
            output_norm = float(np.linalg.norm(output_centered))
            correlation = (
                float(
                    np.dot(source_centered, output_centered)
                    / (source_norm * output_norm)
                )
                if source_norm > 1e-10 and output_norm > 1e-10
                else 0.0
            )
            shape_error_ratio = float(
                np.sqrt(np.mean(np.square(output_centered - source_centered)))
                / max(local_source_rms, 1e-10)
            )
            records.append(
                {
                    "row": int(row_start),
                    "col": int(col_start),
                    "samples": samples,
                    "reference_rms_mm": local_source_rms,
                    "correlation": correlation,
                    "shape_error_ratio": shape_error_ratio,
                }
            )

    failed = [
        record
        for record in records
        if record["correlation"] < float(minimum_correlation)
        or record["shape_error_ratio"] > float(maximum_shape_error_ratio)
    ]
    correlations = [record["correlation"] for record in records]
    error_ratios = [record["shape_error_ratio"] for record in records]
    worst = (
        max(
            records,
            key=lambda record: max(
                float(minimum_correlation) - record["correlation"],
                record["shape_error_ratio"] - float(maximum_shape_error_ratio),
            ),
        )
        if records
        else None
    )
    return {
        "available": True,
        "passed": not failed,
        "window_mm": float(window_mm),
        "window_px": int(window_px),
        "stride_px": int(stride_px),
        "border_exclusion_px": int(border_exclusion_px),
        "minimum_samples_per_window": int(minimum_samples),
        "minimum_window_coverage_ratio": float(minimum_window_coverage_ratio),
        "measured_window_count": int(measured_windows),
        "structured_window_count": int(len(records)),
        "failed_window_count": int(len(failed)),
        "minimum_structure_rms_mm": float(minimum_structure_rms),
        "minimum_correlation": float(minimum_correlation),
        "maximum_shape_error_ratio": float(maximum_shape_error_ratio),
        "minimum_window_correlation": min(correlations) if correlations else None,
        "maximum_window_shape_error_ratio": max(error_ratios) if error_ratios else None,
        "worst_window": worst,
        "flat_or_smooth_reference": not records,
    }


def _surface_lighting_agreement_metrics(
    reference_values,
    candidate_values,
    region_mask,
    *,
    sample_pitch_mm,
    boundary_exclusion_mm=0.8,
    minimum_samples=64,
    component_metrics=False,
):
    """Compare physical surface normals and deterministic Lambertian renders."""
    stats = {
        "available": False,
        "samples": 0,
        "normal_mean_cosine": None,
        "normal_p05_cosine": None,
        "normal_angle_median_deg": None,
        "normal_angle_p95_deg": None,
        "minimum_lighting_correlation": None,
        "maximum_lighting_mae": None,
        "minimum_lighting_rms_retention": None,
        "maximum_lighting_rms_retention": None,
        "lights": [],
    }
    if region_mask is None:
        stats["reason"] = "no_region_mask"
        return stats

    reference = np.asarray(reference_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError("Surface lighting agreement requires identical shapes")
    try:
        pitch_mm = float(sample_pitch_mm)
        exclusion_mm = float(boundary_exclusion_mm)
    except (TypeError, ValueError):
        stats["reason"] = "invalid_physical_scale"
        return stats
    if (
        not np.isfinite(pitch_mm)
        or pitch_mm <= 0
        or not np.isfinite(exclusion_mm)
        or exclusion_mm < 0
    ):
        stats["reason"] = "invalid_physical_scale"
        return stats

    region = _resize_binary_mask(region_mask, reference.shape)
    reference_valid = np.isfinite(reference)
    candidate_valid = np.isfinite(candidate)
    reference_neighbor_valid = binary_erosion(
        reference_valid,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    candidate_neighbor_valid = binary_erosion(
        candidate_valid,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    exclusion_px = max(1, int(np.ceil(exclusion_mm / pitch_mm)))
    interior = binary_erosion(
        region,
        structure=np.ones((3, 3), dtype=bool),
        iterations=exclusion_px,
        border_value=0,
    )
    expected = interior & reference_neighbor_valid
    measured = expected & candidate_neighbor_valid
    reference_samples = int(np.count_nonzero(expected))
    samples = int(np.count_nonzero(measured))
    coverage_ratio = samples / max(reference_samples, 1)
    stats.update(
        {
            "samples": samples,
            "reference_samples": reference_samples,
            "missing_candidate_samples": int(reference_samples - samples),
            "candidate_coverage_ratio": float(coverage_ratio),
            "minimum_coverage_ratio": 1.0,
            "region_pixels": int(np.count_nonzero(region)),
            "boundary_exclusion_mm": exclusion_mm,
            "boundary_exclusion_px": exclusion_px,
            "minimum_samples": int(minimum_samples),
        }
    )
    if reference_samples < int(minimum_samples):
        stats["reason"] = "insufficient_reference_surface_samples"
        return stats
    if samples < int(minimum_samples):
        stats["reason"] = "insufficient_candidate_surface_samples"
        return stats

    def gradients(values):
        dx = np.zeros(values.shape, dtype=np.float64)
        dy = np.zeros(values.shape, dtype=np.float64)
        dx[:, 1:-1] = (
            values[:, 2:].astype(np.float64) - values[:, :-2].astype(np.float64)
        ) / (2.0 * pitch_mm)
        dy[1:-1, :] = (
            values[2:, :].astype(np.float64) - values[:-2, :].astype(np.float64)
        ) / (2.0 * pitch_mm)
        return dx, dy

    reference_dx, reference_dy = gradients(reference)
    candidate_dx, candidate_dy = gradients(candidate)

    def normals(dx, dy):
        vectors = np.column_stack(
            (
                -dx[measured],
                -dy[measured],
                np.ones(samples, dtype=np.float64),
            )
        )
        lengths = np.linalg.norm(vectors, axis=1)
        return vectors / np.maximum(lengths[:, None], 1e-12)

    reference_normals = normals(reference_dx, reference_dy)
    candidate_normals = normals(candidate_dx, candidate_dy)
    normal_cosines = np.clip(
        np.einsum("ij,ij->i", reference_normals, candidate_normals),
        -1.0,
        1.0,
    )
    normal_angles = np.degrees(np.arccos(normal_cosines))

    def centered_correlation(reference_signal, candidate_signal):
        reference_centered = reference_signal - float(np.mean(reference_signal))
        candidate_centered = candidate_signal - float(np.mean(candidate_signal))
        reference_norm = float(np.linalg.norm(reference_centered))
        candidate_norm = float(np.linalg.norm(candidate_centered))
        if reference_norm <= 1e-12:
            return 1.0 if candidate_norm <= 1e-12 else 0.0
        if candidate_norm <= 1e-12:
            return 0.0
        return float(
            np.dot(reference_centered, candidate_centered)
            / (reference_norm * candidate_norm)
        )

    light_specs = (
        ("frontal", (0.0, 0.0, 1.0)),
        ("upper_left", (-0.45, -0.35, 0.82)),
        ("upper_right", (0.45, -0.25, 0.82)),
        ("grazing_right", (0.67, 0.10, 0.74)),
    )
    light_records = []
    for name, direction in light_specs:
        light = np.asarray(direction, dtype=np.float64)
        light /= max(float(np.linalg.norm(light)), 1e-12)
        reference_lighting = 0.25 + 0.75 * np.clip(
            reference_normals @ light,
            0.0,
            1.0,
        )
        candidate_lighting = 0.25 + 0.75 * np.clip(
            candidate_normals @ light,
            0.0,
            1.0,
        )
        reference_centered = reference_lighting - float(np.mean(reference_lighting))
        candidate_centered = candidate_lighting - float(np.mean(candidate_lighting))
        reference_rms = float(np.sqrt(np.mean(np.square(reference_centered))))
        candidate_rms = float(np.sqrt(np.mean(np.square(candidate_centered))))
        rms_retention = (
            candidate_rms / reference_rms
            if reference_rms > 1e-12
            else (1.0 if candidate_rms <= 1e-12 else None)
        )
        light_records.append(
            {
                "name": name,
                "correlation": centered_correlation(
                    reference_lighting,
                    candidate_lighting,
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(candidate_lighting - reference_lighting))
                ),
                "rms_retention": rms_retention,
            }
        )

    correlations = [record["correlation"] for record in light_records]
    errors = [record["mean_absolute_error"] for record in light_records]
    retentions = [
        record["rms_retention"]
        for record in light_records
        if record["rms_retention"] is not None
        and np.isfinite(record["rms_retention"])
    ]
    stats.update(
        {
            "available": True,
            "normal_mean_cosine": float(np.mean(normal_cosines)),
            "normal_p05_cosine": float(np.percentile(normal_cosines, 5.0)),
            "normal_angle_median_deg": float(np.median(normal_angles)),
            "normal_angle_p95_deg": float(np.percentile(normal_angles, 95.0)),
            "minimum_lighting_correlation": float(min(correlations)),
            "maximum_lighting_mae": float(max(errors)),
            "minimum_lighting_rms_retention": (
                float(min(retentions)) if retentions else None
            ),
            "maximum_lighting_rms_retention": (
                float(max(retentions)) if retentions else None
            ),
            "lights": light_records,
        }
    )
    if component_metrics:
        component_labels, component_count = label(
            region,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        components = []
        for component_index in range(1, component_count + 1):
            component = component_labels == component_index
            component_stats = _surface_lighting_agreement_metrics(
                reference,
                candidate,
                component,
                sample_pitch_mm=pitch_mm,
                boundary_exclusion_mm=exclusion_mm,
                minimum_samples=minimum_samples,
                component_metrics=False,
            )
            components.append(
                {
                    "component": int(component_index),
                    **component_stats,
                }
            )
        stats.update(
            {
                "components": components,
                "component_count": int(component_count),
                "measured_component_count": int(
                    sum(record.get("available", False) for record in components)
                ),
                "unavailable_component_count": int(
                    sum(not record.get("available", False) for record in components)
                ),
            }
        )
    return stats


def _background_relief_preservation_metrics(
    reference_values,
    candidate_values,
    foreground_mask,
    *,
    sample_pitch_mm=None,
    boundary_exclusion_mm=1.5,
    minimum_correlation=0.8,
    minimum_rms_retention=0.5,
    minimum_span_retention=0.5,
    minimum_gradient_correlation=0.58,
    minimum_gradient_rms_retention=0.25,
    maximum_rms_retention=1.5,
    maximum_span_retention=1.5,
    maximum_gradient_rms_retention=1.75,
    maximum_mean_shift_mm=2.0,
    maximum_boundary_jump_mm=6.0,
    minimum_coverage_ratio=1.0,
    local_window_mm=6.0,
    minimum_local_correlation=0.35,
    maximum_local_shape_error_ratio=0.95,
):
    """Measure broad scene structure outside a protected foreground subject."""
    stats = {
        "available": False,
        "passed": False,
        "samples": 0,
        "correlation": None,
        "rms_retention": None,
        "span_retention": None,
        "gradient_correlation": None,
        "gradient_rms_retention": None,
        "quality_failures": [],
    }
    if foreground_mask is None:
        stats["reason"] = "no_foreground_mask"
        return stats

    reference = np.asarray(reference_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if reference.shape != candidate.shape:
        raise ValueError("Background preservation surfaces must have identical shapes")
    foreground = _resize_binary_mask(foreground_mask, reference.shape)
    reference_valid = np.isfinite(reference)
    candidate_valid = np.isfinite(candidate)
    valid = reference_valid & candidate_valid
    try:
        pitch_mm = float(sample_pitch_mm)
    except (TypeError, ValueError):
        pitch_mm = 0.0
    exclusion_px = (
        max(2, int(np.ceil(float(boundary_exclusion_mm) / pitch_mm)))
        if np.isfinite(pitch_mm) and pitch_mm > 0
        else 3
    )
    reference_context = reference_valid & ~foreground
    reference_context &= distance_transform_edt(~foreground) >= float(exclusion_px)
    border = max(2, min(exclusion_px // 2, 8))
    reference_context[:border, :] = False
    reference_context[-border:, :] = False
    reference_context[:, :border] = False
    reference_context[:, -border:] = False
    reference_samples = int(np.count_nonzero(reference_context))
    context = reference_context & candidate_valid
    samples = int(np.count_nonzero(context))
    coverage_ratio = samples / max(reference_samples, 1)
    stats.update(
        {
            "samples": samples,
            "reference_samples": reference_samples,
            "missing_candidate_samples": int(reference_samples - samples),
            "candidate_coverage_ratio": float(coverage_ratio),
            "minimum_coverage_ratio": float(minimum_coverage_ratio),
            "boundary_exclusion_mm": float(boundary_exclusion_mm),
            "boundary_exclusion_px": int(exclusion_px),
            "minimum_correlation": float(minimum_correlation),
            "minimum_rms_retention": float(minimum_rms_retention),
            "minimum_span_retention": float(minimum_span_retention),
            "minimum_gradient_correlation": float(minimum_gradient_correlation),
            "minimum_gradient_rms_retention": float(minimum_gradient_rms_retention),
            "maximum_rms_retention": float(maximum_rms_retention),
            "maximum_span_retention": float(maximum_span_retention),
            "maximum_gradient_rms_retention": float(maximum_gradient_rms_retention),
            "maximum_mean_shift_mm": float(maximum_mean_shift_mm),
            "maximum_boundary_jump_mm": float(maximum_boundary_jump_mm),
            "local_window_mm": float(local_window_mm),
            "minimum_local_correlation": float(minimum_local_correlation),
            "maximum_local_shape_error_ratio": float(
                maximum_local_shape_error_ratio
            ),
        }
    )
    if reference_samples < 64:
        stats["reason"] = "insufficient_reference_background_samples"
        return stats

    quality_failures = []
    if coverage_ratio < float(minimum_coverage_ratio):
        quality_failures.append("coverage")
    if samples < 64:
        stats.update(
            {
                "available": True,
                "reason": "insufficient_candidate_background_samples",
                "quality_failures": quality_failures or ["coverage"],
            }
        )
        return stats

    source = reference[context].astype(np.float64)
    output = candidate[context].astype(np.float64)
    mean_shift_mm = float(np.mean(output - source))
    source_centered = source - float(np.mean(source))
    output_centered = output - float(np.mean(output))
    source_rms = float(np.sqrt(np.mean(np.square(source_centered))))
    output_rms = float(np.sqrt(np.mean(np.square(output_centered))))
    source_norm = float(np.linalg.norm(source_centered))
    output_norm = float(np.linalg.norm(output_centered))
    correlation = (
        float(np.dot(source_centered, output_centered) / (source_norm * output_norm))
        if source_norm > 1e-10 and output_norm > 1e-10
        else (1.0 if source_norm <= 1e-10 and output_norm <= 1e-8 else 0.0)
    )
    rms_retention = (
        output_rms / source_rms
        if source_rms > 1e-10
        else (1.0 if output_rms <= 1e-8 else None)
    )
    source_low, source_high = (float(value) for value in np.percentile(source, [2.0, 98.0]))
    output_low, output_high = (float(value) for value in np.percentile(output, [2.0, 98.0]))
    source_span = source_high - source_low
    output_span = output_high - output_low
    span_retention = (
        output_span / source_span
        if source_span > 1e-10
        else (1.0 if output_span <= 1e-8 else None)
    )

    horizontal = context[:, :-1] & context[:, 1:]
    vertical = context[:-1, :] & context[1:, :]
    source_gradients = np.concatenate(
        (
            (reference[:, 1:] - reference[:, :-1])[horizontal],
            (reference[1:, :] - reference[:-1, :])[vertical],
        )
    ).astype(np.float64)
    output_gradients = np.concatenate(
        (
            (candidate[:, 1:] - candidate[:, :-1])[horizontal],
            (candidate[1:, :] - candidate[:-1, :])[vertical],
        )
    ).astype(np.float64)
    if source_gradients.size < 32:
        quality_failures.append("gradient_coverage")
        stats.update(
            {
                "available": True,
                "reason": "insufficient_background_gradient_samples",
                "quality_failures": quality_failures,
            }
        )
        return stats
    source_gradient_rms = float(np.sqrt(np.mean(np.square(source_gradients))))
    output_gradient_rms = float(np.sqrt(np.mean(np.square(output_gradients))))
    source_gradient_centered = source_gradients - float(np.mean(source_gradients))
    output_gradient_centered = output_gradients - float(np.mean(output_gradients))
    source_gradient_norm = float(np.linalg.norm(source_gradient_centered))
    output_gradient_norm = float(np.linalg.norm(output_gradient_centered))
    gradient_correlation = (
        float(
            np.dot(source_gradient_centered, output_gradient_centered)
            / (source_gradient_norm * output_gradient_norm)
        )
        if source_gradient_norm > 1e-10 and output_gradient_norm > 1e-10
        else (
            1.0
            if source_gradient_norm <= 1e-10 and output_gradient_norm <= 1e-8
            else 0.0
        )
    )
    gradient_rms_retention = (
        output_gradient_rms / source_gradient_rms
        if source_gradient_rms > 1e-10
        else (1.0 if output_gradient_rms <= 1e-8 else None)
    )
    localized_structure = _localized_background_structure_metrics(
        reference,
        candidate,
        context,
        sample_pitch_mm=pitch_mm,
        window_mm=local_window_mm,
        minimum_correlation=minimum_local_correlation,
        maximum_shape_error_ratio=maximum_local_shape_error_ratio,
    )

    boundary_horizontal = (
        (foreground[:, :-1] ^ foreground[:, 1:])
        & valid[:, :-1]
        & valid[:, 1:]
    )
    boundary_vertical = (
        (foreground[:-1, :] ^ foreground[1:, :])
        & valid[:-1, :]
        & valid[1:, :]
    )
    reference_boundary_jumps = np.concatenate(
        (
            np.abs(reference[:, 1:] - reference[:, :-1])[boundary_horizontal],
            np.abs(reference[1:, :] - reference[:-1, :])[boundary_vertical],
        )
    ).astype(np.float64)
    output_boundary_jumps = np.concatenate(
        (
            np.abs(candidate[:, 1:] - candidate[:, :-1])[boundary_horizontal],
            np.abs(candidate[1:, :] - candidate[:-1, :])[boundary_vertical],
        )
    ).astype(np.float64)
    boundary_samples = int(output_boundary_jumps.size)
    reference_boundary_p99 = (
        float(np.percentile(reference_boundary_jumps, 99.0))
        if boundary_samples
        else None
    )
    reference_boundary_max = (
        float(np.max(reference_boundary_jumps)) if boundary_samples else None
    )
    output_boundary_p99 = (
        float(np.percentile(output_boundary_jumps, 99.0))
        if boundary_samples
        else None
    )
    output_boundary_max = (
        float(np.max(output_boundary_jumps)) if boundary_samples else None
    )
    effective_boundary_p99_limit = (
        max(float(maximum_boundary_jump_mm), reference_boundary_p99 * 1.25)
        if boundary_samples
        else float(maximum_boundary_jump_mm)
    )
    effective_boundary_max_limit = (
        max(float(maximum_boundary_jump_mm) * 2.0, reference_boundary_max * 1.25)
        if boundary_samples
        else float(maximum_boundary_jump_mm) * 2.0
    )

    if correlation < float(minimum_correlation):
        quality_failures.append("correlation")
    if (
        rms_retention is None
        or not float(minimum_rms_retention)
        <= rms_retention
        <= float(maximum_rms_retention)
    ):
        quality_failures.append("rms_retention")
    if (
        span_retention is None
        or not float(minimum_span_retention)
        <= span_retention
        <= float(maximum_span_retention)
    ):
        quality_failures.append("span_retention")
    if gradient_correlation < float(minimum_gradient_correlation):
        quality_failures.append("gradient_correlation")
    if (
        gradient_rms_retention is None
        or not float(minimum_gradient_rms_retention)
        <= gradient_rms_retention
        <= float(maximum_gradient_rms_retention)
    ):
        quality_failures.append("gradient_rms_retention")
    if abs(mean_shift_mm) > float(maximum_mean_shift_mm):
        quality_failures.append("mean_shift")
    if not localized_structure["passed"]:
        quality_failures.append("localized_structure")
    if (
        boundary_samples
        and (
            output_boundary_p99 > effective_boundary_p99_limit
            or output_boundary_max > effective_boundary_max_limit
        )
    ):
        quality_failures.append("boundary_jump")
    stats.update(
        {
            "available": True,
            "passed": not quality_failures,
            "correlation": correlation,
            "rms_retention": float(rms_retention) if rms_retention is not None else None,
            "span_retention": float(span_retention) if span_retention is not None else None,
            "gradient_samples": int(source_gradients.size),
            "gradient_correlation": gradient_correlation,
            "gradient_rms_retention": (
                float(gradient_rms_retention)
                if gradient_rms_retention is not None
                else None
            ),
            "mean_shift_mm": mean_shift_mm,
            "boundary_samples": boundary_samples,
            "reference_boundary_jump_p99_mm": reference_boundary_p99,
            "reference_boundary_jump_max_mm": reference_boundary_max,
            "output_boundary_jump_p99_mm": output_boundary_p99,
            "output_boundary_jump_max_mm": output_boundary_max,
            "effective_boundary_jump_p99_limit_mm": effective_boundary_p99_limit,
            "effective_boundary_jump_max_limit_mm": effective_boundary_max_limit,
            "reference_rms_mm": source_rms,
            "output_rms_mm": output_rms,
            "reference_span_p02_p98_mm": source_span,
            "output_span_p02_p98_mm": output_span,
            "reference_gradient_rms_mm": source_gradient_rms,
            "output_gradient_rms_mm": output_gradient_rms,
            "localized_structure": localized_structure,
            "quality_failures": quality_failures,
        }
    )
    return stats


def _compress_relief_gradients(
    values,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    structural_region_mask=None,
    detail_region_mask=None,
    screening_weight=0.01,
    gradient_threshold_ratio=1.2,
    boundary_anchor_weight=64.0,
    solver_tolerance=1e-7,
    max_iterations=2400,
    minimum_detail_correlation=0.8,
    minimum_detail_rms_retention=0.6,
    maximum_detail_rms_retention=2.0,
    detail_gradient_retention=0.8,
    maximum_detail_gradient_ratio=12.0,
    maximum_output_edge_p99_ratio=12.0,
    maximum_output_edge_ratio=24.0,
    minimum_height_span_ratio=0.5,
    maximum_height_span_ratio=1.15,
    maximum_correction_span_ratio=0.9,
    allow_edge_only_candidate=False,
):
    """Compress large relief gradients and reconstruct a coherent height field."""
    stats = {
        "enabled": False,
        "method": "screened_gradient_domain_compression",
        "hard_slope_limit_enforced": False,
    }
    try:
        pitch_mm = float(sample_pitch_mm)
        max_slope = float(max_slope_mm_per_mm)
        screening = float(screening_weight)
        threshold_ratio = float(gradient_threshold_ratio)
        anchor_weight = float(boundary_anchor_weight)
        tolerance = float(solver_tolerance)
        iteration_limit = int(max_iterations)
        min_detail_correlation = float(minimum_detail_correlation)
        min_detail_retention = float(minimum_detail_rms_retention)
        max_detail_retention = float(maximum_detail_rms_retention)
        detail_retention = float(detail_gradient_retention)
        max_detail_gradient_ratio = float(maximum_detail_gradient_ratio)
        max_edge_p99_ratio = float(maximum_output_edge_p99_ratio)
        max_edge_ratio = float(maximum_output_edge_ratio)
        min_span_ratio = float(minimum_height_span_ratio)
        max_span_ratio = float(maximum_height_span_ratio)
        max_correction_ratio = float(maximum_correction_span_ratio)
    except (TypeError, ValueError):
        stats["reason"] = "invalid_configuration"
        return values, stats
    if (
        not np.isfinite(pitch_mm)
        or pitch_mm <= 0
        or not np.isfinite(max_slope)
        or max_slope <= 0
        or not np.isfinite(screening)
        or screening <= 0
        or not np.isfinite(threshold_ratio)
        or threshold_ratio <= 0
        or not np.isfinite(anchor_weight)
        or anchor_weight <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or iteration_limit < 1
        or not np.isfinite(min_detail_correlation)
        or min_detail_correlation < -1
        or min_detail_correlation > 1
        or not np.isfinite(min_detail_retention)
        or min_detail_retention < 0
        or not np.isfinite(max_detail_retention)
        or max_detail_retention < min_detail_retention
        or not np.isfinite(detail_retention)
        or detail_retention < 0
        or detail_retention > 1
        or not np.isfinite(max_detail_gradient_ratio)
        or max_detail_gradient_ratio <= 0
        or not np.isfinite(max_edge_p99_ratio)
        or max_edge_p99_ratio <= 0
        or not np.isfinite(max_edge_ratio)
        or max_edge_ratio < max_edge_p99_ratio
        or not np.isfinite(min_span_ratio)
        or min_span_ratio <= 0
        or not np.isfinite(max_span_ratio)
        or max_span_ratio < min_span_ratio
        or not np.isfinite(max_correction_ratio)
        or max_correction_ratio <= 0
    ):
        stats["reason"] = "invalid_configuration"
        return values, stats

    source = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(source)
    sample_count = int(np.count_nonzero(valid))
    if sample_count < 4:
        stats["reason"] = "insufficient_finite_samples"
        return values, stats

    max_step = pitch_mm * max_slope
    soft_threshold = max_step * threshold_ratio
    horizontal_barrier, vertical_barrier, edge_stats = _structural_relief_edge_barriers(
        source,
        max_step,
        structural_region_mask=structural_region_mask,
    )
    sample_ids = np.full(source.shape, -1, dtype=np.int64)
    sample_ids[valid] = np.arange(sample_count, dtype=np.int64)
    source_values = source[valid].astype(np.float64)
    diagonal = np.full(sample_count, screening, dtype=np.float64)
    rhs = screening * source_values
    matrix_rows = []
    matrix_cols = []
    matrix_values = []
    input_edge_values = []
    target_edge_values = []
    retained_detail_pairs = 0
    detail_core = None
    if detail_region_mask is not None:
        detail_core = _resize_binary_mask(detail_region_mask, source.shape) & valid
        detail_core = binary_erosion(
            detail_core,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )

    def add_edges(
        first_values,
        second_values,
        first_ids,
        second_ids,
        active,
        barriers,
        detail_pairs,
    ):
        nonlocal retained_detail_pairs
        first = first_ids[active]
        second = second_ids[active]
        raw_gradient = (second_values - first_values)[active].astype(np.float64)
        desired_gradient = (
            np.sign(raw_gradient)
            * soft_threshold
            * np.arcsinh(np.abs(raw_gradient) / soft_threshold)
        )
        preserved = barriers[active]
        desired_gradient[preserved] = raw_gradient[preserved]
        if detail_pairs is not None and detail_retention > 0:
            detail_active = detail_pairs[active]
            recoverable = (
                detail_active
                & ~preserved
                & (np.abs(raw_gradient) <= max_step * max_detail_gradient_ratio)
            )
            desired_gradient[recoverable] += detail_retention * (
                raw_gradient[recoverable] - desired_gradient[recoverable]
            )
            retained_detail_pairs += int(np.count_nonzero(recoverable))

        np.add.at(diagonal, first, 1.0)
        np.add.at(diagonal, second, 1.0)
        np.add.at(rhs, first, -desired_gradient)
        np.add.at(rhs, second, desired_gradient)
        matrix_rows.extend((first, second))
        matrix_cols.extend((second, first))
        matrix_values.extend(
            (
                np.full(first.size, -1.0, dtype=np.float64),
                np.full(first.size, -1.0, dtype=np.float64),
            )
        )
        input_edge_values.append(np.abs(raw_gradient))
        target_edge_values.append(np.abs(desired_gradient))

    horizontal_active = valid[:, :-1] & valid[:, 1:]
    add_edges(
        source[:, :-1],
        source[:, 1:],
        sample_ids[:, :-1],
        sample_ids[:, 1:],
        horizontal_active,
        horizontal_barrier,
        (
            detail_core[:, :-1] & detail_core[:, 1:]
            if detail_core is not None
            else None
        ),
    )
    vertical_active = valid[:-1, :] & valid[1:, :]
    add_edges(
        source[:-1, :],
        source[1:, :],
        sample_ids[:-1, :],
        sample_ids[1:, :],
        vertical_active,
        vertical_barrier,
        (
            detail_core[:-1, :] & detail_core[1:, :]
            if detail_core is not None
            else None
        ),
    )
    if not input_edge_values or not any(values.size for values in input_edge_values):
        stats["reason"] = "no_connected_grid_edges"
        return values, stats

    anchors = np.zeros(source.shape, dtype=bool)
    anchors[0, :] = valid[0, :]
    anchors[-1, :] = valid[-1, :]
    anchors[:, 0] |= valid[:, 0]
    anchors[:, -1] |= valid[:, -1]
    if not np.any(anchors):
        finite_rows, finite_cols = np.where(valid)
        minimum_index = int(np.argmin(source_values))
        anchors[finite_rows[minimum_index], finite_cols[minimum_index]] = True
    anchor_ids = sample_ids[anchors]
    diagonal[anchor_ids] += anchor_weight
    rhs[anchor_ids] += anchor_weight * source[anchors]

    matrix_rows.insert(0, np.arange(sample_count, dtype=np.int64))
    matrix_cols.insert(0, np.arange(sample_count, dtype=np.int64))
    matrix_values.insert(0, diagonal)
    matrix = coo_matrix(
        (
            np.concatenate(matrix_values),
            (np.concatenate(matrix_rows), np.concatenate(matrix_cols)),
        ),
        shape=(sample_count, sample_count),
    ).tocsr()
    solver_iterations = 0

    def count_iteration(_current):
        nonlocal solver_iterations
        solver_iterations += 1

    try:
        try:
            solution, solver_info = cg(
                matrix,
                rhs,
                rtol=tolerance,
                atol=0.0,
                maxiter=iteration_limit,
                callback=count_iteration,
            )
        except TypeError:  # pragma: no cover - older SciPy compatibility
            solution, solver_info = cg(
                matrix,
                rhs,
                tol=tolerance,
                maxiter=iteration_limit,
                callback=count_iteration,
            )
    except Exception as exc:
        stats.update(
            {
                "reason": "solver_error",
                "solver_error": type(exc).__name__,
                "solver_iterations": int(solver_iterations),
            }
        )
        return values, stats
    if int(solver_info) != 0 or not np.all(np.isfinite(solution)):
        stats.update(
            {
                "reason": "solver_nonconvergence",
                "solver_info": int(solver_info),
                "solver_iterations": int(solver_iterations),
            }
        )
        return values, stats

    compressed = np.full(source.shape, np.nan, dtype=np.float32)
    compressed[valid] = solution.astype(np.float32)
    output_edges = np.concatenate(
        (
            np.abs(compressed[:, 1:] - compressed[:, :-1])[horizontal_active],
            np.abs(compressed[1:, :] - compressed[:-1, :])[vertical_active],
        )
    ).astype(np.float64)
    input_edges = np.concatenate(input_edge_values)
    target_edges = np.concatenate(target_edge_values)
    correction = np.abs(solution - source_values)
    output_ratio = output_edges / max(max_step, 1e-12)
    diagonal_edges = np.concatenate(
        (
            np.abs(compressed[1:, 1:] - compressed[:-1, :-1])[
                valid[1:, 1:] & valid[:-1, :-1]
            ],
            np.abs(compressed[1:, :-1] - compressed[:-1, 1:])[
                valid[1:, :-1] & valid[:-1, 1:]
            ],
        )
    ).astype(np.float64)
    diagonal_ratio = diagonal_edges / max(max_step * np.sqrt(2.0), 1e-12)
    input_height_span = float(np.ptp(source_values))
    output_height_span = float(np.ptp(solution))
    height_span_ratio = output_height_span / max(input_height_span, 1e-12)
    correction_span_ratio = float(np.max(correction)) / max(input_height_span, 1e-12)

    detail_stats = _face_detail_preservation_metrics(
        source,
        compressed,
        detail_region_mask,
    )

    output_edge_ratio_p99 = float(np.percentile(output_ratio, 99.0))
    output_edge_ratio_max = float(np.max(output_ratio))
    diagonal_metrics_available = bool(diagonal_ratio.size)
    diagonal_edge_p99_mm = (
        float(np.percentile(diagonal_edges, 99.0))
        if diagonal_metrics_available
        else None
    )
    diagonal_edge_max_mm = (
        float(np.max(diagonal_edges)) if diagonal_metrics_available else None
    )
    diagonal_edge_ratio_p99 = (
        float(np.percentile(diagonal_ratio, 99.0))
        if diagonal_metrics_available
        else None
    )
    diagonal_edge_ratio_max = (
        float(np.max(diagonal_ratio)) if diagonal_metrics_available else None
    )
    quality_failures = []
    if output_edge_ratio_p99 > max_edge_p99_ratio:
        quality_failures.append("cardinal_edge_p99")
    if output_edge_ratio_max > max_edge_ratio:
        quality_failures.append("cardinal_edge_max")
    if diagonal_metrics_available:
        if diagonal_edge_ratio_p99 > max_edge_p99_ratio:
            quality_failures.append("diagonal_edge_p99")
        if diagonal_edge_ratio_max > max_edge_ratio:
            quality_failures.append("diagonal_edge_max")
    if not min_span_ratio <= height_span_ratio <= max_span_ratio:
        quality_failures.append("height_span_ratio")
    if correction_span_ratio > max_correction_ratio:
        quality_failures.append("correction_span_ratio")
    if detail_region_mask is not None and not detail_stats["available"]:
        quality_failures.append("detail_metric_unavailable")
    if detail_stats.get("measured_component_count", 0):
        if detail_stats.get("flat_reference_violation", False):
            quality_failures.append("detail_flat_reference_violation")
        aggregate_correlation = detail_stats.get("correlation")
        if (
            aggregate_correlation is None
            or not np.isfinite(float(aggregate_correlation))
            or float(aggregate_correlation) < min_detail_correlation
        ):
            quality_failures.append("detail_correlation")
        aggregate_retention = detail_stats.get("rms_retention")
        if (
            aggregate_retention is None
            or not np.isfinite(float(aggregate_retention))
            or not min_detail_retention <= float(aggregate_retention) <= max_detail_retention
        ):
            quality_failures.append("detail_rms_retention")
        component_records = detail_stats.get("components", [])
        if any(
            not record.get("available", False)
            or record.get("correlation") is None
            or not np.isfinite(float(record["correlation"]))
            or float(record["correlation"]) < min_detail_correlation
            for record in component_records
        ):
            quality_failures.append("detail_component_correlation")
        if any(
            not _detail_component_passes(
                record,
                min_detail_correlation,
                min_detail_retention,
                max_detail_retention,
            )
            for record in component_records
        ):
            quality_failures.append("detail_component_rms_retention")
    stats.update(
        {
            "enabled": not quality_failures,
            "max_slope_mm_per_mm": max_slope,
            "max_neighbor_step_mm": max_step,
            "gradient_soft_threshold_mm": soft_threshold,
            "gradient_threshold_ratio": threshold_ratio,
            "detail_gradient_retention": detail_retention,
            "maximum_detail_gradient_ratio": max_detail_gradient_ratio,
            "retained_detail_gradient_pairs": int(retained_detail_pairs),
            "screening_weight": screening,
            "boundary_anchor_weight": anchor_weight,
            "anchor_pixels": int(np.count_nonzero(anchors)),
            "solver_tolerance": tolerance,
            "solver_info": int(solver_info),
            "solver_iterations": int(solver_iterations),
            "input_height_span_mm": input_height_span,
            "output_height_span_mm": output_height_span,
            "height_span_ratio": height_span_ratio,
            "input_edge_p99_mm": float(np.percentile(input_edges, 99.0)),
            "input_edge_max_mm": float(np.max(input_edges)),
            "target_edge_p99_mm": float(np.percentile(target_edges, 99.0)),
            "target_edge_max_mm": float(np.max(target_edges)),
            "output_edge_p99_mm": float(np.percentile(output_edges, 99.0)),
            "output_edge_max_mm": float(np.max(output_edges)),
            "output_edge_ratio_p99": output_edge_ratio_p99,
            "output_edge_ratio_max": output_edge_ratio_max,
            "diagonal_metrics_available": diagonal_metrics_available,
            "diagonal_edge_p99_mm": diagonal_edge_p99_mm,
            "diagonal_edge_max_mm": diagonal_edge_max_mm,
            "diagonal_edge_ratio_p99": diagonal_edge_ratio_p99,
            "diagonal_edge_ratio_max": diagonal_edge_ratio_max,
            "correction_p95_mm": float(np.percentile(correction, 95.0)),
            "correction_max_mm": float(np.max(correction)),
            "correction_span_ratio": correction_span_ratio,
            "detail_preservation": detail_stats,
            "quality_gates": {
                "passed": not quality_failures,
                "failures": quality_failures,
                "minimum_detail_correlation": min_detail_correlation,
                "minimum_detail_rms_retention": min_detail_retention,
                "maximum_detail_rms_retention": max_detail_retention,
                "maximum_output_edge_p99_ratio": max_edge_p99_ratio,
                "maximum_output_edge_ratio": max_edge_ratio,
                "minimum_height_span_ratio": min_span_ratio,
                "maximum_height_span_ratio": max_span_ratio,
                "maximum_correction_span_ratio": max_correction_ratio,
            },
            **edge_stats,
        }
    )
    edge_quality_failures = {
        "cardinal_edge_p99",
        "cardinal_edge_max",
        "diagonal_edge_p99",
        "diagonal_edge_max",
    }
    provisional_edge_only_candidate = bool(
        allow_edge_only_candidate
        and quality_failures
        and set(quality_failures).issubset(edge_quality_failures)
    )
    stats["provisional_edge_only_candidate"] = provisional_edge_only_candidate
    if provisional_edge_only_candidate:
        stats["enabled"] = True
        stats["reason"] = "pending_baseline_aware_post_blend_audit"
        return compressed, stats
    if quality_failures:
        stats["reason"] = "quality_gate"
        return values, stats
    return compressed, stats


def _audit_bounded_compression_surface(
    source_values,
    candidate_values,
    detail_region_mask,
    *,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    quality_gates,
    reject_direction_reversals=False,
):
    """Recheck the exact post-blend surface, exempting unchanged baseline edges."""
    source = np.asarray(source_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    stats = {
        "enabled": False,
        "method": "post_blend_updated_edges_compression_audit",
        "quality_gates": {
            **dict(quality_gates or {}),
            "passed": False,
            "failures": ["audit_unavailable"],
        },
    }
    if source.shape != candidate.shape:
        stats["reason"] = "shape_mismatch"
        return stats
    source_valid = np.isfinite(source)
    candidate_valid = np.isfinite(candidate)
    source_finite_count = int(np.count_nonzero(source_valid))
    candidate_finite_count = int(np.count_nonzero(candidate_valid))
    stats.update(
        {
            "source_finite_samples": source_finite_count,
            "candidate_finite_samples": candidate_finite_count,
            "finite_mask_match": bool(np.array_equal(source_valid, candidate_valid)),
            "candidate_finite_coverage_ratio": float(
                candidate_finite_count / max(source_finite_count, 1)
            ),
        }
    )
    if not stats["finite_mask_match"]:
        stats["reason"] = "finite_coverage_mismatch"
        stats["quality_gates"]["failures"] = ["finite_coverage_mismatch"]
        return stats
    valid = source_valid
    if np.count_nonzero(valid) < 4:
        stats["reason"] = "insufficient_finite_samples"
        return stats
    max_step = float(sample_pitch_mm) * float(max_slope_mm_per_mm)
    if not np.isfinite(max_step) or max_step <= 0:
        stats["reason"] = "invalid_slope_configuration"
        return stats

    changed = valid & (np.abs(candidate - source) > 1e-6)
    horizontal_all = valid[:, :-1] & valid[:, 1:]
    vertical_all = valid[:-1, :] & valid[1:, :]
    horizontal = horizontal_all & (changed[:, :-1] | changed[:, 1:])
    vertical = vertical_all & (changed[:-1, :] | changed[1:, :])
    output_signed_edges = np.concatenate(
        (
            (candidate[:, 1:] - candidate[:, :-1])[horizontal],
            (candidate[1:, :] - candidate[:-1, :])[vertical],
        )
    ).astype(np.float64)
    source_signed_edges = np.concatenate(
        (
            (source[:, 1:] - source[:, :-1])[horizontal],
            (source[1:, :] - source[:-1, :])[vertical],
        )
    ).astype(np.float64)
    output_edges = np.abs(output_signed_edges)
    source_edges = np.abs(source_signed_edges)
    diagonal_down_all = valid[1:, 1:] & valid[:-1, :-1]
    diagonal_up_all = valid[1:, :-1] & valid[:-1, 1:]
    diagonal_down = diagonal_down_all & (changed[1:, 1:] | changed[:-1, :-1])
    diagonal_up = diagonal_up_all & (changed[1:, :-1] | changed[:-1, 1:])
    output_signed_diagonal_edges = np.concatenate(
        (
            (candidate[1:, 1:] - candidate[:-1, :-1])[diagonal_down],
            (candidate[1:, :-1] - candidate[:-1, 1:])[diagonal_up],
        )
    ).astype(np.float64)
    source_signed_diagonal_edges = np.concatenate(
        (
            (source[1:, 1:] - source[:-1, :-1])[diagonal_down],
            (source[1:, :-1] - source[:-1, 1:])[diagonal_up],
        )
    ).astype(np.float64)
    diagonal_edges = np.abs(output_signed_diagonal_edges)
    source_diagonal_edges = np.abs(source_signed_diagonal_edges)
    if not output_edges.size:
        stats["reason"] = "no_updated_cardinal_edges"
        return stats

    source_samples = source[valid].astype(np.float64)
    candidate_samples = candidate[valid].astype(np.float64)
    input_span = float(np.ptp(source_samples))
    output_span = float(np.ptp(candidate_samples))
    height_span_ratio = output_span / max(input_span, 1e-12)
    correction = np.abs(candidate_samples - source_samples)
    correction_span_ratio = float(np.max(correction)) / max(input_span, 1e-12)
    physical_cardinal_ratio = output_edges / max(max_step, 1e-12)
    cardinal_ratio = output_edges / np.maximum(source_edges, max_step)
    cardinal_excess_ratio = np.maximum(output_edges - source_edges, 0.0) / max_step
    cardinal_direction_reversal = (
        (source_signed_edges * output_signed_edges < 0.0)
        & (source_edges > max_step * 0.25)
        & (output_edges > max_step)
    )
    diagonal_step = max_step * np.sqrt(2.0)
    physical_diagonal_ratio = diagonal_edges / max(diagonal_step, 1e-12)
    diagonal_ratio = diagonal_edges / np.maximum(
        source_diagonal_edges,
        diagonal_step,
    )
    diagonal_excess_ratio = (
        np.maximum(diagonal_edges - source_diagonal_edges, 0.0) / diagonal_step
    )
    diagonal_direction_reversal = (
        (source_signed_diagonal_edges * output_signed_diagonal_edges < 0.0)
        & (source_diagonal_edges > diagonal_step * 0.25)
        & (diagonal_edges > diagonal_step)
    )
    cardinal_reversal_count = int(np.count_nonzero(cardinal_direction_reversal))
    diagonal_reversal_count = int(np.count_nonzero(diagonal_direction_reversal))
    total_reversal_count = cardinal_reversal_count + diagonal_reversal_count
    minimum_reversal_support_edges = 4
    cardinal_reversal_max_physical_ratio = (
        float(np.max(physical_cardinal_ratio[cardinal_direction_reversal]))
        if cardinal_reversal_count
        else 0.0
    )
    diagonal_reversal_max_physical_ratio = (
        float(np.max(physical_diagonal_ratio[diagonal_direction_reversal]))
        if diagonal_reversal_count
        else 0.0
    )
    detail_stats = _face_detail_preservation_metrics(
        source,
        candidate,
        detail_region_mask,
    )

    minimum_detail_correlation = float(
        quality_gates.get("minimum_detail_correlation", 0.8)
    )
    minimum_detail_rms = float(
        quality_gates.get("minimum_detail_rms_retention", 0.6)
    )
    maximum_detail_rms = float(
        quality_gates.get("maximum_detail_rms_retention", 2.0)
    )
    maximum_edge_p99 = float(
        quality_gates.get("maximum_output_edge_p99_ratio", 12.0)
    )
    maximum_sparse_reversal_physical_ratio = min(maximum_edge_p99, 4.0)
    maximum_edge = float(quality_gates.get("maximum_output_edge_ratio", 24.0))
    minimum_span = float(quality_gates.get("minimum_height_span_ratio", 0.5))
    maximum_span = float(quality_gates.get("maximum_height_span_ratio", 1.15))
    maximum_correction = float(
        quality_gates.get("maximum_correction_span_ratio", 0.9)
    )
    cardinal_p99 = float(np.percentile(cardinal_ratio, 99.0))
    cardinal_max = float(np.max(cardinal_ratio))
    cardinal_excess_p99 = float(np.percentile(cardinal_excess_ratio, 99.0))
    cardinal_excess_max = float(np.max(cardinal_excess_ratio))
    diagonal_p99 = (
        float(np.percentile(diagonal_ratio, 99.0)) if diagonal_ratio.size else None
    )
    diagonal_max = float(np.max(diagonal_ratio)) if diagonal_ratio.size else None
    diagonal_excess_p99 = (
        float(np.percentile(diagonal_excess_ratio, 99.0))
        if diagonal_excess_ratio.size
        else None
    )
    diagonal_excess_max = (
        float(np.max(diagonal_excess_ratio)) if diagonal_excess_ratio.size else None
    )
    failures = []
    if cardinal_p99 > maximum_edge_p99:
        failures.append("cardinal_edge_p99")
    if cardinal_max > maximum_edge:
        failures.append("cardinal_edge_max")
    if cardinal_excess_p99 > maximum_edge_p99:
        failures.append("cardinal_edge_excess_p99")
    if cardinal_excess_max > maximum_edge:
        failures.append("cardinal_edge_excess_max")
    if reject_direction_reversals and (
        (cardinal_reversal_count and total_reversal_count >= minimum_reversal_support_edges)
        or cardinal_reversal_max_physical_ratio
        > maximum_sparse_reversal_physical_ratio
    ):
        failures.append("cardinal_edge_direction_reversal")
    if diagonal_p99 is not None and diagonal_p99 > maximum_edge_p99:
        failures.append("diagonal_edge_p99")
    if diagonal_max is not None and diagonal_max > maximum_edge:
        failures.append("diagonal_edge_max")
    if diagonal_excess_p99 is not None and diagonal_excess_p99 > maximum_edge_p99:
        failures.append("diagonal_edge_excess_p99")
    if diagonal_excess_max is not None and diagonal_excess_max > maximum_edge:
        failures.append("diagonal_edge_excess_max")
    if reject_direction_reversals and (
        (diagonal_reversal_count and total_reversal_count >= minimum_reversal_support_edges)
        or diagonal_reversal_max_physical_ratio
        > maximum_sparse_reversal_physical_ratio
    ):
        failures.append("diagonal_edge_direction_reversal")
    if not minimum_span <= height_span_ratio <= maximum_span:
        failures.append("height_span_ratio")
    if correction_span_ratio > maximum_correction:
        failures.append("correction_span_ratio")
    if detail_region_mask is not None and not detail_stats.get("available", False):
        failures.append("detail_metric_unavailable")
    component_records = detail_stats.get("components", [])
    if detail_stats.get("measured_component_count", 0):
        if detail_stats.get("flat_reference_violation", False):
            failures.append("detail_flat_reference_violation")
        aggregate_correlation = detail_stats.get("correlation")
        if (
            aggregate_correlation is None
            or not np.isfinite(float(aggregate_correlation))
            or float(aggregate_correlation) < minimum_detail_correlation
        ):
            failures.append("detail_correlation")
        aggregate_rms = detail_stats.get("rms_retention")
        if (
            aggregate_rms is None
            or not np.isfinite(float(aggregate_rms))
            or not minimum_detail_rms
            <= float(aggregate_rms)
            <= maximum_detail_rms
        ):
            failures.append("detail_rms_retention")
        if any(
            not _detail_component_passes(
                record,
                minimum_detail_correlation,
                minimum_detail_rms,
                maximum_detail_rms,
            )
            for record in component_records
        ):
            failures.append("detail_component_quality")

    stats.update(
        {
            "enabled": not failures,
            "reason": None if not failures else "quality_gate",
            "changed_pixels": int(np.count_nonzero(changed)),
            "cardinal_edge_count": int(output_edges.size),
            "baseline_exempt_cardinal_edge_count": int(
                np.count_nonzero(horizontal_all)
                + np.count_nonzero(vertical_all)
                - output_edges.size
            ),
            "diagonal_edge_count": int(diagonal_edges.size),
            "output_edge_ratio_p99": cardinal_p99,
            "output_edge_ratio_max": cardinal_max,
            "output_edge_excess_ratio_p99": cardinal_excess_p99,
            "output_edge_excess_ratio_max": cardinal_excess_max,
            "cardinal_edge_direction_reversal_count": cardinal_reversal_count,
            "cardinal_edge_direction_reversal_max_physical_ratio": (
                cardinal_reversal_max_physical_ratio
            ),
            "physical_output_edge_ratio_p99": float(
                np.percentile(physical_cardinal_ratio, 99.0)
            ),
            "physical_output_edge_ratio_max": float(
                np.max(physical_cardinal_ratio)
            ),
            "diagonal_edge_ratio_p99": diagonal_p99,
            "diagonal_edge_ratio_max": diagonal_max,
            "diagonal_edge_excess_ratio_p99": diagonal_excess_p99,
            "diagonal_edge_excess_ratio_max": diagonal_excess_max,
            "diagonal_edge_direction_reversal_count": diagonal_reversal_count,
            "diagonal_edge_direction_reversal_max_physical_ratio": (
                diagonal_reversal_max_physical_ratio
            ),
            "total_edge_direction_reversal_count": total_reversal_count,
            "minimum_direction_reversal_support_edges": (
                minimum_reversal_support_edges
            ),
            "maximum_sparse_direction_reversal_physical_ratio": (
                maximum_sparse_reversal_physical_ratio
            ),
            "physical_diagonal_edge_ratio_p99": (
                float(np.percentile(physical_diagonal_ratio, 99.0))
                if physical_diagonal_ratio.size
                else None
            ),
            "physical_diagonal_edge_ratio_max": (
                float(np.max(physical_diagonal_ratio))
                if physical_diagonal_ratio.size
                else None
            ),
            "height_span_ratio": height_span_ratio,
            "correction_span_ratio": correction_span_ratio,
            "detail_preservation": detail_stats,
            "quality_gates": {
                **dict(quality_gates),
                "passed": not failures,
                "failures": failures,
            },
        }
    )
    return stats


def _attenuate_direction_reversals_to_source(
    source_values,
    candidate_values,
    adjustable_region_mask,
    *,
    max_neighbor_step_mm,
    iterations=64,
):
    """Locally pull reversed compressed edges toward the accepted source."""
    source = np.asarray(source_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    stats = {
        "enabled": False,
        "method": "local_source_delta_attenuation",
        "iterations": 0,
        "corrected_pixels": 0,
    }
    if source.shape != candidate.shape:
        stats["reason"] = "shape_mismatch"
        return candidate_values, stats
    try:
        max_step = float(max_neighbor_step_mm)
    except (TypeError, ValueError):
        max_step = 0.0
    if not np.isfinite(max_step) or max_step <= 0:
        stats["reason"] = "invalid_max_neighbor_step"
        return candidate_values, stats
    valid = np.isfinite(source) & np.isfinite(candidate)
    if not np.array_equal(np.isfinite(source), np.isfinite(candidate)):
        stats["reason"] = "finite_coverage_mismatch"
        return candidate_values, stats
    adjustable = _resize_binary_mask(adjustable_region_mask, source.shape) & valid
    if not np.any(adjustable):
        stats["reason"] = "empty_adjustable_region"
        return candidate_values, stats

    work = candidate.copy()
    corrected = np.zeros(source.shape, dtype=bool)

    def reversal_masks(values):
        horizontal_valid = valid[:, :-1] & valid[:, 1:]
        vertical_valid = valid[:-1, :] & valid[1:, :]
        diagonal_down_valid = valid[:-1, :-1] & valid[1:, 1:]
        diagonal_up_valid = valid[:-1, 1:] & valid[1:, :-1]
        source_horizontal = source[:, 1:] - source[:, :-1]
        output_horizontal = values[:, 1:] - values[:, :-1]
        source_vertical = source[1:, :] - source[:-1, :]
        output_vertical = values[1:, :] - values[:-1, :]
        source_diagonal_down = source[1:, 1:] - source[:-1, :-1]
        output_diagonal_down = values[1:, 1:] - values[:-1, :-1]
        source_diagonal_up = source[1:, :-1] - source[:-1, 1:]
        output_diagonal_up = values[1:, :-1] - values[:-1, 1:]
        diagonal_step = max_step * np.sqrt(2.0)
        return (
            horizontal_valid
            & (source_horizontal * output_horizontal < 0.0)
            & (np.abs(source_horizontal) > max_step * 0.25)
            & (np.abs(output_horizontal) > max_step),
            vertical_valid
            & (source_vertical * output_vertical < 0.0)
            & (np.abs(source_vertical) > max_step * 0.25)
            & (np.abs(output_vertical) > max_step),
            diagonal_down_valid
            & (source_diagonal_down * output_diagonal_down < 0.0)
            & (np.abs(source_diagonal_down) > diagonal_step * 0.25)
            & (np.abs(output_diagonal_down) > diagonal_step),
            diagonal_up_valid
            & (source_diagonal_up * output_diagonal_up < 0.0)
            & (np.abs(source_diagonal_up) > diagonal_step * 0.25)
            & (np.abs(output_diagonal_up) > diagonal_step),
        )

    def reversal_counts(masks):
        cardinal = int(np.count_nonzero(masks[0]) + np.count_nonzero(masks[1]))
        diagonal = int(np.count_nonzero(masks[2]) + np.count_nonzero(masks[3]))
        return cardinal, diagonal

    initial_masks = reversal_masks(work)
    initial_cardinal, initial_diagonal = reversal_counts(initial_masks)
    stats.update(
        {
            "enabled": bool(initial_cardinal or initial_diagonal),
            "initial_cardinal_reversals": initial_cardinal,
            "initial_diagonal_reversals": initial_diagonal,
        }
    )
    if not stats["enabled"]:
        stats["reason"] = "no_direction_reversals"
        stats["passed"] = True
        return candidate_values, stats

    for iteration in range(max(1, int(iterations))):
        masks = reversal_masks(work)
        cardinal, diagonal = reversal_counts(masks)
        if not cardinal and not diagonal:
            stats["iterations"] = iteration
            break
        marked = np.zeros(source.shape, dtype=bool)
        marked[:, :-1] |= masks[0]
        marked[:, 1:] |= masks[0]
        marked[:-1, :] |= masks[1]
        marked[1:, :] |= masks[1]
        marked[:-1, :-1] |= masks[2]
        marked[1:, 1:] |= masks[2]
        marked[:-1, 1:] |= masks[3]
        marked[1:, :-1] |= masks[3]
        marked &= adjustable
        if not np.any(marked):
            stats["reason"] = "reversals_outside_adjustable_region"
            stats["iterations"] = iteration
            break
        corrected |= marked
        work[marked] = source[marked] + 0.5 * (work[marked] - source[marked])
        stats["iterations"] = iteration + 1

    final_masks = reversal_masks(work)
    final_cardinal, final_diagonal = reversal_counts(final_masks)
    correction = np.abs(work - candidate)
    stats.update(
        {
            "corrected_pixels": int(np.count_nonzero(corrected)),
            "final_cardinal_reversals": final_cardinal,
            "final_diagonal_reversals": final_diagonal,
            "correction_p95_mm": (
                float(np.percentile(correction[corrected], 95.0))
                if np.any(corrected)
                else 0.0
            ),
            "correction_max_mm": (
                float(np.max(correction[corrected])) if np.any(corrected) else 0.0
            ),
            "passed": bool(final_cardinal == 0 and final_diagonal == 0),
        }
    )
    if not stats["passed"] and "reason" not in stats:
        stats["reason"] = "iteration_limit"
    return work.astype(candidate.dtype, copy=False), stats


def _blend_updates_outside_protected_region(
    baseline_values,
    candidate_values,
    protected_region_mask,
    feather_pixels=6.0,
):
    baseline = np.asarray(baseline_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if baseline.shape != candidate.shape:
        raise ValueError("Protected-region blend surfaces must have identical shapes")
    if protected_region_mask is None:
        return candidate_values, {"enabled": False, "reason": "no_protected_region"}

    protected = _resize_binary_mask(protected_region_mask, baseline.shape)
    if not np.any(protected):
        return candidate_values, {"enabled": False, "reason": "empty_protected_region"}
    feather = max(float(feather_pixels), 1.0)
    candidate_weight = np.clip(distance_transform_edt(~protected) / feather, 0.0, 1.0)
    candidate_weight[protected] = 0.0
    finite = np.isfinite(baseline) & np.isfinite(candidate)
    blended = baseline.copy()
    blended[finite] = (
        baseline[finite]
        + candidate_weight[finite] * (candidate[finite] - baseline[finite])
    )
    correction = np.abs(blended - baseline)
    return blended.astype(candidate.dtype, copy=False), {
        "enabled": True,
        "protected_pixels": int(np.count_nonzero(protected)),
        "feather_pixels": feather,
        "protected_correction_max_mm": float(np.max(correction[protected])),
        "outside_correction_max_mm": float(np.max(correction[~protected & finite]))
        if np.any(~protected & finite)
        else 0.0,
    }


def _blend_updates_inside_region(
    baseline_values,
    candidate_values,
    active_region_mask,
    feather_pixels=6.0,
):
    """Keep solver corrections inside a feathered subject region."""
    baseline = np.asarray(baseline_values, dtype=np.float32)
    candidate = np.asarray(candidate_values, dtype=np.float32)
    if baseline.shape != candidate.shape:
        raise ValueError("Active-region blend surfaces must have identical shapes")
    if active_region_mask is None:
        return baseline_values, {"enabled": False, "reason": "no_active_region"}

    active = _resize_binary_mask(active_region_mask, baseline.shape)
    if not np.any(active):
        return baseline_values, {"enabled": False, "reason": "empty_active_region"}
    feather = max(float(feather_pixels), 1.0)
    outside_distance = distance_transform_edt(~active)
    candidate_weight = np.clip(1.0 - outside_distance / feather, 0.0, 1.0)
    candidate_weight[active] = 1.0
    finite = np.isfinite(baseline) & np.isfinite(candidate)
    blended = baseline.copy()
    blended[finite] = (
        baseline[finite]
        + candidate_weight[finite] * (candidate[finite] - baseline[finite])
    )
    correction = np.abs(blended - baseline)
    return blended.astype(candidate.dtype, copy=False), {
        "enabled": True,
        "active_pixels": int(np.count_nonzero(active)),
        "feather_pixels": feather,
        "inside_correction_max_mm": float(np.max(correction[active & finite]))
        if np.any(active & finite)
        else 0.0,
        "outside_correction_max_mm": float(np.max(correction[~active & finite]))
        if np.any(~active & finite)
        else 0.0,
    }


def _compress_selected_relief_surface(
    values,
    selected_region_mask,
    *,
    sample_pitch_mm,
    max_slope_mm_per_mm,
    sample_pitch_source,
    protected_region_mask=None,
):
    selected = _resize_binary_mask(selected_region_mask, np.asarray(values).shape)
    protected = (
        _resize_binary_mask(protected_region_mask, selected.shape)
        if protected_region_mask is not None
        else np.zeros(selected.shape, dtype=bool)
    )
    raw_detail_region = selected & ~protected
    detail_components, detail_component_count = label(
        raw_detail_region,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    detail_region = np.zeros(selected.shape, dtype=bool)
    dropped_detail_components = 0
    dropped_detail_pixels = 0
    for component_index in range(1, detail_component_count + 1):
        component = detail_components == component_index
        interior = binary_erosion(
            component,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        if np.count_nonzero(interior) >= 16:
            detail_region |= component
        else:
            dropped_detail_components += 1
            dropped_detail_pixels += int(np.count_nonzero(component))
    if np.count_nonzero(detail_region) < 16:
        stats = {
            "enabled": False,
            "reason": "no_unprotected_selection_detail",
            "selected_pixels": int(np.count_nonzero(selected)),
            "protected_pixels": int(np.count_nonzero(protected)),
            "dropped_detail_components": int(dropped_detail_components),
            "dropped_detail_pixels": int(dropped_detail_pixels),
            "sample_pitch_source": sample_pitch_source,
        }
        if np.any(protected):
            return values, stats, None
        limited, slope_stats = _limit_positive_relief_slope(
            values,
            sample_pitch_mm=sample_pitch_mm,
            max_slope_mm_per_mm=max_slope_mm_per_mm,
            structural_region_mask=selected,
        )
        limited, localization_stats = _blend_updates_inside_region(
            values,
            limited,
            selected,
            feather_pixels=max(2.0, 1.5 / max(float(sample_pitch_mm), 1e-6)),
        )
        stats["selected_region_blend"] = localization_stats
        return limited, stats, slope_stats

    selection_surface, stats = _compress_relief_gradients(
        values,
        sample_pitch_mm=sample_pitch_mm,
        max_slope_mm_per_mm=max_slope_mm_per_mm,
        structural_region_mask=selected,
        detail_region_mask=detail_region,
        screening_weight=HIGH_RELIEF_SELECTION_SCREENING_WEIGHT,
        minimum_detail_correlation=0.65,
        minimum_detail_rms_retention=0.15,
        detail_gradient_retention=HIGH_RELIEF_SELECTION_DETAIL_GRADIENT_RETENTION,
        maximum_detail_gradient_ratio=8.0,
        allow_edge_only_candidate=True,
    )
    stats["sample_pitch_source"] = sample_pitch_source
    stats["selected_pixels"] = int(np.count_nonzero(selected))
    stats["protected_pixels"] = int(np.count_nonzero(protected))
    stats["dropped_detail_components"] = int(dropped_detail_components)
    stats["dropped_detail_pixels"] = int(dropped_detail_pixels)
    if stats.get("enabled", False):
        pre_restoration_detail = stats.get("detail_preservation", {})
        restored_surface, restoration_stats = _restore_face_laplacian_detail(
            selection_surface,
            values,
            detail_region,
            max_neighbor_step_mm=stats.get("max_neighbor_step_mm"),
            max_correction_mm=1.0,
        )
        restored_detail = _face_detail_preservation_metrics(
            values,
            restored_surface,
            detail_region,
        )
        components = restored_detail.get("components", [])
        restoration_accepted = bool(
            restoration_stats.get("enabled", False)
            and components
            and all(_detail_component_passes(record, 0.7, 0.2, 2.0) for record in components)
            and float(restoration_stats.get("applied_correction_max_mm", float("inf")))
            <= 1.0001
            and float(restoration_stats.get("boundary_correction_max_mm", float("inf")))
            <= 1e-5
        )
        restoration_stats["accepted"] = restoration_accepted
        restoration_stats["detail_preservation"] = restored_detail
        stats["pre_restoration_detail_preservation"] = pre_restoration_detail
        stats["post_solve_detail_restoration"] = restoration_stats
        if restoration_accepted:
            selection_surface = restored_surface
            stats["detail_preservation"] = restored_detail
        if np.any(protected):
            selection_surface, protection_stats = _blend_updates_outside_protected_region(
                values,
                selection_surface,
                protected,
            )
            stats["protected_region_blend"] = protection_stats
        selection_surface, localization_stats = _blend_updates_inside_region(
            values,
            selection_surface,
            selected,
            feather_pixels=max(2.0, 1.5 / max(float(sample_pitch_mm), 1e-6)),
        )
        stats["selected_region_blend"] = localization_stats
        pre_blend_quality_gates = dict(stats.get("quality_gates", {}))
        post_blend_audit = _audit_bounded_compression_surface(
            values,
            selection_surface,
            detail_region,
            sample_pitch_mm=sample_pitch_mm,
            max_slope_mm_per_mm=max_slope_mm_per_mm,
            quality_gates=pre_blend_quality_gates,
            reject_direction_reversals=bool(
                stats.get("provisional_edge_only_candidate", False)
            ),
        )
        stats["pre_blend_quality_gates"] = pre_blend_quality_gates
        stats["post_blend_quality_audit"] = post_blend_audit
        if post_blend_audit.get("enabled", False):
            stats["enabled"] = True
            stats["reason"] = None
            stats["quality_gates"] = post_blend_audit["quality_gates"]
            for metric_name in (
                "changed_pixels",
                "cardinal_edge_count",
                "baseline_exempt_cardinal_edge_count",
                "diagonal_edge_count",
                "output_edge_ratio_p99",
                "output_edge_ratio_max",
                "output_edge_excess_ratio_p99",
                "output_edge_excess_ratio_max",
                "cardinal_edge_direction_reversal_count",
                "cardinal_edge_direction_reversal_max_physical_ratio",
                "physical_output_edge_ratio_p99",
                "physical_output_edge_ratio_max",
                "diagonal_edge_ratio_p99",
                "diagonal_edge_ratio_max",
                "diagonal_edge_excess_ratio_p99",
                "diagonal_edge_excess_ratio_max",
                "diagonal_edge_direction_reversal_count",
                "diagonal_edge_direction_reversal_max_physical_ratio",
                "total_edge_direction_reversal_count",
                "minimum_direction_reversal_support_edges",
                "maximum_sparse_direction_reversal_physical_ratio",
                "physical_diagonal_edge_ratio_p99",
                "physical_diagonal_edge_ratio_max",
                "height_span_ratio",
                "correction_span_ratio",
                "detail_preservation",
                "source_finite_samples",
                "candidate_finite_samples",
                "finite_mask_match",
                "candidate_finite_coverage_ratio",
            ):
                if metric_name in post_blend_audit:
                    stats[metric_name] = post_blend_audit[metric_name]
            stats["provisional_edge_only_candidate_accepted"] = bool(
                stats.get("provisional_edge_only_candidate", False)
            )
            return selection_surface, stats, stats
        stats["enabled"] = False
        stats["reason"] = "quality_gate"
        stats["post_blend_rejection_reason"] = "post_blend_quality_gate"
        stats["quality_gates"] = post_blend_audit.get(
            "quality_gates",
            pre_blend_quality_gates,
        )

    if np.any(protected):
        return values, stats, None
    limited, slope_stats = _limit_positive_relief_slope(
        values,
        sample_pitch_mm=sample_pitch_mm,
        max_slope_mm_per_mm=max_slope_mm_per_mm,
        structural_region_mask=selected,
    )
    limited, localization_stats = _blend_updates_inside_region(
        values,
        limited,
        selected,
        feather_pixels=max(2.0, 1.5 / max(float(sample_pitch_mm), 1e-6)),
    )
    stats["selected_region_blend"] = localization_stats
    return limited, stats, slope_stats


def _add_triangle(faces, a, b, c):
    faces.append([a, b, c])


def _signed_triangle_volume(triangles):
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return 0.0
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def _diagnostic_signed_volume(triangles):
    try:
        import trimesh

        # Match the signed-volume convention used by benchmark diagnostics.
        triangles = np.asarray(triangles, dtype=np.float64)
        vertices = triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
        return float(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).volume)
    except Exception:
        import warnings

        warnings.warn(
            "Trimesh signed-volume check failed; falling back to a local STL volume convention.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _signed_triangle_volume(triangles)


def _force_positive_stl_volume(stl_mesh):
    volume = _diagnostic_signed_volume(stl_mesh.vectors)
    if np.isfinite(volume) and volume < 0:
        stl_mesh.vectors[:] = stl_mesh.vectors[:, [0, 2, 1], :]
        if hasattr(stl_mesh, "update_normals"):
            stl_mesh.update_normals()
    return stl_mesh


def depth_data_to_3d_model(
    npy_file,
    output_stl_path='output_3d_model.stl',
    target_dimension=300,
    z_scale=50,
    invert=False,
    sigma=0.6,
    max_xy_size=None,
    relief_gamma=0.75,
    detail_boost=0.8,
    detail_radius=2.0,
    detail_edge_threshold=0.12,
    low_percentile=1.0,
    high_percentile=99.0,
    base_border_px=2,
    value_transform="auto",
    minimum_feature_mm=0.8,
    max_relief_slope=2.0,
    face_region_mask=None,
    selection_region_mask=None,
    selection_background_depth_ratio=DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    selection_subject_lock=False,
    background_detail_boost=1.0,
    source_image=None,
    background_photo_detail_mm=0.0,
    trim_top_background=False,
    feature_weight_mask=None,
    printable_feature_depth_mm=0.0,
    feature_bridge_depth_mm=0.0,
    feature_exclusion_mask=None,
    surface_output_path=None,
    reference_surface_output_path=None,
    normalization_reference_depth=None,
):
    # Load the .npy file
    data = np.load(npy_file).astype(np.float32)
    input_depth_shape = [int(data.shape[0]), int(data.shape[1])]
    normalization_reference = None
    if normalization_reference_depth is not None:
        if isinstance(
            normalization_reference_depth,
            (str, os.PathLike),
        ):
            normalization_reference = np.load(
                normalization_reference_depth
            ).astype(np.float32)
        else:
            normalization_reference = np.asarray(
                normalization_reference_depth,
                dtype=np.float32,
            )
        normalization_reference = np.squeeze(normalization_reference)
        if (
            normalization_reference.ndim != 2
            or normalization_reference.shape != data.shape
        ):
            raise ValueError(
                "Normalization reference depth must match the input depth grid"
            )
    region_mask = None
    if face_region_mask is not None:
        if isinstance(face_region_mask, (str, os.PathLike)):
            from PIL import Image

            region_mask = np.asarray(Image.open(face_region_mask).convert("L")) > 0
        else:
            region_mask = np.asarray(face_region_mask) > 0
        region_mask = _resize_binary_mask(region_mask, data.shape)
    selected_region = None
    if selection_region_mask is not None:
        if isinstance(selection_region_mask, (str, os.PathLike)):
            from PIL import Image

            selected_region = np.asarray(Image.open(selection_region_mask).convert("L")) > 0
        else:
            selected_region = np.asarray(selection_region_mask) > 0
        selected_region = _resize_binary_mask(selected_region, data.shape)

    # Skip downsampling if target_dimension is -1
    if target_dimension != -1:
        target_shape = _target_shape_for_max_dimension(data.shape, target_dimension)
        if target_shape != data.shape:
            print(f"Resizing depth grid from {data.shape} to {target_shape}")
            data = _resize_nan_aware(data, target_shape)
            if normalization_reference is not None:
                normalization_reference = _resize_nan_aware(
                    normalization_reference,
                    target_shape,
                )
            if region_mask is not None:
                region_mask = _resize_binary_mask(region_mask, target_shape)
            if selected_region is not None:
                selected_region = _resize_binary_mask(selected_region, target_shape)
        else:
            print(f"Keeping depth grid resolution: {data.shape}")
    else:
        print("Skipping downsampling as target_dimension is -1")
    target_depth_shape = [int(data.shape[0]), int(data.shape[1])]

    # Flip the x axis
    data = np.flip(data, axis=1)
    if normalization_reference is not None:
        normalization_reference = np.flip(
            normalization_reference,
            axis=1,
        )
    if region_mask is not None:
        region_mask = np.flip(region_mask, axis=1)
    if selected_region is not None:
        selected_region = np.flip(selected_region, axis=1)
    subject_surface_locked = bool(
        selection_subject_lock and selected_region is not None
    )
    if trim_top_background:
        top_silhouette_mask, top_silhouette_stats = _top_silhouette_mask(source_image, data.shape)
    else:
        top_silhouette_mask = np.ones(data.shape, dtype=bool)
        top_silhouette_stats = {"enabled": False, "reason": "disabled"}

    detail_protection_mask = None
    if region_mask is not None or (
        selected_region is not None and not subject_surface_locked
    ):
        detail_protection_mask = np.zeros(data.shape, dtype=bool)
        if region_mask is not None:
            detail_protection_mask |= region_mask
        if selected_region is not None and not subject_surface_locked:
            detail_protection_mask |= selected_region

    resolved_value_transform = _resolve_relief_value_transform(npy_file, value_transform)
    relief = _shape_relief_values(
        data,
        invert=invert,
        gamma=relief_gamma,
        detail_boost=detail_boost,
        detail_radius=detail_radius,
        detail_edge_threshold=detail_edge_threshold,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        value_transform=resolved_value_transform,
        detail_protection_mask=detail_protection_mask,
        background_detail_boost=background_detail_boost,
        normalization_mask=(
            selected_region
            if selected_region is not None
            and float(selection_background_depth_ratio) > 0
            and not subject_surface_locked
            else None
        ),
        normalization_reference_values=normalization_reference,
    )
    requested_photo_detail_mm = float(background_photo_detail_mm)
    has_photo_detail_protection = _has_background_photo_detail_protection(
        detail_protection_mask
    )
    effective_photo_detail_mm = _effective_background_photo_detail_mm(
        requested_photo_detail_mm,
        detail_protection_mask,
    )
    photo_detail_ratio = (
        effective_photo_detail_mm / float(z_scale)
        if z_scale and effective_photo_detail_mm > 0
        else 0.0
    )
    photo_detail_halo_mm = BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM
    input_sample_pitch_mm, photo_detail_halo_px = _photo_detail_sampling(
        max_xy_size,
        relief.shape,
        photo_detail_halo_mm,
    )
    relief, photo_detail_stats = _inject_photo_relief_detail(
        relief,
        source_image,
        max_detail_ratio=photo_detail_ratio,
        protection_mask=(
            detail_protection_mask if has_photo_detail_protection else None
        ),
        protection_halo_px=photo_detail_halo_px,
    )
    photo_detail_stats.update(
        {
            "requested_detail_mm": requested_photo_detail_mm,
            "effective_detail_mm": effective_photo_detail_mm,
            "unprotected_detail_limited": bool(
                not has_photo_detail_protection
                and effective_photo_detail_mm < requested_photo_detail_mm
            ),
            "protection_halo_mm": (
                photo_detail_halo_mm if has_photo_detail_protection else 0.0
            ),
            "input_sample_pitch_mm": input_sample_pitch_mm,
        }
    )
    relief = np.where(top_silhouette_mask, relief, np.nan)
    border_detail_region = region_mask if region_mask is not None else selected_region
    preserve_detail_border = (
        border_detail_region
        if selected_region is not None
        and float(selection_background_depth_ratio) > 0
        and not subject_surface_locked
        else None
    )
    relief = _flatten_border(
        relief,
        base_border_px,
        preserve_mask=preserve_detail_border,
    )
    relief = np.where(top_silhouette_mask, relief, np.nan)
    relief, print_filter_stats = _prepare_relief_for_printing(
        relief,
        max_xy_size=max_xy_size,
        minimum_feature_mm=minimum_feature_mm,
    )
    mesh_shape_before_crop = [int(relief.shape[0]), int(relief.shape[1])]
    gradient_sample_pitch_mm = print_filter_stats["mesh_sample_pitch_mm"]
    gradient_sample_pitch_source = "physical_size"
    if gradient_sample_pitch_mm is None and max_xy_size is None:
        # Without an explicit physical footprint, STL X/Y coordinates are the
        # integer grid coordinates below, so adjacent samples are one unit apart.
        gradient_sample_pitch_mm = 1.0
        gradient_sample_pitch_source = "implicit_stl_grid_unit"
    if region_mask is not None:
        region_mask = _resize_binary_mask(region_mask, relief.shape)
    if selected_region is not None:
        selected_region = _resize_binary_mask(selected_region, relief.shape)
    border_detail_region = region_mask if region_mask is not None else selected_region
    preserve_detail_border = (
        border_detail_region
        if selected_region is not None
        and float(selection_background_depth_ratio) > 0
        and not subject_surface_locked
        else None
    )
    top_silhouette_mask = _resize_binary_mask(top_silhouette_mask, relief.shape)
    relief = np.where(top_silhouette_mask, relief, np.nan)
    relief = _flatten_border(
        relief,
        base_border_px,
        preserve_mask=preserve_detail_border,
    )
    relief = np.where(top_silhouette_mask, relief, np.nan)
    z = relief * z_scale
    
    # Add a small offset to create buffer
    z = z + 0.01

    # Apply a Gaussian filter to smooth the data
    z = _smooth_nan_aware(z, sigma=sigma)
    z = np.where(top_silhouette_mask, z, np.nan)
    if base_border_px:
        z = _flatten_border(
            np.maximum(z - 0.01, 0.0),
            base_border_px,
            preserve_mask=preserve_detail_border,
        ) + 0.01
        z = np.where(top_silhouette_mask, z, np.nan)
    reference_face_height_mm = 12.0
    face_reference_surface = np.where(
        np.isfinite(z),
        (z - 0.01) * min(1.0, reference_face_height_mm / max(float(z_scale), 1e-6)) + 0.01,
        np.nan,
    )
    head_region_mask, head_region_stats = _expand_face_region_to_depth_connected_head(
        region_mask,
        face_reference_surface,
    )
    face_translation_core_mask, face_translation_core_stats = _face_translation_core_mask(
        region_mask,
        z.shape,
    )
    post_feature_slope_guard_stats = {
        "enabled": False,
        "reason": "no_feature_update",
    }
    face_detail_guard_stats = {
        "enabled": False,
        "reason": "height_within_reference",
    }
    unstabilized_scene = z.copy()
    z, face_height_stabilization_stats = _stabilize_face_relief_height(
        z,
        head_region_mask,
        relief_height_mm=z_scale,
        reference_face_height_mm=reference_face_height_mm,
    )

    selection_gradient_compression_stats = {
        "enabled": False,
        "reason": "not_requested",
    }
    if face_height_stabilization_stats.get("enabled", False):
        face_boundary_attachment_stats = {
            "enabled": False,
            "reason": "reference_boundary_transfer",
        }
        gradient_surface, gradient_compression_stats = _compress_relief_gradients(
            unstabilized_scene,
            sample_pitch_mm=gradient_sample_pitch_mm,
            max_slope_mm_per_mm=max_relief_slope,
            structural_region_mask=head_region_mask,
            detail_region_mask=region_mask,
            screening_weight=HIGH_RELIEF_FACE_SCREENING_WEIGHT,
            minimum_detail_correlation=HIGH_RELIEF_FACE_MIN_DETAIL_CORRELATION,
        )
        primary_gradient_compression_stats = gradient_compression_stats
        primary_quality_failures = set(
            primary_gradient_compression_stats.get("quality_gates", {}).get(
                "failures", []
            )
        )
        adaptive_screening_retry_stats = {
            "attempted": False,
            "reason": "primary_candidate_accepted",
            "trigger_failures": sorted(primary_quality_failures),
            "primary_screening_weight": float(HIGH_RELIEF_FACE_SCREENING_WEIGHT),
            "retry_screening_weight": float(
                HIGH_RELIEF_FACE_CARDINAL_EDGE_RETRY_WEIGHT
            ),
            "detail_retry_screening_weight": float(
                HIGH_RELIEF_FACE_DETAIL_RETRY_WEIGHT
            ),
        }
        if not gradient_compression_stats.get("enabled", False):
            adaptive_screening_retry_stats["reason"] = "ineligible_quality_failure"
            detail_retry_trigger_failures = set(primary_quality_failures)
            if primary_quality_failures in (
                {"cardinal_edge_p99"},
                {"cardinal_edge_max"},
            ):
                retry_surface, retry_stats = _compress_relief_gradients(
                    unstabilized_scene,
                    sample_pitch_mm=gradient_sample_pitch_mm,
                    max_slope_mm_per_mm=max_relief_slope,
                    structural_region_mask=head_region_mask,
                    detail_region_mask=region_mask,
                    screening_weight=HIGH_RELIEF_FACE_CARDINAL_EDGE_RETRY_WEIGHT,
                    minimum_detail_correlation=(
                        HIGH_RELIEF_FACE_MIN_DETAIL_CORRELATION
                    ),
                    allow_edge_only_candidate=True,
                )
                retry_failures = retry_stats.get("quality_gates", {}).get(
                    "failures", []
                )
                adaptive_screening_retry_stats.update(
                    {
                        "attempted": True,
                        "reason": (
                            "retry_candidate_accepted"
                            if retry_stats.get("enabled", False)
                            else "retry_candidate_rejected"
                        ),
                        "primary_output_edge_ratio_p99": (
                            primary_gradient_compression_stats.get(
                                "output_edge_ratio_p99"
                            )
                        ),
                        "retry_quality_failures": list(retry_failures),
                        "retry_output_edge_ratio_p99": retry_stats.get(
                            "output_edge_ratio_p99"
                        ),
                    }
                )
                if retry_stats.get("enabled", False):
                    gradient_surface = retry_surface
                    gradient_compression_stats = retry_stats
                else:
                    detail_retry_trigger_failures = set(retry_failures)
            edge_failures = {
                "cardinal_edge_p99",
                "cardinal_edge_max",
                "diagonal_edge_p99",
                "diagonal_edge_max",
            }
            detail_failures = {
                "detail_correlation",
                "detail_rms_retention",
                "detail_component_correlation",
                "detail_component_rms_retention",
            }
            eligible_detail_retry = bool(
                not gradient_compression_stats.get("enabled", False)
                and detail_retry_trigger_failures & detail_failures
                and detail_retry_trigger_failures
                <= edge_failures | detail_failures
            )
            if eligible_detail_retry:
                retry_surface, retry_stats = _compress_relief_gradients(
                    unstabilized_scene,
                    sample_pitch_mm=gradient_sample_pitch_mm,
                    max_slope_mm_per_mm=max_relief_slope,
                    structural_region_mask=head_region_mask,
                    detail_region_mask=region_mask,
                    screening_weight=HIGH_RELIEF_FACE_DETAIL_RETRY_WEIGHT,
                    minimum_detail_correlation=(
                        HIGH_RELIEF_FACE_MIN_DETAIL_CORRELATION
                    ),
                    allow_edge_only_candidate=True,
                )
                retry_failures = retry_stats.get("quality_gates", {}).get(
                    "failures", []
                )
                adaptive_screening_retry_stats.update(
                    {
                        "attempted": True,
                        "kind": "high_detail_edge_audited",
                        "detail_retry_trigger_failures": sorted(
                            detail_retry_trigger_failures
                        ),
                        "reason": (
                            "retry_candidate_pending_post_blend_audit"
                            if retry_stats.get("enabled", False)
                            else "retry_candidate_rejected"
                        ),
                        "retry_quality_failures": list(retry_failures),
                        "retry_output_edge_ratio_p99": retry_stats.get(
                            "output_edge_ratio_p99"
                        ),
                        "retry_detail_correlation": retry_stats.get(
                            "detail_preservation", {}
                        ).get("correlation"),
                        "retry_detail_rms_retention": retry_stats.get(
                            "detail_preservation", {}
                        ).get("rms_retention"),
                        "retry_provisional_edge_only_candidate": bool(
                            retry_stats.get(
                                "provisional_edge_only_candidate", False
                            )
                        ),
                    }
                )
                if retry_stats.get("enabled", False):
                    gradient_surface = retry_surface
                    gradient_compression_stats = retry_stats
        gradient_compression_stats[
            "adaptive_screening_retry"
        ] = adaptive_screening_retry_stats
        face_height_stabilization_stats[
            "gradient_compression_attempt"
        ] = primary_gradient_compression_stats
        face_height_stabilization_stats[
            "gradient_compression_selected"
        ] = gradient_compression_stats
        if gradient_compression_stats.get("enabled", False):
            pre_restoration_detail = gradient_compression_stats.get(
                "detail_preservation",
                {},
            )
            restored_gradient_surface, gradient_detail_restoration_stats = (
                _restore_face_laplacian_detail(
                    gradient_surface,
                    unstabilized_scene,
                    region_mask,
                    max_neighbor_step_mm=gradient_compression_stats.get(
                        "max_neighbor_step_mm"
                    ),
                    max_correction_mm=0.6,
                )
            )
            restored_detail = _face_detail_preservation_metrics(
                unstabilized_scene,
                restored_gradient_surface,
                region_mask,
            )
            restoration_components = restored_detail.get("components", [])
            restoration_accepted = bool(
                gradient_detail_restoration_stats.get("enabled", False)
                and restoration_components
                and all(
                    _detail_component_passes(record, 0.8, 0.6, 2.0)
                    for record in restoration_components
                )
                and float(
                    gradient_detail_restoration_stats.get(
                        "applied_correction_max_mm",
                        float("inf"),
                    )
                )
                <= 0.6001
                and float(
                    gradient_detail_restoration_stats.get(
                        "boundary_correction_max_mm",
                        float("inf"),
                    )
                )
                <= 1e-5
            )
            gradient_detail_restoration_stats["accepted"] = restoration_accepted
            gradient_detail_restoration_stats["detail_preservation"] = restored_detail
            gradient_compression_stats[
                "pre_restoration_detail_preservation"
            ] = pre_restoration_detail
            gradient_compression_stats[
                "post_solve_detail_restoration"
            ] = gradient_detail_restoration_stats
            if restoration_accepted:
                gradient_surface = restored_gradient_surface
                gradient_compression_stats["detail_preservation"] = restored_detail
            face_blend_region = (
                head_region_mask if head_region_mask is not None else region_mask
            )
            if selected_region is not None:
                face_blend_region = (
                    _resize_binary_mask(face_blend_region, selected_region.shape)
                    & selected_region
                )
            gradient_surface, face_region_blend_stats = _blend_updates_inside_region(
                unstabilized_scene,
                gradient_surface,
                face_blend_region,
                feather_pixels=6.0,
            )
            if selected_region is not None:
                selection_bound = _resize_binary_mask(
                    selected_region,
                    gradient_surface.shape,
                )
                outside_selection = (
                    ~selection_bound
                    & np.isfinite(unstabilized_scene)
                    & np.isfinite(gradient_surface)
                )
                outside_selection_correction = np.abs(
                    gradient_surface - unstabilized_scene
                )
                face_region_blend_stats[
                    "outside_selection_correction_before_clamp_max_mm"
                ] = (
                    float(np.max(outside_selection_correction[outside_selection]))
                    if np.any(outside_selection)
                    else 0.0
                )
                gradient_surface = np.where(
                    selection_bound,
                    gradient_surface,
                    unstabilized_scene,
                )
                face_region_blend_stats[
                    "outside_selection_correction_max_mm"
                ] = 0.0
                face_region_blend_stats["selection_bounded"] = True
            else:
                face_region_blend_stats["selection_bounded"] = False
            gradient_compression_stats["face_region_blend"] = face_region_blend_stats
            gradient_compression_stats["sample_pitch_source"] = gradient_sample_pitch_source
            direction_reversal_projection_stats = {
                "enabled": False,
                "reason": "not_provisional_edge_only_candidate",
            }
            if gradient_compression_stats.get(
                "provisional_edge_only_candidate",
                False,
            ):
                projected_gradient_surface, direction_reversal_projection_stats = (
                    _attenuate_direction_reversals_to_source(
                        unstabilized_scene,
                        gradient_surface,
                        face_blend_region,
                        max_neighbor_step_mm=gradient_compression_stats.get(
                            "max_neighbor_step_mm"
                        ),
                    )
                )
                if direction_reversal_projection_stats.get("passed", False):
                    gradient_surface = projected_gradient_surface
            gradient_compression_stats["direction_reversal_projection"] = (
                direction_reversal_projection_stats
            )
            post_blend_audit = _audit_bounded_compression_surface(
                unstabilized_scene,
                gradient_surface,
                region_mask,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                quality_gates=gradient_compression_stats["quality_gates"],
                reject_direction_reversals=bool(
                    gradient_compression_stats.get(
                        "provisional_edge_only_candidate",
                        False,
                    )
                ),
            )
            projection_required = bool(
                gradient_compression_stats.get(
                    "provisional_edge_only_candidate",
                    False,
                )
            )
            projection_passed = bool(
                direction_reversal_projection_stats.get("passed", False)
            )
            if projection_required and not projection_passed:
                post_blend_audit["enabled"] = False
                post_blend_audit["reason"] = "direction_reversal_projection_failed"
                audit_quality = post_blend_audit.setdefault("quality_gates", {})
                audit_failures = list(audit_quality.get("failures", []))
                if "direction_reversal_projection" not in audit_failures:
                    audit_failures.append("direction_reversal_projection")
                audit_quality["failures"] = audit_failures
                audit_quality["passed"] = False
            gradient_compression_stats[
                "post_blend_quality_audit"
            ] = post_blend_audit
            if post_blend_audit.get("enabled", False):
                gradient_compression_stats[
                    "provisional_edge_only_candidate_accepted"
                ] = bool(projection_required and projection_passed)
                z = gradient_surface
                processed_scene = gradient_surface
                slope_limit_stats = gradient_compression_stats
                face_boundary_attachment_stats = {
                    "enabled": False,
                    "reason": "global_gradient_reconstruction",
                }
                legacy_shape_scale = face_height_stabilization_stats.get("shape_scale")
                face_height_stabilization_stats = {
                    "enabled": True,
                    "method": "screened_gradient_domain_compression",
                    "relief_height_mm": float(z_scale),
                    "reference_face_height_mm": float(reference_face_height_mm),
                    "legacy_shape_scale": legacy_shape_scale,
                    "gradient_compression": gradient_compression_stats,
                    "gradient_compression_attempt": (
                        primary_gradient_compression_stats
                    ),
                    "gradient_compression_selected": gradient_compression_stats,
                }
                rigid_face_surface = gradient_surface
                rigid_alignment_stats = {
                    "enabled": True,
                    "method": "screened_gradient_domain_compression",
                    "selection_reason": ["face_aware_high_relief"],
                    "attachment_slope_ratio_p99": post_blend_audit.get(
                        "output_edge_ratio_p99"
                    ),
                    "attachment_slope_ratio_max": post_blend_audit.get(
                        "output_edge_ratio_max"
                    ),
                    "slope_projection": [],
                    "gradient_compression": gradient_compression_stats,
                }
            else:
                gradient_compression_stats["enabled"] = False
                gradient_compression_stats["reason"] = "post_blend_quality_gate"
                z = np.where(top_silhouette_mask, z, np.nan)
                processed_scene, slope_limit_stats = _limit_positive_relief_slope(
                    z,
                    sample_pitch_mm=gradient_sample_pitch_mm,
                    max_slope_mm_per_mm=max_relief_slope,
                    structural_region_mask=head_region_mask,
                )
                rigid_face_surface, rigid_alignment_stats = (
                    _align_stabilized_head_to_reference_boundary(
                        z,
                        face_reference_surface,
                        processed_scene,
                        head_region_mask,
                        protected_core_mask=face_translation_core_mask,
                        protected_face_mask=region_mask,
                        max_neighbor_step_mm=slope_limit_stats.get(
                            "max_neighbor_step_mm"
                        ),
                    )
                )
                rigid_alignment_stats[
                    "gradient_compression"
                ] = gradient_compression_stats
        else:
            z = np.where(top_silhouette_mask, z, np.nan)
            processed_scene, slope_limit_stats = _limit_positive_relief_slope(
                z,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                structural_region_mask=head_region_mask,
            )
            rigid_face_surface, rigid_alignment_stats = _align_stabilized_head_to_reference_boundary(
                z,
                face_reference_surface,
                processed_scene,
                head_region_mask,
                protected_core_mask=face_translation_core_mask,
                protected_face_mask=region_mask,
                max_neighbor_step_mm=slope_limit_stats.get("max_neighbor_step_mm"),
            )
            rigid_alignment_stats["gradient_compression"] = gradient_compression_stats
        projection_corrections = [
            float(record.get("correction_max_mm", 0.0))
            for record in rigid_alignment_stats.get("slope_projection", [])
            if record.get("enabled", False)
        ]
        rigid_shape_rmse = float(
            rigid_alignment_stats.get("translation_aligned_shape_distortion_rmse_mm", 0.0)
            or 0.0
        )
        gradient_reconstruction_selected = (
            rigid_alignment_stats.get("method") == "screened_gradient_domain_compression"
        )
        use_detail_restoration = not gradient_reconstruction_selected and (
            not rigid_alignment_stats.get("enabled", False)
            or rigid_shape_rmse > 3.0
            or max(projection_corrections, default=0.0) > 1.5
        )
        if use_detail_restoration:
            global_processed_scene, global_slope_stats = _limit_positive_relief_slope(
                unstabilized_scene,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                structural_region_mask=head_region_mask,
            )
            detail_surface, detail_restoration_stats = _restore_face_laplacian_detail(
                global_processed_scene,
                face_reference_surface,
                region_mask,
                max_neighbor_step_mm=global_slope_stats.get("max_neighbor_step_mm"),
            )
            if detail_restoration_stats.get("enabled", False):
                z = detail_surface
                slope_limit_stats = global_slope_stats
                selection_reasons = []
                if not rigid_alignment_stats.get("enabled", False):
                    selection_reasons.append("rigid_alignment_rejected")
                if rigid_shape_rmse > 3.0:
                    selection_reasons.append("rigid_outer_band_distortion")
                if max(projection_corrections, default=0.0) > 1.5:
                    selection_reasons.append("rigid_projection_correction")
                face_boundary_alignment_stats = {
                    "enabled": True,
                    "method": "frequency_separated_face_detail",
                    "selection_reason": selection_reasons,
                    "attachment_slope_ratio_p99": detail_restoration_stats.get(
                        "attachment_slope_ratio_p99"
                    ),
                    "attachment_slope_ratio_max": detail_restoration_stats.get(
                        "attachment_slope_ratio_max"
                    ),
                    "protected_core_region": face_translation_core_stats,
                    "detail_restoration": detail_restoration_stats,
                    "rigid_candidate": rigid_alignment_stats,
                }
            else:
                z = rigid_face_surface if rigid_alignment_stats.get("enabled", False) else global_processed_scene
                face_boundary_alignment_stats = rigid_alignment_stats
                face_boundary_alignment_stats["detail_restoration"] = detail_restoration_stats
        else:
            z = rigid_face_surface
            face_boundary_alignment_stats = rigid_alignment_stats
        face_boundary_alignment_stats["protected_core_region"] = face_translation_core_stats
        face_surface_protection_stats = {
            "enabled": False,
            "reason": (
                "superseded_by_gradient_domain_compression"
                if face_boundary_alignment_stats.get("method")
                == "screened_gradient_domain_compression"
                else "superseded_by_reference_boundary_transfer"
            ),
        }
        accepted_face_surface = z.copy()
        effective_printable_feature_depth_mm = float(printable_feature_depth_mm)
        feature_emboss_suppressed = False
        z, printable_feature_stats = _enhance_weighted_relief_features(
            z,
            feature_weight_mask,
            max_feature_depth_mm=effective_printable_feature_depth_mm,
            feature_exclusion_mask=feature_exclusion_mask,
        )
        printable_feature_stats.update(
            {
                "requested_max_feature_depth_mm": float(printable_feature_depth_mm),
                "effective_max_feature_depth_mm": effective_printable_feature_depth_mm,
                "suppressed_after_screened_face_reconstruction": (
                    feature_emboss_suppressed
                ),
                "screened_face_reconstruction": bool(
                    gradient_reconstruction_selected
                ),
            }
        )
        z, feature_bridge_stats = _bridge_weighted_face_features(
            z,
            region_mask,
            feature_weight_mask,
            max_bridge_depth_mm=feature_bridge_depth_mm,
            feature_exclusion_mask=feature_exclusion_mask,
        )
        z, post_feature_slope_guard_stats = _guard_weighted_feature_updates(
            accepted_face_surface,
            z,
            slope_limit_stats.get("max_neighbor_step_mm"),
        )
        z, face_detail_guard_stats = _guard_face_detail_updates(
            accepted_face_surface,
            z,
            unstabilized_scene,
            region_mask,
            minimum_correlation=0.8,
            minimum_rms_retention=0.6,
            maximum_rms_retention=2.0,
        )
    elif (
        selected_region is not None
        and region_mask is None
        and not subject_surface_locked
    ):
        face_boundary_alignment_stats = {
            "enabled": False,
            "reason": "no_face_region",
        }
        face_boundary_attachment_stats = {
            "enabled": False,
            "reason": "no_face_region",
        }
        face_surface_protection_stats = {
            "enabled": False,
            "reason": "selection_gradient_domain",
        }
        printable_feature_stats = {
            "enabled": False,
            "reason": "no_face_region",
        }
        feature_bridge_stats = {
            "enabled": False,
            "reason": "no_face_region",
        }
        z, selection_gradient_compression_stats, selection_slope_stats = (
            _compress_selected_relief_surface(
                z,
                selected_region,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                sample_pitch_source=gradient_sample_pitch_source,
            )
        )
        if selection_slope_stats is not None:
            slope_limit_stats = selection_slope_stats
        face_detail_guard_stats = {
            "enabled": False,
            "reason": "no_face_region",
        }
    else:
        face_boundary_alignment_stats = {"enabled": False, "reason": "height_within_reference"}
        z, face_boundary_attachment_stats = _attach_face_boundary_to_local_surface(
            z,
            region_mask,
            max_separation_mm=feature_bridge_depth_mm,
            sample_pitch_mm=print_filter_stats["mesh_sample_pitch_mm"],
        )
        accepted_face_surface = z.copy()
        z, printable_feature_stats = _enhance_weighted_relief_features(
            z,
            feature_weight_mask,
            max_feature_depth_mm=printable_feature_depth_mm,
            feature_exclusion_mask=feature_exclusion_mask,
        )
        z, feature_bridge_stats = _bridge_weighted_face_features(
            z,
            region_mask,
            feature_weight_mask,
            max_bridge_depth_mm=feature_bridge_depth_mm,
            feature_exclusion_mask=feature_exclusion_mask,
        )
        try:
            configured_feature_step = (
                float(print_filter_stats["mesh_sample_pitch_mm"])
                * float(max_relief_slope)
            )
        except (TypeError, ValueError):
            configured_feature_step = None
        z, post_feature_slope_guard_stats = _guard_weighted_feature_updates(
            accepted_face_surface,
            z,
            configured_feature_step,
        )
        z = np.where(top_silhouette_mask, z, np.nan)
        z, slope_limit_stats = _limit_positive_relief_slope(
            z,
            sample_pitch_mm=print_filter_stats["mesh_sample_pitch_mm"],
            max_slope_mm_per_mm=max_relief_slope,
            structural_region_mask=region_mask,
        )
        face_surface_protection_stats = {"enabled": False, "reason": "height_within_reference"}
    if (
        selected_region is not None
        and region_mask is not None
        and not subject_surface_locked
    ):
        face_protected_selection_baseline = z.copy()
        protected_selection_region = (
            head_region_mask
            if head_region_mask is not None and np.any(head_region_mask)
            else region_mask
        )
        selection_candidate, selection_gradient_compression_stats, selection_slope_stats = (
            _compress_selected_relief_surface(
                z,
                selected_region,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                sample_pitch_source=gradient_sample_pitch_source,
                protected_region_mask=protected_selection_region,
            )
        )
        post_selection_face_detail = _face_detail_preservation_metrics(
            unstabilized_scene,
            selection_candidate,
            region_mask,
        )
        post_selection_components = post_selection_face_detail.get("components", [])
        face_protection_passed = bool(
            post_selection_face_detail.get("available", False)
            and post_selection_components
            and all(
                _detail_component_passes(record, 0.8, 0.6, 2.0)
                for record in post_selection_components
            )
        )
        selection_gradient_compression_stats["face_detail_after_selection"] = (
            post_selection_face_detail
        )
        selection_gradient_compression_stats["face_protection_passed"] = face_protection_passed
        if face_protection_passed:
            z = selection_candidate
            if selection_slope_stats is not None:
                slope_limit_stats = selection_slope_stats
        else:
            z = face_protected_selection_baseline
            selection_gradient_compression_stats["enabled"] = False
            selection_gradient_compression_stats["reason"] = "face_protection_gate"
    elif subject_surface_locked:
        selection_gradient_compression_stats = {
            "enabled": False,
            "reason": "subject_surface_locked",
            "selected_pixels": int(np.count_nonzero(selected_region)),
            "face_protection_passed": True,
        }
    z = np.where(top_silhouette_mask, z, np.nan)
    if base_border_px:
        z = _flatten_border(
            np.maximum(z - 0.01, 0.0),
            base_border_px,
            preserve_mask=preserve_detail_border,
        ) + 0.01
        z = np.where(top_silhouette_mask, z, np.nan)
    background_reference_surface = unstabilized_scene
    selection_background_cap_stats = {
        "enabled": False,
        "reason": "no_selection_mask",
    }
    if selected_region is not None:
        # Compare background retention with identical final foreground geometry so
        # required printable support ramps are not mistaken for context loss.
        background_reference_surface = np.where(
            selected_region,
            z,
            background_reference_surface,
        )
        z, initial_selection_background_cap_stats = _cap_selection_background_relief(
            z,
            selected_region,
            relief_height_mm=z_scale,
            sample_pitch_mm=gradient_sample_pitch_mm,
            max_slope_mm_per_mm=max_relief_slope,
            background_depth_ratio=selection_background_depth_ratio,
        )
        attachment_conflict_relaxation_stats = {
            "enabled": False,
            "reason": "background_context_disabled",
        }
        if float(selection_background_depth_ratio) > 0:
            z, attachment_conflict_relaxation_stats = (
                _relax_selection_attachment_conflicts(
                    z,
                    selected_region,
                    sample_pitch_mm=gradient_sample_pitch_mm,
                    max_slope_mm_per_mm=max_relief_slope,
                    feather_pixels=1,
                )
            )
        z, selection_background_cap_stats = _cap_selection_background_relief(
            z,
            selected_region,
            relief_height_mm=z_scale,
            sample_pitch_mm=gradient_sample_pitch_mm,
            max_slope_mm_per_mm=max_relief_slope,
            background_depth_ratio=selection_background_depth_ratio,
        )
        selection_background_cap_stats["pre_relaxation"] = (
            initial_selection_background_cap_stats
        )
        selection_background_cap_stats["attachment_conflict_relaxation"] = (
            attachment_conflict_relaxation_stats
        )
        background_reference_surface = np.where(
            selected_region,
            z,
            unstabilized_scene,
        )
        background_reference_surface, reference_background_cap_stats = (
            _cap_selection_background_relief(
                background_reference_surface,
                selected_region,
                relief_height_mm=z_scale,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                background_depth_ratio=selection_background_depth_ratio,
            )
        )
        reference_background_cap_stats["foreground_geometry_source"] = (
            "final_conflict_relaxed_selection"
        )
        selection_background_cap_stats["reference_cap"] = reference_background_cap_stats
        if not reference_background_cap_stats.get("emission_passed", False):
            raise ValueError("Bounded background reference failed its physical emission gates")
        if not selection_background_cap_stats.get("emission_passed", False):
            raise ValueError("Background relief failed its physical emission gates")
    background_foreground_mask = selected_region
    background_context_enforced = bool(
        selected_region is not None
        and float(selection_background_depth_ratio) > 0
    )
    if background_context_enforced:
        background_preservation_stats = _background_relief_preservation_metrics(
            background_reference_surface,
            z,
            background_foreground_mask,
            sample_pitch_mm=gradient_sample_pitch_mm,
        )
        background_preservation_stats["enforced"] = True
        if not background_preservation_stats.get("available", False):
            raise ValueError(
                "Background relief preservation telemetry is unavailable"
            )
    else:
        background_preservation_stats = {
            "available": False,
            "passed": True,
            "enforced": False,
            "reason": "background_context_disabled",
            "quality_failures": [],
        }
    if (
        background_preservation_stats.get("available", False)
        and not background_preservation_stats.get("passed", False)
    ):
        initial_background_stats = background_preservation_stats
        restored_background, fallback_stats = _restore_background_from_reference(
            z,
            background_reference_surface,
            background_foreground_mask,
            sample_pitch_mm=gradient_sample_pitch_mm,
            max_neighbor_step_mm=gradient_sample_pitch_mm * float(max_relief_slope),
            reference_is_accepted=True,
        )
        if selected_region is not None:
            restored_background, fallback_cap_stats = _cap_selection_background_relief(
                restored_background,
                selected_region,
                relief_height_mm=z_scale,
                sample_pitch_mm=gradient_sample_pitch_mm,
                max_slope_mm_per_mm=max_relief_slope,
                background_depth_ratio=selection_background_depth_ratio,
            )
            fallback_stats["physical_cap"] = fallback_cap_stats
        fallback_metrics = _background_relief_preservation_metrics(
            background_reference_surface,
            restored_background,
            background_foreground_mask,
            sample_pitch_mm=gradient_sample_pitch_mm,
        )
        fallback_cap_accepted = bool(
            fallback_stats.get("physical_cap", {}).get("emission_passed", False)
        )
        fallback_slope_guard = fallback_stats.get("slope_guard", {})
        fallback_slope_audit = fallback_slope_guard.get("final_audit", {})
        fallback_slope_accepted = bool(
            not fallback_slope_guard.get("enabled", False)
            or float(fallback_slope_audit.get("accepted_surface_ratio_max", float("inf")))
            <= float(fallback_slope_guard.get("max_ratio", 1.0)) + 1e-6
        )
        fallback_stats["physical_cap_accepted"] = fallback_cap_accepted
        fallback_stats["slope_guard_accepted"] = fallback_slope_accepted
        fallback_stats["accepted"] = bool(
            fallback_metrics.get("passed", False)
            and fallback_cap_accepted
            and fallback_slope_accepted
        )
        fallback_metrics["initial_candidate"] = initial_background_stats
        fallback_metrics["fallback"] = fallback_stats
        if not fallback_stats["accepted"]:
            raise ValueError(
                "Background relief preservation failed after bounded reference fallback"
            )
        z = restored_background
        background_preservation_stats = fallback_metrics

    face_appearance_stats = _surface_lighting_agreement_metrics(
        unstabilized_scene,
        z,
        region_mask,
        sample_pitch_mm=gradient_sample_pitch_mm,
        boundary_exclusion_mm=0.8,
        component_metrics=True,
    )
    if selected_region is not None:
        selection_appearance_region = _resize_binary_mask(selected_region, z.shape)
        if region_mask is not None:
            selection_appearance_region &= ~_resize_binary_mask(region_mask, z.shape)
        selection_appearance_stats = _surface_lighting_agreement_metrics(
            unstabilized_scene,
            z,
            selection_appearance_region,
            sample_pitch_mm=gradient_sample_pitch_mm,
            boundary_exclusion_mm=0.8,
            component_metrics=True,
        )
    else:
        selection_appearance_stats = {
            "available": False,
            "reason": "no_selection_region",
            "samples": 0,
        }
    if background_context_enforced:
        background_appearance_region = (
            np.isfinite(background_reference_surface)
            & ~_resize_binary_mask(selected_region, background_reference_surface.shape)
        )
        background_appearance_stats = _surface_lighting_agreement_metrics(
            background_reference_surface,
            z,
            background_appearance_region,
            sample_pitch_mm=gradient_sample_pitch_mm,
            boundary_exclusion_mm=1.5,
            component_metrics=True,
        )
    else:
        background_appearance_stats = {
            "available": False,
            "reason": "background_context_disabled",
            "samples": 0,
        }
    surface_appearance_stats = {
        "method": "physical_heightfield_normals_lambertian_v1",
        "face": face_appearance_stats,
        "selection_nonface": selection_appearance_stats,
        "background": background_appearance_stats,
    }

    # Create a mask for non-NaN values
    mask = np.isfinite(z)
    if not np.any(mask):
        raise ValueError("Depth data has no finite values to convert into a mesh")

    # Find the bounding box of non-NaN values
    rows, cols = np.where(mask)
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1

    # Crop the data to the bounding box
    z = z[top:bottom, left:right]
    mask = mask[top:bottom, left:right]
    reference_surface = unstabilized_scene[top:bottom, left:right]
    surface_grid_transform = {
        "schema_version": 1,
        "input_depth_shape": input_depth_shape,
        "target_depth_shape": target_depth_shape,
        "flip_x": True,
        "mesh_shape_before_crop": mesh_shape_before_crop,
        "crop_bbox_rc": [int(top), int(left), int(bottom), int(right)],
        "emitted_shape": [int(z.shape[0]), int(z.shape[1])],
        "mask_interpolation": "nearest",
    }
    if surface_output_path is not None:
        surface_output_path = os.fspath(surface_output_path)
        os.makedirs(os.path.dirname(surface_output_path) or ".", exist_ok=True)
        np.save(surface_output_path, z.astype(np.float32, copy=False))
    if reference_surface_output_path is not None:
        reference_surface_output_path = os.fspath(reference_surface_output_path)
        os.makedirs(
            os.path.dirname(reference_surface_output_path) or ".",
            exist_ok=True,
        )
        np.save(
            reference_surface_output_path,
            reference_surface.astype(np.float32, copy=False),
        )

    # Adjust the X, Y grid to match the cropped data. target_dimension controls
    # sampling/detail; max_xy_size controls the final physical STL footprint.
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    x, y = np.meshgrid(x, y)
    if max_xy_size is not None:
        coordinate_max = max(z.shape[1] - 1, z.shape[0] - 1)
        if coordinate_max > 0:
            xy_scale = float(max_xy_size) / float(coordinate_max)
            x = x * xy_scale
            y = y * xy_scale

    valid_cells = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    faces = []

    # Generate a watertight height-field mesh. Internal cell borders are shared
    # by adjacent top faces; side walls are emitted only around exposed edges.
    for i in range(z.shape[0] - 1):
        for j in range(z.shape[1] - 1):
            if valid_cells[i, j]:
                # Top surface vertices
                v0 = [x[i, j], y[i, j], z[i, j]]
                v1 = [x[i+1, j], y[i+1, j], z[i+1, j]]
                v2 = [x[i, j+1], y[i, j+1], z[i, j+1]]
                v3 = [x[i+1, j+1], y[i+1, j+1], z[i+1, j+1]]

                # Bottom surface vertices
                v0_bottom = [x[i, j], y[i, j], 0]
                v1_bottom = [x[i+1, j], y[i+1, j], 0]
                v2_bottom = [x[i, j+1], y[i, j+1], 0]
                v3_bottom = [x[i+1, j+1], y[i+1, j+1], 0]

                # Top and bottom are wound outward so trimesh recognizes the
                # exported relief as a valid volume, not merely watertight.
                _add_triangle(faces, v0, v2, v1)
                _add_triangle(faces, v1, v2, v3)

                # Bottom surface
                _add_triangle(faces, v2_bottom, v0_bottom, v1_bottom)
                _add_triangle(faces, v2_bottom, v1_bottom, v3_bottom)

                # Front boundary
                if i == 0 or not valid_cells[i - 1, j]:
                    _add_triangle(faces, v0, v0_bottom, v2)
                    _add_triangle(faces, v2, v0_bottom, v2_bottom)

                # Back boundary
                if i == valid_cells.shape[0] - 1 or not valid_cells[i + 1, j]:
                    _add_triangle(faces, v1, v3, v1_bottom)
                    _add_triangle(faces, v3, v3_bottom, v1_bottom)

                # Left boundary
                if j == 0 or not valid_cells[i, j - 1]:
                    _add_triangle(faces, v0, v1, v0_bottom)
                    _add_triangle(faces, v1, v1_bottom, v0_bottom)

                # Right boundary
                if j == valid_cells.shape[1] - 1 or not valid_cells[i, j + 1]:
                    _add_triangle(faces, v2, v2_bottom, v3)
                    _add_triangle(faces, v3, v2_bottom, v3_bottom)

    if not faces:
        raise ValueError("Depth data produced no valid mesh faces")

    print(f"Number of faces: {len(faces)}")

    stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    stl_mesh.vectors[:] = np.asarray(faces, dtype=np.float32)
    _force_positive_stl_volume(stl_mesh)

    # Save the mesh to an STL file
    os.makedirs(os.path.dirname(output_stl_path) or ".", exist_ok=True)
    stl_mesh.save(output_stl_path)
    print(f"STL file saved as {output_stl_path}")
    return {
        **print_filter_stats,
        "max_relief_slope": slope_limit_stats,
        "detail_edge_threshold": float(detail_edge_threshold),
        "detail_edge_window_px": _detail_edge_window_size(detail_radius),
        "background_detail_boost": float(background_detail_boost),
        "background_detail_face_protected": region_mask is not None,
        "background_photo_detail_mm": float(background_photo_detail_mm),
        "background_photo_detail": photo_detail_stats,
        "selection_background_depth_ratio": float(selection_background_depth_ratio),
        "normalization_reference_depth": {
            "enabled": normalization_reference is not None,
            "method": (
                "external-pre-refinement-depth-subject-interior-blend"
                if normalization_reference is not None
                and selected_region is not None
                and float(selection_background_depth_ratio) > 0
                and not subject_surface_locked
                else "external-pre-refinement-depth"
                if normalization_reference is not None
                else "input-depth"
            ),
            "subject_boundary_px": (
                NORMALIZATION_REFERENCE_BOUNDARY_PX
                if normalization_reference is not None
                and selected_region is not None
                and float(selection_background_depth_ratio) > 0
                and not subject_surface_locked
                else None
            ),
            "subject_taper_px": (
                NORMALIZATION_REFERENCE_TAPER_PX
                if normalization_reference is not None
                and selected_region is not None
                and float(selection_background_depth_ratio) > 0
                and not subject_surface_locked
                else None
            ),
        },
        "selection_subject_lock": bool(subject_surface_locked),
        "selection_background_physical_cap": selection_background_cap_stats,
        "background_depth_preservation": background_preservation_stats,
        "surface_appearance_agreement": surface_appearance_stats,
        "top_silhouette": top_silhouette_stats,
        "printable_feature_depth_mm": float(printable_feature_depth_mm),
        "effective_printable_feature_depth_mm": float(
            printable_feature_stats.get(
                "effective_max_feature_depth_mm",
                printable_feature_depth_mm,
            )
        ),
        "printable_feature_depth": printable_feature_stats,
        "feature_bridge_depth_mm": float(feature_bridge_depth_mm),
        "feature_bridge": feature_bridge_stats,
        "feature_exclusion_masked": feature_exclusion_mask is not None,
        "post_feature_slope_guard": post_feature_slope_guard_stats,
        "face_detail_guard": face_detail_guard_stats,
        "selection_gradient_compression": selection_gradient_compression_stats,
        "face_boundary_attachment": face_boundary_attachment_stats,
        "face_height_stabilization": face_height_stabilization_stats,
        "head_stabilization_region": head_region_stats,
        "face_boundary_alignment": face_boundary_alignment_stats,
        "face_surface_protection": face_surface_protection_stats,
        "surface_grid_transform": surface_grid_transform,
        "reference_surface": {
            "kind": "pre_high_relief_post_shape_surface",
            "emitted": reference_surface_output_path is not None,
            "shape": [int(reference_surface.shape[0]), int(reference_surface.shape[1])],
        },
        "mesh_grid_shape": [int(z.shape[0]), int(z.shape[1])],
        "face_count": int(len(faces)),
    }

# Create interactive 3D plot
def create_interactive_3d_plot(npy_file, output_html='interactive_3d_plot.html'):
    import plotly.graph_objs as go

    # Load the .npy file
    data = np.load(npy_file)

    # Create x and y values corresponding to the data dimensions
    x = np.linspace(0, data.shape[1] - 1, data.shape[1])
    y = np.linspace(0, data.shape[0] - 1, data.shape[0])
    x, y = np.meshgrid(x, y)
    z = data  # Use the loaded data as z-values

    # Create the 3D surface plot using Plotly
    surface = go.Surface(z=z, x=x, y=y, colorscale='Viridis')

    # Create the layout for the plot
    layout = go.Layout(
        title='Interactive 3D Surface Plot',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
    )

    # Combine the plot and layout into a figure
    fig = go.Figure(data=[surface], layout=layout)

    # Save the figure as an HTML file
    fig.write_html(output_html)
    print(f"Interactive 3D plot saved as {output_html}")

# Main script to call the functions
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process an image, generate depth data, and optionally create a 3D model and interactive plot.')
    parser.add_argument('--input', type=str, default="./input_image.jpg", help='Path to the input image file')
    parser.add_argument('--interactive-plot', action='store_true', help='Generate interactive 3D plot (default: False)')
    parser.add_argument('--output-dir', type=str, default="./output", help='Directory to save the output files')
    parser.add_argument('--target-dimension', type=int, default=300, help='Target dimension for the 3D model downsampling')
    parser.add_argument('--skip-gendepth', action='store_true', help='Skip depth data generation and use existing depth data')
    parser.add_argument('--z-scale', type=float, default=50, help='Z scale for adjusting the height in the 3D model')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the image and get depth data (unless skip-gendepth is True)
    if not args.skip_gendepth:
        npy_file = process_image_get_depth_data(args.input, args.output_dir)
    else:
        # If skipping depth generation, ensure the depth data exists
        npy_file = os.path.join(args.output_dir, "output_depth_data.npy")
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Depth data not found at {npy_file}. Please generate depth data first or provide the correct path.")

    # Conditionally create an interactive 3D plot
    if args.interactive_plot:
        create_interactive_3d_plot(npy_file, output_html=os.path.join(args.output_dir, 'interactive_3d_plot.html'))

    # Convert depth data to 3D model, passing the target dimension and z-scale
    depth_data_to_3d_model(npy_file, output_stl_path=os.path.join(args.output_dir, 'output_3d_model.stl'), target_dimension=args.target_dimension, z_scale=args.z_scale)

if __name__ == "__main__":
    main()
