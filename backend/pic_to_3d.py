import os
import shutil
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
except ImportError:  # pragma: no cover - supports running from backend/
    if __package__:
        raise
    from face_relief_geometry import align_face_to_scene_gradient_domain

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
METRIC_FAR_HIGH_DEPTH_MODELS = frozenset(
    {
        DEPTHPRO_MODEL_ID,
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    }
)


def relief_value_transform_for_model(model_name):
    if model_name in METRIC_FAR_HIGH_DEPTH_MODELS:
        return RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH
    return RELIEF_VALUE_TRANSFORM_LINEAR

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
    if provider in ("flux-fill", "qwen-image-inpaint", "qwen-image-edit"):
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
        from PIL import Image
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
    image = Image.open(input_image_path).convert("RGB")
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
        from PIL import Image
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
        image = Image.open(input_image_path).convert("RGB")
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


def _normalize_relief_values(values, low_percentile=1.0, high_percentile=99.0):
    normalized = values.astype(np.float32, copy=True)
    finite = normalized[np.isfinite(normalized)]
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
):
    transformed, reverses_order = _transform_relief_values(values, value_transform)
    relief = _normalize_relief_values(
        transformed,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
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


def _inject_photo_relief_detail(
    relief,
    source_image,
    max_detail_ratio=0.0,
    protection_mask=None,
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
        protected = np.asarray(protection_mask) > 0
        if protected.shape != relief.shape:
            protected = _resize_binary_mask(protected, relief.shape)
        feather = gaussian_filter(maximum_filter(protected.astype(np.float32), size=9), sigma=3.0)
        feather[protected] = 1.0
        background_gate *= 1.0 - np.clip(feather, 0.0, 1.0)

    usable = finite & (background_gate > 0.5)
    if not np.any(usable):
        return relief, {"enabled": False, "reason": "no_background_pixels"}
    detail_scale = float(np.percentile(np.abs(photo_detail[usable]), 98.0))
    if not np.isfinite(detail_scale) or detail_scale <= 1e-6:
        return relief, {"enabled": False, "reason": "no_photo_detail"}

    normalized_detail = np.clip(photo_detail / detail_scale, -1.0, 1.0)
    fused = relief + float(max_detail_ratio) * normalized_detail * background_gate
    return np.clip(fused, 0.0, 1.0), {
        "enabled": True,
        "max_detail_ratio": float(max_detail_ratio),
        "detail_scale": detail_scale,
        "background_coverage_ratio": float(np.mean(usable)),
        "face_protected": protected is not None,
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
            and float(audit["accepted_surface_ratio_max"]) <= ratio_limit
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
        projected_delta = np.where(movable, output - baseline64, 0.0)
        low_scale = 0.0
        high_scale = 1.0
        for _ in range(max(1, int(iterations))):
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
        "max_ratio": ratio_limit,
        "candidate_audit": candidate_audit,
        "final_audit": final_audit,
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
    applied = np.abs(np.asarray(guarded, dtype=np.float64) - processed)
    weighted = total_weight > 1e-4
    boundary = valid & ~binary_erosion(
        valid,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    final_audit = guard_stats.get("final_audit", {})
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


def _flatten_border(values, border_px):
    border_px = int(border_px or 0)
    if border_px <= 0:
        return values
    border_px = min(border_px, values.shape[0] // 2, values.shape[1] // 2)
    if border_px <= 0:
        return values
    values = values.copy()
    values[:border_px, :] = 0.0
    values[-border_px:, :] = 0.0
    values[:, :border_px] = 0.0
    values[:, -border_px:] = 0.0
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
    minimum_detail_rms_retention=0.25,
    maximum_detail_rms_retention=2.0,
    maximum_output_edge_p99_ratio=12.0,
    maximum_output_edge_ratio=24.0,
    minimum_height_span_ratio=0.5,
    maximum_height_span_ratio=1.15,
    maximum_correction_span_ratio=0.9,
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

    def add_edges(first_values, second_values, first_ids, second_ids, active, barriers):
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
    )
    vertical_active = valid[:-1, :] & valid[1:, :]
    add_edges(
        source[:-1, :],
        source[1:, :],
        sample_ids[:-1, :],
        sample_ids[1:, :],
        vertical_active,
        vertical_barrier,
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

    detail_stats = {
        "available": False,
        "correlation": None,
        "rms_retention": None,
        "samples": 0,
    }
    if detail_region_mask is not None:
        detail_region = _resize_binary_mask(detail_region_mask, source.shape) & valid
        detail_region = binary_erosion(
            detail_region,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        detail_region &= binary_erosion(
            valid,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        if np.count_nonzero(detail_region) >= 16:
            source_curvature = laplace(np.where(valid, source, 0.0))[detail_region].astype(
                np.float64
            )
            output_curvature = laplace(np.where(valid, compressed, 0.0))[detail_region].astype(
                np.float64
            )
            source_curvature -= np.mean(source_curvature)
            output_curvature -= np.mean(output_curvature)
            source_rms = float(np.sqrt(np.mean(np.square(source_curvature))))
            output_rms = float(np.sqrt(np.mean(np.square(output_curvature))))
            if source_rms <= 1e-10:
                detail_correlation = 1.0 if output_rms <= 1e-8 else 0.0
                detail_retention = (
                    1.0 if output_rms <= 1e-8 else max_detail_retention + 1.0
                )
            else:
                detail_correlation = float(
                    np.dot(source_curvature, output_curvature)
                    / max(
                        np.linalg.norm(source_curvature) * np.linalg.norm(output_curvature),
                        1e-12,
                    )
                )
                detail_retention = output_rms / source_rms
            detail_stats = {
                "available": True,
                "correlation": detail_correlation,
                "rms_retention": detail_retention,
                "samples": int(np.count_nonzero(detail_region)),
            }

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
    elif detail_stats["available"]:
        if detail_stats["correlation"] < min_detail_correlation:
            quality_failures.append("detail_correlation")
        if not min_detail_retention <= detail_stats["rms_retention"] <= max_detail_retention:
            quality_failures.append("detail_rms_retention")
    stats.update(
        {
            "enabled": not quality_failures,
            "max_slope_mm_per_mm": max_slope,
            "max_neighbor_step_mm": max_step,
            "gradient_soft_threshold_mm": soft_threshold,
            "gradient_threshold_ratio": threshold_ratio,
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
    if quality_failures:
        stats["reason"] = "quality_gate"
        return values, stats
    return compressed, stats


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
    background_detail_boost=1.0,
    source_image=None,
    background_photo_detail_mm=0.0,
    trim_top_background=False,
    feature_weight_mask=None,
    printable_feature_depth_mm=0.0,
    feature_bridge_depth_mm=0.0,
    feature_exclusion_mask=None,
):
    # Load the .npy file
    data = np.load(npy_file).astype(np.float32)
    region_mask = None
    if face_region_mask is not None:
        if isinstance(face_region_mask, (str, os.PathLike)):
            from PIL import Image

            region_mask = np.asarray(Image.open(face_region_mask).convert("L")) > 0
        else:
            region_mask = np.asarray(face_region_mask) > 0
        region_mask = _resize_binary_mask(region_mask, data.shape)

    # Skip downsampling if target_dimension is -1
    if target_dimension != -1:
        target_shape = _target_shape_for_max_dimension(data.shape, target_dimension)
        if target_shape != data.shape:
            print(f"Resizing depth grid from {data.shape} to {target_shape}")
            data = _resize_nan_aware(data, target_shape)
            if region_mask is not None:
                region_mask = _resize_binary_mask(region_mask, target_shape)
        else:
            print(f"Keeping depth grid resolution: {data.shape}")
    else:
        print("Skipping downsampling as target_dimension is -1")

    # Flip the x axis
    data = np.flip(data, axis=1)
    if region_mask is not None:
        region_mask = np.flip(region_mask, axis=1)
    if trim_top_background:
        top_silhouette_mask, top_silhouette_stats = _top_silhouette_mask(source_image, data.shape)
    else:
        top_silhouette_mask = np.ones(data.shape, dtype=bool)
        top_silhouette_stats = {"enabled": False, "reason": "disabled"}

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
        detail_protection_mask=region_mask,
        background_detail_boost=background_detail_boost,
    )
    photo_detail_ratio = (
        float(background_photo_detail_mm) / float(z_scale)
        if z_scale and background_photo_detail_mm > 0
        else 0.0
    )
    relief, photo_detail_stats = _inject_photo_relief_detail(
        relief,
        source_image,
        max_detail_ratio=photo_detail_ratio,
        protection_mask=region_mask,
    )
    relief = np.where(top_silhouette_mask, relief, np.nan)
    relief = _flatten_border(relief, base_border_px)
    relief = np.where(top_silhouette_mask, relief, np.nan)
    relief, print_filter_stats = _prepare_relief_for_printing(
        relief,
        max_xy_size=max_xy_size,
        minimum_feature_mm=minimum_feature_mm,
    )
    gradient_sample_pitch_mm = print_filter_stats["mesh_sample_pitch_mm"]
    gradient_sample_pitch_source = "physical_size"
    if gradient_sample_pitch_mm is None and max_xy_size is None:
        # Without an explicit physical footprint, STL X/Y coordinates are the
        # integer grid coordinates below, so adjacent samples are one unit apart.
        gradient_sample_pitch_mm = 1.0
        gradient_sample_pitch_source = "implicit_stl_grid_unit"
    if region_mask is not None:
        region_mask = _resize_binary_mask(region_mask, relief.shape)
    top_silhouette_mask = _resize_binary_mask(top_silhouette_mask, relief.shape)
    relief = np.where(top_silhouette_mask, relief, np.nan)
    relief = _flatten_border(relief, base_border_px)
    relief = np.where(top_silhouette_mask, relief, np.nan)
    z = relief * z_scale
    
    # Add a small offset to create buffer
    z = z + 0.01

    # Apply a Gaussian filter to smooth the data
    z = _smooth_nan_aware(z, sigma=sigma)
    z = np.where(top_silhouette_mask, z, np.nan)
    if base_border_px:
        z = _flatten_border(np.maximum(z - 0.01, 0.0), base_border_px) + 0.01
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
    unstabilized_scene = z.copy()
    z, face_height_stabilization_stats = _stabilize_face_relief_height(
        z,
        head_region_mask,
        relief_height_mm=z_scale,
        reference_face_height_mm=reference_face_height_mm,
    )

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
        )
        face_height_stabilization_stats[
            "gradient_compression_attempt"
        ] = gradient_compression_stats
        if gradient_compression_stats.get("enabled", False):
            gradient_compression_stats["sample_pitch_source"] = gradient_sample_pitch_source
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
            }
            rigid_face_surface = gradient_surface
            rigid_alignment_stats = {
                "enabled": True,
                "method": "screened_gradient_domain_compression",
                "selection_reason": ["face_aware_high_relief"],
                "attachment_slope_ratio_p99": gradient_compression_stats.get(
                    "output_edge_ratio_p99"
                ),
                "attachment_slope_ratio_max": gradient_compression_stats.get(
                    "output_edge_ratio_max"
                ),
                "slope_projection": [],
                "gradient_compression": gradient_compression_stats,
            }
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
        z, post_feature_slope_guard_stats = _guard_weighted_feature_updates(
            accepted_face_surface,
            z,
            slope_limit_stats.get("max_neighbor_step_mm"),
        )
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
    z = np.where(top_silhouette_mask, z, np.nan)
    if base_border_px:
        z = _flatten_border(np.maximum(z - 0.01, 0.0), base_border_px) + 0.01
        z = np.where(top_silhouette_mask, z, np.nan)

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
        "top_silhouette": top_silhouette_stats,
        "printable_feature_depth_mm": float(printable_feature_depth_mm),
        "printable_feature_depth": printable_feature_stats,
        "feature_bridge_depth_mm": float(feature_bridge_depth_mm),
        "feature_bridge": feature_bridge_stats,
        "feature_exclusion_masked": feature_exclusion_mask is not None,
        "post_feature_slope_guard": post_feature_slope_guard_stats,
        "face_boundary_attachment": face_boundary_attachment_stats,
        "face_height_stabilization": face_height_stabilization_stats,
        "head_stabilization_region": head_region_stats,
        "face_boundary_alignment": face_boundary_alignment_stats,
        "face_surface_protection": face_surface_protection_stats,
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
