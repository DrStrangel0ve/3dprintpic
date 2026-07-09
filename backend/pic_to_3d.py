import os
import shutil
import numpy as np
from stl import mesh
from scipy.ndimage import gaussian_filter
import argparse

_DEPTH_PIPELINE_CACHE = {}
_INPAINT_PIPELINE_CACHE = {}

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
        edit_prompt = (
            f"{prompt} Treat the white blank region as the area to fill. Do not alter the visible half."
        )
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
    else:
        raise ValueError(f"Unsupported edit mask fill mode: {fill_mode}")
    return Image.composite(fill, image, mask)


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
):
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

    depth_data = np.squeeze(depth_data).astype(np.float32)
    finite_mask = np.isfinite(depth_data)
    if not np.any(finite_mask):
        raise ValueError("Depth model returned no finite depth values")

    finite_values = depth_data[finite_mask]
    depth_min = float(np.min(finite_values))
    depth_max = float(np.max(finite_values))
    if depth_max > depth_min:
        depth_data = (depth_data - depth_min) / (depth_max - depth_min)

    npy_path = os.path.join(output_dir, "output_depth_data.npy")
    preview_path = os.path.join(output_dir, "output_depth_preview.png")
    np.save(npy_path, depth_data)

    preview_data = np.nan_to_num(depth_data, nan=0.0, posinf=1.0, neginf=0.0)
    if np.max(preview_data) > np.min(preview_data):
        preview_data = (preview_data - np.min(preview_data)) / (np.max(preview_data) - np.min(preview_data))
    Image.fromarray((preview_data * 255).astype(np.uint8)).save(preview_path)

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


def depth_data_to_3d_model(npy_file, output_stl_path='output_3d_model.stl', target_dimension=300, z_scale=50, invert=True, sigma=4.0):
    # Load the .npy file
    data = np.load(npy_file).astype(np.float32)

    # Skip downsampling if target_dimension is -1
    if target_dimension != -1:
        # Adjust the downsample resolution dynamically based on the maximum dimension
        max_dimension = max(data.shape)
        downsample_res = max(1, -(-max_dimension // target_dimension))
        print(f"Downsampling resolution: {downsample_res}")

        # Downsample the data to reduce the resolution
        data = data[::downsample_res, ::downsample_res]
    else:
        print("Skipping downsampling as target_dimension is -1")

    # Flip the x axis
    data = np.flip(data, axis=1)

    # Adjust height scaling (z-scale) and option to invert the heights
    if invert:
        z_max = np.nanmax(data)
        z = (z_max - data) * z_scale
    else:
        z = data * z_scale
    
    # Add a small offset to create buffer
    z = z + 0.01

    # Apply a Gaussian filter to smooth the data
    z = _smooth_nan_aware(z, sigma=sigma)

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

    # Adjust the X, Y grid to match the cropped data
    x = np.arange(z.shape[1])
    y = np.arange(z.shape[0])
    x, y = np.meshgrid(x, y)

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

                # Top surface
                _add_triangle(faces, v0, v1, v2)
                _add_triangle(faces, v1, v3, v2)

                # Bottom surface
                _add_triangle(faces, v2_bottom, v1_bottom, v0_bottom)
                _add_triangle(faces, v2_bottom, v3_bottom, v1_bottom)

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
