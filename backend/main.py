import os
import json
from dotenv import load_dotenv
import aiohttp
import asyncio
import re
import math
import time
from collections import deque
from threading import Lock, Thread
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import logging
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse
try:
    from .pic_to_3d import (
        MODERN_INPAINT_MODELS,
        complete_image,
        depth_data_to_3d_model,
        process_image_get_depth_data,
        relief_value_transform_for_model,
    )
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from pic_to_3d import (
        MODERN_INPAINT_MODELS,
        complete_image,
        depth_data_to_3d_model,
        process_image_get_depth_data,
        relief_value_transform_for_model,
    )
try:
    from .stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
DEFAULT_LOCAL_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
CORS_ORIGINS = list(
    dict.fromkeys(
        origin.strip()
        for origin in [*DEFAULT_LOCAL_ORIGINS, *os.getenv("CORS_ORIGINS", "").split(",")]
        if origin.strip()
    )
)

# Updated CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get API key and team ID from environment variables
MASV_API_KEY = os.getenv("MASV_API_KEY")
MASV_TEAM_ID = os.getenv("MASV_TEAM_ID")
RBC_ACCESS_TOKEN = os.getenv("RBC_ACCESS_TOKEN")
RBC_API_BASE_URL = "https://paywithpretendpointsapi.onrender.com/api/v1"
DEFAULT_DEPTH_PROVIDER = os.getenv("DEPTH_PROVIDER", "transformers")
DEFAULT_DEPTH_MODEL = os.getenv("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
DEPTHPRO_MODEL_ID = "apple/DepthPro-hf"
DEPTHPRO_PRIMARY_WEIGHT_FILE = "model.safetensors"
DEPTHPRO_REQUIRED_FILES = ("model.safetensors", "config.json", "preprocessor_config.json")
DEPTHPRO_EXPECTED_BYTES = 1_904_996_876
DEPTH_PRELOAD_LOCK = Lock()
DEPTH_PRELOAD_STATE = {
    "model_id": DEPTHPRO_MODEL_ID,
    "status": "idle",
    "message": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "total_bytes": DEPTHPRO_EXPECTED_BYTES,
}
SELECTION_MODEL_LOCK = Lock()
SELECTION_MODEL_CACHE = {}
SELECTION_MODEL_IDS = {
    "sam2.1-hiera-large": "facebook/sam2.1-hiera-large",
    "sam2.1-hiera-base-plus": "facebook/sam2.1-hiera-base-plus",
    "grounding-dino-sam2": "facebook/sam2.1-hiera-large",
}

DEPTH_MODELS = [
    {
        "id": "apple/DepthPro-hf",
        "label": "Apple Depth Pro",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "metric_far_high",
        "notes": "Sharp distance edges with inverse-depth relief shaping; large first download.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        "label": "Depth Anything V2 Metric Indoor Large",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "metric_far_high",
        "notes": "Strong metric indoor model; good fallback for room/person/object photos.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        "label": "Depth Anything V2 Metric Outdoor Large",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "metric_far_high",
        "notes": "Metric outdoor model for street, landscape, and larger-scene photos.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Small-hf",
        "label": "Depth Anything V2 Small",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "relative_close_high",
        "notes": "Fast local default for iterative STL generation.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Base-hf",
        "label": "Depth Anything V2 Base",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "relative_close_high",
        "notes": "Better detail with a larger download and slower first run.",
    },
    {
        "id": "depth-anything/Depth-Anything-V2-Large-hf",
        "label": "Depth Anything V2 Large",
        "provider": "transformers",
        "recommended": True,
        "depth_value_semantics": "relative_close_high",
        "notes": "Best verified local quality option for CUDA relief generation.",
    },
]


def depth_model_far_is_high(model_id: str | None) -> bool:
    for model in DEPTH_MODELS:
        if model.get("id") == model_id:
            return model.get("depth_value_semantics") == "metric_far_high"
    return False


def relief_invert_for_model(model_id: str | None, relief_polarity: str, requested_invert: bool) -> bool:
    if relief_polarity not in ("raised-print", "mold"):
        return requested_invert
    far_is_high = depth_model_far_is_high(model_id)
    return far_is_high if relief_polarity == "raised-print" else not far_is_high


def _hf_model_cache_dir(model_id: str) -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"


def _hf_cached_file_path(model_id: str, filename: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, filename)
        if isinstance(cached, str):
            path = Path(cached)
            if path.exists() and not path.name.endswith(".incomplete"):
                return path
    except Exception:
        return None
    return None


def _depthpro_remote_total_bytes() -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(DEPTHPRO_MODEL_ID, files_metadata=True)
        total = sum(getattr(sibling, "size", None) or 0 for sibling in (info.siblings or []))
        return int(total or DEPTHPRO_EXPECTED_BYTES)
    except Exception:
        return DEPTHPRO_EXPECTED_BYTES


def _depthpro_cache_status() -> dict:
    cache_dir = _hf_model_cache_dir(DEPTHPRO_MODEL_ID)
    total_bytes = int(DEPTH_PRELOAD_STATE.get("total_bytes") or DEPTHPRO_EXPECTED_BYTES)
    complete_weight_path = _hf_cached_file_path(DEPTHPRO_MODEL_ID, DEPTHPRO_PRIMARY_WEIGHT_FILE)
    complete_files = {
        filename: _hf_cached_file_path(DEPTHPRO_MODEL_ID, filename) is not None
        for filename in DEPTHPRO_REQUIRED_FILES
    }
    missing_files = [filename for filename, present in complete_files.items() if not present]

    incomplete_files = list(cache_dir.glob("**/*.incomplete")) if cache_dir.exists() else []
    incomplete_weight_bytes = max((path.stat().st_size for path in incomplete_files), default=0)
    complete_weight_bytes = complete_weight_path.stat().st_size if complete_weight_path else 0
    downloaded_bytes = complete_weight_bytes or min(incomplete_weight_bytes, total_bytes)
    progress = 100.0 if complete_weight_path else min(99.0, (downloaded_bytes / total_bytes) * 100.0 if total_bytes else 0.0)

    with DEPTH_PRELOAD_LOCK:
        state = dict(DEPTH_PRELOAD_STATE)

    if complete_weight_path and not missing_files:
        derived_status = "ready"
        message = "Apple Depth Pro weights are cached."
    elif state.get("status") == "downloading":
        derived_status = "downloading"
        message = state.get("message") or "Downloading Apple Depth Pro weights."
    elif state.get("status") == "error":
        derived_status = "error"
        message = state.get("message") or "Apple Depth Pro preload failed."
    else:
        derived_status = "missing"
        message = "Apple Depth Pro weights are not fully cached."

    return {
        "model_id": DEPTHPRO_MODEL_ID,
        "status": derived_status,
        "message": message,
        "downloaded_bytes": int(downloaded_bytes),
        "total_bytes": int(total_bytes),
        "progress_percent": round(progress, 1),
        "complete": bool(complete_weight_path and not missing_files),
        "complete_files": complete_files,
        "missing_files": missing_files,
        "incomplete_file_count": len(incomplete_files),
        "cache_dir": str(cache_dir),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "error": state.get("error"),
    }


def _depthpro_preload_worker(force: bool = False) -> None:
    total_bytes = _depthpro_remote_total_bytes()
    with DEPTH_PRELOAD_LOCK:
        DEPTH_PRELOAD_STATE.update(
            {
                "status": "downloading",
                "message": "Downloading Apple Depth Pro weights.",
                "started_at": datetime.utcnow().isoformat() + "Z",
                "finished_at": None,
                "error": None,
                "total_bytes": total_bytes,
            }
        )

    try:
        from huggingface_hub import hf_hub_download

        for filename in DEPTHPRO_REQUIRED_FILES:
            hf_hub_download(
                DEPTHPRO_MODEL_ID,
                filename,
                resume_download=True,
                force_download=force,
            )
        status = _depthpro_cache_status()
        if not status["complete"]:
            raise RuntimeError("Depth Pro download finished but required files are still missing.")
        with DEPTH_PRELOAD_LOCK:
            DEPTH_PRELOAD_STATE.update(
                {
                    "status": "ready",
                    "message": "Apple Depth Pro weights are cached.",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "error": None,
                }
            )
    except Exception as exc:
        logger.exception("Apple Depth Pro preload failed")
        with DEPTH_PRELOAD_LOCK:
            DEPTH_PRELOAD_STATE.update(
                {
                    "status": "error",
                    "message": "Apple Depth Pro preload failed.",
                    "finished_at": datetime.utcnow().isoformat() + "Z",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_output_file(file_path: str, allowed_suffixes: tuple[str, ...]) -> Path:
    resolved_path = (OUTPUT_DIR / file_path).resolve()
    if not is_relative_to(resolved_path, OUTPUT_DIR):
        raise HTTPException(status_code=400, detail="Invalid output file path")
    if resolved_path.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Unsupported output file type")
    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return resolved_path


def output_relative_path(file_path: Path | str) -> str:
    path = Path(file_path).resolve()
    return path.relative_to(OUTPUT_DIR).as_posix()


def parse_selection_points(points_json: str) -> list[dict[str, float]]:
    try:
        raw_points = json.loads(points_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid selection points JSON: {exc}") from exc

    if not isinstance(raw_points, list):
        raise HTTPException(status_code=400, detail="Selection points must be a JSON list")

    points = []
    for index, point in enumerate(raw_points):
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[:2]
        else:
            raise HTTPException(status_code=400, detail=f"Selection point {index} must include x and y")
        try:
            points.append({"x": float(x), "y": float(y)})
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Selection point {index} has non-numeric coordinates") from exc
    return points


def selection_points_to_pixels(points: list[dict[str, float]], width: int, height: int) -> list[list[float]]:
    pixels = []
    for point in points:
        x = point["x"]
        y = point["y"]
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px = x * max(0, width - 1)
            py = y * max(0, height - 1)
        else:
            px = x
            py = y
        pixels.append([float(np.clip(px, 0, max(0, width - 1))), float(np.clip(py, 0, max(0, height - 1)))])
    return pixels


def connected_component_from_seed(region_mask: np.ndarray, seed_x: int, seed_y: int, max_pixels: int) -> np.ndarray:
    height, width = region_mask.shape
    component = np.zeros_like(region_mask, dtype=bool)
    if not (0 <= seed_x < width and 0 <= seed_y < height) or not region_mask[seed_y, seed_x]:
        return component

    visited = np.zeros_like(region_mask, dtype=bool)
    queue = deque([(seed_x, seed_y)])
    accepted = 0
    while queue and accepted < max_pixels:
        x, y = queue.popleft()
        if x < 0 or y < 0 or x >= width or y >= height or visited[y, x]:
            continue
        visited[y, x] = True
        if not region_mask[y, x]:
            continue
        component[y, x] = True
        accepted += 1
        queue.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    return component


def fallback_selection_mask(image: Image.Image, points: list[dict[str, float]], max_dimension: int = 1024) -> Image.Image:
    width, height = image.size
    max_dimension = max(128, min(2048, int(max_dimension or 1024)))
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1.0:
        work_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        work_image = image.resize(work_size, Image.Resampling.LANCZOS)
    else:
        work_size = image.size
        work_image = image

    work_width, work_height = work_size
    data = np.asarray(work_image.convert("RGB"), dtype=np.float32)
    mask = np.zeros((work_height, work_width), dtype=bool)
    pixels = selection_points_to_pixels(points, work_width, work_height)
    yy, xx = np.ogrid[:work_height, :work_width]
    global_std = float(np.std(data.reshape(-1, 3)))
    threshold = max(28.0, min(74.0, 22.0 + global_std * 0.42))
    click_radius = max(18, round(max(work_width, work_height) * 0.055))
    max_component_pixels = max(1024, int(work_width * work_height * 0.55))

    for px_float, py_float in pixels:
        px = int(round(px_float))
        py = int(round(py_float))
        seed = data[py, px]
        color_distance = np.linalg.norm(data - seed, axis=2)
        color_region = color_distance <= threshold
        component = connected_component_from_seed(color_region, px, py, max_component_pixels)
        click_circle = (xx - px) ** 2 + (yy - py) ** 2 <= click_radius**2
        component |= click_circle & (color_distance <= threshold * 1.8)
        if int(component.sum()) < max(64, int(work_width * work_height * 0.001)):
            component |= click_circle
        mask |= component

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(7))
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
    return mask_image


def load_sam2_selection_model(model_id: str, device: str):
    import torch
    from transformers import Sam2Model, Sam2Processor

    hf_model_id = SELECTION_MODEL_IDS.get(model_id, model_id if "/" in model_id else SELECTION_MODEL_IDS["sam2.1-hiera-large"])
    cache_key = (hf_model_id, device)
    with SELECTION_MODEL_LOCK:
        if cache_key in SELECTION_MODEL_CACHE:
            return SELECTION_MODEL_CACHE[cache_key]

        allow_download = os.getenv("SELECTION_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
        local_files_only = not allow_download
        processor = Sam2Processor.from_pretrained(hf_model_id, local_files_only=local_files_only)
        torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        model = Sam2Model.from_pretrained(hf_model_id, torch_dtype=torch_dtype, local_files_only=local_files_only)
        model.to(device)
        model.eval()
        SELECTION_MODEL_CACHE[cache_key] = (processor, model, hf_model_id)
        return SELECTION_MODEL_CACHE[cache_key]


def sam2_selection_mask(image: Image.Image, points: list[dict[str, float]], model_id: str, device: str = "auto") -> tuple[Image.Image, str]:
    import torch

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, hf_model_id = load_sam2_selection_model(model_id, selected_device)
    pixel_points = selection_points_to_pixels(points, *image.size)
    input_labels = [[[1 for _ in pixel_points]]]
    inputs = processor(
        images=image,
        input_points=[[pixel_points]],
        input_labels=input_labels,
        return_tensors="pt",
    )
    inputs = inputs.to(selected_device)

    with torch.inference_mode():
        outputs = model(**inputs)

    post_masks = processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=256.0,
        max_sprinkle_area=128.0,
    )
    masks = post_masks[0].detach().cpu().numpy().astype(bool)
    candidates = masks.reshape((-1, masks.shape[-2], masks.shape[-1]))
    scores = getattr(outputs, "iou_scores", None)
    if scores is not None and scores.numel() == len(candidates):
        best_index = int(torch.argmax(scores.detach().cpu().reshape(-1)).item())
    else:
        areas = candidates.reshape((len(candidates), -1)).sum(axis=1)
        best_index = int(np.argmax(areas))

    selected_mask = candidates[best_index]
    if not np.any(selected_mask):
        raise RuntimeError("SAM2 returned an empty selection mask")
    return Image.fromarray((selected_mask.astype(np.uint8) * 255), mode="L"), hf_model_id


def selected_image_from_mask(image: Image.Image, mask: Image.Image, background_mode: str) -> Image.Image:
    background_colors = {
        "neutral": (245, 245, 245),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    background = Image.new("RGB", image.size, background_colors.get(background_mode, background_colors["neutral"]))
    soft_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
    return Image.composite(image.convert("RGB"), background, soft_mask)


def selection_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (16, 185, 129, 105))
    clear = Image.new("RGBA", image.size, (0, 0, 0, 0))
    highlight = Image.composite(tint, clear, mask.convert("L"))
    return Image.alpha_composite(base, highlight)


def selection_tint(mask: Image.Image, alpha: int = 210) -> Image.Image:
    tint = Image.new("RGBA", mask.size, (16, 185, 129, max(0, min(255, alpha))))
    clear = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    return Image.composite(tint, clear, mask.convert("L"))


def normalize_output_reference(file_path: str) -> str:
    value = str(file_path or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Empty output file reference")
    if value.startswith("http://") or value.startswith("https://"):
        value = urlparse(value).path
    if value.startswith("/depth_data/"):
        value = value[len("/depth_data/") :]
    elif value.startswith("/diagnostics/"):
        value = value[len("/diagnostics/") :]
    return value.lstrip("/")


def union_selection_masks(mask_paths: list[str], image_size: tuple[int, int]) -> Image.Image:
    if not mask_paths:
        raise HTTPException(status_code=400, detail="Select at least one object to keep")

    width, height = image_size
    union = np.zeros((height, width), dtype=bool)
    for mask_reference in mask_paths:
        resolved_mask_path = resolve_output_file(normalize_output_reference(mask_reference), (".png", ".webp"))
        with Image.open(resolved_mask_path) as mask_image:
            loaded_mask = mask_image.convert("L")
            if loaded_mask.size != image_size:
                loaded_mask = loaded_mask.resize(image_size, Image.Resampling.NEAREST)
            union |= np.asarray(loaded_mask) > 0
    return Image.fromarray((union.astype(np.uint8) * 255), mode="L")


def get_runtime_info() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        }
    except Exception as exc:
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "unknown",
            "error": str(exc),
        }


@app.post("/selection/keep")
async def keep_selected_objects(
    file: UploadFile = File(...),
    points_json: str = Form("[]"),
    model_id: str = Form("sam2.1-hiera-large"),
    device: str = Form("auto"),
    background_mode: str = Form("neutral"),
    mask_max_dimension: int = Form(1024),
):
    points = parse_selection_points(points_json)
    if not points:
        raise HTTPException(status_code=400, detail="Click at least one object or image part to keep")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        with Image.open(file.file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc

    model_status = "fallback-click-region"
    model_error = None
    try:
        if model_id.startswith("sam2") or model_id == "grounding-dino-sam2":
            mask, resolved_model_id = sam2_selection_mask(image, points, model_id=model_id, device=device)
            model_status = "sam2-point-prompt"
            model_id = resolved_model_id
        else:
            mask = fallback_selection_mask(image, points, max_dimension=mask_max_dimension)
    except Exception as exc:
        model_error = str(exc)
        if "cache" in model_error.lower() and "local" in model_error.lower():
            model_error = "SAM2 checkpoint is not cached locally"
        elif len(model_error) > 220:
            model_error = f"{model_error[:217]}..."
        logger.warning("Selection model failed; using fallback click-region mask: %s", model_error)
        mask = fallback_selection_mask(image, points, max_dimension=mask_max_dimension)

    selected = selected_image_from_mask(image, mask, background_mode=background_mode)
    overlay = selection_overlay(image, mask)
    source_path = job_dir / "source.png"
    selected_path = job_dir / "selected_image.png"
    mask_path = job_dir / "selection_mask.png"
    overlay_path = job_dir / "selection_overlay.png"
    tint_path = job_dir / "selection_tint.png"
    metadata_path = job_dir / "selection.json"
    image.save(source_path)
    selected.save(selected_path)
    mask.save(mask_path)
    overlay.save(overlay_path)
    selection_tint(mask).save(tint_path)

    mask_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": file.filename,
        "points": points,
        "model_id": model_id,
        "model_status": model_status,
        "model_error": model_error,
        "background_mode": background_mode,
        "image_size": {"width": image.width, "height": image.height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, image.width * image.height)),
        "timings": {"selection_seconds": round(time.perf_counter() - started, 3)},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    selected_relative_path = output_relative_path(selected_path)
    mask_relative_path = output_relative_path(mask_path)
    overlay_relative_path = output_relative_path(overlay_path)
    tint_relative_path = output_relative_path(tint_path)
    metadata_relative_path = output_relative_path(metadata_path)
    return {
        **metadata,
        "selected_image": selected_relative_path,
        "selected_image_url": f"/depth_data/{selected_relative_path}",
        "mask": mask_relative_path,
        "mask_url": f"/depth_data/{mask_relative_path}",
        "overlay": overlay_relative_path,
        "overlay_url": f"/depth_data/{overlay_relative_path}",
        "tint": tint_relative_path,
        "tint_url": f"/depth_data/{tint_relative_path}",
        "metadata": metadata_relative_path,
        "metadata_url": f"/diagnostics/{metadata_relative_path}",
    }


@app.post("/selection/mask")
async def preview_selection_mask(
    file: UploadFile = File(...),
    points_json: str = Form("[]"),
    model_id: str = Form("sam2.1-hiera-large"),
    device: str = Form("auto"),
    mask_max_dimension: int = Form(1024),
):
    points = parse_selection_points(points_json)
    if not points:
        raise HTTPException(status_code=400, detail="Hover or click an object to preview a mask")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        with Image.open(file.file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc

    model_status = "fallback-click-region"
    model_error = None
    try:
        if model_id.startswith("sam2") or model_id == "grounding-dino-sam2":
            mask, resolved_model_id = sam2_selection_mask(image, points, model_id=model_id, device=device)
            model_status = "sam2-point-prompt"
            model_id = resolved_model_id
        else:
            mask = fallback_selection_mask(image, points, max_dimension=mask_max_dimension)
    except Exception as exc:
        model_error = str(exc)
        if "cache" in model_error.lower() and "local" in model_error.lower():
            model_error = "SAM2 checkpoint is not cached locally"
        elif len(model_error) > 220:
            model_error = f"{model_error[:217]}..."
        logger.warning("Selection preview model failed; using fallback click-region mask: %s", model_error)
        mask = fallback_selection_mask(image, points, max_dimension=mask_max_dimension)

    mask_path = job_dir / "selection_mask.png"
    overlay_path = job_dir / "selection_overlay.png"
    tint_path = job_dir / "selection_tint.png"
    metadata_path = job_dir / "selection.json"
    mask.save(mask_path)
    selection_overlay(image, mask).save(overlay_path)
    selection_tint(mask).save(tint_path)

    mask_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": file.filename,
        "points": points,
        "model_id": model_id,
        "model_status": model_status,
        "model_error": model_error,
        "image_size": {"width": image.width, "height": image.height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, image.width * image.height)),
        "timings": {"selection_seconds": round(time.perf_counter() - started, 3)},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    mask_relative_path = output_relative_path(mask_path)
    overlay_relative_path = output_relative_path(overlay_path)
    tint_relative_path = output_relative_path(tint_path)
    metadata_relative_path = output_relative_path(metadata_path)
    return {
        **metadata,
        "mask": mask_relative_path,
        "mask_url": f"/depth_data/{mask_relative_path}",
        "overlay": overlay_relative_path,
        "overlay_url": f"/depth_data/{overlay_relative_path}",
        "tint": tint_relative_path,
        "tint_url": f"/depth_data/{tint_relative_path}",
        "metadata": metadata_relative_path,
        "metadata_url": f"/diagnostics/{metadata_relative_path}",
    }


@app.post("/selection/compose")
async def compose_selected_objects(
    file: UploadFile = File(...),
    mask_paths_json: str = Form("[]"),
    background_mode: str = Form("neutral"),
):
    try:
        mask_paths = json.loads(mask_paths_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid mask paths JSON: {exc}") from exc
    if not isinstance(mask_paths, list) or not all(isinstance(path, str) for path in mask_paths):
        raise HTTPException(status_code=400, detail="Mask paths must be a JSON list of strings")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        with Image.open(file.file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc

    mask = union_selection_masks(mask_paths, image.size)
    selected = selected_image_from_mask(image, mask, background_mode=background_mode)
    overlay = selection_overlay(image, mask)

    source_path = job_dir / "source.png"
    selected_path = job_dir / "selected_image.png"
    mask_path = job_dir / "selection_mask.png"
    overlay_path = job_dir / "selection_overlay.png"
    tint_path = job_dir / "selection_tint.png"
    metadata_path = job_dir / "selection.json"
    image.save(source_path)
    selected.save(selected_path)
    mask.save(mask_path)
    overlay.save(overlay_path)
    selection_tint(mask).save(tint_path)

    mask_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": file.filename,
        "mask_count": len(mask_paths),
        "model_status": "composed-clicked-masks",
        "background_mode": background_mode,
        "image_size": {"width": image.width, "height": image.height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, image.width * image.height)),
        "timings": {"selection_seconds": round(time.perf_counter() - started, 3)},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    selected_relative_path = output_relative_path(selected_path)
    mask_relative_path = output_relative_path(mask_path)
    overlay_relative_path = output_relative_path(overlay_path)
    tint_relative_path = output_relative_path(tint_path)
    metadata_relative_path = output_relative_path(metadata_path)
    return {
        **metadata,
        "selected_image": selected_relative_path,
        "selected_image_url": f"/depth_data/{selected_relative_path}",
        "mask": mask_relative_path,
        "mask_url": f"/depth_data/{mask_relative_path}",
        "overlay": overlay_relative_path,
        "overlay_url": f"/depth_data/{overlay_relative_path}",
        "tint": tint_relative_path,
        "tint_url": f"/depth_data/{tint_relative_path}",
        "metadata": metadata_relative_path,
        "metadata_url": f"/diagnostics/{metadata_relative_path}",
    }


@app.post("/process_image")
async def process_image(
    file: UploadFile = File(...),
    depth_provider: str = Form(DEFAULT_DEPTH_PROVIDER),
    depth_model: str | None = Form(None),
    device: str = Form("auto"),
    target_dimension: int = Form(300),
    z_scale: float = Form(50),
    max_xy_size: float | None = Form(None),
    printer_profile: str | None = Form(None),
    printer_max_x_mm: float | None = Form(None),
    printer_max_y_mm: float | None = Form(None),
    printer_max_z_mm: float | None = Form(None),
    printer_clearance_mm: float | None = Form(None),
    print_scale_percent: float | None = Form(None),
    relief_polarity: str = Form("raised-print"),
    mesh_resolution_multiplier: float | None = Form(None),
    invert: bool = Form(False),
    sigma: float = Form(0.6),
    relief_gamma: float = Form(0.75),
    detail_boost: float = Form(1.4),
    detail_radius: float = Form(2.0),
    low_percentile: float = Form(1.0),
    high_percentile: float = Form(99.0),
    base_border_px: int = Form(2),
    completion_mode: str = Form("none"),
    completion_provider: str = Form("mirror"),
    completion_prompt: str | None = Form(None),
    completion_model: str | None = Form(None),
    completion_lora_weights: str | None = Form(None),
    completion_lora_scale: float | None = Form(None),
    completion_steps: int = Form(24),
    completion_guidance: float | None = Form(None),
    completion_seed: int | None = Form(None),
    completion_inpaint_max_dimension: int = Form(768),
):
    logger.info(f"Received file: {file.filename}")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.perf_counter()
    timings: dict[str, float] = {}

    def record_timing(name: str, started: float) -> None:
        timings[name] = round(time.perf_counter() - started, 3)

    upload_suffix = Path(file.filename or "").suffix or ".jpg"
    with NamedTemporaryFile(delete=False, suffix=upload_suffix, dir=job_dir) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the image and get depth data
        selected_model = depth_model or DEFAULT_DEPTH_MODEL
        logger.info(
            "Processing image to get depth data with provider=%s model=%s device=%s",
            depth_provider,
            selected_model,
            device,
        )
        stage_started = time.perf_counter()
        completed_image_path, applied_completion_mode = complete_image(
            temp_file_path,
            output_dir=str(job_dir),
            mode=completion_mode,
            provider=completion_provider,
            prompt=completion_prompt,
            model_name=completion_model,
            lora_weights=completion_lora_weights,
            lora_scale=completion_lora_scale,
            device=device,
            num_inference_steps=completion_steps,
            guidance_scale=completion_guidance,
            seed=completion_seed,
            inpaint_max_dimension=completion_inpaint_max_dimension,
        )
        record_timing("completion_seconds", stage_started)
        image_for_depth = completed_image_path

        stage_started = time.perf_counter()
        depth_data_path = process_image_get_depth_data(
            image_for_depth,
            output_dir=str(job_dir),
            provider=depth_provider,
            model_name=selected_model,
            device=device,
        )
        record_timing("depth_seconds", stage_started)
        logger.info(f"Depth data saved as: {depth_data_path}")
        depth_metadata_path = job_dir / "output_depth_metadata.json"
        depth_metadata = {}
        if depth_metadata_path.exists():
            with open(depth_metadata_path, encoding="utf-8") as depth_metadata_file:
                depth_metadata = json.load(depth_metadata_file)
        effective_depth_model = depth_metadata.get("effective_model") or selected_model
        effective_invert = relief_invert_for_model(effective_depth_model, relief_polarity, invert)
        relief_value_transform = depth_metadata.get("relief_value_transform")
        if not relief_value_transform:
            relief_value_transform = relief_value_transform_for_model(effective_depth_model)
        
        # Generate 3D model
        logger.info("Generating 3D model...")
        stl_path = job_dir / "output_model.stl"
        stage_started = time.perf_counter()
        depth_data_to_3d_model(
            depth_data_path,
            output_stl_path=str(stl_path),
            target_dimension=target_dimension,
            z_scale=z_scale,
            max_xy_size=max_xy_size,
            invert=effective_invert,
            sigma=sigma,
            relief_gamma=relief_gamma,
            detail_boost=detail_boost,
            detail_radius=detail_radius,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            base_border_px=base_border_px,
            value_transform=relief_value_transform,
        )
        record_timing("stl_seconds", stage_started)
        logger.info(f"3D model saved as: {stl_path}")
        
        # Check if the STL file was actually created
        if not os.path.exists(stl_path):
            raise FileNotFoundError(f"STL file was not created at {stl_path}")

        stage_started = time.perf_counter()
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "depth-relief",
                "artifact_contract": "output_model.stl + diagnostics.json",
            }
        )
        record_timing("diagnostics_seconds", stage_started)
        timings["total_seconds"] = round(time.perf_counter() - request_started, 3)
        runtime = get_runtime_info()
        diagnostics_path = job_dir / "diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as diagnostics_file:
            json.dump(diagnostics, diagnostics_file, indent=2, allow_nan=False)

        metadata = {
            "job_id": job_id,
            "source_filename": file.filename,
            "depth_provider": depth_provider,
            "depth_model": effective_depth_model,
            "requested_depth_model": selected_model,
            "depth_fallback_model": depth_metadata.get("fallback_model"),
            "depth_fallback_reason": depth_metadata.get("fallback_reason"),
            "depth_metadata": depth_metadata,
            "device": device,
            "target_dimension": target_dimension,
            "z_scale": z_scale,
            "max_xy_size": max_xy_size,
            "printer": {
                "profile": printer_profile,
                "max_x_mm": printer_max_x_mm,
                "max_y_mm": printer_max_y_mm,
                "max_z_mm": printer_max_z_mm,
                "clearance_mm": printer_clearance_mm,
                "print_scale_percent": print_scale_percent,
            },
            "relief_polarity": relief_polarity,
            "mesh_resolution_multiplier": mesh_resolution_multiplier,
            "invert": effective_invert,
            "requested_invert": invert,
            "relief_value_transform": relief_value_transform,
            "sigma": sigma,
            "relief_gamma": relief_gamma,
            "detail_boost": detail_boost,
            "detail_radius": detail_radius,
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
            "base_border_px": base_border_px,
            "completion_mode": completion_mode,
            "completion_provider": completion_provider,
            "completion_model": completion_model,
            "completion_lora_weights": completion_lora_weights,
            "completion_lora_scale": completion_lora_scale,
            "completion_steps": completion_steps,
            "completion_guidance": completion_guidance,
            "completion_seed": completion_seed,
            "completion_inpaint_max_dimension": completion_inpaint_max_dimension,
            "applied_completion_mode": applied_completion_mode,
            "runtime": runtime,
            "timings": timings,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(job_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        depth_relative_path = output_relative_path(depth_data_path)
        stl_relative_path = output_relative_path(stl_path)
        diagnostics_relative_path = output_relative_path(diagnostics_path)
        completed_image_relative_path = (
            output_relative_path(completed_image_path)
            if completed_image_path and applied_completion_mode
            else None
        )
        
        # Return paths to the generated files
        response = {
            **metadata,
            "depth_data": depth_relative_path,
            "depth_data_url": f"/depth_data/{depth_relative_path}",
            "stl_model": stl_relative_path,
            "stl_url": f"/stl_model/{stl_relative_path}",
            "diagnostics": diagnostics_relative_path,
            "diagnostics_url": f"/diagnostics/{diagnostics_relative_path}",
            "stl_diagnostics": diagnostics,
        }
        if completed_image_relative_path:
            response["completed_image"] = completed_image_relative_path
            response["completed_image_url"] = f"/depth_data/{completed_image_relative_path}"
        return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@app.post("/upload_to_masv")
async def upload_to_masv_endpoint(file_name: str = Form(...)):
    logger.info(f"Received request to upload file to MASV: {file_name}")
    
    file_path = resolve_output_file(file_name, (".stl",))
    
    try:
        # Upload to MASV
        logger.info("Uploading to MASV...")
        masv_package_id = await upload_to_masv(str(file_path), file_path.name)
        logger.info(f"Uploaded to MASV. Package ID: {masv_package_id}")
        
        return {"masv_package_id": masv_package_id}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during MASV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during MASV upload: {str(e)}")

async def upload_to_masv(file_path: str, file_name: str):
    # MASV API endpoints
    create_package_url = "https://api.massive.app/v1/teams/{team_id}/packages"
    add_file_url = "https://api.massive.app/v1/packages/{package_id}/files"
    finalize_package_url = "https://api.massive.app/v1/packages/{package_id}/finalize"

    headers = {
        "X-API-KEY": MASV_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Create a package
            package_data = {
                "name": "3D Print Model",
                "description": "Uploaded 3D print model",
                "recipients": ["jennylive158@gmail.com"]
            }
            async with session.post(create_package_url.format(team_id=MASV_TEAM_ID), json=package_data, headers=headers) as response:
                response.raise_for_status()
                package = await response.json()
                package_id = package["id"]
                package_token = package["access_token"]
                logger.info(f"Package created successfully. ID: {package_id}")

            # Step 2: Add file to the package
            file_data = {
                "kind": "file",
                "name": file_name,
                "path": "",
                "last_modified": datetime.utcnow().isoformat() + "Z"
            }
            headers["X-Package-Token"] = package_token
            async with session.post(add_file_url.format(package_id=package_id), json=file_data, headers=headers) as response:
                response.raise_for_status()
                file_info = await response.json()
                file_id = file_info["file"]["id"]
                logger.info(f"File added to package successfully. File ID: {file_id}")

            # Step 3: Create the file in MASV's cloud storage
            create_blueprint = file_info["create_blueprint"]
            async with session.request(
                create_blueprint["method"],
                create_blueprint["url"],
                headers=create_blueprint.get("headers", {})
            ) as response:
                response.raise_for_status()
                create_response = await response.text()
                logger.info("File created in MASV's cloud storage")
                
                # Parse the XML response to get the UploadId
                upload_id = re.search("<UploadId>(.*?)</UploadId>", create_response).group(1)

            # Step 4: Obtain upload URLs
            chunk_size = 5 * 1024 * 1024  # 5 MB chunks
            file_size = os.path.getsize(file_path)
            chunk_count = math.ceil(file_size / chunk_size)
            
            async with session.post(
                f"{add_file_url.format(package_id=package_id)}/{file_id}",
                params={"start": 0, "count": chunk_count},
                json={"upload_id": upload_id},
                headers=headers
            ) as response:
                response.raise_for_status()
                upload_urls = await response.json()
                logger.info(f"Obtained {len(upload_urls)} upload URLs")

            # Step 5: Upload file chunks
            chunk_extras = []
            with open(file_path, 'rb') as f:
                for i, upload_info in enumerate(upload_urls, start=1):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    logger.info(f"Uploading chunk {i}")
                    async with session.request(
                        upload_info["method"],
                        upload_info["url"],
                        data=chunk
                    ) as response:
                        response.raise_for_status()
                        etag = response.headers.get("ETag")
                        chunk_extras.append({"partNumber": str(i), "etag": etag})
                    logger.info(f"Chunk {i} uploaded successfully")

            # Step 6: Finalize the file
            finalize_data = {
                "chunk_extras": chunk_extras,
                "file_extras": {"upload_id": upload_id},
                "size": file_size,
                "chunk_size": chunk_size
            }
            async with session.post(
                f"{add_file_url.format(package_id=package_id)}/{file_id}/finalize",
                json=finalize_data,
                headers=headers
            ) as response:
                response.raise_for_status()
                logger.info("File upload finalized successfully")

            # Step 7: Finalize the package
            async with session.post(finalize_package_url.format(package_id=package_id), headers=headers) as response:
                response.raise_for_status()
                logger.info("Package finalized successfully")

        return package_id
    except aiohttp.ClientResponseError as e:
        logger.error(f"MASV API error: {e.status}, message='{e.message}', url='{e.request_info.url}'")
        logger.error(f"Request headers: {e.request_info.headers}")
        logger.error(f"Response headers: {e.headers}")
        raise HTTPException(status_code=500, detail=f"MASV API error: {e.status}, {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error during MASV upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during MASV upload: {str(e)}")

@app.get("/depth_data/{file_path:path}")
async def get_depth_data(file_path: str):
    resolved_path = resolve_output_file(file_path, (".npy", ".png", ".webp"))
    return FileResponse(resolved_path)

@app.get("/stl_model/{file_path:path}")
async def get_stl_model(file_path: str):
    resolved_path = resolve_output_file(file_path, (".stl",))
    return FileResponse(resolved_path)

@app.get("/diagnostics/{file_path:path}")
async def get_diagnostics(file_path: str):
    resolved_path = resolve_output_file(file_path, (".json",))
    return FileResponse(resolved_path, media_type="application/json")

def sanitize_float(x):
    if np.isnan(x) or np.isinf(x):
        return -12345678  # or another appropriate default value
    return float(x)


def completion_provider_rows() -> list[dict]:
    notes = {
        "sdxl-inpaint": "Public SDXL inpainting checkpoint. Practical middle tier for a 12 GB GPU when run at 384-512 px with fp16 and CPU offload.",
        "dreamshaper-inpaint": "Public SD1.5-style inpainting checkpoint. Smaller fp16 footprint than SDXL and useful as the first learned baseline to beat mirror/biharmonic.",
        "amused-inpaint": "Small public masked-token inpainting model. Useful as a fast non-diffusion learned baseline against mirror/biharmonic.",
        "flux-fill": "Modern rectified-flow inpainting/outpainting model. Large and may require Hugging Face access plus CPU offload.",
        "qwen-image-inpaint": "Qwen Image model through the Diffusers inpaint pipeline. Large; preserves visible pixels after generation.",
        "qwen-image-edit": "Official Qwen Image Edit pipeline prompted to fill the blank half. Large; visible pixels are restored after generation.",
    }
    rows = [
        {
            "id": "mirror",
            "label": "Mirror prior",
            "local": True,
            "gpu_supported": False,
            "notes": "Fast geometric symmetry prior. No diffusion model.",
        },
        {
            "id": "mirror-seam-repair",
            "label": "Mirror seam repair",
            "local": True,
            "gpu_supported": False,
            "notes": "Experimental fast mirror prior with a small classical inpaint repair band along the generated seam. Benchmark before making it the default.",
        }
    ]
    for provider_id, provider in MODERN_INPAINT_MODELS.items():
        rows.append(
            {
                "id": provider_id,
                "label": provider["label"],
                "model": provider["model"],
                "local": True,
                "gpu_supported": True,
                "notes": notes.get(provider_id, "Modern diffusion-based completion provider."),
            }
        )
    return rows

@app.get("/depth_data_downsampled/{file_path:path}")
async def get_depth_data_downsampled(file_path: str):
    resolved_path = resolve_output_file(file_path, (".npy",))
    
    try:
        # Load the depth data
        depth_data = np.load(resolved_path)
        
        # Get original dimensions
        original_height, original_width = depth_data.shape
        
        # Calculate the scaling factor
        max_dimension = max(original_height, original_width)
        scale_factor = max(1, int(max_dimension / 100))
        
        # Downsample the depth data by skipping pixels
        downsampled_data = depth_data[::scale_factor, ::scale_factor]
        
        # Get new dimensions
        new_height, new_width = downsampled_data.shape
        
        # Sanitize and convert to list for JSON serialization
        downsampled_list = [[sanitize_float(x) for x in row] for row in downsampled_data]
        
        return JSONResponse(content={
            "depth_data": downsampled_list,
            "original_dimensions": {
                "height": original_height,
                "width": original_width
            },
            "downsampled_dimensions": {
                "height": new_height,
                "width": new_width
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing depth data: {str(e)}")

async def get_rbc_session():
    return aiohttp.ClientSession(headers={"Authorization": f"Bearer {RBC_ACCESS_TOKEN}"})

async def create_transaction(session, member_id, points):
    transaction_data = {
        "partnerRefId": f"AWARD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "amount": points,
        "note": "Points awarded",
        "type": "PAYMENT"
    }
    try:
        async with session.post(f"{RBC_API_BASE_URL}/loyalty/{member_id}/transactions", json=transaction_data) as response:
            response_text = await response.text()
            logger.info(f"RBC API Response: Status {response.status}, Body: {response_text}")
            if response.status != 200:
                error_detail = f"Error creating transaction. Status: {response.status}, Response: {response_text}"
                logger.error(error_detail)
                raise HTTPException(status_code=response.status, detail=error_detail)
            return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Network error when calling RBC API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error when calling RBC API: {str(e)}")

@app.post("/award_rbc_points")
async def award_rbc_points(
    member_id: int = Query(..., description="The ID of the member to award points to"),
    points: int = Query(..., description="The number of points to award")
):
    logger.info(f"Awarding {points} points to member {member_id}")
    
    async with await get_rbc_session() as session:
        try:
            # Create a transaction record (which also awards the points)
            transaction = await create_transaction(session, member_id, points)
            
            return {
                "status": "success",
                "message": f"Awarded {points} points to member {member_id}",
                "member_id": member_id,
                "points_awarded": points,
                "transaction": transaction["transaction"]
            }
        except HTTPException as e:
            logger.error(f"Error awarding points: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_depth_provider": DEFAULT_DEPTH_PROVIDER,
        "default_depth_model": DEFAULT_DEPTH_MODEL,
        "output_dir": str(OUTPUT_DIR),
        "runtime": get_runtime_info(),
    }


@app.get("/depth/preload/depthpro/status")
async def depthpro_preload_status():
    return _depthpro_cache_status()


@app.post("/depth/preload/depthpro")
async def start_depthpro_preload(force: bool = Query(False)):
    status = _depthpro_cache_status()
    if status["complete"]:
        return status

    already_downloading = False
    with DEPTH_PRELOAD_LOCK:
        if DEPTH_PRELOAD_STATE.get("status") == "downloading":
            already_downloading = True
        else:
            DEPTH_PRELOAD_STATE.update(
                {
                    "status": "downloading",
                    "message": "Starting Apple Depth Pro preload.",
                    "started_at": datetime.utcnow().isoformat() + "Z",
                    "finished_at": None,
                    "error": None,
                }
            )

    if already_downloading:
        return _depthpro_cache_status()

    thread = Thread(target=_depthpro_preload_worker, kwargs={"force": force}, daemon=True)
    thread.start()
    return _depthpro_cache_status()


@app.get("/models")
async def get_models():
    return {
        "default_provider": DEFAULT_DEPTH_PROVIDER,
        "providers": [
            {
                "id": "transformers",
                "label": "Local Transformers depth",
                "model": DEFAULT_DEPTH_MODEL,
                "local": True,
                "gpu_supported": True,
                "models": DEPTH_MODELS,
            },
            {
                "id": "sapiens",
                "label": "Sapiens Depth",
                "model": "facebook/sapiens_depth",
                "local": False,
                "gpu_supported": False,
            },
        ],
        "completion_modes": [
            {
                "id": "none",
                "label": "No completion",
                "notes": "Estimate depth only from the input pixels.",
            },
            {
                "id": "mirror-auto",
                "label": "Auto mirror completion",
                "notes": "Mirror the visually richer half across the center before depth estimation.",
            },
            {
                "id": "mirror-left-to-right",
                "label": "Mirror left to right",
                "notes": "Use the left half to synthesize the right half.",
            },
            {
                "id": "mirror-right-to-left",
                "label": "Mirror right to left",
                "notes": "Use the right half to synthesize the left half.",
            },
        ],
        "completion_providers": completion_provider_rows(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
