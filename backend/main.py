import os
import json
import hashlib
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
        complete_selection_context_with_modern_inpaint,
        compose_selection_depth_with_context,
        depth_data_to_3d_model,
        process_image_get_depth_data,
        release_depth_pipelines,
        release_inpaint_pipelines,
        relief_value_transform_for_model,
    )
    from .face_depth_refinement import (
        DEFAULT_FACE_DETAIL_STRENGTH,
        DEFAULT_FACE_FEATHER_RATIO,
        DEFAULT_FACE_MAX_CORRECTION_RATIO,
        refine_depth_for_faces,
    )
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from pic_to_3d import (
        MODERN_INPAINT_MODELS,
        complete_image,
        complete_selection_context_with_modern_inpaint,
        compose_selection_depth_with_context,
        depth_data_to_3d_model,
        process_image_get_depth_data,
        release_depth_pipelines,
        release_inpaint_pipelines,
        relief_value_transform_for_model,
    )
    from face_depth_refinement import (
        DEFAULT_FACE_DETAIL_STRENGTH,
        DEFAULT_FACE_FEATHER_RATIO,
        DEFAULT_FACE_MAX_CORRECTION_RATIO,
        refine_depth_for_faces,
    )
try:
    from .stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
try:
    from .runtime_provenance import runtime_source_provenance
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from runtime_provenance import runtime_source_provenance
try:
    from .scene_diorama import build_scene_diorama
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    from scene_diorama import build_scene_diorama
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
DEFAULT_LOCAL_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
DEFAULT_LOCAL_ORIGIN_REGEX = (
    r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$"
)
CORS_ORIGINS = list(
    dict.fromkeys(
        origin.strip()
        for origin in [*DEFAULT_LOCAL_ORIGINS, *os.getenv("CORS_ORIGINS", "").split(",")]
        if origin.strip()
    )
)
CORS_ORIGIN_REGEX = (
    os.getenv("CORS_ORIGIN_REGEX", "").strip() or DEFAULT_LOCAL_ORIGIN_REGEX
)

# Updated CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
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
PANOPTIC_SELECTION_MODEL_ID = os.getenv("SELECTION_PANOPTIC_MODEL", "facebook/detr-resnet-50-panoptic")
SELECTION_PRECOMPUTE_LOCK = Lock()
SELECTION_PRECOMPUTE_CACHE: dict[str, dict] = {}
SELECTION_PRECOMPUTE_MAX_ENTRIES = int(os.getenv("SELECTION_PRECOMPUTE_MAX_ENTRIES", "12"))
RELIEF_MIN_DETAIL_DIMENSION = int(os.getenv("RELIEF_MIN_DETAIL_DIMENSION", "192"))
RELIEF_MAX_DETAIL_DIMENSION = int(os.getenv("RELIEF_MAX_DETAIL_DIMENSION", "900"))
DEFAULT_NOZZLE_DIAMETER_MM = float(os.getenv("DEFAULT_NOZZLE_DIAMETER_MM", "0.4"))
DEFAULT_MAX_RELIEF_SLOPE = float(os.getenv("DEFAULT_MAX_RELIEF_SLOPE", "2.0"))

DEPTH_MODELS = [
    {
        "id": "depth-anything/DA3-LARGE-1.1",
        "label": "Depth Anything V3 Large 1.1",
        "provider": "transformers",
        "recommended": False,
        "depth_value_semantics": "relative_distance_far_high",
        "notes": "Modern any-view depth with direct distance output; 1.64 GB first download.",
    },
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


def _positive_float(value: float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def resolve_relief_target_dimension(
    target_dimension: int,
    *,
    max_xy_size: float | None = None,
    printer_max_x_mm: float | None = None,
    printer_max_y_mm: float | None = None,
    printer_clearance_mm: float | None = None,
    mesh_resolution_multiplier: float | None = None,
) -> int:
    if int(target_dimension) == -1:
        return -1

    requested = max(2, int(round(float(target_dimension))))
    multiplier = max(1.0, _positive_float(mesh_resolution_multiplier) or 1.0)
    floors = [requested, RELIEF_MIN_DETAIL_DIMENSION]

    physical_xy = _positive_float(max_xy_size)
    if physical_xy is not None:
        floors.append(int(round(physical_xy * multiplier)))

    max_x = _positive_float(printer_max_x_mm)
    max_y = _positive_float(printer_max_y_mm)
    if max_x is not None and max_y is not None:
        clearance = max(0.0, float(printer_clearance_mm or 0.0))
        usable_xy = max(1.0, min(max_x - clearance * 2, max_y - clearance * 2))
        floors.append(int(round(usable_xy * multiplier)))

    return min(max(floors), max(RELIEF_MIN_DETAIL_DIMENSION, RELIEF_MAX_DETAIL_DIMENSION))


def relief_sample_pitch_mm(max_xy_size: float | None, target_dimension: int) -> float | None:
    physical_xy = _positive_float(max_xy_size)
    if physical_xy is None or target_dimension in (-1, 0, 1):
        return None
    return physical_xy / float(max(1, target_dimension - 1))


def resolve_minimum_feature_mm(
    nozzle_diameter_mm: float | None,
    requested_minimum_feature_mm: float | None,
) -> float:
    nozzle = _positive_float(nozzle_diameter_mm) or DEFAULT_NOZZLE_DIAMETER_MM
    requested = _positive_float(requested_minimum_feature_mm) or 0.0
    return max(requested, nozzle * 2.0)


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


def selection_source_fingerprint(image: Image.Image) -> str:
    source = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{source.width}x{source.height}:".encode("ascii"))
    digest.update(source.tobytes())
    return digest.hexdigest()


def resolve_selection_compose_job(job_id: str) -> dict:
    normalized_job_id = str(job_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", normalized_job_id):
        raise HTTPException(status_code=400, detail="Invalid selection compose job")

    job_dir = (OUTPUT_DIR / "selection" / normalized_job_id).resolve()
    if not is_relative_to(job_dir, OUTPUT_DIR / "selection"):
        raise HTTPException(status_code=400, detail="Invalid selection compose job")

    source_path = job_dir / "source.png"
    selected_path = job_dir / "selected_image.png"
    face_detection_path = job_dir / "selection_face_detection.png"
    mask_path = job_dir / "selection_mask.png"
    metadata_path = job_dir / "selection.json"
    if not all(path.is_file() for path in (source_path, selected_path, mask_path, metadata_path)):
        raise HTTPException(status_code=404, detail="Selection compose job is incomplete or expired")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="Selection compose metadata is invalid") from exc
    if (
        metadata.get("job_id") != normalized_job_id
        or metadata.get("model_status") != "composed-clicked-masks"
    ):
        raise HTTPException(status_code=409, detail="Selection compose provenance does not match")

    try:
        with Image.open(source_path) as source_image:
            source_size = source_image.size
            source_fingerprint = selection_source_fingerprint(source_image)
        with Image.open(selected_path) as selected_image:
            selected_size = selected_image.size
        face_detection_size = None
        if face_detection_path.is_file():
            with Image.open(face_detection_path) as face_detection_image:
                face_detection_size = face_detection_image.size
        with Image.open(mask_path) as mask_image:
            mask_size = mask_image.size
            mask_pixels = int(np.count_nonzero(np.asarray(mask_image.convert("L")) > 0))
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Selection compose artifacts are invalid") from exc
    if (
        source_size != selected_size
        or source_size != mask_size
        or (face_detection_size is not None and source_size != face_detection_size)
        or mask_pixels < 4
    ):
        raise HTTPException(status_code=409, detail="Selection compose artifacts do not align")
    if metadata.get("source_fingerprint") != source_fingerprint:
        raise HTTPException(status_code=409, detail="Selection compose source provenance does not match")

    return {
        "job_id": normalized_job_id,
        "job_dir": job_dir,
        "source_path": source_path,
        "selected_path": selected_path,
        "face_detection_path": (
            face_detection_path if face_detection_path.is_file() else selected_path
        ),
        "mask_path": mask_path,
        "metadata": metadata,
    }


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


def load_panoptic_selection_model(device: str):
    import torch
    from transformers import DetrForSegmentation, DetrImageProcessor

    cache_key = (PANOPTIC_SELECTION_MODEL_ID, device)
    with SELECTION_MODEL_LOCK:
        if cache_key in SELECTION_MODEL_CACHE:
            return SELECTION_MODEL_CACHE[cache_key]

        allow_download = os.getenv("SELECTION_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
        local_files_only = not allow_download
        processor = DetrImageProcessor.from_pretrained(PANOPTIC_SELECTION_MODEL_ID, local_files_only=local_files_only)
        model = DetrForSegmentation.from_pretrained(PANOPTIC_SELECTION_MODEL_ID, local_files_only=local_files_only)
        model.to(device)
        model.eval()
        SELECTION_MODEL_CACHE[cache_key] = (processor, model, PANOPTIC_SELECTION_MODEL_ID)
        return SELECTION_MODEL_CACHE[cache_key]


def nearest_panoptic_segment_id(segmentation: np.ndarray, seed_x: int, seed_y: int) -> int | None:
    height, width = segmentation.shape
    if 0 <= seed_x < width and 0 <= seed_y < height:
        segment_id = int(segmentation[seed_y, seed_x])
        if segment_id >= 0:
            return segment_id

    for radius in (8, 16, 32, 64, 96):
        left = max(0, seed_x - radius)
        right = min(width, seed_x + radius + 1)
        top = max(0, seed_y - radius)
        bottom = min(height, seed_y + radius + 1)
        window = segmentation[top:bottom, left:right]
        valid_y, valid_x = np.where(window >= 0)
        if len(valid_x) == 0:
            continue
        absolute_x = valid_x + left
        absolute_y = valid_y + top
        distances = (absolute_x - seed_x) ** 2 + (absolute_y - seed_y) ** 2
        nearest_index = int(np.argmin(distances))
        return int(window[valid_y[nearest_index], valid_x[nearest_index]])
    return None


def compute_panoptic_segmentation(image: Image.Image, device: str = "auto") -> tuple[np.ndarray, dict[int, str], str]:
    import torch

    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, model_id = load_panoptic_selection_model(selected_device)

    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(selected_device)
    with torch.inference_mode():
        outputs = model(**inputs)

    processed = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[(image.height, image.width)],
        label_ids_to_fuse=set(),
    )[0]
    segmentation = processed["segmentation"].detach().cpu().numpy().astype(np.int32)
    segment_labels = {
        int(segment["id"]): str(model.config.id2label.get(int(segment["label_id"]), segment.get("label_id", "object")))
        for segment in processed.get("segments_info", [])
    }
    return segmentation, segment_labels, model_id


def panoptic_mask_from_segmentation(
    segmentation: np.ndarray,
    segment_labels: dict[int, str],
    points: list[dict[str, float]],
    image_size: tuple[int, int],
) -> tuple[Image.Image, list[str], list[int]]:
    width, height = image_size
    pixels = selection_points_to_pixels(points, width, height)
    selected_ids: set[int] = set()
    for px_float, py_float in pixels:
        segment_id = nearest_panoptic_segment_id(segmentation, int(round(px_float)), int(round(py_float)))
        if segment_id is not None:
            selected_ids.add(segment_id)

    if not selected_ids:
        raise RuntimeError("Panoptic segmenter did not find an object under the cursor")

    mask_data = np.isin(segmentation, list(selected_ids))
    if not np.any(mask_data):
        raise RuntimeError("Panoptic segmenter returned an empty selection mask")

    mask = Image.fromarray((mask_data.astype(np.uint8) * 255), mode="L")
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    labels = sorted({segment_labels.get(segment_id, f"segment-{segment_id}") for segment_id in selected_ids})
    return mask, labels, sorted(selected_ids)


def panoptic_selection_mask(
    image: Image.Image,
    points: list[dict[str, float]],
    device: str = "auto",
) -> tuple[Image.Image, str, list[str]]:
    segmentation, segment_labels, model_id = compute_panoptic_segmentation(image, device=device)
    mask, labels, _selected_ids = panoptic_mask_from_segmentation(segmentation, segment_labels, points, image.size)
    return mask, model_id, labels


def compact_selection_error(exc: Exception, model_label: str = "model") -> str:
    message = str(exc)
    if "cache" in message.lower() and "local" in message.lower():
        return f"{model_label} checkpoint is not cached locally"
    if "couldn't connect" in message.lower() and "huggingface.co" in message.lower():
        return f"{model_label} checkpoint is not cached locally"
    if len(message) > 220:
        return f"{message[:217]}..."
    return message


def selection_mask_for_points(
    image: Image.Image,
    points: list[dict[str, float]],
    model_id: str,
    device: str = "auto",
    mask_max_dimension: int = 1024,
) -> tuple[Image.Image, str, str | None, str, list[str]]:
    model_error = None
    if model_id.startswith("sam2") or model_id == "grounding-dino-sam2":
        try:
            mask, resolved_model_id = sam2_selection_mask(image, points, model_id=model_id, device=device)
            return mask, resolved_model_id, model_error, "sam2-point-prompt", []
        except Exception as exc:
            model_error = compact_selection_error(exc, "SAM2")
            logger.warning("SAM2 selection failed; trying panoptic segmenter: %s", model_error)

    try:
        mask, resolved_model_id, labels = panoptic_selection_mask(image, points, device=device)
        return mask, resolved_model_id, model_error, "panoptic-click-segment", labels
    except Exception as exc:
        panoptic_error = compact_selection_error(exc, "Panoptic")
        logger.warning("Panoptic selection failed; using fallback click-region mask: %s", panoptic_error)
        model_error = f"{model_error}; panoptic: {panoptic_error}" if model_error else panoptic_error

    mask = fallback_selection_mask(image, points, max_dimension=mask_max_dimension)
    return mask, model_id, model_error, "fallback-click-region", []


def trim_selection_precompute_cache() -> None:
    if len(SELECTION_PRECOMPUTE_CACHE) <= SELECTION_PRECOMPUTE_MAX_ENTRIES:
        return
    sorted_items = sorted(SELECTION_PRECOMPUTE_CACHE.items(), key=lambda item: item[1].get("created_at_epoch", 0.0))
    for cache_id, _cache_entry in sorted_items[: max(0, len(sorted_items) - SELECTION_PRECOMPUTE_MAX_ENTRIES)]:
        SELECTION_PRECOMPUTE_CACHE.pop(cache_id, None)


def cache_panoptic_precompute(
    segmentation: np.ndarray,
    segment_labels: dict[int, str],
    model_id: str,
    image_size: tuple[int, int],
) -> str:
    precompute_id = uuid4().hex
    with SELECTION_PRECOMPUTE_LOCK:
        SELECTION_PRECOMPUTE_CACHE[precompute_id] = {
            "segmentation": segmentation,
            "segment_labels": segment_labels,
            "model_id": model_id,
            "image_size": image_size,
            "created_at_epoch": time.time(),
        }
        trim_selection_precompute_cache()
    return precompute_id


def get_selection_precompute(precompute_id: str) -> dict:
    with SELECTION_PRECOMPUTE_LOCK:
        cached = SELECTION_PRECOMPUTE_CACHE.get(precompute_id)
        if cached:
            cached["created_at_epoch"] = time.time()
    if not cached:
        raise HTTPException(status_code=404, detail="Selection precompute session not found")
    return cached


def save_selection_mask_artifacts(
    job_dir: Path,
    mask: Image.Image,
    metadata: dict,
) -> dict:
    mask_path = job_dir / "selection_mask.png"
    tint_path = job_dir / "selection_tint.png"
    metadata_path = job_dir / "selection.json"
    mask.save(mask_path)
    selection_tint(mask).save(tint_path)
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    mask_relative_path = output_relative_path(mask_path)
    tint_relative_path = output_relative_path(tint_path)
    metadata_relative_path = output_relative_path(metadata_path)
    return {
        "mask": mask_relative_path,
        "mask_url": f"/depth_data/{mask_relative_path}",
        "tint": tint_relative_path,
        "tint_url": f"/depth_data/{tint_relative_path}",
        "metadata": metadata_relative_path,
        "metadata_url": f"/diagnostics/{metadata_relative_path}",
    }


SELECTION_INFILL_MODES = {
    "none",
    "structural-context",
    "clean-context",
    "generative-context",
}


def selected_image_from_mask(image: Image.Image, mask: Image.Image, background_mode: str) -> Image.Image:
    background_colors = {
        "neutral": (245, 245, 245),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    background = Image.new("RGB", image.size, background_colors.get(background_mode, background_colors["neutral"]))
    soft_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
    return Image.composite(image.convert("RGB"), background, soft_mask)


def _fill_small_selection_holes(
    selected: np.ndarray,
    *,
    max_hole_area_ratio: float,
) -> tuple[np.ndarray, int, int]:
    height, width = selected.shape
    pixel_count = max(1, width * height)
    max_hole_pixels = max(64, int(math.floor(pixel_count * max_hole_area_ratio)))
    filled = selected.copy()
    inverse = (~selected).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=8,
    )
    filled_hole_pixels = 0
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if not touches_border and area <= max_hole_pixels:
            filled[labels == component_index] = True
            filled_hole_pixels += area
    return filled, filled_hole_pixels, max_hole_pixels


def release_selection_models():
    import gc

    with SELECTION_MODEL_LOCK:
        SELECTION_MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def structural_context_selection_infill(
    image: Image.Image,
    mask: Image.Image,
    *,
    context_max_dimension: int = 192,
    max_hole_area_ratio: float = 0.001,
    mask_feather_sigma_px: float = 2.5,
    context_gaussian_sigma_px: float = 3.0,
    core_erosion_px: int = 4,
) -> tuple[Image.Image, Image.Image, dict]:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    height, width = selected.shape
    pixel_count = max(1, width * height)
    filled, filled_hole_pixels, max_hole_pixels = _fill_small_selection_holes(
        selected,
        max_hole_area_ratio=max_hole_area_ratio,
    )

    maximum_dimension = max(width, height)
    if maximum_dimension > context_max_dimension:
        resize_scale = context_max_dimension / float(maximum_dimension)
        context_width = max(1, int(round(width * resize_scale)))
        context_height = max(1, int(round(height * resize_scale)))
        low_resolution = cv2.resize(
            source,
            (context_width, context_height),
            interpolation=cv2.INTER_AREA,
        )
        context = cv2.resize(
            low_resolution,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        context_width = width
        context_height = height
        context = source.copy()
    context = cv2.GaussianBlur(
        context,
        (0, 0),
        sigmaX=context_gaussian_sigma_px,
        sigmaY=context_gaussian_sigma_px,
        borderType=cv2.BORDER_REFLECT101,
    )

    soft_mask = cv2.GaussianBlur(
        filled.astype(np.float32),
        (0, 0),
        sigmaX=mask_feather_sigma_px,
        sigmaY=mask_feather_sigma_px,
        borderType=cv2.BORDER_REFLECT101,
    )
    alpha = np.clip(soft_mask, 0.0, 1.0)[..., None]
    composed = np.rint(
        source.astype(np.float32) * alpha
        + context.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    core_kernel_size = max(1, core_erosion_px * 2 + 1)
    core = cv2.erode(
        filled.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_kernel_size, core_kernel_size)),
        iterations=1,
    ) > 0
    composed[filled] = source[filled]

    filled_mask = Image.fromarray(filled.astype(np.uint8) * 255, mode="L")
    metadata = {
        "mode": "structural-context",
        "enabled": True,
        "context_max_dimension": int(context_max_dimension),
        "context_dimensions": {"width": int(context_width), "height": int(context_height)},
        "context_gaussian_sigma_px": float(context_gaussian_sigma_px),
        "mask_feather_sigma_px": float(mask_feather_sigma_px),
        "core_erosion_px": int(core_erosion_px),
        "max_hole_pixels": int(max_hole_pixels),
        "original_mask_pixels": int(np.count_nonzero(selected)),
        "final_mask_pixels": int(np.count_nonzero(filled)),
        "filled_hole_pixels": int(filled_hole_pixels),
        "context_pixels": int(pixel_count - np.count_nonzero(filled)),
        "selected_core_pixels": int(np.count_nonzero(core)),
        "selected_core_exact": bool(np.array_equal(composed[core], source[core])),
        "selected_pixels_exact": bool(np.array_equal(composed[filled], source[filled])),
    }
    return Image.fromarray(composed, mode="RGB"), filled_mask, metadata


def clean_context_selection_infill(
    image: Image.Image,
    mask: Image.Image,
    *,
    context_max_dimension: int = 192,
    max_hole_area_ratio: float = 0.001,
    canvas_top_rgb: tuple[int, int, int] = (238, 240, 242),
    canvas_bottom_rgb: tuple[int, int, int] = (214, 218, 222),
) -> tuple[Image.Image, Image.Image, dict]:
    """Fill removed pixels without sampling them from the source photograph."""
    from skimage.restoration import inpaint_biharmonic

    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    height, width = selected.shape
    pixel_count = max(1, width * height)
    filled, filled_hole_pixels, max_hole_pixels = _fill_small_selection_holes(
        selected,
        max_hole_area_ratio=max_hole_area_ratio,
    )

    maximum_dimension = max(width, height)
    resize_scale = min(1.0, context_max_dimension / float(maximum_dimension))
    context_width = max(8, int(round(width * resize_scale)))
    context_height = max(8, int(round(height * resize_scale)))
    selected_weight = cv2.resize(
        selected.astype(np.float32),
        (context_width, context_height),
        interpolation=cv2.INTER_AREA,
    )
    weighted_source = cv2.resize(
        source.astype(np.float32) * selected[..., None],
        (context_width, context_height),
        interpolation=cv2.INTER_AREA,
    )
    selected_values = weighted_source / np.maximum(selected_weight[..., None], 1e-6)
    known_selected = selected_weight >= 0.20

    top = np.asarray(canvas_top_rgb, dtype=np.float32) / 255.0
    bottom = np.asarray(canvas_bottom_rgb, dtype=np.float32) / 255.0
    row_weight = np.linspace(0.0, 1.0, context_height, dtype=np.float32)[:, None, None]
    canvas = top[None, None, :] * (1.0 - row_weight) + bottom[None, None, :] * row_weight
    canvas = np.broadcast_to(canvas, (context_height, context_width, 3)).copy()
    canvas[known_selected] = np.clip(selected_values[known_selected] / 255.0, 0.0, 1.0)

    pad = 2
    padded = np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    unknown = np.zeros(padded.shape[:2], dtype=bool)
    unknown[pad:-pad, pad:-pad] = ~known_selected
    if np.any(unknown):
        generated_small = inpaint_biharmonic(
            padded,
            unknown,
            channel_axis=-1,
        )[pad:-pad, pad:-pad]
    else:
        generated_small = canvas
    generated = cv2.resize(
        np.clip(generated_small * 255.0, 0.0, 255.0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    composed = generated.copy()
    composed[selected] = source[selected]
    core = cv2.erode(
        selected.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    filled_mask = Image.fromarray(filled.astype(np.uint8) * 255, mode="L")
    metadata = {
        "mode": "clean-context",
        "enabled": True,
        "method": "source-free-biharmonic-canvas",
        "source_free": True,
        "excluded_source_pixels_used": False,
        "context_max_dimension": int(context_max_dimension),
        "context_dimensions": {"width": int(context_width), "height": int(context_height)},
        "canvas_top_rgb": [int(value) for value in canvas_top_rgb],
        "canvas_bottom_rgb": [int(value) for value in canvas_bottom_rgb],
        "max_hole_pixels": int(max_hole_pixels),
        "original_mask_pixels": int(np.count_nonzero(selected)),
        "final_mask_pixels": int(np.count_nonzero(filled)),
        "filled_hole_pixels": int(filled_hole_pixels),
        "generated_hole_pixels": int(filled_hole_pixels),
        "filled_hole_source_pixels_preserved": 0,
        "context_pixels": int(pixel_count - np.count_nonzero(filled)),
        "selected_core_pixels": int(np.count_nonzero(core)),
        "selected_core_exact": bool(np.array_equal(composed[core], source[core])),
        "selected_pixels_exact": bool(np.array_equal(composed[selected], source[selected])),
    }
    return Image.fromarray(composed, mode="RGB"), filled_mask, metadata


def generative_context_selection_infill(
    image: Image.Image,
    mask: Image.Image,
    *,
    max_hole_area_ratio: float = 0.001,
) -> tuple[Image.Image, Image.Image, dict]:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    selected = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    filled, filled_hole_pixels, max_hole_pixels = _fill_small_selection_holes(
        selected,
        max_hole_area_ratio=max_hole_area_ratio,
    )
    provider = "flux2-klein-inpaint"
    release_selection_models()
    release_depth_pipelines()
    try:
        generated, generation = complete_selection_context_with_modern_inpaint(
            image,
            Image.fromarray(selected.astype(np.uint8) * 255, mode="L"),
            provider=provider,
        )
    finally:
        release_inpaint_pipelines(provider=provider)
    composed = np.asarray(generated.convert("RGB"), dtype=np.uint8).copy()
    composed[selected] = source[selected]
    filled_mask = Image.fromarray(filled.astype(np.uint8) * 255, mode="L")
    metadata = {
        "mode": "generative-context",
        "enabled": True,
        "method": generation["method"],
        "source_free": bool(generation["source_free_removed_context"]),
        "excluded_source_pixels_used": bool(
            generation["removed_source_pixels_conditioned"]
        ),
        "provider": generation["provider"],
        "model": generation["model"],
        "generation": generation,
        "max_hole_pixels": int(max_hole_pixels),
        "original_mask_pixels": int(np.count_nonzero(selected)),
        "final_mask_pixels": int(np.count_nonzero(filled)),
        "filled_hole_pixels": int(filled_hole_pixels),
        "generated_hole_pixels": int(filled_hole_pixels),
        "filled_hole_source_pixels_preserved": 0,
        "context_pixels": int(selected.size - np.count_nonzero(filled)),
        "selected_pixels_exact": bool(np.array_equal(composed[selected], source[selected])),
    }
    return Image.fromarray(composed, mode="RGB"), filled_mask, metadata


def compose_selected_image(
    image: Image.Image,
    mask: Image.Image,
    *,
    background_mode: str,
    selection_infill_mode: str,
) -> tuple[Image.Image, Image.Image, dict]:
    normalized_mode = str(selection_infill_mode or "none").strip().lower()
    if normalized_mode not in SELECTION_INFILL_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported selection infill mode '{selection_infill_mode}'. "
                f"Choose one of: {', '.join(sorted(SELECTION_INFILL_MODES))}"
            ),
        )
    if normalized_mode == "structural-context":
        return structural_context_selection_infill(image, mask)
    if normalized_mode == "clean-context":
        return clean_context_selection_infill(image, mask)
    if normalized_mode == "generative-context":
        return generative_context_selection_infill(image, mask)

    selected = selected_image_from_mask(image, mask, background_mode)
    unchanged_mask = mask.convert("L")
    mask_pixels = int(np.count_nonzero(np.asarray(unchanged_mask) > 0))
    return selected, unchanged_mask, {
        "mode": "none",
        "enabled": False,
        "original_mask_pixels": mask_pixels,
        "final_mask_pixels": mask_pixels,
        "filled_hole_pixels": 0,
    }


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


def crop_selection_artifacts(
    source_path: Path,
    selected_path: Path,
    mask_path: Path,
    output_dir: Path,
    *,
    padding_ratio: float = 0.06,
    minimum_padding_px: int = 8,
) -> dict:
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        mask_values = np.asarray(mask) > 0
    rows, columns = np.where(mask_values)
    if rows.size < 4 or columns.size < 4:
        raise HTTPException(status_code=409, detail="Selection mask is too small to crop")

    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB")
    with Image.open(selected_path) as selected_image:
        selected = selected_image.convert("RGB")
    if source.size != selected.size or source.size != mask.size:
        raise HTTPException(status_code=409, detail="Selection crop artifacts do not align")

    width, height = source.size
    raw_left = int(columns.min())
    raw_top = int(rows.min())
    raw_right = int(columns.max()) + 1
    raw_bottom = int(rows.max()) + 1
    object_span = max(raw_right - raw_left, raw_bottom - raw_top)
    padding = max(int(minimum_padding_px), int(round(object_span * float(padding_ratio))))
    left = max(0, raw_left - padding)
    top = max(0, raw_top - padding)
    right = min(width, raw_right + padding)
    bottom = min(height, raw_bottom + padding)
    crop_box = (left, top, right, bottom)

    output_dir.mkdir(parents=True, exist_ok=True)
    cropped_source_path = output_dir / "selection_source_crop.png"
    cropped_selected_path = output_dir / "selection_preview_crop.png"
    cropped_mask_path = output_dir / "selection_mask_crop.png"
    source.crop(crop_box).save(cropped_source_path)
    selected.crop(crop_box).save(cropped_selected_path)
    mask.crop(crop_box).save(cropped_mask_path)

    return {
        "source_path": cropped_source_path,
        "selected_path": cropped_selected_path,
        "mask_path": cropped_mask_path,
        "source_size": [int(width), int(height)],
        "raw_bbox_xyxy": [raw_left, raw_top, raw_right, raw_bottom],
        "crop_bbox_xyxy": [left, top, right, bottom],
        "crop_size": [right - left, bottom - top],
        "padding_px": int(padding),
        "mask_coverage": float(np.mean(mask_values[top:bottom, left:right])),
    }


def get_runtime_info() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
            "implementation_provenance": runtime_source_provenance(),
        }
    except Exception as exc:
        return {
            "torch": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "unknown",
            "error": str(exc),
            "implementation_provenance": runtime_source_provenance(),
        }


@app.post("/selection/precompute")
async def precompute_selection_model(
    file: UploadFile = File(...),
    model_id: str = Form("detr-resnet-50-panoptic"),
    device: str = Form("auto"),
):
    job_id = uuid4().hex
    started = time.perf_counter()
    try:
        with Image.open(file.file) as uploaded_image:
            image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc

    if model_id.startswith("sam2") or model_id == "grounding-dino-sam2":
        try:
            selected_device = device
            if selected_device == "auto":
                import torch

                selected_device = "cuda" if torch.cuda.is_available() else "cpu"
            _processor, _model, resolved_model_id = load_sam2_selection_model(model_id, selected_device)
            return {
                "job_id": job_id,
                "precompute_id": None,
                "model_id": resolved_model_id,
                "model_status": "sam2-model-ready",
                "precompute_supported": False,
                "message": "SAM2 model is loaded; point masks still run per hover until image embeddings are cached.",
                "image_size": {"width": image.width, "height": image.height},
                "timings": {"precompute_seconds": round(time.perf_counter() - started, 3)},
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=compact_selection_error(exc, "SAM2")) from exc

    try:
        segmentation, segment_labels, resolved_model_id = compute_panoptic_segmentation(image, device=device)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=compact_selection_error(exc, "Panoptic")) from exc

    precompute_id = cache_panoptic_precompute(segmentation, segment_labels, resolved_model_id, image.size)
    segment_ids = sorted(int(segment_id) for segment_id in np.unique(segmentation) if int(segment_id) >= 0)
    labels = sorted({segment_labels.get(segment_id, f"segment-{segment_id}") for segment_id in segment_ids})
    return {
        "job_id": job_id,
        "precompute_id": precompute_id,
        "model_id": resolved_model_id,
        "model_status": "panoptic-precomputed",
        "precompute_supported": True,
        "image_size": {"width": image.width, "height": image.height},
        "segment_count": len(segment_ids),
        "selection_labels": labels,
        "timings": {"precompute_seconds": round(time.perf_counter() - started, 3)},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/selection/precomputed_mask")
async def preview_precomputed_selection_mask(
    precompute_id: str = Form(...),
    points_json: str = Form("[]"),
):
    points = parse_selection_points(points_json)
    if not points:
        raise HTTPException(status_code=400, detail="Hover or click an object to preview a mask")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cached = get_selection_precompute(precompute_id)

    mask, selection_labels, selected_segment_ids = panoptic_mask_from_segmentation(
        cached["segmentation"],
        cached["segment_labels"],
        points,
        cached["image_size"],
    )
    mask_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    width, height = cached["image_size"]
    metadata = {
        "job_id": job_id,
        "precompute_id": precompute_id,
        "points": points,
        "model_id": cached["model_id"],
        "model_status": "panoptic-precomputed-point",
        "selection_labels": selection_labels,
        "selected_segment_ids": selected_segment_ids,
        "image_size": {"width": width, "height": height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, width * height)),
        "timings": {"selection_seconds": round(time.perf_counter() - started, 3)},
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    artifacts = save_selection_mask_artifacts(job_dir, mask, metadata)
    return {**metadata, **artifacts}


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

    mask, model_id, model_error, model_status, selection_labels = selection_mask_for_points(
        image,
        points,
        model_id=model_id,
        device=device,
        mask_max_dimension=mask_max_dimension,
    )

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
        "selection_labels": selection_labels,
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

    mask, model_id, model_error, model_status, selection_labels = selection_mask_for_points(
        image,
        points,
        model_id=model_id,
        device=device,
        mask_max_dimension=mask_max_dimension,
    )

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
        "selection_labels": selection_labels,
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
    selection_infill_mode: str = Form("none"),
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
    selected, mask, selection_infill = compose_selected_image(
        image,
        mask,
        background_mode=background_mode,
        selection_infill_mode=selection_infill_mode,
    )
    overlay = selection_overlay(image, mask)

    source_path = job_dir / "source.png"
    selected_path = job_dir / "selected_image.png"
    face_detection_path = job_dir / "selection_face_detection.png"
    mask_path = job_dir / "selection_mask.png"
    overlay_path = job_dir / "selection_overlay.png"
    tint_path = job_dir / "selection_tint.png"
    metadata_path = job_dir / "selection.json"
    image.save(source_path)
    selected.save(selected_path)
    face_detection_source = "selected-image"
    if selection_infill["mode"] in {
        "structural-context",
        "clean-context",
        "generative-context",
    }:
        selected_image_from_mask(image, mask, background_mode="neutral").save(
            face_detection_path
        )
        face_detection_source = "neutral-selection-cutout"
    mask.save(mask_path)
    overlay.save(overlay_path)
    selection_tint(mask).save(tint_path)

    mask_pixels = int(np.count_nonzero(np.asarray(mask) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": file.filename,
        "source_fingerprint": selection_source_fingerprint(image),
        "mask_count": len(mask_paths),
        "model_status": "composed-clicked-masks",
        "background_mode": background_mode,
        "selection_infill_mode": selection_infill["mode"],
        "selection_infill": selection_infill,
        "face_detection_source": face_detection_source,
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
    selection_job_id: str | None = Form(None),
    selection_mode: str = Form("context"),
    selection_subject_lock: bool = Form(False),
    depth_provider: str = Form(DEFAULT_DEPTH_PROVIDER),
    depth_model: str | None = Form(None),
    device: str = Form("auto"),
    target_dimension: int = Form(300),
    z_scale: float = Form(10),
    max_xy_size: float | None = Form(None),
    printer_profile: str | None = Form(None),
    printer_max_x_mm: float | None = Form(None),
    printer_max_y_mm: float | None = Form(None),
    printer_max_z_mm: float | None = Form(None),
    printer_clearance_mm: float | None = Form(None),
    nozzle_diameter_mm: float = Form(DEFAULT_NOZZLE_DIAMETER_MM),
    minimum_feature_mm: float | None = Form(None),
    print_scale_percent: float | None = Form(None),
    relief_polarity: str = Form("raised-print"),
    mesh_resolution_multiplier: float | None = Form(None),
    invert: bool = Form(False),
    sigma: float = Form(0.6),
    relief_gamma: float = Form(0.75),
    detail_boost: float = Form(0.8),
    background_detail_boost: float = Form(2.4),
    background_photo_detail_mm: float = Form(0.60, ge=0.0, le=0.60),
    selection_background_depth_ratio: float = Form(0.65),
    selection_background_feather_mm: float = Form(1.5),
    selection_background_smoothing_mm: float = Form(0.6),
    trim_top_background: bool = Form(True),
    printable_feature_depth_mm: float = Form(0.4),
    feature_bridge_depth_mm: float = Form(0.8),
    detail_radius: float = Form(2.0),
    detail_edge_threshold: float = Form(0.12),
    max_relief_slope: float = Form(DEFAULT_MAX_RELIEF_SLOPE),
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
    face_refinement_mode: str = Form("auto"),
    face_detail_strength: float = Form(DEFAULT_FACE_DETAIL_STRENGTH),
    face_feather_ratio: float = Form(DEFAULT_FACE_FEATHER_RATIO),
    face_max_correction_ratio: float = Form(DEFAULT_FACE_MAX_CORRECTION_RATIO),
):
    logger.info(f"Received file: {file.filename}")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.perf_counter()
    timings: dict[str, float] = {}

    def record_timing(name: str, started: float) -> None:
        timings[name] = round(time.perf_counter() - started, 3)

    temp_file_path = None
    normalized_input_path = None
    try:
        selection_job = (
            resolve_selection_compose_job(selection_job_id)
            if str(selection_job_id or "").strip()
            else None
        )
        resolved_selection_mode = str(selection_mode or "context").strip().lower()
        if resolved_selection_mode not in {"context", "isolate"}:
            raise HTTPException(
                status_code=400,
                detail="Selection mode must be either context or isolate",
            )
        if selection_job is None and resolved_selection_mode != "context":
            raise HTTPException(
                status_code=400,
                detail="Selection isolate mode requires a composed selection job",
            )
        if selection_job and completion_mode not in ("", "none"):
            raise HTTPException(
                status_code=400,
                detail="Image completion and object-selection context cannot be combined in one relief run",
            )
        if selection_job is None:
            upload_suffix = Path(file.filename or "").suffix or ".jpg"
            with NamedTemporaryFile(delete=False, suffix=upload_suffix, dir=job_dir) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
            normalized_input_path = job_dir / "input_oriented.png"
            try:
                with Image.open(temp_file_path) as uploaded_image:
                    ImageOps.exif_transpose(uploaded_image).convert("RGB").save(normalized_input_path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc
        selection_crop = None
        if selection_job and resolved_selection_mode == "isolate":
            selection_crop = crop_selection_artifacts(
                selection_job["source_path"],
                selection_job["selected_path"],
                selection_job["mask_path"],
                job_dir,
            )
            image_input_path = str(selection_crop["selected_path"])
        else:
            image_input_path = (
                str(selection_job["selected_path"])
                if selection_job
                else str(normalized_input_path)
            )

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
            image_input_path,
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
        # The edited artifact drives global and local depth. Selection jobs may
        # provide a detector-only cutout so background infill cannot suppress a face.
        depth_inference_source = image_for_depth
        face_refinement_source = image_for_depth
        face_detection_source = None
        if selection_job is not None:
            face_detection_source = selection_job["face_detection_path"]
            if selection_crop is not None:
                face_detection_crop_path = job_dir / "selection_face_detection_crop.png"
                with Image.open(face_detection_source) as detection_image:
                    detection_image.crop(
                        tuple(selection_crop["crop_bbox_xyxy"])
                    ).save(face_detection_crop_path)
                face_detection_source = face_detection_crop_path

        stage_started = time.perf_counter()
        depth_data_path = process_image_get_depth_data(
            depth_inference_source,
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
        stage_started = time.perf_counter()

        def infer_face_depth(crop_path: Path, face_output_dir: Path):
            return process_image_get_depth_data(
                str(crop_path),
                output_dir=str(face_output_dir),
                provider=depth_provider,
                model_name=effective_depth_model,
                device=device,
            )

        selection_region_mask_path = (
            selection_crop["mask_path"]
            if selection_crop
            else (selection_job["mask_path"] if selection_job is not None else None)
        )
        depth_data_path, face_refinement = refine_depth_for_faces(
            face_refinement_source,
            depth_data_path,
            job_dir,
            infer_depth=infer_face_depth,
            mode=face_refinement_mode,
            detail_strength=face_detail_strength,
            feather_ratio=face_feather_ratio,
            max_correction_ratio=face_max_correction_ratio,
            detection_roi_mask=selection_region_mask_path,
            detection_image_path=face_detection_source,
        )
        record_timing("face_refinement_seconds", stage_started)
        depth_metadata["face_refinement"] = face_refinement
        selection_depth_context = {"enabled": False, "reason": "not_requested"}
        effective_selection_background_depth_ratio = selection_background_depth_ratio
        if selection_job is not None and resolved_selection_mode == "isolate":
            stage_started = time.perf_counter()
            with Image.open(selection_region_mask_path) as selection_mask_image:
                selection_mask = np.asarray(selection_mask_image.convert("L")) > 0
            isolated_depth = np.load(depth_data_path).astype(np.float32)
            if selection_mask.shape != isolated_depth.shape:
                selection_mask = np.asarray(
                    Image.fromarray(selection_mask.astype(np.uint8) * 255, mode="L").resize(
                        (isolated_depth.shape[1], isolated_depth.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                ) > 0
            isolated_depth = np.where(selection_mask, isolated_depth, np.nan).astype(
                np.float32,
                copy=False,
            )
            isolated_depth_path = job_dir / "output_depth_data_selected_isolate.npy"
            np.save(isolated_depth_path, isolated_depth)
            depth_data_path = str(isolated_depth_path)
            effective_selection_background_depth_ratio = 0.0
            selection_depth_context = {
                "enabled": True,
                "method": "crop_first_isolated_selection_v1",
                "selection_job_id": selection_job["job_id"],
                "selection_mask": output_relative_path(selection_region_mask_path),
                "depth_file": isolated_depth_path.name,
                "depth_source": "selection_edited_image",
                "depth_source_file": output_relative_path(depth_inference_source),
                "background_depth_ratio": 0.0,
                "mask_pixels": int(np.count_nonzero(selection_mask)),
                "mask_coverage_ratio": float(np.mean(selection_mask)),
                "crop": {
                    key: value
                    for key, value in selection_crop.items()
                    if key not in {"source_path", "selected_path", "mask_path"}
                },
            }
            record_timing("selection_depth_context_seconds", stage_started)
        elif selection_job is not None:
            stage_started = time.perf_counter()
            with Image.open(selection_region_mask_path) as selection_mask_image:
                selection_mask = np.asarray(selection_mask_image.convert("L")) > 0
            if selection_subject_lock:
                selection_depth_context = {
                    "enabled": True,
                    "method": "full_scene_subject_locked_background_v1",
                    "selection_job_id": selection_job["job_id"],
                    "source_filename": selection_job["metadata"].get("source_filename"),
                    "source_fingerprint": selection_job["metadata"].get("source_fingerprint"),
                    "selection_mask": output_relative_path(selection_region_mask_path),
                    "depth_file": Path(depth_data_path).name,
                    "depth_source": "selection_edited_image",
                    "depth_source_file": output_relative_path(depth_inference_source),
                    "background_depth_ratio": float(selection_background_depth_ratio),
                    "mask_pixels": int(np.count_nonzero(selection_mask)),
                    "mask_coverage_ratio": float(np.mean(selection_mask)),
                    "subject_surface_locked": True,
                }
            else:
                context_depth = np.load(depth_data_path).astype(np.float32)
                context_sample_pitch_mm = (
                    float(max_xy_size) / max(max(context_depth.shape) - 1, 1)
                    if max_xy_size is not None and float(max_xy_size) > 0
                    else 1.0
                )
                try:
                    context_depth, selection_depth_context = compose_selection_depth_with_context(
                        context_depth,
                        selection_mask,
                        value_transform=depth_metadata.get("relief_value_transform", "linear"),
                        relief_height_mm=z_scale,
                        sample_pitch_mm=context_sample_pitch_mm,
                        max_slope_mm_per_mm=max_relief_slope,
                        background_depth_ratio=selection_background_depth_ratio,
                        background_feather_mm=selection_background_feather_mm,
                        background_smoothing_mm=selection_background_smoothing_mm,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                selected_context_depth_path = job_dir / "output_depth_data_selected_context.npy"
                np.save(selected_context_depth_path, context_depth)
                depth_data_path = str(selected_context_depth_path)

                preview_values = context_depth.astype(np.float32, copy=True)
                if depth_metadata.get("relief_value_transform") == "inverse-depth":
                    positive = np.isfinite(preview_values) & (preview_values > 0)
                    preview_values[positive] = 1.0 / preview_values[positive]
                    preview_values[np.isfinite(preview_values) & ~positive] = np.nan
                finite_preview = preview_values[np.isfinite(preview_values)]
                preview_low, preview_high = np.percentile(finite_preview, [1.0, 99.0])
                preview_span = max(float(preview_high - preview_low), 1e-8)
                preview = np.clip((preview_values - preview_low) / preview_span, 0.0, 1.0)
                preview = np.where(np.isfinite(preview), preview, 0.0)
                Image.fromarray((preview * 255.0).astype(np.uint8), mode="L").save(
                    job_dir / "output_depth_selected_context_preview.png"
                )
                selection_depth_context.update(
                    {
                        "selection_job_id": selection_job["job_id"],
                        "source_filename": selection_job["metadata"].get("source_filename"),
                        "source_fingerprint": selection_job["metadata"].get("source_fingerprint"),
                        "selection_mask": output_relative_path(selection_region_mask_path),
                        "depth_file": selected_context_depth_path.name,
                        "depth_source": "selection_edited_image",
                        "depth_source_file": output_relative_path(depth_inference_source),
                        "preview_file": "output_depth_selected_context_preview.png",
                    }
                )
            record_timing("selection_depth_context_seconds", stage_started)
        depth_metadata["selection_depth_context"] = selection_depth_context
        if depth_metadata_path.exists():
            with open(depth_metadata_path, "w", encoding="utf-8") as depth_metadata_file:
                json.dump(depth_metadata, depth_metadata_file, indent=2)
        effective_invert = relief_invert_for_model(effective_depth_model, relief_polarity, invert)
        relief_value_transform = depth_metadata.get("relief_value_transform")
        if not relief_value_transform:
            relief_value_transform = relief_value_transform_for_model(effective_depth_model)
        requested_target_dimension = target_dimension
        effective_target_dimension = resolve_relief_target_dimension(
            target_dimension,
            max_xy_size=max_xy_size,
            printer_max_x_mm=printer_max_x_mm,
            printer_max_y_mm=printer_max_y_mm,
            printer_clearance_mm=printer_clearance_mm,
            mesh_resolution_multiplier=mesh_resolution_multiplier,
        )
        effective_sample_pitch_mm = relief_sample_pitch_mm(max_xy_size, effective_target_dimension)
        effective_minimum_feature_mm = resolve_minimum_feature_mm(
            nozzle_diameter_mm,
            minimum_feature_mm,
        )
        
        # Generate 3D model
        logger.info("Generating 3D model...")
        stl_path = job_dir / "output_model.stl"
        stage_started = time.perf_counter()
        effective_trim_top_background = bool(trim_top_background) and (
            bool(selection_subject_lock)
            or not bool(selection_depth_context.get("enabled", False))
        )
        relief_postprocess = depth_data_to_3d_model(
            depth_data_path,
            output_stl_path=str(stl_path),
            target_dimension=effective_target_dimension,
            z_scale=z_scale,
            max_xy_size=max_xy_size,
            invert=effective_invert,
            sigma=sigma,
            relief_gamma=relief_gamma,
            detail_boost=detail_boost,
            background_detail_boost=background_detail_boost,
            source_image=depth_inference_source,
            background_photo_detail_mm=background_photo_detail_mm,
            trim_top_background=effective_trim_top_background,
            feature_weight_mask=(
                job_dir / face_refinement["weight_file"]
                if face_refinement.get("applied") and face_refinement.get("weight_file")
                else None
            ),
            feature_exclusion_mask=(
                job_dir / face_refinement["occlusion_file"]
                if face_refinement.get("applied") and face_refinement.get("occlusion_file")
                else None
            ),
            printable_feature_depth_mm=printable_feature_depth_mm,
            feature_bridge_depth_mm=feature_bridge_depth_mm,
            detail_radius=detail_radius,
            detail_edge_threshold=detail_edge_threshold,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            base_border_px=base_border_px,
            value_transform=relief_value_transform,
            minimum_feature_mm=effective_minimum_feature_mm,
            max_relief_slope=max_relief_slope,
            face_region_mask=(
                job_dir / face_refinement["region_file"]
                if face_refinement.get("applied") and face_refinement.get("region_file")
                else None
            ),
            selection_region_mask=(
                None if resolved_selection_mode == "isolate" else selection_region_mask_path
            ),
            selection_background_depth_ratio=effective_selection_background_depth_ratio,
            selection_subject_lock=selection_subject_lock,
            surface_output_path=job_dir / "output_surface.npy",
            reference_surface_output_path=job_dir / "output_reference_surface.npy",
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
            "target_dimension": effective_target_dimension,
            "requested_target_dimension": requested_target_dimension,
            "relief_sample_pitch_mm": effective_sample_pitch_mm,
            "z_scale": z_scale,
            "max_xy_size": max_xy_size,
            "printer": {
                "profile": printer_profile,
                "max_x_mm": printer_max_x_mm,
                "max_y_mm": printer_max_y_mm,
                "max_z_mm": printer_max_z_mm,
                "clearance_mm": printer_clearance_mm,
                "nozzle_diameter_mm": nozzle_diameter_mm,
                "minimum_feature_mm": effective_minimum_feature_mm,
                "print_scale_percent": print_scale_percent,
            },
            "relief_polarity": relief_polarity,
            "mesh_resolution_multiplier": mesh_resolution_multiplier,
            "minimum_feature_mm": effective_minimum_feature_mm,
            "requested_minimum_feature_mm": minimum_feature_mm,
            "max_relief_slope": max_relief_slope,
            "relief_postprocess": relief_postprocess,
            "size_aware_detail": {
                "min_detail_dimension": RELIEF_MIN_DETAIL_DIMENSION,
                "max_detail_dimension": RELIEF_MAX_DETAIL_DIMENSION,
                "applied": effective_target_dimension != requested_target_dimension,
            },
            "invert": effective_invert,
            "requested_invert": invert,
            "relief_value_transform": relief_value_transform,
            "sigma": sigma,
            "relief_gamma": relief_gamma,
            "detail_boost": detail_boost,
            "background_detail_boost": background_detail_boost,
            "background_photo_detail_mm": background_photo_detail_mm,
            "selection_background_depth_ratio": selection_background_depth_ratio,
            "effective_selection_background_depth_ratio": effective_selection_background_depth_ratio,
            "selection_mode": resolved_selection_mode,
            "selection_subject_lock": bool(selection_subject_lock),
            "selection_crop": (
                {
                    key: value
                    for key, value in selection_crop.items()
                    if key not in {"source_path", "selected_path", "mask_path"}
                }
                if selection_crop
                else None
            ),
            "selection_background_feather_mm": selection_background_feather_mm,
            "selection_background_smoothing_mm": selection_background_smoothing_mm,
            "trim_top_background": trim_top_background,
            "effective_trim_top_background": effective_trim_top_background,
            "printable_feature_depth_mm": printable_feature_depth_mm,
            "feature_bridge_depth_mm": feature_bridge_depth_mm,
            "detail_radius": detail_radius,
            "detail_edge_threshold": detail_edge_threshold,
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
            "face_refinement_mode": face_refinement_mode,
            "face_detail_strength": face_detail_strength,
            "face_feather_ratio": face_feather_ratio,
            "face_max_correction_ratio": face_max_correction_ratio,
            "face_refinement": face_refinement,
            "selection_depth_context": selection_depth_context,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if normalized_input_path and normalized_input_path.exists():
            normalized_input_path.unlink()


@app.post("/process_scene_diorama")
async def process_scene_diorama(
    file: UploadFile = File(...),
    mask_paths_json: str = Form("[]"),
    selection_labels_json: str = Form("[]"),
    depth_provider: str = Form(DEFAULT_DEPTH_PROVIDER),
    depth_model: str = Form(DEFAULT_DEPTH_MODEL),
    device: str = Form("auto"),
    max_size_mm: float = Form(180.0),
    scene_depth_mm: float = Form(64.0),
    base_thickness_mm: float = Form(2.4),
    facade_detail_mm: float = Form(0.8),
    subject_depth_mm: float = Form(12.0),
    depth_compression: float = Form(0.65),
    nozzle_diameter_mm: float = Form(DEFAULT_NOZZLE_DIAMETER_MM),
    minimum_feature_mm: float | None = Form(None),
    max_samples: int = Form(240),
):
    try:
        mask_references = json.loads(mask_paths_json or "[]")
        label_groups = json.loads(selection_labels_json or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid scene selection JSON: {exc}") from exc
    if not isinstance(mask_references, list) or not all(isinstance(value, str) for value in mask_references):
        raise HTTPException(status_code=400, detail="Scene mask paths must be a JSON list of strings")
    if not isinstance(label_groups, list):
        raise HTTPException(status_code=400, detail="Scene selection labels must be a JSON list")
    if label_groups and len(label_groups) != len(mask_references):
        raise HTTPException(status_code=400, detail="Scene masks and label groups must have the same length")
    if not label_groups:
        label_groups = [[] for _ in mask_references]

    selections = []
    for index, (mask_reference, raw_labels) in enumerate(zip(mask_references, label_groups)):
        if isinstance(raw_labels, str):
            labels = [raw_labels]
        elif isinstance(raw_labels, list) and all(isinstance(label, str) for label in raw_labels):
            labels = raw_labels
        else:
            raise HTTPException(status_code=400, detail=f"Scene label group {index} must contain strings")
        mask_path = resolve_output_file(
            normalize_output_reference(mask_reference),
            (".png", ".webp"),
        )
        selections.append({"mask_path": mask_path, "labels": labels})

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
        stage_started = time.perf_counter()
        depth_data_path = process_image_get_depth_data(
            temp_file_path,
            output_dir=str(job_dir),
            provider=depth_provider,
            model_name=depth_model,
            device=device,
        )
        record_timing("depth_seconds", stage_started)

        depth_metadata_path = job_dir / "output_depth_metadata.json"
        depth_metadata = {}
        if depth_metadata_path.exists():
            with open(depth_metadata_path, encoding="utf-8") as depth_metadata_file:
                depth_metadata = json.load(depth_metadata_file)
        effective_depth_model = depth_metadata.get("effective_model") or depth_model
        effective_minimum_feature_mm = resolve_minimum_feature_mm(
            nozzle_diameter_mm,
            minimum_feature_mm,
        )

        stl_path = job_dir / "output_scene.stl"
        glb_path = job_dir / "output_scene.glb"
        preview_path = job_dir / "output_scene_preview.png"
        stage_started = time.perf_counter()
        scene_reconstruction = build_scene_diorama(
            temp_file_path,
            depth_data_path,
            stl_path,
            glb_path,
            preview_path,
            selections=selections,
            far_is_high=depth_model_far_is_high(effective_depth_model),
            max_size_mm=max_size_mm,
            scene_depth_mm=scene_depth_mm,
            base_thickness_mm=base_thickness_mm,
            facade_detail_mm=facade_detail_mm,
            subject_depth_mm=subject_depth_mm,
            depth_compression=depth_compression,
            minimum_feature_mm=effective_minimum_feature_mm,
            max_samples=max_samples,
        )
        record_timing("scene_mesh_seconds", stage_started)

        stage_started = time.perf_counter()
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "single-photo-scene-diorama",
                "artifact_contract": "output_scene.glb + output_scene.stl + diagnostics.json",
            }
        )
        record_timing("diagnostics_seconds", stage_started)
        timings["total_seconds"] = round(time.perf_counter() - request_started, 3)

        diagnostics_path = job_dir / "diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as diagnostics_file:
            json.dump(diagnostics, diagnostics_file, indent=2, allow_nan=False)

        metadata = {
            "job_id": job_id,
            "source_filename": file.filename,
            "depth_provider": depth_provider,
            "requested_depth_model": depth_model,
            "depth_model": effective_depth_model,
            "depth_metadata": depth_metadata,
            "device": device,
            "selection_count": len(selections),
            "selection_labels": [selection["labels"] for selection in selections],
            "minimum_feature_mm": effective_minimum_feature_mm,
            "scene_reconstruction": scene_reconstruction,
            "runtime": get_runtime_info(),
            "timings": timings,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(job_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, allow_nan=False)

        depth_relative_path = output_relative_path(depth_data_path)
        stl_relative_path = output_relative_path(stl_path)
        glb_relative_path = output_relative_path(glb_path)
        preview_relative_path = output_relative_path(preview_path)
        diagnostics_relative_path = output_relative_path(diagnostics_path)
        return {
            **metadata,
            "depth_data": depth_relative_path,
            "depth_data_url": f"/depth_data/{depth_relative_path}",
            "stl_model": stl_relative_path,
            "stl_url": f"/stl_model/{stl_relative_path}",
            "scene_model": glb_relative_path,
            "scene_url": f"/scene_model/{glb_relative_path}",
            "preview": preview_relative_path,
            "preview_url": f"/depth_data/{preview_relative_path}",
            "diagnostics": diagnostics_relative_path,
            "diagnostics_url": f"/diagnostics/{diagnostics_relative_path}",
            "stl_diagnostics": diagnostics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scene diorama generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_file_path):
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


@app.get("/scene_model/{file_path:path}")
async def get_scene_model(file_path: str):
    resolved_path = resolve_output_file(file_path, (".glb",))
    return FileResponse(resolved_path, media_type="model/gltf-binary")


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
