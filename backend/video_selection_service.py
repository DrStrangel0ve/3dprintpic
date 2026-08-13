import os
import importlib.util
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

try:
    from .benchmark.run_image_to_mesh_provider import (
        CLI_PROVIDERS,
        HUNYUAN3D_SHAPE_PROVIDER,
        MULTIVIEW_VISUAL_HULL_PROVIDER,
        PROVIDERS,
        SOURCE_MESH_BUNDLE_ORACLE_PROVIDER,
        TRIPOSR_API_PROVIDER,
        provider_dir_config_key,
        resolve_provider_dir,
        run_provider,
    )
    from .benchmark.direct_mesh import MESH_REPAIR_MODES
    from .model_profiles import (
        DEFAULT_MODEL_PROFILE_ID,
        model_profile_catalog,
        resolve_model_profile,
        route_settings,
    )
    from .stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
    from .video_pipeline import (
        FRAME_SELECTION_MODES,
        SUPPORTED_VIDEO_SUFFIXES,
        VIDEO_SEGMENTATION_PROVIDERS,
        VideoDecodeError,
        VideoPipelineError,
        VideoSegmentationError,
        prepare_turntable_video,
        video_model_preflight,
    )
    from .video_subject_relief import generate_video_subject_relief
except ImportError:  # pragma: no cover - supports running uvicorn from backend/
    if __package__:
        raise
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.benchmark.run_image_to_mesh_provider import (
        CLI_PROVIDERS,
        HUNYUAN3D_SHAPE_PROVIDER,
        MULTIVIEW_VISUAL_HULL_PROVIDER,
        PROVIDERS,
        SOURCE_MESH_BUNDLE_ORACLE_PROVIDER,
        TRIPOSR_API_PROVIDER,
        provider_dir_config_key,
        resolve_provider_dir,
        run_provider,
    )
    from backend.benchmark.direct_mesh import MESH_REPAIR_MODES
    from backend.model_profiles import (
        DEFAULT_MODEL_PROFILE_ID,
        model_profile_catalog,
        resolve_model_profile,
        route_settings,
    )
    from backend.stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
    from backend.video_pipeline import (
        FRAME_SELECTION_MODES,
        SUPPORTED_VIDEO_SUFFIXES,
        VIDEO_SEGMENTATION_PROVIDERS,
        VideoDecodeError,
        VideoPipelineError,
        VideoSegmentationError,
        prepare_turntable_video,
        video_model_preflight,
    )
    from backend.video_subject_relief import generate_video_subject_relief


load_dotenv()

app = FastAPI(title="3D Print Pic Video and Selection Planner")
OUTPUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR", "./output/video-selection-runs")).resolve()
IMAGE_TO_MESH_RUN_LOCK = Lock()
SINGLE_IMAGE_PROVIDERS = tuple(
    provider
    for provider in PROVIDERS
    if provider not in {SOURCE_MESH_BUNDLE_ORACLE_PROVIDER, MULTIVIEW_VISUAL_HULL_PROVIDER}
)
MULTIVIEW_PROVIDERS = (MULTIVIEW_VISUAL_HULL_PROVIDER,)
RUNNER_MODES = ("image-to-mesh", "multiview-to-mesh", "video-to-mesh", "video-to-relief")
try:
    VIDEO_MAX_UPLOAD_BYTES = max(1, int(os.getenv("VIDEO_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))))
except (TypeError, ValueError):
    VIDEO_MAX_UPLOAD_BYTES = 500 * 1024 * 1024
STL_HARD_CHECKS = (
    "stl_exists",
    "stl_is_watertight",
    "stl_is_volume",
    "stl_is_manifold",
    "stl_winding_consistent",
    "stl_positive_volume",
    "stl_single_component",
)

DEFAULT_LOCAL_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]
DEFAULT_LOCAL_ORIGIN_REGEX = (
    r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$"
)
CORS_ORIGINS = list(
    dict.fromkeys(
        origin.strip()
        for origin in [
            *DEFAULT_LOCAL_ORIGINS,
            *os.getenv("CORS_ORIGINS", "").split(","),
            *os.getenv("VIDEO_CORS_ORIGINS", "").split(","),
        ]
        if origin.strip()
    )
)
CORS_ORIGIN_REGEX = (
    os.getenv("VIDEO_CORS_ORIGIN_REGEX", "").strip()
    or os.getenv("CORS_ORIGIN_REGEX", "").strip()
    or DEFAULT_LOCAL_ORIGIN_REGEX
)


def model_profile_for_request(profile_id: str | None) -> dict:
    try:
        return resolve_model_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc.args[0])) from exc


def profile_setting(value, settings: dict, key: str):
    return settings[key] if value is None else value


def resolve_output_file(file_path: str, allowed_suffixes: tuple[str, ...]) -> Path:
    requested = Path(file_path)
    if requested.is_absolute() or ".." in requested.parts:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    resolved = (OUTPUT_DIR / requested).resolve()
    try:
        resolved.relative_to(OUTPUT_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Artifact path escapes output directory") from exc
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    if allowed_suffixes and resolved.suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported artifact type: {resolved.suffix}")
    return resolved


def output_relative_path(path: Path | str) -> str:
    return Path(path).resolve().relative_to(OUTPUT_DIR).as_posix()


def provider_for_model(model_id: str | None) -> str:
    provider = str(model_id or DEFAULTS["image_to_mesh"]).strip()
    aliases = {
        "hunyuan3d": "hunyuan3d-shape",
        "tripo-sr": "triposr",
        "triposr-local": "triposr",
        "tripo-sr-api": "triposr-api",
    }
    provider = aliases.get(provider, provider)
    if provider not in SINGLE_IMAGE_PROVIDERS:
        expected = ", ".join(SINGLE_IMAGE_PROVIDERS)
        raise HTTPException(status_code=400, detail=f"Unsupported image_to_mesh provider '{provider}'. Valid: {expected}")
    return provider


def provider_for_multiview_model(model_id: str | None) -> str:
    provider = str(model_id or MULTIVIEW_VISUAL_HULL_PROVIDER).strip()
    aliases = {
        "visual-hull": MULTIVIEW_VISUAL_HULL_PROVIDER,
        "visual_hull": MULTIVIEW_VISUAL_HULL_PROVIDER,
    }
    provider = aliases.get(provider, provider)
    if provider not in MULTIVIEW_PROVIDERS:
        expected = ", ".join(MULTIVIEW_PROVIDERS)
        raise HTTPException(status_code=400, detail=f"Unsupported multiview_to_mesh provider '{provider}'. Valid: {expected}")
    return provider


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def provider_env_prefix(provider: str) -> str:
    return provider.upper().replace("-", "_")


def provider_dir_env_names(provider: str) -> list[str]:
    names = [f"{provider_env_prefix(provider)}_DIR"]
    config = CLI_PROVIDERS.get(provider_dir_config_key(provider), {})
    canonical = config.get("env")
    if canonical and canonical not in names:
        names.append(str(canonical))
    if provider == HUNYUAN3D_SHAPE_PROVIDER and "HUNYUAN3D_DIR" not in names:
        names.append("HUNYUAN3D_DIR")
    if "IMAGE_TO_MESH_PROVIDER_DIR" not in names:
        names.append("IMAGE_TO_MESH_PROVIDER_DIR")
    return names


def provider_python_env_names(provider: str) -> list[str]:
    return [f"{provider_env_prefix(provider)}_PYTHON", "IMAGE_TO_MESH_PROVIDER_PYTHON"]


def configured_env_value(names: list[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def provider_runtime_config(provider: str) -> dict[str, object]:
    env_prefix = provider_env_prefix(provider)
    provider_dir = configured_env_value(provider_dir_env_names(provider))
    provider_python = configured_env_value(provider_python_env_names(provider)) or sys.executable
    timeout = _env_int(f"{env_prefix}_TIMEOUT_SECONDS", _env_int("IMAGE_TO_MESH_TIMEOUT_SECONDS", 3600))
    return {
        "provider_dir": provider_dir,
        "provider_python": provider_python,
        "timeout": timeout,
    }


def multiview_provider_runtime_config(provider: str) -> dict[str, object]:
    env_prefix = provider_env_prefix(provider)
    timeout = _env_int(f"{env_prefix}_TIMEOUT_SECONDS", _env_int("MULTIVIEW_TO_MESH_TIMEOUT_SECONDS", 3600))
    return {
        "provider_dir": None,
        "provider_python": sys.executable,
        "timeout": timeout,
    }


def stl_gate_result(diagnostics: dict) -> tuple[bool, list[str]]:
    failed_checks = [key for key in STL_HARD_CHECKS if not bool(diagnostics.get(key))]
    return not failed_checks, failed_checks


def run_provider_job(args):
    return run_provider(args)


def release_competing_model_caches(provider: str, provider_device: str) -> dict:
    if provider != "triposg" or not str(provider_device).lower().startswith("cuda"):
        return {"attempted": False, "status": "not-required"}

    backend_url = os.getenv("RELIEF_BACKEND_URL", "http://127.0.0.1:8014").rstrip("/")
    endpoint = f"{backend_url}/runtime/release-models"
    request = Request(endpoint, data=b"", method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "attempted": True,
            "status": "released",
            "endpoint": endpoint,
            "details": payload,
        }
    except HTTPError as exc:
        error = f"HTTP {exc.code}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "attempted": True,
        "status": "unavailable",
        "endpoint": endpoint,
        "error": error,
    }


def run_serialized_image_to_mesh_job(args):
    with IMAGE_TO_MESH_RUN_LOCK:
        resource_release = release_competing_model_caches(
            args.provider,
            args.provider_device,
        )
        mesh_path, stl_path = run_provider_job(args)
        return mesh_path, stl_path, resource_release


def safe_upload_filename(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or fallback


def unique_child_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem or "upload"
    suffix = candidate.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise HTTPException(status_code=500, detail=f"Could not choose a unique upload name for {filename}")


def save_upload(upload: UploadFile, directory: Path, fallback: str) -> tuple[Path, str]:
    original_name = safe_upload_filename(upload.filename, fallback)
    target = unique_child_path(directory, original_name)
    with target.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)
    return target, original_name


def save_upload_limited(
    upload: UploadFile,
    directory: Path,
    fallback: str,
    *,
    max_bytes: int,
) -> tuple[Path, str, int]:
    original_name = safe_upload_filename(upload.filename, fallback)
    target = unique_child_path(directory, original_name)
    written = 0
    try:
        with target.open("wb") as output_file:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video upload exceeds the configured {max_bytes / (1024 * 1024):.0f} MB limit",
                    )
                output_file.write(chunk)
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    if written == 0:
        try:
            target.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    return target, original_name, written


def upload_lookup(items: list[tuple[str, Path]]) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for original_name, path in items:
        lookup[original_name] = path
        lookup[Path(original_name).name] = path
        lookup[path.name] = path
    return lookup


def parse_json_form(value: str | None, *, default):
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON form field: {exc}") from exc


async def load_bundle_payload(bundle_json: str | None, bundle_file: UploadFile | None) -> dict:
    payload = None
    if bundle_json and str(bundle_json).strip():
        payload = parse_json_form(bundle_json, default={})
    elif bundle_file is not None:
        content = await bundle_file.read()
        if content:
            try:
                payload = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid bundle_file JSON: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Bundle JSON must be an object.")
    return payload


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def normalize_bundle_path(value, lookup: dict[str, Path], job_dir: Path) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    candidates = [text, Path(text).name]
    for candidate in candidates:
        if candidate in lookup:
            return str(lookup[candidate])
    raw = Path(text)
    if raw.is_absolute():
        resolved = raw.resolve()
        if resolved.exists() and (is_within(resolved, job_dir) or is_within(resolved, OUTPUT_DIR)):
            return str(resolved)
        raise HTTPException(status_code=400, detail=f"Bundle path is not an uploaded or output artifact: {text}")
    for candidate in (job_dir / raw, OUTPUT_DIR / raw):
        resolved = candidate.resolve()
        if resolved.exists() and (is_within(resolved, job_dir) or is_within(resolved, OUTPUT_DIR)):
            return str(resolved)
    raise HTTPException(status_code=400, detail=f"Bundle path is not an uploaded or output artifact: {text}")


def normalize_multiview_bundle(
    bundle: dict,
    *,
    primary_image: Path,
    uploaded_views: list[Path],
    uploaded_masks: list[Path],
    uploaded_lookup: dict[str, Path],
    cameras,
    view_ids,
    job_dir: Path,
) -> dict:
    normalized = dict(bundle)
    normalized["primary_image"] = str(primary_image)
    normalized.setdefault("sample_id", "")

    raw_views = normalized.get("views") or []
    if not isinstance(raw_views, list):
        raise HTTPException(status_code=400, detail="Bundle field 'views' must be a list.")
    if not raw_views:
        source_views = uploaded_views or [primary_image]
        raw_views = [
            {
                "index": index,
                "sample_id": str(view_ids[index]) if isinstance(view_ids, list) and index < len(view_ids) else "",
                "image": str(path),
                "mask": str(uploaded_masks[index]) if index < len(uploaded_masks) else "",
                "camera": cameras[index] if isinstance(cameras, list) and index < len(cameras) else {},
            }
            for index, path in enumerate(source_views)
        ]

    views = []
    for index, view in enumerate(raw_views):
        if not isinstance(view, dict):
            raise HTTPException(status_code=400, detail="Each bundle view must be an object.")
        normalized_view = dict(view)
        if normalized_view.get("image"):
            normalized_view["image"] = normalize_bundle_path(normalized_view["image"], uploaded_lookup, job_dir)
        elif index < len(uploaded_views):
            normalized_view["image"] = str(uploaded_views[index])
        elif index == 0:
            normalized_view["image"] = str(primary_image)
        else:
            raise HTTPException(status_code=400, detail=f"Bundle view {index} has no image.")

        if normalized_view.get("mask"):
            normalized_view["mask"] = normalize_bundle_path(normalized_view["mask"], uploaded_lookup, job_dir)
        elif index < len(uploaded_masks):
            normalized_view["mask"] = str(uploaded_masks[index])
        else:
            normalized_view["mask"] = ""

        if not normalized_view.get("camera") and isinstance(cameras, list) and index < len(cameras):
            normalized_view["camera"] = cameras[index]
        if not normalized_view.get("sample_id") and isinstance(view_ids, list) and index < len(view_ids):
            normalized_view["sample_id"] = str(view_ids[index])
        normalized_view["index"] = normalized_view.get("index", index)
        views.append(normalized_view)

    normalized["views"] = views
    for key in ("masked_image", "full_image", "mask", "video_path"):
        if normalized.get(key):
            normalized[key] = normalize_bundle_path(normalized[key], uploaded_lookup, job_dir)
    return normalized

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_GROUPS = {
    "selection": [
        {
            "id": "sam3-person-aware",
            "label": "SAM 3 Person-aware",
            "model": "facebook/sam3",
            "role": "cached full-person masks with point-tracker fallback for non-person objects",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Pinned gated checkpoint; promoted after the exact shirt-omission regression reached 1.0 face/torso consistency.",
        },
        {
            "id": "turntable-grabcut",
            "label": "Turntable foreground",
            "model": "OpenCV GrabCut with temporal mask prior",
            "role": "automatic centered-object masks for controlled turntable videos",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Live deterministic video baseline with mask drift and coverage gates; best on a static, contrasting background.",
        },
        {
            "id": "sam2.1-hiera-tiny",
            "label": "SAM 2.1 Tiny",
            "model": "facebook/sam2.1-hiera-tiny",
            "role": "whole-object point-prompted masks for still photos",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Pinned still-image selector; favors complete click-anchored objects and removes disconnected mask islands.",
        },
        {
            "id": "detr-resnet-50-panoptic",
            "label": "DETR Panoptic",
            "model": "facebook/detr-resnet-50-panoptic",
            "role": "cached click-to-segment object masks for people, buildings, foliage, and scene parts",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Practical default while SAM2 checkpoints are not cached; selects the panoptic segment under the cursor.",
        },
        {
            "id": "sam2.1-hiera-large",
            "label": "SAM 2.1 Hiera Large",
            "model": "facebook/sam2.1-hiera-large",
            "role": "promptable object and video mask propagation",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Primary interactive selection model for selected-frame and object-selection routes.",
        },
        {
            "id": "sam2.1-hiera-base-plus",
            "label": "SAM 2.1 Hiera Base+",
            "model": "facebook/sam2.1-hiera-base-plus",
            "role": "faster promptable mask propagation",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Lower-latency option when the large checkpoint is too slow.",
        },
        {
            "id": "sam2.1-hiera-tiny-video",
            "label": "SAM 2.1 Tiny Video",
            "model": "facebook/sam2.1-hiera-tiny",
            "role": "point-prompted temporal mask propagation through sampled video frames",
            "local": True,
            "gpu_supported": True,
            "availability": "setup-required",
            "notes": "Live adapter; readiness requires Transformers Sam2VideoModel support and a compatible CUDA runtime.",
        },
        {
            "id": "sam2.1-hiera-base-plus-video",
            "label": "SAM 2.1 Base+ Video",
            "model": "facebook/sam2.1-hiera-base-plus",
            "role": "higher-quality point-prompted temporal mask propagation",
            "local": True,
            "gpu_supported": True,
            "availability": "setup-required",
            "notes": "Live adapter with higher memory use than Tiny; mask quality gates still decide whether reconstruction may run.",
        },
        {
            "id": "sam3.1-video",
            "label": "SAM 3.1 Video",
            "model": "facebookresearch/sam3 3.1",
            "role": "text/point-prompted modern video object segmentation and tracking",
            "local": True,
            "gpu_supported": True,
            "availability": "setup-required",
            "notes": "Preferred modern segmentation candidate; gated checkpoint access and the external SAM3 runtime are required.",
        },
        {
            "id": "grounding-dino-sam2",
            "label": "Grounding DINO + SAM 2.1",
            "model": "IDEA-Research/GroundingDINO + facebook/sam2.1",
            "role": "text-prompted object box plus mask",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Useful when the user names an object instead of clicking it.",
        },
        {
            "id": "panoptic-detr",
            "label": "DETR Panoptic",
            "model": "facebook/detr-resnet-50-panoptic",
            "role": "click nearest panoptic segment",
            "local": True,
            "gpu_supported": True,
            "availability": "configured",
            "notes": "Opt-in cached panoptic segment fallback for object selection when SAM2 is unavailable or too broad.",
        },
        {
            "id": "rmbg-2.0",
            "label": "RMBG 2.0",
            "model": "briaai/RMBG-2.0",
            "role": "automatic foreground matte",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast fallback for product-style foreground/background separation.",
        },
    ],
    "frame_selection": [
        {
            "id": "uniform-frame-sampler",
            "label": "Uniform frame sampler",
            "model": "opencv-videoio",
            "role": "deterministic every-n-frame sampling",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Baseline sampler for full-video routes.",
        },
        {
            "id": "scenedetect-adaptive",
            "label": "PySceneDetect adaptive",
            "model": "scenedetect-adaptive",
            "role": "shot and motion change sampling",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Keeps coverage without flooding reconstruction with near-duplicate frames.",
        },
        {
            "id": "sharpness-motion-selector",
            "label": "Sharpness/motion selector",
            "model": "opencv-laplacian-optical-flow",
            "role": "blur rejection and parallax coverage",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Live deterministic sampler that balances focus, exposure, temporal coverage, and appearance diversity.",
        },
    ],
    "camera_pose": [
        {
            "id": "turntable-orbit",
            "label": "Turntable orbit",
            "model": "frame-time orbit prior",
            "role": "deterministic cameras for a fixed camera and rotating object",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Live camera lane for controlled full-rotation clips; it does not claim general handheld camera recovery.",
        },
        {
            "id": "colmap-sift",
            "label": "COLMAP SIFT",
            "model": "COLMAP",
            "role": "camera matching and sparse reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Classic robust camera recovery for full frames.",
        },
        {
            "id": "hloc-lightglue",
            "label": "hloc + LightGlue",
            "model": "SuperPoint/DISK + LightGlue",
            "role": "learned feature matching for camera poses",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Modern learned matching lane for selected-frame camera/keypoint retention.",
        },
        {
            "id": "vggt-camera",
            "label": "VGGT camera head",
            "model": "VGGT",
            "role": "feed-forward camera/depth/point prediction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast learned geometry prior for short clips and sparse image sets.",
        },
        {
            "id": "vggt-omega-camera",
            "label": "VGGT-Omega camera",
            "model": "VGGT-Omega-1B-512",
            "role": "2026 feed-forward camera and depth prediction for selected video frames",
            "local": True,
            "gpu_supported": True,
            "availability": "setup-required",
            "notes": "Modern gated-checkpoint candidate; preflight reports source/checkpoint readiness separately from STL extraction.",
        },
    ],
    "video_reconstruction": [
        {
            "id": "multiview-visual-hull",
            "label": "Multiview visual hull",
            "model": "silhouette visual hull",
            "role": "deterministic multiview mesh from object silhouettes",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "First live STL-emitting multiview baseline; useful for selected-frame masks before learned reconstruction is attached.",
        },
        {
            "id": "colmap-openmvs",
            "label": "COLMAP + OpenMVS",
            "model": "COLMAP/OpenMVS",
            "role": "photogrammetry mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "STL-first baseline because it naturally produces a mesh.",
        },
        {
            "id": "gaussian-splatting-mesh",
            "label": "Gaussian Splatting + mesh",
            "model": "3D Gaussian Splatting + TSDF/Poisson extraction",
            "role": "reconstruct splats, extract mesh, repair STL",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Useful only if extracted STL quality beats photogrammetry/depth baselines.",
        },
        {
            "id": "vggt-fusion",
            "label": "VGGT fusion",
            "model": "VGGT",
            "role": "multi-frame depth/point fusion",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Candidate learned reconstruction path for short videos.",
        },
        {
            "id": "vggt-omega",
            "label": "VGGT-Omega fusion",
            "model": "VGGT-Omega-1B-512",
            "role": "2026 feed-forward cameras, depth, and points before mesh extraction",
            "local": True,
            "gpu_supported": True,
            "availability": "setup-required",
            "notes": "The model runtime can be preflighted, but this lane stays non-runnable until point/depth fusion emits a gated STL.",
        },
        {
            "id": "dust3r-mast3r",
            "label": "DUSt3R/MASt3R",
            "model": "DUSt3R or MASt3R",
            "role": "dense correspondence and 3D point prediction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Modern pairwise/multiview matching candidate before mesh extraction.",
        },
        {
            "id": "nerfstudio-mesh",
            "label": "Nerfstudio mesh",
            "model": "Nerfstudio",
            "role": "NeRF training, mesh extraction, STL repair",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Kept as an optional backend; visual quality alone is not enough for promotion.",
        },
    ],
    "image_to_mesh": [
        {
            "id": "triposg",
            "label": "TripoSG",
            "model": "VAST-AI/TripoSG",
            "role": "single image to mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "provider-setup",
            "notes": "Verified profile default; its paired 10-object STL run passed every printability gate.",
        },
        {
            "id": "pixal3d",
            "label": "Pixal3D",
            "model": "TencentARC/Pixal3D",
            "role": "single image to detailed geometry mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "provider-setup",
            "notes": "Provisional candidate: printable on one measured object but not promoted over TripoSG.",
        },
        {
            "id": "trellis2",
            "label": "TRELLIS.2 4B",
            "model": "microsoft/TRELLIS.2-4B",
            "role": "single image to open-topology geometry mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "provider-setup",
            "notes": "Pinned geometry adapter is available; paired STL-quality validation is still pending.",
        },
        {
            "id": "hunyuan3d-shape",
            "label": "Hunyuan3D Shape",
            "model": "Hunyuan3D Shape",
            "role": "single image to shape mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Direct image-to-3D candidate for full STL output.",
        },
        {
            "id": "triposr",
            "label": "TripoSR",
            "model": "TripoSR",
            "role": "single image sparse-view reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast local baseline for direct mesh smoke tests.",
        },
        {
            "id": "stable-fast-3d",
            "label": "Stable Fast 3D",
            "model": "SF3D",
            "role": "single image to textured mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Fast mesh candidate to compare against TripoSR/TripoSG-style outputs.",
        },
        {
            "id": "spar3d",
            "label": "SPAR3D",
            "model": "SPAR3D",
            "role": "single image sparse 3D reconstruction",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Candidate already represented in benchmark configs.",
        },
    ],
    "stl_postprocess": [
        {
            "id": "trimesh-repair",
            "label": "Trimesh repair",
            "model": "trimesh",
            "role": "mesh cleanup and STL export",
            "local": True,
            "gpu_supported": False,
            "availability": "configured",
            "notes": "Default lightweight repair and diagnostics path.",
        },
        {
            "id": "manifold3d",
            "label": "Manifold3D repair",
            "model": "manifold3d",
            "role": "watertight boolean/manifold conversion",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Promotion target for bad direct-mesh outputs.",
        },
        {
            "id": "pymeshlab-remesh",
            "label": "PyMeshLab remesh",
            "model": "pymeshlab",
            "role": "surface repair, decimation, simplification",
            "local": True,
            "gpu_supported": False,
            "availability": "adapter-planned",
            "notes": "Useful for printable complexity controls before STL export.",
        },
    ],
}

DEFAULTS = {
    "selection": "sam3-person-aware",
    "frame_selection": "uniform-frame-sampler",
    "camera_pose": "turntable-orbit",
    "video_reconstruction": "multiview-visual-hull",
    "image_to_mesh": "triposg",
    "stl_postprocess": "trimesh-repair",
}

STL_METRICS = [
    "watertightness",
    "manifoldness",
    "positive volume",
    "single component",
    "minimum printable thickness",
    "bbox aspect ratio",
    "surface Chamfer when ground truth exists",
]


def _model_index() -> dict[str, dict]:
    return {
        row["id"]: {**row, "group": group}
        for group, rows in MODEL_GROUPS.items()
        for row in rows
    }


def command_exists(executable: object) -> bool:
    text = str(executable or "").strip()
    if not text:
        return False
    path = Path(text)
    has_path_separator = any(separator and separator in text for separator in (os.sep, os.altsep))
    if path.is_absolute() or has_path_separator:
        return path.exists()
    return shutil.which(text) is not None


def safe_find_spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def image_to_mesh_provider_preflight(provider: str) -> dict:
    selected_provider = provider_for_model(provider)
    runtime = provider_runtime_config(selected_provider)
    model = _model_index().get(selected_provider, {})
    setup_errors: list[str] = []
    checks: dict[str, object] = {}
    provider_dir_envs = provider_dir_env_names(selected_provider)
    provider_python_envs = provider_python_env_names(selected_provider)

    provider_python_found = command_exists(runtime["provider_python"])
    checks["provider_python_found"] = provider_python_found
    if not provider_python_found:
        setup_errors.append("Provider Python executable is not available from server configuration.")

    if selected_provider in CLI_PROVIDERS or selected_provider == TRIPOSR_API_PROVIDER:
        try:
            provider_dir = resolve_provider_dir(selected_provider, runtime["provider_dir"])
            checks["provider_dir_resolved"] = True
        except FileNotFoundError:
            provider_dir = None
            checks["provider_dir_resolved"] = False
            setup_errors.append(f"Provider repo is missing. Configure one of: {', '.join(provider_dir_envs)}.")

        if provider_dir is not None:
            if selected_provider in CLI_PROVIDERS:
                entrypoint = (
                    Path("scripts/inference_triposg.py")
                    if CLI_PROVIDERS[selected_provider].get("runner") == "triposg-module"
                    else Path("run.py")
                )
            else:
                entrypoint = Path("tsr/system.py")
            entrypoint_found = (provider_dir / entrypoint).exists()
            checks["entrypoint"] = entrypoint.as_posix()
            checks["entrypoint_found"] = entrypoint_found
            if not entrypoint_found:
                setup_errors.append(f"Provider repo is missing expected entrypoint: {entrypoint.as_posix()}.")
    elif selected_provider == HUNYUAN3D_SHAPE_PROVIDER:
        provider_dir_value = runtime["provider_dir"] or os.getenv("HUNYUAN3D_DIR")
        source_available = False
        if provider_dir_value:
            provider_dir = Path(str(provider_dir_value))
            checks["provider_dir_resolved"] = provider_dir.exists()
            source_available = (provider_dir / "hy3dshape").exists()
        else:
            checks["provider_dir_resolved"] = False
        importable = safe_find_spec("hy3dshape")
        checks["hy3dshape_source_found"] = source_available
        checks["hy3dshape_importable"] = importable
        if not (source_available or importable):
            setup_errors.append(
                "Hunyuan3D Shape is not importable. Configure HUNYUAN3D_DIR or install hy3dshape in the server environment."
            )

    runnable = not setup_errors
    return {
        "id": selected_provider,
        "label": model.get("label", selected_provider),
        "status": "available" if runnable else "missing",
        "runnable": runnable,
        "setup_errors": setup_errors,
        "checks": checks,
        "env": {
            "provider_dir_env_names": provider_dir_envs,
            "provider_python_env_names": provider_python_envs,
            "timeout_seconds": runtime["timeout"],
            "provider_dir_configured": bool(runtime["provider_dir"]),
            "provider_python_configured": bool(configured_env_value(provider_python_envs)),
        },
    }


def _selected_model(model_id: str | None, group: str) -> dict:
    candidate_id = model_id or DEFAULTS[group]
    row = _model_index().get(candidate_id)
    if not row or row["group"] != group:
        valid = ", ".join(model["id"] for model in MODEL_GROUPS[group])
        raise HTTPException(status_code=400, detail=f"Unsupported {group} model '{candidate_id}'. Valid: {valid}")
    return row


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "video-selection-planner",
        "mode": "planner-plus-runner",
        "runner_modes": list(RUNNER_MODES),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/models")
async def models():
    profiles = model_profile_catalog()
    return {
        "service": "video-selection-planner",
        "mode": "planner-plus-runner",
        "defaults": DEFAULTS,
        "groups": MODEL_GROUPS,
        "model_profiles": profiles,
        "default_model_profile": profiles["default_profile"],
        "metrics": STL_METRICS,
        "runner_modes": list(RUNNER_MODES),
        "notes": "This companion service exposes the video, selection, camera, direct-mesh, multiview-mesh, and STL-repair model surface. The image-to-mesh and multiview-to-mesh runners can attach providers behind the same ids.",
    }


@app.get("/profiles")
async def profiles():
    return {
        "service": "video-selection-planner",
        **model_profile_catalog(),
    }


@app.get("/providers/image-to-mesh")
async def image_to_mesh_providers():
    return {
        "service": "video-selection-planner",
        "runner": "image-to-mesh",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "default_provider": DEFAULTS["image_to_mesh"],
        "default_model_profile": DEFAULT_MODEL_PROFILE_ID,
        "providers": [image_to_mesh_provider_preflight(provider) for provider in SINGLE_IMAGE_PROVIDERS],
    }


@app.get("/providers/multiview-to-mesh")
async def multiview_to_mesh_providers():
    return {
        "service": "video-selection-planner",
        "runner": "multiview-to-mesh",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "default_provider": MULTIVIEW_VISUAL_HULL_PROVIDER,
        "providers": [
            {
                "id": MULTIVIEW_VISUAL_HULL_PROVIDER,
                "label": "Multiview visual hull",
                "status": "available",
                "runnable": True,
                "setup_errors": [],
                "checks": {"builtin_provider": True, "requires_input_bundle": True},
                "env": {
                    "timeout_seconds": multiview_provider_runtime_config(MULTIVIEW_VISUAL_HULL_PROVIDER)["timeout"],
                    "provider_dir_configured": False,
                    "provider_python_configured": False,
                },
            }
        ],
    }


@app.get("/providers/video-to-mesh")
async def video_to_mesh_providers():
    return {
        "service": "video-selection-planner",
        "runner": "video-to-mesh",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "default_frame_selection": DEFAULTS["frame_selection"],
        "default_segmentation": "turntable-grabcut",
        "default_camera_pose": DEFAULTS["camera_pose"],
        "default_reconstruction": DEFAULTS["video_reconstruction"],
        "preflight": video_model_preflight(),
    }


@app.post("/plan")
async def plan(payload: dict):
    route = payload.get("route") or {}
    print_volume = payload.get("print_volume") or {}
    media_type = (payload.get("input") or {}).get("media_type", "photo")
    model_ids = payload.get("models") or {}
    profile = model_profile_for_request(payload.get("model_profile") or payload.get("profile_id"))
    profile_models = profile["models"]

    selection = _selected_model(model_ids.get("selection") or profile_models.get("selection"), "selection")
    stl_postprocess = _selected_model(
        model_ids.get("stl_postprocess") or profile_models.get("stl_postprocess"),
        "stl_postprocess",
    )

    stages = []
    if media_type == "video":
        frame_selection = _selected_model(model_ids.get("frame_selection"), "frame_selection")
        camera_pose = _selected_model(model_ids.get("camera_pose"), "camera_pose")
        video_reconstruction = _selected_model(
            model_ids.get("video_reconstruction") or profile_models.get("video_reconstruction"),
            "video_reconstruction",
        )
        stages = [
            {"id": "frame-selection", "model": frame_selection},
            {"id": "object-selection", "model": selection},
            {"id": "camera-pose", "model": camera_pose},
            {"id": "video-reconstruction", "model": video_reconstruction},
            {"id": "stl-postprocess", "model": stl_postprocess},
        ]
    else:
        image_to_mesh = _selected_model(
            model_ids.get("image_to_mesh") or profile_models.get("image_to_mesh"),
            "image_to_mesh",
        )
        stages = [
            {"id": "object-selection", "model": selection},
            {"id": "image-to-mesh", "model": image_to_mesh},
            {"id": "stl-postprocess", "model": stl_postprocess},
        ]

    return {
        "status": "planned",
        "run_id": uuid4().hex,
        "service": "video-selection-planner",
        "execution_mode": "planner-only",
        "model_profile": {
            "id": profile["id"],
            "label": profile["label"],
            "status": profile["status"],
        },
        "route": route,
        "print_volume": print_volume,
        "stages": stages,
        "metrics": STL_METRICS,
        "next_backend_contract": {
            "input": "uploaded media path plus selected frame/mask artifacts",
            "output": "watertight STL path plus diagnostics JSON",
            "promotion_gate": "all hard STL checks must pass before model promotion",
        },
    }


@app.post("/run/image-to-mesh")
async def run_image_to_mesh(
    file: UploadFile = File(...),
    profile_id: str | None = Form(DEFAULT_MODEL_PROFILE_ID),
    provider: str | None = Form(None),
    provider_device: str = Form("cuda"),
    model_name: str | None = Form(None),
    mesh_repair: str | None = Form(None),
    mesh_repair_preconditioner: str | None = Form(None),
    mesh_repair_voxel_resolution: int | None = Form(None),
    mesh_repair_voxel_fill_method: str | None = Form(None),
    mesh_repair_smoothing_iterations: int | None = Form(None),
    mesh_allow_convex_hull_fallback: bool | None = Form(None),
    mesh_target_max_dimension: float | None = Form(None),
    mesh_min_bbox_dimension: float | None = Form(None),
    mesh_max_bbox_aspect_ratio: float | None = Form(None),
    mesh_target_bbox_mode: str | None = Form(None),
    mesh_target_faces: int | None = Form(None),
    mesh_max_normalized_face_density_log1p: float | None = Form(None),
    low_vram: bool | None = Form(None),
    chunk_size: int | None = Form(None),
    mc_resolution: int | None = Form(None),
    texture_resolution: int | None = Form(None),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    octree_resolution: int | None = Form(None),
    num_chunks: int | None = Form(None),
    seed: int | None = Form(None),
    disable_progress: bool | None = Form(None),
):
    profile = model_profile_for_request(profile_id)
    settings = route_settings(profile, "photo-full-mesh")
    selected_provider = provider_for_model(provider or settings["provider"])
    mesh_repair = str(profile_setting(mesh_repair, settings, "mesh_repair"))
    triposg_profile_defaults = selected_provider == "triposg"
    mesh_repair_preconditioner = str(
        mesh_repair_preconditioner
        if mesh_repair_preconditioner is not None
        else settings["mesh_repair_preconditioner"]
        if triposg_profile_defaults
        else "legacy"
    )
    mesh_repair_voxel_resolution = int(
        mesh_repair_voxel_resolution
        if mesh_repair_voxel_resolution is not None
        else settings["mesh_repair_voxel_resolution"]
        if triposg_profile_defaults
        else 64
    )
    mesh_repair_voxel_fill_method = str(
        mesh_repair_voxel_fill_method
        if mesh_repair_voxel_fill_method is not None
        else settings["mesh_repair_voxel_fill_method"]
        if triposg_profile_defaults
        else "orthographic"
    )
    mesh_repair_smoothing_iterations = int(
        mesh_repair_smoothing_iterations
        if mesh_repair_smoothing_iterations is not None
        else settings["mesh_repair_smoothing_iterations"]
        if triposg_profile_defaults
        else 0
    )
    mesh_allow_convex_hull_fallback = bool(
        mesh_allow_convex_hull_fallback
        if mesh_allow_convex_hull_fallback is not None
        else settings["mesh_allow_convex_hull_fallback"]
        if triposg_profile_defaults
        else True
    )
    mesh_target_max_dimension = float(
        profile_setting(mesh_target_max_dimension, settings, "mesh_target_max_dimension")
    )
    mesh_min_bbox_dimension = float(
        profile_setting(mesh_min_bbox_dimension, settings, "mesh_min_bbox_dimension")
    )
    mesh_max_bbox_aspect_ratio = float(
        profile_setting(mesh_max_bbox_aspect_ratio, settings, "mesh_max_bbox_aspect_ratio")
    )
    mesh_target_bbox_mode = str(
        mesh_target_bbox_mode
        if mesh_target_bbox_mode is not None
        else settings["mesh_target_bbox_mode"]
        if triposg_profile_defaults
        else "exact"
    )
    mesh_target_faces = int(profile_setting(mesh_target_faces, settings, "mesh_target_faces"))
    mesh_max_normalized_face_density_log1p = float(
        profile_setting(
            mesh_max_normalized_face_density_log1p,
            settings,
            "mesh_max_normalized_face_density_log1p",
        )
    )
    low_vram = bool(profile_setting(low_vram, settings, "low_vram"))
    chunk_size = int(profile_setting(chunk_size, settings, "chunk_size"))
    mc_resolution = int(profile_setting(mc_resolution, settings, "mc_resolution"))
    texture_resolution = profile_setting(texture_resolution, settings, "texture_resolution")
    num_inference_steps = int(profile_setting(num_inference_steps, settings, "num_inference_steps"))
    guidance_scale = float(profile_setting(guidance_scale, settings, "guidance_scale"))
    octree_resolution = int(profile_setting(octree_resolution, settings, "octree_resolution"))
    num_chunks = int(profile_setting(num_chunks, settings, "num_chunks"))
    seed = profile_setting(seed, settings, "seed")
    disable_progress = bool(profile_setting(disable_progress, settings, "disable_progress"))
    if mesh_repair not in MESH_REPAIR_MODES:
        expected = ", ".join(MESH_REPAIR_MODES)
        raise HTTPException(status_code=400, detail=f"Unsupported mesh_repair '{mesh_repair}'. Valid: {expected}")
    runtime_config = provider_runtime_config(selected_provider)

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.perf_counter()
    upload_suffix = Path(file.filename or "").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=upload_suffix, dir=job_dir) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        input_path = Path(temp_file.name)

    output_mesh = job_dir / "output_mesh.glb"
    raw_output_mesh = job_dir / "output_mesh_raw.glb"
    output_stl = job_dir / "output_model.stl"
    provider_args = SimpleNamespace(
        provider=selected_provider,
        input_image=input_path,
        input_bundle=None,
        output_mesh=output_mesh,
        output_stl=output_stl,
        raw_output_mesh=raw_output_mesh,
        provider_dir=runtime_config["provider_dir"],
        provider_output_dir=job_dir / f"{selected_provider}_raw",
        provider_mesh_cache_dir=None,
        python=runtime_config["provider_python"],
        timeout=runtime_config["timeout"],
        low_vram=bool(low_vram),
        provider_device=provider_device,
        chunk_size=int(chunk_size),
        mc_resolution=int(mc_resolution),
        texture_resolution=texture_resolution,
        remesh_option=None,
        mesh_repair=mesh_repair,
        mesh_repair_preconditioner=mesh_repair_preconditioner,
        mesh_repair_voxel_resolution=mesh_repair_voxel_resolution,
        mesh_repair_voxel_fill_method=mesh_repair_voxel_fill_method,
        mesh_repair_smoothing_iterations=mesh_repair_smoothing_iterations,
        mesh_allow_convex_hull_fallback=mesh_allow_convex_hull_fallback,
        mesh_target_max_dimension=float(mesh_target_max_dimension or 0.0),
        mesh_min_bbox_dimension=float(mesh_min_bbox_dimension or 0.0),
        mesh_max_bbox_aspect_ratio=float(mesh_max_bbox_aspect_ratio or 0.0),
        mesh_target_bbox_extents=None,
        mesh_target_bbox_mode=mesh_target_bbox_mode,
        mesh_target_faces=int(mesh_target_faces or 0),
        mesh_max_normalized_face_density_log1p=mesh_max_normalized_face_density_log1p,
        provider_arg=[],
        model_name=model_name,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        seed=seed,
        mc_algo=None,
        disable_progress=bool(disable_progress),
        pixal3d_resolution=None,
        pixal3d_fov=None,
        pixal3d_model_path=None,
        pixal3d_model_revision=None,
        pixal3d_moge_revision=None,
        pixal3d_dinov3_revision=None,
        pixal3d_rembg_model=None,
        pixal3d_rembg_revision=None,
        triposg_model_revision=(
            settings.get("triposg_model_revision") if selected_provider == "triposg" else None
        ),
        triposg_rembg_revision=(
            settings.get("triposg_rembg_revision") if selected_provider == "triposg" else None
        ),
        trellis2_model_path=None,
        trellis2_model_revision=None,
        trellis2_resolution=None,
        prefetch_only=False,
    )

    started_at = datetime.utcnow().isoformat() + "Z"
    try:
        stage_started = time.perf_counter()
        mesh_path, stl_path, resource_release = await run_in_threadpool(
            run_serialized_image_to_mesh_job,
            provider_args,
        )
        provider_seconds = round(time.perf_counter() - stage_started, 3)
        stl_path = Path(stl_path or output_stl)
        if not stl_path.exists():
            raise FileNotFoundError(f"Provider did not write an STL at {stl_path}")
        stage_started = time.perf_counter()
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "image-to-mesh",
                "provider": selected_provider,
                "model_profile": profile["id"],
                "artifact_contract": "output_model.stl + diagnostics.json",
            }
        )
        stl_passes_hard_checks, stl_failed_checks = stl_gate_result(diagnostics)
        diagnostics["stl_passes_hard_checks"] = stl_passes_hard_checks
        diagnostics["stl_failed_checks"] = stl_failed_checks
        diagnostics_seconds = round(time.perf_counter() - stage_started, 3)
        provider_metrics = json_safe_stl_diagnostics(
            dict(getattr(provider_args, "_provider_metrics", {}) or {})
        )
        timings = {
            "provider_seconds": provider_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "total_seconds": round(time.perf_counter() - request_started, 3),
        }
        diagnostics_path = job_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
        provider_metrics_path = job_dir / "provider_metrics.json"
        provider_metrics_path.write_text(
            json.dumps(provider_metrics, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        provider_python_configured = bool(configured_env_value(provider_python_env_names(selected_provider)))
        metadata = {
            "job_id": job_id,
            "source_filename": file.filename,
            "service": "video-selection-planner",
            "runner": "image-to-mesh",
            "provider": selected_provider,
            "model_profile": profile["id"],
            "model_profile_label": profile["label"],
            "model_profile_status": profile["status"],
            "model_revisions": profile.get("model_revisions", {}),
            "provider_dir_configured": bool(runtime_config["provider_dir"]),
            "provider_python_configured": provider_python_configured,
            "provider_device": provider_device,
            "provider_timeout_seconds": runtime_config["timeout"],
            "resource_release": resource_release,
            "mesh_repair": mesh_repair,
            "mesh_repair_preconditioner": mesh_repair_preconditioner,
            "mesh_repair_voxel_resolution": mesh_repair_voxel_resolution,
            "mesh_repair_voxel_fill_method": mesh_repair_voxel_fill_method,
            "mesh_repair_smoothing_iterations": mesh_repair_smoothing_iterations,
            "mesh_allow_convex_hull_fallback": mesh_allow_convex_hull_fallback,
            "mesh_target_max_dimension": mesh_target_max_dimension,
            "mesh_min_bbox_dimension": mesh_min_bbox_dimension,
            "mesh_max_bbox_aspect_ratio": mesh_max_bbox_aspect_ratio,
            "mesh_target_bbox_mode": mesh_target_bbox_mode,
            "mesh_target_faces": mesh_target_faces,
            "mesh_max_normalized_face_density_log1p": mesh_max_normalized_face_density_log1p,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "provider_metrics": provider_metrics,
            "timings": timings,
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

        mesh_relative = output_relative_path(mesh_path)
        stl_relative = output_relative_path(stl_path)
        diagnostics_relative = output_relative_path(diagnostics_path)
        provider_metrics_relative = output_relative_path(provider_metrics_path)
        metadata_relative = output_relative_path(metadata_path)
        return {
            **metadata,
            "status": "printable" if stl_passes_hard_checks else "stl-emitted",
            "stl_passes_hard_checks": stl_passes_hard_checks,
            "stl_failed_checks": stl_failed_checks,
            "output_mesh": mesh_relative,
            "output_mesh_url": f"/artifacts/{mesh_relative}",
            "stl_model": stl_relative,
            "stl_url": f"/artifacts/{stl_relative}",
            "diagnostics": diagnostics_relative,
            "diagnostics_url": f"/artifacts/{diagnostics_relative}",
            "provider_metrics": provider_metrics,
            "provider_metrics_artifact": provider_metrics_relative,
            "provider_metrics_url": f"/artifacts/{provider_metrics_relative}",
            "metadata": metadata_relative,
            "metadata_url": f"/artifacts/{metadata_relative}",
            "stl_diagnostics": diagnostics,
            "timings": timings,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        try:
            input_path.unlink()
        except OSError:
            pass


@app.post("/run/multiview-to-mesh")
async def run_multiview_to_mesh(
    primary_file: UploadFile = File(...),
    bundle_json: str | None = Form(None),
    bundle_file: UploadFile | None = File(None),
    view_files: list[UploadFile] | None = File(None),
    mask_files: list[UploadFile] | None = File(None),
    profile_id: str | None = Form(DEFAULT_MODEL_PROFILE_ID),
    provider: str | None = Form(None),
    provider_device: str = Form("cuda"),
    mesh_repair: str | None = Form(None),
    mesh_target_max_dimension: float | None = Form(None),
    mesh_min_bbox_dimension: float | None = Form(None),
    mesh_max_bbox_aspect_ratio: float | None = Form(None),
    mesh_target_faces: int | None = Form(None),
    cameras_json: str | None = Form(None),
    view_ids_json: str | None = Form(None),
    visual_hull_resolution: int | None = Form(None),
    visual_hull_grid_extent: float | None = Form(None),
    visual_hull_ortho_scale: float | None = Form(None),
    visual_hull_mask_dilate: int | None = Form(None),
):
    profile = model_profile_for_request(profile_id)
    settings = route_settings(profile, "multiview-full-mesh")
    selected_provider = provider_for_multiview_model(provider or settings["provider"])
    mesh_repair = str(profile_setting(mesh_repair, settings, "mesh_repair"))
    mesh_target_max_dimension = float(
        profile_setting(mesh_target_max_dimension, settings, "mesh_target_max_dimension")
    )
    mesh_min_bbox_dimension = float(
        profile_setting(mesh_min_bbox_dimension, settings, "mesh_min_bbox_dimension")
    )
    mesh_max_bbox_aspect_ratio = float(
        profile_setting(mesh_max_bbox_aspect_ratio, settings, "mesh_max_bbox_aspect_ratio")
    )
    mesh_target_faces = int(profile_setting(mesh_target_faces, settings, "mesh_target_faces"))
    visual_hull_resolution = int(
        profile_setting(visual_hull_resolution, settings, "visual_hull_resolution")
    )
    visual_hull_grid_extent = float(
        profile_setting(visual_hull_grid_extent, settings, "visual_hull_grid_extent")
    )
    visual_hull_ortho_scale = float(
        profile_setting(visual_hull_ortho_scale, settings, "visual_hull_ortho_scale")
    )
    visual_hull_mask_dilate = int(
        profile_setting(visual_hull_mask_dilate, settings, "visual_hull_mask_dilate")
    )
    if mesh_repair not in MESH_REPAIR_MODES:
        expected = ", ".join(MESH_REPAIR_MODES)
        raise HTTPException(status_code=400, detail=f"Unsupported mesh_repair '{mesh_repair}'. Valid: {expected}")
    runtime_config = multiview_provider_runtime_config(selected_provider)

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.perf_counter()
    upload_records: list[tuple[str, Path]] = []

    primary_path, primary_name = save_upload(primary_file, job_dir / "inputs", "primary.png")
    upload_records.append((primary_name, primary_path))
    uploaded_views = []
    for index, upload in enumerate(view_files or []):
        path, name = save_upload(upload, job_dir / "inputs" / "views", f"view{index}.png")
        upload_records.append((name, path))
        uploaded_views.append(path)
    uploaded_masks = []
    for index, upload in enumerate(mask_files or []):
        path, name = save_upload(upload, job_dir / "inputs" / "masks", f"mask{index}.png")
        upload_records.append((name, path))
        uploaded_masks.append(path)

    bundle = await load_bundle_payload(bundle_json, bundle_file)
    cameras = parse_json_form(cameras_json, default=[])
    view_ids = parse_json_form(view_ids_json, default=[])
    if cameras and not isinstance(cameras, list):
        raise HTTPException(status_code=400, detail="cameras_json must be a list when provided.")
    if view_ids and not isinstance(view_ids, list):
        raise HTTPException(status_code=400, detail="view_ids_json must be a list when provided.")
    normalized_bundle = normalize_multiview_bundle(
        bundle,
        primary_image=primary_path,
        uploaded_views=uploaded_views,
        uploaded_masks=uploaded_masks,
        uploaded_lookup=upload_lookup(upload_records),
        cameras=cameras,
        view_ids=view_ids,
        job_dir=job_dir,
    )
    bundle_path = job_dir / "multiview_input.json"
    bundle_path.write_text(json.dumps(normalized_bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    output_mesh = job_dir / "output_mesh.ply"
    raw_output_mesh = job_dir / "output_mesh_raw.ply"
    output_stl = job_dir / "output_model.stl"
    provider_args = SimpleNamespace(
        provider=selected_provider,
        input_image=primary_path,
        input_bundle=bundle_path,
        output_mesh=output_mesh,
        output_stl=output_stl,
        raw_output_mesh=raw_output_mesh,
        provider_dir=runtime_config["provider_dir"],
        provider_output_dir=job_dir / f"{selected_provider}_raw",
        provider_mesh_cache_dir=None,
        python=runtime_config["provider_python"],
        timeout=runtime_config["timeout"],
        low_vram=False,
        provider_device=provider_device,
        chunk_size=8192,
        mc_resolution=256,
        texture_resolution=None,
        remesh_option=None,
        mesh_repair=mesh_repair,
        mesh_target_max_dimension=float(mesh_target_max_dimension or 0.0),
        mesh_min_bbox_dimension=float(mesh_min_bbox_dimension or 0.0),
        mesh_max_bbox_aspect_ratio=float(mesh_max_bbox_aspect_ratio or 0.0),
        mesh_target_bbox_extents=None,
        mesh_target_faces=int(mesh_target_faces or 0),
        mesh_max_normalized_face_density_log1p=0.0,
        provider_arg=[],
        model_name=None,
        num_inference_steps=1,
        guidance_scale=0.0,
        octree_resolution=256,
        num_chunks=8000,
        seed=None,
        mc_algo=None,
        disable_progress=True,
        prefetch_only=False,
        visual_hull_resolution=int(visual_hull_resolution),
        visual_hull_grid_extent=float(visual_hull_grid_extent),
        visual_hull_ortho_scale=float(visual_hull_ortho_scale),
        visual_hull_mask_dilate=int(visual_hull_mask_dilate),
    )

    started_at = datetime.utcnow().isoformat() + "Z"
    try:
        stage_started = time.perf_counter()
        mesh_path, stl_path = await run_in_threadpool(run_provider_job, provider_args)
        provider_seconds = round(time.perf_counter() - stage_started, 3)
        stl_path = Path(stl_path or output_stl)
        if not stl_path.exists():
            raise FileNotFoundError(f"Provider did not write an STL at {stl_path}")
        stage_started = time.perf_counter()
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "multiview-to-mesh",
                "provider": selected_provider,
                "model_profile": profile["id"],
                "artifact_contract": "output_model.stl + diagnostics.json",
                "view_count": len(normalized_bundle.get("views", [])),
            }
        )
        stl_passes_hard_checks, stl_failed_checks = stl_gate_result(diagnostics)
        diagnostics["stl_passes_hard_checks"] = stl_passes_hard_checks
        diagnostics["stl_failed_checks"] = stl_failed_checks
        diagnostics_seconds = round(time.perf_counter() - stage_started, 3)
        timings = {
            "provider_seconds": provider_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "total_seconds": round(time.perf_counter() - request_started, 3),
        }
        diagnostics_path = job_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
        metadata = {
            "job_id": job_id,
            "source_filename": primary_file.filename,
            "service": "video-selection-planner",
            "runner": "multiview-to-mesh",
            "provider": selected_provider,
            "model_profile": profile["id"],
            "model_profile_label": profile["label"],
            "model_profile_status": profile["status"],
            "provider_dir_configured": False,
            "provider_python_configured": False,
            "provider_device": provider_device,
            "provider_timeout_seconds": runtime_config["timeout"],
            "mesh_repair": mesh_repair,
            "mesh_target_max_dimension": mesh_target_max_dimension,
            "mesh_min_bbox_dimension": mesh_min_bbox_dimension,
            "mesh_max_bbox_aspect_ratio": mesh_max_bbox_aspect_ratio,
            "mesh_target_faces": mesh_target_faces,
            "visual_hull_resolution": visual_hull_resolution,
            "visual_hull_grid_extent": visual_hull_grid_extent,
            "visual_hull_ortho_scale": visual_hull_ortho_scale,
            "visual_hull_mask_dilate": visual_hull_mask_dilate,
            "view_count": len(normalized_bundle.get("views", [])),
            "timings": timings,
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

        mesh_relative = output_relative_path(mesh_path)
        stl_relative = output_relative_path(stl_path)
        bundle_relative = output_relative_path(bundle_path)
        diagnostics_relative = output_relative_path(diagnostics_path)
        metadata_relative = output_relative_path(metadata_path)
        return {
            **metadata,
            "status": "printable" if stl_passes_hard_checks else "stl-emitted",
            "stl_passes_hard_checks": stl_passes_hard_checks,
            "stl_failed_checks": stl_failed_checks,
            "input_bundle": bundle_relative,
            "input_bundle_url": f"/artifacts/{bundle_relative}",
            "output_mesh": mesh_relative,
            "output_mesh_url": f"/artifacts/{mesh_relative}",
            "stl_model": stl_relative,
            "stl_url": f"/artifacts/{stl_relative}",
            "diagnostics": diagnostics_relative,
            "diagnostics_url": f"/artifacts/{diagnostics_relative}",
            "metadata": metadata_relative,
            "metadata_url": f"/artifacts/{metadata_relative}",
            "stl_diagnostics": diagnostics,
            "timings": timings,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/run/video-to-relief")
async def run_video_to_relief(
    file: UploadFile = File(...),
    frame_selection: str = Form("sharpness-motion-selector"),
    segmentation_provider: str = Form("sam2.1-hiera-tiny-video"),
    segmentation_device: str = Form("auto"),
    selected_frame_count: int = Form(12),
    object_point_x: float = Form(0.5),
    object_point_y: float = Form(0.4),
    strict_segmentation: bool = Form(True),
    frame_max_side: int = Form(960),
    depth_model: str = Form("depth-anything/Depth-Anything-V2-Large-hf"),
    depth_device: str = Form("auto"),
    target_dimension: int = Form(360),
    max_xy_size_mm: float = Form(120.0),
    relief_height_mm: float = Form(8.0),
    minimum_feature_mm: float = Form(0.4),
):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        expected = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported video extension '{suffix}'. Valid: {expected}")
    if frame_selection not in FRAME_SELECTION_MODES:
        expected = ", ".join(sorted(FRAME_SELECTION_MODES))
        raise HTTPException(status_code=400, detail=f"Unsupported frame_selection '{frame_selection}'. Valid: {expected}")
    if segmentation_provider not in VIDEO_SEGMENTATION_PROVIDERS:
        expected = ", ".join(sorted(VIDEO_SEGMENTATION_PROVIDERS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported segmentation_provider '{segmentation_provider}'. Valid: {expected}",
        )
    if not 3 <= int(selected_frame_count) <= 64:
        raise HTTPException(status_code=400, detail="selected_frame_count must be between 3 and 64")
    if not 256 <= int(frame_max_side) <= 2048:
        raise HTTPException(status_code=400, detail="frame_max_side must be between 256 and 2048")
    if not 64 <= int(target_dimension) <= 2048:
        raise HTTPException(status_code=400, detail="target_dimension must be between 64 and 2048")
    if not 10.0 <= float(max_xy_size_mm) <= 1000.0:
        raise HTTPException(status_code=400, detail="max_xy_size_mm must be between 10 and 1000")
    if not 0.5 <= float(relief_height_mm) <= 100.0:
        raise HTTPException(status_code=400, detail="relief_height_mm must be between 0.5 and 100")
    if not 0.1 <= float(minimum_feature_mm) <= 10.0:
        raise HTTPException(status_code=400, detail="minimum_feature_mm must be between 0.1 and 10")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path, original_name, uploaded_bytes = save_upload_limited(
        file,
        job_dir / "inputs",
        f"video{suffix or '.mp4'}",
        max_bytes=VIDEO_MAX_UPLOAD_BYTES,
    )
    try:
        result = await run_in_threadpool(
            generate_video_subject_relief,
            video_path,
            job_dir,
            selected_frame_count=int(selected_frame_count),
            frame_selection=frame_selection,
            segmentation_provider=segmentation_provider,
            segmentation_device=segmentation_device,
            object_point=(float(object_point_x), float(object_point_y)),
            strict_segmentation=bool(strict_segmentation),
            frame_max_side=int(frame_max_side),
            depth_model=depth_model,
            depth_device=depth_device,
            target_dimension=int(target_dimension),
            max_xy_size_mm=float(max_xy_size_mm),
            relief_height_mm=float(relief_height_mm),
            minimum_feature_mm=float(minimum_feature_mm),
        )
    except (VideoDecodeError, VideoSegmentationError, VideoPipelineError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "job_id": job_id,
                "message": str(exc),
                "error_type": type(exc).__name__,
                "segmentation_report": getattr(exc, "report", None) or None,
            },
        ) from exc
    except (ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    def artifact(path: Path) -> tuple[str, str]:
        relative = output_relative_path(path)
        return relative, f"/artifacts/{relative}"

    stl_relative, stl_url = artifact(result.stl_path)
    diagnostics_relative, diagnostics_url = artifact(result.diagnostics_path)
    report_relative, report_url = artifact(result.report_path)
    preview_relative, preview_url = artifact(result.preview_path)
    frame_relative, frame_url = artifact(result.selected_frame_path)
    mask_relative, mask_url = artifact(result.selected_mask_path)
    diagnostics = result.report["stl_diagnostics"]
    return {
        "job_id": job_id,
        "runner": "video-to-relief",
        "status": result.report["status"],
        "source_filename": original_name,
        "uploaded_bytes": uploaded_bytes,
        "motion": result.report["motion"],
        "selection": result.report["selection"],
        "face_refinement": result.report["face_refinement"],
        "stl_model": stl_relative,
        "stl_url": stl_url,
        "diagnostics": diagnostics_relative,
        "diagnostics_url": diagnostics_url,
        "report": report_relative,
        "report_url": report_url,
        "preview": preview_relative,
        "preview_url": preview_url,
        "selected_frame": frame_relative,
        "selected_frame_url": frame_url,
        "selected_mask": mask_relative,
        "selected_mask_url": mask_url,
        "stl_diagnostics": diagnostics,
        "stl_passes_hard_checks": diagnostics["stl_passes_hard_checks"],
        "stl_failed_checks": diagnostics["stl_failed_checks"],
        "timings": result.report["timings"],
    }


@app.post("/run/video-to-mesh")
async def run_video_to_mesh(
    file: UploadFile = File(...),
    provider: str | None = Form(None),
    frame_selection: str = Form("uniform-frame-sampler"),
    segmentation_provider: str = Form("turntable-grabcut"),
    segmentation_device: str = Form("auto"),
    selected_frame_count: int = Form(12),
    object_point_x: float = Form(0.5),
    object_point_y: float = Form(0.5),
    strict_segmentation: bool = Form(True),
    rotation_degrees: float = Form(360.0),
    rotation_direction: str = Form("counter-clockwise"),
    start_azimuth_deg: float = Form(0.0),
    elevation_deg: float = Form(0.0),
    max_duration_seconds: float = Form(180.0),
    max_decode_frames: int = Form(12000),
    frame_max_side: int = Form(960),
    provider_device: str = Form("cpu"),
    mesh_repair: str = Form("printable"),
    mesh_target_max_dimension: float = Form(96.0),
    mesh_min_bbox_dimension: float = Form(12.0),
    mesh_max_bbox_aspect_ratio: float = Form(0.0),
    mesh_target_faces: int = Form(40000),
    visual_hull_resolution: int = Form(32),
    visual_hull_grid_extent: float = Form(1.9),
    visual_hull_ortho_scale: float = Form(2.0),
    visual_hull_mask_dilate: int = Form(1),
):
    selected_provider = provider_for_multiview_model(provider)
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        expected = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported video extension '{suffix}'. Valid: {expected}")
    if frame_selection not in FRAME_SELECTION_MODES:
        expected = ", ".join(sorted(FRAME_SELECTION_MODES))
        raise HTTPException(status_code=400, detail=f"Unsupported frame_selection '{frame_selection}'. Valid: {expected}")
    if segmentation_provider not in VIDEO_SEGMENTATION_PROVIDERS:
        expected = ", ".join(sorted(VIDEO_SEGMENTATION_PROVIDERS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported segmentation_provider '{segmentation_provider}'. Valid: {expected}",
        )
    if mesh_repair not in MESH_REPAIR_MODES:
        expected = ", ".join(MESH_REPAIR_MODES)
        raise HTTPException(status_code=400, detail=f"Unsupported mesh_repair '{mesh_repair}'. Valid: {expected}")
    if not 3 <= int(selected_frame_count) <= 64:
        raise HTTPException(status_code=400, detail="selected_frame_count must be between 3 and 64")
    if not 256 <= int(frame_max_side) <= 2048:
        raise HTTPException(status_code=400, detail="frame_max_side must be between 256 and 2048")
    if not 3 <= int(visual_hull_resolution) <= 256:
        raise HTTPException(status_code=400, detail="visual_hull_resolution must be between 3 and 256")
    if max_duration_seconds <= 0 or max_duration_seconds > 1800:
        raise HTTPException(status_code=400, detail="max_duration_seconds must be in (0, 1800]")
    if max_decode_frames < 3 or max_decode_frames > 100000:
        raise HTTPException(status_code=400, detail="max_decode_frames must be between 3 and 100000")

    runtime_config = multiview_provider_runtime_config(selected_provider)
    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    request_started = time.perf_counter()
    video_path, original_name, uploaded_bytes = save_upload_limited(
        file,
        job_dir / "inputs",
        f"video{suffix or '.mp4'}",
        max_bytes=VIDEO_MAX_UPLOAD_BYTES,
    )

    preparation_started = time.perf_counter()
    try:
        prepared = await run_in_threadpool(
            prepare_turntable_video,
            video_path,
            job_dir / "prepared",
            selected_frame_count=int(selected_frame_count),
            frame_selection=frame_selection,
            segmentation_provider=segmentation_provider,
            object_point=(float(object_point_x), float(object_point_y)),
            segmentation_device=segmentation_device,
            rotation_degrees=float(rotation_degrees),
            rotation_direction=rotation_direction,
            start_azimuth_deg=float(start_azimuth_deg),
            elevation_deg=float(elevation_deg),
            strict_segmentation=bool(strict_segmentation),
            max_duration_seconds=float(max_duration_seconds),
            max_decode_frames=int(max_decode_frames),
            frame_max_side=int(frame_max_side),
        )
    except (VideoDecodeError, VideoSegmentationError, VideoPipelineError) as exc:
        report_path = job_dir / "prepared" / "video_preparation.json"
        detail = {
            "job_id": job_id,
            "message": str(exc),
            "error_type": type(exc).__name__,
            "segmentation_report": getattr(exc, "report", None) or None,
        }
        if report_path.exists():
            report_relative = output_relative_path(report_path)
            detail["report_url"] = f"/artifacts/{report_relative}"
        status_code = 400 if isinstance(exc, VideoDecodeError) else 422
        if isinstance(exc, VideoSegmentationError) and (
            "dependencies are unavailable" in str(exc) or "requested CUDA" in str(exc)
        ):
            status_code = 503
        raise HTTPException(status_code=status_code, detail=detail) from exc
    preparation_seconds = round(time.perf_counter() - preparation_started, 3)

    output_mesh = job_dir / "output_mesh.ply"
    raw_output_mesh = job_dir / "output_mesh_raw.ply"
    output_stl = job_dir / "output_model.stl"
    provider_args = SimpleNamespace(
        provider=selected_provider,
        input_image=prepared.frame_paths[0],
        input_bundle=prepared.bundle_path,
        output_mesh=output_mesh,
        output_stl=output_stl,
        raw_output_mesh=raw_output_mesh,
        provider_dir=runtime_config["provider_dir"],
        provider_output_dir=job_dir / f"{selected_provider}_raw",
        python=runtime_config["provider_python"],
        timeout=runtime_config["timeout"],
        low_vram=False,
        provider_device=provider_device,
        chunk_size=8192,
        mc_resolution=256,
        texture_resolution=None,
        remesh_option=None,
        mesh_repair=mesh_repair,
        mesh_target_max_dimension=float(mesh_target_max_dimension or 0.0),
        mesh_min_bbox_dimension=float(mesh_min_bbox_dimension or 0.0),
        mesh_max_bbox_aspect_ratio=float(mesh_max_bbox_aspect_ratio or 0.0),
        mesh_target_bbox_extents=None,
        mesh_target_faces=int(mesh_target_faces or 0),
        provider_arg=[],
        model_name=None,
        num_inference_steps=1,
        guidance_scale=0.0,
        octree_resolution=256,
        num_chunks=8000,
        seed=None,
        mc_algo=None,
        disable_progress=True,
        prefetch_only=False,
        visual_hull_resolution=int(visual_hull_resolution),
        visual_hull_grid_extent=float(visual_hull_grid_extent),
        visual_hull_ortho_scale=float(visual_hull_ortho_scale),
        visual_hull_mask_dilate=int(visual_hull_mask_dilate),
    )

    started_at = datetime.utcnow().isoformat() + "Z"
    try:
        stage_started = time.perf_counter()
        mesh_path, stl_path = await run_in_threadpool(run_provider_job, provider_args)
        provider_seconds = round(time.perf_counter() - stage_started, 3)
        stl_path = Path(stl_path or output_stl)
        if not stl_path.exists():
            raise FileNotFoundError(f"Provider did not write an STL at {stl_path}")
        stage_started = time.perf_counter()
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(stl_path))
        mask_quality = prepared.report["segmentation"]["quality"]
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "video-to-mesh",
                "provider": selected_provider,
                "segmentation_provider": segmentation_provider,
                "artifact_contract": "output_model.stl + diagnostics.json + video_preparation.json",
                "view_count": len(prepared.frame_paths),
                "mask_quality_status": mask_quality["status"],
                "mask_quality_passes_hard_checks": mask_quality["passes_hard_checks"],
            }
        )
        stl_passes_hard_checks, stl_failed_checks = stl_gate_result(diagnostics)
        diagnostics["stl_passes_hard_checks"] = stl_passes_hard_checks
        diagnostics["stl_failed_checks"] = stl_failed_checks
        diagnostics_seconds = round(time.perf_counter() - stage_started, 3)
        timings = {
            "preparation_seconds": preparation_seconds,
            "provider_seconds": provider_seconds,
            "diagnostics_seconds": diagnostics_seconds,
            "total_seconds": round(time.perf_counter() - request_started, 3),
        }
        diagnostics_path = job_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding="utf-8")
        metadata = {
            "job_id": job_id,
            "source_filename": original_name,
            "source_bytes": uploaded_bytes,
            "source_content_type": file.content_type,
            "service": "video-selection-planner",
            "runner": "video-to-mesh",
            "provider": selected_provider,
            "frame_selection": frame_selection,
            "segmentation_provider": segmentation_provider,
            "segmentation_device": segmentation_device,
            "selected_frame_count": len(prepared.frame_paths),
            "strict_segmentation": strict_segmentation,
            "rotation_degrees": rotation_degrees,
            "rotation_direction": rotation_direction,
            "mesh_repair": mesh_repair,
            "mesh_target_max_dimension": mesh_target_max_dimension,
            "mesh_min_bbox_dimension": mesh_min_bbox_dimension,
            "mesh_max_bbox_aspect_ratio": mesh_max_bbox_aspect_ratio,
            "mesh_target_faces": mesh_target_faces,
            "visual_hull_resolution": visual_hull_resolution,
            "visual_hull_grid_extent": visual_hull_grid_extent,
            "visual_hull_ortho_scale": visual_hull_ortho_scale,
            "visual_hull_mask_dilate": visual_hull_mask_dilate,
            "timings": timings,
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

        mesh_relative = output_relative_path(mesh_path)
        stl_relative = output_relative_path(stl_path)
        bundle_relative = output_relative_path(prepared.bundle_path)
        report_relative = output_relative_path(prepared.report_path)
        diagnostics_relative = output_relative_path(diagnostics_path)
        metadata_relative = output_relative_path(metadata_path)
        frame_relatives = [output_relative_path(path) for path in prepared.frame_paths]
        mask_relatives = [output_relative_path(path) for path in prepared.mask_paths]
        return {
            **metadata,
            "status": "printable" if stl_passes_hard_checks else "stl-emitted",
            "stl_passes_hard_checks": stl_passes_hard_checks,
            "stl_failed_checks": stl_failed_checks,
            "input_bundle": bundle_relative,
            "input_bundle_url": f"/artifacts/{bundle_relative}",
            "video_preparation": report_relative,
            "video_preparation_url": f"/artifacts/{report_relative}",
            "selected_frames": frame_relatives,
            "selected_frame_urls": [f"/artifacts/{path}" for path in frame_relatives],
            "selected_masks": mask_relatives,
            "selected_mask_urls": [f"/artifacts/{path}" for path in mask_relatives],
            "mask_quality": mask_quality,
            "sampling": prepared.report["sampling"],
            "output_mesh": mesh_relative,
            "output_mesh_url": f"/artifacts/{mesh_relative}",
            "stl_model": stl_relative,
            "stl_url": f"/artifacts/{stl_relative}",
            "diagnostics": diagnostics_relative,
            "diagnostics_url": f"/artifacts/{diagnostics_relative}",
            "metadata": metadata_relative,
            "metadata_url": f"/artifacts/{metadata_relative}",
            "stl_diagnostics": diagnostics,
            "timings": timings,
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/artifacts/{file_path:path}")
async def get_artifact(file_path: str):
    resolved_path = resolve_output_file(file_path, (".stl", ".json", ".glb", ".gltf", ".obj", ".ply", ".png", ".jpg", ".jpeg", ".webp"))
    media_type = "application/json" if resolved_path.suffix.lower() == ".json" else None
    return FileResponse(resolved_path, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VIDEO_SELECTION_PORT", "8005")))
