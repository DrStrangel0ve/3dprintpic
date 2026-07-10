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
from types import SimpleNamespace
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
    from .stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics
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
    from backend.stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics


load_dotenv()

app = FastAPI(title="3D Print Pic Video and Selection Planner")
OUTPUT_DIR = Path(os.getenv("VIDEO_OUTPUT_DIR", "./output/video-selection-runs")).resolve()
SINGLE_IMAGE_PROVIDERS = tuple(
    provider
    for provider in PROVIDERS
    if provider not in {SOURCE_MESH_BUNDLE_ORACLE_PROVIDER, MULTIVIEW_VISUAL_HULL_PROVIDER}
)
MULTIVIEW_PROVIDERS = (MULTIVIEW_VISUAL_HULL_PROVIDER,)
RUNNER_MODES = ("image-to-mesh", "multiview-to-mesh")
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_GROUPS = {
    "selection": [
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
            "availability": "adapter-planned",
            "notes": "Ranks selected frames by focus, motion, and pose diversity.",
        },
    ],
    "camera_pose": [
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
            "model": "TripoSG-style direct mesh",
            "role": "single image to mesh",
            "local": True,
            "gpu_supported": True,
            "availability": "adapter-planned",
            "notes": "Preferred direct mesh lane when available.",
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
    "selection": "detr-resnet-50-panoptic",
    "frame_selection": "uniform-frame-sampler",
    "camera_pose": "hloc-lightglue",
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
    return {
        "service": "video-selection-planner",
        "mode": "planner-plus-runner",
        "defaults": DEFAULTS,
        "groups": MODEL_GROUPS,
        "metrics": STL_METRICS,
        "runner_modes": list(RUNNER_MODES),
        "notes": "This companion service exposes the video, selection, camera, direct-mesh, multiview-mesh, and STL-repair model surface. The image-to-mesh and multiview-to-mesh runners can attach providers behind the same ids.",
    }


@app.get("/providers/image-to-mesh")
async def image_to_mesh_providers():
    return {
        "service": "video-selection-planner",
        "runner": "image-to-mesh",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "default_provider": DEFAULTS["image_to_mesh"],
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


@app.post("/plan")
async def plan(payload: dict):
    route = payload.get("route") or {}
    print_volume = payload.get("print_volume") or {}
    media_type = (payload.get("input") or {}).get("media_type", "photo")
    model_ids = payload.get("models") or {}

    selection = _selected_model(model_ids.get("selection"), "selection")
    stl_postprocess = _selected_model(model_ids.get("stl_postprocess"), "stl_postprocess")

    stages = []
    if media_type == "video":
        frame_selection = _selected_model(model_ids.get("frame_selection"), "frame_selection")
        camera_pose = _selected_model(model_ids.get("camera_pose"), "camera_pose")
        video_reconstruction = _selected_model(model_ids.get("video_reconstruction"), "video_reconstruction")
        stages = [
            {"id": "frame-selection", "model": frame_selection},
            {"id": "object-selection", "model": selection},
            {"id": "camera-pose", "model": camera_pose},
            {"id": "video-reconstruction", "model": video_reconstruction},
            {"id": "stl-postprocess", "model": stl_postprocess},
        ]
    else:
        image_to_mesh = _selected_model(model_ids.get("image_to_mesh"), "image_to_mesh")
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
    provider: str | None = Form(None),
    provider_device: str = Form("cuda"),
    model_name: str | None = Form(None),
    mesh_repair: str = Form("printable"),
    mesh_target_max_dimension: float = Form(96.0),
    mesh_min_bbox_dimension: float = Form(12.0),
    mesh_max_bbox_aspect_ratio: float = Form(0.0),
    mesh_target_faces: int = Form(40000),
    low_vram: bool = Form(True),
    chunk_size: int = Form(8192),
    mc_resolution: int = Form(256),
    texture_resolution: int | None = Form(None),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.0),
    octree_resolution: int = Form(256),
    num_chunks: int = Form(8000),
    seed: int | None = Form(None),
    disable_progress: bool = Form(True),
):
    selected_provider = provider_for_model(provider)
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
        python=runtime_config["provider_python"],
        timeout=runtime_config["timeout"],
        low_vram=bool(low_vram),
        provider_device=provider_device,
        chunk_size=int(chunk_size),
        mc_resolution=int(mc_resolution),
        texture_resolution=texture_resolution,
        remesh_option=None,
        mesh_repair=mesh_repair,
        mesh_target_max_dimension=float(mesh_target_max_dimension or 0.0),
        mesh_min_bbox_dimension=float(mesh_min_bbox_dimension or 0.0),
        mesh_max_bbox_aspect_ratio=float(mesh_max_bbox_aspect_ratio or 0.0),
        mesh_target_bbox_extents=None,
        mesh_target_faces=int(mesh_target_faces or 0),
        provider_arg=[],
        model_name=model_name,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        seed=seed,
        mc_algo=None,
        disable_progress=bool(disable_progress),
        prefetch_only=False,
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
                "runner": "image-to-mesh",
                "provider": selected_provider,
                "artifact_contract": "output_model.stl + diagnostics.json",
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
        provider_python_configured = bool(configured_env_value(provider_python_env_names(selected_provider)))
        metadata = {
            "job_id": job_id,
            "source_filename": file.filename,
            "service": "video-selection-planner",
            "runner": "image-to-mesh",
            "provider": selected_provider,
            "provider_dir_configured": bool(runtime_config["provider_dir"]),
            "provider_python_configured": provider_python_configured,
            "provider_device": provider_device,
            "provider_timeout_seconds": runtime_config["timeout"],
            "mesh_repair": mesh_repair,
            "mesh_target_max_dimension": mesh_target_max_dimension,
            "mesh_min_bbox_dimension": mesh_min_bbox_dimension,
            "mesh_max_bbox_aspect_ratio": mesh_max_bbox_aspect_ratio,
            "mesh_target_faces": mesh_target_faces,
            "timings": timings,
            "started_at": started_at,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        metadata_path = job_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

        mesh_relative = output_relative_path(mesh_path)
        stl_relative = output_relative_path(stl_path)
        diagnostics_relative = output_relative_path(diagnostics_path)
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
    provider: str | None = Form(None),
    provider_device: str = Form("cuda"),
    mesh_repair: str = Form("printable"),
    mesh_target_max_dimension: float = Form(96.0),
    mesh_min_bbox_dimension: float = Form(12.0),
    mesh_max_bbox_aspect_ratio: float = Form(0.0),
    mesh_target_faces: int = Form(40000),
    cameras_json: str | None = Form(None),
    view_ids_json: str | None = Form(None),
    visual_hull_resolution: int = Form(32),
    visual_hull_grid_extent: float = Form(1.9),
    visual_hull_ortho_scale: float = Form(2.0),
    visual_hull_mask_dilate: int = Form(1),
):
    selected_provider = provider_for_multiview_model(provider)
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
        diagnostics.update(
            {
                "job_id": job_id,
                "runner": "multiview-to-mesh",
                "provider": selected_provider,
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


@app.get("/artifacts/{file_path:path}")
async def get_artifact(file_path: str):
    resolved_path = resolve_output_file(file_path, (".stl", ".json", ".glb", ".gltf", ".obj", ".ply", ".png", ".jpg", ".jpeg", ".webp"))
    media_type = "application/json" if resolved_path.suffix.lower() == ".json" else None
    return FileResponse(resolved_path, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VIDEO_SELECTION_PORT", "8005")))
