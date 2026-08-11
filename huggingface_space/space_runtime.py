from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageOps


PROJECT_REPOSITORY = "https://github.com/DrStrangel0ve/3dprintpic.git"
PROJECT_REVISION = "3808d76bf02d31b9a6724e825bfd325bf6c3d412"
TRIPOSG_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSG.git"
TRIPOSG_SOURCE_REVISION = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
TRIPOSG_MODEL_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
TRIPOSG_REMBG_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
SAM3_MODEL = "facebook/sam3"

SPACE_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(os.getenv("THREEDPRINTPIC_RUNTIME_DIR", Path(tempfile.gettempdir()) / "3dprintpic-space"))
OUTPUT_DIR = RUNTIME_ROOT / "output"
SOURCE_DIR = RUNTIME_ROOT / "source"
TRIPOSG_DIR = RUNTIME_ROOT / "TripoSG"

_SOURCE_LOCK = Lock()
_TRIPOSG_LOCK = Lock()


def _run_git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout.strip()


def _repository_root_if_local() -> Path | None:
    explicit = os.getenv("THREEDPRINTPIC_SOURCE_DIR")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend((SPACE_DIR.parent, Path.cwd()))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "backend" / "main.py").is_file():
            return resolved
    return None


def _clone_exact_repository(repository: str, revision: str, destination: Path) -> Path:
    with _SOURCE_LOCK:
        if (destination / ".git").is_dir():
            current = _run_git("rev-parse", "HEAD", cwd=destination)
            if current == revision:
                return destination
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", "--filter=blob:none", "--no-checkout", repository, str(destination))
        _run_git("checkout", "--detach", revision, cwd=destination)
        current = _run_git("rev-parse", "HEAD", cwd=destination)
        if current != revision:
            raise RuntimeError(f"Source checkout mismatch: expected {revision}, got {current}")
        return destination


def ensure_project_source() -> Path:
    local = _repository_root_if_local()
    if local is not None:
        return local
    return _clone_exact_repository(PROJECT_REPOSITORY, PROJECT_REVISION, SOURCE_DIR)


PROJECT_DIR = ensure_project_source()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OUTPUT_DIR", str(OUTPUT_DIR))
os.environ.setdefault("DEPTH_PROVIDER", "transformers")
os.environ.setdefault("DEPTH_MODEL", DEPTH_MODEL)
os.environ.setdefault("SELECTION_ALLOW_MODEL_DOWNLOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from backend import main as backend_main  # noqa: E402
from backend.benchmark.direct_mesh import (  # noqa: E402
    convert_mesh_to_stl,
    postprocess_mesh_for_stl,
    repair_mesh_for_printable_stl,
)
from backend.stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics  # noqa: E402


BACKEND_CLIENT = TestClient(backend_main.app)


def image_dimensions_mm(image_path: str | Path | None, long_edge_mm: float) -> tuple[float, float]:
    if not image_path:
        return float(long_edge_mm), float(long_edge_mm)
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    scale = float(long_edge_mm) / float(max(width, height, 1))
    return round(width * scale, 2), round(height * scale, 2)


def _safe_file(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected output was not created: {resolved.name}")
    return str(resolved)


def _response_error(response) -> RuntimeError:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload
    else:
        detail = payload
    return RuntimeError(f"Generation failed ({response.status_code}): {detail}")


def _diagnostic_summary(diagnostics: dict, *, model: str) -> dict:
    keys = (
        "is_watertight",
        "is_volume",
        "winding_consistent",
        "component_count",
        "nonmanifold_edge_count",
        "degenerate_face_count",
        "positive_volume",
        "bbox_extents",
        "face_count",
        "vertex_count",
        "normalized_bbox_complexity_log1p",
        "stl_passes_hard_checks",
        "stl_failed_checks",
    )
    return {
        "model": model,
        **{key: diagnostics[key] for key in keys if key in diagnostics},
    }


def _selection_job(
    image_path: str | Path,
    point_x: float,
    point_y: float,
) -> dict:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "SAM 3 needs the Space owner's read-only HF_TOKEN secret before object selection can run."
        )
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    normalized_point = {
        "x": float(np.clip(point_x / max(width, 1), 0.0, 1.0)),
        "y": float(np.clip(point_y / max(height, 1), 0.0, 1.0)),
    }
    mask, resolved_model, model_status, labels = backend_main.sam3_person_aware_selection_mask(
        image,
        [normalized_point],
        device="cuda",
    )
    if resolved_model != SAM3_MODEL or not model_status.startswith("sam3"):
        raise RuntimeError("SAM 3 did not produce a verified selection mask")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    selected = backend_main.selected_image_from_mask(image, mask, background_mode="neutral")
    overlay = backend_main.selection_overlay(image, mask)
    mask_pixels = int(np.count_nonzero(np.asarray(mask.convert("L")) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": Path(image_path).name,
        "source_fingerprint": backend_main.selection_source_fingerprint(image),
        "mask_count": 1,
        "model_id": resolved_model,
        "selection_model_status": model_status,
        "model_status": "composed-clicked-masks",
        "selection_labels": labels,
        "background_mode": "neutral",
        "selection_infill_mode": "none",
        "selection_infill": {"mode": "none", "enabled": False},
        "face_detection_source": "neutral-selection-cutout",
        "image_size": {"width": width, "height": height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, width * height)),
    }
    image.save(job_dir / "source.png")
    selected.save(job_dir / "selected_image.png")
    selected.save(job_dir / "selection_face_detection.png")
    mask.save(job_dir / "selection_mask.png")
    overlay.save(job_dir / "selection_overlay.png")
    backend_main.selection_tint(mask).save(job_dir / "selection_tint.png")
    (job_dir / "selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "job_id": job_id,
        "overlay": _safe_file(job_dir / "selection_overlay.png"),
        "mask": backend_main.output_relative_path(job_dir / "selection_mask.png"),
        "labels": labels,
        "model": resolved_model,
        "model_status": model_status,
        "mask_coverage": metadata["mask_coverage"],
    }


def select_object(image_path: str | Path | None, point_x: float, point_y: float) -> dict:
    if not image_path:
        raise ValueError("Upload a photo before selecting an object")
    return _selection_job(image_path, point_x, point_y)


def release_gpu_models() -> None:
    try:
        backend_main.release_selection_models()
    except Exception:
        pass
    try:
        backend_main.release_depth_pipelines()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_relief(
    image_path: str | Path | None,
    scope: str,
    selection: dict | None,
    long_edge_mm: float,
    relief_height_mm: float,
    detail_samples: int,
    background_depth_ratio: float,
) -> tuple[str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a relief")
    selected = str(scope).lower().startswith("select")
    if selected and not selection:
        raise ValueError("Click the object in the image before generating the selected-object relief")
    x_mm, y_mm = image_dimensions_mm(image_path, long_edge_mm)
    data = {
        "selection_mode": "context",
        "selection_subject_lock": "true" if selected else "false",
        "depth_provider": "transformers",
        "depth_model": DEPTH_MODEL,
        "device": "cuda",
        "target_dimension": str(int(detail_samples)),
        "z_scale": str(float(relief_height_mm)),
        "max_xy_size": str(max(x_mm, y_mm)),
        "relief_polarity": "raised-print",
        "selection_background_depth_ratio": str(float(background_depth_ratio)),
        "completion_mode": "none",
        "face_refinement_mode": "auto",
    }
    if selected:
        data["selection_job_id"] = selection["job_id"]
    release_gpu_models()
    with open(image_path, "rb") as image_file:
        response = BACKEND_CLIENT.post(
            "/process_image",
            files={"file": (Path(image_path).name, image_file, "image/png")},
            data=data,
        )
    if response.status_code != 200:
        raise _response_error(response)
    result = response.json()
    job_dir = OUTPUT_DIR / result["job_id"]
    stl_path = OUTPUT_DIR / result["stl_model"]
    preview_path = job_dir / "output_relief_preview.png"
    if not preview_path.is_file():
        preview_path = job_dir / "output_depth_preview.png"
    diagnostics_path = OUTPUT_DIR / result["diagnostics"]
    diagnostics = result.get("stl_diagnostics", {})
    summary = _diagnostic_summary(diagnostics, model=DEPTH_MODEL)
    summary["dimensions_mm"] = {"x": x_mm, "y": y_mm, "z": float(relief_height_mm)}
    summary["scope"] = "selected-object-with-scene-context" if selected else "full-scene"
    summary["inpainting"] = False
    return (
        _safe_file(stl_path),
        _safe_file(stl_path),
        _safe_file(preview_path),
        {"summary": summary, "full_report": _safe_file(diagnostics_path)},
    )


def generate_diorama(
    image_path: str | Path | None,
    selection: dict | None,
    max_size_mm: float,
    scene_depth_mm: float,
    base_thickness_mm: float,
) -> tuple[str, str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a scene diorama")
    release_gpu_models()
    data = {
        "mask_paths_json": json.dumps([selection["mask"]] if selection else []),
        "selection_labels_json": json.dumps([selection.get("labels", [])] if selection else []),
        "depth_provider": "transformers",
        "depth_model": DEPTH_MODEL,
        "device": "cuda",
        "max_size_mm": str(float(max_size_mm)),
        "scene_depth_mm": str(float(scene_depth_mm)),
        "base_thickness_mm": str(float(base_thickness_mm)),
    }
    with open(image_path, "rb") as image_file:
        response = BACKEND_CLIENT.post(
            "/process_scene_diorama",
            files={"file": (Path(image_path).name, image_file, "image/png")},
            data=data,
        )
    if response.status_code != 200:
        raise _response_error(response)
    result = response.json()
    stl_path = OUTPUT_DIR / result["stl_model"]
    glb_path = OUTPUT_DIR / result["scene_model"]
    preview_path = OUTPUT_DIR / result["preview"]
    diagnostics = result.get("stl_diagnostics", {})
    summary = _diagnostic_summary(diagnostics, model=DEPTH_MODEL)
    summary["scene_mode"] = "layered-camera-free-diorama"
    summary["selected_layers"] = 1 if selection else 0
    summary["inpainting"] = False
    return (
        _safe_file(glb_path),
        _safe_file(stl_path),
        _safe_file(glb_path),
        _safe_file(preview_path),
        summary,
    )


def _patch_triposg_source(source_dir: Path) -> None:
    inference_path = source_dir / "triposg" / "inference_utils.py"
    vae_path = source_dir / "triposg" / "models" / "autoencoders" / "autoencoder_kl_triposg.py"
    inference_text = inference_path.read_text(encoding="utf-8")
    inference_text = inference_text.replace(
        "from diso import DiffDMC",
        "try:\n    from diso import DiffDMC\nexcept ImportError:\n    DiffDMC = None",
        1,
    )
    inference_path.write_text(inference_text, encoding="utf-8")
    vae_text = vae_path.read_text(encoding="utf-8")
    vae_text = vae_text.replace(
        "            q = self.proj_query(q)",
        "            q = self.proj_query(q.to(dtype=self.proj_query.weight.dtype))",
        1,
    )
    vae_path.write_text(vae_text, encoding="utf-8")


def ensure_triposg_source() -> Path:
    with _TRIPOSG_LOCK:
        source = _clone_exact_repository(TRIPOSG_REPOSITORY, TRIPOSG_SOURCE_REVISION, TRIPOSG_DIR)
        _patch_triposg_source(source)
        return source


def _load_triposg_models():
    import torch
    from huggingface_hub import snapshot_download

    source = ensure_triposg_source()
    scripts_dir = source / "scripts"
    for path in (source, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from briarmbg import BriaRMBG
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    model_root = RUNTIME_ROOT / "models"
    triposg_weights = Path(
        snapshot_download(
            repo_id="VAST-AI/TripoSG",
            revision=TRIPOSG_MODEL_REVISION,
            local_dir=model_root / "TripoSG",
            token=os.getenv("HF_TOKEN") or None,
        )
    )
    rmbg_weights = Path(
        snapshot_download(
            repo_id="briaai/RMBG-1.4",
            revision=TRIPOSG_REMBG_REVISION,
            local_dir=model_root / "RMBG-1.4",
            allow_patterns=["config.json", "model.safetensors"],
            token=os.getenv("HF_TOKEN") or None,
        )
    )
    rmbg = BriaRMBG.from_pretrained(rmbg_weights).to("cuda")
    rmbg.eval()
    pipeline = TripoSGPipeline.from_pretrained(
        triposg_weights,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    return pipeline, rmbg


def generate_full_mesh(
    image_path: str | Path | None,
    scope: str,
    selection: dict | None,
    max_dimension_mm: float,
    seed: int,
) -> tuple[str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a full mesh")
    selected = str(scope).lower().startswith("select")
    if selected and not selection:
        raise ValueError("Click the object in the image before generating the selected-object mesh")
    mesh_input_path = Path(image_path)
    if selected:
        mesh_input_path = OUTPUT_DIR / "selection" / selection["job_id"] / "selected_image.png"
        _safe_file(mesh_input_path)
    import torch
    import trimesh

    release_gpu_models()
    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "full-mesh" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    raw_glb = job_dir / "output_mesh_raw.glb"
    repaired_mesh = job_dir / "output_mesh_repaired.ply"
    final_glb = job_dir / "output_mesh.glb"
    final_stl = job_dir / "output_model.stl"
    metrics: dict = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pipeline = None
    rmbg = None
    try:
        pipeline, rmbg = _load_triposg_models()
        scripts_dir = ensure_triposg_source() / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from image_process import prepare_image

        prepared = prepare_image(
            mesh_input_path,
            bg_color=np.array([1.0, 1.0, 1.0]),
            rmbg_net=rmbg,
        )
        with torch.inference_mode():
            samples = pipeline(
                image=prepared,
                generator=torch.Generator(device="cuda").manual_seed(int(seed)),
                num_inference_steps=50,
                guidance_scale=7.0,
                use_flash_decoder=False,
            ).samples[0]
        mesh = trimesh.Trimesh(
            samples[0].astype(np.float32),
            np.ascontiguousarray(samples[1]),
            process=False,
        )
        if not len(mesh.vertices) or not len(mesh.faces):
            raise RuntimeError("TripoSG returned an empty mesh")
        mesh.export(raw_glb)
        repair_mesh_for_printable_stl(
            raw_glb,
            repaired_mesh,
            "printable",
            max_normalized_face_density_log1p=9.95,
            preconditioner="adaptive-voxel-close",
            voxel_resolution=128,
            voxel_fill_method="orthographic",
            smoothing_iterations=2,
            allow_convex_hull_fallback=False,
            metrics=metrics,
        )
        postprocess_mesh_for_stl(
            repaired_mesh,
            final_glb,
            target_max_dimension=float(max_dimension_mm),
            target_faces=10000,
            max_normalized_face_density_log1p=9.95,
            preserve_printability=True,
        )
        convert_mesh_to_stl(final_glb, final_stl)
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(final_stl))
        summary = _diagnostic_summary(diagnostics, model="VAST-AI/TripoSG")
        summary.update(
            {
                "source_revision": TRIPOSG_SOURCE_REVISION,
                "model_revision": TRIPOSG_MODEL_REVISION,
                "foreground_model_revision": TRIPOSG_REMBG_REVISION,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
                "repair": metrics,
                "scope": "selected-object" if selected else "full-visible-subject",
                "inpainting": False,
            }
        )
        (job_dir / "diagnostics.json").write_text(
            json.dumps({"stl": diagnostics, "summary": summary}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return _safe_file(final_stl), _safe_file(final_stl), _safe_file(final_glb), summary
    finally:
        del pipeline, rmbg
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_expired_outputs(max_age_seconds: int = 6 * 60 * 60) -> int:
    if not OUTPUT_DIR.is_dir():
        return 0
    now = time.time()
    removed = 0
    for child in OUTPUT_DIR.iterdir():
        if child.name == "selection" or not child.is_dir():
            continue
        try:
            if now - child.stat().st_mtime > max_age_seconds:
                shutil.rmtree(child)
                removed += 1
        except OSError:
            continue
    return removed
