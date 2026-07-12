from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from backend.benchmark.patch_step1x3d_sources import (
    EAGER_IMPORTS,
    GEOMETRY_ONLY_IMPORTS,
)


DEFAULT_STEP1X3D_SOURCE_REVISION = "cb5ac944709c6c913109070c7b90c3447f57f3d4"
DEFAULT_STEP1X3D_MODEL = "stepfun-ai/Step1X-3D"
DEFAULT_STEP1X3D_MODEL_REVISION = "bf7084495b3a72222f36549b7942948aa4d9daa7"
DEFAULT_STEP1X3D_SUBFOLDER = "Step1X-3D-Geometry-1300m"
DEFAULT_STEP1X3D_STEPS = 50
DEFAULT_STEP1X3D_GUIDANCE = 7.5
DEFAULT_STEP1X3D_OCTREE_RESOLUTION = 384
DEFAULT_STEP1X3D_MAX_FACES = 0
DEFAULT_STEP1X3D_SEED = 2025
PROVIDER_NATIVE_METRICS_FILENAME = "provider_native_metrics.json"

_STEP1X3D_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "visual_eature_extractor/preprocessor_config.json",
    "visual_encoder/config.json",
    "visual_encoder/diffusion_pytorch_model.safetensors",
)
_STEP1X3D_PATCHED_SOURCE = "step1x3d_geometry/__init__.py"


def step1x3d_model_specs(
    *,
    model_repo: str = DEFAULT_STEP1X3D_MODEL,
    model_revision: str = DEFAULT_STEP1X3D_MODEL_REVISION,
    subfolder: str = DEFAULT_STEP1X3D_SUBFOLDER,
) -> dict[str, dict[str, str | tuple[str, ...]]]:
    return {
        "step1x3d": {
            "repo_id": model_repo,
            "revision": model_revision,
            "subfolder": subfolder,
            "required_file": f"{subfolder}/model_index.json",
            "required_files": tuple(
                f"{subfolder}/{relative_path}"
                for relative_path in _STEP1X3D_REQUIRED_FILES
            ),
        }
    }


def resolve_step1x3d_model_snapshot(
    model_repo: str = DEFAULT_STEP1X3D_MODEL,
    model_revision: str = DEFAULT_STEP1X3D_MODEL_REVISION,
    subfolder: str = DEFAULT_STEP1X3D_SUBFOLDER,
) -> Path:
    configured_path = Path(model_repo).expanduser()
    if configured_path.exists():
        snapshot_path = configured_path.resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=model_repo,
                revision=model_revision,
                allow_patterns=[f"{subfolder}/*"],
            )
        ).resolve()

    spec = step1x3d_model_specs(
        model_repo=model_repo,
        model_revision=model_revision,
        subfolder=subfolder,
    )["step1x3d"]
    missing = [
        snapshot_path / str(relative_path)
        for relative_path in spec["required_files"]
        if not (snapshot_path / str(relative_path)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Pinned Step1X-3D snapshot is missing required files: "
            + ", ".join(str(path) for path in missing)
        )

    print(
        json.dumps(
            {
                "event": "step1x3d_model_snapshot",
                "model": {
                    "repo_id": model_repo,
                    "revision": model_revision,
                    "subfolder": subfolder,
                },
                "path": str(snapshot_path),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return snapshot_path


def verify_step1x3d_source_integrity(provider_dir: Path) -> dict[str, object]:
    provider_dir = provider_dir.resolve()
    source_path = provider_dir / _STEP1X3D_PATCHED_SOURCE
    if not source_path.is_file():
        raise FileNotFoundError(f"Step1X-3D patched source is missing: {source_path}")

    original = subprocess.check_output(
        ["git", "show", f"HEAD:{_STEP1X3D_PATCHED_SOURCE}"],
        cwd=provider_dir,
    ).decode("utf-8")
    if original.count(EAGER_IMPORTS) != 1:
        raise ValueError("Pinned Step1X-3D source has an unexpected eager import marker")
    expected = original.replace(EAGER_IMPORTS, GEOMETRY_ONLY_IMPORTS)
    actual = source_path.read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError(
            "Step1X-3D patched package initializer does not match the expected geometry-only source"
        )

    tracked_changes = tuple(
        line.strip()
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD", "--"],
            cwd=provider_dir,
            text=True,
        ).splitlines()
        if line.strip()
    )
    if tracked_changes != (_STEP1X3D_PATCHED_SOURCE,):
        raise ValueError(
            "Step1X-3D checkout has unexpected tracked modifications: "
            + (", ".join(tracked_changes) or "none")
        )
    untracked = tuple(
        line.strip()
        for line in subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=provider_dir,
            text=True,
        ).splitlines()
        if line.strip()
    )
    unexpected_untracked = tuple(
        path
        for path in untracked
        if "__pycache__/" not in path.replace("\\", "/")
        and not path.endswith((".pyc", ".pyo"))
    )
    if unexpected_untracked:
        raise ValueError(
            "Step1X-3D checkout has unexpected untracked source: "
            + ", ".join(unexpected_untracked)
        )

    return {
        "patched_source": _STEP1X3D_PATCHED_SOURCE,
        "patched_source_sha256": hashlib.sha256(actual.encode("utf-8")).hexdigest(),
        "tracked_changes": list(tracked_changes),
        "ignored_bytecode_files": len(untracked) - len(unexpected_untracked),
    }


def _cuda_metrics(torch_module, device) -> dict[str, object]:
    if not torch_module.cuda.is_available():
        return {
            "provider_peak_cuda_vram_gib": None,
            "provider_peak_cuda_vram_supported": False,
            "provider_peak_cuda_vram_measurement": "unsupported",
            "provider_peak_cuda_allocated_gib": None,
            "provider_peak_cuda_reserved_gib": None,
        }
    divisor = float(1024**3)
    allocated = float(torch_module.cuda.max_memory_allocated(device)) / divisor
    reserved = float(torch_module.cuda.max_memory_reserved(device)) / divisor
    return {
        "provider_peak_cuda_vram_gib": max(allocated, reserved),
        "provider_peak_cuda_vram_supported": True,
        "provider_peak_cuda_vram_measurement": "torch-max-memory-reserved",
        "provider_peak_cuda_allocated_gib": allocated,
        "provider_peak_cuda_reserved_gib": reserved,
    }


def run_step1x3d(
    *,
    provider_dir: Path,
    input_image: Path,
    output_mesh: Path,
    model_repo: str = DEFAULT_STEP1X3D_MODEL,
    model_revision: str = DEFAULT_STEP1X3D_MODEL_REVISION,
    subfolder: str = DEFAULT_STEP1X3D_SUBFOLDER,
    num_inference_steps: int = DEFAULT_STEP1X3D_STEPS,
    guidance_scale: float = DEFAULT_STEP1X3D_GUIDANCE,
    octree_resolution: int = DEFAULT_STEP1X3D_OCTREE_RESOLUTION,
    max_faces: int = DEFAULT_STEP1X3D_MAX_FACES,
    seed: int = DEFAULT_STEP1X3D_SEED,
    device: str = "cuda",
) -> Path:
    verify_step1x3d_source_integrity(provider_dir)
    snapshot_path = resolve_step1x3d_model_snapshot(
        model_repo,
        model_revision,
        subfolder,
    )
    provider_dir = provider_dir.resolve()
    sys.path.insert(0, str(provider_dir))
    os.environ["USE_SAGEATTN"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import trimesh
    from step1x3d_geometry.models.pipelines.pipeline import (
        Step1X3DGeometryPipeline,
    )

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Step1X-3D requested CUDA but torch.cuda.is_available() is false")
    cuda_device = torch.device(device) if str(device).startswith("cuda") else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    started = time.perf_counter()
    pipeline = Step1X3DGeometryPipeline.from_pretrained(
        str(snapshot_path),
        subfolder=subfolder,
        torch_dtype=dtype,
    ).to(device)
    generator_device = device if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    output = pipeline(
        str(input_image),
        guidance_scale=float(guidance_scale),
        num_inference_steps=max(1, int(num_inference_steps)),
        octree_resolution=max(32, int(octree_resolution)),
        max_facenum=max(0, int(max_faces)),
        generator=generator,
        do_remove_floater=False,
        do_remove_degenerate_face=False,
        do_reduce_face=False,
        do_shade_smooth=False,
        output_type="raw",
    )
    meshes = output.mesh
    if not isinstance(meshes, (list, tuple)) or len(meshes) != 1:
        count = len(meshes) if isinstance(meshes, (list, tuple)) else type(meshes).__name__
        raise RuntimeError(f"Step1X-3D returned {count} meshes; expected exactly one")
    raw_mesh = meshes[0]
    vertices = raw_mesh.verts.detach().cpu().numpy()
    faces = raw_mesh.faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Step1X-3D returned an empty mesh")

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    metrics = {
        "provider": "step1x3d",
        "provider_inference_runtime_seconds": time.perf_counter() - started,
        "provider_raw_vertex_count": int(len(mesh.vertices)),
        "provider_raw_face_count": int(len(mesh.faces)),
        **_cuda_metrics(torch, cuda_device),
    }
    (output_mesh.parent / PROVIDER_NATIVE_METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned Step1X-3D geometry-only image-to-mesh inference."
    )
    parser.add_argument("--provider-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--output-mesh", type=Path, default=None)
    parser.add_argument("--model-path", default=DEFAULT_STEP1X3D_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_STEP1X3D_MODEL_REVISION)
    parser.add_argument("--subfolder", default=DEFAULT_STEP1X3D_SUBFOLDER)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEP1X3D_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_STEP1X3D_GUIDANCE)
    parser.add_argument("--octree-resolution", type=int, default=DEFAULT_STEP1X3D_OCTREE_RESOLUTION)
    parser.add_argument("--max-faces", type=int, default=DEFAULT_STEP1X3D_MAX_FACES)
    parser.add_argument("--seed", type=int, default=DEFAULT_STEP1X3D_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefetch-only", action="store_true")
    args = parser.parse_args()

    if args.prefetch_only:
        verify_step1x3d_source_integrity(args.provider_dir)
        snapshot_path = resolve_step1x3d_model_snapshot(
            args.model_path,
            args.model_revision,
            args.subfolder,
        )
        print(f"snapshot={snapshot_path}")
        return
    if args.input_image is None or args.output_mesh is None:
        raise ValueError(
            "--input-image and --output-mesh are required unless --prefetch-only is set"
        )
    output_mesh = run_step1x3d(
        provider_dir=args.provider_dir,
        input_image=args.input_image,
        output_mesh=args.output_mesh,
        model_repo=args.model_path,
        model_revision=args.model_revision,
        subfolder=args.subfolder,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        max_faces=args.max_faces,
        seed=args.seed,
        device=args.device,
    )
    print(f"mesh={output_mesh}")


if __name__ == "__main__":
    main()
