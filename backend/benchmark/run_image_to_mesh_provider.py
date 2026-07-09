from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.direct_mesh import (
    MESH_REPAIR_MODES,
    convert_mesh_to_stl,
    postprocess_mesh_for_stl,
    repair_mesh_for_printable_stl,
)
from backend.benchmark.mesh_rendering import load_mesh


MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl")
MESH_EXTENSION_PRIORITY = {".glb": 5, ".gltf": 4, ".obj": 3, ".ply": 2, ".stl": 1}
TRIPOSR_API_PROVIDER = "triposr-api"
HUNYUAN3D_SHAPE_PROVIDER = "hunyuan3d-shape"
DEFAULT_TRIPOSR_MODEL = "stabilityai/TripoSR"
DEFAULT_HUNYUAN3D_MODEL = "tencent/Hunyuan3D-2.1"

CLI_PROVIDERS = {
    "spar3d": {
        "env": "SPAR3D_DIR",
        "default_dirs": ("/content/stable-point-aware-3d", "/content/SPAR3D"),
        "supports_low_vram": True,
        "supports_remesh": True,
        "supports_texture_resolution": True,
    },
    "stable-fast-3d": {
        "env": "SF3D_DIR",
        "default_dirs": ("/content/stable-fast-3d", "/content/SF3D"),
        "supports_low_vram": False,
        "supports_remesh": True,
        "supports_texture_resolution": True,
    },
    "triposr": {
        "env": "TRIPOSR_DIR",
        "default_dirs": ("/content/TripoSR", "/content/triposr"),
        "supports_low_vram": False,
        "supports_remesh": False,
        "supports_texture_resolution": True,
    },
}

PROVIDERS = tuple(sorted((*CLI_PROVIDERS, TRIPOSR_API_PROVIDER, HUNYUAN3D_SHAPE_PROVIDER)))


def parse_bbox_extents(value: str) -> tuple[float, float, float] | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--mesh-target-bbox-extents expects three positive numbers")
    try:
        extents = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--mesh-target-bbox-extents expects numeric values") from exc
    if not all(math.isfinite(extent) and extent > 0 for extent in extents):
        raise argparse.ArgumentTypeError("--mesh-target-bbox-extents values must be finite and positive")
    return extents


def provider_dir_config_key(provider: str) -> str:
    return "triposr" if provider == TRIPOSR_API_PROVIDER else provider


def resolve_provider_dir(provider: str, explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    config = CLI_PROVIDERS.get(provider_dir_config_key(provider), {})
    env_name = config.get("env")
    if env_name and os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    candidates.extend(Path(path) for path in config.get("default_dirs", ()))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hint = f" Set --provider-dir or ${env_name}." if env_name else " Set --provider-dir."
    raise FileNotFoundError(f"Could not find provider repo for {provider}.{hint}")


def find_mesh_output(output_dir: Path, started_at: float | None = None) -> Path:
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    ]
    if started_at is not None:
        fresh = [path for path in candidates if path.stat().st_mtime >= started_at - 1.0]
        if fresh:
            candidates = fresh
    if not candidates:
        raise FileNotFoundError(f"No mesh output found under {output_dir}")

    def candidate_key(path: Path) -> tuple[int, int, float, int]:
        name = path.stem.lower()
        name_score = 2 if "mesh" in name or "model" in name else 0
        if "point" in name or "pcd" in name:
            name_score -= 1
        return (
            name_score,
            MESH_EXTENSION_PRIORITY.get(path.suffix.lower(), 0),
            path.stat().st_mtime,
            path.stat().st_size,
        )

    candidates.sort(key=candidate_key, reverse=True)
    return candidates[0]


def export_mesh(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    if source.suffix.lower() == target.suffix.lower():
        shutil.copy2(source, target)
        return target
    mesh = load_mesh(source)
    mesh.export(target)
    return target


def cli_provider_command(args: argparse.Namespace, provider_dir: Path, raw_output_dir: Path) -> list[str]:
    config = CLI_PROVIDERS[args.provider]
    command = [
        args.python,
        str(provider_dir / "run.py"),
        str(args.input_image),
        "--output-dir",
        str(raw_output_dir),
    ]
    if args.low_vram and config["supports_low_vram"]:
        command.append("--low-vram-mode")
    if args.texture_resolution and config["supports_texture_resolution"]:
        command.extend(["--texture-resolution", str(args.texture_resolution)])
    if args.remesh_option and config["supports_remesh"]:
        command.extend(["--remesh_option", args.remesh_option])
    if args.provider_device:
        command.extend(["--device", args.provider_device])
    command.extend(args.provider_arg or [])
    return command


def run_cli_provider(args: argparse.Namespace) -> Path:
    provider_dir = resolve_provider_dir(args.provider, args.provider_dir)
    run_py = provider_dir / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"{args.provider} provider repo has no run.py: {run_py}")
    raw_output_dir = args.provider_output_dir or args.output_mesh.parent / f"{args.provider}_raw"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    command = cli_provider_command(args, provider_dir, raw_output_dir)
    subprocess.run(command, cwd=provider_dir, check=True, timeout=args.timeout)
    return find_mesh_output(raw_output_dir, started_at=started_at)


def run_hunyuan_shape(args: argparse.Namespace) -> Path:
    provider_dir_value = args.provider_dir or os.environ.get("HUNYUAN3D_DIR")
    if provider_dir_value:
        provider_dir = Path(provider_dir_value)
        sys.path.insert(0, str(provider_dir))
        sys.path.insert(0, str(provider_dir / "hy3dshape"))
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    device = args.provider_device or "cuda"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_name or DEFAULT_HUNYUAN3D_MODEL,
        device=device,
        dtype=dtype,
    )
    if args.low_vram and str(device).startswith("cuda") and hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload(device=device)
    with torch.no_grad():
        mesh = pipeline(image=str(args.input_image))[0]
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_mesh)
    return args.output_mesh


def install_rembg_stub() -> None:
    if "rembg" in sys.modules:
        return
    rembg = types.ModuleType("rembg")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("rembg is unavailable in triposr-api mode; pass preprocessed RGB/RGBA input instead")

    rembg.remove = unavailable
    rembg.new_session = unavailable
    sys.modules["rembg"] = rembg


def triposr_input_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        rgba = np.asarray(image).astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:4] + (1.0 - rgba[..., 3:4]) * 0.5
        return Image.fromarray((rgb * 255.0).astype(np.uint8)).convert("RGB")
    return image.convert("RGB")


def run_triposr_api(args: argparse.Namespace) -> Path:
    provider_dir = resolve_provider_dir(TRIPOSR_API_PROVIDER, args.provider_dir)
    sys.path.insert(0, str(provider_dir))
    install_rembg_stub()

    import torch
    from tsr.system import TSR

    device = args.provider_device or "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = TSR.from_pretrained(
        args.model_name or DEFAULT_TRIPOSR_MODEL,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)

    image = triposr_input_image(args.input_image)
    with torch.no_grad():
        scene_codes = model([image], device=device)
        meshes = model.extract_mesh(scene_codes, True, resolution=args.mc_resolution)
    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    meshes[0].export(args.output_mesh)
    return args.output_mesh


def run_provider(args: argparse.Namespace) -> tuple[Path, Path | None]:
    args.input_image = Path(args.input_image)
    args.output_mesh = Path(args.output_mesh)
    args.output_stl = Path(args.output_stl) if args.output_stl else None
    args.raw_output_mesh = Path(args.raw_output_mesh) if args.raw_output_mesh else None
    if not args.input_image.exists():
        raise FileNotFoundError(f"Input image does not exist: {args.input_image}")

    if args.provider in CLI_PROVIDERS:
        provider_mesh = run_cli_provider(args)
        output_mesh = export_mesh(provider_mesh, args.output_mesh)
    elif args.provider == TRIPOSR_API_PROVIDER:
        output_mesh = run_triposr_api(args)
    elif args.provider == HUNYUAN3D_SHAPE_PROVIDER:
        output_mesh = run_hunyuan_shape(args)
    else:
        raise ValueError(f"Unsupported provider: {args.provider}")

    if args.mesh_repair != "none":
        raw_output_mesh = args.raw_output_mesh or output_mesh.with_name(
            f"{output_mesh.stem}_raw{output_mesh.suffix}"
        )
        raw_output_mesh.parent.mkdir(parents=True, exist_ok=True)
        if output_mesh.resolve() != raw_output_mesh.resolve():
            shutil.copy2(output_mesh, raw_output_mesh)
        output_mesh = repair_mesh_for_printable_stl(raw_output_mesh, args.output_mesh, args.mesh_repair)

    if (
        args.mesh_target_max_dimension > 0
        or args.mesh_min_bbox_dimension > 0
        or args.mesh_max_bbox_aspect_ratio > 0
        or args.mesh_target_bbox_extents is not None
        or args.mesh_target_faces > 0
    ):
        if args.raw_output_mesh and args.mesh_repair == "none" and output_mesh.resolve() != args.raw_output_mesh.resolve():
            args.raw_output_mesh.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_mesh, args.raw_output_mesh)
        output_mesh = postprocess_mesh_for_stl(
            output_mesh,
            args.output_mesh,
            target_max_dimension=args.mesh_target_max_dimension,
            min_bbox_dimension=args.mesh_min_bbox_dimension,
            max_bbox_aspect_ratio=args.mesh_max_bbox_aspect_ratio,
            target_bbox_extents=args.mesh_target_bbox_extents,
            target_faces=args.mesh_target_faces,
        )

    output_stl = None
    if args.output_stl:
        output_stl = convert_mesh_to_stl(output_mesh, args.output_stl)
    return output_mesh, output_stl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an external single-image-to-mesh provider and normalize its output for the benchmark."
    )
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--output-stl", default=None)
    parser.add_argument("--raw-output-mesh", default=None)
    parser.add_argument("--provider-dir", default=None)
    parser.add_argument("--provider-output-dir", type=Path, default=None)
    parser.add_argument(
        "--python",
        "--provider-python",
        dest="python",
        default=sys.executable,
        help="Python executable used to launch CLI provider repos. Use this to isolate provider dependencies in a venv.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--provider-device", default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--texture-resolution", type=int, default=None)
    parser.add_argument("--remesh-option", choices=("none", "triangle", "quad"), default=None)
    parser.add_argument(
        "--mesh-repair",
        choices=MESH_REPAIR_MODES,
        default="none",
        help=(
            "Postprocess the provider mesh before STL export. 'basic' keeps the largest connected body and "
            "runs Trimesh cleanup; 'convex-hull' forces a watertight hull; 'printable' tries basic repair and "
            "falls back to a hull only if watertight/volume/single-component checks still fail."
        ),
    )
    parser.add_argument(
        "--mesh-target-max-dimension",
        type=float,
        default=0.0,
        help=(
            "If positive, center and uniformly scale the provider mesh so its longest bounding-box side "
            "matches this STL-space dimension before STL export."
        ),
    )
    parser.add_argument(
        "--mesh-min-bbox-dimension",
        type=float,
        default=0.0,
        help=(
            "If positive, anisotropically thicken any bounding-box axis below this dimension after max-size "
            "scaling. This is an opt-in printable-compactness probe and may distort shape."
        ),
    )
    parser.add_argument(
        "--mesh-max-bbox-aspect-ratio",
        type=float,
        default=0.0,
        help=(
            "If positive, anisotropically thicken small bounding-box axes until max_axis/min_axis is at most "
            "this ratio after size scaling. This is an opt-in STL calibration probe and may distort shape."
        ),
    )
    parser.add_argument(
        "--mesh-target-bbox-extents",
        type=parse_bbox_extents,
        default=None,
        help=(
            "Optional comma- or space-separated X,Y,Z STL-space bbox extents. When set, the normalized "
            "provider mesh is anisotropically scaled to these exact final extents before STL export."
        ),
    )
    parser.add_argument(
        "--mesh-target-faces",
        type=int,
        default=0,
        help=(
            "If positive, attempt quadric decimation to this face count after scaling. Environments without "
            "the optional Trimesh simplification backend leave the mesh unchanged."
        ),
    )
    parser.add_argument("--provider-arg", action="append", default=[])
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()
    output_mesh, output_stl = run_provider(args)
    print(f"mesh={output_mesh}")
    if output_stl is not None:
        print(f"stl={output_stl}")


if __name__ == "__main__":
    main()
