from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from backend.benchmark.mesh_rendering import load_mesh, mesh_in_render_frame


DIRECT_MESH_METHODS = {"source-mesh-oracle", "external-image-to-mesh"}


def is_direct_mesh_method(method: str) -> bool:
    return method in DIRECT_MESH_METHODS


def resolve_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sample_mesh_path(sample: dict) -> Path | None:
    return resolve_existing_path(sample.get("mesh") or sample.get("asset_path"))


def direct_mesh_input_path(sample: dict, mode: str = "masked") -> Path:
    if mode == "full":
        key = "full_image"
    elif mode == "masked":
        key = "masked_image"
    else:
        raise ValueError(f"Unsupported direct mesh input mode: {mode}")
    path = resolve_existing_path(sample.get(key))
    if path is None:
        raise FileNotFoundError(f"Missing {key} for sample {sample.get('id', '')}: {sample.get(key, '')}")
    return path


def convert_mesh_to_stl(mesh_path: Path, stl_path: Path) -> Path:
    if mesh_path.resolve() == stl_path.resolve():
        return stl_path
    mesh = load_mesh(mesh_path)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Direct mesh output has no triangles: {mesh_path}")
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)
    return stl_path


def run_direct_mesh(sample: dict, method: str, output_dir: Path, args) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = direct_mesh_input_path(sample, getattr(args, "direct_mesh_input", "masked"))
    stl_path = output_dir / "output_model.stl"
    output_ext = str(getattr(args, "direct_mesh_output_ext", "glb") or "glb").lstrip(".")
    mesh_output_path = output_dir / f"output_mesh.{output_ext}"

    if method == "source-mesh-oracle":
        source_path = sample_mesh_path(sample)
        if source_path is None:
            raise FileNotFoundError(f"Sample {sample.get('id', '')} does not include an existing source mesh")
        mesh_output_path = output_dir / f"source_mesh{source_path.suffix or '.mesh'}"
        if sample.get("camera"):
            source_mesh = mesh_in_render_frame(load_mesh(source_path), sample.get("camera"))
            source_mesh.export(mesh_output_path)
        else:
            shutil.copy2(source_path, mesh_output_path)
        convert_mesh_to_stl(mesh_output_path, stl_path)
        return input_image, mesh_output_path, stl_path

    if method == "external-image-to-mesh":
        command_template = getattr(args, "direct_mesh_command", None)
        if not command_template:
            raise ValueError("--direct-mesh-command is required for external-image-to-mesh")
        values = {
            "input_image": str(input_image),
            "masked_image": str(direct_mesh_input_path(sample, "masked")),
            "full_image": str(direct_mesh_input_path(sample, "full")),
            "mask": str(resolve_existing_path(sample.get("mask")) or ""),
            "output_dir": str(output_dir),
            "output_mesh": str(mesh_output_path),
            "output_stl": str(stl_path),
            "sample_id": str(sample.get("id", "")),
            "method": method,
        }
        command = command_template.format(**values)
        subprocess.run(
            command,
            shell=True,
            check=True,
            timeout=max(1, int(getattr(args, "direct_mesh_timeout", 1800))),
        )
        if not mesh_output_path.exists() and stl_path.exists():
            return input_image, stl_path, stl_path
        if not mesh_output_path.exists():
            raise FileNotFoundError(
                f"External image-to-mesh command produced neither {mesh_output_path} nor {stl_path}"
            )
        if not stl_path.exists():
            convert_mesh_to_stl(mesh_output_path, stl_path)
        return input_image, mesh_output_path, stl_path

    raise ValueError(f"Unsupported direct mesh method: {method}")
