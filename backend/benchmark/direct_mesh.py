from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import load_mesh, mesh_in_render_frame
from backend.pic_to_3d import _masked_edit_image


DIRECT_MESH_METHODS = {"source-mesh-oracle", "external-image-to-mesh"}
DIRECT_MESH_INPUT_MODES = ("masked", "full", "mirror", "biharmonic")
MESH_REPAIR_MODES = ("none", "basic", "convex-hull", "printable")


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


def direct_mesh_input_path(sample: dict, mode: str = "masked", output_dir: Path | None = None) -> Path:
    if mode == "full":
        key = "full_image"
    elif mode == "masked":
        key = "masked_image"
    elif mode in ("mirror", "biharmonic"):
        if output_dir is None:
            raise ValueError(f"direct mesh input mode {mode!r} requires an output directory")
        masked_path = direct_mesh_input_path(sample, "masked")
        mask_path = resolve_existing_path(sample.get("mask"))
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for sample {sample.get('id', '')}: {sample.get('mask', '')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        prefill_path = output_dir / f"direct_mesh_input_{mode}.png"
        image = Image.open(masked_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        _masked_edit_image(image, mask, mode).save(prefill_path)
        return prefill_path
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


def _finite_positive_volume(mesh) -> bool:
    try:
        volume = float(mesh.volume)
    except Exception:
        return False
    return math.isfinite(volume) and volume > 0


def mesh_is_printable_volume(mesh) -> bool:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or not np.all(extents > 0):
        return False
    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = math.inf
    return bool(
        len(mesh.vertices)
        and len(mesh.faces)
        and mesh.is_watertight
        and mesh.is_volume
        and mesh.is_winding_consistent
        and component_count == 1
        and _finite_positive_volume(mesh)
    )


def _update_faces(mesh, mask) -> None:
    if mask is None:
        return
    if hasattr(mesh, "update_faces"):
        mesh.update_faces(mask)
    else:
        mesh.faces = mesh.faces[mask]


def _component_score(mesh) -> tuple[float, float, int]:
    try:
        volume = abs(float(mesh.volume))
    except Exception:
        volume = 0.0
    try:
        area = float(mesh.area)
    except Exception:
        area = 0.0
    volume = volume if math.isfinite(volume) else 0.0
    area = area if math.isfinite(area) else 0.0
    return (volume, area, int(len(mesh.faces)))


def _largest_component(mesh):
    try:
        components = [
            component
            for component in mesh.split(only_watertight=False)
            if len(component.vertices) and len(component.faces)
        ]
    except Exception:
        components = []
    if len(components) <= 1:
        return mesh
    return max(components, key=_component_score)


def _clean_mesh(mesh):
    import trimesh

    cleaned = mesh.copy()
    if hasattr(cleaned, "remove_infinite_values"):
        cleaned.remove_infinite_values()
    if hasattr(cleaned, "unique_faces"):
        _update_faces(cleaned, cleaned.unique_faces())
    elif hasattr(cleaned, "remove_duplicate_faces"):
        cleaned.remove_duplicate_faces()
    if hasattr(cleaned, "nondegenerate_faces"):
        _update_faces(cleaned, cleaned.nondegenerate_faces())
    elif hasattr(cleaned, "remove_degenerate_faces"):
        cleaned.remove_degenerate_faces()
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned.process(validate=True)
    cleaned = _largest_component(cleaned).copy()
    trimesh.repair.fill_holes(cleaned)
    trimesh.repair.fix_winding(cleaned)
    trimesh.repair.fix_normals(cleaned)
    trimesh.repair.fix_inversion(cleaned)
    cleaned.process(validate=True)
    return _largest_component(cleaned).copy()


def _convex_hull_mesh(mesh):
    import trimesh

    source = _largest_component(mesh).copy()
    try:
        hull = source.convex_hull
        if callable(hull):
            hull = hull()
    except Exception:
        hull = source.bounding_box.to_mesh()
    if not len(hull.vertices) or not len(hull.faces):
        hull = trimesh.creation.box(extents=np.maximum(np.asarray(source.extents), 1e-6))
        hull.apply_translation(source.bounds.mean(axis=0))
    hull.process(validate=True)
    trimesh.repair.fix_winding(hull)
    trimesh.repair.fix_normals(hull)
    trimesh.repair.fix_inversion(hull)
    return hull


def repair_mesh_for_printable_stl(mesh_path: Path, output_path: Path, mode: str = "printable") -> Path:
    if mode not in MESH_REPAIR_MODES or mode == "none":
        raise ValueError(f"Unsupported mesh repair mode: {mode}")
    mesh = load_mesh(mesh_path)
    if mode == "convex-hull":
        repaired = _convex_hull_mesh(mesh)
    else:
        repaired = _clean_mesh(mesh)
        if mode == "printable" and not mesh_is_printable_volume(repaired):
            repaired = _convex_hull_mesh(repaired)
    if not len(repaired.vertices) or not len(repaired.faces):
        raise ValueError(f"Mesh repair produced no triangles: {mesh_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_path)
    return output_path


def run_direct_mesh(sample: dict, method: str, output_dir: Path, args) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = direct_mesh_input_path(sample, getattr(args, "direct_mesh_input", "masked"), output_dir)
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
