from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

SOURCE_SPLIT_NAMES = {"train", "test", "val", "validation"}


@dataclass(frozen=True)
class CameraSpec:
    azimuth_deg: float
    elevation_deg: float
    roll_deg: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RenderConfig:
    size: int = 256
    ortho_scale: float = 2.0
    background_rgb: tuple[float, float, float] = (0.91, 0.93, 0.96)
    ambient: float = 0.45
    diffuse: float = 0.55
    light_direction: tuple[float, float, float] = (0.35, -0.45, 0.82)


@dataclass
class RenderResult:
    rgb: np.ndarray
    depth: np.ndarray
    silhouette: np.ndarray


def scene_to_mesh(scene: trimesh.Scene, path: str | Path = "") -> trimesh.Trimesh:
    dumped = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    if isinstance(dumped, trimesh.Trimesh):
        mesh = dumped
    else:
        meshes = [
            geometry
            for geometry in dumped
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.vertices) and len(geometry.faces)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Scene has no renderable triangles: {path}")
    return mesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), process=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = scene_to_mesh(loaded, path)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"Unsupported mesh object loaded from {path}: {type(loaded)!r}")

    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Mesh has no renderable triangles: {path}")
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, target_extent: float = 1.45) -> trimesh.Trimesh:
    normalized = mesh.copy()
    bounds = normalized.bounds
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    normalized.apply_translation(-center)
    if extent > 0:
        normalized.apply_scale(target_extent / extent)
    return normalized


def make_procedural_mesh(index: int) -> trimesh.Trimesh:
    shape = index % 6
    if shape == 0:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.72)
    elif shape == 1:
        mesh = trimesh.creation.box(extents=(1.2, 0.78, 0.62))
    elif shape == 2:
        mesh = trimesh.creation.cylinder(radius=0.52, height=1.2, sections=48)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif shape == 3:
        mesh = trimesh.creation.cone(radius=0.58, height=1.2, sections=48)
        mesh.apply_translation([0, 0, -0.2])
    elif shape == 4:
        mesh = trimesh.creation.torus(
            major_radius=0.46,
            minor_radius=0.16,
            major_sections=64,
            minor_sections=16,
        )
    else:
        left = trimesh.creation.icosphere(subdivisions=2, radius=0.42)
        left.apply_translation([-0.32, 0, 0.0])
        right = trimesh.creation.icosphere(subdivisions=2, radius=0.42)
        right.apply_translation([0.32, 0, 0.08])
        stem = trimesh.creation.cylinder(radius=0.22, height=1.0, sections=32)
        stem.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        mesh = trimesh.util.concatenate([left, right, stem])
    return normalize_mesh(mesh)


def sample_camera(index: int, seed: int = 2026) -> CameraSpec:
    rng = np.random.default_rng(seed + index * 9973)
    return CameraSpec(
        azimuth_deg=float((index * 37 + rng.uniform(-12, 12)) % 360),
        elevation_deg=float(-18 + (index % 5) * 9 + rng.uniform(-4, 4)),
        roll_deg=float((index % 3 - 1) * 7 + rng.uniform(-2, 2)),
    )


def camera_from_value(value) -> CameraSpec | None:
    if value is None or value == "":
        return None
    if isinstance(value, CameraSpec):
        return value
    if isinstance(value, dict):
        return CameraSpec(
            azimuth_deg=float(value.get("azimuth_deg", 0.0)),
            elevation_deg=float(value.get("elevation_deg", 0.0)),
            roll_deg=float(value.get("roll_deg", 0.0)),
        )
    raise ValueError(f"Unsupported camera value: {value!r}")


def camera_transform(camera: CameraSpec | dict) -> np.ndarray:
    spec = camera_from_value(camera)
    if spec is None:
        return np.eye(4)
    return trimesh.transformations.euler_matrix(
        np.deg2rad(spec.elevation_deg),
        np.deg2rad(spec.azimuth_deg),
        np.deg2rad(spec.roll_deg),
        axes="sxyz",
    )


def mesh_in_render_frame(mesh: trimesh.Trimesh, camera: CameraSpec | dict | None) -> trimesh.Trimesh:
    transformed = normalize_mesh(mesh)
    if camera is not None:
        transformed.apply_transform(camera_transform(camera))
    return transformed


def iter_mesh_paths(asset_root: str | Path, asset_glob: str, limit: int | None, seed: int) -> Iterable[Path]:
    paths = sorted(Path(asset_root).glob(asset_glob))
    if not paths:
        raise ValueError(f"No mesh assets found under {asset_root} with glob {asset_glob!r}")
    rng = np.random.default_rng(seed)
    shuffled = [paths[index] for index in rng.permutation(len(paths))]
    if limit is not None:
        shuffled = shuffled[:limit]
    return shuffled


def mesh_category(path: str | Path, asset_root: str | Path | None = None) -> str:
    path = Path(path)
    root = Path(asset_root).resolve() if asset_root is not None else None
    parent = path.parent
    if parent.name.lower() in SOURCE_SPLIT_NAMES:
        parent = parent.parent
    if root is not None and parent.resolve() == root:
        return "mesh"
    return parent.name or "mesh"


def mesh_source_split(path: str | Path) -> str:
    split = Path(path).parent.name.lower()
    if split == "validation":
        return "val"
    if split in SOURCE_SPLIT_NAMES:
        return split
    return "unknown"


def iter_mesh_paths_balanced(asset_root: str | Path, asset_glob: str, limit: int | None, seed: int) -> Iterable[Path]:
    paths = sorted(Path(asset_root).glob(asset_glob))
    if not paths:
        raise ValueError(f"No mesh assets found under {asset_root} with glob {asset_glob!r}")

    grouped = defaultdict(list)
    for path in paths:
        grouped[mesh_category(path, asset_root)].append(path)

    rng = np.random.default_rng(seed)
    categories = sorted(grouped)
    categories = [categories[index] for index in rng.permutation(len(categories))]
    for category in categories:
        category_paths = grouped[category]
        grouped[category] = [category_paths[index] for index in rng.permutation(len(category_paths))]

    selected = []
    while categories and (limit is None or len(selected) < limit):
        next_categories = []
        for category in categories:
            category_paths = grouped[category]
            if category_paths:
                selected.append(category_paths.pop(0))
                next_categories.append(category)
                if limit is not None and len(selected) >= limit:
                    break
        categories = next_categories
    return selected


def render_mesh(
    mesh: trimesh.Trimesh,
    camera: CameraSpec,
    config: RenderConfig,
    base_color: tuple[int, int, int],
) -> RenderResult:
    mesh = mesh_in_render_frame(mesh, camera)
    rgb, depth, silhouette = _render_orthographic(mesh, config=config, base_color=base_color)
    return RenderResult(rgb=rgb, depth=depth, silhouette=silhouette)


def make_half_mask(size: int, side: str) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if side == "right":
        mask[:, size // 2 :] = 255
    elif side == "left":
        mask[:, : size // 2] = 255
    else:
        raise ValueError(f"Unsupported mask side: {side}")
    return mask


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    pixels[mask > 127] = 255
    return pixels


def _render_orthographic(
    mesh: trimesh.Trimesh,
    config: RenderConfig,
    base_color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = config.size
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    z_buffer = np.full((size, size), -np.inf, dtype=np.float32)
    rgb = np.ones((size, size, 3), dtype=np.float32)
    rgb[:] = np.asarray(config.background_rgb, dtype=np.float32)

    projected = np.empty((len(vertices), 3), dtype=np.float32)
    projected[:, 0] = (vertices[:, 0] / config.ortho_scale + 0.5) * (size - 1)
    projected[:, 1] = (1.0 - (vertices[:, 1] / config.ortho_scale + 0.5)) * (size - 1)
    projected[:, 2] = vertices[:, 2]

    light = np.asarray(config.light_direction, dtype=np.float32)
    light /= max(1e-8, float(np.linalg.norm(light)))
    base = np.asarray(base_color, dtype=np.float32) / 255.0

    for face, normal in zip(faces, face_normals):
        pts = projected[face]
        min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
        max_x = min(size - 1, int(np.ceil(np.max(pts[:, 0]))))
        min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
        max_y = min(size - 1, int(np.ceil(np.max(pts[:, 1]))))
        if max_x < min_x or max_y < min_y:
            continue

        p0, p1, p2 = pts
        denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(float(denom)) < 1e-8:
            continue

        xs, ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
        w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) / denom
        w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue

        z = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
        current = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (z > current)
        if not np.any(update):
            continue

        shade = config.ambient + config.diffuse * max(0.0, float(np.dot(normal, light)))
        face_color = np.clip(base * shade, 0.0, 1.0)
        current[update] = z[update]
        rgb_region = rgb[min_y : max_y + 1, min_x : max_x + 1]
        rgb_region[update] = face_color

    silhouette = np.isfinite(z_buffer)
    depth = np.ones((size, size), dtype=np.float32)
    if np.any(silhouette):
        z_values = z_buffer[silhouette]
        z_near = float(np.max(z_values))
        z_far = float(np.min(z_values))
        denom = max(1e-6, z_near - z_far)
        depth[silhouette] = (z_near - z_buffer[silhouette]) / denom

    return rgb, depth, silhouette
