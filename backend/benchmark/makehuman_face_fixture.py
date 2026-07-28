"""Build and load a compact CC0 MakeHuman face benchmark fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


MAKEHUMAN_SOURCE_COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
MAKEHUMAN_SOURCE_URL = "https://github.com/makehumancommunity/makehuman"
BASE_MESH_RELATIVE_PATH = Path("makehuman/data/3dobjs/base.obj")
ASSET_LICENSE_RELATIVE_PATH = Path("LICENSE.ASSETS.md")
HEAD_MINIMUM_Y = 5.35
RENDER_GROUPS = frozenset(("body", "helper-l-eye", "helper-r-eye"))
FACE_PART_NAMES = (
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "nose",
    "mouth",
)


@dataclass(frozen=True)
class MakeHumanProfileSpec:
    name: str
    skin_tone: tuple[float, float, float]
    targets: tuple[tuple[str, float], ...]


MAKEHUMAN_PROFILES = (
    MakeHumanProfileSpec(
        name="caucasian_female_smile",
        skin_tone=(0.74, 0.50, 0.39),
        targets=(
            ("macrodetails/caucasian-female-young.target", 1.0),
            ("expression/units/caucasian/mouth-corner-puller.target", 0.55),
        ),
    ),
    MakeHumanProfileSpec(
        name="african_male_neutral",
        skin_tone=(0.34, 0.20, 0.15),
        targets=(("macrodetails/african-male-young.target", 1.0),),
    ),
    MakeHumanProfileSpec(
        name="asian_female_asymmetric",
        skin_tone=(0.66, 0.44, 0.32),
        targets=(
            ("macrodetails/asian-female-young.target", 1.0),
            ("expression/units/asian/eyebrows-left-up.target", 0.40),
            ("expression/units/asian/mouth-corner-puller.target", 0.30),
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _verify_source_commit(source_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("MakeHuman source must be a readable git checkout") from exc
    revision = completed.stdout.strip().lower()
    if revision != MAKEHUMAN_SOURCE_COMMIT:
        raise ValueError(
            "MakeHuman source revision mismatch: "
            f"expected {MAKEHUMAN_SOURCE_COMMIT}, got {revision or '<empty>'}"
        )
    return revision


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    face_groups: list[str] = []
    group = ""
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if stripped.startswith("v "):
                vertices.append([float(value) for value in stripped.split()[1:4]])
            elif stripped.startswith("g "):
                group = stripped.split(maxsplit=1)[1]
            elif stripped.startswith("f ") and group in RENDER_GROUPS:
                indices = [
                    int(value.split("/", 1)[0]) - 1
                    for value in stripped.split()[1:]
                ]
                for index in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[index], indices[index + 1]))
                    face_groups.append(group)
    vertex_array = np.asarray(vertices, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int32)
    group_array = np.asarray(face_groups, dtype="U24")
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not len(face_array):
        raise ValueError(f"Invalid MakeHuman base OBJ: {path}")
    return vertex_array, face_array, group_array


def _read_target(path: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    offsets: list[list[float]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 4:
                raise ValueError(f"Invalid target row in {path}: {stripped!r}")
            index = int(fields[0])
            if not 0 <= index < vertex_count:
                raise ValueError(f"Target vertex {index} is outside {vertex_count}: {path}")
            indices.append(index)
            offsets.append([float(value) for value in fields[1:]])
    return np.asarray(indices, dtype=np.int32), np.asarray(offsets, dtype=np.float32)


def _target_directory_digest(paths: list[Path], root: Path) -> dict:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return {
        "file_count": len(paths),
        "sha256": digest.hexdigest(),
    }


def _target_vertex_set(paths: list[Path], vertex_count: int) -> set[int]:
    indices: set[int] = set()
    for path in paths:
        target_indices, _ = _read_target(path, vertex_count)
        indices.update(int(value) for value in target_indices)
    return indices


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.save(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_makehuman_face_fixture(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    profiles: tuple[MakeHumanProfileSpec, ...] = MAKEHUMAN_PROFILES,
    fixture_filename: str = "makehuman_cc0_heads.npz",
) -> dict:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    profiles = tuple(profiles)
    if not profiles:
        raise ValueError("At least one MakeHuman profile is required")
    profile_names = [profile.name for profile in profiles]
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("MakeHuman profile names must be unique")
    if (
        Path(fixture_filename).name != fixture_filename
        or not fixture_filename.endswith(".npz")
    ):
        raise ValueError("fixture_filename must be a local .npz filename")
    source_revision = _verify_source_commit(source_root)
    base_path = source_root / BASE_MESH_RELATIVE_PATH
    license_path = source_root / ASSET_LICENSE_RELATIVE_PATH
    target_root = source_root / "makehuman/data/targets"
    if not base_path.exists() or not license_path.exists():
        raise FileNotFoundError("MakeHuman source checkout is incomplete")

    vertices, faces, face_groups = _read_obj(base_path)
    retained = np.all(vertices[faces, 1] >= HEAD_MINIMUM_Y, axis=1)
    faces = faces[retained]
    face_groups = face_groups[retained]
    original_indices = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[original_indices] = np.arange(len(original_indices), dtype=np.int32)
    head_faces = remap[faces]

    part_sources = {
        "left_eye": sorted((target_root / "eyes").glob("l-eye-*.target")),
        "right_eye": sorted((target_root / "eyes").glob("r-eye-*.target")),
        "eyebrows": sorted((target_root / "eyebrows").glob("*.target")),
        "nose": sorted((target_root / "nose").glob("*.target")),
        "mouth": sorted((target_root / "mouth").glob("*.target")),
    }
    if any(not paths for paths in part_sources.values()):
        missing = [name for name, paths in part_sources.items() if not paths]
        raise FileNotFoundError(f"Missing MakeHuman target directories: {missing}")

    original_to_head = {int(value): index for index, value in enumerate(original_indices)}
    weights: dict[str, np.ndarray] = {}
    for name in ("left_eye", "right_eye", "nose", "mouth"):
        selected = _target_vertex_set(part_sources[name], len(vertices))
        weight = np.zeros(len(original_indices), dtype=np.float32)
        for original in selected:
            if original in original_to_head:
                weight[original_to_head[original]] = 1.0
        weights[name] = weight

    brow_indices = _target_vertex_set(part_sources["eyebrows"], len(vertices))
    for name, positive_x in (("left_eyebrow", True), ("right_eyebrow", False)):
        weight = np.zeros(len(original_indices), dtype=np.float32)
        for original in brow_indices:
            if original in original_to_head and ((vertices[original, 0] >= 0) == positive_x):
                weight[original_to_head[original]] = 1.0
        weights[name] = weight

    eye_surface_weights = {}
    for name, group_name in (
        ("left_eye_surface", "helper-l-eye"),
        ("right_eye_surface", "helper-r-eye"),
    ):
        group_vertices = np.unique(faces[face_groups == group_name])
        weight = np.zeros(len(original_indices), dtype=np.float32)
        for original in group_vertices:
            weight[original_to_head[int(original)]] = 1.0
        eye_surface_weights[name] = weight
    weights["left_eye"] = np.maximum(weights["left_eye"], eye_surface_weights["left_eye_surface"])
    weights["right_eye"] = np.maximum(weights["right_eye"], eye_surface_weights["right_eye_surface"])

    arrays: dict[str, np.ndarray] = {
        "faces": head_faces.astype(np.int32),
        "original_vertex_indices": original_indices.astype(np.int32),
    }
    arrays.update({f"weight__{name}": value for name, value in weights.items()})
    arrays.update(
        {f"surface__{name}": value for name, value in eye_surface_weights.items()}
    )

    target_records = {}
    profile_records = []
    for profile in profiles:
        profile_vertices = vertices.copy()
        applied = []
        for relative, scale in profile.targets:
            target_path = target_root / relative
            target_indices, offsets = _read_target(target_path, len(vertices))
            profile_vertices[target_indices] += float(scale) * offsets
            record = {
                "path": target_path.relative_to(source_root).as_posix(),
                "scale": float(scale),
                "sha256": _sha256(target_path),
            }
            applied.append(record)
            target_records[record["path"]] = record["sha256"]
        head_vertices = profile_vertices[original_indices].astype(np.float32)
        arrays[f"vertices__{profile.name}"] = head_vertices
        profile_records.append(
            {
                "name": profile.name,
                "skin_tone": list(profile.skin_tone),
                "targets": applied,
                "vertices_sha256": _array_sha256(head_vertices),
                "bounds": head_vertices.min(axis=0).tolist()
                + head_vertices.max(axis=0).tolist(),
            }
        )

    fixture_path = output_dir / fixture_filename
    _deterministic_npz(fixture_path, arrays)
    license_output = output_dir / "LICENSE.ASSETS.md"
    license_output.write_bytes(license_path.read_bytes())
    manifest = {
        "schema_version": 1,
        "license": "CC0-1.0",
        "source": {
            "repository": MAKEHUMAN_SOURCE_URL,
            "commit": source_revision,
            "base_mesh_path": BASE_MESH_RELATIVE_PATH.as_posix(),
            "base_mesh_sha256": _sha256(base_path),
            "asset_license_path": ASSET_LICENSE_RELATIVE_PATH.as_posix(),
            "asset_license_sha256": _sha256(license_path),
            "profile_target_files": target_records,
            "part_target_directories": {
                name: _target_directory_digest(paths, source_root)
                for name, paths in part_sources.items()
            },
        },
        "derivation": {
            "head_minimum_y": HEAD_MINIMUM_Y,
            "render_groups": sorted(RENDER_GROUPS),
            "triangulation": "fan triangulation preserving OBJ face order",
            "coordinate_units": "MakeHuman native decimeters",
            "axes": "+x anatomical left, +y up, +z forward",
        },
        "fixture": {
            "file": fixture_path.name,
            "sha256": _sha256(fixture_path),
            "size_bytes": fixture_path.stat().st_size,
            "vertex_count": len(original_indices),
            "face_count": len(head_faces),
            "faces_sha256": _array_sha256(head_faces),
            "part_support_vertices": {
                name: int(np.count_nonzero(weights[name])) for name in FACE_PART_NAMES
            },
            "profiles": profile_records,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "asset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_makehuman_face_fixture(asset_dir: str | Path) -> dict:
    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / "asset.json").read_text(encoding="utf-8"))
    fixture_path = asset_dir / manifest["fixture"]["file"]
    if _sha256(fixture_path) != manifest["fixture"]["sha256"]:
        raise ValueError("MakeHuman fixture checksum mismatch")
    with np.load(fixture_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    if _array_sha256(faces.astype(np.int32)) != manifest["fixture"]["faces_sha256"]:
        raise ValueError("MakeHuman fixture topology checksum mismatch")
    profiles = {}
    for record in manifest["fixture"]["profiles"]:
        name = record["name"]
        vertices = np.asarray(arrays[f"vertices__{name}"], dtype=np.float64)
        if _array_sha256(vertices.astype(np.float32)) != record["vertices_sha256"]:
            raise ValueError(f"MakeHuman profile checksum mismatch: {name}")
        profiles[name] = {
            "mesh": trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            "skin_tone": tuple(float(value) for value in record["skin_tone"]),
            "record": record,
        }
    weights = {
        name: np.asarray(arrays[f"weight__{name}"], dtype=np.float32)
        for name in FACE_PART_NAMES
    }
    surfaces = {
        name: np.asarray(arrays[f"surface__{name}"], dtype=np.float32)
        for name in ("left_eye_surface", "right_eye_surface")
    }
    expected_shape = (manifest["fixture"]["vertex_count"],)
    if any(values.shape != expected_shape for values in (*weights.values(), *surfaces.values())):
        raise ValueError("MakeHuman fixture weight shape mismatch")
    if any(not np.any(values >= 0.5) for values in weights.values()):
        raise ValueError("MakeHuman fixture contains an empty face part")
    return {
        "manifest": manifest,
        "profiles": profiles,
        "part_weights": weights,
        "surface_weights": surfaces,
    }


def make_profile_vertex_colors(
    vertices: np.ndarray,
    skin_tone: tuple[float, float, float],
    part_weights: dict[str, np.ndarray],
    surface_weights: dict[str, np.ndarray],
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    color = np.broadcast_to(np.asarray(skin_tone, dtype=np.float64), vertices.shape).copy()
    minimum = vertices.min(axis=0)
    span = np.maximum(vertices.max(axis=0) - minimum, 1e-6)
    normalized = (vertices - minimum) / span
    pore = (
        0.025 * np.sin(91.0 * normalized[:, 0] + 47.0 * normalized[:, 1])
        + 0.018 * np.sin(73.0 * normalized[:, 1] - 59.0 * normalized[:, 2])
    )
    color *= 1.0 + pore[:, None]

    def localized_visual_weight(
        values: np.ndarray,
        lower_percentiles: tuple[float, float, float],
        upper_percentiles: tuple[float, float, float],
    ) -> np.ndarray:
        selected = np.asarray(values) >= 0.5
        coordinates = vertices[selected]
        if not len(coordinates):
            return np.zeros((len(vertices), 1), dtype=np.float64)
        lower = np.asarray(
            [
                np.percentile(coordinates[:, axis], lower_percentiles[axis])
                for axis in range(3)
            ]
        )
        upper = np.asarray(
            [
                np.percentile(coordinates[:, axis], upper_percentiles[axis])
                for axis in range(3)
            ]
        )
        localized = selected & np.all((vertices >= lower) & (vertices <= upper), axis=1)
        return localized.astype(np.float64)[:, None]

    mouth = localized_visual_weight(
        part_weights["mouth"], (5.0, 5.0, 20.0), (95.0, 95.0, 100.0)
    )
    lip_color = np.asarray((0.42, 0.16, 0.14), dtype=np.float64)
    color = color * (1.0 - 0.48 * mouth) + lip_color * (0.48 * mouth)
    brows = localized_visual_weight(
        np.maximum(part_weights["left_eyebrow"], part_weights["right_eyebrow"]),
        (2.0, 45.0, 20.0),
        (98.0, 78.0, 100.0),
    )
    brow_source = np.maximum(
        part_weights["left_eyebrow"], part_weights["right_eyebrow"]
    ) >= 0.5
    brow_x = vertices[brow_source, 0]
    if len(brow_x):
        brow_center = float(np.median(brow_x))
        brow_width = max(float(np.ptp(brow_x)), 1e-6)
        brows *= (np.abs(vertices[:, 0] - brow_center) > 0.10 * brow_width)[:, None]
    hair_color = np.asarray((0.075, 0.052, 0.040), dtype=np.float64)
    color = color * (1.0 - 0.72 * brows) + hair_color * (0.72 * brows)

    eye_surface = np.maximum(
        surface_weights["left_eye_surface"], surface_weights["right_eye_surface"]
    )[:, None]
    eye_color = np.asarray((0.86, 0.87, 0.82), dtype=np.float64)
    color = color * (1.0 - 0.92 * eye_surface) + eye_color * (0.92 * eye_surface)

    for surface_name in ("left_eye_surface", "right_eye_surface"):
        selected = np.asarray(surface_weights[surface_name]) >= 0.5
        eye_vertices = vertices[selected]
        if not len(eye_vertices):
            continue
        center = np.median(eye_vertices[:, :2], axis=0)
        extent = np.maximum(np.ptp(eye_vertices[:, :2], axis=0), 1e-6)
        radius = np.square((vertices[:, 0] - center[0]) / (0.22 * extent[0]))
        radius += np.square((vertices[:, 1] - center[1]) / (0.24 * extent[1]))
        front = vertices[:, 2] >= np.percentile(eye_vertices[:, 2], 62.0)
        iris = selected & front & (radius <= 1.0)
        pupil = selected & front & (radius <= 0.20)
        color[iris] = 0.25 * color[iris] + 0.75 * np.asarray((0.24, 0.14, 0.08))
        color[pupil] = np.asarray((0.025, 0.018, 0.015))

    hair_threshold = 0.82 + 0.07 * (1.0 - np.abs(2.0 * normalized[:, 0] - 1.0))
    hairline = (
        (normalized[:, 1] > hair_threshold)
        & (normalized[:, 2] < 0.78)
    )[:, None]
    color = np.where(hairline, 0.82 * hair_color + 0.18 * color, color)
    return np.clip(color, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    manifest = build_makehuman_face_fixture(args.source_root, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
