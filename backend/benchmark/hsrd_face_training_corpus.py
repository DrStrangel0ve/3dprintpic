"""Build a camera-depth face corpus from pinned HSRD-100 LOD1 scans."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image
import trimesh

from backend.benchmark.c3i_synface_corpus import (
    _border_rgb,
    _resize_and_place,
    _select_face_region,
    _transform_bbox,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.face_depth_refinement import (
    FACE_LANDMARKER_LICENSE,
    FACE_LANDMARKER_MODEL_SHA256,
    FACE_PART_NAMES,
    YUNET_MODEL_REVISION,
    YUNET_MODEL_SHA256,
)


HSRD_REPOSITORY = "digitalrealitylab/HSRD-100"
HSRD_REPOSITORY_URL = "https://huggingface.co/datasets/digitalrealitylab/HSRD-100"
HSRD_REVISION = "9cc5e9c138dd310dc88c95c80e5db0a3ecb7971a"
HSRD_LICENSE = "CC BY 4.0"
HSRD_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
HSRD_ATTRIBUTION = "HSRD-100, Digital Reality Lab (2025)"

MANIFEST_HASHES = {
    "README.md": "359b5b10fcb8deb7dded4f65299103c34e4e4c2ec8a6d7e9b5a1bbef2a7e8c40",
    "manifest/files.csv": "12f40c0c8aa049e73bc31e47a6235786cd758b9f391302049b82ce0875bc445e",
    "manifest/persons.jsonl": "d98866b4d33cb5003220c84048db460825211b171f653015c40f331e1a4dd107",
    "manifest/poses.jsonl": "ceb963e6a5fe5c0421d99ae57c2717acbec1bb1a1fa881c96e317c52d0f65a1e",
}

RENDER_SIZE = 384
OUTPUT_SIZE = 256
PERSPECTIVE_FOV_Y_DEGREES = 32.0
CAMERA_DISTANCE = 2.6
HEAD_CROP_HEIGHT_METERS = 0.43
TARGET_FACE_HEIGHTS = (74, 75)
CAMERA_YAWS_DEGREES = (-35.0, -30.0, 30.0, 35.0)
MINIMUM_SOURCE_PART_PIXELS = 20
MINIMUM_VIEW_DETECTION_RATE = 0.80
FACE_SELECTION_FUNCTION_SHA256 = hashlib.sha256(
    inspect.getsource(_select_face_region).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class HSRDIdentitySpec:
    person_id: str
    pose_id: str
    split: str
    archive_bytes: int
    archive_sha256: str

    @property
    def archive_relative_path(self) -> Path:
        return (
            Path("data")
            / self.person_id
            / self.pose_id
            / "scans"
            / f"{self.pose_id}-Scan-LOD1.zip"
        )

    @property
    def obj_name(self) -> str:
        return f"{self.pose_id}_LOD1.obj"


IDENTITY_SPECS = (
    HSRDIdentitySpec(
        "HSR0015",
        "HSR0015-Body-032",
        "train",
        99_931_520,
        "e509b1e91bb33c35b85d36fb7967aaa670d096290e6de8d47d8d6bec43049bb7",
    ),
    HSRDIdentitySpec(
        "HSR0023",
        "HSR0023-Body-017",
        "train",
        94_203_595,
        "ff08db840cff8e57cb99a66b559f4f3b4fe7bef2636bc9926db0f0144ade5c38",
    ),
    HSRDIdentitySpec(
        "HSR0027",
        "HSR0027-Body-025",
        "train",
        90_478_336,
        "c23db1c8253a6f32cb0706466061946c7a405f9427078c44b7714fc86090b7df",
    ),
    HSRDIdentitySpec(
        "HSR0028",
        "HSR0028-Body-052",
        "train",
        89_980_310,
        "f0fc8772dd8fa54310538f30944d3a55f688e5bf659cc4e54fe650d0cad4e5c2",
    ),
    HSRDIdentitySpec(
        "HSR0040",
        "HSR0040-Body-023",
        "train",
        99_466_743,
        "1e1c09383107444f83c7dd83a13cae1fb866dccc0ed35546ab166b2b24b8a130",
    ),
    HSRDIdentitySpec(
        "HSR0042",
        "HSR0042-Body-018",
        "train",
        67_042_943,
        "e6465892644894ddd43ca0205327c33d8f0aee3994a66bfd53f70198e4c4f78f",
    ),
    HSRDIdentitySpec(
        "HSR0081",
        "HSR0081-Body-020",
        "validation",
        95_188_928,
        "c587c11ab2f734ab01757070cc005b506533316b1f56bb36b4ca31b61526a37f",
    ),
    HSRDIdentitySpec(
        "HSR0102",
        "HSR0102-Body-034",
        "validation",
        88_991_594,
        "fc97639cd11ddf1e7d8677d1e2c6ca034c697c6b8fc91582c47aa438bf4cc501",
    ),
    HSRDIdentitySpec(
        "HSR0112",
        "HSR0112-Body-044",
        "sealed",
        71_530_822,
        "16193210750b18c59da9e33d1bb97e6f655a69eca4742d23c64122049dbd1736",
    ),
)

EXCLUDED_IDENTITIES = {
    "HSR0161": (
        "All ten published poses have a hat covering the cranial surface; "
        "using one as face-shape supervision would encode the hat as anatomy."
    )
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_specs(
    pose_ids: tuple[str, ...] | list[str] | None = None,
    limit: int | None = None,
) -> tuple[HSRDIdentitySpec, ...]:
    if limit is not None and limit < 1:
        raise ValueError("HSRD identity limit must be positive")
    specs = IDENTITY_SPECS
    if pose_ids:
        requested = tuple(str(value) for value in pose_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("HSRD pose selection contains duplicates")
        by_pose = {spec.pose_id: spec for spec in IDENTITY_SPECS}
        missing = sorted(set(requested) - set(by_pose))
        if missing:
            raise ValueError("Unknown HSRD pose IDs: " + ", ".join(missing))
        specs = tuple(by_pose[pose_id] for pose_id in requested)
    if limit is not None:
        specs = specs[:limit]
    return specs


def hsrd_preflight(
    manifest_root: str | Path,
    dataset_root: str | Path,
    *,
    specs: tuple[HSRDIdentitySpec, ...] = IDENTITY_SPECS,
) -> dict:
    manifest_root = Path(manifest_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    manifest_files = {}
    checks = {}
    for relative, expected_hash in MANIFEST_HASHES.items():
        path = manifest_root / Path(relative)
        actual_hash = _sha256(path) if path.is_file() else None
        manifest_files[relative] = {
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        checks[f"manifest:{relative}"] = actual_hash == expected_hash

    archives = []
    for spec in specs:
        path = dataset_root / spec.archive_relative_path
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_hash = _sha256(path) if exists and actual_bytes == spec.archive_bytes else None
        archives.append(
            {
                **asdict(spec),
                "relative_path": spec.archive_relative_path.as_posix(),
                "path": str(path),
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_hash,
            }
        )
        checks[f"archive:{spec.pose_id}:bytes"] = actual_bytes == spec.archive_bytes
        checks[f"archive:{spec.pose_id}:sha256"] = actual_hash == spec.archive_sha256

    split_people = {
        split: sorted(spec.person_id for spec in specs if spec.split == split)
        for split in ("train", "validation", "sealed")
    }
    disjoint = all(
        not (set(split_people[left]) & set(split_people[right]))
        for left, right in (("train", "validation"), ("train", "sealed"), ("validation", "sealed"))
    )
    checks["identity_splits_disjoint"] = disjoint
    checks["selected_people_unique"] = len({spec.person_id for spec in specs}) == len(specs)
    checks["selected_poses_unique"] = len({spec.pose_id for spec in specs}) == len(specs)
    return {
        "schema_version": 1,
        "provider": "hsrd100-lod1-camera-depth",
        "repository": HSRD_REPOSITORY,
        "repository_url": HSRD_REPOSITORY_URL,
        "revision": HSRD_REVISION,
        "license": HSRD_LICENSE,
        "license_url": HSRD_LICENSE_URL,
        "attribution": HSRD_ATTRIBUTION,
        "manifest_files": manifest_files,
        "archives": archives,
        "identity_splits": split_people,
        "checks": checks,
        "ready": all(checks.values()),
    }


def _safe_extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(
                    f"HSRD archive contains an unsafe path: {member.filename!r}"
                )
        archive.extractall(root)


def _archive_required_paths(
    output_dir: Path, spec: HSRDIdentitySpec
) -> tuple[tuple[str, Path], ...]:
    return (
        (spec.obj_name, output_dir / spec.obj_name),
        (
            f"{spec.pose_id}_LOD1_u0_v0_diffuse.png",
            output_dir / f"{spec.pose_id}_LOD1_u0_v0_diffuse.png",
        ),
        ("person_metadata.json", output_dir / "person_metadata.json"),
        ("pose_metadata.json", output_dir / "pose_metadata.json"),
    )


def _verify_extracted_members(
    archive_path: Path,
    output_dir: Path,
    spec: HSRDIdentitySpec,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        members = {member.filename: member for member in archive.infolist()}
        for member_name, path in _archive_required_paths(output_dir, spec):
            member = members.get(member_name)
            if member is None:
                raise ValueError(
                    f"HSRD archive {spec.pose_id} omitted required file {member_name}"
                )
            if not path.is_file():
                raise ValueError(
                    f"HSRD extraction {spec.pose_id} omitted required file {member_name}"
                )
            digest = hashlib.sha256()
            with archive.open(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != member.file_size or _sha256(path) != digest.hexdigest():
                raise ValueError(
                    f"HSRD extracted file {member_name} differs from the pinned archive"
                )


def extract_identity_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    spec: HSRDIdentitySpec,
) -> Path:
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    if archive_path.stat().st_size != spec.archive_bytes:
        raise ValueError(f"HSRD archive byte count changed for {spec.pose_id}")
    if _sha256(archive_path) != spec.archive_sha256:
        raise ValueError(f"HSRD archive hash changed for {spec.pose_id}")
    obj_path = output_dir / spec.obj_name
    if not obj_path.is_file():
        _safe_extract_archive(archive_path, output_dir)
    _verify_extracted_members(archive_path, output_dir, spec)
    return obj_path


def _obj_geometry_and_colors(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    colors = []
    faces = []
    with Path(path).open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) < 7:
                    raise ValueError("HSRD OBJ vertex omitted direct RGB values")
                vertices.append(tuple(float(value) for value in fields[1:4]))
                colors.append(tuple(float(value) for value in fields[4:7]))
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) != 3:
                    raise ValueError("HSRD LOD1 OBJ contains a non-triangle face")
                indices = []
                for field in fields:
                    vertex_index = int(field.split("/", 1)[0])
                    if vertex_index <= 0:
                        raise ValueError(
                            "HSRD LOD1 OBJ uses unsupported relative face indices"
                        )
                    indices.append(vertex_index - 1)
                faces.append(tuple(indices))
            else:
                continue
    vertex_array = np.asarray(vertices, dtype=np.float64)
    color_array = np.asarray(colors, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise ValueError("HSRD OBJ contains no valid vertices")
    if color_array.shape != vertex_array.shape:
        raise ValueError("HSRD OBJ vertex/color counts disagree")
    if not np.all(np.isfinite(vertex_array)) or not np.all(np.isfinite(color_array)):
        raise ValueError("HSRD OBJ vertices or colors are non-finite")
    if np.min(color_array) < 0.0 or np.max(color_array) > 1.0:
        raise ValueError("HSRD OBJ direct RGB values are outside [0, 1]")
    if face_array.ndim != 2 or face_array.shape[1:] != (3,):
        raise ValueError("HSRD OBJ contains no valid triangle faces")
    if np.any(face_array < 0) or np.any(face_array >= len(vertex_array)):
        raise ValueError("HSRD OBJ faces reference invalid vertices")
    return vertex_array, face_array, color_array


def _obj_vertices_and_colors(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, _, colors = _obj_geometry_and_colors(path)
    return vertices, colors


def load_hsrd_head(path: str | Path) -> tuple[trimesh.Trimesh, np.ndarray, dict]:
    path = Path(path)
    vertices, faces, vertex_colors = _obj_geometry_and_colors(path)
    if len(vertices) < 10_000 or len(faces) < 20_000:
        raise ValueError("HSRD LOD1 mesh is unexpectedly coarse")
    source = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    top = float(np.max(vertices[:, 2]))
    threshold = top - HEAD_CROP_HEIGHT_METERS
    keep_faces = np.all(vertices[faces, 2] >= threshold, axis=1)
    retained_faces = faces[keep_faces]
    used, inverse = np.unique(retained_faces, return_inverse=True)
    if len(used) < 2_000 or len(retained_faces) < 4_000:
        raise ValueError("HSRD head crop retained insufficient geometry")

    native = vertices[used]
    # RealityCapture exports Z-up scans. The published frontal view faces +Y.
    render_vertices = np.stack(
        (native[:, 0], native[:, 2], native[:, 1]),
        axis=1,
    )
    head = trimesh.Trimesh(
        vertices=render_vertices,
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    diagnostics = {
        "source_vertices": int(len(vertices)),
        "source_faces": int(len(faces)),
        "head_vertices": int(len(head.vertices)),
        "head_faces": int(len(head.faces)),
        "native_bounds_meters": source.bounds.tolist(),
        "native_top_meters": top,
        "native_head_threshold_meters": threshold,
        "native_to_render_axes": ["+x", "+z", "+y"],
        "source_degenerate_faces": int(np.count_nonzero(source.area_faces <= 1e-12)),
        "head_degenerate_faces": int(np.count_nonzero(head.area_faces <= 1e-12)),
    }
    return head, vertex_colors[used], diagnostics


def _render_identity(
    mesh: trimesh.Trimesh,
    colors: np.ndarray,
    yaw_degrees: float,
):
    return render_mesh(
        mesh,
        CameraSpec(float(yaw_degrees), 0.0),
        RenderConfig(
            size=RENDER_SIZE,
            projection="perspective",
            perspective_fov_y_deg=PERSPECTIVE_FOV_Y_DEGREES,
            camera_distance=CAMERA_DISTANCE,
            background_rgb=(0.30, 0.30, 0.30),
            ambient=0.75,
            diffuse=0.25,
            specular=0.0,
            light_direction=(0.35, -0.45, 0.82),
        ),
        (180, 120, 100),
        vertex_colors=colors,
    )


def _resize_valid_field(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    scale: float,
    dimension: int,
    center_xy: tuple[float, float],
    output_center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    resized, transform = _resize_and_place(
        np.where(valid, values, 0.0).astype(np.float32),
        scale=scale,
        dimension=dimension,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_NEAREST,
        fill_value=0.0,
    )
    support, support_transform = _resize_and_place(
        valid.astype(np.float32),
        scale=scale,
        dimension=dimension,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_NEAREST,
        fill_value=0.0,
    )
    if support_transform != transform:
        raise AssertionError("HSRD field/support transforms disagree")
    output = np.full((dimension, dimension), np.nan, dtype=np.float32)
    stable = support > 0.999
    output[stable] = resized[stable]
    return output, stable, transform


def _transformed_intrinsics(transform: dict) -> np.ndarray:
    resized_height, resized_width = (
        int(value) for value in transform["resized_shape"]
    )
    if resized_height < 1 or resized_width < 1:
        raise ValueError("HSRD resize transform has an empty shape")
    scale_x = resized_width / float(RENDER_SIZE)
    scale_y = resized_height / float(RENDER_SIZE)
    origin_x, origin_y = (float(value) for value in transform["origin_xy"])
    focal_source = 0.5 * (RENDER_SIZE - 1) / math.tan(
        math.radians(PERSPECTIVE_FOV_Y_DEGREES) / 2.0
    )
    principal = 0.5 * (RENDER_SIZE - 1)
    return np.asarray(
        (
            (
                focal_source * scale_x,
                0.0,
                origin_x + (principal + 0.5) * scale_x - 0.5,
            ),
            (
                0.0,
                focal_source * scale_y,
                origin_y + (principal + 0.5) * scale_y - 0.5,
            ),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detector_provenance(region: dict, errors: list[str]) -> dict:
    name = str(region.get("detector") or "unknown")
    models = {}
    if "mediapipe" in name.lower():
        models["face_landmarker"] = {
            "sha256": FACE_LANDMARKER_MODEL_SHA256,
            "license": FACE_LANDMARKER_LICENSE,
        }
    if "yunet" in name.lower():
        models["yunet"] = {
            "revision": YUNET_MODEL_REVISION,
            "sha256": YUNET_MODEL_SHA256,
        }
    return {
        "name": name,
        "landmark_count": int(region.get("landmark_count") or 0),
        "errors": list(errors),
        "models": models,
        "packages": {
            "mediapipe": _package_version("mediapipe"),
            "opencv_python": cv2.__version__,
        },
        "selection_function_sha256": FACE_SELECTION_FUNCTION_SHA256,
    }


def _view_coverage_contract(rows: list[dict], specs: tuple[HSRDIdentitySpec, ...]) -> dict:
    expected = len(specs) * len(CAMERA_YAWS_DEGREES)
    accepted = len(rows) // len(TARGET_FACE_HEIGHTS)
    required_splits = {spec.split for spec in specs}
    signs_by_split = {
        split: sorted(
            {
                -1 if float(row["rendering"]["camera_yaw_degrees"]) < 0 else 1
                for row in rows
                if row["split"] == split
            }
        )
        for split in required_splits
    }
    rate = float(accepted / max(expected, 1))
    return {
        "expected_view_count": expected,
        "accepted_view_count": accepted,
        "view_detection_rate": rate,
        "minimum_view_detection_rate": MINIMUM_VIEW_DETECTION_RATE,
        "yaw_signs_by_split": signs_by_split,
        "rate_passed": rate >= MINIMUM_VIEW_DETECTION_RATE,
        "split_yaw_sign_coverage_passed": all(
            signs == [-1, 1] for signs in signs_by_split.values()
        ),
    }


def _camera_space_normals(
    camera_z: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    values = np.asarray(camera_z, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError("HSRD camera intrinsics must be 3x3")
    rows, columns = np.indices(values.shape, dtype=np.float32)
    points = np.stack(
        (
            (columns - intrinsics[0, 2]) * values / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * values / intrinsics[1, 1],
            values,
        ),
        axis=-1,
    )
    interior = np.zeros_like(valid)
    interior[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    tangent_u = np.zeros_like(points)
    tangent_v = np.zeros_like(points)
    tangent_u[:, 1:-1] = points[:, 2:] - points[:, :-2]
    tangent_v[1:-1] = points[2:] - points[:-2]
    normals = np.cross(tangent_v, tangent_u)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    normals[~interior] = np.nan
    return normals.astype(np.float32)


def _write_array(path: Path, root: Path, values: np.ndarray) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        Image.fromarray(values).save(path)
    else:
        np.save(path, values, allow_pickle=False)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _yaw_token(yaw_degrees: float) -> str:
    return ("p" if yaw_degrees >= 0 else "m") + str(abs(int(round(yaw_degrees))))


def _exact_face_height_scale(
    bbox: list[int],
    target_height: int,
    *,
    center_xy: tuple[float, float],
    output_center_xy: tuple[float, float],
) -> float:
    x0, y0, x1, y1 = bbox
    del x0, x1
    source_height = y1 - y0
    if source_height <= 0:
        raise ValueError("HSRD source face bbox is empty")
    base = float(target_height / source_height)
    candidates = np.linspace(base * 0.97, base * 1.03, 1201)
    exact = []
    for scale in candidates:
        origin_y = int(round(output_center_xy[1] - center_xy[1] * scale))
        transformed_height = int(round(origin_y + y1 * scale)) - int(
            round(origin_y + y0 * scale)
        )
        if transformed_height == target_height:
            exact.append(float(scale))
    if not exact:
        raise ValueError(
            f"HSRD cannot place an exact {target_height}px face from {source_height}px"
        )
    return min(exact, key=lambda value: abs(value - base))


def _emit_row(
    rendered,
    spec: HSRDIdentitySpec,
    yaw_degrees: float,
    face_height: int,
    output_dir: Path,
    mesh_diagnostics: dict,
    *,
    detected_region: dict | None = None,
    detector_errors: list[str] | None = None,
) -> dict:
    source_high = np.clip(rendered.rgb * 255.0, 0, 255).astype(np.uint8)
    if detected_region is None:
        region, detected_errors = _select_face_region(source_high)
        detector_errors = detected_errors
    else:
        region = detected_region
        detector_errors = list(detector_errors or ())
    x0, y0, x1, y1 = (int(value) for value in region["bbox"])
    detected_height = y1 - y0
    if detected_height < 90:
        raise ValueError(
            f"HSRD source face is too small before reduction: {detected_height}px"
        )
    center_xy = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    output_center_xy = (0.5 * OUTPUT_SIZE, 0.46 * OUTPUT_SIZE)
    scale = _exact_face_height_scale(
        region["bbox"],
        face_height,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
    )
    source, transform = _resize_and_place(
        source_high,
        scale=scale,
        dimension=OUTPUT_SIZE,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_AREA,
        fill_value=_border_rgb(source_high),
    )
    source_camera_z = -np.asarray(rendered.surface_z, dtype=np.float32)
    source_valid = np.asarray(rendered.silhouette, dtype=bool) & np.isfinite(
        source_camera_z
    )
    camera_z, camera_support, depth_transform = _resize_valid_field(
        source_camera_z,
        source_valid,
        scale=scale,
        dimension=OUTPUT_SIZE,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
    )
    if depth_transform != transform:
        raise AssertionError("HSRD RGB/depth transforms disagree")
    source_face_mask = np.asarray(region["face_mask"]) > 0
    for name in FACE_PART_NAMES:
        if name not in region["part_masks"]:
            raise ValueError(f"HSRD detector omitted face part {name!r}")
        source_face_mask |= np.asarray(region["part_masks"][name]) > 0
    source_face_mask &= source_valid
    face_mask, mask_transform = _resize_and_place(
        source_face_mask.astype(np.uint8),
        scale=scale,
        dimension=OUTPUT_SIZE,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_NEAREST,
        fill_value=0,
    )
    if mask_transform != transform:
        raise AssertionError("HSRD RGB/mask transforms disagree")
    face_mask = (face_mask > 0) & camera_support
    if np.count_nonzero(face_mask) < 600:
        raise ValueError("HSRD transformed face support is too small")

    part_masks = {}
    for name in FACE_PART_NAMES:
        part, part_transform = _resize_and_place(
            (np.asarray(region["part_masks"][name]) > 0).astype(np.float32),
            scale=scale,
            dimension=OUTPUT_SIZE,
            center_xy=center_xy,
            output_center_xy=output_center_xy,
            interpolation=cv2.INTER_AREA,
            fill_value=0.0,
        )
        if part_transform != transform:
            raise AssertionError("HSRD RGB/part transforms disagree")
        part_masks[name] = (part > 1e-4) & face_mask
        if not np.any(part_masks[name]):
            raise ValueError(f"HSRD transformed face part {name!r} is empty")

    transformed_bbox = _transform_bbox(
        region["bbox"],
        scale=scale,
        origin_xy=tuple(transform["origin_xy"]),
        dimension=OUTPUT_SIZE,
    )
    actual_height = transformed_bbox[3] - transformed_bbox[1]
    if actual_height != face_height:
        raise ValueError(
            f"HSRD transformed face height {actual_height} missed {face_height}"
        )
    camera_z[~face_mask] = np.nan
    finite_depth = camera_z[face_mask]
    if not np.all(np.isfinite(finite_depth)) or float(np.ptp(finite_depth)) <= 1e-5:
        raise ValueError("HSRD face camera-Z is incomplete or flat")
    normalized_depth = np.full_like(camera_z, np.nan)
    normalized_depth[face_mask] = (
        finite_depth - float(np.min(finite_depth))
    ) / float(np.ptp(finite_depth))

    intrinsics = _transformed_intrinsics(transform)
    normals = _camera_space_normals(camera_z, face_mask, intrinsics)

    row_id = (
        f"hsrd_{spec.person_id.lower()}_{_yaw_token(yaw_degrees)}_h{face_height}"
    )
    row_dir = output_dir / "rows" / row_id
    parts_dir = row_dir / "exact_face_parts"
    source_record = _write_array(row_dir / "source.png", output_dir, source)
    selection_record = _write_array(
        row_dir / "selection_mask.png",
        output_dir,
        face_mask.astype(np.uint8) * 255,
    )
    exact_depth_record = _write_array(
        row_dir / "exact_depth.npy",
        output_dir,
        normalized_depth.astype(np.float32),
    )
    camera_depth_record = _write_array(
        row_dir / "exact_camera_depth.npy",
        output_dir,
        camera_z.astype(np.float32),
    )
    camera_normals_record = _write_array(
        row_dir / "exact_camera_normals.npy",
        output_dir,
        normals,
    )
    part_records = {
        name: _write_array(
            parts_dir / f"{name}.png",
            output_dir,
            values.astype(np.uint8) * 255,
        )
        for name, values in part_masks.items()
    }
    return {
        "row_id": row_id,
        "identity_group": spec.person_id,
        "split": spec.split,
        "source": source_record,
        "selection_mask": selection_record,
        "exact_depth": exact_depth_record,
        "exact_camera_depth": camera_depth_record,
        "exact_camera_normals": camera_normals_record,
        "exact_face_parts": part_records,
        "detector": _detector_provenance(region, detector_errors),
        "selection_geometry": {
            "selection_bbox_xyxy": transformed_bbox,
            "selection_bbox_width_pixels": transformed_bbox[2] - transformed_bbox[0],
            "selection_bbox_height_pixels": actual_height,
            "face_bbox_xyxy": transformed_bbox,
            "face_bbox_width_pixels": transformed_bbox[2] - transformed_bbox[0],
            "face_bbox_height_pixels": actual_height,
            "camera": {
                "projection": "perspective-opencv",
                "intrinsics": intrinsics.tolist(),
                "fov_y_degrees": PERSPECTIVE_FOV_Y_DEGREES,
                "camera_distance_normalized_units": CAMERA_DISTANCE,
                "target_depth": "floating-relative-camera-z",
                "metric_scale_claimed": False,
                "resize_pixel_center_convention": "opencv-half-pixel",
            },
            "detector_errors": detector_errors,
            "render_face_height_pixels": detected_height,
            "source_to_output_transform": transform,
            "camera_z_span": float(np.ptp(finite_depth)),
            **mesh_diagnostics,
        },
        "rendering": {
            "provider": "hsrd100-lod1-camera-depth",
            "person_id": spec.person_id,
            "pose_id": spec.pose_id,
            "camera_yaw_degrees": float(yaw_degrees),
            "target_face_height_pixels": int(face_height),
            "source_archive_sha256": spec.archive_sha256,
            "repository": HSRD_REPOSITORY,
            "source_revision": HSRD_REVISION,
            "license": HSRD_LICENSE,
            "attribution": HSRD_ATTRIBUTION,
            "source_geometry_training_and_evaluation_only": True,
        },
    }


def build_hsrd_face_corpus(
    manifest_root: str | Path,
    dataset_root: str | Path,
    extracted_root: str | Path,
    output_dir: str | Path,
    *,
    pose_ids: tuple[str, ...] | list[str] | None = None,
    identity_limit: int | None = None,
) -> dict:
    manifest_root = Path(manifest_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    extracted_root = Path(extracted_root).resolve()
    output_dir = Path(output_dir).resolve()
    specs = selected_specs(pose_ids, identity_limit)
    preflight = hsrd_preflight(manifest_root, dataset_root, specs=specs)
    if not preflight["ready"]:
        failed = sorted(name for name, passed in preflight["checks"].items() if not passed)
        raise RuntimeError("HSRD preflight failed: " + ", ".join(failed))
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    rows = []
    view_rejections = []
    identity_diagnostics = []
    for spec in specs:
        archive_path = dataset_root / spec.archive_relative_path
        obj_path = extract_identity_archive(
            archive_path,
            extracted_root / spec.pose_id,
            spec,
        )
        mesh, colors, mesh_diagnostics = load_hsrd_head(obj_path)
        identity_diagnostics.append(
            {
                **asdict(spec),
                "archive_relative_path": spec.archive_relative_path.as_posix(),
                "obj_sha256": _sha256(obj_path),
                **mesh_diagnostics,
            }
        )
        for yaw_degrees in CAMERA_YAWS_DEGREES:
            rendered = _render_identity(mesh, colors, yaw_degrees)
            source_high = np.clip(rendered.rgb * 255.0, 0, 255).astype(np.uint8)
            try:
                region, detector_errors = _select_face_region(source_high)
            except RuntimeError as exc:
                view_rejections.append(
                    {
                        "person_id": spec.person_id,
                        "pose_id": spec.pose_id,
                        "split": spec.split,
                        "camera_yaw_degrees": float(yaw_degrees),
                        "reason": f"{type(exc).__name__}: {exc}",
                        "kind": "incomplete-face-landmarks",
                    }
                )
                continue
            source_part_counts = {
                name: int(np.count_nonzero(region["part_masks"].get(name, 0)))
                for name in FACE_PART_NAMES
            }
            minimum_part_pixels = min(source_part_counts.values())
            if minimum_part_pixels < MINIMUM_SOURCE_PART_PIXELS:
                view_rejections.append(
                    {
                        "person_id": spec.person_id,
                        "pose_id": spec.pose_id,
                        "split": spec.split,
                        "camera_yaw_degrees": float(yaw_degrees),
                        "reason": (
                            f"minimum source part pixels {minimum_part_pixels} "
                            f"is below {MINIMUM_SOURCE_PART_PIXELS}"
                        ),
                        "kind": "insufficient-visible-part-area",
                        "source_part_pixel_counts": source_part_counts,
                    }
                )
                continue
            for face_height in TARGET_FACE_HEIGHTS:
                rows.append(
                    _emit_row(
                        rendered,
                        spec,
                        yaw_degrees,
                        face_height,
                        output_dir,
                        mesh_diagnostics,
                        detected_region=region,
                        detector_errors=detector_errors,
                    )
                )
        print(f"rendered {spec.pose_id}: {len(rows)} rows", flush=True)

    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("HSRD corpus emitted duplicate row IDs")
    split_identities = {
        split: sorted({row["identity_group"] for row in rows if row["split"] == split})
        for split in ("train", "validation", "sealed")
    }
    identity_disjoint = all(
        not (set(split_identities[left]) & set(split_identities[right]))
        for left, right in (("train", "validation"), ("train", "sealed"), ("validation", "sealed"))
    )
    rows_per_identity = Counter(row["identity_group"] for row in rows)
    minimum_rows_per_identity = min(rows_per_identity.values(), default=0)
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    full_run = len(specs) == len(IDENTITY_SPECS) and set(specs) == set(IDENTITY_SPECS)
    view_coverage = _view_coverage_contract(rows, specs)
    expected_view_count = view_coverage["expected_view_count"]
    accepted_view_count = view_coverage["accepted_view_count"]
    source_gate_passed = (
        identity_disjoint
        and len(rows_per_identity) == len(specs)
        and minimum_rows_per_identity >= 4
        and len(rows) == accepted_view_count * len(TARGET_FACE_HEIGHTS)
        and view_coverage["rate_passed"]
        and view_coverage["split_yaw_sign_coverage_passed"]
    )
    summary = {
        "schema_version": 1,
        "provider": "hsrd100-lod1-camera-depth",
        "repository": HSRD_REPOSITORY,
        "repository_url": HSRD_REPOSITORY_URL,
        "source_revision": HSRD_REVISION,
        "dataset_license": HSRD_LICENSE,
        "dataset_license_url": HSRD_LICENSE_URL,
        "attribution": HSRD_ATTRIBUTION,
        "privacy": "public CC BY 4.0 photogrammetry scans; no private user artifacts",
        "production_training_eligible": True,
        "source_geometry_training_and_evaluation_only": True,
        "target_depth": "floating-relative-camera-z",
        "target_dimension": OUTPUT_SIZE,
        "target_face_heights_pixels": list(TARGET_FACE_HEIGHTS),
        "camera_yaws_degrees": list(CAMERA_YAWS_DEGREES),
        "selected_identity_count": len(specs),
        "selected_pose_ids": [spec.pose_id for spec in specs],
        "full_expected_identity_count": len(IDENTITY_SPECS),
        "full_expected_row_count": len(IDENTITY_SPECS)
        * len(CAMERA_YAWS_DEGREES)
        * len(TARGET_FACE_HEIGHTS),
        "row_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "split_identities": split_identities,
        "identity_disjoint": identity_disjoint,
        "promotion_eligible": full_run and source_gate_passed,
        "source_gate_passed": source_gate_passed,
        "minimum_source_part_pixels": MINIMUM_SOURCE_PART_PIXELS,
        "expected_view_count": expected_view_count,
        "accepted_view_count": accepted_view_count,
        "view_detection_rate": view_coverage["view_detection_rate"],
        "minimum_view_detection_rate": MINIMUM_VIEW_DETECTION_RATE,
        "yaw_signs_by_split": view_coverage["yaw_signs_by_split"],
        "view_rate_passed": view_coverage["rate_passed"],
        "split_yaw_sign_coverage_passed": view_coverage[
            "split_yaw_sign_coverage_passed"
        ],
        "view_rejection_count": len(view_rejections),
        "view_rejections": view_rejections,
        "rows_per_identity": dict(sorted(rows_per_identity.items())),
        "minimum_rows_per_identity": minimum_rows_per_identity,
        "manifest_sha256": _sha256(manifest_root / "README.md"),
        "preflight_sha256": _sha256(preflight_path),
        "selected_asset_manifest_sha256": _canonical_sha256(
            [
                {
                    "person_id": spec.person_id,
                    "pose_id": spec.pose_id,
                    "split": spec.split,
                    "archive_bytes": spec.archive_bytes,
                    "archive_sha256": spec.archive_sha256,
                }
                for spec in specs
            ]
        ),
        "identity_diagnostics": identity_diagnostics,
        "excluded_identities": EXCLUDED_IDENTITIES,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pose-id", action="append")
    parser.add_argument("--identity-limit", type=int)
    args = parser.parse_args()
    summary = build_hsrd_face_corpus(
        args.manifest_root,
        args.dataset_root,
        args.extracted_root,
        args.output_dir,
        pose_ids=args.pose_id,
        identity_limit=args.identity_limit,
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
