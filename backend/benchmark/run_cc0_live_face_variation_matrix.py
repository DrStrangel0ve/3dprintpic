"""Run a privacy-safe CC0 MakeHuman face matrix against a live backend.

All rendered inputs, request records, and raw responses are kept below the
specified clean server checkout's ignored ``backend/output`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import httpx
import numpy as np
from PIL import Image

from backend.benchmark.makehuman_face_fixture import (
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.run_makehuman_face_depth_smoke import (
    _correlation,
    _make_scene,
    _resize_nan_aware,
)
from backend.benchmark.run_background_photo_detail_sweep import _detail_metrics
from backend.benchmark.run_private_background_photo_detail_replay import (
    _require_ignored,
    _selection_source_fingerprint,
    _sha256,
)
from backend.benchmark.run_relief_visual_sweep import (
    BACKGROUND_APPEARANCE_GATES,
    FACE_APPEARANCE_GATES,
    _appearance_checks,
    _stl_heightfield_agreement,
)
from backend.benchmark.summarize_private_live_api_background_replay import (
    _independent_background_checks,
    _independent_cap_checks,
    _topology_record,
)


DEFAULT_ASSET_DIR = Path(__file__).parent / "assets" / "makehuman_cc0_heads"
DEFAULT_OUTPUT_NAME = "cc0-live-face-variation-matrix"
BACKGROUND_FIELDS = (
    "background_photo_detail_mm",
    "selection_background_depth_ratio",
)
BASELINE_DETAIL_MM = 0.0
CANDIDATE_DEFAULT_DETAIL_MM = 0.60
DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO = 0.65
RELIEF_HEIGHT_MM = 30.0
MAX_XY_SIZE_MM = 96.0
EXACT_FACE_DEPTH_GATES = {
    "minimum_coverage_ratio": 0.98,
    "minimum_shape_correlation": 0.60,
    "minimum_gradient_correlation": 0.20,
    "maximum_normalized_rmse": 0.35,
}
_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
PRODUCER_PATHS = (
    "backend/benchmark/run_cc0_live_face_variation_matrix.py",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/assets/makehuman_cc0_heads/asset.json",
)


@dataclass(frozen=True)
class OccluderSpec:
    """A normalized, deliberately small rectangular foreground occluder."""

    left: float
    top: float
    right: float
    bottom: float
    rgb: tuple[int, int, int] = (47, 71, 83)


@dataclass(frozen=True)
class FaceSceneSpec:
    row_id: str
    profile_name: str
    target_dimension: int
    camera_yaw_deg: float
    camera_distance: float
    camera_scale: float = 1.0
    horizontal_offset: float = 0.0
    occluder: OccluderSpec | None = None


# Keep the recommended cheap smoke first: a small, off-axis, yawed face at 256 px.
DEFAULT_MATRIX = (
    FaceSceneSpec(
        row_id="small_off_axis_yaw_256",
        profile_name="asian_female_asymmetric",
        target_dimension=256,
        camera_yaw_deg=24.0,
        camera_distance=7.5,
        horizontal_offset=0.16,
    ),
    FaceSceneSpec(
        row_id="centered_neutral_384",
        profile_name="african_male_neutral",
        target_dimension=384,
        camera_yaw_deg=0.0,
        camera_distance=3.6,
    ),
    FaceSceneSpec(
        row_id="offset_occluded_turn_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=-20.0,
        camera_distance=4.0,
        camera_scale=1.05,
        horizontal_offset=-0.13,
        occluder=OccluderSpec(0.39, 0.49, 0.61, 0.62),
    ),
    FaceSceneSpec(
        row_id="close_positive_turn_384",
        profile_name="caucasian_female_smile",
        target_dimension=384,
        camera_yaw_deg=14.0,
        camera_distance=3.25,
        camera_scale=1.08,
        horizontal_offset=0.06,
    ),
)


def _validate_scene_spec(spec: FaceSceneSpec) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", spec.row_id):
        raise ValueError(f"Unsafe row identifier: {spec.row_id!r}")
    if not 64 <= int(spec.target_dimension) <= 1024:
        raise ValueError("target_dimension must be between 64 and 1024")
    numeric = (
        spec.camera_yaw_deg,
        spec.camera_distance,
        spec.camera_scale,
        spec.horizontal_offset,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("Scene camera controls must be finite")
    if not -75.0 <= float(spec.camera_yaw_deg) <= 75.0:
        raise ValueError("camera_yaw_deg must be between -75 and 75")
    if not 1.5 <= float(spec.camera_distance) <= 12.0:
        raise ValueError("camera_distance must be between 1.5 and 12")
    if not 0.5 <= float(spec.camera_scale) <= 2.0:
        raise ValueError("camera_scale must be between 0.5 and 2")
    if not -0.45 <= float(spec.horizontal_offset) <= 0.45:
        raise ValueError("horizontal_offset must be between -0.45 and 0.45")
    if spec.occluder is not None:
        occ = spec.occluder
        coordinates = (occ.left, occ.top, occ.right, occ.bottom)
        if not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError("Occluder bounds must be finite")
        if not (
            0.0 <= occ.left < occ.right <= 1.0 and 0.0 <= occ.top < occ.bottom <= 1.0
        ):
            raise ValueError("Occluder bounds must be ordered normalized coordinates")
        width = float(occ.right - occ.left)
        height = float(occ.bottom - occ.top)
        if width > 0.35 or height > 0.35 or width * height > 0.10:
            raise ValueError("Occluder exceeds the bounded foreground allowance")
        if len(occ.rgb) != 3 or any(not 0 <= int(value) <= 255 for value in occ.rgb):
            raise ValueError("Occluder RGB values must be bytes")


def _translate(values: np.ndarray, columns: int, fill) -> np.ndarray:
    array = np.asarray(values)
    shifted = np.empty_like(array)
    shifted[...] = fill
    if columns == 0:
        shifted[...] = array
    elif columns > 0 and columns < array.shape[1]:
        shifted[:, columns:] = array[:, : array.shape[1] - columns]
    elif columns < 0 and -columns < array.shape[1]:
        shifted[:, :columns] = array[:, -columns:]
    return shifted


def _scene_phase(row_id: str) -> float:
    # A stable identifier-derived phase avoids Python's randomized hash seed.
    import hashlib

    prefix = hashlib.sha256(row_id.encode("ascii")).digest()[:8]
    return int.from_bytes(prefix, "big") / float(2**64) * 2.0 * math.pi


def _render_scene_arrays(
    spec: FaceSceneSpec, fixture: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    _validate_scene_spec(spec)
    profile = fixture["profiles"].get(spec.profile_name)
    if profile is None:
        raise ValueError(f"Unknown MakeHuman profile: {spec.profile_name}")
    colors = make_profile_vertex_colors(
        profile["mesh"].vertices,
        profile["skin_tone"],
        fixture["part_weights"],
        fixture["surface_weights"],
    )
    effective_distance = float(spec.camera_distance) / float(spec.camera_scale)
    background = (0.84, 0.87, 0.91)
    rendered = render_mesh(
        profile["mesh"],
        CameraSpec(azimuth_deg=float(spec.camera_yaw_deg), elevation_deg=0.0),
        RenderConfig(
            size=int(spec.target_dimension),
            projection="perspective",
            perspective_fov_y_deg=32.0,
            camera_distance=effective_distance,
            background_rgb=background,
            ambient=0.42,
            diffuse=0.53,
            specular=0.035,
            shininess=48.0,
            light_direction=(-0.35, -0.20, 0.90),
        ),
        (180, 120, 100),
        vertex_part_weights=fixture["part_weights"],
        vertex_colors=colors,
    )
    offset_columns = int(round(float(spec.horizontal_offset) * spec.target_dimension))
    face_mask = _translate(
        np.asarray(rendered.silhouette, dtype=bool), offset_columns, False
    )
    face_rgb = _translate(
        np.asarray(rendered.rgb, dtype=np.float32), offset_columns, background
    )
    rendered_depth = _translate(
        np.asarray(rendered.depth, dtype=np.float32), offset_columns, 1.0
    )
    occluder_pixels = 0
    occluder_bounds = None
    if spec.occluder is not None:
        occ = spec.occluder
        size = int(spec.target_dimension)
        left = max(0, min(size - 1, int(math.floor(occ.left * size))))
        top = max(0, min(size - 1, int(math.floor(occ.top * size))))
        right = max(left + 1, min(size, int(math.ceil(occ.right * size))))
        bottom = max(top + 1, min(size, int(math.ceil(occ.bottom * size))))
        occluder_pixels = int((right - left) * (bottom - top))
        occluder_bounds = (top, bottom, left, right)

    exact_depth, source_rgb = _make_scene(
        rendered_depth,
        face_mask,
        face_rgb,
        phase=_scene_phase(spec.row_id),
    )
    if occluder_bounds is not None:
        top, bottom, left, right = occluder_bounds
        source_rgb[top:bottom, left:right] = (
            np.asarray(spec.occluder.rgb, dtype=np.float32) / 255.0
        )

    mask_pixels = int(np.count_nonzero(face_mask))
    if mask_pixels == 0:
        raise ValueError(f"Scene {spec.row_id!r} has an empty visible face mask")
    source = np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)
    mask = face_mask.astype(np.uint8) * 255
    rows, columns = np.where(face_mask)
    face_bbox = [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    ]
    face_width = face_bbox[2] - face_bbox[0]
    face_height = face_bbox[3] - face_bbox[1]
    return (
        source,
        mask,
        exact_depth.astype(np.float32, copy=False),
        {
            "effective_camera_distance": effective_distance,
            "horizontal_offset_columns": offset_columns,
            "visible_face_pixels": mask_pixels,
            "visible_face_fraction": mask_pixels / float(mask.size),
            "face_bbox_xyxy": face_bbox,
            "face_bbox_width_pixels": int(face_width),
            "face_bbox_height_pixels": int(face_height),
            "face_bbox_width_ratio": float(face_width / spec.target_dimension),
            "face_bbox_height_ratio": float(face_height / spec.target_dimension),
            "occluder_pixels": occluder_pixels,
        },
    )


def _artifact_record(path: Path, root: Path) -> dict:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _git_output(command: list[str], repository: Path) -> str:
    try:
        return subprocess.run(
            command,
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Clean server repository must be a readable git checkout"
        ) from exc


def _clean_server_provenance(repository: Path) -> dict:
    repository = repository.resolve()
    top_level = Path(
        _git_output(["git", "rev-parse", "--show-toplevel"], repository)
    ).resolve()
    if top_level != repository:
        raise ValueError("Clean server repository must be the git checkout root")
    revision = _git_output(["git", "rev-parse", "HEAD"], repository).lower()
    status = _git_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], repository
    )
    provenance = {
        "available": bool(re.fullmatch(r"[a-f0-9]{40}", revision)),
        "revision": revision,
        "clean": not bool(status),
        "status": status.splitlines() if status else [],
    }
    if not provenance["available"] or not provenance["clean"]:
        raise ValueError("Clean server repository provenance is unavailable or dirty")
    return provenance


def _producer_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]
    provenance = _clean_server_provenance(repository)
    provenance["files"] = [
        {
            "path": relative,
            "sha256": _sha256(repository / relative),
        }
        for relative in PRODUCER_PATHS
    ]
    return provenance


def _prepare_output_directory(
    clean_repository: Path, output_dir: Path | None
) -> tuple[Path, Path]:
    server_output = (clean_repository / "backend" / "output").resolve()
    selected = (
        (server_output / DEFAULT_OUTPUT_NAME).resolve()
        if output_dir is None
        else output_dir.resolve()
    )
    try:
        relative = selected.relative_to(server_output)
    except ValueError as exc:
        raise ValueError(
            "Harness output must be beneath the clean server backend/output"
        ) from exc
    if not relative.parts:
        raise ValueError("Harness output must use a child directory of backend/output")
    _require_ignored(server_output, clean_repository)
    _require_ignored(selected, clean_repository)
    selected.mkdir(parents=True, exist_ok=True)
    return server_output, selected


def _json_object(content: bytes, label: str) -> dict:
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} did not return a JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} did not return a JSON object")
    return payload


def _request_json(
    client,
    *,
    method: str,
    endpoint: str,
    output_dir: Path,
    record_name: str,
    data: dict[str, str] | None = None,
    upload_path: Path | None = None,
    omitted_background_fields: Iterable[str] = (),
) -> tuple[dict, dict, Path]:
    started = time.perf_counter()
    if upload_path is None:
        response = client.get(endpoint)
        upload_record = None
    else:
        with upload_path.open("rb") as upload_handle:
            response = client.post(
                endpoint,
                files={"file": (upload_path.name, upload_handle, "image/png")},
                data=data or {},
            )
        upload_record = {
            "field": "file",
            "filename": upload_path.name,
            "content_type": "image/png",
            "size_bytes": int(upload_path.stat().st_size),
            "sha256": _sha256(upload_path),
        }
    elapsed = time.perf_counter() - started
    response_path = output_dir / f"{record_name}_response.json"
    response_path.write_bytes(bytes(response.content))
    fields = {str(key): str(value) for key, value in sorted((data or {}).items())}
    record = {
        "schema_version": 1,
        "method": method.upper(),
        "endpoint": endpoint,
        "request_fields": fields,
        "posted_form_fields": sorted(fields),
        "omitted_background_fields": sorted(set(omitted_background_fields)),
        "upload": upload_record,
        "http_status": int(response.status_code),
        "elapsed_seconds": round(float(elapsed), 6),
        "response_sha256": _sha256(response_path),
        "response_size_bytes": int(response_path.stat().st_size),
    }
    record_path = output_dir / f"{record_name}_request.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not 200 <= int(response.status_code) < 300:
        text = bytes(response.content)[:300].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{endpoint} failed with HTTP {response.status_code}: {text}"
        )
    return _json_object(bytes(response.content), endpoint), record, record_path


def _server_runtime_provenance(payload: dict) -> dict:
    provenance = payload.get("runtime", {}).get("implementation_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("Server response has no runtime implementation provenance")
    return provenance


def _assert_runtime_provenance(payload: dict, expected: dict) -> dict:
    actual = _server_runtime_provenance(payload)
    if not (
        actual.get("available") is True
        and actual.get("clean") is True
        and str(actual.get("revision", "")).lower() == expected["revision"]
        and not actual.get("status")
    ):
        raise RuntimeError(
            "Live server runtime provenance is dirty or does not match the clean checkout"
        )
    return actual


def _assert_number(payload: dict, key: str, expected: float) -> None:
    try:
        value = float(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Live response has no numeric {key}") from exc
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Live response {key} mismatch: expected {expected}, got {value}"
        )


def _assert_process_contract(
    payload: dict,
    *,
    spec: FaceSceneSpec,
    selection_job_id: str,
    expected_provenance: dict,
    expected_detail_mm: float,
) -> None:
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise RuntimeError("Live process response has no valid job identifier")
    requested_dimension = payload.get(
        "requested_target_dimension", payload.get("target_dimension")
    )
    if int(requested_dimension) != int(spec.target_dimension):
        raise RuntimeError("Live response target_dimension does not match the request")
    _assert_number(payload, "z_scale", RELIEF_HEIGHT_MM)
    _assert_number(payload, "max_xy_size", MAX_XY_SIZE_MM)
    _assert_number(payload, "background_photo_detail_mm", expected_detail_mm)
    _assert_number(
        payload,
        "selection_background_depth_ratio",
        DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    )
    context = payload.get("selection_depth_context", {})
    if context.get("selection_job_id") != selection_job_id:
        raise RuntimeError("Live response does not preserve the composed selection job")
    _assert_runtime_provenance(payload, expected_provenance)


def _stage_scene(
    row_dir: Path, spec: FaceSceneSpec, fixture: dict
) -> tuple[Path, Path, Path, dict]:
    source, mask, exact_depth, render_record = _render_scene_arrays(spec, fixture)
    source_path = row_dir / "source.png"
    mask_path = row_dir / "selection_mask.png"
    exact_depth_path = row_dir / "exact_depth.npy"
    Image.fromarray(source, mode="RGB").save(source_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    np.save(exact_depth_path, exact_depth)
    return source_path, mask_path, exact_depth_path, render_record


def _process_form(
    spec: FaceSceneSpec, selection_job_id: str, *, baseline: bool
) -> dict[str, str]:
    form = {
        "selection_job_id": selection_job_id,
        "target_dimension": str(int(spec.target_dimension)),
        "z_scale": f"{RELIEF_HEIGHT_MM:g}",
        "max_xy_size": f"{MAX_XY_SIZE_MM:g}",
    }
    if baseline:
        form["background_photo_detail_mm"] = f"{BASELINE_DETAIL_MM:g}"
    if not baseline and any(field in form for field in BACKGROUND_FIELDS):
        raise AssertionError("Candidate request must omit both background fields")
    return form


def _job_artifacts(server_output: Path, response: dict) -> dict[str, Path]:
    job_id = str(response.get("job_id", ""))
    if not _JOB_ID.fullmatch(job_id):
        raise RuntimeError("Cannot resolve artifacts for an invalid job identifier")
    job_dir = (server_output / job_id).resolve()
    if job_dir.parent != server_output.resolve():
        raise RuntimeError("Live job artifacts escaped the clean server output root")
    artifacts = {
        "surface": job_dir / "output_surface.npy",
        "reference_surface": job_dir / "output_reference_surface.npy",
        "stl": job_dir / "output_model.stl",
    }
    refinement = response.get("face_refinement", {})
    depth_name = str(refinement.get("depth_file") or "")
    if depth_name:
        if Path(depth_name).name != depth_name:
            raise RuntimeError("Live face refinement has an unsafe depth artifact name")
        artifacts["refined_depth"] = job_dir / depth_name
    elif bool(refinement.get("applied", False)):
        raise RuntimeError("Applied live face refinement has no depth artifact name")
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise RuntimeError("Live job is missing artifacts: " + ", ".join(missing))
    return artifacts


def _mask_on_emitted_grid(
    mask_path: Path,
    surface_grid_transform: dict,
    emitted_shape: tuple[int, int],
) -> np.ndarray:
    required = {
        "input_depth_shape",
        "target_depth_shape",
        "mesh_shape_before_crop",
        "crop_bbox_rc",
        "emitted_shape",
        "flip_x",
    }
    missing = sorted(required - set(surface_grid_transform))
    if missing:
        raise ValueError("Surface-grid transform is missing: " + ", ".join(missing))

    input_rows, input_columns = (
        int(value) for value in surface_grid_transform["input_depth_shape"]
    )
    target_rows, target_columns = (
        int(value) for value in surface_grid_transform["target_depth_shape"]
    )
    with Image.open(mask_path) as loaded:
        mask = (
            np.asarray(
                loaded.convert("L").resize(
                    (input_columns, input_rows), Image.Resampling.NEAREST
                )
            )
            >= 128
        )
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (target_columns, target_rows), Image.Resampling.NEAREST
            )
        )
        >= 128
    )
    if bool(surface_grid_transform["flip_x"]):
        mask = np.flip(mask, axis=1)

    mesh_rows, mesh_columns = (
        int(value) for value in surface_grid_transform["mesh_shape_before_crop"]
    )
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask = (
        np.asarray(
            mask_image.resize((mesh_columns, mesh_rows), Image.Resampling.NEAREST)
        )
        >= 128
    )
    top, left, bottom, right = (
        int(value) for value in surface_grid_transform["crop_bbox_rc"]
    )
    mask = mask[top:bottom, left:right]
    transform_shape = tuple(
        int(value) for value in surface_grid_transform["emitted_shape"]
    )
    if mask.shape != transform_shape or mask.shape != emitted_shape:
        raise ValueError(
            f"Transformed selection mask has shape {mask.shape}, expected "
            f"{transform_shape} and emitted surface shape {emitted_shape}"
        )
    return mask


def _exact_face_depth_quality(
    refined_depth_path: Path,
    exact_depth_path: Path,
    mask_path: Path,
) -> dict:
    predicted = np.load(refined_depth_path).astype(np.float32)
    exact = np.load(exact_depth_path).astype(np.float32)
    with Image.open(mask_path) as loaded:
        face = (
            np.asarray(
                loaded.convert("L").resize(
                    (exact.shape[1], exact.shape[0]), Image.Resampling.NEAREST
                )
            )
            >= 128
        )
    if predicted.shape != exact.shape:
        predicted = _resize_nan_aware(predicted, exact.shape)
    valid = face & np.isfinite(exact) & np.isfinite(predicted)
    coverage = float(np.count_nonzero(valid) / max(np.count_nonzero(face), 1))
    if np.count_nonzero(valid) < 64:
        return {
            "available": False,
            "coverage_ratio": coverage,
            "gates": dict(EXACT_FACE_DEPTH_GATES),
            "checks": {"passed": False},
        }

    reference = exact.astype(np.float64)
    candidate = predicted.astype(np.float64)
    design = np.column_stack((candidate[valid], np.ones(np.count_nonzero(valid))))
    scale, shift = np.linalg.lstsq(design, reference[valid], rcond=None)[0]
    aligned = candidate * float(scale) + float(shift)
    reference_span = float(
        np.percentile(reference[valid], 95.0) - np.percentile(reference[valid], 5.0)
    )
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(aligned[valid] - reference[valid])))
        / max(reference_span, 1e-8)
    )

    interior = valid.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    interior[1:-1, 1:-1] &= (
        valid[:-2, 1:-1] & valid[2:, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
    )
    reference_gradients = np.gradient(reference)
    candidate_gradients = np.gradient(aligned)
    gradient_correlation = min(
        _correlation(ref[interior], cand[interior])
        for ref, cand in zip(reference_gradients, candidate_gradients)
    )
    metrics = {
        "available": True,
        "coverage_ratio": coverage,
        "shape_correlation": _correlation(reference[valid], aligned[valid]),
        "gradient_correlation": float(gradient_correlation),
        "normalized_rmse": normalized_rmse,
        "affine_scale": float(scale),
        "affine_shift": float(shift),
        "samples": int(np.count_nonzero(valid)),
        "gates": dict(EXACT_FACE_DEPTH_GATES),
    }
    checks = {
        "coverage": coverage >= EXACT_FACE_DEPTH_GATES["minimum_coverage_ratio"],
        "positive_orientation": bool(np.isfinite(scale) and scale > 0),
        "shape_correlation": metrics["shape_correlation"]
        >= EXACT_FACE_DEPTH_GATES["minimum_shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"]
        >= EXACT_FACE_DEPTH_GATES["minimum_gradient_correlation"],
        "normalized_rmse": normalized_rmse
        <= EXACT_FACE_DEPTH_GATES["maximum_normalized_rmse"],
    }
    checks["passed"] = bool(all(checks.values()))
    return {**metrics, "checks": checks}


def _score_variant(
    response: dict,
    *,
    server_output: Path,
    exact_depth_path: Path,
    mask_path: Path,
    require_occlusion: bool,
) -> dict:
    artifacts = _job_artifacts(server_output, response)
    surface = np.load(artifacts["surface"])
    reference = np.load(artifacts["reference_surface"])
    surfaces_valid = bool(
        surface.ndim == 2
        and surface.shape == reference.shape
        and min(surface.shape, default=0) >= 2
        and np.all(np.isfinite(surface))
        and np.all(np.isfinite(reference))
    )

    refinement = response.get("face_refinement", {})
    validated_faces = min(
        int(refinement.get("detected_faces", 0)),
        int(refinement.get("refined_faces", 0)),
    )
    selected_detail = int(refinement.get("refined_selection_detail_regions", 0))
    semantic_key = "face"
    postprocess = response.get("relief_postprocess", {})
    appearance = postprocess.get("surface_appearance_agreement", {}).get(
        semantic_key, {}
    )
    appearance_checks = _appearance_checks(appearance, FACE_APPEARANCE_GATES)
    background_appearance = postprocess.get("surface_appearance_agreement", {}).get(
        "background", {}
    )
    background_appearance_checks = _appearance_checks(
        background_appearance, BACKGROUND_APPEARANCE_GATES
    )
    background_checks = _independent_background_checks(
        postprocess.get("background_depth_preservation", {})
    )
    cap_checks = _independent_cap_checks(
        postprocess.get("selection_background_physical_cap", {})
    )
    topology = _topology_record(response.get("stl_diagnostics", {}))
    shell = _stl_heightfield_agreement(
        artifacts["stl"],
        artifacts["surface"],
        expected_max_xy_size_mm=MAX_XY_SIZE_MM,
    )
    exact_face_depth = (
        _exact_face_depth_quality(
            artifacts["refined_depth"], exact_depth_path, mask_path
        )
        if "refined_depth" in artifacts
        else {
            "available": False,
            "reason": "no_refined_depth_artifact",
            "gates": dict(EXACT_FACE_DEPTH_GATES),
            "checks": {"passed": False},
        }
    )
    occlusion_passed = bool(
        not require_occlusion
        or (
            validated_faces > 0
            and int(refinement.get("eyewear_deoccluded_faces", 0)) >= 1
        )
    )
    checks = {
        "finite_surface_contract": surfaces_valid,
        "refinement_applied": bool(refinement.get("applied", False)),
        "validated_human_face_refined": validated_faces > 0,
        "exact_face_depth": bool(exact_face_depth["checks"].get("passed", False)),
        "semantic_appearance": bool(appearance_checks.get("passed", False)),
        "background_appearance": bool(
            background_appearance_checks.get("passed", False)
        ),
        "background_depth": bool(background_checks.get("passed", False)),
        "physical_cap": bool(cap_checks.get("passed", False)),
        "topology": bool(topology["checks"].get("passed", False)),
        "exact_stl_shell": bool(shell.get("passed", False)),
        "occlusion_deoccluded": occlusion_passed,
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "semantic_scope": semantic_key,
        "validated_face_regions": validated_faces,
        "selected_detail_regions": selected_detail,
        "eyewear_deoccluded_faces": int(refinement.get("eyewear_deoccluded_faces", 0)),
        "appearance": appearance,
        "appearance_checks": appearance_checks,
        "background_appearance": background_appearance,
        "background_appearance_checks": background_appearance_checks,
        "background_depth": postprocess.get("background_depth_preservation", {}),
        "background_depth_checks": background_checks,
        "physical_cap": postprocess.get("selection_background_physical_cap", {}),
        "physical_cap_checks": cap_checks,
        "topology": topology,
        "stl_heightfield_agreement": shell,
        "exact_face_depth": exact_face_depth,
        "artifacts": {
            name: _artifact_record(path, server_output)
            for name, path in artifacts.items()
        },
        "checks": checks,
    }


def _score_pair(
    baseline_response: dict,
    candidate_response: dict,
    *,
    source_path: Path,
    mask_path: Path,
    server_output: Path,
) -> dict:
    baseline_artifacts = _job_artifacts(server_output, baseline_response)
    candidate_artifacts = _job_artifacts(server_output, candidate_response)
    baseline_surface = np.load(baseline_artifacts["surface"])
    candidate_surface = np.load(candidate_artifacts["surface"])
    transform = candidate_response.get("relief_postprocess", {}).get(
        "surface_grid_transform", {}
    )
    baseline_transform = baseline_response.get("relief_postprocess", {}).get(
        "surface_grid_transform", {}
    )
    transforms_match = baseline_transform == transform
    face_mask = _mask_on_emitted_grid(mask_path, transform, candidate_surface.shape)
    detail_telemetry = candidate_response.get("relief_postprocess", {}).get(
        "background_photo_detail", {}
    )
    protection_halo_px = float(detail_telemetry.get("protection_halo_px", 13.0))
    detail = _detail_metrics(
        baseline_surface,
        candidate_surface,
        face_mask,
        source_path,
        CANDIDATE_DEFAULT_DETAIL_MM,
        protection_halo_px,
        surface_grid_transform=transform,
    )
    checks = {
        "shared_surface_shape": baseline_surface.shape == candidate_surface.shape,
        "shared_surface_grid_transform": transforms_match,
        "background_photo_detail": bool(detail["checks"].get("passed", False)),
    }
    checks["passed"] = bool(all(checks.values()))
    return {"background_photo_detail": detail, "checks": checks}


def _matrix_coverage(specs: tuple[FaceSceneSpec, ...]) -> dict:
    rows = [spec.row_id for spec in specs]
    dimensions = sorted({int(spec.target_dimension) for spec in specs})
    yaw_signs = sorted(
        {
            "negative"
            if spec.camera_yaw_deg < 0
            else "positive"
            if spec.camera_yaw_deg > 0
            else "neutral"
            for spec in specs
        }
    )
    return {
        "row_ids": rows,
        "target_dimensions": dimensions,
        "yaw_signs": yaw_signs,
        "occluded_rows": [spec.row_id for spec in specs if spec.occluder is not None],
        "unique_rows": len(rows) == len(set(rows)),
    }


def run(
    clean_server_repository: str | Path,
    *,
    output_dir: str | Path | None = None,
    base_url: str = "http://127.0.0.1:8005",
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    limit: int | None = None,
    timeout_seconds: float = 900.0,
    specs: Iterable[FaceSceneSpec] = DEFAULT_MATRIX,
    client=None,
) -> dict:
    clean_repository = Path(clean_server_repository).resolve()
    expected_provenance = _clean_server_provenance(clean_repository)
    server_output, run_output = _prepare_output_directory(
        clean_repository,
        Path(output_dir) if output_dir is not None else None,
    )
    all_specs = tuple(specs)
    selected_specs = all_specs
    if not all_specs:
        raise ValueError("Face variation matrix must contain at least one row")
    if len({spec.row_id for spec in selected_specs}) != len(selected_specs):
        raise ValueError("Face variation matrix row identifiers must be unique")
    for spec in selected_specs:
        _validate_scene_spec(spec)
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be at least one")
        selected_specs = selected_specs[: int(limit)]

    owns_client = client is None
    live_client = client or httpx.Client(
        base_url=base_url.rstrip("/"), timeout=float(timeout_seconds)
    )
    try:
        health, health_record, health_record_path = _request_json(
            live_client,
            method="GET",
            endpoint="/health",
            output_dir=run_output,
            record_name="health",
        )
        runtime_provenance = _assert_runtime_provenance(health, expected_provenance)
        reported_output = health.get("output_dir")
        if (
            not isinstance(reported_output, str)
            or Path(reported_output).resolve() != server_output
        ):
            raise RuntimeError(
                "Live server output directory does not match the clean checkout"
            )
        producer_provenance = _producer_provenance()
        if producer_provenance["revision"] != expected_provenance["revision"]:
            raise RuntimeError(
                "Matrix producer revision does not match the clean live server"
            )

        fixture = load_makehuman_face_fixture(asset_dir)
        rows = []
        for spec in selected_specs:
            row_dir = run_output / "rows" / spec.row_id
            row_dir.mkdir(parents=True, exist_ok=True)
            source_path, mask_path, exact_depth_path, render_record = _stage_scene(
                row_dir,
                spec,
                fixture,
            )
            mask_reference = mask_path.resolve().relative_to(server_output).as_posix()
            compose_form = {
                "mask_paths_json": json.dumps([mask_reference], separators=(",", ":")),
                "background_mode": "neutral",
            }
            compose, compose_record, compose_record_path = _request_json(
                live_client,
                method="POST",
                endpoint="/selection/compose",
                output_dir=row_dir,
                record_name="compose",
                data=compose_form,
                upload_path=source_path,
            )
            selection_job_id = compose.get("job_id")
            if not isinstance(selection_job_id, str) or not _JOB_ID.fullmatch(
                selection_job_id
            ):
                raise RuntimeError(
                    "Selection compose response has no valid job identifier"
                )
            if (
                compose.get("model_status") != "composed-clicked-masks"
                or int(compose.get("mask_count", -1)) != 1
                or compose.get("source_fingerprint")
                != _selection_source_fingerprint(source_path)
            ):
                raise RuntimeError(
                    "Selection compose response does not match the staged source and mask"
                )

            variants = {}
            response_payloads = {}
            for variant, baseline in (("baseline", True), ("candidate", False)):
                form = _process_form(spec, selection_job_id, baseline=baseline)
                omitted = [field for field in BACKGROUND_FIELDS if field not in form]
                response, request_record, request_record_path = _request_json(
                    live_client,
                    method="POST",
                    endpoint="/process_image",
                    output_dir=row_dir,
                    record_name=variant,
                    data=form,
                    upload_path=source_path,
                    omitted_background_fields=omitted,
                )
                expected_detail = (
                    BASELINE_DETAIL_MM if baseline else CANDIDATE_DEFAULT_DETAIL_MM
                )
                _assert_process_contract(
                    response,
                    spec=spec,
                    selection_job_id=selection_job_id,
                    expected_provenance=expected_provenance,
                    expected_detail_mm=expected_detail,
                )
                response_payloads[variant] = response
                variants[variant] = {
                    "job_id": response["job_id"],
                    "request": request_record,
                    "request_artifact": _artifact_record(
                        request_record_path, run_output
                    ),
                    "response_artifact": _artifact_record(
                        row_dir / f"{variant}_response.json", run_output
                    ),
                    "runtime_provenance": _server_runtime_provenance(response),
                }

            for variant, response in response_payloads.items():
                variants[variant]["quality"] = _score_variant(
                    response,
                    server_output=server_output,
                    exact_depth_path=exact_depth_path,
                    mask_path=mask_path,
                    require_occlusion=spec.occluder is not None,
                )
            pair_quality = _score_pair(
                response_payloads["baseline"],
                response_payloads["candidate"],
                source_path=source_path,
                mask_path=mask_path,
                server_output=server_output,
            )
            row_checks = {
                "baseline": bool(variants["baseline"]["quality"]["checks"]["passed"]),
                "candidate": bool(variants["candidate"]["quality"]["checks"]["passed"]),
                "paired_background_detail": bool(pair_quality["checks"]["passed"]),
            }
            row_checks["passed"] = bool(all(row_checks.values()))

            rows.append(
                {
                    "row_id": spec.row_id,
                    "scene": asdict(spec),
                    "render": render_record,
                    "selection_job_id": selection_job_id,
                    "source": _artifact_record(source_path, run_output),
                    "selection_mask": _artifact_record(mask_path, run_output),
                    "exact_depth": _artifact_record(exact_depth_path, run_output),
                    "compose": {
                        "request": compose_record,
                        "request_artifact": _artifact_record(
                            compose_record_path, run_output
                        ),
                        "response_artifact": _artifact_record(
                            row_dir / "compose_response.json", run_output
                        ),
                    },
                    "variants": variants,
                    "pair_quality": pair_quality,
                    "checks": row_checks,
                }
            )
    finally:
        if owns_client:
            live_client.close()

    final_provenance = _clean_server_provenance(clean_repository)
    if final_provenance["revision"] != expected_provenance["revision"]:
        raise RuntimeError("Clean server repository changed revision during the run")
    final_producer_provenance = _producer_provenance()
    if final_producer_provenance != producer_provenance:
        raise RuntimeError("Matrix producer provenance changed during the run")
    coverage = _matrix_coverage(selected_specs)
    all_rows_passed = bool(rows) and all(row["checks"]["passed"] for row in rows)
    summary = {
        "schema_version": 1,
        "run_kind": "cc0_makehuman_live_face_variation_matrix",
        "privacy": "CC0 MakeHuman synthetic heads and deterministic procedural scene materials only",
        "server_provenance": expected_provenance,
        "server_runtime_provenance": runtime_provenance,
        "producer_provenance": producer_provenance,
        "final_producer_provenance": final_producer_provenance,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "httpx": httpx.__version__,
            "numpy": np.__version__,
        },
        "fixture": fixture["manifest"],
        "matrix": {
            "available_rows": len(all_specs),
            "executed_rows": len(rows),
            "limit": int(limit) if limit is not None else None,
            "z_scale": RELIEF_HEIGHT_MM,
            "max_xy_size": MAX_XY_SIZE_MM,
            "baseline_background_photo_detail_mm": BASELINE_DETAIL_MM,
            "candidate_omitted_background_fields": list(BACKGROUND_FIELDS),
            "coverage": coverage,
        },
        "health": {
            "request": health_record,
            "request_artifact": _artifact_record(health_record_path, run_output),
            "response_artifact": _artifact_record(
                run_output / "health_response.json", run_output
            ),
        },
        "checks": {
            "server_repository_clean": True,
            "server_revision_unchanged": True,
            "server_runtime_matches_checkout": True,
            "producer_matches_server_revision": True,
            "producer_unchanged_during_run": True,
            "raw_artifacts_beneath_ignored_output": True,
            "requested_rows_executed_once": bool(
                coverage["unique_rows"]
                and coverage["row_ids"] == [row["row_id"] for row in rows]
            ),
            "all_behavioral_rows_passed": all_rows_passed,
            "passed": all_rows_passed,
        },
        "rows": rows,
    }
    summary_path = run_output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-server-repository", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    summary = run(
        args.clean_server_repository,
        output_dir=args.output_dir,
        base_url=args.base_url,
        asset_dir=args.asset_dir,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                "passed": summary["checks"]["passed"],
                "executed_rows": summary["matrix"]["executed_rows"],
                "first_row": summary["rows"][0]["row_id"],
            },
            indent=2,
        )
    )
    if not summary["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
