"""Import a bounded RAP3DF V2 RGB-depth slice for evaluation-only use.

The importer is intentionally offline. Callers provide a local mirror plus a
Mendeley-derived manifest whose records add safe relative paths to the
publisher's file metadata. No directory scanning or filename-based discovery
is used: identities, poses, and RGB/depth pairs come only from database.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.face_depth_refinement import FACE_PART_NAMES, detect_face_regions


DATASET_DOI = "10.17632/kpdkpcs8zb.4"
DATASET_VERSION = 4
DATASET_TITLE = "RAP3DF V2"
DATASET_LICENSE = "CC BY 4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_PAGE = "https://data.mendeley.com/datasets/kpdkpcs8zb/4"
DATABASE_FILENAME = "database.json"
DATABASE_BYTES = 273_343
DATABASE_SHA256 = (
    "1366f0496078a250b43bafffc3483d3f949c33afb32520a041d92d353598e3ea"
)
DEPTH_SHAPE = (119, 149)
DEPTH_DTYPE = np.dtype("<u2")
DEPTH_ORIENTATION = "lower-is-nearer"
DEPTH_SCALE_STATUS = "sensor scale is not established by the dataset release"
DEPTH_REGISTRATION_STATUS = (
    "RGB/depth registration accuracy and calibration are not published"
)
POSES = ("front", "left", "right", "up", "down", "burned")
DEFAULT_DIMENSION = 256
DEFAULT_FACE_HEIGHT = 75
DEFAULT_IDENTITY_LIMIT = 3
MAX_IDENTITY_LIMIT = 16
EVALUATION_SPLIT = "research-evaluation-only"

_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_RGB_PATTERN = re.compile(r"^rgb_([A-Za-z0-9]+)\.bmp$")
_DEPTH_PATTERN = re.compile(r"^depth_bgRm_([A-Za-z0-9]+)\.data$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    text = str(value or "")
    if not text or text != text.strip() or "\\" in text or "\x00" in text:
        raise ValueError(f"RAP3DF {label} must be a safe relative POSIX path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ValueError(f"RAP3DF {label} must be a safe relative POSIX path")
    return relative


def _resolve_local_asset(root: Path, relative: PurePosixPath) -> Path:
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("RAP3DF asset resolves outside its root") from exc
    if not path.is_file():
        raise ValueError(f"RAP3DF asset is missing: {relative.as_posix()}")
    return path


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"RAP3DF JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json_strict(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"RAP3DF JSON is invalid: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"RAP3DF JSON root must be an object: {path.name}")
    return value


def _manifest_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("RAP3DF Mendeley manifest file record must be an object")
    relative = _safe_relative_path(record.get("path"), label="manifest path")
    details = record.get("content_details")
    if not isinstance(details, dict):
        raise ValueError(
            f"RAP3DF manifest record lacks content_details: {relative.as_posix()}"
        )
    filename = str(record.get("filename") or "")
    if filename != relative.name:
        raise ValueError(
            f"RAP3DF manifest filename/path mismatch: {relative.as_posix()}"
        )
    file_id = str(record.get("id") or "")
    content_id = str(details.get("id") or "")
    digest = str(details.get("sha256_hash") or "").lower()
    size = details.get("size")
    if not file_id or not content_id:
        raise ValueError(
            f"RAP3DF manifest record lacks Mendeley identifiers: {relative.as_posix()}"
        )
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(
            f"RAP3DF manifest SHA256 is invalid: {relative.as_posix()}"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"RAP3DF manifest size is invalid: {relative.as_posix()}")
    status = record.get("status")
    if status is not None and status != "COMPLETED":
        raise ValueError(
            f"RAP3DF manifest asset is not completed: {relative.as_posix()}"
        )
    return {
        "path": relative.as_posix(),
        "filename": filename,
        "mendeley_file_id": file_id,
        "mendeley_content_id": content_id,
        "sha256": digest,
        "size": size,
    }


def load_mendeley_manifest(manifest: str | Path | dict) -> tuple[dict, dict]:
    """Normalize and authenticate caller-supplied Mendeley file metadata."""

    if isinstance(manifest, dict):
        value = manifest
    else:
        value = _read_json_strict(Path(manifest).resolve())
    if value.get("dataset_doi") != DATASET_DOI:
        raise ValueError("RAP3DF manifest DOI does not match the pinned dataset")
    if value.get("dataset_version") != DATASET_VERSION:
        raise ValueError("RAP3DF manifest version does not match the pinned dataset")
    if value.get("license") != DATASET_LICENSE:
        raise ValueError("RAP3DF manifest license does not match CC BY 4.0")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("RAP3DF manifest must contain a nonempty files list")

    records = {}
    casefolded_paths = set()
    for raw_record in files:
        record = _manifest_record(raw_record)
        relative = record["path"]
        folded = relative.casefold()
        if relative in records or folded in casefolded_paths:
            raise ValueError(f"RAP3DF manifest has ambiguous path: {relative}")
        records[relative] = record
        casefolded_paths.add(folded)

    canonical = {
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "files": [records[path] for path in sorted(records)],
    }
    canonical_bytes = json.dumps(
        canonical,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    provenance = {
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "canonical_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "file_count": len(records),
    }
    return records, provenance


def verified_manifest_asset(
    source_root: str | Path,
    relative_path: str | PurePosixPath,
    records: dict[str, dict],
) -> tuple[Path, dict]:
    source_root = Path(source_root).resolve()
    relative = _safe_relative_path(relative_path, label="asset path")
    record = records.get(relative.as_posix())
    if record is None:
        raise ValueError(
            f"RAP3DF asset is absent from the Mendeley manifest: {relative.as_posix()}"
        )
    path = _resolve_local_asset(source_root, relative)
    actual_size = path.stat().st_size
    if actual_size != record["size"]:
        raise ValueError(f"RAP3DF asset size mismatch: {relative.as_posix()}")
    actual_digest = _sha256(path)
    if actual_digest != record["sha256"]:
        raise ValueError(f"RAP3DF asset hash mismatch: {relative.as_posix()}")
    return path, record


def decode_depth_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode one publisher raw depth frame as 119x149 little-endian uint16."""

    path = Path(path)
    expected_bytes = int(np.prod(DEPTH_SHAPE) * DEPTH_DTYPE.itemsize)
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"RAP3DF depth shape mismatch: expected {DEPTH_SHAPE} uint16"
        )
    values = np.fromfile(path, dtype=DEPTH_DTYPE)
    if values.size != int(np.prod(DEPTH_SHAPE)):
        raise ValueError("RAP3DF depth frame has an incomplete uint16 payload")
    depth = values.reshape(DEPTH_SHAPE).astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        raise ValueError("RAP3DF depth frame contains no nonzero sensor samples")
    return depth, valid


def _database_pair(identity: str, pose: str, records: Any) -> dict:
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(
            f"RAP3DF {identity}/{pose} pairing is missing or ambiguous"
        )
    pair = records[0]
    if not isinstance(pair, dict):
        raise ValueError(f"RAP3DF {identity}/{pose} pair must be an object")
    rgb = _safe_relative_path(pair.get("rgb"), label="database RGB path")
    depth = _safe_relative_path(
        pair.get("depth_data_with_bg"),
        label="database depth path",
    )
    expected_parent = PurePosixPath("rap3df_data_02") / identity
    if rgb.parent != expected_parent or depth.parent != expected_parent:
        raise ValueError(f"RAP3DF {identity}/{pose} pair crosses identity folders")
    rgb_match = _RGB_PATTERN.fullmatch(rgb.name)
    depth_match = _DEPTH_PATTERN.fullmatch(depth.name)
    if (
        rgb_match is None
        or depth_match is None
        or rgb_match.group(1) != depth_match.group(1)
    ):
        raise ValueError(f"RAP3DF {identity}/{pose} pair has mismatched sample IDs")
    return {
        "identity": identity,
        "pose": pose,
        "sample_id": rgb_match.group(1),
        "rgb_path": rgb.as_posix(),
        "depth_path": depth.as_posix(),
    }


def select_database_rows(
    database: dict,
    *,
    identity_limit: int = DEFAULT_IDENTITY_LIMIT,
    poses: tuple[str, ...] = POSES,
) -> list[dict]:
    """Select complete identity bundles deterministically and without leakage."""

    if (
        not isinstance(identity_limit, int)
        or isinstance(identity_limit, bool)
        or not 1 <= identity_limit <= MAX_IDENTITY_LIMIT
    ):
        raise ValueError(
            f"RAP3DF identity_limit must be between 1 and {MAX_IDENTITY_LIMIT}"
        )
    poses = tuple(poses)
    if (
        not poses
        or any(not isinstance(pose, str) or pose not in POSES for pose in poses)
        or len(set(poses)) != len(poses)
    ):
        raise ValueError("RAP3DF poses must be a unique nonempty subset of POSES")
    identities = database.get("_faces")
    if not isinstance(identities, list):
        raise ValueError("RAP3DF database _faces must contain unique identities")
    if any(
        not isinstance(identity, str)
        or _IDENTITY_PATTERN.fullmatch(identity) is None
        for identity in identities
    ):
        raise ValueError("RAP3DF database contains an unsafe identity")
    if len(set(identities)) != len(identities):
        raise ValueError("RAP3DF database _faces must contain unique identities")
    selected_identities = sorted(identities)[:identity_limit]
    if len(selected_identities) != identity_limit:
        raise ValueError("RAP3DF database has fewer identities than requested")

    rows = []
    for identity in selected_identities:
        identity_record = database.get(identity)
        if not isinstance(identity_record, dict):
            raise ValueError(f"RAP3DF database identity {identity!r} is missing")
        for pose in poses:
            rows.append(_database_pair(identity, pose, identity_record.get(pose)))
    if len({row["identity"] for row in rows}) != identity_limit:
        raise ValueError("RAP3DF identity-disjoint selection is incomplete")
    return rows


def _select_face_region(image_rgb: np.ndarray) -> tuple[dict, list[str]]:
    regions, errors = detect_face_regions(
        image_rgb,
        max_faces=3,
        min_face_pixels=48,
    )
    complete = []
    for region in regions:
        masks = region.get("part_masks") or {}
        if int(region.get("landmark_count", 0)) < 468:
            continue
        face_mask = np.asarray(region.get("face_mask", 0))
        if face_mask.shape != image_rgb.shape[:2] or not np.any(face_mask > 0):
            continue
        if any(
            name not in masks
            or np.asarray(masks[name]).shape != image_rgb.shape[:2]
            or not np.any(np.asarray(masks[name]) > 0)
            for name in FACE_PART_NAMES
        ):
            continue
        complete.append(region)
    if len(complete) != 1:
        raise RuntimeError(
            "RAP3DF row requires exactly one complete landmark face; "
            f"found {len(complete)}; detector errors: {'; '.join(errors)}"
        )
    return complete[0], errors


def _validate_depth_orientation(
    depth: np.ndarray,
    valid: np.ndarray,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
) -> dict:
    face_valid = valid & face_mask
    if np.count_nonzero(face_valid) < 32:
        raise ValueError("RAP3DF face has insufficient valid depth coverage")
    part_coverage = {}
    for name in FACE_PART_NAMES:
        count = int(np.count_nonzero(valid & part_masks[name]))
        part_coverage[name] = count
        if count == 0:
            raise ValueError(f"RAP3DF depth has no valid samples for face part {name!r}")
    nose_values = depth[valid & part_masks["nose"]]
    face_values = depth[face_valid]
    nose_median = float(np.median(nose_values))
    face_median = float(np.median(face_values))
    if not nose_median < face_median:
        raise ValueError(
            "RAP3DF depth orientation check failed: expected lower-is-nearer"
        )
    return {
        "convention": DEPTH_ORIENTATION,
        "nose_median_sensor_value": nose_median,
        "face_median_sensor_value": face_median,
        "part_valid_pixel_counts": part_coverage,
    }


def _border_rgb(image_rgb: np.ndarray) -> tuple[int, int, int]:
    border = np.concatenate(
        (image_rgb[0], image_rgb[-1], image_rgb[:, 0], image_rgb[:, -1]),
        axis=0,
    )
    values = np.median(border, axis=0).round().astype(np.uint8)
    return tuple(int(value) for value in values)


def _face_transform(
    bbox: list[int],
    *,
    source_shape: tuple[int, int],
    dimension: int,
    face_height: int,
) -> tuple[np.ndarray, dict]:
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if not (0 <= x0 < x1 <= source_shape[1] and 0 <= y0 < y1 <= source_shape[0]):
        raise ValueError("RAP3DF detector returned an invalid face box")
    source_face_height = y1 - y0
    scale = float(face_height / source_face_height)
    source_center_x = 0.5 * (x0 + x1)
    source_center_y = 0.5 * (y0 + y1)
    output_center_x = 0.5 * dimension
    output_center_y = 0.46 * dimension
    matrix = np.asarray(
        [
            [scale, 0.0, output_center_x - source_center_x * scale],
            [0.0, scale, output_center_y - source_center_y * scale],
        ],
        dtype=np.float64,
    )
    transformed_bbox = [
        int(round(matrix[0, 0] * x0 + matrix[0, 2])),
        int(round(matrix[1, 1] * y0 + matrix[1, 2])),
        int(round(matrix[0, 0] * x1 + matrix[0, 2])),
        int(round(matrix[1, 1] * y1 + matrix[1, 2])),
    ]
    transformed_bbox = [int(np.clip(value, 0, dimension)) for value in transformed_bbox]
    actual_height = transformed_bbox[3] - transformed_bbox[1]
    if actual_height not in (74, 75) or abs(actual_height - face_height) > 1:
        raise ValueError("RAP3DF shared transform missed the 74-75 px face gate")
    return matrix, {
        "kind": "shared-full-frame-affine",
        "matrix_source_to_output": matrix.tolist(),
        "source_shape": [int(source_shape[0]), int(source_shape[1])],
        "canvas_shape": [int(dimension), int(dimension)],
        "face_bbox_xyxy": transformed_bbox,
        "face_bbox_height_pixels": actual_height,
        "requested_face_height_pixels": int(face_height),
    }


def _warp_mask(mask: np.ndarray, matrix: np.ndarray, dimension: int) -> np.ndarray:
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (dimension, dimension),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _warp_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    matrix: np.ndarray,
    dimension: int,
) -> np.ndarray:
    weighted = cv2.warpAffine(
        depth * valid.astype(np.float32),
        matrix,
        (dimension, dimension),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    weights = cv2.warpAffine(
        valid.astype(np.float32),
        matrix,
        (dimension, dimension),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    output = np.full((dimension, dimension), np.nan, dtype=np.float32)
    usable = weights > 1e-6
    output[usable] = weighted[usable] / weights[usable]
    return output


def _asset_provenance(record: dict) -> dict:
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "size": record["size"],
        "mendeley_file_id": record["mendeley_file_id"],
        "mendeley_content_id": record["mendeley_content_id"],
    }


def import_rap3df_corpus(
    source_root: str | Path,
    mendeley_manifest: str | Path | dict,
    output_dir: str | Path,
    *,
    identity_limit: int = DEFAULT_IDENTITY_LIMIT,
    poses: tuple[str, ...] = POSES,
    dimension: int = DEFAULT_DIMENSION,
    face_height: int = DEFAULT_FACE_HEIGHT,
) -> dict:
    """Create a bounded, authenticated, identity-disjoint evaluation corpus."""

    if dimension < 128 or face_height not in (74, 75) or face_height >= dimension:
        raise ValueError("RAP3DF output must use a 74-75 px face on a bounded canvas")
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    records, manifest_provenance = load_mendeley_manifest(mendeley_manifest)
    database_path, database_record = verified_manifest_asset(
        source_root,
        DATABASE_FILENAME,
        records,
    )
    if (
        database_path.stat().st_size != DATABASE_BYTES
        or _sha256(database_path) != DATABASE_SHA256
        or database_record["size"] != DATABASE_BYTES
        or database_record["sha256"] != DATABASE_SHA256
    ):
        raise ValueError("RAP3DF database.json does not match the pinned release")
    database = _read_json_strict(database_path)
    selected = select_database_rows(
        database,
        identity_limit=identity_limit,
        poses=poses,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for selected_row in selected:
        identity = selected_row["identity"]
        pose = selected_row["pose"]
        sample_id = selected_row["sample_id"]
        rgb_path, rgb_record = verified_manifest_asset(
            source_root,
            selected_row["rgb_path"],
            records,
        )
        depth_path, depth_record = verified_manifest_asset(
            source_root,
            selected_row["depth_path"],
            records,
        )
        image_rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth, valid = decode_depth_data(depth_path)
        if image_rgb.shape[:2] != DEPTH_SHAPE:
            raise ValueError(
                f"RAP3DF {identity}/{pose} RGB/depth shape mismatch"
            )
        region, detector_errors = _select_face_region(image_rgb)
        face_mask = np.asarray(region["face_mask"]) > 0
        part_masks = {
            name: np.asarray(region["part_masks"][name]) > 0
            for name in FACE_PART_NAMES
        }
        orientation = _validate_depth_orientation(
            depth,
            valid,
            face_mask,
            part_masks,
        )
        matrix, transform = _face_transform(
            region["bbox"],
            source_shape=DEPTH_SHAPE,
            dimension=dimension,
            face_height=face_height,
        )
        source = cv2.warpAffine(
            image_rgb,
            matrix,
            (dimension, dimension),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_border_rgb(image_rgb),
        )
        transformed_valid = _warp_mask(valid, matrix, dimension)
        transformed_face = _warp_mask(face_mask, matrix, dimension)
        selection_mask = transformed_valid & transformed_face
        exact_depth = _warp_depth(depth, valid, matrix, dimension)
        exact_depth[~selection_mask] = np.nan
        if not np.any(np.isfinite(exact_depth[selection_mask])):
            raise ValueError(f"RAP3DF {identity}/{pose} lost all valid depth")

        transformed_parts = {}
        for name in FACE_PART_NAMES:
            transformed = _warp_mask(part_masks[name], matrix, dimension)
            transformed &= selection_mask
            if not np.any(transformed):
                raise ValueError(
                    f"RAP3DF {identity}/{pose} lost face part {name!r}"
                )
            transformed_parts[name] = transformed

        row_id = f"rap3df_{identity}_{pose}_{sample_id}"
        row_dir = output_dir / "rows" / row_id
        part_dir = row_dir / "exact_face_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        source_out = row_dir / "source.png"
        selection_out = row_dir / "selection_mask.png"
        exact_out = row_dir / "exact_depth.npy"
        Image.fromarray(source.astype(np.uint8)).save(source_out)
        Image.fromarray(selection_mask.astype(np.uint8) * 255).save(selection_out)
        np.save(exact_out, exact_depth.astype(np.float32))
        part_records = {}
        for name in FACE_PART_NAMES:
            part_path = part_dir / f"{name}.png"
            Image.fromarray(transformed_parts[name].astype(np.uint8) * 255).save(
                part_path
            )
            part_records[name] = {
                "path": part_path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(part_path),
            }

        rows.append(
            {
                "row_id": row_id,
                "split": EVALUATION_SPLIT,
                "identity_group": f"rap3df:{identity}",
                "identity_provenance": (
                    "published RAP3DF V2 volunteer pseudonym; all poses remain "
                    "in one evaluation-only split"
                ),
                "expression": pose,
                "render": {
                    **transform,
                    "depth_semantics": (
                        "raw Kinect One uint16 sensor values; zero is invalid; "
                        "lower nonzero values are nearer"
                    ),
                    "metric_scale_available": False,
                    "sensor_depth_scale_status": DEPTH_SCALE_STATUS,
                    "rgb_depth_registration_status": DEPTH_REGISTRATION_STATUS,
                    "depth_orientation_check": orientation,
                },
                "detector": {
                    "name": str(region.get("detector") or "unknown"),
                    "landmark_count": int(region["landmark_count"]),
                    "errors": detector_errors,
                },
                "source_asset": {
                    "database": _asset_provenance(database_record),
                    "rgb": _asset_provenance(rgb_record),
                    "depth": _asset_provenance(depth_record),
                },
                "provenance": {
                    "dataset_doi": DATASET_DOI,
                    "dataset_version": DATASET_VERSION,
                    "license": DATASET_LICENSE,
                    "mendeley_manifest_sha256": manifest_provenance[
                        "canonical_sha256"
                    ],
                    "evaluation_only_real_volunteer_data": True,
                    "training_eligible": False,
                },
                "source": {
                    "path": source_out.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(source_out),
                },
                "selection_mask": {
                    "path": selection_out.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(selection_out),
                },
                "exact_depth": {
                    "path": exact_out.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(exact_out),
                },
                "exact_face_parts": part_records,
            }
        )

    identity_splits: dict[str, set[str]] = {}
    for row in rows:
        identity_splits.setdefault(row["identity_group"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in identity_splits.values()):
        raise RuntimeError("RAP3DF identity leaked across corpus splits")

    summary = {
        "schema_version": 1,
        "provider": "rap3df-v2-kinect-one-raw-depth",
        "privacy": (
            "real volunteer data; ethics approval is stated by the publisher; "
            "evaluation-only in this project"
        ),
        "source_geometry_training_and_evaluation_only": True,
        "source_geometry_evaluation_only": True,
        "training_eligible": False,
        "source": {
            "title": DATASET_TITLE,
            "doi": DATASET_DOI,
            "version": DATASET_VERSION,
            "page": DATASET_PAGE,
            "license": DATASET_LICENSE,
            "license_url": DATASET_LICENSE_URL,
            "database_json_size": DATABASE_BYTES,
            "database_json_sha256": DATABASE_SHA256,
        },
        "sensor_provenance": {
            "device": "Kinect One",
            "raw_dtype": "little-endian uint16",
            "raw_shape": list(DEPTH_SHAPE),
            "zero_is_invalid": True,
            "orientation": DEPTH_ORIENTATION,
            "metric_scale_available": False,
            "sensor_depth_scale_status": DEPTH_SCALE_STATUS,
            "rgb_depth_registration_status": DEPTH_REGISTRATION_STATUS,
        },
        "asset_manifest_sha256": manifest_provenance["canonical_sha256"],
        "target_dimension": int(dimension),
        "target_face_height": int(face_height),
        "selection": {
            "identity_limit": int(identity_limit),
            "identity_groups": sorted(identity_splits),
            "identity_disjoint": True,
            "poses": list(poses),
            "split": EVALUATION_SPLIT,
        },
        "row_count": len(rows),
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--mendeley-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--identity-limit", type=int, default=DEFAULT_IDENTITY_LIMIT)
    parser.add_argument("--pose", action="append", choices=POSES)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--face-height", type=int, default=DEFAULT_FACE_HEIGHT)
    args = parser.parse_args()
    summary = import_rap3df_corpus(
        args.source_root,
        args.mendeley_manifest,
        args.output_dir,
        identity_limit=args.identity_limit,
        poses=tuple(args.pose) if args.pose else POSES,
        dimension=args.dimension,
        face_height=args.face_height,
    )
    print(
        json.dumps(
            {
                "row_count": summary["row_count"],
                "identity_groups": summary["selection"]["identity_groups"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
