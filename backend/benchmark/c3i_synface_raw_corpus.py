"""Build a small-face corpus from the licensed C3I-SynFace raw EXR release."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image

from backend.face_depth_refinement import FACE_PART_NAMES
from backend.benchmark.c3i_synface_corpus import (
    _border_rgb,
    _resize_and_place,
    _select_face_region,
    _sha256,
    _transform_bbox,
)


DATASET_DOI = "10.17632/yzjdjj5w39.1"
DATASET_TITLE = "C3I-SynFace female data part 2"
DATASET_LICENSE = "CC BY 4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_PAGE = "https://data.mendeley.com/datasets/yzjdjj5w39/1"
ARCHIVE_FILENAME = "female_data_part2.7z"
ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/yzjdjj5w39/files/"
    "63b625bd-ca8a-40ed-baa7-968665924c94/file_downloaded"
)
ARCHIVE_BYTES = 7_425_750_802
ARCHIVE_SHA256 = (
    "971c643a253e7939f357fb21831b73d44445ae935939cc6f50b662d8b89735a9"
)
EXTRACTED_FILE_COUNT = 19_950
EXTRACTED_BYTES = 8_435_597_152
PAIR_COUNT = 6_650
IDENTITIES = ("0020", "0024", "0025", "0029")
SPLIT_BY_IDENTITY = {
    "0020": "train",
    "0024": "train",
    "0025": "validation",
    "0029": "sealed",
}
BACKGROUNDS = ("Barbershop", "Classroom")
EXPRESSIONS = ("Angry", "Happy", "Neutral", "Sad", "Scared")
MOTIONS = ("CameraTran", "HeadCameraRotTran", "HeadRot")
DEFAULT_DIMENSION = 256
DEFAULT_FACE_HEIGHT = 75
BLENDER_NO_HIT_DEPTH = 1.0e9

_POSE_PATTERN = re.compile(
    r"\ACamera Location: \((?P<camera>[^)]+)\)\n"
    r"Head Point Location: \((?P<head>[^)]+)\)\n"
    r"Camera Rotation: Yaw (?P<cy>-?[0-9.]+) Pitch (?P<cp>-?[0-9.]+) "
    r"Roll (?P<cr>-?[0-9.]+)\n"
    r"Head Rotation: Yaw (?P<hy>-?[0-9.]+) Pitch (?P<hp>-?[0-9.]+) "
    r"Roll (?P<hr>-?[0-9.]+)\s*\Z"
)
_FRAME_PATTERN = re.compile(r"data_(?P<frame>[0-9]{4})\.txt\Z")


class FaceRowRejected(RuntimeError):
    """A valid source row whose face cannot satisfy the exact benchmark mask."""


def _float_triplet(value: str) -> tuple[float, float, float]:
    parts = tuple(float(item.strip()) for item in value.split(","))
    if len(parts) != 3 or not all(math.isfinite(item) for item in parts):
        raise ValueError("C3I pose vector must contain three finite values")
    return parts


def parse_pose_text(value: str) -> dict:
    normalized = value.replace("\r\n", "\n")
    match = _POSE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError("C3I pose metadata does not match the published schema")
    pose = {
        "camera_location": _float_triplet(match.group("camera")),
        "head_point_location": _float_triplet(match.group("head")),
        "camera_rotation_yaw_pitch_roll": tuple(
            float(match.group(name)) for name in ("cy", "cp", "cr")
        ),
        "head_rotation_yaw_pitch_roll": tuple(
            float(match.group(name)) for name in ("hy", "hp", "hr")
        ),
    }
    if not all(
        math.isfinite(item)
        for values in pose.values()
        for item in values
    ):
        raise ValueError("C3I pose metadata contains a non-finite value")
    camera = pose["camera_location"]
    head = pose["head_point_location"]
    forward = float(head[1] - camera[1])
    lateral = float(camera[0] - head[0])
    if abs(forward) <= 1e-8:
        raise ValueError("C3I camera and head have an ambiguous viewing direction")
    pose["camera_view_yaw_degrees"] = float(
        math.degrees(math.atan2(lateral, forward))
    )
    return pose


def _relative_path(path: Path, root: Path) -> PurePosixPath:
    path = path.resolve()
    root = root.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("C3I raw asset resolves outside its release root") from exc
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("C3I raw asset path is unsafe")
    return pure


def raw_release_preflight(
    archive_path: str | Path,
    extracted_root: str | Path,
    *,
    verify_archive_hash: bool = True,
) -> dict:
    archive_path = Path(archive_path).resolve()
    extracted_root = Path(extracted_root).resolve()
    archive_present = archive_path.is_file()
    archive_size = archive_path.stat().st_size if archive_present else None
    size_exact = archive_size == ARCHIVE_BYTES
    archive_digest = (
        _sha256(archive_path) if size_exact and verify_archive_hash else None
    )
    files = (
        [path for path in extracted_root.rglob("*") if path.is_file()]
        if extracted_root.is_dir()
        else []
    )
    extensions = Counter(path.suffix.lower() for path in files)
    extracted_bytes = sum(path.stat().st_size for path in files)
    checks = {
        "archive_present": archive_present,
        "archive_size_exact": size_exact,
        "archive_sha256_exact": (
            archive_digest == ARCHIVE_SHA256 if verify_archive_hash else False
        ),
        "extracted_file_count_exact": len(files) == EXTRACTED_FILE_COUNT,
        "extracted_byte_count_exact": extracted_bytes == EXTRACTED_BYTES,
        "paired_extensions_exact": all(
            extensions[suffix] == PAIR_COUNT
            for suffix in (".jpg", ".exr", ".txt")
        ),
    }
    return {
        "dataset_doi": DATASET_DOI,
        "dataset_title": DATASET_TITLE,
        "dataset_license": DATASET_LICENSE,
        "archive_filename": ARCHIVE_FILENAME,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": archive_size,
        "archive_sha256": archive_digest,
        "expected_archive_bytes": ARCHIVE_BYTES,
        "expected_archive_sha256": ARCHIVE_SHA256,
        "extracted_files": len(files),
        "extracted_bytes": extracted_bytes,
        "extension_counts": dict(sorted(extensions.items())),
        "checks": checks,
        "ready": bool(all(checks.values())),
    }


def _record_from_pose_path(path: Path, root: Path) -> dict:
    relative = _relative_path(path, root)
    parts = relative.parts
    if len(parts) != 6:
        raise ValueError(f"Unexpected C3I raw path: {relative.as_posix()}")
    identity, scene_type, background, expression, motion, filename = parts
    frame_match = _FRAME_PATTERN.fullmatch(filename)
    if (
        identity not in IDENTITIES
        or scene_type != "Complex"
        or background not in BACKGROUNDS
        or expression not in EXPRESSIONS
        or motion not in MOTIONS
        or frame_match is None
    ):
        raise ValueError(f"Unexpected C3I raw taxonomy: {relative.as_posix()}")
    frame = frame_match.group("frame")
    rgb_path = path.with_name(f"rgb_{frame}.jpg")
    depth_path = path.with_name(f"depthExr_{frame}.exr")
    if not rgb_path.is_file() or not depth_path.is_file():
        raise ValueError(f"Incomplete C3I RGB/EXR/pose triplet: {relative.as_posix()}")
    pose = parse_pose_text(path.read_text(encoding="utf-8"))
    head_yaw, head_pitch, head_roll = pose[
        "head_rotation_yaw_pitch_roll"
    ]
    view_yaw = pose["camera_view_yaw_degrees"]
    effective_yaw = (
        view_yaw
        if motion == "CameraTran"
        else head_yaw
        if motion == "HeadRot"
        else head_yaw - view_yaw
    )
    return {
        "identity": identity,
        "split": SPLIT_BY_IDENTITY[identity],
        "background": background,
        "expression": expression,
        "motion": motion,
        "frame": frame,
        "pose_path": path,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "pose": pose,
        "effective_yaw_degrees": float(effective_yaw),
        "head_pitch_degrees": float(head_pitch),
        "head_roll_degrees": float(head_roll),
    }


def discover_raw_records(extracted_root: str | Path) -> list[dict]:
    root = Path(extracted_root).resolve()
    records = [
        _record_from_pose_path(path, root)
        for path in sorted(root.rglob("data_*.txt"))
    ]
    if len(records) != PAIR_COUNT:
        raise ValueError(
            f"C3I raw release has {len(records)} triplets, expected {PAIR_COUNT}"
        )
    keys = {
        (
            record["identity"],
            record["background"],
            record["expression"],
            record["motion"],
            record["frame"],
        )
        for record in records
    }
    if len(keys) != len(records):
        raise ValueError("C3I raw release contains duplicate triplets")
    return records


def _target_yaw(record: dict) -> float:
    key = ":".join(
        str(record[name])
        for name in ("identity", "background", "expression", "motion")
    )
    sign = -1.0 if hashlib.sha256(key.encode("ascii")).digest()[0] & 1 else 1.0
    magnitude = 18.0 if record["motion"] == "CameraTran" else 26.0
    return sign * magnitude


def _selection_score(record: dict, target_yaw: float) -> tuple:
    view_penalty = (
        0.15 * abs(record["pose"]["camera_view_yaw_degrees"])
        if record["motion"] == "HeadCameraRotTran"
        else 0.0
    )
    return (
        abs(record["effective_yaw_degrees"] - target_yaw)
        + 0.50 * abs(record["head_pitch_degrees"])
        + 0.20 * abs(record["head_roll_degrees"])
        + view_penalty,
        abs(record["effective_yaw_degrees"] - target_yaw),
        abs(record["head_pitch_degrees"]),
        abs(record["head_roll_degrees"]),
        record["frame"],
    )


def _ranked_balanced_cells(records: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for record in records:
        key = tuple(
            record[name]
            for name in ("identity", "background", "expression", "motion")
        )
        grouped.setdefault(key, []).append(record)
    cells = []
    for identity in IDENTITIES:
        for background in BACKGROUNDS:
            for expression in EXPRESSIONS:
                for motion in MOTIONS:
                    key = (identity, background, expression, motion)
                    candidates = grouped.get(key, [])
                    if not candidates:
                        raise ValueError(f"C3I raw release is missing balanced cell {key}")
                    target = _target_yaw(candidates[0])
                    ranked = sorted(
                        candidates,
                        key=lambda item: _selection_score(item, target),
                    )
                    cell = []
                    for rank, candidate in enumerate(ranked):
                        candidate = dict(candidate)
                        candidate["selection_target_yaw_degrees"] = target
                        candidate["selection_candidate_count"] = len(ranked)
                        candidate["selection_rank"] = rank
                        cell.append(candidate)
                    cells.append(cell)
    if len(cells) != 120:
        raise ValueError("C3I balanced selector did not build 120 cells")
    return cells


def select_balanced_records(records: list[dict]) -> list[dict]:
    selected = [cell[0] for cell in _ranked_balanced_cells(records)]
    selected_keys = {
        (
            item["identity"],
            item["background"],
            item["expression"],
            item["motion"],
            item["frame"],
        )
        for item in selected
    }
    if len(selected) != 120 or len(selected_keys) != 120:
        raise ValueError("C3I balanced selector did not emit 120 unique rows")
    split_counts = Counter(item["split"] for item in selected)
    if split_counts != Counter({"train": 60, "validation": 30, "sealed": 30}):
        raise ValueError("C3I balanced selector lost identity-disjoint splits")
    return selected


def _load_raw_exr(path: Path) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"C3I raw depth {path.name} is not a three-channel EXR")
    if values.dtype != np.float32:
        raise ValueError(f"C3I raw depth {path.name} is not float32")
    if not (
        np.allclose(values[..., 0], values[..., 1], rtol=0.0, atol=1e-7)
        and np.allclose(values[..., 0], values[..., 2], rtol=0.0, atol=1e-7)
    ):
        raise ValueError(f"C3I raw depth {path.name} channels disagree")
    depth = values[..., 0].astype(np.float32)
    geometry = depth[
        np.isfinite(depth) & (depth > 0.0) & (depth < BLENDER_NO_HIT_DEPTH)
    ]
    if (
        not np.all(np.isfinite(depth))
        or geometry.size < 4
        or float(np.ptp(geometry)) <= 1e-6
    ):
        raise ValueError(f"C3I raw depth {path.name} has invalid geometry")
    return depth


def _resize_depth_and_place(
    depth: np.ndarray,
    *,
    scale: float,
    dimension: int,
    center_xy: tuple[float, float],
    output_center_xy: tuple[float, float],
) -> tuple[np.ndarray, dict, dict]:
    valid = (
        np.isfinite(depth)
        & (np.asarray(depth) > 0.0)
        & (np.asarray(depth) < BLENDER_NO_HIT_DEPTH)
    )
    weighted, transform = _resize_and_place(
        np.where(valid, depth, 0.0).astype(np.float32),
        scale=scale,
        dimension=dimension,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_LINEAR,
        fill_value=0.0,
    )
    weights, weight_transform = _resize_and_place(
        valid.astype(np.float32),
        scale=scale,
        dimension=dimension,
        center_xy=center_xy,
        output_center_xy=output_center_xy,
        interpolation=cv2.INTER_LINEAR,
        fill_value=0.0,
    )
    if transform != weight_transform:
        raise AssertionError("C3I depth values and validity mask lost alignment")
    complete = weights >= (1.0 - 1e-6)
    exact = np.full(weighted.shape, np.nan, dtype=np.float32)
    exact[complete] = weighted[complete] / weights[complete]
    return exact, transform, {
        "source_no_hit_pixels": int(np.count_nonzero(~valid)),
        "source_valid_pixels": int(np.count_nonzero(valid)),
        "output_valid_pixels": int(np.count_nonzero(complete)),
        "no_hit_threshold": float(BLENDER_NO_HIT_DEPTH),
        "resampling_policy": "linear-only-with-complete-valid-support",
    }


def _save_row(
    record: dict,
    *,
    raw_root: Path,
    output_dir: Path,
    dimension: int,
    face_height: int,
) -> dict:
    image_rgb = np.asarray(Image.open(record["rgb_path"]).convert("RGB"))
    exact_depth = _load_raw_exr(record["depth_path"])
    if exact_depth.shape != image_rgb.shape[:2]:
        raise ValueError("C3I raw RGB and EXR dimensions disagree")
    try:
        region, detector_errors = _select_face_region(image_rgb)
    except RuntimeError as exc:
        raise FaceRowRejected(str(exc)) from exc
    x0, y0, x1, y1 = (int(value) for value in region["bbox"])
    source_face_height = y1 - y0
    if source_face_height <= 0:
        raise FaceRowRejected("C3I raw detector emitted an invalid face box")
    scale = float(face_height / source_face_height)
    face_center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    output_center = (0.5 * dimension, 0.46 * dimension)
    source, transform = _resize_and_place(
        image_rgb,
        scale=scale,
        dimension=dimension,
        center_xy=face_center,
        output_center_xy=output_center,
        interpolation=cv2.INTER_AREA,
        fill_value=_border_rgb(image_rgb),
    )
    exact, _, depth_validity = _resize_depth_and_place(
        exact_depth,
        scale=scale,
        dimension=dimension,
        center_xy=face_center,
        output_center_xy=output_center,
    )
    face_mask, _ = _resize_and_place(
        (np.asarray(region["face_mask"]) > 0).astype(np.uint8),
        scale=scale,
        dimension=dimension,
        center_xy=face_center,
        output_center_xy=output_center,
        interpolation=cv2.INTER_NEAREST,
        fill_value=0,
    )
    face_mask = (face_mask > 0) & np.isfinite(exact)
    if np.count_nonzero(face_mask) < 64:
        raise FaceRowRejected("C3I raw row has too little valid exact face depth")
    part_masks = {}
    for name in FACE_PART_NAMES:
        values, _ = _resize_and_place(
            (np.asarray(region["part_masks"][name]) > 0).astype(np.uint8),
            scale=scale,
            dimension=dimension,
            center_xy=face_center,
            output_center_xy=output_center,
            interpolation=cv2.INTER_NEAREST,
            fill_value=0,
        )
        part_masks[name] = (values > 0) & face_mask
        if not np.any(part_masks[name]):
            raise FaceRowRejected(f"C3I raw row lost face part {name!r}")
    transformed_bbox = _transform_bbox(
        region["bbox"],
        scale=scale,
        origin_xy=tuple(transform["origin_xy"]),
        dimension=dimension,
    )
    actual_height = transformed_bbox[3] - transformed_bbox[1]
    if abs(actual_height - face_height) > 2:
        raise FaceRowRejected("C3I raw row missed the target face height")

    row_id = "c3i_f2_" + "_".join(
        str(record[name]).lower()
        for name in ("identity", "background", "expression", "motion", "frame")
    )
    row_dir = output_dir / "rows" / row_id
    part_dir = row_dir / "exact_face_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    source_out = row_dir / "source.png"
    selection_out = row_dir / "selection_mask.png"
    exact_out = row_dir / "exact_depth.npy"
    Image.fromarray(source).save(source_out)
    Image.fromarray(face_mask.astype(np.uint8) * 255).save(selection_out)
    np.save(exact_out, exact.astype(np.float32))
    part_records = {}
    for name, values in part_masks.items():
        path = part_dir / f"{name}.png"
        Image.fromarray(values.astype(np.uint8) * 255).save(path)
        part_records[name] = {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(path),
        }
    return {
        "row_id": row_id,
        "split": record["split"],
        "identity_group": f"c3i_female_part2_{record['identity']}",
        "identity_provenance": "synthetic source model directory",
        "expression": record["expression"].lower(),
        "background": record["background"].lower(),
        "motion": record["motion"],
        "pose": {
            **record["pose"],
            "effective_yaw_degrees": record["effective_yaw_degrees"],
            "selection_target_yaw_degrees": record[
                "selection_target_yaw_degrees"
            ],
            "selection_candidate_count": record["selection_candidate_count"],
            "selection_rank": record["selection_rank"],
        },
        "render": {
            "face_bbox_xyxy": transformed_bbox,
            "face_bbox_height_pixels": actual_height,
            "depth_semantics": (
                "raw Blender Z-pass float EXR; lower is nearer; affine relief-mm "
                "fit required"
            ),
            "metric_scale_available": False,
            "source_depth_representation": "three equal float32 EXR channels",
            "depth_validity": depth_validity,
            "transform": transform,
        },
        "detector": {
            "name": region["detector"],
            "landmark_count": int(region["landmark_count"]),
            "errors": detector_errors,
        },
        "source_asset": {
            "rgb_path": _relative_path(record["rgb_path"], raw_root).as_posix(),
            "rgb_sha256": _sha256(record["rgb_path"]),
            "depth_path": _relative_path(record["depth_path"], raw_root).as_posix(),
            "depth_sha256": _sha256(record["depth_path"]),
            "pose_path": _relative_path(record["pose_path"], raw_root).as_posix(),
            "pose_sha256": _sha256(record["pose_path"]),
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


def _emit_balanced_rows(
    cells: list[list[dict]],
    save_row,
) -> tuple[list[dict], list[dict]]:
    rows = []
    rejections = []
    for cell in cells:
        emitted = None
        for candidate in cell:
            try:
                emitted = save_row(candidate)
            except FaceRowRejected as exc:
                rejections.append(
                    {
                        "identity": candidate["identity"],
                        "background": candidate["background"],
                        "expression": candidate["expression"],
                        "motion": candidate["motion"],
                        "frame": candidate["frame"],
                        "selection_rank": candidate["selection_rank"],
                        "reason": str(exc),
                    }
                )
                continue
            break
        if emitted is None:
            first = cell[0]
            key = tuple(
                first[name]
                for name in ("identity", "background", "expression", "motion")
            )
            raise RuntimeError(f"C3I raw balanced cell has no usable face row: {key}")
        rows.append(emitted)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("C3I raw fallback emitted duplicate rows")
    return rows, rejections


def import_raw_corpus(
    archive_path: str | Path,
    extracted_root: str | Path,
    output_dir: str | Path,
    *,
    dimension: int = DEFAULT_DIMENSION,
    face_height: int = DEFAULT_FACE_HEIGHT,
    limit: int | None = None,
) -> dict:
    archive_path = Path(archive_path).resolve()
    extracted_root = Path(extracted_root).resolve()
    output_dir = Path(output_dir).resolve()
    preflight = raw_release_preflight(archive_path, extracted_root)
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError("C3I raw preflight failed: " + ", ".join(failed))
    if dimension < 128 or face_height < 48 or face_height >= dimension:
        raise ValueError("C3I raw output dimensions are outside the bounded range")
    if limit is not None and limit < 1:
        raise ValueError("C3I raw limit must be positive")
    cells = _ranked_balanced_cells(discover_raw_records(extracted_root))
    if limit is not None:
        cells = cells[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, selection_rejections = _emit_balanced_rows(
        cells,
        lambda record: _save_row(
            record,
            raw_root=extracted_root,
            output_dir=output_dir,
            dimension=dimension,
            face_height=face_height,
        ),
    )
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "provider": "c3i-synface-female-part2-raw-exr",
        "dataset_doi": DATASET_DOI,
        "dataset_page": DATASET_PAGE,
        "dataset_license": DATASET_LICENSE,
        "dataset_license_url": DATASET_LICENSE_URL,
        "privacy": "synthetic identities only; no real-person source artifacts",
        "source_geometry_training_and_evaluation_only": True,
        "identity_disjoint_splits": True,
        "asset_manifest_sha256": _sha256(preflight_path),
        "target_dimension": int(dimension),
        "target_face_height": int(face_height),
        "full_balanced_row_count": 120,
        "promotion_eligible": limit is None,
        "row_count": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "selection_rejection_count": len(selection_rejections),
        "selection_rejections": selection_rejections,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--face-height", type=int, default=DEFAULT_FACE_HEIGHT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = import_raw_corpus(
        args.archive_path,
        args.extracted_root,
        args.output_dir,
        dimension=args.dimension,
        face_height=args.face_height,
        limit=args.limit,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
