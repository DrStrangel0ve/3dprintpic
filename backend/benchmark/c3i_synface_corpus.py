"""Import the official C3I-SynFace demo pairs into the face-depth corpus schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.face_depth_refinement import FACE_PART_NAMES, detect_face_regions


SOURCE_REPOSITORY = "https://github.com/khan9048/Facial_depth_estimation"
SOURCE_REVISION = "dc8adfffbfd38818b72d0ad3776eb66724fafe95"
DEMO_ASSET_LICENSE = "not stated in source repository"
RAW_DATASET_DOI = "10.17632/z4454fyd8b.1"
RAW_DATASET_TITLE = "C3I-SynFace female data part 1"
RAW_DATASET_LICENSE = "CC BY 4.0"
RAW_DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
FULL_COLLECTION_PAIR_COUNT = 37_670
FULL_COLLECTION_REFERENCE = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10070519/"
)
RAW_ARCHIVE_FILENAME = "female_data_part1.7z"
RAW_ARCHIVE_URL = (
    "https://data.mendeley.com/public-files/datasets/z4454fyd8b/files/"
    "e2415b29-a475-4c3e-b8cd-cd9f79e82dba/file_downloaded"
)
RAW_ARCHIVE_BYTES = 7_738_200_071
RAW_ARCHIVE_SHA256 = (
    "5448d6d6577172a3f93c237c8cfe3e1896b238d256810e775855c890bf94b478"
)
DEFAULT_DIMENSION = 256
DEFAULT_FACE_HEIGHT = 78

DEMO_PAIRS = (
    ("0000", "image0000.jpg", "depth_0000.png"),
    ("0001", "Image0001.jpg", "depth_0001.png"),
    ("0002", "image0002.jpg", "depth_0002.png"),
    ("0003", "image0003.jpg", "depth_0003.png"),
    ("0015", "Image0015.jpg", "depth_0015.png"),
    ("0036", "image0036.jpg", "depth_0036.png"),
    ("0037", "image0037.jpg", "depth_0037.png"),
    ("0039", "image0039.jpg", "depth_0039.png"),
    ("0052", "image0052.jpg", "depth_0052.png"),
    ("0097", "image0097.jpg", "depth_0097.png"),
)

EXPECTED_SHA256 = {
    "depth_0000.png": "f835ff6b436847821d85f18895a3d5d9d6063aea701f0f75817fc07124983cbc",
    "depth_0001.png": "016beb50c4688a9244809329fc320cac483412de1ce6e5e84fd5f896ce131831",
    "depth_0002.png": "99ad25ad1ff8e8dfe315ab4d4b3e140ea61a4ef40ed31dd7445b7a01948db51c",
    "depth_0003.png": "ae655bfd3a3a427a0d38e3b9ab2febd29bc13587fd8411c44579182d21b9e9cc",
    "depth_0015.png": "338eab156f06b6cc5718aa06d789c02d1a5d3e23963ac92d66c09845545505bb",
    "depth_0036.png": "e12c50def49259c5d0b74f05d44a432ec3a81ec0b5253840d6143a79407a5e0d",
    "depth_0037.png": "28a7ab62d52a54522f1660554f2b94b6b89e47c50fc88eeecffd9ebb95cb1fc5",
    "depth_0039.png": "ad01f670c94aab441d77e39458efb6b014b74bfed9cec2ff3c5573877ca9c7a5",
    "depth_0052.png": "59786a6b9e92aa096dfbe426d4951e1e062bdfe40841218b0e8dddd36872bb30",
    "depth_0097.png": "6eaf18b29385dad916099ba6c11182de2a293a7ecfcd5ae60dfe4f77a09e4e6f",
    "Image0001.jpg": "370c5085c8edec795104e1a5f95bc97911a9da38f78d57452b1f49a096e0d2b4",
    "Image0015.jpg": "efa4f779ad23a87c1b0f9479e8fdace585d0e32e6b2825810af92c8d1800ea25",
    "image0000.jpg": "b54c8b50fa6268b58e78d44c0436a25f74db45c932f38f52606cd7cc623cb7a2",
    "image0002.jpg": "0f3ff1847dfd201a6f31bc6f08d63b1280b9937c0584299573ff5f334c3d64b4",
    "image0003.jpg": "415c17633c1ac787cba16eba4bf4c670653475680e971b68804b1add59a041e8",
    "image0036.jpg": "60b9f92268bc60be3b1a1eb3bf88b4536f506c48d87ea11a37e394d9fc4d4911",
    "image0037.jpg": "8acbf0a98e7736d35b0111382b15bdb4c902b55f286e73ad0256f24b4fe8a878",
    "image0039.jpg": "46115373ff65f6210b0102e398a5f7a495707d737aa8b5a7a797c8fd7420614f",
    "image0052.jpg": "ad4c99f266f7c30e2091f1aedca03f04b1b8faaa6e2208c1a45d58bb167beabb",
    "image0097.jpg": "20ccd8c7a6c77a855df7feffbdf9914b27f55659ab3606e4f1b45b44e33e71bb",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def c3i_raw_archive_preflight(archive_path: str | Path) -> dict:
    archive_path = Path(archive_path).resolve()
    exists = archive_path.is_file()
    size = archive_path.stat().st_size if exists else None
    size_exact = size == RAW_ARCHIVE_BYTES
    digest = _sha256(archive_path) if size_exact else None
    checks = {
        "archive_present": exists,
        "archive_size_exact": size_exact,
        "archive_sha256_exact": digest == RAW_ARCHIVE_SHA256,
    }
    return {
        "dataset_doi": RAW_DATASET_DOI,
        "dataset_title": RAW_DATASET_TITLE,
        "dataset_license": RAW_DATASET_LICENSE,
        "archive_filename": RAW_ARCHIVE_FILENAME,
        "archive_url": RAW_ARCHIVE_URL,
        "archive_bytes": size,
        "archive_sha256": digest,
        "expected_archive_bytes": RAW_ARCHIVE_BYTES,
        "expected_archive_sha256": RAW_ARCHIVE_SHA256,
        "checks": checks,
        "ready": bool(all(checks.values())),
    }


def c3i_demo_preflight(source_root: str | Path) -> dict:
    source_root = Path(source_root).resolve()
    rgb_root = source_root / "FaceDepth" / "rgb_syn_test"
    depth_root = source_root / "FaceDepth" / "gt_syn_test"
    checks = {}
    evidence = {
        "source_repository": SOURCE_REPOSITORY,
        "required_revision": SOURCE_REVISION,
        "demo_asset_license": DEMO_ASSET_LICENSE,
        "raw_dataset_reference": {
            "doi": RAW_DATASET_DOI,
            "title": RAW_DATASET_TITLE,
            "license": RAW_DATASET_LICENSE,
            "license_url": RAW_DATASET_LICENSE_URL,
        },
        "checks": checks,
    }
    try:
        revision = _git_output(source_root, "rev-parse", "HEAD")
        tracked_status = _git_output(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        evidence["ready"] = False
        return evidence
    evidence["actual_revision"] = revision
    checks["revision_exact"] = revision == SOURCE_REVISION
    checks["tracked_source_clean"] = tracked_status == ""

    files = {}
    for _row_id, rgb_name, depth_name in DEMO_PAIRS:
        for path in (rgb_root / rgb_name, depth_root / depth_name):
            exists = path.is_file()
            digest = _sha256(path) if exists else None
            expected = EXPECTED_SHA256[path.name]
            files[path.name] = {
                "path": path.relative_to(source_root).as_posix(),
                "exists": exists,
                "sha256": digest,
                "expected_sha256": expected,
                "verified": digest == expected,
            }
    evidence["files"] = files
    checks["all_assets_present"] = all(item["exists"] for item in files.values())
    checks["all_assets_verified"] = all(
        item["verified"] for item in files.values()
    )
    evidence["ready"] = bool(all(checks.values()))
    return evidence


def verified_corpus_asset(
    corpus_root: str | Path,
    record: dict,
) -> Path:
    corpus_root = Path(corpus_root).resolve()
    relative = Path(str(record.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("C3I corpus asset path must be a safe relative path")
    path = (corpus_root / relative).resolve()
    try:
        path.relative_to(corpus_root)
    except ValueError as exc:
        raise ValueError("C3I corpus asset resolves outside its root") from exc
    if not path.is_file():
        raise ValueError(f"C3I corpus asset is missing: {relative.as_posix()}")
    expected = str(record.get("sha256", "")).lower()
    actual = _sha256(path)
    if not expected or actual != expected:
        raise ValueError(
            f"C3I corpus asset hash mismatch: {relative.as_posix()}"
        )
    return path


def _load_exact_depth(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path))
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(f"C3I depth preview {path.name} is not RGB(A)")
    if not (
        np.array_equal(values[..., 0], values[..., 1])
        and np.array_equal(values[..., 0], values[..., 2])
    ):
        raise ValueError(f"C3I depth preview {path.name} is not grayscale")
    depth = values[..., 0].astype(np.float32) / 255.0
    if not np.all(np.isfinite(depth)) or float(np.ptp(depth)) <= 1e-6:
        raise ValueError(f"C3I depth preview {path.name} has no usable span")
    return depth


def _select_face_region(image_rgb: np.ndarray) -> tuple[dict, list[str]]:
    regions, errors = detect_face_regions(
        image_rgb,
        max_faces=3,
        min_face_pixels=32,
    )
    regions = [
        region
        for region in regions
        if region.get("landmark_count", 0) >= 468
        and all(
            np.any(np.asarray(region["part_masks"].get(name, 0)) > 0)
            for name in FACE_PART_NAMES
        )
    ]
    if not regions:
        raise RuntimeError(
            "C3I demo row has no complete landmark face: " + "; ".join(errors)
        )
    return max(
        regions,
        key=lambda region: (
            int(region["bbox"][2]) - int(region["bbox"][0])
        )
        * (int(region["bbox"][3]) - int(region["bbox"][1])),
    ), errors


def _border_rgb(image_rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        (
            image_rgb[0],
            image_rgb[-1],
            image_rgb[:, 0],
            image_rgb[:, -1],
        ),
        axis=0,
    )
    return np.median(border, axis=0).round().astype(np.uint8)


def _resize_and_place(
    values: np.ndarray,
    *,
    scale: float,
    dimension: int,
    center_xy: tuple[float, float],
    output_center_xy: tuple[float, float],
    interpolation: int,
    fill_value,
) -> tuple[np.ndarray, dict]:
    source_height, source_width = values.shape[:2]
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        values,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    channels = () if values.ndim == 2 else (values.shape[2],)
    canvas = np.empty((dimension, dimension, *channels), dtype=values.dtype)
    canvas[...] = fill_value
    origin_x = int(round(output_center_xy[0] - center_xy[0] * scale))
    origin_y = int(round(output_center_xy[1] - center_xy[1] * scale))
    dst_x0 = max(0, origin_x)
    dst_y0 = max(0, origin_y)
    dst_x1 = min(dimension, origin_x + resized_width)
    dst_y1 = min(dimension, origin_y + resized_height)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        raise ValueError("C3I resize transform placed the source outside the canvas")
    src_x0 = dst_x0 - origin_x
    src_y0 = dst_y0 - origin_y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
        src_y0:src_y1,
        src_x0:src_x1,
    ]
    return canvas, {
        "scale": float(scale),
        "source_shape": [source_height, source_width],
        "resized_shape": [resized_height, resized_width],
        "canvas_shape": [dimension, dimension],
        "origin_xy": [origin_x, origin_y],
        "source_copy_xyxy": [src_x0, src_y0, src_x1, src_y1],
        "canvas_copy_xyxy": [dst_x0, dst_y0, dst_x1, dst_y1],
    }


def _transform_bbox(
    bbox: list[int],
    *,
    scale: float,
    origin_xy: tuple[int, int],
    dimension: int,
) -> list[int]:
    x0, y0, x1, y1 = bbox
    origin_x, origin_y = origin_xy
    return [
        int(np.clip(round(origin_x + x0 * scale), 0, dimension)),
        int(np.clip(round(origin_y + y0 * scale), 0, dimension)),
        int(np.clip(round(origin_x + x1 * scale), 0, dimension)),
        int(np.clip(round(origin_y + y1 * scale), 0, dimension)),
    ]


def import_c3i_demo_corpus(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    dimension: int = DEFAULT_DIMENSION,
    face_height: int = DEFAULT_FACE_HEIGHT,
    limit: int | None = None,
) -> dict:
    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    preflight = c3i_demo_preflight(source_root)
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError("C3I demo preflight failed: " + ", ".join(failed))
    if dimension < 128 or face_height < 48 or face_height >= dimension:
        raise ValueError("C3I output dimensions are outside the bounded smoke range")
    if limit is not None and limit < 1:
        raise ValueError("C3I demo limit must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = DEMO_PAIRS[:limit] if limit is not None else DEMO_PAIRS
    records = []
    for row_id, rgb_name, depth_name in pairs:
        rgb_path = source_root / "FaceDepth" / "rgb_syn_test" / rgb_name
        depth_path = source_root / "FaceDepth" / "gt_syn_test" / depth_name
        image_rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        exact_depth = _load_exact_depth(depth_path)
        if exact_depth.shape != image_rgb.shape[:2]:
            raise ValueError(f"C3I row {row_id} has mismatched RGB/depth shapes")
        region, detector_errors = _select_face_region(image_rgb)
        x0, y0, x1, y1 = (int(value) for value in region["bbox"])
        source_face_height = y1 - y0
        if source_face_height <= 0:
            raise ValueError(f"C3I row {row_id} has an invalid face box")
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
        exact, _ = _resize_and_place(
            exact_depth,
            scale=scale,
            dimension=dimension,
            center_xy=face_center,
            output_center_xy=output_center,
            interpolation=cv2.INTER_LINEAR,
            fill_value=float(np.nanmax(exact_depth)),
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
        part_masks = {}
        for name in FACE_PART_NAMES:
            part_masks[name], _ = _resize_and_place(
                (np.asarray(region["part_masks"][name]) > 0).astype(np.uint8),
                scale=scale,
                dimension=dimension,
                center_xy=face_center,
                output_center_xy=output_center,
                interpolation=cv2.INTER_NEAREST,
                fill_value=0,
            )
            part_masks[name] = (part_masks[name] > 0) & (face_mask > 0)
            if not np.any(part_masks[name]):
                raise ValueError(f"C3I row {row_id} lost face part {name!r}")

        transformed_bbox = _transform_bbox(
            region["bbox"],
            scale=scale,
            origin_xy=tuple(transform["origin_xy"]),
            dimension=dimension,
        )
        actual_height = transformed_bbox[3] - transformed_bbox[1]
        if abs(actual_height - face_height) > 2:
            raise ValueError(f"C3I row {row_id} missed the target face height")

        row_dir = output_dir / "rows" / f"c3i_demo_{row_id}"
        part_dir = row_dir / "exact_face_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        source_out = row_dir / "source.png"
        selection_out = row_dir / "selection_mask.png"
        exact_out = row_dir / "exact_depth.npy"
        Image.fromarray(source).save(source_out)
        Image.fromarray((face_mask > 0).astype(np.uint8) * 255).save(selection_out)
        np.save(exact_out, exact.astype(np.float32))
        part_records = {}
        for name in FACE_PART_NAMES:
            path = part_dir / f"{name}.png"
            Image.fromarray(part_masks[name].astype(np.uint8) * 255).save(path)
            part_records[name] = {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
            }
        records.append(
            {
                "row_id": f"c3i_demo_{row_id}",
                "split": "research-smoke",
                "identity_group": "c3i_demo_identity_unknown",
                "identity_provenance": (
                    "identity is not published in the demo filename"
                ),
                "expression": "source-unknown",
                "render": {
                    "face_bbox_xyxy": transformed_bbox,
                    "face_bbox_height_pixels": actual_height,
                    "depth_semantics": (
                        "nonmetric normalized 8-bit preview intensity; "
                        "lower is nearer in the observed convention"
                    ),
                    "metric_scale_available": False,
                    "source_depth_representation": (
                        "official grayscale 8-bit demo preview"
                    ),
                    "transform": transform,
                },
                "detector": {
                    "name": region["detector"],
                    "landmark_count": int(region["landmark_count"]),
                    "errors": detector_errors,
                },
                "source_asset": {
                    "rgb_path": rgb_path.relative_to(source_root).as_posix(),
                    "rgb_sha256": _sha256(rgb_path),
                    "depth_path": depth_path.relative_to(source_root).as_posix(),
                    "depth_sha256": _sha256(depth_path),
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

    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "provider": "c3i-synface-official-demo-depth",
        "privacy": "synthetic identities only; no real-person source artifacts",
        "source_geometry_training_and_evaluation_only": True,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "demo_asset_license": DEMO_ASSET_LICENSE,
        "raw_dataset_reference": {
            "doi": RAW_DATASET_DOI,
            "title": RAW_DATASET_TITLE,
            "license": RAW_DATASET_LICENSE,
            "license_url": RAW_DATASET_LICENSE_URL,
        },
        "asset_manifest_sha256": _sha256(preflight_path),
        "target_dimension": int(dimension),
        "target_face_height": int(face_height),
        "row_count": len(records),
        "rows": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--face-height", type=int, default=DEFAULT_FACE_HEIGHT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summary = import_c3i_demo_corpus(
        args.source_root,
        args.output_dir,
        dimension=args.dimension,
        face_height=args.face_height,
        limit=args.limit,
    )
    print(json.dumps({"row_count": summary["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
