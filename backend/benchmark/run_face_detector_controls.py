from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.run_relief_scene_regression import _git_provenance
from backend.face_depth_refinement import (
    YUNET_MODEL_REVISION,
    YUNET_MODEL_SHA256,
    _detect_faces_mediapipe,
    _detect_faces_opencv,
    _detect_faces_yunet,
    _effective_min_face_pixels,
    detect_face_regions,
)


HAAR_CASCADE_REVISION = "opencv-4.10.0"
HAAR_CASCADE_SHA256 = "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0"
FACE_CASES = {
    "centered": "photo_detail_0p00mm_caucasian_female_smile_centered_30p0mm",
    "left": "photo_detail_0p00mm_african_male_neutral_left_frame_30p0mm",
    "right": "photo_detail_0p00mm_asian_female_asymmetric_right_frame_30p0mm",
}
PROVENANCE_PATHS = (
    "backend/face_depth_refinement.py",
    "backend/benchmark/run_face_detector_controls.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb_sha256(image_rgb: np.ndarray) -> str:
    image = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    return hashlib.sha256(image.tobytes()).hexdigest()


def _load_rgb(path: Path, size: int = 256, *, nearest: bool = False) -> np.ndarray:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), resampling)).copy()


def _subject_removed_background(scene_dir: Path, size: int = 256) -> np.ndarray:
    image = _load_rgb(scene_dir / "source.png", size)
    with Image.open(scene_dir / "face_parts" / "face.png") as mask_image:
        mask = np.asarray(
            mask_image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        ) > 0
    image[mask] = 245
    return image


def _checkerboard(size: int = 256) -> np.ndarray:
    yy, xx = np.indices((size, size))
    tile = max(1, size // 16)
    values = (((xx // tile + yy // tile) % 2) * 180 + 40).astype(np.uint8)
    return np.stack(
        (values, np.roll(values, tile // 2, axis=0), np.roll(values, tile // 2, axis=1)),
        axis=-1,
    )


def _detector_record(name: str, image_rgb: np.ndarray, minimum_face_pixels: int) -> dict:
    mediapipe = _detect_faces_mediapipe(image_rgb, 3, minimum_face_pixels)
    yunet = _detect_faces_yunet(image_rgb, 3, minimum_face_pixels)
    haar = _detect_faces_opencv(image_rgb, 3, minimum_face_pixels)
    final_regions, final_errors = detect_face_regions(
        image_rgb,
        max_faces=3,
        min_face_pixels=minimum_face_pixels,
    )
    return {
        "control": name,
        "shape": list(image_rgb.shape),
        "rgb_sha256": _rgb_sha256(image_rgb),
        "mediapipe_count": len(mediapipe),
        "yunet_count": len(yunet),
        "yunet_confidences": [float(region["confidence"]) for region in yunet],
        "haar_count": len(haar),
        "final_chain_count": len(final_regions),
        "final_chain_detectors": [str(region["detector"]) for region in final_regions],
        "final_chain_errors": [str(error) for error in final_errors],
    }


def _checks_pass(checks: dict[str, bool], *, allow_dirty: bool) -> bool:
    return all(
        value
        for name, value in checks.items()
        if not (allow_dirty and name == "implementation_provenance_clean")
    )


def run(
    *,
    fixture_root: str | Path,
    procedural_dataset: str | Path,
    yunet_model: str | Path,
    haar_cascade: str | Path,
    output: str | Path,
    allow_dirty: bool = False,
) -> dict:
    fixture_root = Path(fixture_root)
    procedural_dataset = Path(procedural_dataset)
    yunet_model = Path(yunet_model)
    haar_cascade = Path(haar_cascade)
    output = Path(output)
    if _sha256_file(yunet_model) != YUNET_MODEL_SHA256:
        raise RuntimeError("YuNet model checksum mismatch")
    if _sha256_file(haar_cascade) != HAAR_CASCADE_SHA256:
        raise RuntimeError("OpenCV 4.10 Haar cascade checksum mismatch")

    negative_images = {
        f"cc0_{name}_background_only": _subject_removed_background(
            fixture_root / directory
        )
        for name, directory in FACE_CASES.items()
    }
    for index in range(3):
        negative_images[f"procedural_object_{index}"] = _load_rgb(
            procedural_dataset / f"procedural_mesh_{index:04d}_v00_full.png",
            nearest=True,
        )
    negative_images["high_contrast_checkerboard"] = _checkerboard()
    negative_images["blank_neutral"] = np.full((256, 256, 3), 245, dtype=np.uint8)

    centered = _load_rgb(fixture_root / FACE_CASES["centered"] / "source.png")
    minimum_face_pixels = _effective_min_face_pixels(centered.shape, 96)
    previous_yunet_path = os.environ.get("YUNET_FACE_DETECTOR_MODEL_PATH")
    previous_haar_path = getattr(cv2.data, "haarcascades", "")
    try:
        os.environ["YUNET_FACE_DETECTOR_MODEL_PATH"] = str(yunet_model)
        cv2.data.haarcascades = str(haar_cascade.parent) + os.sep
        positive_yunet = _detect_faces_yunet(centered, 3, minimum_face_pixels)
        positive_final, positive_errors = detect_face_regions(
            centered,
            max_faces=3,
            min_face_pixels=minimum_face_pixels,
        )
        negatives = [
            _detector_record(name, image, minimum_face_pixels)
            for name, image in negative_images.items()
        ]
    finally:
        if previous_yunet_path is None:
            os.environ.pop("YUNET_FACE_DETECTOR_MODEL_PATH", None)
        else:
            os.environ["YUNET_FACE_DETECTOR_MODEL_PATH"] = previous_yunet_path
        cv2.data.haarcascades = previous_haar_path

    provenance = _git_provenance(PROVENANCE_PATHS)
    checks = {
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "positive_yunet_detected": len(positive_yunet) == 1,
        "positive_final_chain_detected": len(positive_final) == 1,
        "positive_final_chain_clean": not positive_errors,
        "all_mediapipe_negative": all(row["mediapipe_count"] == 0 for row in negatives),
        "all_yunet_negative": all(row["yunet_count"] == 0 for row in negatives),
        "all_haar_negative": all(row["haar_count"] == 0 for row in negatives),
        "all_final_chain_negative": all(row["final_chain_count"] == 0 for row in negatives),
        "all_final_chain_clean": all(not row["final_chain_errors"] for row in negatives),
    }
    checks["passed"] = _checks_pass(checks, allow_dirty=allow_dirty)
    summary = {
        "schema_version": 1,
        "run_kind": "face_detector_positive_and_negative_controls",
        "privacy": "CC0 generated heads, deterministic backgrounds, and procedural objects",
        "implementation_provenance": provenance,
        "minimum_face_pixels": minimum_face_pixels,
        "models": {
            "yunet": {
                "revision": YUNET_MODEL_REVISION,
                "sha256": YUNET_MODEL_SHA256,
                "score_threshold": 0.9,
            },
            "haar": {
                "revision": HAAR_CASCADE_REVISION,
                "sha256": HAAR_CASCADE_SHA256,
            },
        },
        "positive_control": {
            "control": "cc0_centered_face",
            "shape": list(centered.shape),
            "rgb_sha256": _rgb_sha256(centered),
            "yunet_count": len(positive_yunet),
            "yunet_confidences": [
                float(region["confidence"]) for region in positive_yunet
            ],
            "final_chain_count": len(positive_final),
            "final_chain_detectors": [
                str(region["detector"]) for region in positive_final
            ],
            "final_chain_errors": [str(error) for error in positive_errors],
        },
        "negative_controls": negatives,
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["checks"]["passed"]:
        raise RuntimeError("Face detector controls failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", required=True)
    parser.add_argument("--procedural-dataset", required=True)
    parser.add_argument("--yunet-model", required=True)
    parser.add_argument("--haar-cascade", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run(
        fixture_root=args.fixture_root,
        procedural_dataset=args.procedural_dataset,
        yunet_model=args.yunet_model,
        haar_cascade=args.haar_cascade,
        output=args.output,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
