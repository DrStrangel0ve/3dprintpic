import unittest
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark.hsrd_face_training_corpus import (
    HSRD_LICENSE,
    HSRD_REVISION,
)
from backend.benchmark.train_hsrd_face_spatial_pyramid import (
    _rgb_laplacian_luma,
    _per_row_non_regression,
    _nvidia_driver_version,
    _select_rows,
    _select_blend,
    _square_padded_box,
    _strictly_beats,
    _validate_corpus,
    _validate_corpus_summary_hash,
)


def _row(row_id, identity, split, height=74, yaw=-35.0):
    return {
        "row_id": row_id,
        "identity_group": identity,
        "split": split,
        "selection_geometry": {"face_bbox_height_pixels": height},
        "rendering": {"camera_yaw_degrees": yaw},
    }


class TrainHSRDFaceSpatialPyramidTests(unittest.TestCase):
    def _summary(self):
        rows = [
            _row("train-74", "train-id", "train", 74, -35.0),
            _row("train-75", "train-id", "train", 75, 35.0),
            _row("validation-74", "validation-id", "validation", 74, -30.0),
            _row("validation-75", "validation-id", "validation", 75, 30.0),
            _row("sealed-74", "sealed-id", "sealed", 74, -35.0),
            _row("sealed-75", "sealed-id", "sealed", 75, 35.0),
        ]
        return {
            "provider": "hsrd100-lod1-camera-depth",
            "source_revision": HSRD_REVISION,
            "dataset_license": HSRD_LICENSE,
            "source_gate_passed": True,
            "promotion_eligible": True,
            "production_training_eligible": True,
            "identity_disjoint": True,
            "target_depth": "floating-relative-camera-z",
            "source_geometry_training_and_evaluation_only": True,
            "rows": rows,
        }

    def test_square_crop_is_bounded_and_remains_square(self):
        box = _square_padded_box((4, 80, 64, 154), 256, 256)
        self.assertEqual(box[2] - box[0], box[3] - box[1])
        self.assertGreaterEqual(box[0], 0)
        self.assertLessEqual(box[2], 256)

    def test_corpus_gate_rejects_identity_leakage_and_nonturned_rows(self):
        summary = self._summary()
        contract = _validate_corpus(summary)
        self.assertEqual(contract["identity_count"], 3)
        summary["rows"][-1]["identity_group"] = "train-id"
        summary["identity_disjoint"] = True
        with self.assertRaisesRegex(ValueError, "identity_disjoint"):
            _validate_corpus(summary)
        summary = self._summary()
        summary["rows"][0]["rendering"]["camera_yaw_degrees"] = 0.0
        with self.assertRaisesRegex(ValueError, "turned_views"):
            _validate_corpus(summary)

    def test_row_limit_retains_every_split(self):
        selected = _select_rows(self._summary()["rows"], 1)
        self.assertEqual(len(selected), 3)
        self.assertEqual({row["split"] for row in selected}, {"train", "validation", "sealed"})

    def test_corpus_summary_hash_must_match_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_bytes(b'{"source":"pinned"}\n')
            expected = "c8cd813d8151638bc825b33f0dfd768c1905e12b1a620592b00994c8fc735ed0"
            self.assertEqual(_validate_corpus_summary_hash(path, expected), expected)
            path.write_bytes(b'{"source":"substituted"}\n')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _validate_corpus_summary_hash(path, expected)
            with self.assertRaisesRegex(ValueError, "64 lowercase hex"):
                _validate_corpus_summary_hash(path, "not-a-hash")

    def test_driver_provenance_requires_one_consistent_version(self):
        with patch(
            "backend.benchmark.train_hsrd_face_spatial_pyramid.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="591.74\n591.74\n"),
        ):
            self.assertEqual(_nvidia_driver_version(), "591.74")
        with patch(
            "backend.benchmark.train_hsrd_face_spatial_pyramid.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="591.74\n592.00\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "one consistent"):
                _nvidia_driver_version()

    def test_laplacian_conditioning_is_bounded_and_nonconstant(self):
        values = np.zeros((32, 32, 3), dtype=np.uint8)
        values[:, 16:] = 255
        levels = _rgb_laplacian_luma(Image.fromarray(values), size=32)
        self.assertEqual(levels.shape, (3, 32, 32))
        self.assertTrue(np.isfinite(levels).all())
        self.assertLessEqual(float(np.max(np.abs(levels))), 1.0)
        self.assertGreater(float(np.max(np.abs(levels))), 0.05)

    def test_strict_gate_requires_failure_reduction_and_row_nonregression(self):
        reference = {
            "combined_part_failures": 2,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.7,
            "median_normalized_rmse": 0.2,
            "source_background_bit_exact": True,
            "rows": [
                {"row_id": "one", "shape_failed_parts": ["nose"], "affine_failed_parts": ["mouth"]}
            ],
        }
        candidate = {
            **reference,
            "combined_part_failures": 1,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.71,
            "median_normalized_rmse": 0.19,
            "rows": [
                {"row_id": "one", "shape_failed_parts": ["nose"], "affine_failed_parts": []}
            ],
        }
        self.assertTrue(_strictly_beats(candidate, reference))
        candidate["source_background_bit_exact"] = False
        self.assertFalse(_strictly_beats(candidate, reference))

    def test_row_nonregression_rejects_part_swaps_and_row_set_changes(self):
        reference = {
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["nose"],
                    "affine_failed_parts": ["mouth"],
                }
            ]
        }
        swapped = {
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["left_eye"],
                    "affine_failed_parts": [],
                }
            ]
        }
        missing = {"rows": []}
        duplicate = {"rows": [reference["rows"][0], reference["rows"][0]]}
        self.assertFalse(_per_row_non_regression(swapped, reference))
        self.assertFalse(_per_row_non_regression(missing, reference))
        self.assertFalse(_per_row_non_regression(duplicate, reference))

    def test_blend_selector_prefers_eligible_baseline_safe_candidate(self):
        baseline = {
            "combined_part_failures": 4,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.7,
            "median_normalized_rmse": 0.2,
            "source_background_bit_exact": True,
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["nose", "mouth"],
                    "affine_failed_parts": ["left_eye", "right_eye"],
                }
            ],
        }
        regressing = {
            **baseline,
            "alpha": 1.0,
            "combined_part_failures": 2,
            "median_normalized_rmse": 0.1,
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["nose", "mouth", "left_eye", "right_eye", "left_eyebrow"],
                    "affine_failed_parts": [],
                }
            ],
        }
        safe = {
            **baseline,
            "alpha": 0.7,
            "combined_part_failures": 3,
            "median_shape_correlation": 0.82,
            "median_gradient_correlation": 0.72,
            "median_normalized_rmse": 0.18,
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["nose", "mouth"],
                    "affine_failed_parts": ["left_eye"],
                }
            ],
        }
        selected = _select_blend([regressing, safe], baseline)
        self.assertEqual(selected["alpha"], 0.7)
        self.assertTrue(selected["eligible_vs_baseline"])


if __name__ == "__main__":
    unittest.main()
