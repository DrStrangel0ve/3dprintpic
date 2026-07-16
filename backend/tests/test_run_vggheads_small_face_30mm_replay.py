import unittest
from unittest.mock import patch

import numpy as np

from backend.benchmark.run_vggheads_small_face_30mm_replay import (
    _correlation,
    _named_part_failure_count,
    _variant_quality,
)


class RunVGGHeadsSmallFace30mmReplayTests(unittest.TestCase):
    def test_centered_correlation_ignores_translation(self):
        first = np.arange(16, dtype=np.float64)
        second = first + 8.0

        self.assertAlmostEqual(_correlation(first, second), 1.0)

    def test_named_part_failure_count_combines_shape_and_affine(self):
        count = _named_part_failure_count(
            {"failed_parts": ["left_eye", "mouth"]},
            {"failed_parts": ["nose"]},
        )

        self.assertEqual(count, 3)

    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "_independent_cap_checks",
        return_value={"passed": True},
    )
    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "_independent_background_checks",
        return_value={"passed": True},
    )
    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "_appearance_checks",
        return_value={"passed": True},
    )
    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "face_part_affine_surface_error_metrics",
        return_value={
            "passed": False,
            "failed_parts": ["right_eye"],
        },
    )
    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "face_part_cross_height_metrics",
        return_value={
            "passed": False,
            "failed_parts": ["nose"],
        },
    )
    @patch(
        "backend.benchmark.run_vggheads_small_face_30mm_replay."
        "_emitted_masks",
    )
    def test_variant_quality_keeps_paired_face_gate_separate(
        self,
        emitted_masks,
        _shape_metrics,
        _affine_metrics,
        _appearance,
        _background,
        _cap,
    ):
        face = np.ones((4, 4), dtype=bool)
        emitted_masks.return_value = (
            face,
            {"nose": face, "right_eye": face},
        )
        variant = {
            "_surface": np.zeros((4, 4), dtype=np.float64),
            "postprocess": {
                "mesh_sample_pitch_mm": 0.4,
                "surface_appearance_agreement": {"face": {}},
                "background_depth_preservation": {},
                "selection_background_physical_cap": {},
            },
            "topology": {"printable": True},
            "shell": {"passed": True},
            "surface_max_mm": 30.0,
        }

        quality = _variant_quality(
            variant,
            {"_surface": np.zeros((4, 4), dtype=np.float64)},
            "selection.png",
            {},
        )

        self.assertEqual(quality["combined_named_part_failures"], 2)
        self.assertFalse(quality["checks"]["named_part_shape"])
        self.assertFalse(quality["checks"]["named_part_affine_mm"])
        self.assertTrue(quality["checks"]["passed"])


if __name__ == "__main__":
    unittest.main()
