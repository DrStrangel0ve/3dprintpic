import unittest
from unittest.mock import patch

import numpy as np

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    RELIEF_HEIGHT_MM,
)
from backend.benchmark.run_face_depth_head_30mm_replay import (
    BASE_RELIEF_HEIGHT_MM,
    _validate_replay_binding,
)
from backend.benchmark.run_vggheads_small_face_30mm_replay import (
    PRINTABLE_FEATURE_DEPTH_MM,
    _correlation,
    _eligible_for_replay,
    _named_part_failure_count,
    _variant_quality,
)


class RunVGGHeadsSmallFace30mmReplayTests(unittest.TestCase):
    def test_trained_head_replay_reserves_feature_depth_inside_height_budget(self):
        self.assertAlmostEqual(
            BASE_RELIEF_HEIGHT_MM + PRINTABLE_FEATURE_DEPTH_MM,
            RELIEF_HEIGHT_MM,
        )

    def test_trained_head_replay_binds_source_and_checkpoint(self):
        evidence = {
            "source_summary_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
        }
        _validate_replay_binding("a" * 64, "b" * 64, evidence)
        with self.assertRaisesRegex(ValueError, "source summary"):
            _validate_replay_binding("c" * 64, "b" * 64, evidence)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            _validate_replay_binding("a" * 64, "c" * 64, evidence)

    def test_eligibility_supports_legacy_and_generalization_evidence(self):
        self.assertTrue(
            _eligible_for_replay(
                {"decision": {"eligible_for_30mm_stl_replay": True}}
            )
        )
        self.assertTrue(
            _eligible_for_replay(
                {"decision": {"eligible_for_full_stl_replay": True}}
            )
        )
        self.assertTrue(
            _eligible_for_replay(
                {
                    "decision": {
                        "checks": {
                            "eligible_for_30mm_stl_replay": True
                        }
                    }
                }
            )
        )
        self.assertFalse(_eligible_for_replay({"decision": {}}))

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
