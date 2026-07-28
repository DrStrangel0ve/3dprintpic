import unittest

import numpy as np

from backend.pic_to_3d import _shape_relief_values


class ReliefNormalizationReferenceTests(unittest.TestCase):
    def test_external_reference_prevents_local_update_rescaling_background(self):
        baseline = np.linspace(
            0.0,
            1.0,
            100,
            dtype=np.float32,
        ).reshape(10, 10)
        selection = np.zeros((10, 10), dtype=bool)
        selection[2:8, 2:8] = True
        candidate = baseline.copy()
        candidate[4:6, 4:6] += 0.4
        updated = candidate != baseline

        baseline_relief = _shape_relief_values(
            baseline,
            gamma=1.0,
            detail_boost=0.0,
            low_percentile=1.0,
            high_percentile=99.0,
            normalization_mask=selection,
        )
        candidate_relief = _shape_relief_values(
            candidate,
            gamma=1.0,
            detail_boost=0.0,
            low_percentile=1.0,
            high_percentile=99.0,
            normalization_mask=selection,
            normalization_reference_values=baseline,
        )

        np.testing.assert_array_equal(
            candidate_relief[~selection],
            baseline_relief[~selection],
        )
        self.assertTrue(
            np.all(
                candidate_relief[updated]
                > baseline_relief[updated]
            )
        )

    def test_reference_shape_mismatch_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "normalization reference",
        ):
            _shape_relief_values(
                np.zeros((8, 8), dtype=np.float32),
                detail_boost=0.0,
                normalization_reference_values=np.zeros(
                    (7, 8),
                    dtype=np.float32,
                ),
            )

    def test_subject_interior_uses_candidate_normalization_at_full_taper(self):
        baseline = np.linspace(
            0.0,
            1.0,
            400,
            dtype=np.float32,
        ).reshape(20, 20)
        selection = np.zeros((20, 20), dtype=bool)
        selection[2:18, 2:18] = True
        candidate = baseline.copy()
        candidate[7:13, 7:13] += 0.4

        candidate_relief = _shape_relief_values(
            candidate,
            gamma=1.0,
            detail_boost=0.0,
            normalization_mask=selection,
        )
        dual_relief = _shape_relief_values(
            candidate,
            gamma=1.0,
            detail_boost=0.0,
            normalization_mask=selection,
            normalization_reference_values=baseline,
        )
        baseline_relief = _shape_relief_values(
            baseline,
            gamma=1.0,
            detail_boost=0.0,
            normalization_mask=selection,
        )

        np.testing.assert_array_equal(
            dual_relief[~selection],
            baseline_relief[~selection],
        )
        self.assertEqual(
            float(dual_relief[10, 10]),
            float(candidate_relief[10, 10]),
        )


if __name__ == "__main__":
    unittest.main()
