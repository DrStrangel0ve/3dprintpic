import unittest

import numpy as np

from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    _correlation,
    _minmax_normalize,
)


class EvaluateFaceDepthHeadExactGateTests(unittest.TestCase):
    def test_minmax_normalize_matches_production_linear_storage(self):
        values = np.asarray([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
        normalized = _minmax_normalize(values)

        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)
        self.assertAlmostEqual(float(normalized[0, 1]), 0.25)

    def test_correlation_ignores_nonfinite_values(self):
        left = np.asarray([0.0, 1.0, np.nan, 2.0], dtype=np.float32)
        right = np.asarray([0.0, 2.0, 8.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(_correlation(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()
