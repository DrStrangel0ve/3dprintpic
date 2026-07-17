import unittest

import numpy as np

from backend.benchmark.mediapipe_expression_features import (
    BLENDSHAPE_NAMES,
    BLENDSHAPE_SCHEMA_SHA256,
    blendshape_provenance,
    validated_blendshape_features,
)


class MediaPipeExpressionFeatureTests(unittest.TestCase):
    def test_blendshape_schema_is_sorted_and_unique(self):
        self.assertEqual(len(BLENDSHAPE_NAMES), 52)
        self.assertEqual(tuple(sorted(BLENDSHAPE_NAMES)), BLENDSHAPE_NAMES)
        self.assertEqual(len(set(BLENDSHAPE_NAMES)), len(BLENDSHAPE_NAMES))
        self.assertIn("jawOpen", BLENDSHAPE_NAMES)
        self.assertIn("mouthSmileLeft", BLENDSHAPE_NAMES)
        self.assertIn("eyeBlinkRight", BLENDSHAPE_NAMES)
        self.assertEqual(len(BLENDSHAPE_SCHEMA_SHA256), 64)

    def test_blendshape_schema_has_no_nonfinite_sentinel(self):
        values = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(values)))

        validated = validated_blendshape_features(BLENDSHAPE_NAMES, values)

        np.testing.assert_array_equal(validated, values)
        self.assertEqual(
            blendshape_provenance()["extraction"],
            "same-pass-same-face-index-as-landmarks",
        )

    def test_blendshape_validation_fails_closed(self):
        values = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "schema"):
            validated_blendshape_features(
                tuple(reversed(BLENDSHAPE_NAMES)),
                values,
            )
        values[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validated_blendshape_features(BLENDSHAPE_NAMES, values)


if __name__ == "__main__":
    unittest.main()
