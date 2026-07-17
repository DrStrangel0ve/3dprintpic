import unittest
from unittest import mock

import numpy as np

from backend import face_depth_refinement as face_module


class FaceTransformationMatrixTests(unittest.TestCase):
    def test_validated_transform_accepts_a_rigid_affine_matrix(self):
        angle = np.deg2rad(23.0)
        matrix = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle), 22.0],
                [0.0, 1.0, 0.0, 4.0],
                [-np.sin(angle), 0.0, np.cos(angle), -97.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        validated = face_module._validated_face_transformation_matrix(matrix)

        np.testing.assert_allclose(validated, matrix, atol=1e-6)
        self.assertEqual(validated.dtype, np.float32)

    def test_validated_transform_rejects_non_rigid_rotation(self):
        matrix = np.eye(4)
        matrix[0, 0] = 1.1

        with self.assertRaisesRegex(ValueError, "rotation is invalid"):
            face_module._validated_face_transformation_matrix(matrix)

    def test_detection_requests_transform_in_the_landmark_pass(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        with mock.patch.object(
            face_module,
            "_detect_faces_mediapipe",
            return_value=[{"bbox": [80, 50, 175, 190]}],
        ) as detector:
            regions, errors = face_module.detect_face_regions(
                image,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(regions), 1)
        detector.assert_called_once_with(
            image,
            3,
            64,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )


if __name__ == "__main__":
    unittest.main()
