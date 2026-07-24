import unittest
from unittest import mock

import numpy as np

import backend.face_depth_refinement as refinement


class YunetProfileSelectionRoiTests(unittest.TestCase):
    def test_profile_keypoints_are_allowed_only_with_profile_policy(self):
        box = (0.0, 0.0, 100.0, 120.0)
        keypoints = np.asarray(
            [
                [54.0, 36.0],
                [64.0, 36.0],
                [67.0, 61.0],
                [53.0, 91.0],
                [70.0, 91.0],
            ],
            dtype=np.float32,
        )

        self.assertFalse(
            refinement._yunet_keypoints_are_face_like(keypoints, box)
        )
        self.assertTrue(
            refinement._yunet_keypoints_are_face_like(
                keypoints,
                box,
                minimum_eye_separation_ratio=(
                    refinement.YUNET_PROFILE_MINIMUM_EYE_SEPARATION_RATIO
                ),
                minimum_mouth_separation_ratio=(
                    refinement.YUNET_PROFILE_MINIMUM_MOUTH_SEPARATION_RATIO
                ),
            )
        )

    def test_global_detection_keeps_conservative_yunet_policy(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        with (
            mock.patch.object(
                refinement, "_detect_faces_mediapipe", return_value=[]
            ),
            mock.patch.object(
                refinement, "_detect_faces_yunet", return_value=[]
            ) as yunet,
            mock.patch.object(
                refinement, "_detect_faces_opencv", return_value=[]
            ),
        ):
            refinement.detect_face_regions(image, max_faces=1)

        self.assertEqual(
            yunet.call_args.kwargs["score_threshold"],
            refinement.YUNET_SCORE_THRESHOLD,
        )
        self.assertEqual(
            yunet.call_args.kwargs["minimum_eye_separation_ratio"],
            refinement.YUNET_MINIMUM_EYE_SEPARATION_RATIO,
        )
        self.assertEqual(
            yunet.call_args.kwargs["minimum_mouth_separation_ratio"],
            refinement.YUNET_MINIMUM_MOUTH_SEPARATION_RATIO,
        )

    def test_selection_roi_uses_profile_policy_and_overlap_gate(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        roi = np.zeros((160, 160), dtype=np.uint8)
        roi[40:120, 50:110] = 255
        local_face = {
            "bbox": [80, 70, 240, 280],
            "face_mask": np.zeros((384, 288), dtype=np.uint8),
            "feature_mask": np.zeros((384, 288), dtype=np.uint8),
            "detector": "opencv-yunet-2023mar",
            "landmark_count": 5,
        }
        local_face["face_mask"][60:290, 70:250] = 255
        local_face["feature_mask"][60:290, 70:250] = 255

        def detector(values, **kwargs):
            self.assertEqual(
                kwargs["yunet_score_threshold"],
                refinement.YUNET_SELECTION_ROI_SCORE_THRESHOLD,
            )
            self.assertEqual(
                kwargs["yunet_minimum_eye_separation_ratio"],
                refinement.YUNET_PROFILE_MINIMUM_EYE_SEPARATION_RATIO,
            )
            self.assertEqual(
                kwargs["yunet_minimum_mouth_separation_ratio"],
                refinement.YUNET_PROFILE_MINIMUM_MOUTH_SEPARATION_RATIO,
            )
            return [local_face], []

        with mock.patch.object(
            refinement, "detect_face_regions", side_effect=detector
        ):
            regions, _errors, stats = refinement.detect_face_regions_in_roi(
                image,
                roi,
                max_faces=1,
                min_face_pixels=48,
            )

        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(regions[0]["selection_overlap_ratio"], 0.50)
        self.assertEqual(
            stats["selection_roi_yunet_policy"]["score_threshold"],
            refinement.YUNET_SELECTION_ROI_SCORE_THRESHOLD,
        )

    def test_selection_roi_rejects_low_overlap_detection(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        roi = np.zeros((160, 160), dtype=np.uint8)
        roi[40:120, 50:110] = 255
        local_face = {
            "bbox": [0, 0, 80, 80],
            "face_mask": np.zeros((384, 288), dtype=np.uint8),
            "feature_mask": np.zeros((384, 288), dtype=np.uint8),
            "detector": "opencv-yunet-2023mar",
            "landmark_count": 5,
        }
        local_face["face_mask"][:80, :80] = 255
        local_face["feature_mask"][:80, :80] = 255

        with mock.patch.object(
            refinement,
            "detect_face_regions",
            return_value=([local_face], []),
        ):
            regions, _errors, stats = refinement.detect_face_regions_in_roi(
                image,
                roi,
                max_faces=1,
                min_face_pixels=48,
            )

        self.assertEqual(regions, [])
        self.assertEqual(stats["validated_face_regions"], 0)


if __name__ == "__main__":
    unittest.main()
