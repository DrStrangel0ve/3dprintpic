import unittest
from unittest import mock
from pathlib import Path
import tempfile

import numpy as np

from backend.benchmark import train_gnm_structured_geometry as structured
from backend.benchmark.gnm_structured_geometry_provider import (
    StructuredGNMFaceFoundation,
    predict_checkpoint_head,
)
from backend.benchmark.mediapipe_expression_features import BLENDSHAPE_NAMES


class StructuredGNMTrainingTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(path: Path, *, encoder_kind: str) -> None:
        landmark_dimension = 68 * 3 + 52
        encoder_dimension = (
            structured.ENCODER_FEATURE_DIMENSION
            if encoder_kind == structured.FEATURE_ENCODER_DAV2_SMALL
            else 0
        )
        feature_dimension = landmark_dimension + encoder_dimension
        arrays = {
            "metadata_schema_version": np.asarray(
                structured.CHECKPOINT_SCHEMA_VERSION,
                dtype=np.int32,
            ),
            "metadata_landmark_feature_kind": np.asarray(
                structured.LANDMARK_FEATURE_DLIB68_BLENDSHAPES52
            ),
            "metadata_feature_dimension": np.asarray(
                feature_dimension, dtype=np.int32
            ),
            "metadata_feature_encoder_kind": np.asarray(encoder_kind),
            "metadata_feature_encoder_dimension": np.asarray(
                encoder_dimension, dtype=np.int32
            ),
            "metadata_mediapipe_version": np.asarray(
                structured.package_version("mediapipe")
            ),
        }
        encoder_metadata = (
            (
                structured.ENCODER_ID,
                structured.ENCODER_REVISION,
                structured.ENCODER_MODEL_SHA256,
                structured.ENCODER_LICENSE,
            )
            if encoder_kind == structured.FEATURE_ENCODER_DAV2_SMALL
            else ("", "", "", "")
        )
        for key, value in zip(
            (
                "model_id",
                "revision",
                "model_sha256",
                "license",
            ),
            encoder_metadata,
        ):
            arrays[f"metadata_feature_encoder_{key}"] = np.asarray(value)
        provenance = structured.blendshape_provenance()
        arrays["metadata_face_landmarker_sha256"] = np.asarray(
            provenance["model_sha256"]
        )
        arrays["metadata_blendshape_names_sha256"] = np.asarray(
            provenance["names_sha256"]
        )
        for prefix, target_dimension in (
            ("identity", structured.IDENTITY_SKIN_DIMENSION),
            ("expression", structured.EXPRESSION_SKIN_DIMENSION),
        ):
            arrays.update(
                {
                    f"{prefix}_feature_mean": np.zeros(
                        feature_dimension, dtype=np.float32
                    ),
                    f"{prefix}_feature_scale": np.ones(
                        feature_dimension, dtype=np.float32
                    ),
                    f"{prefix}_target_mean": np.zeros(
                        target_dimension, dtype=np.float32
                    ),
                    f"{prefix}_components": np.empty(
                        (0, target_dimension), dtype=np.float32
                    ),
                    f"{prefix}_score_mean": np.empty(0, dtype=np.float32),
                    f"{prefix}_weights": np.empty(
                        (feature_dimension, 0), dtype=np.float32
                    ),
                    f"{prefix}_score_minimum": np.empty(
                        0, dtype=np.float32
                    ),
                    f"{prefix}_score_maximum": np.empty(
                        0, dtype=np.float32
                    ),
                }
            )
        np.savez(path, **arrays)

    def test_landmark_features_are_translation_and_scale_invariant(self):
        indices = np.arange(478, dtype=np.float64)
        landmarks = np.column_stack(
            (
                0.45 + 0.16 * np.cos(indices * 0.07),
                0.51 + 0.22 * np.sin(indices * 0.09),
                0.04 * np.sin(indices * 0.13),
            )
        )
        transformed = landmarks.copy()
        transformed[:, :2] = transformed[:, :2] * 2.75 + (1.2, -0.8)
        transformed[:, 2] = transformed[:, 2] * 4.0 + 0.7

        first = structured.normalized_landmark_features(landmarks)
        second = structured.normalized_landmark_features(transformed)

        np.testing.assert_allclose(first, second, atol=1e-5)
        self.assertEqual(first.shape, (68 * 3,))
        self.assertTrue(np.all(np.isfinite(first)))

    def test_full_landmark_features_keep_all_468_points(self):
        indices = np.arange(478, dtype=np.float64)
        landmarks = np.column_stack(
            (
                0.45 + 0.16 * np.cos(indices * 0.07),
                0.51 + 0.22 * np.sin(indices * 0.09),
                0.04 * np.sin(indices * 0.13),
            )
        )
        transformed = landmarks.copy()
        transformed[:, :2] = transformed[:, :2] * 2.75 + (1.2, -0.8)
        transformed[:, 2] = transformed[:, 2] * 4.0 + 0.7

        first = structured.normalized_landmark_features(
            landmarks,
            feature_kind=structured.LANDMARK_FEATURE_FULL468,
        )
        second = structured.normalized_landmark_features(
            transformed,
            feature_kind=structured.LANDMARK_FEATURE_FULL468,
        )

        np.testing.assert_allclose(first, second, atol=1e-5)
        self.assertEqual(first.shape, (468 * 3,))
        self.assertTrue(np.all(np.isfinite(first)))

    def test_landmark_features_reject_unknown_schema(self):
        landmarks = np.zeros((478, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "feature kind"):
            structured.normalized_landmark_features(
                landmarks,
                feature_kind="unknown-v1",
            )

    def test_landmark_feature_dimensions_match_the_structured_contract(self):
        self.assertEqual(
            structured.landmark_feature_dimension(
                structured.LANDMARK_FEATURE_DLIB68
            ),
            68 * 3,
        )
        self.assertEqual(
            structured.landmark_feature_dimension(
                structured.LANDMARK_FEATURE_DLIB68_BLENDSHAPES52
            ),
            68 * 3 + 52,
        )
        self.assertEqual(
            structured.landmark_feature_dimension(
                structured.LANDMARK_FEATURE_FULL468
            ),
            468 * 3,
        )

    def test_blendshape_schema_appends_52_bounded_features(self):
        indices = np.arange(478, dtype=np.float64)
        landmarks = np.column_stack(
            (
                0.45 + 0.16 * np.cos(indices * 0.07),
                0.51 + 0.22 * np.sin(indices * 0.09),
                0.04 * np.sin(indices * 0.13),
            )
        )
        scores = np.linspace(0.0, 1.0, 52, dtype=np.float32)

        features, stats = structured.structured_face_features(
            landmarks,
            feature_kind=(
                structured.LANDMARK_FEATURE_DLIB68_BLENDSHAPES52
            ),
            blendshape_names=BLENDSHAPE_NAMES,
            blendshape_scores=scores,
        )

        self.assertEqual(features.shape, (68 * 3 + 52,))
        np.testing.assert_array_equal(features[-52:], scores)
        self.assertTrue(stats["blendshapes_enabled"])

    def test_pose_schema_appends_validated_rotation_without_translation(self):
        indices = np.arange(478, dtype=np.float64)
        landmarks = np.column_stack(
            (
                0.45 + 0.16 * np.cos(indices * 0.07),
                0.51 + 0.22 * np.sin(indices * 0.09),
                0.04 * np.sin(indices * 0.13),
            )
        )
        angle = np.deg2rad(24.0)
        matrix = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle), 12.0],
                [0.0, 1.0, 0.0, -3.0],
                [-np.sin(angle), 0.0, np.cos(angle), -90.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        features, stats = structured.structured_face_features(
            landmarks,
            feature_kind=(
                structured.LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9
            ),
            blendshape_names=BLENDSHAPE_NAMES,
            blendshape_scores=np.zeros(52, dtype=np.float32),
            facial_transformation_matrix=matrix,
        )

        self.assertEqual(features.shape, (68 * 3 + 52 + 9,))
        np.testing.assert_allclose(features[-9:], matrix[:3, :3].reshape(-1))
        self.assertTrue(stats["pose_transform_enabled"])

    def test_pose_schema_rejects_reflection(self):
        landmarks = np.column_stack(
            (
                np.linspace(0.2, 0.8, 478),
                np.linspace(0.8, 0.2, 478),
                np.linspace(-0.1, 0.1, 478),
            )
        )
        matrix = np.eye(4)
        matrix[0, 0] = -1.0

        with self.assertRaisesRegex(ValueError, "not rigid"):
            structured.structured_face_features(
                landmarks,
                feature_kind=(
                    structured.LANDMARK_FEATURE_DLIB68_BLENDSHAPES52_POSE9
                ),
                blendshape_names=BLENDSHAPE_NAMES,
                blendshape_scores=np.zeros(52, dtype=np.float32),
                facial_transformation_matrix=matrix,
            )

    def test_identity_split_audit_rejects_overlap(self):
        clean = structured.audit_identity_splits(
            [
                {"identity_group": "train-a", "split": "train"},
                {"identity_group": "validation-a", "split": "validation"},
                {"identity_group": "sealed-a", "split": "sealed"},
            ]
        )
        self.assertTrue(clean["passed"])

        overlap = structured.audit_identity_splits(
            [
                {"identity_group": "shared", "split": "train"},
                {"identity_group": "shared", "split": "validation"},
                {"identity_group": "sealed-a", "split": "sealed"},
            ]
        )
        self.assertFalse(overlap["passed"])
        self.assertIn("shared", overlap["conflicting_identities"])

    def test_ridge_head_fits_a_bounded_linear_target(self):
        features = np.asarray(
            [
                [-2.0, 1.0],
                [-1.0, 0.5],
                [0.0, 0.0],
                [1.0, -0.5],
                [2.0, -1.0],
            ],
            dtype=np.float64,
        )
        targets = np.column_stack(
            (features[:, 0] * 0.4, features[:, 1] * -0.6)
        )
        head = structured.fit_ridge_head(
            features,
            targets,
            target_mean=np.zeros(2),
            components=np.eye(2),
            rank=2,
            alpha=1e-6,
            sample_weights=np.ones(len(features)),
        )

        predicted = head.predict_targets(features)

        np.testing.assert_allclose(predicted, targets, atol=1e-5)
        self.assertEqual(head.rank, 2)

    def test_zero_rank_head_emits_training_mean(self):
        features = np.asarray([[0.0], [1.0], [2.0]])
        targets = np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        mean = np.asarray([2.0, 3.0])
        head = structured.fit_ridge_head(
            features,
            targets,
            target_mean=mean,
            components=np.eye(2),
            rank=0,
            alpha=1.0,
            sample_weights=np.ones(len(features)),
        )

        predicted = head.predict_targets(np.asarray([[10.0], [-4.0]]))

        np.testing.assert_array_equal(predicted, np.broadcast_to(mean, (2, 2)))
        self.assertEqual(head.rank, 0)

    def test_checkpoint_head_clips_scores_before_decoding(self):
        arrays = {
            "feature_mean": np.asarray([0.0, 0.0]),
            "feature_scale": np.asarray([1.0, 2.0]),
            "target_mean": np.asarray([0.5, -0.5]),
            "components": np.asarray([[1.0, 2.0]]),
            "score_mean": np.asarray([0.0]),
            "weights": np.asarray([[2.0], [0.0]]),
            "score_minimum": np.asarray([-1.0]),
            "score_maximum": np.asarray([1.0]),
        }

        target, scores = predict_checkpoint_head(
            np.asarray([10.0, 4.0]),
            arrays,
        )

        np.testing.assert_array_equal(scores, np.asarray([1.0], dtype=np.float32))
        np.testing.assert_array_equal(
            target,
            np.asarray([1.5, 1.5], dtype=np.float32),
        )

    def test_structured_only_checkpoint_skips_image_encoder_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "structured-only.npz"
            self._write_checkpoint(
                checkpoint,
                encoder_kind=structured.FEATURE_ENCODER_NONE,
            )
            with mock.patch(
                "backend.benchmark.gnm_structured_geometry_provider."
                "GNMMeanFaceFoundation"
            ) as geometry:
                provider = StructuredGNMFaceFoundation(
                    checkpoint,
                    expected_checkpoint_sha256=structured._sha256(checkpoint),
                    device="cpu",
                )

            geometry.assert_called_once_with(load_geometry_bases=True)
            self.assertEqual(
                provider.feature_encoder_kind,
                structured.FEATURE_ENCODER_NONE,
            )
            embedding, stats = provider._embedding(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.ones((8, 8), dtype=np.uint8),
            )
            self.assertEqual(embedding.shape, (0,))
            self.assertEqual(stats["peak_vram_gib"], 0.0)

    def test_checkpoint_rejects_encoder_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bad-encoder-dimension.npz"
            self._write_checkpoint(
                checkpoint,
                encoder_kind=structured.FEATURE_ENCODER_NONE,
            )
            with np.load(checkpoint, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["metadata_feature_encoder_dimension"] = np.asarray(
                1, dtype=np.int32
            )
            np.savez(checkpoint, **arrays)

            with self.assertRaisesRegex(ValueError, "encoder dimension"):
                StructuredGNMFaceFoundation(
                    checkpoint,
                    expected_checkpoint_sha256=structured._sha256(checkpoint),
                    device="cpu",
                )

    def test_checkpoint_rejects_future_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "future-schema.npz"
            self._write_checkpoint(
                checkpoint,
                encoder_kind=structured.FEATURE_ENCODER_NONE,
            )
            with np.load(checkpoint, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["metadata_schema_version"] = np.asarray(
                structured.CHECKPOINT_SCHEMA_VERSION + 1,
                dtype=np.int32,
            )
            np.savez(checkpoint, **arrays)

            with self.assertRaisesRegex(ValueError, "unsupported schema"):
                StructuredGNMFaceFoundation(
                    checkpoint,
                    expected_checkpoint_sha256=structured._sha256(checkpoint),
                    device="cpu",
                )

    def test_checkpoint_rejects_missing_schema_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing-schema-field.npz"
            self._write_checkpoint(
                checkpoint,
                encoder_kind=structured.FEATURE_ENCODER_NONE,
            )
            with np.load(checkpoint, allow_pickle=False) as source:
                arrays = {
                    name: source[name]
                    for name in source.files
                    if name != "metadata_mediapipe_version"
                }
            np.savez(checkpoint, **arrays)

            with self.assertRaisesRegex(ValueError, "schema-v4 metadata"):
                StructuredGNMFaceFoundation(
                    checkpoint,
                    expected_checkpoint_sha256=structured._sha256(checkpoint),
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
