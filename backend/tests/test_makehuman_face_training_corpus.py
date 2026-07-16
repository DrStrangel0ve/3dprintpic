import unittest

from backend.benchmark import makehuman_face_training_corpus as corpus
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FaceSceneSpec,
    VARIED_CONTEXT_MATRIX,
)


class MakeHumanFaceTrainingCorpusTests(unittest.TestCase):
    def test_profiles_cover_identity_disjoint_expression_groups(self):
        self.assertEqual(len(corpus.TRAINING_PROFILES), 32)
        self.assertEqual(len({profile.name for profile in corpus.TRAINING_PROFILES}), 32)
        grouped = {}
        for profile in corpus.TRAINING_PROFILES:
            identity, expression = profile.name.split("__", maxsplit=1)
            grouped.setdefault(identity, set()).add(expression)
            self.assertTrue(profile.targets)
            self.assertTrue(
                all(0.0 < float(scale) <= 1.0 for _, scale in profile.targets)
            )
        self.assertEqual(set(grouped), set(corpus.SPLIT_BY_IDENTITY))
        self.assertTrue(
            all(expressions == set(corpus.EXPRESSIONS) for expressions in grouped.values())
        )

    def test_matrix_has_identity_disjoint_240_40_40_splits(self):
        self.assertEqual(len(corpus.TRAINING_MATRIX), 320)
        counts = {
            split: sum(row.split == split for row in corpus.TRAINING_MATRIX)
            for split in ("train", "validation", "sealed")
        }
        self.assertEqual(counts, {"train": 240, "validation": 40, "sealed": 40})
        identities = {
            split: {
                row.identity_group
                for row in corpus.TRAINING_MATRIX
                if row.split == split
            }
            for split in counts
        }
        self.assertEqual(
            [len(identities[name]) for name in ("train", "validation", "sealed")],
            [6, 1, 1],
        )
        self.assertFalse(identities["train"] & identities["validation"])
        self.assertFalse(identities["train"] & identities["sealed"])
        self.assertFalse(identities["validation"] & identities["sealed"])

    def test_matrix_oversamples_small_turned_faces_and_crosses_context(self):
        small = [
            row
            for row in corpus.TRAINING_MATRIX
            if (
                row.target_dimension == 256
                and row.camera_distance / row.camera_scale >= 6.5
            )
            or (
                row.target_dimension == 384
                and row.camera_distance / row.camera_scale >= 10.0
            )
        ]
        self.assertGreaterEqual(len(small) / len(corpus.TRAINING_MATRIX), 0.60)
        self.assertTrue(all(abs(row.camera_yaw_deg) >= 22.0 for row in small))
        self.assertEqual(
            {row.target_dimension for row in corpus.TRAINING_MATRIX},
            {256, 384},
        )
        self.assertEqual(
            {row.background_profile for row in corpus.TRAINING_MATRIX},
            {"structured_room", "deep_shelves", "layered_studio"},
        )
        self.assertEqual(
            {row.lighting_profile for row in corpus.TRAINING_MATRIX},
            {"soft_left", "side_right", "overhead"},
        )
        self.assertGreaterEqual(
            sum(row.occlusion is not None for row in corpus.TRAINING_MATRIX),
            60,
        )

    def test_development_rows_are_not_semantic_duplicates_of_held_out(self):
        held_out = [
            row for row in VARIED_CONTEXT_MATRIX if isinstance(row, FaceSceneSpec)
        ]
        for candidate in corpus.TRAINING_MATRIX:
            for reference in held_out:
                same_context = (
                    candidate.target_dimension == reference.target_dimension
                    and candidate.background_profile == reference.background_profile
                    and candidate.lighting_profile == reference.lighting_profile
                )
                camera_near_duplicate = (
                    abs(candidate.camera_yaw_deg - reference.camera_yaw_deg) <= 2.0
                    and abs(
                        candidate.camera_distance / candidate.camera_scale
                        - reference.camera_distance / reference.camera_scale
                    )
                    <= 0.25
                )
                self.assertFalse(same_context and camera_near_duplicate)

    def test_builders_are_deterministic(self):
        self.assertEqual(
            corpus.build_training_profiles(),
            corpus.TRAINING_PROFILES,
        )
        self.assertEqual(
            corpus.build_training_matrix(),
            corpus.TRAINING_MATRIX,
        )

    def test_bounded_slice_is_evenly_distributed_not_prefix_truncated(self):
        selected = corpus.select_training_rows(limit=40)
        self.assertEqual(len(selected), 40)
        self.assertEqual(
            {row.split for row in selected},
            {"train", "validation", "sealed"},
        )
        self.assertEqual(
            {row.identity_group for row in selected},
            set(corpus.SPLIT_BY_IDENTITY),
        )
        self.assertEqual(
            {row.expression for row in selected},
            set(corpus.EXPRESSIONS),
        )
        self.assertEqual(len({row.row_id for row in selected}), 40)


if __name__ == "__main__":
    unittest.main()
