import tempfile
import unittest
from collections import Counter
from pathlib import Path

from backend.benchmark import gnm_face_training_corpus as corpus
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FaceSceneSpec,
    VARIED_CONTEXT_MATRIX,
)


class GNMFaceTrainingCorpusTests(unittest.TestCase):
    def test_identity_specs_are_demographically_balanced_and_disjoint(self):
        self.assertEqual(len(corpus.IDENTITIES), 40)
        counts = {
            split: sum(identity.split == split for identity in corpus.IDENTITIES)
            for split in ("train", "validation", "sealed")
        }
        self.assertEqual(counts, {"train": 24, "validation": 8, "sealed": 8})
        for demographic in corpus.DEMOGRAPHICS:
            matches = [
                identity
                for identity in corpus.IDENTITIES
                if (identity.gender, identity.ethnicity) == demographic
            ]
            self.assertEqual(len(matches), 5)
            self.assertEqual(
                {identity.split for identity in matches},
                {"train", "validation", "sealed"},
            )
        grouped = {
            split: {
                identity.identity_group
                for identity in corpus.IDENTITIES
                if identity.split == split
            }
            for split in counts
        }
        self.assertFalse(grouped["train"] & grouped["validation"])
        self.assertFalse(grouped["train"] & grouped["sealed"])
        self.assertFalse(grouped["validation"] & grouped["sealed"])

    def test_matrix_has_identity_disjoint_960_320_320_splits(self):
        self.assertEqual(len(corpus.TRAINING_MATRIX), 1600)
        counts = {
            split: sum(row.split == split for row in corpus.TRAINING_MATRIX)
            for split in ("train", "validation", "sealed")
        }
        self.assertEqual(counts, {"train": 960, "validation": 320, "sealed": 320})
        self.assertEqual(
            {row.expression for row in corpus.TRAINING_MATRIX},
            set(corpus.EXPRESSIONS),
        )
        self.assertEqual(
            len({row.row_id for row in corpus.TRAINING_MATRIX}),
            len(corpus.TRAINING_MATRIX),
        )

    def test_matrix_oversamples_small_turned_faces_and_crosses_context(self):
        small_conditions = {
            condition
            for condition in corpus.SCENE_CONDITIONS
            if (
                condition.target_dimension == 256
                and condition.camera_distance >= 7.6
            )
            or (
                condition.target_dimension == 384
                and condition.camera_distance >= 13.0
            )
        }
        small_rows = [
            row
            for row in corpus.TRAINING_MATRIX
            if any(
                row.target_dimension == condition.target_dimension
                and row.camera_distance == condition.camera_distance
                and row.camera_yaw_deg == condition.camera_yaw_deg
                for condition in small_conditions
            )
        ]
        self.assertGreaterEqual(
            len(small_rows) / len(corpus.TRAINING_MATRIX),
            0.60,
        )
        self.assertTrue(all(abs(row.camera_yaw_deg) >= 30.0 for row in small_rows))
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
        self.assertEqual(
            sum(row.occlusion == "eye_band" for row in corpus.TRAINING_MATRIX),
            640,
        )

    def test_development_conditions_are_not_held_out_near_duplicates(self):
        held_out = [
            row for row in VARIED_CONTEXT_MATRIX if isinstance(row, FaceSceneSpec)
        ]
        for condition in corpus.SCENE_CONDITIONS:
            for reference in held_out:
                same_context = (
                    condition.target_dimension == reference.target_dimension
                    and condition.background_profile
                    == reference.background_profile
                    and condition.lighting_profile == reference.lighting_profile
                )
                camera_near_duplicate = (
                    abs(condition.camera_yaw_deg - reference.camera_yaw_deg)
                    <= 2.0
                    and abs(condition.camera_distance - reference.camera_distance)
                    <= 0.25
                )
                self.assertFalse(same_context and camera_near_duplicate)

    def test_builders_and_bounded_selection_are_deterministic(self):
        self.assertEqual(corpus.build_identity_specs(), corpus.IDENTITIES)
        self.assertEqual(corpus.build_training_matrix(), corpus.TRAINING_MATRIX)
        selected = corpus.select_training_rows(limit=80)
        self.assertEqual(len(selected), 80)
        self.assertEqual(
            {row.split for row in selected},
            {"train", "validation", "sealed"},
        )
        self.assertEqual(
            {row.expression for row in selected},
            set(corpus.EXPRESSIONS),
        )
        self.assertEqual(len({row.row_id for row in selected}), 80)

    def test_identity_stratified_selection_balances_the_80_row_audit(self):
        selected = corpus.select_training_rows(
            limit=80,
            strategy="identity-stratified",
        )
        self.assertEqual(len(selected), 80)
        self.assertEqual(
            Counter(row.split for row in selected),
            {"train": 48, "validation": 16, "sealed": 16},
        )
        self.assertEqual(
            Counter(row.identity_group for row in selected),
            {identity.identity_group: 2 for identity in corpus.IDENTITIES},
        )
        self.assertEqual(
            Counter(row.expression for row in selected),
            {expression: 10 for expression in corpus.EXPRESSIONS},
        )
        self.assertEqual(
            Counter(corpus._scene_condition_index(row) for row in selected),
            {index: 16 for index in range(len(corpus.SCENE_CONDITIONS))},
        )
        small_count = sum(
            (
                row.target_dimension == 256
                and row.camera_distance >= 7.6
            )
            or (
                row.target_dimension == 384
                and row.camera_distance >= 13.0
            )
            for row in selected
        )
        self.assertEqual(small_count, 48)
        self.assertEqual(len({row.row_id for row in selected}), 80)

    def test_novel_identity_selection_is_disjoint_and_balanced(self):
        selected = corpus.select_training_rows(
            split="train",
            limit=32,
            strategy=corpus.NOVEL_IDENTITY_SELECTION_STRATEGY,
        )
        existing_identities = {
            row.identity_group
            for row in corpus.select_training_rows(
                limit=80,
                strategy="identity-stratified",
            )
        }
        counts = Counter(row.identity_group for row in selected)

        self.assertEqual(len(selected), 32)
        self.assertEqual(len(counts), 8)
        self.assertEqual(set(counts.values()), {4})
        self.assertFalse(set(counts) & existing_identities)
        self.assertTrue(all("_v05" in identity for identity in counts))
        self.assertEqual(
            Counter(corpus._scene_condition_index(row) for row in selected),
            {0: 7, 1: 7, 2: 6, 3: 6, 4: 6},
        )

    def test_identity_stratified_selection_rejects_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unsupported row selection"):
            corpus.select_training_rows(limit=8, strategy="unknown")

    def test_preflight_fails_closed_for_missing_official_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            result = corpus.preflight_gnm_root(Path(directory))
        self.assertFalse(result["runnable"])
        self.assertFalse(result["checks"]["source_revision_matches"])
        self.assertFalse(result["checks"]["tracked_source_clean"])
        self.assertFalse(result["checks"]["assets_exist"])
        self.assertFalse(result["checks"]["asset_hashes_match"])


if __name__ == "__main__":
    unittest.main()
