import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

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

    def test_geometry_target_emission_rejects_mixed_and_non_train_rows(self):
        train_row = next(
            row for row in corpus.TRAINING_MATRIX if row.split == "train"
        )
        validation_row = next(
            row for row in corpus.TRAINING_MATRIX if row.split == "validation"
        )
        sealed_row = next(
            row for row in corpus.TRAINING_MATRIX if row.split == "sealed"
        )
        selections = {
            "mixed": (train_row, validation_row),
            "non_train": (sealed_row,),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, selected in selections.items():
                with self.subTest(selection=name):
                    output = root / name
                    with patch.object(
                        corpus,
                        "preflight_gnm_root",
                        return_value={"runnable": True},
                    ):
                        with patch.object(
                            corpus,
                            "select_training_rows",
                            return_value=selected,
                        ):
                            with patch.object(
                                corpus,
                                "_load_gnm_model",
                            ) as load_model:
                                with self.assertRaisesRegex(
                                    ValueError,
                                    "every selected row must have split='train'",
                                ):
                                    corpus.render_training_slice(
                                        root / "gnm",
                                        output,
                                        emit_geometry_targets=True,
                                    )
                    load_model.assert_not_called()
                    self.assertFalse(output.exists())

    def test_geometry_target_contract_pins_generator_layout_and_assets(self):
        contract = corpus._geometry_target_contract()

        self.assertEqual(
            contract["version"],
            corpus.GNM_GEOMETRY_TARGET_CONTRACT_VERSION,
        )
        self.assertTrue(contract["enabled"])
        self.assertTrue(contract["training_only"])
        self.assertEqual(
            contract["provenance"],
            {
                "provider": "google-gnm-head-v3",
                "gnm_revision": corpus.GNM_SOURCE_REVISION,
                "model_sha256": corpus.GNM_MODEL_SHA256,
                "identity_decoder_sha256": (
                    corpus.GNM_IDENTITY_DECODER_SHA256
                ),
                "expression_decoder_sha256": (
                    corpus.GNM_EXPRESSION_DECODER_SHA256
                ),
            },
        )
        self.assertEqual(
            contract["rng"]["bit_generator"],
            "numpy.PCG64",
        )
        self.assertEqual(contract["rng"]["latent_dtype"], "<f4")
        self.assertEqual(contract["target_layout"]["dtype"], "<f4")
        self.assertEqual(
            contract["target_layout"]["dimensions"],
            {
                "identity": corpus.GNM_IDENTITY_DIMENSION,
                "expression": corpus.GNM_EXPRESSION_DIMENSION,
                "combined": corpus.GNM_GEOMETRY_TARGET_DIMENSION,
            },
        )
        self.assertEqual(
            contract["target_layout"]["block_slices"],
            {
                "identity": [0, corpus.GNM_IDENTITY_DIMENSION],
                "expression": [
                    corpus.GNM_IDENTITY_DIMENSION,
                    corpus.GNM_GEOMETRY_TARGET_DIMENSION,
                ],
            },
        )
        self.assertEqual(
            contract["target_layout"]["coefficient_blocks"]["identity"],
            {
                "head": [0, 170],
                "eyes": [170, 173],
                "teeth": [173, 253],
            },
        )
        self.assertEqual(
            contract["target_layout"]["coefficient_blocks"]["expression"],
            {
                "left_eye": [0, 100],
                "right_eye": [100, 200],
                "lower_face": [200, 350],
                "tongue": [350, 382],
                "pupils": [382, 383],
            },
        )

    def test_geometry_targets_are_deterministic_and_auditable(self):
        identity = np.linspace(
            -1.0,
            1.0,
            corpus.GNM_IDENTITY_DIMENSION,
            dtype=np.float32,
        )
        expression = np.linspace(
            0.5,
            -0.5,
            corpus.GNM_EXPRESSION_DIMENSION,
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = corpus._write_geometry_targets(
                root / "first",
                root,
                identity,
                expression,
            )
            second = corpus._write_geometry_targets(
                root / "second",
                root,
                identity,
                expression,
            )

            self.assertEqual(
                first["identity"]["sha256"],
                second["identity"]["sha256"],
            )
            self.assertEqual(
                first["expression"]["sha256"],
                second["expression"]["sha256"],
            )
            first_identity = np.load(root / first["identity"]["path"])
            first_expression = np.load(root / first["expression"]["path"])
            self.assertEqual(
                first_identity.shape,
                (corpus.GNM_IDENTITY_DIMENSION,),
            )
            self.assertEqual(
                first_expression.shape,
                (corpus.GNM_EXPRESSION_DIMENSION,),
            )
            self.assertEqual(first_identity.dtype.str, "<f4")
            self.assertEqual(first_expression.dtype.str, "<f4")
            np.testing.assert_array_equal(first_identity, identity)
            np.testing.assert_array_equal(first_expression, expression)

    def test_training_supervision_is_private_and_hash_deterministic(self):
        spec = next(
            row for row in corpus.TRAINING_MATRIX if row.split == "train"
        )
        identity = np.linspace(
            -1.0,
            1.0,
            corpus.GNM_IDENTITY_DIMENSION,
            dtype=np.float32,
        )
        expression = np.linspace(
            0.5,
            -0.5,
            corpus.GNM_EXPRESSION_DIMENSION,
            dtype=np.float32,
        )
        rendered = (
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4), 255, dtype=np.uint8),
            np.zeros((4, 4), dtype=np.float32),
            {"left_eye": np.eye(4, dtype=bool)},
            {
                "source_mesh_vertices": 10,
                "mesh_path": "private/source_mesh.obj",
            },
            identity,
            expression,
        )
        preflight = {
            "runnable": True,
            "root": "private/gnm/source",
            "assets": {"model": {"path": "private/gnm_head.npz"}},
        }

        def render(output: Path) -> dict:
            with patch.object(
                corpus,
                "preflight_gnm_root",
                return_value=preflight,
            ):
                with patch.object(
                    corpus,
                    "select_training_rows",
                    return_value=(spec,),
                ):
                    with patch.object(
                        corpus,
                        "_load_gnm_model",
                        return_value=object(),
                    ):
                        with patch.object(
                            corpus,
                            "_face_part_weights",
                            return_value={},
                        ):
                            with patch.object(
                                corpus,
                                "_render_row",
                                return_value=rendered,
                            ):
                                return corpus.render_training_slice(
                                    output / "gnm",
                                    output,
                                    split="train",
                                    limit=1,
                                    emit_geometry_targets=True,
                                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_summary = render(root / "first")
            second_summary = render(root / "second")
            first_record = first_summary["training_supervision"]
            second_record = second_summary["training_supervision"]
            first_path = root / "first" / first_record["path"]
            second_path = root / "second" / second_record["path"]
            first_manifest = json.loads(first_path.read_text(encoding="utf-8"))

            self.assertEqual(first_record["path"], "training_supervision.json")
            self.assertEqual(first_record["sha256"], corpus._sha256(first_path))
            self.assertEqual(first_record["sha256"], second_record["sha256"])
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(
                first_summary["geometry_target_contract"],
                first_manifest["geometry_target_contract"],
            )
            self.assertNotIn("geometry_targets", first_summary["rows"][0])
            self.assertEqual(
                set(first_manifest),
                {"schema_version", "geometry_target_contract", "rows"},
            )
            manifest_row = first_manifest["rows"][0]
            self.assertEqual(set(manifest_row), {"row_id", "spec", "targets"})
            self.assertEqual(manifest_row["row_id"], spec.row_id)
            self.assertEqual(manifest_row["spec"]["split"], "train")
            for target in manifest_row["targets"].values():
                self.assertEqual(set(target), {"path", "sha256"})
                target_path = root / "first" / target["path"]
                self.assertEqual(target["sha256"], corpus._sha256(target_path))

            def nested_keys(value):
                if isinstance(value, dict):
                    keys = set(value)
                    for child in value.values():
                        keys.update(nested_keys(child))
                    return keys
                if isinstance(value, list):
                    keys = set()
                    for child in value:
                        keys.update(nested_keys(child))
                    return keys
                return set()

            forbidden = {
                "source",
                "selection",
                "selection_mask",
                "exact_depth",
                "exact_face_parts",
                "part_masks",
                "mesh",
                "mesh_path",
            }
            self.assertFalse(forbidden & nested_keys(first_manifest))
            self.assertNotIn(
                "private/source_mesh.obj",
                first_path.read_text(encoding="utf-8"),
            )

    def test_geometry_target_validation_fails_closed(self):
        class Model:
            identity_dim = corpus.GNM_IDENTITY_DIMENSION
            expression_dim = corpus.GNM_EXPRESSION_DIMENSION

        with self.assertRaisesRegex(ValueError, "identity target"):
            corpus._validate_geometry_targets(
                np.zeros(corpus.GNM_IDENTITY_DIMENSION - 1),
                np.zeros(corpus.GNM_EXPRESSION_DIMENSION),
                model=Model(),
            )
        invalid_expression = np.zeros(corpus.GNM_EXPRESSION_DIMENSION)
        invalid_expression[0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            corpus._validate_geometry_targets(
                np.zeros(corpus.GNM_IDENTITY_DIMENSION),
                invalid_expression,
                model=Model(),
            )


if __name__ == "__main__":
    unittest.main()
