import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

from backend.benchmark import mhr_face_training_corpus as corpus


class MHRFaceTrainingCorpusTests(unittest.TestCase):
    def test_scene_matrix_is_deterministic_balanced_and_identity_disjoint(self):
        first = corpus.build_scene_matrix()
        second = corpus.build_scene_matrix()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({row.row_id for row in first}), 16)
        split_identities = {
            split: {row.identity_group for row in first if row.split == split}
            for split in ("train", "validation", "sealed")
        }
        self.assertEqual([len(split_identities[name]) for name in split_identities], [2, 1, 1])
        self.assertFalse(split_identities["train"] & split_identities["validation"])
        self.assertFalse(split_identities["train"] & split_identities["sealed"])
        self.assertFalse(split_identities["validation"] & split_identities["sealed"])
        for identity in {row.identity_group for row in first}:
            identity_rows = [row for row in first if row.identity_group == identity]
            self.assertEqual(
                {row.face_height_pixels for row in identity_rows},
                set(corpus.FACE_HEIGHTS),
            )
            self.assertEqual(
                {row.camera_yaw_deg for row in identity_rows},
                set(corpus.CAMERA_YAWS),
            )

    def test_limited_selection_is_stratified_and_full_selection_is_exact(self):
        smoke = corpus.select_scene_rows(4)
        self.assertEqual(len(smoke), 4)
        self.assertEqual({row.split for row in smoke}, {"train", "validation", "sealed"})
        self.assertEqual(corpus.select_scene_rows(), corpus.SCENE_MATRIX)
        self.assertEqual(corpus.select_scene_rows(99), corpus.SCENE_MATRIX)

    def test_training_matrix_has_exact_240_40_40_identity_disjoint_contract(self):
        rows = corpus.build_training_matrix()
        self.assertEqual(len(rows), 320)
        split_counts = {
            split: sum(row.split == split for row in rows)
            for split in ("train", "validation", "sealed")
        }
        self.assertEqual(
            split_counts,
            {"train": 240, "validation": 40, "sealed": 40},
        )
        identities = {
            split: {row.identity_group for row in rows if row.split == split}
            for split in split_counts
        }
        self.assertEqual(
            {split: len(values) for split, values in identities.items()},
            {"train": 30, "validation": 5, "sealed": 5},
        )
        self.assertFalse(identities["train"] & identities["validation"])
        self.assertFalse(identities["train"] & identities["sealed"])
        self.assertFalse(identities["validation"] & identities["sealed"])
        for identity in set().union(*identities.values()):
            identity_rows = [row for row in rows if row.identity_group == identity]
            self.assertEqual(len(identity_rows), 8)
            self.assertEqual(
                {row.face_height_pixels for row in identity_rows},
                set(corpus.TRAINING_FACE_HEIGHTS),
            )
            self.assertEqual(
                {row.camera_yaw_deg for row in identity_rows},
                set(corpus.TRAINING_CAMERA_YAWS),
            )
        self.assertEqual(
            corpus.select_scene_rows(matrix_kind="training"),
            corpus.TRAINING_MATRIX,
        )
        smoke = corpus.select_scene_rows(8, matrix_kind="training")
        self.assertEqual(
            {row.face_height_pixels for row in smoke},
            set(corpus.TRAINING_FACE_HEIGHTS),
        )
        self.assertEqual(
            {row.camera_yaw_deg for row in smoke},
            set(corpus.TRAINING_CAMERA_YAWS),
        )
        self.assertEqual(
            {row.split for row in smoke},
            {"train", "validation", "sealed"},
        )

    def test_coefficients_are_deterministic_and_only_head_identity_varies(self):
        neutral = corpus.SCENE_MATRIX[0]
        expressive = corpus.SCENE_MATRIX[1]
        identity_a, expression_a = corpus.coefficients_for_spec(neutral)
        identity_b, expression_b = corpus.coefficients_for_spec(neutral)
        identity_c, expression_c = corpus.coefficients_for_spec(expressive)
        np.testing.assert_array_equal(identity_a, identity_b)
        np.testing.assert_array_equal(expression_a, expression_b)
        np.testing.assert_array_equal(identity_a, identity_c)
        np.testing.assert_array_equal(identity_a[:20], 0.0)
        np.testing.assert_array_equal(identity_a[40:], 0.0)
        self.assertGreater(np.linalg.norm(identity_a[20:40]), 0.0)
        np.testing.assert_array_equal(expression_a, 0.0)
        self.assertGreater(np.linalg.norm(expression_c), 0.0)
        self.assertLessEqual(float(np.max(np.abs(expression_c))), 0.60)

    def test_compact_head_mesh_reindexes_only_selected_faces(self):
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        head_mask = np.asarray((1.0, 1.0, 1.0, 0.0), dtype=np.float32)
        with (
            mock.patch.object(corpus, "MIN_HEAD_VERTICES", 3),
            mock.patch.object(corpus, "MIN_HEAD_FACES", 1),
        ):
            mesh = corpus._compact_head_mesh(vertices, faces, head_mask)
        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.shape, (1, 3))
        self.assertEqual(int(mesh.faces.max()), 2)

    def test_face_part_weights_cover_all_six_regions(self):
        x, y, z = np.meshgrid(
            np.linspace(-1.0, 1.0, 41),
            np.linspace(0.0, 1.0, 41),
            np.asarray((0.45, 0.75, 1.0)),
            indexing="ij",
        )
        mesh = trimesh.Trimesh(
            vertices=np.column_stack((x.ravel(), y.ravel(), z.ravel())),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
        weights = corpus._face_part_weights(mesh)
        self.assertEqual(set(weights), set(corpus.FACE_PART_NAMES))
        for values in weights.values():
            self.assertEqual(values.shape, (len(mesh.vertices),))
            self.assertGreater(np.count_nonzero(values >= 0.5), 0)
            self.assertTrue(np.all(np.isfinite(values)))

    def test_vertex_color_palette_cycles_for_expanded_identity_indices(self):
        mesh = trimesh.creation.icosphere(subdivisions=1)
        first = corpus._vertex_colors(mesh, 0)
        cycled = corpus._vertex_colors(mesh, len(corpus.SKIN_TONES))
        np.testing.assert_array_equal(first, cycled)

    def test_semantic_appearance_is_deterministic_and_distinct_from_clay(self):
        mesh = trimesh.creation.icosphere(subdivisions=3)
        parts = corpus._face_part_weights(mesh)
        clay = corpus._vertex_colors(mesh, 2)
        semantic_a = corpus._vertex_colors(
            mesh,
            2,
            appearance_kind="semantic-procedural",
            part_weights=parts,
        )
        semantic_b = corpus._vertex_colors(
            mesh,
            2,
            appearance_kind="semantic-procedural",
            part_weights=parts,
        )
        np.testing.assert_array_equal(semantic_a, semantic_b)
        self.assertGreater(float(np.max(np.abs(semantic_a - clay))), 0.1)
        self.assertTrue(np.all(np.isfinite(semantic_a)))
        with self.assertRaisesRegex(ValueError, "requires all"):
            corpus._vertex_colors(
                mesh,
                2,
                appearance_kind="semantic-procedural",
                part_weights=None,
            )

    def test_target_height_distance_tracks_image_size_and_requested_height(self):
        small = corpus.SCENE_MATRIX[0]
        larger_frame = corpus.MHRSceneSpec(
            **{**small.__dict__, "target_dimension": 512}
        )
        taller_face = corpus.MHRSceneSpec(
            **{**small.__dict__, "face_height_pixels": 128}
        )
        self.assertGreater(
            corpus._initial_camera_distance(larger_frame),
            corpus._initial_camera_distance(small),
        )
        self.assertLess(
            corpus._initial_camera_distance(taller_face),
            corpus._initial_camera_distance(small),
        )

    def test_camera_normals_keep_invalid_background_non_finite_and_face_camera(self):
        camera_z = np.asarray(
            (
                (np.nan, np.nan, np.nan, np.nan, np.nan),
                (np.nan, 2.0, 2.0, 2.0, np.nan),
                (np.nan, 2.0, 2.0, 2.0, np.nan),
                (np.nan, 2.0, 2.0, 2.0, np.nan),
                (np.nan, np.nan, np.nan, np.nan, np.nan),
            ),
            dtype=np.float32,
        )
        mask = np.isfinite(camera_z)
        intrinsics = np.asarray(
            ((100.0, 0.0, 2.0), (0.0, 100.0, 2.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        normals = corpus._camera_space_normals(camera_z, mask, intrinsics)
        self.assertEqual(normals.shape, (5, 5, 3))
        self.assertTrue(np.all(np.isnan(normals[~mask])))
        self.assertTrue(np.all(np.isfinite(normals[2, 2])))
        np.testing.assert_allclose(normals[2, 2], (0.0, 0.0, -1.0), atol=1e-6)

    def test_camera_record_projects_with_offset_and_expected_yaw_sign(self):
        mesh = trimesh.creation.box(extents=(2.0, 3.0, 1.0))
        spec = corpus.SCENE_MATRIX[0]
        record = corpus._camera_record(mesh, spec, 8.0, -11)
        matrix = np.asarray(record["object_to_camera"], dtype=np.float64)
        intrinsics = np.asarray(record["intrinsics"], dtype=np.float64)
        origin = matrix @ np.asarray((0.0, 0.0, 0.0, 1.0))
        pixel = intrinsics @ (origin[:3] / origin[2])
        self.assertAlmostEqual(pixel[0], intrinsics[0, 2], places=6)
        self.assertAlmostEqual(pixel[1], intrinsics[1, 2], places=6)
        self.assertAlmostEqual(
            intrinsics[0, 2], 0.5 * (spec.target_dimension - 1) - 11
        )
        positive_x = matrix @ np.asarray((1.0, 0.0, 0.0, 1.0))
        self.assertGreater(positive_x[0], origin[0])

    def test_preflight_fails_closed_without_pinned_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            preflight = corpus.preflight_mhr_root(temporary)
        self.assertFalse(preflight["runnable"])
        self.assertFalse(preflight["checks"]["source_revision_matches"])
        self.assertFalse(preflight["checks"]["assets_exist"])
        self.assertFalse(preflight["checks"]["asset_hashes_match"])

    def test_render_preflight_failure_leaves_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "corpus"
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                corpus.render_training_corpus(root / "missing", output, limit=1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
