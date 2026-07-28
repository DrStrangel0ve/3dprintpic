import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from backend.benchmark import hsrd_face_training_corpus as hsrd
from backend.face_depth_refinement import FACE_PART_NAMES


class HSRDFaceTrainingCorpusTests(unittest.TestCase):
    def test_selected_specs_are_identity_disjoint_and_exclude_hat_geometry(self):
        specs = hsrd.selected_specs()
        self.assertEqual(len(specs), 9)
        self.assertEqual(len({spec.person_id for spec in specs}), 9)
        self.assertEqual(len({spec.pose_id for spec in specs}), 9)
        self.assertEqual(
            {spec.split for spec in specs},
            {"train", "validation", "sealed"},
        )
        split_people = {
            split: {spec.person_id for spec in specs if spec.split == split}
            for split in ("train", "validation", "sealed")
        }
        self.assertFalse(split_people["train"] & split_people["validation"])
        self.assertFalse(split_people["train"] & split_people["sealed"])
        self.assertFalse(split_people["validation"] & split_people["sealed"])
        self.assertIn("HSR0161", hsrd.EXCLUDED_IDENTITIES)
        self.assertIn("hat", hsrd.EXCLUDED_IDENTITIES["HSR0161"].lower())

    def test_selected_specs_reject_unknown_and_duplicate_pose_ids(self):
        with self.assertRaisesRegex(ValueError, "Unknown HSRD pose"):
            hsrd.selected_specs(["not-a-pose"])
        pose_id = hsrd.IDENTITY_SPECS[0].pose_id
        with self.assertRaisesRegex(ValueError, "duplicates"):
            hsrd.selected_specs([pose_id, pose_id])
        with self.assertRaisesRegex(ValueError, "positive"):
            hsrd.selected_specs(limit=0)

    def test_obj_parser_preserves_direct_rgb_vertex_order(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text(
                "v 1 2 3 0.1 0.2 0.3\n"
                "v 4 5 6 0.4 0.5 0.6\n"
                "f 1 2 1\n",
                encoding="utf-8",
            )
            vertices, colors = hsrd._obj_vertices_and_colors(path)
            np.testing.assert_allclose(vertices, [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_allclose(
                colors,
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            )

    def test_obj_parser_rejects_missing_or_invalid_colors(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text("v 1 2 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "omitted direct RGB"):
                hsrd._obj_vertices_and_colors(path)
            path.write_text("v 1 2 3 1.2 0.2 0.3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
                hsrd._obj_vertices_and_colors(path)

    def test_safe_extract_rejects_path_traversal(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.txt", "no")
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                hsrd._safe_extract_archive(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())

    def test_extract_identity_archive_checks_hash_and_required_files(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            pose_id = "HSRTEST-Body-001"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr(f"{pose_id}_LOD1.obj", "v 0 0 0 0 0 0\n")
                output.writestr(f"{pose_id}_LOD1_u0_v0_diffuse.png", b"png")
                output.writestr("person_metadata.json", "{}")
                output.writestr("pose_metadata.json", "{}")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            spec = hsrd.HSRDIdentitySpec(
                "HSRTEST",
                pose_id,
                "train",
                archive.stat().st_size,
                digest,
            )
            obj_path = hsrd.extract_identity_archive(archive, root / "out", spec)
            self.assertTrue(obj_path.is_file())
            obj_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the pinned archive"):
                hsrd.extract_identity_archive(archive, root / "out", spec)
            wrong = hsrd.HSRDIdentitySpec(
                "HSRTEST",
                pose_id,
                "train",
                archive.stat().st_size,
                "0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "hash changed"):
                hsrd.extract_identity_archive(archive, root / "other", wrong)

    def test_preflight_verifies_manifests_archives_and_split_contract(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_root = root / "manifest-root"
            dataset_root = root / "dataset-root"
            manifest_path = manifest_root / "README.md"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("license fixture\n", encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            pose_id = "HSRTEST-Body-001"
            archive_path = (
                dataset_root
                / "data"
                / "HSRTEST"
                / pose_id
                / "scans"
                / f"{pose_id}-Scan-LOD1.zip"
            )
            archive_path.parent.mkdir(parents=True)
            archive_path.write_bytes(b"archive fixture")
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            spec = hsrd.HSRDIdentitySpec(
                "HSRTEST",
                pose_id,
                "train",
                archive_path.stat().st_size,
                archive_hash,
            )
            with patch.object(
                hsrd,
                "MANIFEST_HASHES",
                {"README.md": manifest_hash},
            ):
                result = hsrd.hsrd_preflight(
                    manifest_root,
                    dataset_root,
                    specs=(spec,),
                )
            self.assertTrue(result["ready"])
            self.assertTrue(all(result["checks"].values()))
            self.assertEqual(result["revision"], hsrd.HSRD_REVISION)
            self.assertEqual(result["license"], "CC BY 4.0")

    def test_resize_valid_field_does_not_bleed_invalid_values(self):
        values = np.full((16, 16), np.nan, dtype=np.float32)
        valid = np.zeros((16, 16), dtype=bool)
        values[4:12, 5:11] = 2.5
        valid[4:12, 5:11] = True
        resized, support, _ = hsrd._resize_valid_field(
            values,
            valid,
            scale=2.0,
            dimension=48,
            center_xy=(8.0, 8.0),
            output_center_xy=(24.0, 24.0),
        )
        self.assertTrue(np.allclose(resized[support], 2.5))
        self.assertTrue(np.all(np.isnan(resized[~support])))
        self.assertGreater(np.count_nonzero(support), np.count_nonzero(valid))

    def test_resize_valid_field_does_not_invent_occlusion_depths(self):
        values = np.ones((16, 16), dtype=np.float32)
        values[:, 8:] = 4.0
        resized, support, _ = hsrd._resize_valid_field(
            values,
            np.ones_like(values, dtype=bool),
            scale=1.7,
            dimension=40,
            center_xy=(8.0, 8.0),
            output_center_xy=(20.0, 20.0),
        )
        self.assertEqual(set(np.unique(resized[support]).tolist()), {1.0, 4.0})

    def test_transformed_intrinsics_follow_opencv_half_pixel_centers(self):
        transform = {
            "resized_shape": [192, 192],
            "origin_xy": [7, 11],
        }
        intrinsics = hsrd._transformed_intrinsics(transform)
        scale = 192 / hsrd.RENDER_SIZE
        principal = 0.5 * (hsrd.RENDER_SIZE - 1)
        self.assertAlmostEqual(
            intrinsics[0, 2],
            7 + (principal + 0.5) * scale - 0.5,
        )
        self.assertAlmostEqual(
            intrinsics[1, 2],
            11 + (principal + 0.5) * scale - 0.5,
        )

    def test_view_gate_requires_rate_and_both_yaw_signs_per_split(self):
        specs = tuple(
            hsrd.HSRDIdentitySpec(f"person-{index}", f"pose-{index}", split, 1, "a" * 64)
            for index, split in enumerate(("train", "validation", "sealed"))
        )
        rows = []
        for spec in specs:
            for yaw in hsrd.CAMERA_YAWS_DEGREES:
                for height in hsrd.TARGET_FACE_HEIGHTS:
                    rows.append(
                        {
                            "identity_group": spec.person_id,
                            "split": spec.split,
                            "rendering": {"camera_yaw_degrees": yaw},
                            "height": height,
                        }
                    )
        contract = hsrd._view_coverage_contract(rows, specs)
        self.assertTrue(contract["rate_passed"])
        self.assertTrue(contract["split_yaw_sign_coverage_passed"])
        positive_only = [
            row for row in rows if row["rendering"]["camera_yaw_degrees"] > 0
        ]
        contract = hsrd._view_coverage_contract(positive_only, specs)
        self.assertFalse(contract["rate_passed"])
        self.assertFalse(contract["split_yaw_sign_coverage_passed"])

    def test_camera_normals_are_finite_on_an_interior_plane(self):
        rows, columns = np.indices((12, 12), dtype=np.float32)
        depth = 2.0 + 0.01 * columns + 0.02 * rows
        valid = np.ones_like(depth, dtype=bool)
        intrinsics = np.asarray(((30, 0, 5.5), (0, 30, 5.5), (0, 0, 1)))
        normals = hsrd._camera_space_normals(depth, valid, intrinsics)
        self.assertTrue(np.all(np.isfinite(normals[2:-2, 2:-2])))
        lengths = np.linalg.norm(normals[2:-2, 2:-2], axis=-1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)
        self.assertTrue(np.all(np.isnan(normals[0])))

    def test_scale_solver_places_both_requested_face_heights_exactly(self):
        bbox = [105, 109, 227, 240]
        center = (166.0, 174.5)
        output_center = (128.0, 0.46 * 256)
        for target in hsrd.TARGET_FACE_HEIGHTS:
            scale = hsrd._exact_face_height_scale(
                bbox,
                target,
                center_xy=center,
                output_center_xy=output_center,
            )
            origin_y = int(round(output_center[1] - center[1] * scale))
            height = int(round(origin_y + bbox[3] * scale)) - int(
                round(origin_y + bbox[1] * scale)
            )
            self.assertEqual(height, target)

    def test_emit_row_preserves_parts_camera_depth_and_target_height(self):
        from tempfile import TemporaryDirectory

        source = np.full((hsrd.RENDER_SIZE, hsrd.RENDER_SIZE, 3), 0.3, dtype=np.float32)
        source[70:310, 90:290] = (0.45, 0.30, 0.22)
        silhouette = np.zeros((hsrd.RENDER_SIZE, hsrd.RENDER_SIZE), dtype=bool)
        silhouette[70:310, 90:290] = True
        rows, columns = np.indices(silhouette.shape, dtype=np.float32)
        surface_z = np.full(silhouette.shape, np.nan, dtype=np.float32)
        surface_z[silhouette] = -(
            2.0 + 0.0005 * rows[silhouette] + 0.0007 * columns[silhouette]
        )
        face_mask = np.zeros_like(silhouette)
        face_mask[100:250, 120:260] = True
        part_masks = {}
        for index, name in enumerate(FACE_PART_NAMES):
            part = np.zeros_like(silhouette)
            top = 125 + index * 12
            part[top : top + 8, 145:220] = True
            part_masks[name] = part
        region = {
            "bbox": [120, 100, 260, 250],
            "face_mask": face_mask,
            "part_masks": part_masks,
        }
        rendered = SimpleNamespace(
            rgb=source,
            surface_z=surface_z,
            silhouette=silhouette,
        )
        spec = hsrd.HSRDIdentitySpec(
            "HSRTEST",
            "HSRTEST-Body-001",
            "train",
            1,
            "a" * 64,
        )
        with TemporaryDirectory() as directory, patch.object(
            hsrd,
            "_select_face_region",
            return_value=(region, []),
        ):
            root = Path(directory)
            for yaw, height in ((-35.0, 74), (35.0, 75)):
                row = hsrd._emit_row(
                    rendered,
                    spec,
                    yaw,
                    height,
                    root,
                    {"head_vertices": 10_000, "head_faces": 20_000},
                )
                self.assertEqual(
                    row["selection_geometry"]["face_bbox_height_pixels"], height
                )
                camera_z = np.load(root / row["exact_camera_depth"]["path"])
                selection = np.asarray(
                    Image.open(root / row["selection_mask"]["path"])
                ) > 0
                self.assertTrue(np.all(np.isfinite(camera_z[selection])))
                self.assertTrue(np.all(np.isnan(camera_z[~selection])))
                self.assertGreater(float(np.ptp(camera_z[selection])), 0.0)
                for record in row["exact_face_parts"].values():
                    self.assertGreater(
                        np.count_nonzero(Image.open(root / record["path"])),
                        0,
                    )
                self.assertEqual(
                    row["selection_geometry"]["camera"]["projection"],
                    "perspective-opencv",
                )
                self.assertEqual(
                    row["selection_geometry"]["camera"][
                        "resize_pixel_center_convention"
                    ],
                    "opencv-half-pixel",
                )
                self.assertIn("selection_function_sha256", row["detector"])
                self.assertEqual(row["rendering"]["source_revision"], hsrd.HSRD_REVISION)
                self.assertEqual(row["rendering"]["attribution"], hsrd.HSRD_ATTRIBUTION)


if __name__ == "__main__":
    unittest.main()
