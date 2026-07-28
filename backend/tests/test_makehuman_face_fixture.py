import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from backend.benchmark.makehuman_face_fixture import (
    FACE_PART_NAMES,
    MAKEHUMAN_PROFILES,
    MAKEHUMAN_SOURCE_COMMIT,
    _verify_source_commit,
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh


ASSET_DIR = (
    Path(__file__).parents[1] / "benchmark" / "assets" / "makehuman_cc0_heads"
)


class MakeHumanFaceFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_makehuman_face_fixture(ASSET_DIR)

    def test_manifest_pins_cc0_source_and_derived_archive(self):
        manifest = self.fixture["manifest"]
        self.assertEqual(manifest["license"], "CC0-1.0")
        self.assertEqual(manifest["source"]["commit"], MAKEHUMAN_SOURCE_COMMIT)
        self.assertEqual(manifest["fixture"]["vertex_count"], 4662)
        self.assertEqual(manifest["fixture"]["face_count"], 9262)
        self.assertEqual(
            set(self.fixture["profiles"]),
            {profile.name for profile in MAKEHUMAN_PROFILES},
        )
        license_text = (ASSET_DIR / "LICENSE.ASSETS.md").read_text(encoding="utf-8")
        self.assertIn("Creative Commons CC0 1.0 Universal", license_text)
        self.assertTrue((ASSET_DIR / "PROVENANCE.md").exists())

    @patch("backend.benchmark.makehuman_face_fixture.subprocess.run")
    def test_builder_rejects_checkout_revision_mismatch(self, run_mock):
        run_mock.return_value = Mock(stdout="0" * 40 + "\n")

        with self.assertRaisesRegex(ValueError, "source revision mismatch"):
            _verify_source_commit(Path("makehuman-checkout"))

    @patch("backend.benchmark.makehuman_face_fixture.subprocess.run")
    def test_builder_accepts_exact_checkout_revision(self, run_mock):
        run_mock.return_value = Mock(stdout=MAKEHUMAN_SOURCE_COMMIT + "\n")

        self.assertEqual(
            _verify_source_commit(Path("makehuman-checkout")),
            MAKEHUMAN_SOURCE_COMMIT,
        )
        run_mock.assert_called_once_with(
            ["git", "-C", "makehuman-checkout", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_archive_has_deterministic_order_and_timestamps(self):
        manifest = self.fixture["manifest"]
        archive_path = ASSET_DIR / manifest["fixture"]["file"]
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertTrue(names)
            self.assertTrue(
                all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
            )

    def test_profiles_share_topology_but_not_geometry(self):
        profiles = list(self.fixture["profiles"].values())
        faces = np.asarray(profiles[0]["mesh"].faces)
        geometry = []
        for profile in profiles:
            np.testing.assert_array_equal(faces, profile["mesh"].faces)
            geometry.append(np.asarray(profile["mesh"].vertices))
        self.assertTrue(
            all(not np.array_equal(geometry[0], candidate) for candidate in geometry[1:])
        )
        self.assertEqual(set(self.fixture["part_weights"]), set(FACE_PART_NAMES))
        self.assertTrue(
            all(np.count_nonzero(weight) >= 400 for weight in self.fixture["part_weights"].values())
        )

    def test_perspective_render_has_exact_visible_parts_and_nonflat_color(self):
        profile = self.fixture["profiles"]["caucasian_female_smile"]
        colors = make_profile_vertex_colors(
            profile["mesh"].vertices,
            profile["skin_tone"],
            self.fixture["part_weights"],
            self.fixture["surface_weights"],
        )
        result = render_mesh(
            profile["mesh"],
            CameraSpec(azimuth_deg=0.0, elevation_deg=0.0),
            RenderConfig(
                size=192,
                projection="perspective",
                perspective_fov_y_deg=32.0,
                camera_distance=3.2,
                ambient=0.42,
                diffuse=0.53,
                specular=0.035,
                shininess=48.0,
            ),
            (180, 120, 100),
            vertex_part_weights=self.fixture["part_weights"],
            vertex_colors=colors,
        )

        self.assertGreater(np.count_nonzero(result.silhouette), 5000)
        self.assertGreater(float(np.std(result.rgb[result.silhouette])), 0.08)
        self.assertGreater(float(np.ptp(result.depth[result.silhouette])), 0.95)
        self.assertEqual(set(result.part_masks), set(FACE_PART_NAMES))
        for mask in result.part_masks.values():
            self.assertGreater(np.count_nonzero(mask), 100)
            self.assertTrue(np.all(mask <= result.silhouette))

    def test_manifest_is_machine_readable_without_fixture_loader(self):
        manifest = json.loads((ASSET_DIR / "asset.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["fixture"]["part_support_vertices"]["nose"], 730)


if __name__ == "__main__":
    unittest.main()
