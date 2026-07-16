import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.vggheads_depth_provider import (
    VGGHEADS_REQUIRED_SOURCE_FILES,
    VGGHEADS_SOURCE_REVISION,
    fill_depth_nearest,
    normalize_front_depth,
    rasterize_projected_mesh_depth,
    small_face_policy,
    subject_interior_taper,
    vggheads_preflight,
)


class VGGHeadsDepthProviderTests(unittest.TestCase):
    def test_preflight_rejects_unpinned_model_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            for relative in VGGHEADS_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            model_path = root / "vgg_heads_l.trcd"
            model_path.write_bytes(b"not-the-pinned-model")
            with patch(
                "backend.benchmark.vggheads_depth_provider._git_output",
                side_effect=(VGGHEADS_SOURCE_REVISION, ""),
            ):
                evidence = vggheads_preflight(
                    provider_root,
                    model_path,
                )

        self.assertTrue(evidence["checks"]["source_revision_pinned"])
        self.assertTrue(evidence["checks"]["source_clean"])
        self.assertTrue(evidence["checks"]["source_files_complete"])
        self.assertFalse(evidence["checks"]["model_size_pinned"])
        self.assertFalse(evidence["checks"]["model_hash_pinned"])
        self.assertFalse(evidence["runnable"])
        self.assertTrue(evidence["license"]["research_only"])

    def test_small_face_policy_fails_closed_for_large_or_occluded_faces(self):
        self.assertTrue(small_face_policy(76)["eligible"])
        self.assertEqual(
            small_face_policy(191)["bypass_reason"],
            "face_too_large",
        )
        self.assertEqual(
            small_face_policy(76, occluded=True)["bypass_reason"],
            "occluded_face",
        )

    def test_subject_taper_is_zero_outside_and_on_attachment_boundary(self):
        mask = np.zeros((15, 15), dtype=np.uint8)
        mask[2:13, 2:13] = 255

        taper = subject_interior_taper(
            mask,
            taper_pixels=4.0,
            boundary_pixels=1.0,
        )

        self.assertEqual(float(np.max(taper[mask == 0])), 0.0)
        self.assertEqual(float(np.max(taper[2, 2:13])), 0.0)
        self.assertEqual(float(np.max(taper[12, 2:13])), 0.0)
        self.assertGreater(float(taper[7, 7]), 0.99)

    def test_nearest_fill_and_normalization_make_front_depth_larger(self):
        depth = np.array(
            [
                [1.0, np.nan, 3.0],
                [1.0, np.nan, 3.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=np.float32,
        )

        filled, fill_stats = fill_depth_nearest(depth)
        normalized, normalization = normalize_front_depth(
            filled,
            lower_percentile=0.0,
            upper_percentile=100.0,
        )

        self.assertTrue(np.all(np.isfinite(filled)))
        self.assertEqual(fill_stats["filled_pixels"], 2)
        self.assertGreater(float(normalized[0, 0]), float(normalized[0, 2]))
        self.assertTrue(normalization["near_is_larger"])

    def test_minimum_z_rasterizer_selects_nearer_overlapping_triangle(self):
        vertices = np.array(
            [
                [1.0, 1.0, 4.0],
                [6.0, 1.0, 4.0],
                [1.0, 6.0, 4.0],
                [1.0, 1.0, 2.0],
                [6.0, 1.0, 2.0],
                [1.0, 6.0, 2.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

        depth, stats = rasterize_projected_mesh_depth(
            vertices,
            faces,
            height=8,
            width=8,
            front_surface="minimum-z",
        )

        self.assertAlmostEqual(float(depth[2, 2]), 2.0)
        self.assertEqual(stats["rendered_faces"], 2)
        self.assertGreater(stats["finite_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
