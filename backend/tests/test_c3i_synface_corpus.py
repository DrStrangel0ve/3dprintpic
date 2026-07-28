import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.benchmark import c3i_synface_corpus as corpus


class C3ISynFaceCorpusTests(unittest.TestCase):
    def test_raw_archive_preflight_requires_exact_size_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / corpus.RAW_ARCHIVE_FILENAME
            path.write_bytes(b"calibrated-exr-archive")
            with (
                mock.patch.object(
                    corpus,
                    "RAW_ARCHIVE_BYTES",
                    path.stat().st_size,
                ),
                mock.patch.object(
                    corpus,
                    "RAW_ARCHIVE_SHA256",
                    corpus._sha256(path),
                ),
            ):
                evidence = corpus.c3i_raw_archive_preflight(path)
            self.assertTrue(evidence["ready"])
            self.assertTrue(evidence["checks"]["archive_sha256_exact"])

    def test_raw_archive_preflight_skips_hash_for_wrong_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / corpus.RAW_ARCHIVE_FILENAME
            path.write_bytes(b"truncated")
            with mock.patch.object(
                corpus,
                "_sha256",
                side_effect=AssertionError("wrong-size archive must not be hashed"),
            ):
                evidence = corpus.c3i_raw_archive_preflight(path)
            self.assertFalse(evidence["ready"])
            self.assertIsNone(evidence["archive_sha256"])

    def test_verified_asset_rejects_a_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rows" / "source.png"
            path.parent.mkdir()
            path.write_bytes(b"asset")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                corpus.verified_corpus_asset(
                    root,
                    {
                        "path": "rows/source.png",
                        "sha256": "0" * 64,
                    },
                )

    def test_preflight_requires_exact_revision_clean_tree_and_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_root = root / "FaceDepth" / "rgb_syn_test"
            depth_root = root / "FaceDepth" / "gt_syn_test"
            rgb_root.mkdir(parents=True)
            depth_root.mkdir(parents=True)
            for _row_id, rgb_name, depth_name in corpus.DEMO_PAIRS:
                (rgb_root / rgb_name).write_bytes(rgb_name.encode("ascii"))
                (depth_root / depth_name).write_bytes(depth_name.encode("ascii"))
            expected = {
                path.name: corpus._sha256(path)
                for path in (*rgb_root.iterdir(), *depth_root.iterdir())
            }
            with (
                mock.patch.object(corpus, "EXPECTED_SHA256", expected),
                mock.patch.object(
                    corpus,
                    "_git_output",
                    side_effect=(corpus.SOURCE_REVISION, ""),
                ),
            ):
                evidence = corpus.c3i_demo_preflight(root)
            self.assertTrue(evidence["ready"])
            self.assertTrue(evidence["checks"]["all_assets_verified"])

    def test_preflight_fails_closed_on_dirty_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                corpus,
                "_git_output",
                side_effect=(corpus.SOURCE_REVISION, " M train.py"),
            ):
                evidence = corpus.c3i_demo_preflight(temporary)
            self.assertFalse(evidence["ready"])
            self.assertFalse(evidence["checks"]["tracked_source_clean"])

    def test_resize_and_place_preserves_uniform_geometry(self):
        source = np.arange(12 * 20, dtype=np.float32).reshape(12, 20)
        output, transform = corpus._resize_and_place(
            source,
            scale=0.5,
            dimension=16,
            center_xy=(10.0, 6.0),
            output_center_xy=(8.0, 7.0),
            interpolation=corpus.cv2.INTER_NEAREST,
            fill_value=-1.0,
        )
        self.assertEqual(output.shape, (16, 16))
        self.assertEqual(transform["resized_shape"], [6, 10])
        self.assertEqual(transform["origin_xy"], [3, 4])
        self.assertTrue(np.all(output[:4] == -1.0))

    def test_transform_bbox_hits_requested_face_height(self):
        transformed = corpus._transform_bbox(
            [20, 10, 60, 110],
            scale=0.78,
            origin_xy=(2, -3),
            dimension=256,
        )
        self.assertEqual(transformed[3] - transformed[1], 78)

    def test_load_exact_depth_rejects_colorized_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.png"
            values = np.zeros((8, 8, 3), dtype=np.uint8)
            values[..., 1] = 1
            corpus.Image.fromarray(values).save(path)
            with self.assertRaisesRegex(ValueError, "not grayscale"):
                corpus._load_exact_depth(path)

    def test_importer_is_path_independent_and_marks_identity_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = [root / "source-a", root / "source-b"]
            expected = {}
            source_values = np.full((100, 80, 3), 128, dtype=np.uint8)
            depth_plane = np.linspace(
                0,
                255,
                100 * 80,
                dtype=np.uint8,
            ).reshape(100, 80)
            depth_values = np.repeat(depth_plane[..., None], 3, axis=2)
            for source in sources:
                rgb_root = source / "FaceDepth" / "rgb_syn_test"
                depth_root = source / "FaceDepth" / "gt_syn_test"
                rgb_root.mkdir(parents=True)
                depth_root.mkdir(parents=True)
                rgb_path = rgb_root / "frame.jpg"
                depth_path = depth_root / "depth.png"
                corpus.Image.fromarray(source_values).save(rgb_path)
                corpus.Image.fromarray(depth_values).save(depth_path)
                expected[rgb_path.name] = corpus._sha256(rgb_path)
                expected[depth_path.name] = corpus._sha256(depth_path)

            face_mask = np.zeros((100, 80), dtype=np.uint8)
            face_mask[10:90, 20:60] = 255
            part_masks = {}
            for index, name in enumerate(corpus.FACE_PART_NAMES):
                part = np.zeros_like(face_mask)
                row = 20 + index * 8
                part[row : row + 5, 30:40] = 255
                part_masks[name] = part
            region = {
                "bbox": [20, 10, 60, 90],
                "face_mask": face_mask,
                "part_masks": part_masks,
                "detector": "fixture-landmarks",
                "landmark_count": 478,
            }

            def git_output(_root, *args):
                return corpus.SOURCE_REVISION if args[0] == "rev-parse" else ""

            summaries = []
            with (
                mock.patch.object(
                    corpus,
                    "DEMO_PAIRS",
                    (("fixture", "frame.jpg", "depth.png"),),
                ),
                mock.patch.object(corpus, "EXPECTED_SHA256", expected),
                mock.patch.object(corpus, "_git_output", side_effect=git_output),
                mock.patch.object(
                    corpus,
                    "_select_face_region",
                    return_value=(region, []),
                ),
            ):
                for index, source in enumerate(sources):
                    summaries.append(
                        corpus.import_c3i_demo_corpus(
                            source,
                            root / f"output-{index}",
                            dimension=128,
                            face_height=64,
                        )
                    )

            self.assertEqual(summaries[0], summaries[1])
            row = summaries[0]["rows"][0]
            self.assertEqual(
                row["identity_group"],
                "c3i_demo_identity_unknown",
            )
            self.assertFalse(row["render"]["metric_scale_available"])
            self.assertNotIn("source_root", summaries[0])
            self.assertNotIn("dataset_license", summaries[0])

    def test_importer_rejects_nonpositive_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                corpus,
                "c3i_demo_preflight",
                return_value={"ready": True, "checks": {}},
            ):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    corpus.import_c3i_demo_corpus(
                        temporary,
                        Path(temporary) / "output",
                        limit=0,
                    )


if __name__ == "__main__":
    unittest.main()
