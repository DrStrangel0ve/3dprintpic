import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from backend.benchmark import rap3df_corpus as corpus


class RAP3DFCorpusTests(unittest.TestCase):
    def _asset_record(self, root: Path, relative: str, index: int) -> dict:
        path = root / Path(*Path(relative).parts)
        return {
            "path": relative,
            "filename": Path(relative).name,
            "id": f"mendeley-file-{index}",
            "content_details": {
                "id": f"mendeley-content-{index}",
                "sha256_hash": corpus._sha256(path),
                "size": path.stat().st_size,
            },
            "status": "COMPLETED",
        }

    def _region(self, *, complete: bool = True) -> dict:
        face_mask = np.zeros(corpus.DEPTH_SHAPE, dtype=np.uint8)
        face_mask[20:95, 40:100] = 255
        part_masks = {}
        positions = (
            (42, 40),
            (42, 78),
            (34, 40),
            (34, 78),
            (72, 58),
            (49, 58),
        )
        for name, (row, column) in zip(corpus.FACE_PART_NAMES, positions):
            mask = np.zeros_like(face_mask)
            mask[row : row + 5, column : column + 5] = 255
            part_masks[name] = mask
        if not complete:
            part_masks.pop("mouth")
        return {
            "bbox": [40, 20, 100, 95],
            "face_mask": face_mask,
            "part_masks": part_masks,
            "detector": "fixture-landmarks",
            "landmark_count": 478,
        }

    def _fixture(
        self,
        root: Path,
        *,
        identities=("ID_B", "ID_A"),
        poses=("front",),
        wrong_orientation: bool = False,
        short_depth: bool = False,
    ) -> tuple[dict, dict, dict]:
        database = {"_faces": list(identities)}
        asset_paths = []
        region = self._region()
        for identity in identities:
            identity_record = {"demography": None}
            for pose_index, pose in enumerate(poses):
                sample_id = f"{identity[-1]}{pose_index}"
                base = f"rap3df_data_02/{identity}"
                rgb_relative = f"{base}/rgb_{sample_id}.bmp"
                depth_relative = f"{base}/depth_bgRm_{sample_id}.data"
                rgb_path = root / Path(*Path(rgb_relative).parts)
                depth_path = root / Path(*Path(depth_relative).parts)
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                image = np.zeros((*corpus.DEPTH_SHAPE, 3), dtype=np.uint8)
                image[..., 0] = np.arange(corpus.DEPTH_SHAPE[1], dtype=np.uint8)
                image[..., 1] = 96
                image[..., 2] = np.arange(corpus.DEPTH_SHAPE[0], dtype=np.uint8)[
                    :, None
                ]
                Image.fromarray(image).save(rgb_path)
                depth = np.zeros(corpus.DEPTH_SHAPE, dtype=np.uint16)
                depth[region["face_mask"] > 0] = 1000
                nose = region["part_masks"]["nose"] > 0
                depth[nose] = 1200 if wrong_orientation else 700
                payload = depth.astype("<u2").tobytes()
                if short_depth:
                    payload = payload[:-2]
                depth_path.write_bytes(payload)
                identity_record[pose] = [
                    {
                        "rgb": rgb_relative,
                        "depth_data_with_bg": depth_relative,
                    }
                ]
                asset_paths.extend((rgb_relative, depth_relative))
            database[identity] = identity_record

        database_path = root / corpus.DATABASE_FILENAME
        database_path.write_text(
            json.dumps(database, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "dataset_doi": corpus.DATASET_DOI,
            "dataset_version": corpus.DATASET_VERSION,
            "license": corpus.DATASET_LICENSE,
            "files": [
                self._asset_record(root, relative, index)
                for index, relative in enumerate(
                    (corpus.DATABASE_FILENAME, *asset_paths),
                    start=1,
                )
            ],
        }
        return database, manifest, region

    def _import_fixture(
        self,
        root: Path,
        manifest: dict,
        region: dict,
        *,
        identity_limit: int = 2,
        poses=("front",),
        output_name: str = "output",
    ) -> dict:
        database_path = root / corpus.DATABASE_FILENAME
        with (
            mock.patch.object(corpus, "DATABASE_BYTES", database_path.stat().st_size),
            mock.patch.object(
                corpus,
                "DATABASE_SHA256",
                corpus._sha256(database_path),
            ),
            mock.patch.object(
                corpus,
                "_select_face_region",
                return_value=(region, []),
            ),
        ):
            return corpus.import_rap3df_corpus(
                root,
                manifest,
                root / output_name,
                identity_limit=identity_limit,
                poses=tuple(poses),
                dimension=128,
                face_height=75,
            )

    def test_release_metadata_is_pinned(self):
        self.assertEqual(corpus.DATASET_DOI, "10.17632/kpdkpcs8zb.4")
        self.assertEqual(corpus.DATASET_LICENSE, "CC BY 4.0")
        self.assertEqual(corpus.DEPTH_SHAPE, (149, 119))
        self.assertEqual(corpus.DATABASE_BYTES, 273343)
        self.assertEqual(
            corpus.DATABASE_SHA256,
            "1366f0496078a250b43bafffc3483d3f949c33afb32520a041d92d353598e3ea",
        )

    def test_decode_depth_is_little_endian_and_marks_zero_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.data"
            values = np.zeros(corpus.DEPTH_SHAPE, dtype=np.uint16)
            values[0, 0] = 0x0102
            values[-1, -1] = 999
            path.write_bytes(values.astype("<u2").tobytes())
            depth, valid = corpus.decode_depth_data(path)
            self.assertEqual(depth.shape, corpus.DEPTH_SHAPE)
            self.assertEqual(depth[0, 0], 0x0102)
            self.assertFalse(valid[0, 1])
            self.assertTrue(valid[-1, -1])

    def test_decode_depth_rejects_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.data"
            path.write_bytes(b"\x01\x00" * (np.prod(corpus.DEPTH_SHAPE) - 1))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                corpus.decode_depth_data(path)

    def test_manifest_rejects_unsafe_and_case_ambiguous_paths(self):
        base = {
            "dataset_doi": corpus.DATASET_DOI,
            "dataset_version": corpus.DATASET_VERSION,
            "license": corpus.DATASET_LICENSE,
        }
        record = {
            "path": "rows/source.bmp",
            "filename": "source.bmp",
            "id": "file-id",
            "content_details": {
                "id": "content-id",
                "sha256_hash": "0" * 64,
                "size": 1,
            },
        }
        unsafe = dict(base, files=[dict(record, path="../source.bmp")])
        with self.assertRaisesRegex(ValueError, "safe relative"):
            corpus.load_mendeley_manifest(unsafe)
        ambiguous = dict(
            base,
            files=[
                record,
                dict(
                    record,
                    path="ROWS/SOURCE.BMP",
                    filename="SOURCE.BMP",
                ),
            ],
        )
        with self.assertRaisesRegex(ValueError, "ambiguous path"):
            corpus.load_mendeley_manifest(ambiguous)

    def test_database_selection_is_deterministic_and_identity_disjoint(self):
        database = {
            "_faces": ["ID_B", "ID_A"],
            "ID_A": {
                "front": [
                    {
                        "rgb": "rap3df_data_02/ID_A/rgb_A0.bmp",
                        "depth_data_with_bg": (
                            "rap3df_data_02/ID_A/depth_bgRm_A0.data"
                        ),
                    }
                ]
            },
            "ID_B": {
                "front": [
                    {
                        "rgb": "rap3df_data_02/ID_B/rgb_B0.bmp",
                        "depth_data_with_bg": (
                            "rap3df_data_02/ID_B/depth_bgRm_B0.data"
                        ),
                    }
                ]
            },
        }
        rows = corpus.select_database_rows(
            database,
            identity_limit=2,
            poses=("front",),
        )
        self.assertEqual([row["identity"] for row in rows], ["ID_A", "ID_B"])
        self.assertEqual(len({row["identity"] for row in rows}), 2)

    def test_database_selection_rejects_missing_or_ambiguous_pairing(self):
        pair = {
            "rgb": "rap3df_data_02/ID_A/rgb_A0.bmp",
            "depth_data_with_bg": "rap3df_data_02/ID_A/depth_bgRm_A0.data",
        }
        for records in ([], [pair, pair]):
            database = {"_faces": ["ID_A"], "ID_A": {"front": records}}
            with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
                corpus.select_database_rows(
                    database,
                    identity_limit=1,
                    poses=("front",),
                )

    def test_database_selection_rejects_cross_identity_and_unsafe_paths(self):
        cases = (
            {
                "rgb": "rap3df_data_02/ID_B/rgb_A0.bmp",
                "depth_data_with_bg": "rap3df_data_02/ID_A/depth_bgRm_A0.data",
            },
            {
                "rgb": "../rgb_A0.bmp",
                "depth_data_with_bg": "rap3df_data_02/ID_A/depth_bgRm_A0.data",
            },
        )
        for pair in cases:
            database = {"_faces": ["ID_A"], "ID_A": {"front": [pair]}}
            with self.assertRaises(ValueError):
                corpus.select_database_rows(
                    database,
                    identity_limit=1,
                    poses=("front",),
                )

    def test_importer_rejects_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _database, manifest, region = self._fixture(
                root,
                identities=("ID_A",),
            )
            rgb_path = root / "rap3df_data_02" / "ID_A" / "rgb_A0.bmp"
            tampered = bytearray(rgb_path.read_bytes())
            tampered[-1] ^= 0x01
            rgb_path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self._import_fixture(
                    root,
                    manifest,
                    region,
                    identity_limit=1,
                )

    def test_importer_rejects_wrong_depth_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _database, manifest, region = self._fixture(
                root,
                identities=("ID_A",),
                wrong_orientation=True,
            )
            with self.assertRaisesRegex(ValueError, "lower-is-nearer"):
                self._import_fixture(
                    root,
                    manifest,
                    region,
                    identity_limit=1,
                )
            self.assertFalse((root / "output").exists())

    def test_face_selection_rejects_incomplete_six_part_masks(self):
        image = np.zeros((*corpus.DEPTH_SHAPE, 3), dtype=np.uint8)
        with mock.patch.object(
            corpus,
            "detect_face_regions",
            return_value=([self._region(complete=False)], []),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one complete"):
                corpus._select_face_region(image)

    def test_face_selection_maps_three_x_fallback_to_source(self):
        image = np.zeros((*corpus.DEPTH_SHAPE, 3), dtype=np.uint8)
        scaled = self._region()
        scaled["bbox"] = [value * 3 for value in scaled["bbox"]]
        scaled["face_mask"] = np.repeat(
            np.repeat(scaled["face_mask"], 3, axis=0),
            3,
            axis=1,
        )
        scaled["part_masks"] = {
            name: np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)
            for name, mask in scaled["part_masks"].items()
        }
        with mock.patch.object(
            corpus,
            "detect_face_regions",
            side_effect=[([], []), ([scaled], [])],
        ) as detector:
            region, errors = corpus._select_face_region(image)

        self.assertEqual(errors, [])
        self.assertEqual(region["detection_scale"], 3)
        self.assertEqual(region["bbox"], [40, 20, 100, 95])
        self.assertEqual(region["face_mask"].shape, corpus.DEPTH_SHAPE)
        self.assertTrue(
            np.array_equal(region["face_mask"] > 0, self._region()["face_mask"] > 0)
        )
        for name in corpus.FACE_PART_NAMES:
            self.assertTrue(
                np.array_equal(
                    region["part_masks"][name] > 0,
                    self._region()["part_masks"][name] > 0,
                )
            )
        self.assertEqual(detector.call_args_list[0].kwargs["min_face_pixels"], 24)
        self.assertEqual(detector.call_args_list[1].kwargs["min_face_pixels"], 72)

    def test_face_selection_short_circuits_native_and_rejects_multiple_faces(self):
        image = np.zeros((*corpus.DEPTH_SHAPE, 3), dtype=np.uint8)
        complete = self._region()
        with mock.patch.object(
            corpus,
            "detect_face_regions",
            return_value=([complete], []),
        ) as detector:
            region, _errors = corpus._select_face_region(image)
        self.assertEqual(region["detection_scale"], 1)
        detector.assert_called_once()

        with mock.patch.object(
            corpus,
            "detect_face_regions",
            return_value=([complete, complete], []),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one complete"):
                corpus._select_face_region(image)

    def test_face_selection_rejects_part_lost_during_fallback_mapping(self):
        image = np.zeros((*corpus.DEPTH_SHAPE, 3), dtype=np.uint8)
        scaled = self._region()
        scaled["bbox"] = [value * 3 for value in scaled["bbox"]]
        scaled["face_mask"] = np.repeat(
            np.repeat(scaled["face_mask"], 3, axis=0),
            3,
            axis=1,
        )
        scaled["part_masks"] = {
            name: np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1)
            for name, mask in scaled["part_masks"].items()
        }
        scaled["part_masks"]["mouth"][:] = 0
        scaled["part_masks"]["mouth"][1, 1] = 255
        with mock.patch.object(
            corpus,
            "detect_face_regions",
            side_effect=[([], []), ([scaled], [])],
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one complete"):
                corpus._select_face_region(image)

    def test_importer_emits_authenticated_portable_evaluation_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _database, manifest, region = self._fixture(root)
            summary = self._import_fixture(root, manifest, region)

            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["row_count"], 2)
            self.assertTrue(summary["source_geometry_evaluation_only"])
            self.assertFalse(summary["training_eligible"])
            self.assertTrue(summary["selection"]["identity_disjoint"])
            self.assertEqual(
                summary["selection"]["identity_groups"],
                ["rap3df:ID_A", "rap3df:ID_B"],
            )
            self.assertFalse(summary["sensor_provenance"]["metric_scale_available"])
            row = summary["rows"][0]
            self.assertEqual(row["render"]["face_bbox_height_pixels"], 75)
            self.assertEqual(
                row["render"]["kind"],
                "shared-full-frame-affine",
            )
            self.assertEqual(set(row["exact_face_parts"]), set(corpus.FACE_PART_NAMES))
            self.assertTrue(
                row["provenance"]["evaluation_only_real_volunteer_data"]
            )
            output = root / "output"
            depth_path = output / row["exact_depth"]["path"]
            depth = np.load(depth_path)
            self.assertEqual(depth.shape, (128, 128))
            self.assertTrue(np.any(np.isfinite(depth)))
            self.assertEqual(corpus._sha256(depth_path), row["exact_depth"]["sha256"])
            encoded = json.dumps(summary, sort_keys=True)
            self.assertNotIn(str(root), encoded)

    def test_import_is_repeatable_across_atomic_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _database, manifest, region = self._fixture(root, identities=("ID_A",))
            first = self._import_fixture(
                root,
                manifest,
                region,
                identity_limit=1,
                output_name="output-a",
            )
            second = self._import_fixture(
                root,
                manifest,
                region,
                identity_limit=1,
                output_name="output-b",
            )
            self.assertEqual(
                json.dumps(first, sort_keys=True),
                json.dumps(second, sort_keys=True),
            )
            for row in first["rows"]:
                for key in ("source", "selection_mask", "exact_depth"):
                    self.assertEqual(
                        row[key]["sha256"],
                        second["rows"][0][key]["sha256"],
                    )

    def test_importer_rejects_short_depth_even_when_manifest_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _database, manifest, region = self._fixture(
                root,
                identities=("ID_A",),
                short_depth=True,
            )
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                self._import_fixture(
                    root,
                    manifest,
                    region,
                    identity_limit=1,
                )


if __name__ == "__main__":
    unittest.main()
