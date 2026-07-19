import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark import mhr_photoreal_domain_transfer_corpus as transfer


class MHRPhotorealDomainTransferTests(unittest.TestCase):
    @staticmethod
    def _mask(shape, box):
        output = np.zeros(shape, dtype=np.uint8)
        x0, y0, x1, y1 = box
        output[y0:y1, x0:x1] = 255
        return output

    @classmethod
    def _region(cls, shift_x=0, shift_y=0):
        shape = (64, 64)
        face = cls._mask(shape, (12 + shift_x, 8 + shift_y, 52 + shift_x, 56 + shift_y))
        parts = {
            "left_eye": cls._mask(
                shape, (18 + shift_x, 22 + shift_y, 25 + shift_x, 27 + shift_y)
            ),
            "right_eye": cls._mask(
                shape, (39 + shift_x, 22 + shift_y, 46 + shift_x, 27 + shift_y)
            ),
            "left_eyebrow": cls._mask(
                shape, (17 + shift_x, 17 + shift_y, 26 + shift_x, 20 + shift_y)
            ),
            "right_eyebrow": cls._mask(
                shape, (38 + shift_x, 17 + shift_y, 47 + shift_x, 20 + shift_y)
            ),
            "nose": cls._mask(
                shape, (29 + shift_x, 29 + shift_y, 35 + shift_x, 38 + shift_y)
            ),
            "mouth": cls._mask(
                shape, (25 + shift_x, 43 + shift_y, 39 + shift_x, 48 + shift_y)
            ),
        }
        columns = np.linspace(18 + shift_x, 46 + shift_x, 26)
        rows = np.linspace(16 + shift_y, 49 + shift_y, 18)
        points = np.asarray(
            [(x, y, 0.0) for y in rows for x in columns], dtype=np.float32
        )[:468]
        points[:, 0] /= 63.0
        points[:, 1] /= 63.0
        return {
            "bbox": [12 + shift_x, 8 + shift_y, 52 + shift_x, 56 + shift_y],
            "face_mask": face,
            "feature_mask": face.copy(),
            "part_masks": parts,
            "landmarks_xyz": points,
            "landmark_count": 468,
            "detector": "fixture",
        }

    @staticmethod
    def _detection_result(region):
        return [region], [], {"validated_face_regions": 1}

    def test_inverse_depth_control_is_fixed_range_and_near_bright(self):
        depth = np.asarray([[0.0, 0.25], [0.75, 1.0]], dtype=np.float32)
        control = transfer.make_inverse_depth_control(depth)

        self.assertEqual(control.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            control[..., 0], np.asarray([[255, 191], [64, 0]])
        )
        np.testing.assert_array_equal(control[..., 0], control[..., 1])
        with self.assertRaisesRegex(ValueError, "normalized"):
            transfer.make_inverse_depth_control(np.asarray([[1.2]], dtype=np.float32))

    def test_face_only_composite_preserves_every_background_byte(self):
        parent = np.full((8, 8, 3), 10, dtype=np.uint8)
        generated = np.full((8, 8, 3), 220, dtype=np.uint8)
        mask = self._mask((8, 8), (2, 2, 6, 6))

        output = transfer.composite_face_only(parent, generated, mask)

        np.testing.assert_array_equal(output[mask == 0], parent[mask == 0])
        np.testing.assert_array_equal(output[mask > 0], generated[mask > 0])
        with self.assertRaisesRegex(ValueError, "automatic resize"):
            transfer.composite_face_only(parent, generated[:7], mask)

    def test_face_local_generation_has_exact_inverse_bbox(self):
        parent = np.zeros((40, 48, 3), dtype=np.uint8)
        parent[..., 0] = np.arange(48, dtype=np.uint8)
        depth = np.linspace(0.0, 1.0, 40 * 48, dtype=np.float32).reshape(40, 48)
        mask = np.zeros((40, 48), dtype=bool)
        mask[12:24, 18:26] = True

        rgb_crop, control_crop, metadata = transfer.prepare_face_local_generation(
            parent, depth, mask, target_size=32, context_ratio=2.0
        )
        restored = transfer.restore_face_local_generation(
            parent, np.full_like(rgb_crop, 77), metadata
        )

        self.assertEqual(rgb_crop.shape, (32, 32, 3))
        self.assertEqual(control_crop.shape, (32, 32, 3))
        self.assertEqual(control_crop.dtype, np.uint8)
        self.assertEqual(
            metadata["control_pipeline"],
            "float_depth_resize_then_inverse_uint8",
        )
        x0, y0, x1, y1 = metadata["source_bbox_xyxy"]
        outside = np.ones(parent.shape[:2], dtype=bool)
        outside[y0:y1, x0:x1] = False
        np.testing.assert_array_equal(restored[outside], parent[outside])
        self.assertTrue(np.all(restored[y0:y1, x0:x1] == 77))

    def test_face_local_generation_rejects_crop_that_excludes_selection(self):
        parent = np.zeros((100, 200, 3), dtype=np.uint8)
        depth = np.zeros((100, 200), dtype=np.float32)
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:90, 20:180] = True

        with self.assertRaisesRegex(ValueError, "complete selection"):
            transfer.prepare_face_local_generation(
                parent, depth, mask, target_size=32, context_ratio=1.0
            )

    def test_alignment_accepts_geometry_stable_non_noop_transfer(self):
        parent = np.full((64, 64, 3), 80, dtype=np.uint8)
        selection = self._mask((64, 64), (12, 8, 52, 56))
        candidate = parent.copy()
        candidate[selection > 0] = 130
        region = self._region()

        with patch.object(
            transfer,
            "detect_face_regions_in_roi",
            side_effect=(
                self._detection_result(region),
                self._detection_result(region),
            ),
        ):
            result = transfer.measure_rgb_alignment(parent, candidate, selection)

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["metrics"]["outside_changed_pixels"], 0)
        self.assertGreater(result["metrics"]["inside_rgb_rms"], 3.0)

    def test_alignment_preserves_subject_relative_left_right_ordering(self):
        parent = np.full((64, 64, 3), 80, dtype=np.uint8)
        selection = self._mask((64, 64), (12, 8, 52, 56))
        candidate = parent.copy()
        candidate[selection > 0] = 130
        region = self._region()
        parts = region["part_masks"]
        parts["left_eye"], parts["right_eye"] = parts["right_eye"], parts["left_eye"]
        parts["left_eyebrow"], parts["right_eyebrow"] = (
            parts["right_eyebrow"],
            parts["left_eyebrow"],
        )

        with patch.object(
            transfer,
            "detect_face_regions_in_roi",
            side_effect=(
                self._detection_result(region),
                self._detection_result(region),
            ),
        ):
            result = transfer.measure_rgb_alignment(parent, candidate, selection)

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["anatomical_ordering"])
        self.assertEqual(
            result["metrics"]["parent_anatomical_ordering"],
            result["metrics"]["candidate_anatomical_ordering"],
        )

    def test_alignment_rejects_four_pixel_face_motion(self):
        parent = np.full((64, 64, 3), 80, dtype=np.uint8)
        selection = self._mask((64, 64), (8, 8, 58, 58))
        candidate = parent.copy()
        candidate[selection > 0] = 130
        parent_region = self._region()
        candidate_region = self._region(shift_x=4)

        with patch.object(
            transfer,
            "detect_face_regions_in_roi",
            side_effect=(
                self._detection_result(parent_region),
                self._detection_result(candidate_region),
            ),
        ):
            result = transfer.measure_rgb_alignment(
                parent,
                candidate,
                selection,
                exact_part_masks=parent_region["part_masks"],
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["face_mask_iou"])
        self.assertFalse(result["checks"]["landmark_p95"])
        self.assertFalse(result["checks"]["all_parts"])
        self.assertFalse(result["checks"]["exact_part_geometry"])

    def test_alignment_rejects_candidate_only_anatomical_reversal(self):
        parent = np.full((64, 64, 3), 80, dtype=np.uint8)
        selection = self._mask((64, 64), (12, 8, 52, 56))
        candidate = parent.copy()
        candidate[selection > 0] = 130
        parent_region = self._region()
        candidate_region = copy.deepcopy(parent_region)
        parts = candidate_region["part_masks"]
        parts["left_eye"], parts["right_eye"] = parts["right_eye"], parts["left_eye"]
        parts["left_eyebrow"], parts["right_eyebrow"] = (
            parts["right_eyebrow"],
            parts["left_eyebrow"],
        )

        with patch.object(
            transfer,
            "detect_face_regions_in_roi",
            side_effect=(
                self._detection_result(parent_region),
                self._detection_result(candidate_region),
            ),
        ):
            result = transfer.measure_rgb_alignment(parent, candidate, selection)

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["anatomical_ordering"])

    def test_alignment_rejects_shared_offset_from_exact_parts(self):
        parent = np.full((64, 64, 3), 80, dtype=np.uint8)
        selection = self._mask((64, 64), (4, 4, 63, 63))
        candidate = parent.copy()
        candidate[selection > 0] = 130
        exact_region = self._region()
        shifted_region = self._region(shift_x=8)

        with patch.object(
            transfer,
            "detect_face_regions_in_roi",
            side_effect=(
                self._detection_result(shifted_region),
                self._detection_result(shifted_region),
            ),
        ):
            result = transfer.measure_rgb_alignment(
                parent,
                candidate,
                selection,
                exact_part_masks=exact_region["part_masks"],
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["exact_part_geometry"])
        self.assertFalse(
            result["metrics"]["exact_part_alignment"]["parts"]["nose"][
                "absolute_passed"
            ]
        )

    def test_alignment_rejects_dimension_change_before_detection(self):
        result = transfer.measure_rgb_alignment(
            np.zeros((64, 64, 3), dtype=np.uint8),
            np.zeros((48, 64, 3), dtype=np.uint8),
            np.ones((64, 64), dtype=np.uint8),
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "output_dimensions_changed")

    def test_parent_row_is_hash_pinned_and_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row_dir = root / "rows" / "row"
            parts_dir = row_dir / "exact_face_parts"
            parts_dir.mkdir(parents=True)
            source = np.full((8, 8, 3), 20, dtype=np.uint8)
            mask = np.ones((8, 8), dtype=np.uint8) * 255
            Image.fromarray(source).save(row_dir / "source.png")
            Image.fromarray(mask).save(row_dir / "selection_mask.png")
            np.save(row_dir / "exact_depth.npy", np.full((8, 8), 0.5, dtype=np.float32))
            np.save(
                row_dir / "exact_camera_depth.npy",
                np.full((8, 8), 1.0, dtype=np.float32),
            )
            np.save(
                row_dir / "exact_camera_normals.npy",
                np.ones((8, 8, 3), dtype=np.float32),
            )
            for name in transfer.FACE_PART_NAMES:
                Image.fromarray(mask).save(parts_dir / f"{name}.png")
            geometry_dir = row_dir / "geometry_targets"
            geometry_dir.mkdir()
            np.save(geometry_dir / "identity.npy", np.ones(4, dtype=np.float32))
            np.save(geometry_dir / "expression.npy", np.ones(3, dtype=np.float32))

            def record(path):
                return {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": transfer._sha256(path),
                }

            row = {
                "row_id": "row",
                "split": "train",
                "spec": {"split": "train"},
                "source": record(row_dir / "source.png"),
                "selection_mask": record(row_dir / "selection_mask.png"),
                "exact_depth": record(row_dir / "exact_depth.npy"),
                "exact_camera_depth": record(row_dir / "exact_camera_depth.npy"),
                "exact_camera_normals": record(row_dir / "exact_camera_normals.npy"),
                "exact_face_parts": {
                    name: record(parts_dir / f"{name}.png")
                    for name in transfer.FACE_PART_NAMES
                },
            }
            (root / "summary.json").write_text(
                json.dumps({"rows": [row]}), encoding="utf-8"
            )
            summary_hash = transfer._sha256(root / "summary.json")
            supervision = {
                "rows": [
                    {
                        "row_id": "row",
                        "split": "train",
                        "geometry_targets": {
                            "identity": record(geometry_dir / "identity.npy"),
                            "expression": record(geometry_dir / "expression.npy"),
                        },
                    }
                ]
            }
            (root / "training_supervision.json").write_text(
                json.dumps(supervision), encoding="utf-8"
            )
            supervision_hash = transfer._sha256(root / "training_supervision.json")

            loaded, paths, provenance = transfer.load_parent_row(
                root, summary_hash, "row", supervision_hash
            )
            self.assertEqual(loaded["row_id"], "row")
            self.assertEqual(
                set(paths),
                {
                    "source",
                    "selection_mask",
                    "exact_depth",
                    "exact_camera_depth",
                    "exact_camera_normals",
                    *{f"part:{name}" for name in transfer.FACE_PART_NAMES},
                    "geometry_target:identity",
                    "geometry_target:expression",
                },
            )
            self.assertEqual(provenance["summary_sha256"], summary_hash)
            self.assertEqual(
                provenance["training_supervision_sha256"], supervision_hash
            )
            with self.assertRaisesRegex(ValueError, "summary SHA256"):
                transfer.load_parent_row(root, "0" * 64, "row", supervision_hash)
            with self.assertRaisesRegex(ValueError, "training supervision SHA256"):
                transfer.load_parent_row(root, summary_hash, "row", "0" * 64)
            row["split"] = "sealed"
            row["spec"]["split"] = "sealed"
            (root / "summary.json").write_text(
                json.dumps({"rows": [row]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "train-only"):
                transfer.load_parent_row(
                    root,
                    transfer._sha256(root / "summary.json"),
                    "row",
                    supervision_hash,
                )

    def test_preflight_fails_closed_for_unpinned_source_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "provider"
            model = root / "model"
            provider.mkdir()
            model.mkdir()
            (model / "tokenizer").mkdir()
            (model / "tokenizer" / "added_tokens.json").write_text("{}")
            (model / "extra.safetensors").write_bytes(b"extra")
            control = root / transfer.ZIMAGE_CONTROLNET_FILENAME
            control.write_bytes(b"wrong")
            with patch.object(
                transfer,
                "_git_output",
                side_effect=lambda _root, *args: (
                    "dirty" if args and args[0] == "status" else "wrong"
                ),
            ):
                evidence = transfer.zimage_preflight(provider, model, control)

        self.assertFalse(evidence["runnable"])
        self.assertFalse(evidence["checks"]["source_revision_pinned"])
        self.assertFalse(evidence["checks"]["source_clean"])
        self.assertFalse(evidence["checks"]["model_files_complete"])
        self.assertFalse(evidence["checks"]["model_has_no_unexpected_safetensors"])
        self.assertFalse(evidence["checks"]["model_has_no_unexpected_consumed_files"])
        self.assertFalse(evidence["checks"]["controlnet_hash_pinned"])
        self.assertFalse(evidence["checks"]["controlnet_hash_reads_consistent"])
        self.assertTrue(evidence["license"]["production_eligible"])

    def test_large_asset_record_requires_two_consistent_pinned_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.safetensors"
            path.write_bytes(b"weight")
            with (
                patch.object(transfer, "REDUNDANT_HASH_MIN_BYTES", 1),
                patch.object(
                    transfer,
                    "_sha256",
                    side_effect=("expected", "transient-corruption"),
                ),
            ):
                record = transfer._asset_record(path, len(b"weight"), "expected")

        self.assertEqual(
            record["sha256_observations"],
            ["expected", "transient-corruption"],
        )
        self.assertEqual(record["hash_read_count"], 2)
        self.assertEqual(record["hash_reads_required"], 2)
        self.assertFalse(record["hash_reads_consistent"])
        self.assertFalse(record["hash_pinned"])

    def test_large_asset_record_accepts_two_consistent_pinned_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weight.safetensors"
            path.write_bytes(b"weight")
            with (
                patch.object(transfer, "REDUNDANT_HASH_MIN_BYTES", 1),
                patch.object(
                    transfer,
                    "_sha256",
                    side_effect=("expected", "expected"),
                ),
            ):
                record = transfer._asset_record(path, len(b"weight"), "expected")

        self.assertEqual(record["hash_read_count"], 2)
        self.assertTrue(record["hash_reads_consistent"])
        self.assertTrue(record["hash_pinned"])

    def test_small_asset_record_preserves_single_read_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"config")
            with patch.object(
                transfer,
                "_sha256",
                return_value="expected",
            ) as sha256:
                record = transfer._asset_record(path, len(b"config"), "expected")

        sha256.assert_called_once_with(path)
        self.assertEqual(record["sha256"], "expected")
        self.assertEqual(record["sha256_observations"], ["expected"])
        self.assertEqual(record["hash_reads_required"], 1)
        self.assertTrue(record["hash_reads_consistent"])
        self.assertTrue(record["hash_pinned"])

    def test_preflight_propagates_inconsistent_controlnet_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = root / "provider"
            model = root / "model"
            provider.mkdir()
            model.mkdir()
            for relative in transfer.DIFFSYNTH_REQUIRED_FILES:
                path = provider / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            control = root / transfer.ZIMAGE_CONTROLNET_FILENAME

            def asset_record(path, size, sha256):
                reads = (
                    transfer.REDUNDANT_HASH_READS
                    if size >= transfer.REDUNDANT_HASH_MIN_BYTES
                    else 1
                )
                observations = [sha256] * reads
                consistent = True
                pinned = True
                if Path(path) == control:
                    observations[-1] = "transient-corruption"
                    consistent = False
                    pinned = False
                return {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": size,
                    "expected_size_bytes": size,
                    "size_pinned": True,
                    "sha256": observations[-1],
                    "sha256_observations": observations,
                    "expected_sha256": sha256,
                    "hash_read_count": reads,
                    "hash_reads_required": reads,
                    "hash_reads_consistent": consistent,
                    "hash_pinned": pinned,
                }

            with (
                patch.object(
                    transfer,
                    "_git_output",
                    side_effect=lambda _root, *args: (
                        transfer.DIFFSYNTH_SOURCE_REVISION
                        if args and args[0] == "rev-parse"
                        else ""
                    ),
                ),
                patch.object(
                    transfer,
                    "_adapter_repository_provenance",
                    return_value={
                        "root": str(root),
                        "revision": "fixture",
                        "clean": True,
                        "status": [],
                        "adapter_relative_path": "adapter.py",
                        "adapter_tracked": True,
                    },
                ),
                patch.object(transfer, "_asset_record", side_effect=asset_record),
                patch.object(transfer.importlib.util, "find_spec", return_value=object()),
            ):
                evidence = transfer.zimage_preflight(provider, model, control)

        self.assertFalse(evidence["runnable"])
        self.assertFalse(evidence["checks"]["controlnet_hash_pinned"])
        self.assertFalse(evidence["checks"]["controlnet_hash_reads_consistent"])
        self.assertTrue(evidence["checks"]["model_hash_reads_consistent"])

    def test_publish_requires_both_geometry_gates(self):
        with self.assertRaisesRegex(ValueError, "both geometry gates"):
            transfer._publish_derived_corpus(
                Path("parent"),
                Path("output"),
                {"row_id": "row"},
                np.zeros((1, 1, 3), dtype=np.uint8),
                {},
                {"passed": True},
                {"passed": False},
            )

    def test_disk_maps_refresh_after_provider_device_switch(self):
        events = []

        class Handle:
            def __init__(self, name):
                self.name = name

            def __exit__(self, _exc_type, _exc, _traceback):
                events.append(f"close:{self.name}")

        class DiskMap:
            def __init__(self, name):
                self.name = name
                self.files = [Handle(name)]

            def flush_files(self):
                events.append(f"refresh:{self.name}:{len(self.files)}")
                self.files.append(Handle(self.name))

        first_disk_map = DiskMap("first")
        second_disk_map = DiskMap("second")

        class Child:
            def __init__(self, disk_map):
                self.disk_map = disk_map

        class Model:
            def modules(self):
                return [
                    Child(first_disk_map),
                    Child(first_disk_map),
                    Child(second_disk_map),
                ]

        class Pipe:
            text_encoder = Model()

            @staticmethod
            def load_models_to_device(_model_names):
                events.append("provider_switch")

        pipe = Pipe()
        telemetry = transfer._install_post_empty_cache_disk_map_refresh(pipe)
        pipe.load_models_to_device(["text_encoder"])

        self.assertEqual(
            events,
            [
                "provider_switch",
                "close:first",
                "close:second",
                "refresh:first:0",
                "refresh:second:0",
            ],
        )
        self.assertEqual(telemetry, {"calls": 1, "refreshed_maps": 2})

    def test_precomputed_prompt_processor_accepts_provider_keywords(self):
        embeddings = [object()]
        processor = transfer._make_precomputed_prompt_processor(embeddings)

        result = processor(pipe=object(), prompt="portrait", edit_image=None)

        self.assertIs(result["prompt_embeds"], embeddings)

    def test_materializes_only_direct_meta_parameters_from_pinned_disk_map(self):
        import torch

        class DiskMap:
            def __init__(self):
                self.requested = []

            def __getitem__(self, name):
                self.requested.append(name)
                return torch.full((1, 2), 3.0)

        disk_map = DiskMap()

        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((2, 2), device="meta"))
                self.disk_map = disk_map

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x_pad_token = torch.nn.Parameter(
                    torch.empty((1, 2), device="meta")
                )
                self.child = Child()

        model = Model()
        materialized = transfer._materialize_direct_meta_parameters(
            model,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(materialized, ["x_pad_token"])
        self.assertEqual(disk_map.requested, ["x_pad_token"])
        self.assertEqual(model.x_pad_token.device.type, "cpu")
        self.assertEqual(model.child.weight.device.type, "meta")
        torch.testing.assert_close(model.x_pad_token, torch.full((1, 2), 3.0))

    def test_owned_disk_map_reads_clone_tensors_and_disable_mid_read_refresh(self):
        import sys

        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch

        source = torch.arange(4, dtype=torch.float32)
        source_bf16 = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "weights.safetensors"
            save_file({"bf16": source_bf16, "weight": source}, weights)

            class DiskMap:
                def __init__(self):
                    self.buffer_size = 1
                    self.device = torch.device("cpu")
                    self.files = [
                        safe_open(str(weights), framework="pt", device="cpu")
                    ]
                    self.name_map = {"bf16": 0, "weight": 0}
                    self.rename_dict = None
                    self.path = [str(weights)]
                    self.torch_dtype = None
                    self.num_params = 0

                def flush_files(self):
                    raise AssertionError("on-demand maps must not stay open")

                def __getitem__(self, _name):
                    return source

            disk_map = DiskMap()

            class Child(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.disk_map = disk_map

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.child = Child()

            class Pipe(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = Model()

            telemetry = transfer._install_owned_disk_map_reads(
                Pipe(),
                expected_file_sha256={str(weights): transfer._sha256(weights)},
            )
            fetched = disk_map["weight"]
            fetched_bf16 = disk_map["bf16"]

            self.assertEqual(disk_map.buffer_size, sys.maxsize)
            self.assertEqual(disk_map.files, [])
            self.assertEqual(telemetry["disk_maps"], 1)
            self.assertEqual(telemetry["storage_devices"], ["cpu"])
            self.assertEqual(telemetry["read_mode"], "on_demand_safetensors")
            self.assertEqual(telemetry["supported_dtypes"], ["BF16", "F32"])
            self.assertEqual(
                telemetry["tensor_transport"],
                "file_and_tensor_sha256_validated_bytes_to_owned_torch",
            )
            self.assertEqual(
                telemetry["payload_hash_validation"],
                "manifest_from_pinned_full_read_then_exact_payload",
            )
            self.assertTrue(telemetry["storage_is_file_mapping_independent"])
            self.assertEqual(
                telemetry["observed_reads"],
                {
                    "tensor_count": 2,
                    "payload_bytes": 24,
                    "dtype_counts": {"BF16": 1, "F32": 1},
                    "manifest_file_count": 1,
                    "manifest_bytes_hashed": weights.stat().st_size,
                    "payload_hash_checks": 2,
                    "payload_hash_failures": 0,
                },
            )
            self.assertNotEqual(fetched.data_ptr(), source.data_ptr())
            torch.testing.assert_close(fetched, source)
            self.assertEqual(fetched_bf16.dtype, torch.bfloat16)
            torch.testing.assert_close(fetched_bf16, source_bf16)

    def test_verified_tensor_read_rejects_post_manifest_payload_change(self):
        from safetensors.torch import save_file
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "weights.safetensors"
            save_file({"weight": torch.arange(4, dtype=torch.float32)}, weights)
            manifest = transfer._build_verified_safetensor_manifest(
                weights,
                transfer._sha256(weights),
            )
            entry = manifest["tensors"]["weight"]
            payload_offset = manifest["data_start"] + entry["data_offsets"][0]
            with weights.open("r+b", buffering=0) as stream:
                stream.seek(payload_offset)
                original = stream.read(1)
                stream.seek(payload_offset)
                stream.write(bytes([original[0] ^ 0x01]))
            telemetry = {
                "payload_hash_checks": 0,
                "payload_hash_failures": 0,
            }

            with self.assertRaisesRegex(RuntimeError, "payload hash changed"):
                transfer._read_verified_safetensor_tensor(
                    weights,
                    "weight",
                    manifest,
                    telemetry,
                )

        self.assertEqual(telemetry["payload_hash_checks"], 1)
        self.assertEqual(telemetry["payload_hash_failures"], 1)

    def test_owned_disk_map_reader_rebinds_for_sequential_pipelines(self):
        from safetensors.torch import save_file
        import torch

        class DiskMap:
            def __init__(self, path):
                self.buffer_size = 1
                self.device = torch.device("cpu")
                self.files = []
                self.name_map = {"weight": 0}
                self.rename_dict = None
                self.path = [str(path)]
                self.torch_dtype = None
                self.num_params = 0

            def __getitem__(self, _name):
                raise AssertionError("the pinned reader must be installed")

        class Child(torch.nn.Module):
            def __init__(self, disk_map):
                super().__init__()
                self.disk_map = disk_map

        class Pipe(torch.nn.Module):
            def __init__(self, disk_map):
                super().__init__()
                self.model = Child(disk_map)

        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = Path(temp_dir) / "first.safetensors"
            second_path = Path(temp_dir) / "second.safetensors"
            save_file({"weight": torch.full((2,), 3.0)}, first_path)
            save_file({"weight": torch.full((2,), 9.0)}, second_path)
            first_map = DiskMap(first_path)
            second_map = DiskMap(second_path)
            transfer._install_owned_disk_map_reads(
                Pipe(first_map),
                expected_file_sha256={
                    str(first_path): transfer._sha256(first_path)
                },
            )
            second_telemetry = transfer._install_owned_disk_map_reads(
                Pipe(second_map),
                expected_file_sha256={
                    str(second_path): transfer._sha256(second_path)
                },
            )

            torch.testing.assert_close(first_map["weight"], torch.full((2,), 3.0))
            torch.testing.assert_close(second_map["weight"], torch.full((2,), 9.0))

        self.assertEqual(
            second_telemetry["observed_reads"]["payload_hash_checks"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
