from __future__ import annotations

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from PIL import Image

from huggingface_space import space_runtime


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class SpaceRuntimeTests(unittest.TestCase):
    def test_image_dimensions_preserve_landscape_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "landscape.png"
            Image.new("RGB", (400, 200), "white").save(image_path)
            self.assertEqual(space_runtime.image_dimensions_mm(image_path, 128), (128.0, 64.0))

    def test_image_dimensions_preserve_portrait_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "portrait.png"
            Image.new("RGB", (150, 300), "white").save(image_path)
            self.assertEqual(space_runtime.image_dimensions_mm(image_path, 120), (60.0, 120.0))

    def test_local_relief_dimensions_match_default_printer_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "landscape.png"
            Image.new("RGB", (400, 200), "white").save(image_path)
            self.assertEqual(
                space_runtime.local_relief_dimensions_mm(image_path, 100),
                (256.0, 128.0),
            )
            self.assertEqual(
                space_runtime.local_relief_dimensions_mm(image_path, 50),
                (128.0, 64.0),
            )

    def test_diagnostic_summary_maps_prefixed_stl_contract(self):
        diagnostics = {
            "stl_is_watertight": True,
            "stl_is_volume": True,
            "stl_is_manifold": True,
            "stl_winding_consistent": True,
            "stl_component_count": 1,
            "stl_nonmanifold_edge_count": 0,
            "stl_degenerate_face_count": 0,
            "stl_positive_volume": True,
            "stl_faces": 1200,
            "stl_vertices": 602,
            "stl_bbox_x": 64.0,
            "stl_bbox_y": 40.0,
            "stl_bbox_z": 10.0,
            "stl_faces_per_normalized_bbox_volume_log1p": 8.5,
        }
        summary = space_runtime._diagnostic_summary(diagnostics, model="test/model")
        self.assertTrue(summary["stl_passes_hard_checks"])
        self.assertEqual(summary["stl_failed_checks"], [])
        self.assertEqual(summary["bbox_extents"], [64.0, 40.0, 10.0])
        self.assertEqual(summary["face_count"], 1200)

    def test_sam3_selection_fails_closed_without_owner_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "read-only HF_TOKEN"):
                    space_runtime.select_object(image_path, 10, 10)

    def test_hover_region_map_prefers_scores_and_splits_disconnected_regions(self):
        masks = np.zeros((2, 8, 10), dtype=bool)
        masks[0, 1:4, 1:4] = True
        masks[0, 5:8, 1:4] = True
        masks[1, 2:6, 2:7] = True
        hover_map, regions, size = space_runtime._sam3_hover_region_map(
            masks,
            np.asarray([0.6, 0.9], dtype=np.float32),
            ["person", "furniture"],
            (10, 8),
        )
        encoded = np.asarray(hover_map, dtype=np.uint32)
        region_ids = encoded[..., 0] + (encoded[..., 1] << 8) + (encoded[..., 2] << 16)
        self.assertEqual(size, (10, 8))
        self.assertEqual(regions[str(int(region_ids[3, 3]))]["label"], "furniture")
        person_ids = {
            int(region_id)
            for region_id, metadata in regions.items()
            if metadata["label"] == "person"
        }
        self.assertGreaterEqual(len(person_ids), 2)
        self.assertEqual(int(region_ids[0, 0]), 0)

    def test_hover_region_map_rejects_scene_context_and_prefers_specific_objects(self):
        masks = np.zeros((4, 80, 100), dtype=bool)
        masks[0, :, :] = True
        masks[0, 8:76, 18:48] = False
        masks[0, 28:66, 58:91] = False
        masks[1, 8:76, 18:48] = True
        masks[2, 28:66, 58:91] = True
        masks[3, 5:75, 0:48] = True
        hover_map, regions, _size = space_runtime._sam3_hover_region_map(
            masks,
            np.asarray([0.99, 0.92, 0.90, 0.95], dtype=np.float32),
            ["building", "person", "furniture", "furniture"],
            (100, 80),
        )
        encoded = np.asarray(hover_map, dtype=np.uint32)
        region_ids = encoded[..., 0] + (encoded[..., 1] << 8) + (encoded[..., 2] << 16)

        self.assertEqual(int(region_ids[2, 2]), 0)
        self.assertNotIn("building", {metadata["label"] for metadata in regions.values()})
        self.assertEqual(regions[str(int(region_ids[20, 30]))]["label"], "person")
        self.assertEqual(regions[str(int(region_ids[45, 75]))]["label"], "furniture")

    def test_prepare_object_selection_runs_sam3_once_and_releases_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (20, 12), "white").save(image_path)
            masks = np.zeros((1, 12, 20), dtype=bool)
            masks[0, 2:10, 4:16] = True
            with (
                patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False),
                patch.object(
                    space_runtime.backend_main,
                    "compute_sam3_selection_instances",
                    return_value=(
                        masks,
                        np.asarray([0.95], dtype=np.float32),
                        ["person"],
                        space_runtime.SAM3_MODEL,
                    ),
                ) as compute,
                patch.object(space_runtime.backend_main, "release_selection_models") as release,
            ):
                manifest_json, state = space_runtime.prepare_object_selection(image_path)
            manifest = json.loads(manifest_json)
            self.assertEqual(manifest["version"], space_runtime.SAM3_HOVER_MAP_VERSION)
            self.assertTrue(manifest["hit_map"].startswith("data:image/png;base64,"))
            self.assertEqual(len(state["precompute_id"]), 32)
            self.assertEqual(
                state["state_version"],
                space_runtime.SAM3_PRECOMPUTE_STATE_VERSION,
            )
            self.assertGreater(state["region_count"], 0)
            self.assertEqual(set(state["region_instances"].values()), {0})
            self.assertEqual(
                len(state["selection_precompute"]["compressed_masks"]),
                1,
            )
            self.assertGreater(state["selection_precompute"]["compressed_bytes"], 0)
            self.assertNotIn("image", state["selection_precompute"])
            compute.assert_called_once()
            release.assert_called_once()

    def test_inline_click_survives_zero_gpu_worker_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            image = Image.new("RGB", (16, 10), "white")
            image.save(image_path)
            masks = np.zeros((3, 10, 16), dtype=bool)
            masks[0, 0, 0] = True
            masks[1, 0, 1] = True
            masks[2, 2:9, 3:14] = True
            with (
                patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False),
                patch.object(
                    space_runtime.backend_main,
                    "compute_sam3_selection_instances",
                    return_value=(
                        masks,
                        np.asarray([0.1, 0.1, 0.95], dtype=np.float32),
                        ["artifact-a", "artifact-b", "person"],
                        space_runtime.SAM3_MODEL,
                    ),
                ),
                patch.object(space_runtime, "release_gpu_models"),
            ):
                _manifest, worker_state = space_runtime.prepare_object_selection(image_path)
            main_process_state = pickle.loads(pickle.dumps(worker_state))
            person_region = next(
                region_id
                for region_id, instance_index in main_process_state["region_instances"].items()
                if instance_index == 0
            )
            with (
                patch.object(
                    space_runtime.backend_main,
                    "get_selection_precompute",
                ) as global_cache,
                patch.object(
                    space_runtime,
                    "_save_selection_job",
                    return_value={"job_id": "selected"},
                ) as save,
            ):
                result = space_runtime.select_precomputed_object(
                    image_path,
                    main_process_state,
                    0.5,
                    0.5,
                    int(person_region),
                )
            self.assertEqual(result, {"job_id": "selected"})
            global_cache.assert_not_called()
            self.assertEqual(save.call_args.kwargs["labels"], ["person"])

    def test_inline_precompute_enforces_compressed_payload_bound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            masks = np.ones((1, 32, 32), dtype=bool)
            with (
                patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False),
                patch.object(
                    space_runtime.backend_main,
                    "compute_sam3_selection_instances",
                    return_value=(
                        masks,
                        np.asarray([0.95], dtype=np.float32),
                        ["person"],
                        space_runtime.SAM3_MODEL,
                    ),
                ),
                patch.object(space_runtime, "release_gpu_models"),
                patch.object(space_runtime, "SAM3_PRECOMPUTE_MAX_COMPRESSED_BYTES", 1),
            ):
                with self.assertRaisesRegex(RuntimeError, "too large to cache safely"):
                    space_runtime.prepare_object_selection(image_path)

    def test_multi_click_composes_selected_components_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            image = Image.new("RGB", (12, 8), "white")
            image.save(image_path)
            masks = np.zeros((2, 8, 12), dtype=bool)
            masks[0, 1:4, 1:4] = True
            masks[1, 4:7, 7:11] = True
            compressed_masks = [
                space_runtime.zlib.compress(np.packbits(mask, axis=1).tobytes())
                for mask in masks
            ]
            state = {
                "state_version": space_runtime.SAM3_PRECOMPUTE_STATE_VERSION,
                "precompute_id": "session",
                "source_fingerprint": space_runtime.backend_main.selection_source_fingerprint(image),
                "hover_size": [12, 8],
                "region_instances": {"10": 0, "20": 1},
                "selection_precompute": {
                    "compressed_masks": compressed_masks,
                    "mask_width": 12,
                    "labels": ["person", "furniture"],
                    "model_id": space_runtime.SAM3_MODEL,
                    "image_size": [12, 8],
                    "created_at_epoch": space_runtime.time.time(),
                },
            }
            with patch.object(
                space_runtime,
                "_save_selection_job",
                return_value={"job_id": "combined"},
            ) as save:
                result = space_runtime.select_precomputed_objects(
                    image_path,
                    state,
                    [
                        {"region_id": 10, "x": 0.2, "y": 0.25},
                        {"region_id": 20, "x": 0.75, "y": 0.7},
                    ],
                )
            self.assertEqual(result, {"job_id": "combined"})
            saved_mask = np.asarray(save.call_args.args[2]) > 0
            self.assertEqual(int(np.count_nonzero(saved_mask)), 21)
            self.assertEqual(save.call_args.kwargs["labels"], ["person", "furniture"])
            self.assertEqual(save.call_args.kwargs["mask_count"], 2)
            self.assertEqual(
                save.call_args.kwargs["model_status"],
                "sam3-concept-precomputed-multi-point",
            )

    def test_selection_job_exposes_isolated_pixels_as_preview(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            image = Image.new("RGB", (40, 30), (180, 20, 30))
            image.save(image_path)
            mask_array = np.zeros((30, 40), dtype=np.uint8)
            mask_array[8:22, 10:30] = 255
            mask = Image.fromarray(mask_array, mode="L")
            output_dir = Path(temp_dir) / "output"
            with (
                patch.object(space_runtime, "OUTPUT_DIR", output_dir),
                patch.object(space_runtime.backend_main, "OUTPUT_DIR", output_dir),
            ):
                result = space_runtime._save_selection_job(
                    image_path,
                    image,
                    mask,
                    resolved_model=space_runtime.SAM3_MODEL,
                    model_status="sam3-concept-precomputed-point",
                    labels=["person"],
                )
            self.assertNotEqual(result["selected"], result["overlay"])
            with Image.open(result["selected"]) as preview:
                pixels = preview.convert("RGB")
                self.assertEqual(pixels.getpixel((0, 0)), (245, 245, 245))
                self.assertEqual(pixels.getpixel((20, 15)), (180, 20, 30))

    def test_hover_cache_discards_source_pixels_and_expires_old_entries(self):
        old_id = "hover-old-test"
        fresh_id = "hover-fresh-test"
        cache = space_runtime.backend_main.SELECTION_PRECOMPUTE_CACHE
        lock = space_runtime.backend_main.SELECTION_PRECOMPUTE_LOCK
        try:
            with lock:
                cache[old_id] = {"created_at_epoch": 100.0, "image": object()}
                cache[fresh_id] = {"created_at_epoch": 950.0, "image": object()}
            space_runtime._discard_precompute_source_pixels(fresh_id)
            removed = space_runtime.cleanup_expired_selection_precomputes(
                max_age_seconds=200,
                now=1000.0,
            )
            self.assertEqual(removed, 1)
            with lock:
                self.assertNotIn(old_id, cache)
                self.assertIn(fresh_id, cache)
                self.assertNotIn("image", cache[fresh_id])
        finally:
            with lock:
                cache.pop(old_id, None)
                cache.pop(fresh_id, None)

    def test_cached_click_uses_packed_masks_without_another_model_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            image = Image.new("RGB", (16, 10), "white")
            image.save(image_path)
            mask = np.zeros((1, 10, 16), dtype=bool)
            mask[0, 2:9, 3:14] = True
            fingerprint = space_runtime.backend_main.selection_source_fingerprint(image)
            cached = {
                "packed_masks": np.packbits(mask, axis=2),
                "mask_width": 16,
                "scores": np.asarray([0.9], dtype=np.float32),
                "labels": ["person"],
                "image_size": (16, 10),
                "model_id": space_runtime.SAM3_MODEL,
            }
            with (
                patch.object(
                    space_runtime.backend_main,
                    "get_selection_precompute",
                    return_value=cached,
                ),
                patch.object(
                    space_runtime.backend_main,
                    "compute_sam3_selection_instances",
                ) as compute,
                patch.object(
                    space_runtime,
                    "_save_selection_job",
                    return_value={"job_id": "selected"},
                ) as save,
            ):
                result = space_runtime.select_precomputed_object(
                    image_path,
                    {
                        "precompute_id": "cached-selection",
                        "source_fingerprint": fingerprint,
                        "hover_size": [16, 10],
                        "region_instances": {"7": 0},
                    },
                    0.5,
                    0.5,
                    7,
                )
            self.assertEqual(result, {"job_id": "selected"})
            compute.assert_not_called()
            self.assertEqual(save.call_args.kwargs["model_status"], "sam3-concept-precomputed-point")

    def test_relief_route_matches_local_selected_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            job_id = "a" * 32
            job_dir = space_runtime.OUTPUT_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            stl_path = job_dir / "output_model.stl"
            stl_path.write_bytes(b"solid smoke\nendsolid smoke\n")
            preview_path = job_dir / "output_relief_preview.png"
            Image.new("L", (16, 8), 128).save(preview_path)
            diagnostics_path = job_dir / "diagnostics.json"
            diagnostics_path.write_text("{}", encoding="utf-8")
            selected_path = Path(temp_dir) / "selected.png"
            Image.new("RGB", (200, 100), "gray").save(selected_path)
            selection = {"job_id": "b" * 32, "selected": str(selected_path)}
            payload = {
                "job_id": job_id,
                "stl_model": f"{job_id}/output_model.stl",
                "diagnostics": f"{job_id}/diagnostics.json",
                "stl_diagnostics": {"is_watertight": True, "component_count": 1},
                "selection_mode": "context",
                "selection_subject_lock": True,
                "selection_crop": None,
                "selection_depth_context": {
                    "method": "full_scene_subject_locked_background_v1",
                    "subject_surface_locked": True,
                },
            }
            with (
                patch.object(space_runtime, "release_gpu_models"),
                patch.object(space_runtime, "_depth_model_source", return_value="/models/depth-v2"),
                patch.object(
                    space_runtime.BACKEND_CLIENT,
                    "post",
                    return_value=FakeResponse(payload),
                ) as post,
            ):
                result = space_runtime.generate_relief(
                    image_path,
                    "Select object",
                    selection,
                    50,
                    30,
                    2.4,
                    512,
                )
            request_data = post.call_args.kwargs["data"]
            self.assertEqual(
                set(request_data),
                {
                    "selection_mode",
                    "selection_subject_lock",
                    "selection_job_id",
                    "depth_provider",
                    "depth_model",
                    "device",
                    "depth_downsample_sharpening",
                    "target_dimension",
                    "z_scale",
                    "base_thickness_mm",
                    "max_xy_size",
                    "invert",
                    "relief_polarity",
                    "mesh_resolution_multiplier",
                    "printer_profile",
                    "printer_max_x_mm",
                    "printer_max_y_mm",
                    "printer_max_z_mm",
                    "printer_clearance_mm",
                    "print_scale_percent",
                    "sigma",
                    "detail_boost",
                    "printable_feature_depth_mm",
                    "feature_bridge_depth_mm",
                    "background_detail_boost",
                    "background_photo_detail_mm",
                    "trim_top_background",
                    "relief_gamma",
                    "base_border_px",
                    "detail_radius",
                    "low_percentile",
                    "high_percentile",
                    "max_relief_slope",
                    "nozzle_diameter_mm",
                    "minimum_feature_mm",
                    "face_refinement_mode",
                    "face_detail_strength",
                    "face_feather_ratio",
                    "face_max_correction_ratio",
                },
            )
            self.assertNotIn("completion_mode", request_data)
            self.assertNotIn("selection_background_depth_ratio", request_data)
            self.assertEqual(request_data["selection_mode"], "context")
            self.assertEqual(request_data["selection_subject_lock"], "true")
            self.assertEqual(request_data["device"], "auto")
            self.assertEqual(request_data["invert"], "false")
            self.assertEqual(request_data["target_dimension"], "512")
            self.assertEqual(request_data["base_thickness_mm"], "2.4")
            self.assertEqual(request_data["mesh_resolution_multiplier"], "2.0")
            self.assertEqual(request_data["printer_profile"], "Bambu Lab P1S")
            self.assertEqual(request_data["printer_max_x_mm"], "256")
            self.assertEqual(request_data["printer_max_y_mm"], "256")
            self.assertEqual(request_data["printer_max_z_mm"], "256")
            self.assertEqual(request_data["printer_clearance_mm"], "0")
            self.assertEqual(request_data["print_scale_percent"], "50.0")
            self.assertEqual(request_data["sigma"], "0.35")
            self.assertEqual(request_data["detail_boost"], "0.8")
            self.assertEqual(request_data["depth_downsample_sharpening"], "0.35")
            self.assertEqual(request_data["printable_feature_depth_mm"], "0.4")
            self.assertEqual(request_data["face_detail_strength"], "1.0")
            self.assertEqual(request_data["selection_job_id"], selection["job_id"])
            self.assertEqual(request_data["depth_model"], "/models/depth-v2")
            self.assertEqual(post.call_args.kwargs["files"]["file"][0], "selected.png")
            self.assertEqual(
                result[3]["summary"]["dimensions_mm"],
                {"x": 128.0, "y": 64.0, "z": 32.4},
            )
            self.assertEqual(result[3]["summary"]["relief_height_mm"], 30.0)
            self.assertEqual(result[3]["summary"]["base_thickness_mm"], 2.4)
            self.assertEqual(
                result[3]["summary"]["scope"],
                "selected-objects-local-context",
            )
            self.assertEqual(
                result[3]["summary"]["selection_mode"],
                "context",
            )
            self.assertEqual(
                result[3]["summary"]["local_relief_parity"],
                "full-scene-subject-lock-v1",
            )
            self.assertFalse(result[3]["summary"]["inpainting"])

    def test_relief_route_rejects_nonlocal_mesh_detail_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            with self.assertRaisesRegex(ValueError, "local frontend production preset"):
                space_runtime.generate_relief(
                    image_path,
                    "Full scene",
                    None,
                    50,
                    20,
                    2.4,
                    520,
                )

    def test_selected_relief_matches_local_full_scene_topology_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            image_path = root / "photo.png"
            rows, columns = np.indices((24, 32), dtype=np.uint8)
            rgb = np.stack(
                [
                    40 + columns * 4,
                    50 + rows * 5,
                    70 + ((rows.astype(np.uint16) + columns) % 80).astype(np.uint8),
                ],
                axis=-1,
            )
            image = Image.fromarray(rgb, mode="RGB")
            image.save(image_path)
            mask_values = np.zeros((24, 32), dtype=np.uint8)
            mask_values[5:22, 7:26] = 255
            mask = Image.fromarray(mask_values, mode="L")

            def fake_depth_data(input_path, output_dir, **_kwargs):
                with Image.open(input_path) as source:
                    width, height = source.size
                yy, xx = np.indices((height, width), dtype=np.float32)
                depth = 0.12 + 0.006 * yy + 0.009 * xx
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(depth_path, depth.astype(np.float32))
                Image.fromarray(
                    np.clip(depth / np.max(depth) * 255.0, 0, 255).astype(np.uint8),
                    mode="L",
                ).save(Path(output_dir) / "output_depth_preview.png")
                (Path(output_dir) / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "effective_model": "depth-anything/Depth-Anything-V2-Large-hf",
                            "relief_value_transform": "linear",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            def fake_face_refinement(_image_path, depth_path, _output_dir, **_kwargs):
                return depth_path, {
                    "mode": "auto",
                    "applied": False,
                    "detected_faces": 0,
                    "refined_faces": 0,
                    "faces": [],
                }

            with (
                patch.object(space_runtime, "OUTPUT_DIR", output_dir),
                patch.object(space_runtime.backend_main, "OUTPUT_DIR", output_dir),
                patch.object(space_runtime, "release_gpu_models"),
                patch.object(space_runtime, "_depth_model_source", return_value="depth-anything/Depth-Anything-V2-Large-hf"),
                patch.object(
                    space_runtime.backend_main,
                    "process_image_get_depth_data",
                    side_effect=fake_depth_data,
                ),
                patch.object(
                    space_runtime.backend_main,
                    "refine_depth_for_faces",
                    side_effect=fake_face_refinement,
                ),
                patch.object(space_runtime.backend_main, "RELIEF_MIN_DETAIL_DIMENSION", 32),
                patch.object(space_runtime.backend_main, "RELIEF_MAX_DETAIL_DIMENSION", 64),
            ):
                selection = space_runtime._save_selection_job(
                    image_path,
                    image,
                    mask,
                    resolved_model=space_runtime.SAM3_MODEL,
                    model_status="sam3-concept-precomputed-point",
                    labels=["person"],
                )
                selected_result = space_runtime.generate_relief(
                    image_path,
                    "Select object",
                    selection,
                    50,
                    10,
                    2.4,
                    384,
                )
                full_result = space_runtime.generate_relief(
                    image_path,
                    "Full scene",
                    None,
                    50,
                    10,
                    2.4,
                    384,
                )

            selected_job_dir = Path(selected_result[3]["full_report"]).parent
            full_job_dir = Path(full_result[3]["full_report"]).parent
            selected_surface = np.load(selected_job_dir / "output_surface.npy")
            full_surface = np.load(full_job_dir / "output_surface.npy")
            selected_mesh = trimesh.load_mesh(selected_result[0], force="mesh")
            full_mesh = trimesh.load_mesh(full_result[0], force="mesh")

            self.assertEqual(selected_surface.shape, full_surface.shape)
            np.testing.assert_array_equal(
                np.isfinite(selected_surface),
                np.isfinite(full_surface),
            )
            self.assertEqual(len(selected_mesh.faces), len(full_mesh.faces))
            self.assertTrue(selected_mesh.is_watertight)
            self.assertTrue(selected_mesh.is_winding_consistent)

    def test_model_and_source_revisions_are_immutable(self):
        self.assertEqual(
            space_runtime.PROJECT_REVISION,
            "b349d28e54af70acc1b36290d840ec9fbd2a1bff",
        )
        self.assertEqual(
            space_runtime.TRIPOSG_SOURCE_REVISION,
            "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c",
        )
        self.assertEqual(
            space_runtime.TRIPOSG_MODEL_REVISION,
            "2c1c516d22d58db486a058d98d31bb6177344e06",
        )
        self.assertEqual(
            space_runtime.DEPTH_MODEL_REVISION,
            "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
        )
        self.assertEqual(
            space_runtime.SAM3_MODEL_REVISION,
            "3c879f39826c281e95690f02c7821c4de09afae7",
        )

    def test_full_mesh_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            with self.assertRaisesRegex(ValueError, "requires Select object"):
                space_runtime.generate_full_mesh(image_path, "Full scene", None, 96, 42)

    def test_selected_mesh_input_uses_source_pixels_and_white_background(self):
        job_id = "c" * 32
        selection_dir = space_runtime.OUTPUT_DIR / "selection" / job_id
        selection_dir.mkdir(parents=True, exist_ok=True)
        source = Image.new("RGB", (40, 20), (10, 20, 30))
        source.save(selection_dir / "source.png")
        mask = Image.new("L", source.size, 0)
        for x in range(10, 30):
            for y in range(5, 15):
                mask.putpixel((x, y), 255)
        mask.save(selection_dir / "selection_mask.png")
        output = selection_dir / "mesh-input.png"
        prepared = space_runtime._prepare_selected_mesh_image({"job_id": job_id}, output, 64)
        self.assertEqual(prepared.size, (64, 64))
        self.assertEqual(prepared.getpixel((0, 0)), (255, 255, 255))
        self.assertIn((10, 20, 30), prepared.getdata())


if __name__ == "__main__":
    unittest.main()
