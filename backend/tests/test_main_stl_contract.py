import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.main as main_module


class MainStlContractTest(unittest.TestCase):
    def png_bytes(
        self,
        color=(245, 245, 245),
        accent=(20, 120, 220),
        size=(32, 24),
    ) -> bytes:
        image = Image.new("RGB", size, color)
        for x in range(size[0] // 4, min(size[0], size[0] // 4 + 10)):
            for y in range(size[1] // 4, min(size[1], size[1] // 4 + 12)):
                image.putpixel((x, y), accent)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def exif_rotated_jpeg_bytes(self) -> bytes:
        image = Image.new("RGB", (24, 12), (235, 235, 235))
        for x in range(6, 18):
            for y in range(3, 10):
                image.putpixel((x, y), (180, 40, 70))
        exif = image.getexif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        return buffer.getvalue()

    def test_loopback_frontend_ports_can_read_backend_health(self):
        client = TestClient(main_module.app)

        for origin in (
            "http://localhost:3017",
            "http://127.0.0.1:3017",
            "http://[::1]:3017",
        ):
            with self.subTest(origin=origin):
                response = client.options(
                    "/health",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "GET",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(
                    response.headers.get("access-control-allow-origin"),
                    origin,
                )

        remote_response = client.options(
            "/health",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(remote_response.status_code, 400, remote_response.text)
        self.assertNotIn("access-control-allow-origin", remote_response.headers)

    def test_depth_anything_large_is_default_and_only_recommended_model(self):
        self.assertEqual(main_module.DEFAULT_DEPTH_MODEL, "depth-anything/Depth-Anything-V2-Large-hf")
        recommended = [model for model in main_module.DEPTH_MODELS if model.get("recommended")]

        self.assertEqual([model["id"] for model in recommended], ["depth-anything/Depth-Anything-V2-Large-hf"])
        self.assertFalse(main_module.depth_model_far_is_high("depth-anything/Depth-Anything-V2-Large-hf"))
        self.assertTrue(main_module.depth_model_far_is_high("apple/DepthPro-hf"))
        self.assertEqual(main_module.relief_value_transform_for_model("apple/DepthPro-hf"), "inverse-depth")
        self.assertEqual(
            main_module.relief_value_transform_for_model("depth-anything/Depth-Anything-V2-Large-hf"),
            "linear",
        )

    def test_models_catalog_does_not_expose_image_completion(self):
        response = TestClient(main_module.app).get("/models")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertNotIn("completion_modes", payload)
        self.assertNotIn("completion_providers", payload)

    def test_process_image_rejects_legacy_completion_before_inference(self):
        response = TestClient(main_module.app).post(
            "/process_image",
            files={"file": ("source.png", self.png_bytes(), "image/png")},
            data={"completion_mode": "mirror-auto"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "not available in the production relief route",
            response.json()["detail"],
        )

    def test_relief_invert_tracks_effective_depth_model_semantics(self):
        self.assertTrue(
            main_module.relief_invert_for_model("apple/DepthPro-hf", "raised-print", requested_invert=False)
        )
        self.assertFalse(
            main_module.relief_invert_for_model(
                "depth-anything/Depth-Anything-V2-Large-hf",
                "raised-print",
                requested_invert=True,
            )
        )
        self.assertTrue(
            main_module.relief_invert_for_model(
                "depth-anything/Depth-Anything-V2-Large-hf",
                "mold",
                requested_invert=False,
            )
        )
        self.assertTrue(main_module.relief_invert_for_model("unknown", "manual", requested_invert=True))

    def test_relief_target_dimension_uses_printer_size_before_export(self):
        resolved = main_module.resolve_relief_target_dimension(
            80,
            max_xy_size=40,
            printer_max_x_mm=256,
            printer_max_y_mm=256,
            printer_clearance_mm=0,
            mesh_resolution_multiplier=2,
        )

        self.assertEqual(resolved, 512)
        self.assertAlmostEqual(main_module.relief_sample_pitch_mm(40, resolved), 40 / 511)

    def test_relief_target_dimension_is_capped_for_extreme_custom_printers(self):
        resolved = main_module.resolve_relief_target_dimension(
            200,
            max_xy_size=1000,
            printer_max_x_mm=1000,
            printer_max_y_mm=1000,
            mesh_resolution_multiplier=4,
        )

        self.assertEqual(resolved, main_module.RELIEF_MAX_DETAIL_DIMENSION)

    def test_minimum_feature_is_never_smaller_than_two_nozzle_widths(self):
        self.assertAlmostEqual(main_module.resolve_minimum_feature_mm(0.4, None), 0.8)
        self.assertAlmostEqual(main_module.resolve_minimum_feature_mm(0.6, 0.5), 1.2)
        self.assertAlmostEqual(main_module.resolve_minimum_feature_mm(0.4, 1.4), 1.4)

    def test_process_image_emits_output_model_and_diagnostics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                depth_path = Path(output_dir) / "output_depth_data.npy"
                rows, cols = np.indices((24, 32), dtype=np.float32)
                np.save(depth_path, 0.1 + 0.01 * rows + 0.02 * cols)
                return str(depth_path)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/process_image",
                    files={"file": ("relief.png", self.png_bytes(), "image/png")},
                    data={
                        "target_dimension": "80",
                        "z_scale": "10",
                        "max_xy_size": "40",
                        "invert": "false",
                        "sigma": "0",
                        "base_border_px": "0",
                        "printer_max_x_mm": "256",
                        "printer_max_y_mm": "256",
                        "printer_clearance_mm": "0",
                        "nozzle_diameter_mm": "0.6",
                        "minimum_feature_mm": "0.5",
                        "max_relief_slope": "1.5",
                        "mesh_resolution_multiplier": "2",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

                self.assertTrue(payload["stl_model"].endswith("/output_model.stl"))
                self.assertTrue(payload["diagnostics"].endswith("/diagnostics.json"))
                self.assertEqual(payload["stl_diagnostics"]["runner"], "depth-relief")
                self.assertEqual(payload["stl_diagnostics"]["artifact_contract"], "output_model.stl + diagnostics.json")
                self.assertTrue(payload["stl_diagnostics"]["stl_exists"])
                self.assertTrue(payload["stl_diagnostics"]["stl_is_watertight"])
                self.assertTrue(payload["stl_diagnostics"]["stl_is_volume"])
                self.assertTrue(payload["stl_diagnostics"]["stl_is_manifold"])
                provenance = payload["runtime"]["implementation_provenance"]
                self.assertTrue(provenance["available"])
                self.assertRegex(provenance["revision"], r"^[a-f0-9]{40}$")
                self.assertIn(provenance["clean"], (True, False))
                self.assertEqual(payload["requested_target_dimension"], 80)
                self.assertEqual(payload["target_dimension"], 512)
                self.assertAlmostEqual(payload["relief_sample_pitch_mm"], 40 / 511)
                self.assertTrue(payload["size_aware_detail"]["applied"])
                self.assertAlmostEqual(payload["minimum_feature_mm"], 1.2)
                self.assertAlmostEqual(payload["max_relief_slope"], 1.5)
                self.assertTrue(payload["relief_postprocess"]["enabled"])
                self.assertTrue(
                    payload["relief_postprocess"]["reference_surface"]["emitted"]
                )
                self.assertTrue(
                    (output_root / payload["job_id"] / "output_surface.npy").is_file()
                )
                self.assertTrue(
                    (
                        output_root
                        / payload["job_id"]
                        / "output_reference_surface.npy"
                    ).is_file()
                )
                self.assertAlmostEqual(payload["printer"]["nozzle_diameter_mm"], 0.6)

                diagnostics_response = client.get(payload["diagnostics_url"])
                self.assertEqual(diagnostics_response.status_code, 200)
                diagnostics = diagnostics_response.json()
                self.assertEqual(diagnostics["job_id"], payload["job_id"])
                self.assertTrue(diagnostics["stl_positive_volume"])
                metadata = json.loads((output_root / payload["job_id"] / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["target_dimension"], payload["target_dimension"])
                self.assertEqual(metadata["requested_target_dimension"], payload["requested_target_dimension"])
                self.assertEqual(metadata["relief_postprocess"], payload["relief_postprocess"])
                self.assertEqual(metadata["runtime"], payload["runtime"])

    def test_process_image_forwards_background_photo_detail_default_and_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            observed_detail_mm = []

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                depth_path = Path(output_dir) / "output_depth_data.npy"
                rows, cols = np.indices((24, 32), dtype=np.float32)
                np.save(depth_path, 0.1 + 0.01 * rows + 0.02 * cols)
                return str(depth_path)

            real_depth_to_model = main_module.depth_data_to_3d_model

            def capture_detail(*args, **kwargs):
                observed_detail_mm.append(kwargs["background_photo_detail_mm"])
                return real_depth_to_model(*args, **kwargs)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
                patch.object(main_module, "depth_data_to_3d_model", side_effect=capture_detail),
            ):
                client = TestClient(main_module.app)
                base_request = {
                    "target_dimension": "-1",
                    "z_scale": "10",
                    "max_xy_size": "24",
                    "sigma": "0",
                    "base_border_px": "0",
                }
                default_response = client.post(
                    "/process_image",
                    files={"file": ("default.png", self.png_bytes(), "image/png")},
                    data=base_request,
                )
                override_response = client.post(
                    "/process_image",
                    files={"file": ("override.png", self.png_bytes(), "image/png")},
                    data={**base_request, "background_photo_detail_mm": "0.27"},
                )

            self.assertEqual(default_response.status_code, 200, default_response.text)
            self.assertEqual(override_response.status_code, 200, override_response.text)
            self.assertEqual(observed_detail_mm, [0.60, 0.27])

    def test_process_image_rejects_invalid_background_photo_detail_before_work(self):
        client = TestClient(main_module.app)
        for invalid in ("nan", "inf", "-inf", "-0.01", "0.61"):
            with self.subTest(invalid=invalid):
                response = client.post(
                    "/process_image",
                    files={"file": ("invalid.png", self.png_bytes(), "image/png")},
                    data={"background_photo_detail_mm": invalid},
                )
                self.assertEqual(response.status_code, 422, response.text)

    def test_process_image_uses_original_selection_source_for_context_image_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            mask = Image.new("L", (32, 24), 0)
            for x in range(7, 25):
                for y in range(5, 21):
                    mask.putpixel((x, y), 255)
            mask_path = mask_dir / "selection_mask.png"
            mask.save(mask_path)
            inferred_pixels = []
            refined_pixels = []
            refinement_detection_sources = []
            mesh_source_pixels = []
            refinement_roi_masks = []

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(image_path, output_dir, **_kwargs):
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    inferred_pixels.append(
                        (rgb.getpixel((1, 1)), rgb.getpixel((12, 10)))
                    )
                rows, cols = np.indices((24, 32), dtype=np.float32)
                depth = 0.15 + 0.7 * np.exp(
                    -((rows - 13.0) ** 2 + (cols - 16.0) ** 2) / 95.0
                )
                output_path = Path(output_dir)
                depth_path = output_path / "output_depth_data.npy"
                np.save(depth_path, depth.astype(np.float32))
                (output_path / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "provider": "transformers",
                            "requested_model": "fixture",
                            "effective_model": "fixture",
                            "relief_value_transform": "linear",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            no_faces = {
                "mode": "auto",
                "applied": False,
                "detected_faces": 0,
                "refined_faces": 0,
                "faces": [],
            }

            def fake_face_refinement(image_path, depth, _output, **_kwargs):
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    refined_pixels.append((rgb.getpixel((1, 1)), rgb.getpixel((16, 13))))
                refinement_roi_masks.append(Path(_kwargs["detection_roi_mask"]))
                refinement_detection_sources.append(
                    Path(_kwargs["detection_image_path"])
                )
                return depth, no_faces

            real_depth_to_model = main_module.depth_data_to_3d_model

            def capture_mesh_source(*args, **kwargs):
                with Image.open(kwargs["source_image"]) as image:
                    rgb = image.convert("RGB")
                    mesh_source_pixels.append(
                        (rgb.getpixel((1, 1)), rgb.getpixel((16, 13)))
                    )
                return real_depth_to_model(*args, **kwargs)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
                patch.object(
                    main_module,
                    "refine_depth_for_faces",
                    side_effect=fake_face_refinement,
                ),
                patch.object(
                    main_module,
                    "depth_data_to_3d_model",
                    side_effect=capture_mesh_source,
                ),
            ):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={
                        "file": (
                            "original.png",
                            self.png_bytes(
                                color=(70, 80, 90),
                                accent=(200, 40, 20),
                            ),
                            "image/png",
                        )
                    },
                    data={"mask_paths_json": json.dumps(["selection/fixture/selection_mask.png"])},
                )
                self.assertEqual(compose_response.status_code, 200, compose_response.text)
                selection_job_id = compose_response.json()["job_id"]

                response = client.post(
                    "/process_image",
                    files={
                        "file": (
                            "untrusted-selected.png",
                            self.png_bytes(accent=(20, 120, 220)),
                            "image/png",
                        )
                    },
                    data={
                        "selection_job_id": selection_job_id,
                        "target_dimension": "-1",
                        "z_scale": "30",
                        "max_xy_size": "40",
                        "sigma": "0",
                        "base_border_px": "1",
                        "trim_top_background": "true",
                        "minimum_feature_mm": "0.8",
                        "max_relief_slope": "2.0",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(
                inferred_pixels,
                [((70, 80, 90), (200, 40, 20))],
            )
            self.assertEqual(
                refined_pixels,
                [((70, 80, 90), (200, 40, 20))],
            )
            self.assertEqual(
                mesh_source_pixels,
                [((70, 80, 90), (200, 40, 20))],
            )
            self.assertEqual(
                refinement_roi_masks,
                [output_root / "selection" / selection_job_id / "selection_mask.png"],
            )
            self.assertEqual(
                refinement_detection_sources,
                [
                    output_root
                    / "selection"
                    / selection_job_id
                    / "selected_image.png"
                ],
            )
            self.assertTrue(payload["selection_depth_context"]["enabled"])
            self.assertEqual(payload["selection_depth_context"]["selection_job_id"], selection_job_id)
            self.assertEqual(
                payload["selection_depth_context"]["method"],
                "full_scene_depth_with_bounded_background_context_v3",
            )
            self.assertEqual(
                payload["selection_depth_context"]["depth_source"],
                "selection_original_source",
            )
            self.assertTrue(
                payload["selection_depth_context"]["depth_source_file"].endswith(
                    "source.png"
                )
            )
            self.assertEqual(payload["selection_depth_context"]["background_depth_ratio"], 0.65)
            self.assertFalse(payload["effective_trim_top_background"])
            self.assertTrue(payload["depth_data"].endswith("output_depth_data_selected_context.npy"))
            self.assertTrue(payload["relief_postprocess"]["selection_gradient_compression"]["enabled"])
            self.assertTrue(payload["stl_diagnostics"]["stl_is_watertight"])

    def test_process_image_subject_lock_keeps_full_scene_depth_for_mesh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            mask = Image.new("L", (32, 24), 0)
            for x in range(7, 25):
                for y in range(5, 21):
                    mask.putpixel((x, y), 255)
            mask.save(mask_dir / "selection_mask.png")
            observed_mesh = {}

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                rows, cols = np.indices((24, 32), dtype=np.float32)
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(depth_path, 0.1 + 0.01 * rows + 0.02 * cols)
                (Path(output_dir) / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "effective_model": "fixture-depth",
                            "relief_value_transform": "linear",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            def fake_face_refinement(_image_path, depth_path, _output, **_kwargs):
                return depth_path, {
                    "mode": "auto",
                    "applied": False,
                    "detected_faces": 0,
                    "refined_faces": 0,
                    "faces": [],
                }

            def fake_depth_to_model(depth_path, output_stl_path, **kwargs):
                observed_mesh.update(
                    {
                        "depth_path": Path(depth_path),
                        "selection_subject_lock": kwargs["selection_subject_lock"],
                        "selection_region_mask": Path(kwargs["selection_region_mask"]),
                        "selection_background_depth_ratio": kwargs[
                            "selection_background_depth_ratio"
                        ],
                    }
                )
                Path(output_stl_path).write_bytes(b"solid fixture\nendsolid fixture\n")
                return {
                    "selection_subject_lock": kwargs["selection_subject_lock"],
                    "selection_gradient_compression": {
                        "enabled": False,
                        "reason": "subject_surface_locked",
                    },
                    "surface_grid_transform": {"emitted_shape": [24, 32]},
                }

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(
                    main_module,
                    "process_image_get_depth_data",
                    side_effect=fake_depth_data,
                ),
                patch.object(
                    main_module,
                    "refine_depth_for_faces",
                    side_effect=fake_face_refinement,
                ),
                patch.object(
                    main_module,
                    "depth_data_to_3d_model",
                    side_effect=fake_depth_to_model,
                ),
                patch.object(
                    main_module,
                    "stl_diagnostics",
                    return_value={
                        "stl_is_watertight": True,
                        "stl_passes_hard_checks": True,
                        "stl_failed_checks": [],
                    },
                ),
            ):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={
                        "file": (
                            "original.png",
                            self.png_bytes(
                                color=(70, 80, 90),
                                accent=(200, 40, 20),
                            ),
                            "image/png",
                        )
                    },
                    data={
                        "mask_paths_json": json.dumps(
                            ["selection/fixture/selection_mask.png"]
                        )
                    },
                )
                self.assertEqual(
                    compose_response.status_code,
                    200,
                    compose_response.text,
                )
                selection_job_id = compose_response.json()["job_id"]
                response = client.post(
                    "/process_image",
                    files={
                        "file": (
                            "selected.png",
                            self.png_bytes(accent=(20, 120, 220)),
                            "image/png",
                        )
                    },
                    data={
                        "selection_job_id": selection_job_id,
                        "selection_mode": "context",
                        "selection_subject_lock": "true",
                        "target_dimension": "-1",
                        "trim_top_background": "true",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertTrue(observed_mesh["selection_subject_lock"])
            self.assertEqual(observed_mesh["depth_path"].name, "output_depth_data.npy")
            self.assertEqual(
                observed_mesh["selection_region_mask"],
                output_root
                / "selection"
                / selection_job_id
                / "selection_mask.png",
            )
            self.assertEqual(
                observed_mesh["selection_background_depth_ratio"],
                0.65,
            )
            self.assertTrue(payload["selection_subject_lock"])
            self.assertTrue(payload["effective_trim_top_background"])
            self.assertEqual(
                payload["selection_depth_context"]["method"],
                "full_scene_subject_locked_background_v1",
            )
            self.assertTrue(
                payload["selection_depth_context"]["subject_surface_locked"]
            )
            self.assertTrue(payload["depth_data"].endswith("output_depth_data.npy"))
            self.assertFalse(
                (
                    output_root
                    / payload["job_id"]
                    / "output_depth_data_selected_context.npy"
                ).exists()
            )

    def test_process_image_rejects_tampered_selection_compose_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            Image.new("L", (32, 24), 255).save(mask_dir / "selection_mask.png")

            with patch.object(main_module, "OUTPUT_DIR", output_root):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={"file": ("source.png", self.png_bytes(), "image/png")},
                    data={"mask_paths_json": json.dumps(["selection/fixture/selection_mask.png"])},
                )
                self.assertEqual(compose_response.status_code, 200, compose_response.text)
                selection_job_id = compose_response.json()["job_id"]
                Image.new("RGB", (32, 24), (0, 0, 0)).save(
                    output_root / "selection" / selection_job_id / "source.png"
                )

                response = client.post(
                    "/process_image",
                    files={"file": ("selected.png", self.png_bytes(), "image/png")},
                    data={"selection_job_id": selection_job_id},
                )

            self.assertEqual(response.status_code, 409, response.text)
            self.assertIn("provenance", response.json()["detail"])

    def test_process_image_rejects_legacy_infilled_selection_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            Image.new("L", (32, 24), 255).save(
                mask_dir / "selection_mask.png"
            )

            with patch.object(main_module, "OUTPUT_DIR", output_root):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={
                        "file": (
                            "source.png",
                            self.png_bytes(),
                            "image/png",
                        )
                    },
                    data={
                        "mask_paths_json": json.dumps(
                            ["selection/fixture/selection_mask.png"]
                        )
                    },
                )
                self.assertEqual(
                    compose_response.status_code,
                    200,
                    compose_response.text,
                )
                payload = compose_response.json()
                metadata_path = output_root / payload["metadata"]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["selection_infill_mode"] = "clean-context"
                metadata_path.write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )

                response = client.post(
                    "/process_image",
                    files={
                        "file": (
                            "source.png",
                            self.png_bytes(),
                            "image/png",
                        )
                    },
                    data={"selection_job_id": payload["job_id"]},
                )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn(
            "created with removed image completion",
            response.json()["detail"],
        )

    def test_process_image_isolate_mode_crops_before_depth_and_masks_mesh_surface(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            mask = Image.new("L", (80, 60), 0)
            mask_array = np.asarray(mask).copy()
            mask_array[15:45, 20:60] = 255
            Image.fromarray(mask_array, mode="L").save(mask_dir / "selection_mask.png")
            observed = {"depth_size": None, "face_size": None, "mesh": None}

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(image_path, output_dir, **_kwargs):
                with Image.open(image_path) as image:
                    observed["depth_size"] = image.size
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(
                    depth_path,
                    np.linspace(0.0, 1.0, observed["depth_size"][0] * observed["depth_size"][1])
                    .reshape(observed["depth_size"][1], observed["depth_size"][0])
                    .astype(np.float32),
                )
                (Path(output_dir) / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "effective_model": "fixture-depth",
                            "relief_value_transform": "linear",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            def fake_face_refinement(image_path, depth_path, _output, **kwargs):
                with Image.open(image_path) as image:
                    observed["face_size"] = image.size
                self.assertTrue(str(kwargs["detection_roi_mask"]).endswith("selection_mask_crop.png"))
                return depth_path, {
                    "mode": "auto",
                    "applied": False,
                    "detected_faces": 0,
                    "refined_faces": 0,
                    "faces": [],
                }

            def fake_depth_to_model(depth_path, output_stl_path, **kwargs):
                depth = np.load(depth_path)
                observed["mesh"] = {
                    "shape": depth.shape,
                    "finite_ratio": float(np.mean(np.isfinite(depth))),
                    "selection_region_mask": kwargs["selection_region_mask"],
                    "selection_background_depth_ratio": kwargs[
                        "selection_background_depth_ratio"
                    ],
                }
                Path(output_stl_path).write_bytes(b"solid fixture\nendsolid fixture\n")
                return {
                    "surface_grid_transform": {
                        "emitted_shape": [int(depth.shape[0]), int(depth.shape[1])]
                    }
                }

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(
                    main_module,
                    "process_image_get_depth_data",
                    side_effect=fake_depth_data,
                ),
                patch.object(
                    main_module,
                    "refine_depth_for_faces",
                    side_effect=fake_face_refinement,
                ),
                patch.object(
                    main_module,
                    "depth_data_to_3d_model",
                    side_effect=fake_depth_to_model,
                ),
                patch.object(
                    main_module,
                    "stl_diagnostics",
                    return_value={
                        "stl_is_watertight": True,
                        "stl_passes_hard_checks": True,
                        "stl_failed_checks": [],
                    },
                ),
            ):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={
                        "file": (
                            "source.png",
                            self.png_bytes(size=(80, 60), color=(30, 50, 70)),
                            "image/png",
                        )
                    },
                    data={
                        "mask_paths_json": json.dumps(
                            ["selection/fixture/selection_mask.png"]
                        )
                    },
                )
                self.assertEqual(compose_response.status_code, 200, compose_response.text)
                response = client.post(
                    "/process_image",
                    files={
                        "file": (
                            "selected.png",
                            self.png_bytes(size=(80, 60)),
                            "image/png",
                        )
                    },
                    data={
                        "selection_job_id": compose_response.json()["job_id"],
                        "selection_mode": "isolate",
                        "target_dimension": "-1",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(observed["depth_size"], (56, 46))
            self.assertEqual(observed["face_size"], (56, 46))
            self.assertEqual(observed["mesh"]["shape"], (46, 56))
            self.assertLess(observed["mesh"]["finite_ratio"], 0.6)
            self.assertIsNone(observed["mesh"]["selection_region_mask"])
            self.assertEqual(observed["mesh"]["selection_background_depth_ratio"], 0.0)
            self.assertEqual(payload["selection_mode"], "isolate")
            self.assertEqual(
                payload["selection_depth_context"]["method"],
                "crop_first_isolated_selection_v1",
            )
            self.assertEqual(payload["selection_crop"]["crop_size"], [56, 46])
            self.assertTrue(payload["depth_data"].endswith("output_depth_data_selected_isolate.npy"))

    def test_process_image_source_depth_isolate_crops_full_source_depth_before_mesh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            mask_dir = output_root / "selection" / "fixture"
            mask_dir.mkdir(parents=True)
            mask_array = np.zeros((60, 80), dtype=np.uint8)
            mask_array[15:45, 20:38] = 255
            mask_array[28:45, 38:60] = 255
            Image.fromarray(mask_array, mode="L").save(mask_dir / "selection_mask.png")
            observed = {"depth_size": None, "face": None, "mesh": None}
            full_depth = np.linspace(0.0, 1.0, 40 * 30).reshape(30, 40).astype(np.float32)
            face_weight = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (30, 1))
            face_region = np.zeros((30, 40), dtype=np.uint8)
            face_region[7:24, 11:31] = 255
            face_occlusion = np.flip(face_weight, axis=1).copy()

            def fake_depth_data(image_path, output_dir, **_kwargs):
                with Image.open(image_path) as image:
                    observed["depth_size"] = image.size
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(depth_path, full_depth)
                (Path(output_dir) / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "effective_model": "fixture-depth",
                            "relief_value_transform": "linear",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            def fake_face_refinement(image_path, depth_path, _output, **kwargs):
                with Image.open(image_path) as image:
                    image_size = image.size
                with Image.open(kwargs["detection_image_path"]) as detection_image:
                    detection_size = detection_image.size
                Image.fromarray(face_weight, mode="L").save(
                    Path(_output) / "output_face_refinement_weight.png"
                )
                Image.fromarray(face_region, mode="L").save(
                    Path(_output) / "output_face_refinement_region.png"
                )
                Image.fromarray(face_occlusion, mode="L").save(
                    Path(_output) / "output_face_refinement_occlusion.png"
                )
                (Path(_output) / "output_face_refinement_metadata.json").write_text(
                    json.dumps({"scope": "full-source"}),
                    encoding="utf-8",
                )
                observed["face"] = {
                    "image_size": image_size,
                    "depth_shape": np.load(depth_path).shape,
                    "roi": Path(kwargs["detection_roi_mask"]).name,
                    "detection_size": detection_size,
                }
                return depth_path, {
                    "mode": "auto",
                    "applied": True,
                    "detected_faces": 1,
                    "refined_faces": 1,
                    "faces": [],
                    "weight_file": "output_face_refinement_weight.png",
                    "region_file": "output_face_refinement_region.png",
                    "occlusion_file": "output_face_refinement_occlusion.png",
                    "metadata_file": "output_face_refinement_metadata.json",
                }

            def load_luma(path):
                with Image.open(path) as image:
                    return np.asarray(image.convert("L")).copy()

            def fake_depth_to_model(depth_path, output_stl_path, **kwargs):
                depth = np.load(depth_path)
                with Image.open(kwargs["source_image"]) as source_image:
                    source_size = source_image.size
                observed["mesh"] = {
                    "shape": depth.shape,
                    "depth": depth,
                    "source_size": source_size,
                    "selection_region_mask": kwargs["selection_region_mask"],
                    "selection_background_depth_ratio": kwargs[
                        "selection_background_depth_ratio"
                    ],
                    "weight": load_luma(kwargs["feature_weight_mask"]),
                    "region": load_luma(kwargs["face_region_mask"]),
                    "occlusion": load_luma(kwargs["feature_exclusion_mask"]),
                    "weight_name": Path(kwargs["feature_weight_mask"]).name,
                }
                Path(output_stl_path).write_bytes(b"solid fixture\nendsolid fixture\n")
                return {
                    "surface_grid_transform": {
                        "emitted_shape": [int(depth.shape[0]), int(depth.shape[1])]
                    }
                }

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(
                    main_module,
                    "process_image_get_depth_data",
                    side_effect=fake_depth_data,
                ),
                patch.object(
                    main_module,
                    "refine_depth_for_faces",
                    side_effect=fake_face_refinement,
                ),
                patch.object(
                    main_module,
                    "depth_data_to_3d_model",
                    side_effect=fake_depth_to_model,
                ),
                patch.object(
                    main_module,
                    "stl_diagnostics",
                    return_value={
                        "stl_is_watertight": True,
                        "stl_passes_hard_checks": True,
                        "stl_failed_checks": [],
                    },
                ),
            ):
                client = TestClient(main_module.app)
                compose_response = client.post(
                    "/selection/compose",
                    files={
                        "file": (
                            "source.png",
                            self.png_bytes(size=(80, 60), color=(30, 50, 70)),
                            "image/png",
                        )
                    },
                    data={
                        "mask_paths_json": json.dumps(
                            ["selection/fixture/selection_mask.png"]
                        )
                    },
                )
                self.assertEqual(compose_response.status_code, 200, compose_response.text)
                response = client.post(
                    "/process_image",
                    files={
                        "file": (
                            "selected.png",
                            self.png_bytes(size=(80, 60)),
                            "image/png",
                        )
                    },
                    data={
                        "selection_job_id": compose_response.json()["job_id"],
                        "selection_mode": "source-depth-isolate",
                        "target_dimension": "-1",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(observed["depth_size"], (80, 60))
            self.assertEqual(observed["face"]["image_size"], (80, 60))
            self.assertEqual(observed["face"]["depth_shape"], (30, 40))
            self.assertEqual(observed["face"]["roi"], "selection_mask.png")
            self.assertEqual(observed["face"]["detection_size"], (80, 60))
            self.assertEqual(observed["mesh"]["shape"], (46, 56))
            self.assertEqual(observed["mesh"]["source_size"], (56, 46))
            self.assertIsNone(observed["mesh"]["selection_region_mask"])
            self.assertEqual(observed["mesh"]["selection_background_depth_ratio"], 0.0)

            expected_selection = mask_array[7:53, 12:68] > 0
            self.assertTrue(
                np.array_equal(np.isfinite(observed["mesh"]["depth"]), expected_selection)
            )
            aligned_depth = np.asarray(
                Image.fromarray(full_depth, mode="F").resize(
                    (80, 60),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                observed["mesh"]["depth"][expected_selection],
                aligned_depth[7:53, 12:68][expected_selection],
                rtol=0.0,
                atol=1e-6,
            )

            expected_face_artifacts = {
                "weight": np.asarray(
                    Image.fromarray(face_weight, mode="L").resize(
                        (80, 60), Image.Resampling.BILINEAR
                    )
                )[7:53, 12:68],
                "region": np.asarray(
                    Image.fromarray(face_region, mode="L").resize(
                        (80, 60), Image.Resampling.NEAREST
                    )
                )[7:53, 12:68],
                "occlusion": np.asarray(
                    Image.fromarray(face_occlusion, mode="L").resize(
                        (80, 60), Image.Resampling.BILINEAR
                    )
                )[7:53, 12:68],
            }
            for artifact_name, expected_artifact in expected_face_artifacts.items():
                self.assertTrue(
                    np.array_equal(observed["mesh"][artifact_name], expected_artifact),
                    artifact_name,
                )
            self.assertEqual(
                observed["mesh"]["weight_name"],
                "output_face_refinement_weight_crop.png",
            )
            cropped_face_metadata_path = (
                output_root
                / payload["job_id"]
                / payload["face_refinement"]["metadata_file"]
            )
            cropped_face_metadata = json.loads(
                cropped_face_metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                cropped_face_metadata["weight_file"],
                "output_face_refinement_weight_crop.png",
            )
            self.assertEqual(
                cropped_face_metadata["source_aligned_selection_crop"]["crop_size"],
                [56, 46],
            )
            self.assertEqual(
                cropped_face_metadata["full_source_metadata_file"],
                "output_face_refinement_metadata.json",
            )
            full_source_face_metadata = json.loads(
                (
                    output_root
                    / payload["job_id"]
                    / cropped_face_metadata["full_source_metadata_file"]
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(full_source_face_metadata["scope"], "full-source")
            self.assertEqual(payload["selection_mode"], "source-depth-isolate")
            self.assertEqual(
                payload["selection_depth_context"]["method"],
                "full_source_depth_then_isolated_crop_v1",
            )
            self.assertEqual(
                payload["selection_depth_context"]["depth_source"],
                "selection_original_source",
            )
            self.assertEqual(payload["selection_crop"]["crop_size"], [56, 46])
            self.assertEqual(
                payload["selection_depth_context"]["source_depth_crop"]["source_depth_size"],
                [40, 30],
            )
            self.assertEqual(
                payload["selection_depth_context"]["source_depth_crop"][
                    "aligned_source_depth_size"
                ],
                [80, 60],
            )
            self.assertTrue(
                payload["selection_depth_context"]["source_depth_crop"][
                    "resampled_to_source_crop"
                ]
            )
            self.assertTrue(
                payload["depth_data"].endswith(
                    "output_depth_data_selected_source_isolate.npy"
                )
            )

    def test_process_image_normalizes_exif_orientation_and_skips_completion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            observed_sizes = {"completion": [], "depth": [], "face": [], "mesh": []}

            def image_size(path):
                with Image.open(path) as image:
                    return image.size

            def fake_complete_image(input_path, **_kwargs):
                observed_sizes["completion"].append(image_size(input_path))
                return input_path, None

            def fake_depth_data(image_path, output_dir, **_kwargs):
                width, height = image_size(image_path)
                observed_sizes["depth"].append((width, height))
                rows, cols = np.indices((height, width), dtype=np.float32)
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(depth_path, 0.1 + 0.01 * rows + 0.02 * cols)
                return str(depth_path)

            no_faces = {
                "mode": "auto",
                "applied": False,
                "detected_faces": 0,
                "refined_faces": 0,
                "faces": [],
            }

            def fake_face_refinement(image_path, depth, _output, **_kwargs):
                observed_sizes["face"].append(image_size(image_path))
                return depth, no_faces

            real_depth_to_model = main_module.depth_data_to_3d_model

            def capture_mesh_source(*args, **kwargs):
                observed_sizes["mesh"].append(image_size(kwargs["source_image"]))
                return real_depth_to_model(*args, **kwargs)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
                patch.object(main_module, "refine_depth_for_faces", side_effect=fake_face_refinement),
                patch.object(main_module, "depth_data_to_3d_model", side_effect=capture_mesh_source),
            ):
                response = TestClient(main_module.app).post(
                    "/process_image",
                    files={
                        "file": (
                            "phone.jpg",
                            self.exif_rotated_jpeg_bytes(),
                            "image/jpeg",
                        )
                    },
                    data={
                        "target_dimension": "-1",
                        "z_scale": "10",
                        "max_xy_size": "24",
                        "sigma": "0",
                        "base_border_px": "1",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(observed_sizes, {
                "completion": [],
                "depth": [(12, 24)],
                "face": [(12, 24)],
                "mesh": [(12, 24)],
            })

    def test_process_image_reports_depth_fallback_and_uses_effective_polarity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                output_path = Path(output_dir)
                depth_path = output_path / "output_depth_data.npy"
                rows, cols = np.indices((24, 32), dtype=np.float32)
                np.save(depth_path, 0.1 + 0.01 * rows + 0.02 * cols)
                (output_path / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "provider": "transformers",
                            "requested_model": "apple/DepthPro-hf",
                            "effective_model": "depth-anything/Depth-Anything-V2-Large-hf",
                            "fallback_model": "depth-anything/Depth-Anything-V2-Large-hf",
                            "fallback_reason": "Depth Pro weights are not cached.",
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/process_image",
                    files={"file": ("relief.png", self.png_bytes(), "image/png")},
                    data={
                        "depth_model": "apple/DepthPro-hf",
                        "relief_polarity": "raised-print",
                        "target_dimension": "-1",
                        "z_scale": "10",
                        "invert": "true",
                        "sigma": "0",
                        "base_border_px": "0",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

        self.assertEqual(payload["requested_depth_model"], "apple/DepthPro-hf")
        self.assertEqual(payload["depth_model"], "depth-anything/Depth-Anything-V2-Large-hf")
        self.assertEqual(payload["depth_fallback_model"], "depth-anything/Depth-Anything-V2-Large-hf")
        self.assertEqual(payload["depth_fallback_reason"], "Depth Pro weights are not cached.")
        self.assertFalse(payload["invert"])
        self.assertTrue(payload["requested_invert"])
        self.assertEqual(payload["relief_value_transform"], "linear")
        self.assertEqual(payload["depth_metadata"]["effective_model"], "depth-anything/Depth-Anything-V2-Large-hf")

    def test_depthpro_preload_status_reports_missing_cache_without_network(self):
        original_state = dict(main_module.DEPTH_PRELOAD_STATE)
        with tempfile.TemporaryDirectory() as temp_dir:
            main_module.DEPTH_PRELOAD_STATE.update(
                {
                    "status": "idle",
                    "message": "",
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "total_bytes": 100,
                }
            )
            with (
                patch.object(main_module, "_hf_model_cache_dir", return_value=Path(temp_dir) / "hf-cache"),
                patch.object(main_module, "_hf_cached_file_path", return_value=None),
            ):
                client = TestClient(main_module.app)
                response = client.get("/depth/preload/depthpro/status")

            main_module.DEPTH_PRELOAD_STATE.clear()
            main_module.DEPTH_PRELOAD_STATE.update(original_state)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "missing")
        self.assertFalse(payload["complete"])
        self.assertEqual(payload["missing_files"], list(main_module.DEPTHPRO_REQUIRED_FILES))
        self.assertEqual(payload["total_bytes"], 100)

    def test_depthpro_preload_endpoint_starts_background_download_without_blocking(self):
        original_state = dict(main_module.DEPTH_PRELOAD_STATE)
        started: list[dict] = []

        class FakeThread:
            def __init__(self, *, target, kwargs, daemon):
                started.append({"target": target, "kwargs": kwargs, "daemon": daemon})

            def start(self):
                started[-1]["started"] = True

        try:
            main_module.DEPTH_PRELOAD_STATE.update({"status": "idle", "error": None})
            with (
                patch.object(main_module, "_depthpro_cache_status", return_value={"complete": False, "status": "missing"}),
                patch.object(main_module, "Thread", FakeThread),
            ):
                client = TestClient(main_module.app)
                response = client.post("/depth/preload/depthpro")
        finally:
            main_module.DEPTH_PRELOAD_STATE.clear()
            main_module.DEPTH_PRELOAD_STATE.update(original_state)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "missing")
        self.assertEqual(len(started), 1)
        self.assertIs(started[0]["target"], main_module._depthpro_preload_worker)
        self.assertEqual(started[0]["kwargs"], {"force": False})
        self.assertTrue(started[0]["daemon"])
        self.assertTrue(started[0]["started"])

    def test_selection_keep_requires_at_least_one_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/keep",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={"points_json": "[]"},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Click at least one", response.json()["detail"])

    def test_selection_keep_rejects_malformed_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/keep",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={"points_json": json.dumps([{"x": "left", "y": 0.5}])},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("non-numeric", response.json()["detail"])

    def test_selection_keep_fallback_writes_fetchable_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "sam2_selection_mask", side_effect=RuntimeError("checkpoint not cached locally")),
                patch.object(main_module, "panoptic_selection_mask", side_effect=RuntimeError("panoptic checkpoint not cached locally")),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/keep",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "points_json": json.dumps([{"x": 0.38, "y": 0.5}]),
                        "model_id": "sam2.1-hiera-large",
                        "background_mode": "white",
                        "mask_max_dimension": "64",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

                for url_field in ("selected_image_url", "mask_url", "overlay_url", "tint_url", "metadata_url"):
                    artifact_response = client.get(payload[url_field])
                    self.assertEqual(artifact_response.status_code, 200, url_field)

        self.assertEqual(payload["model_status"], "fallback-click-region")
        self.assertIn("SAM2 checkpoint is not cached locally", payload["model_error"])
        self.assertIn("panoptic", payload["model_error"])
        self.assertGreater(payload["mask_pixels"], 0)
        self.assertGreater(payload["mask_coverage"], 0.0)
        self.assertEqual(payload["background_mode"], "white")

    def test_selection_mask_preview_writes_tint_and_mask_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "sam2_selection_mask", side_effect=RuntimeError("checkpoint not cached locally")),
                patch.object(main_module, "panoptic_selection_mask", side_effect=RuntimeError("panoptic checkpoint not cached locally")),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/mask",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "points_json": json.dumps([{"x": 0.38, "y": 0.5}]),
                        "model_id": "sam2.1-hiera-large",
                        "mask_max_dimension": "64",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

                for url_field in ("mask_url", "overlay_url", "tint_url", "metadata_url"):
                    artifact_response = client.get(payload[url_field])
                    self.assertEqual(artifact_response.status_code, 200, url_field)

        self.assertNotIn("selected_image_url", payload)
        self.assertEqual(payload["model_status"], "fallback-click-region")
        self.assertGreater(payload["mask_pixels"], 0)

    def test_sam2_candidate_selection_prefers_complete_seeded_object(self):
        candidates = np.zeros((3, 20, 20), dtype=bool)
        candidates[0, 4:18, 7:13] = True
        candidates[0, 1:3, 1:3] = True
        candidates[1, :, :] = True
        candidates[2, 8:14, 8:12] = True

        selected, selected_index = main_module.select_sam2_object_candidate(
            candidates,
            np.asarray([0.62, 0.70, 0.90], dtype=np.float32),
            [[9.0, 10.0]],
        )

        self.assertEqual(selected_index, 0)
        self.assertTrue(np.all(selected[4:18, 7:13]))
        self.assertFalse(np.any(selected[1:3, 1:3]))
        self.assertLess(float(selected.mean()), main_module.SAM2_SELECTION_MAX_COVERAGE)

    def test_sam2_candidate_selection_expands_credible_shirt_to_whole_person(self):
        candidates = np.zeros((3, 40, 40), dtype=bool)
        candidates[0, 20:40, 14:28] = True
        candidates[1, 4:40, 7:34] = True
        candidates[2, 22:40, 16:27] = True

        selected, selected_index = main_module.select_sam2_object_candidate(
            candidates,
            np.asarray([0.96, 0.26, 0.38], dtype=np.float32),
            [[20.0, 28.0]],
        )

        self.assertEqual(selected_index, 1)
        self.assertTrue(np.all(selected[4:40, 7:34]))

    def test_sam2_candidate_selection_rejects_uncredible_large_region(self):
        candidates = np.zeros((3, 40, 40), dtype=bool)
        candidates[0, 20:30, 16:24] = True
        candidates[1, 2:38, 2:38] = True
        candidates[2, 18:32, 14:26] = True

        _selected, selected_index = main_module.select_sam2_object_candidate(
            candidates,
            np.asarray([0.82, 0.02, 0.57], dtype=np.float32),
            [[20.0, 24.0]],
        )

        self.assertEqual(selected_index, 2)

    def test_sam3_bounded_hole_fill_repairs_shirt_texture_without_filling_large_gap(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[4:36, 4:36] = True
        mask[10:12, 10:12] = False
        mask[18:28, 18:28] = False

        filled = main_module.fill_bounded_selection_holes(mask, 0.01)

        self.assertTrue(np.all(filled[10:12, 10:12]))
        self.assertFalse(np.any(filled[18:28, 18:28]))

    def test_sam3_person_instance_is_identical_from_face_or_shirt_click(self):
        masks = np.zeros((2, 24, 32), dtype=bool)
        masks[0, 3:24, 4:14] = True
        masks[1, 4:24, 18:29] = True
        scores = np.asarray([0.9, 0.8], dtype=np.float32)

        face_mask, face_ids, face_labels = main_module.sam3_selection_mask_from_instances(
            masks,
            scores,
            ["person", "person"],
            [{"x": 0.25, "y": 0.25}],
            (32, 24),
        )
        shirt_mask, shirt_ids, shirt_labels = main_module.sam3_selection_mask_from_instances(
            masks,
            scores,
            ["person", "person"],
            [{"x": 0.25, "y": 0.75}],
            (32, 24),
        )

        self.assertEqual(face_ids, [0])
        self.assertEqual(shirt_ids, [0])
        self.assertEqual(face_labels, ["person"])
        self.assertEqual(shirt_labels, ["person"])
        np.testing.assert_array_equal(np.asarray(face_mask), np.asarray(shirt_mask))

        packed_face_mask, packed_ids, packed_labels = (
            main_module.sam3_selection_mask_from_packed_instances(
                np.packbits(masks, axis=2),
                32,
                scores,
                ["person", "person"],
                [{"x": 0.25, "y": 0.25}],
                (32, 24),
            )
        )
        self.assertEqual(packed_ids, face_ids)
        self.assertEqual(packed_labels, face_labels)
        np.testing.assert_array_equal(np.asarray(packed_face_mask), np.asarray(face_mask))

    def test_sam3_checkpoint_loading_is_local_by_default_and_opt_in_download(self):
        with (
            patch.dict(main_module.os.environ, {"SELECTION_ALLOW_MODEL_DOWNLOAD": ""}),
            patch.object(
                main_module,
                "_cached_selection_snapshot",
                return_value=Path("cached-sam3"),
            ) as cached_snapshot,
        ):
            source, kwargs = main_module.selection_checkpoint_load_source(
                "facebook/sam3",
                "pinned-revision",
            )
        self.assertEqual(source, "cached-sam3")
        self.assertEqual(kwargs, {"local_files_only": True})
        cached_snapshot.assert_called_once_with("facebook/sam3", "pinned-revision")

        with patch.dict(main_module.os.environ, {"SELECTION_ALLOW_MODEL_DOWNLOAD": "1"}):
            source, kwargs = main_module.selection_checkpoint_load_source(
                "facebook/sam3",
                "pinned-revision",
            )
        self.assertEqual(source, "facebook/sam3")
        self.assertEqual(
            kwargs,
            {"revision": "pinned-revision", "local_files_only": False},
        )

    def test_selection_sam3_precompute_reuses_person_instances(self):
        def fake_compute(image, device="auto"):
            masks = np.zeros((1, image.height, image.width), dtype=bool)
            masks[0, 3:image.height, 6:20] = True
            return masks, np.asarray([0.95], dtype=np.float32), ["person"], "facebook/sam3"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "compute_sam3_selection_instances", side_effect=fake_compute) as compute_mock,
            ):
                main_module.SELECTION_PRECOMPUTE_CACHE.clear()
                client = TestClient(main_module.app)
                precompute_response = client.post(
                    "/selection/precompute",
                    files={"file": ("person.png", self.png_bytes(), "image/png")},
                    data={"model_id": "sam3-person-aware"},
                )
                self.assertEqual(precompute_response.status_code, 200, precompute_response.text)
                precompute_payload = precompute_response.json()
                mask_response = client.post(
                    "/selection/precomputed_mask",
                    data={
                        "precompute_id": precompute_payload["precompute_id"],
                        "points_json": json.dumps([{"x": 0.3, "y": 0.5}]),
                    },
                )
                self.assertEqual(mask_response.status_code, 200, mask_response.text)
                payload = mask_response.json()

        self.assertEqual(compute_mock.call_count, 1)
        self.assertEqual(precompute_payload["model_status"], "sam3-concepts-precomputed")
        self.assertEqual(precompute_payload["segment_count"], 1)
        self.assertEqual(payload["model_status"], "sam3-concept-precomputed-point")
        self.assertEqual(payload["selection_labels"], ["person"])
        self.assertGreater(payload["mask_pixels"], 0)
        main_module.SELECTION_PRECOMPUTE_CACHE.clear()

    def test_selection_sam3_precompute_uses_tracker_outside_person(self):
        masks = np.zeros((1, 24, 32), dtype=bool)
        masks[3:24, 6:16] = True
        tracker_mask = Image.new("L", (32, 24), 0)
        for x in range(22, 30):
            for y in range(6, 18):
                tracker_mask.putpixel((x, y), 255)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(
                    main_module,
                    "compute_sam3_selection_instances",
                    return_value=(masks, np.asarray([0.95], dtype=np.float32), ["person"], "facebook/sam3"),
                ),
                patch.object(
                    main_module,
                    "sam3_tracker_selection_mask",
                    return_value=(tracker_mask, "facebook/sam3"),
                ) as tracker_mock,
            ):
                main_module.SELECTION_PRECOMPUTE_CACHE.clear()
                client = TestClient(main_module.app)
                precompute_payload = client.post(
                    "/selection/precompute",
                    files={"file": ("scene.png", self.png_bytes(), "image/png")},
                    data={"model_id": "sam3-person-aware"},
                ).json()
                response = client.post(
                    "/selection/precomputed_mask",
                    data={
                        "precompute_id": precompute_payload["precompute_id"],
                        "points_json": json.dumps([{"x": 0.8, "y": 0.5}]),
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model_status"], "sam3-tracker-fallback-point")
        self.assertEqual(tracker_mock.call_count, 1)
        main_module.SELECTION_PRECOMPUTE_CACHE.clear()

    def test_selection_sam3_precompute_unions_concept_and_unmatched_tracker_points(self):
        masks = np.zeros((1, 24, 32), dtype=bool)
        masks[0, 3:24, 6:16] = True
        tracker_mask = Image.new("L", (32, 24), 0)
        for x in range(22, 30):
            for y in range(6, 18):
                tracker_mask.putpixel((x, y), 255)

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(
                    main_module,
                    "compute_sam3_selection_instances",
                    return_value=(masks, np.asarray([0.95], dtype=np.float32), ["person"], "facebook/sam3"),
                ),
                patch.object(
                    main_module,
                    "sam3_tracker_selection_mask",
                    return_value=(tracker_mask, "facebook/sam3"),
                ) as tracker_mock,
            ):
                main_module.SELECTION_PRECOMPUTE_CACHE.clear()
                client = TestClient(main_module.app)
                precompute_payload = client.post(
                    "/selection/precompute",
                    files={"file": ("scene.png", self.png_bytes(), "image/png")},
                    data={"model_id": "sam3-person-aware"},
                ).json()
                response = client.post(
                    "/selection/precomputed_mask",
                    data={
                        "precompute_id": precompute_payload["precompute_id"],
                        "points_json": json.dumps([
                            {"x": 0.3, "y": 0.5},
                            {"x": 0.8, "y": 0.5},
                        ]),
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["model_status"], "sam3-concept-precomputed+tracker-point")
        self.assertEqual(payload["selected_segment_ids"], [0])
        self.assertEqual(payload["selection_labels"], ["person"])
        self.assertGreater(payload["mask_pixels"], int(masks[0].sum()))
        tracker_mock.assert_called_once()
        main_module.SELECTION_PRECOMPUTE_CACHE.clear()

    def test_release_selection_models_clears_cached_cuda_models(self):
        original_cache = main_module.SELECTION_MODEL_CACHE
        try:
            main_module.SELECTION_MODEL_CACHE = {("model", "revision", "cuda"): object()}
            main_module.SELECTION_PRECOMPUTE_CACHE["cached"] = {"kind": "test"}
            with patch("torch.cuda.is_available", return_value=False):
                main_module.release_selection_models()
            self.assertEqual(main_module.SELECTION_MODEL_CACHE, {})
            self.assertIn("cached", main_module.SELECTION_PRECOMPUTE_CACHE)
        finally:
            main_module.SELECTION_MODEL_CACHE = original_cache
            main_module.SELECTION_PRECOMPUTE_CACHE.clear()

    def test_runtime_release_endpoint_yields_all_heavy_model_caches(self):
        original_cache = main_module.SELECTION_MODEL_CACHE
        try:
            main_module.SELECTION_MODEL_CACHE = {("model", "revision", "cuda"): object()}
            with (
                patch.object(main_module, "release_selection_models") as release_selection,
                patch.object(main_module, "release_depth_pipelines") as release_depth,
                patch.object(main_module, "release_inpaint_pipelines") as release_inpaint,
                patch("torch.cuda.is_available", return_value=False),
            ):
                response = TestClient(main_module.app).post("/runtime/release-models")

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["status"], "released")
            self.assertEqual(payload["selection_models_released"], 1)
            self.assertEqual(payload["released_caches"], ["selection", "depth", "inpaint"])
            release_selection.assert_called_once_with()
            release_depth.assert_called_once_with()
            release_inpaint.assert_called_once_with()
        finally:
            main_module.SELECTION_MODEL_CACHE = original_cache

    def test_selection_mask_preview_can_use_panoptic_segmenter(self):
        def fake_panoptic(image, points, device="auto"):
            mask = Image.new("L", image.size, 0)
            for x in range(8, 18):
                for y in range(6, 18):
                    mask.putpixel((x, y), 255)
            return mask, "facebook/detr-resnet-50-panoptic", ["chair"]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "panoptic_selection_mask", side_effect=fake_panoptic),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/mask",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "points_json": json.dumps([{"x": 0.38, "y": 0.5}]),
                        "model_id": "panoptic-detr",
                        "mask_max_dimension": "64",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                mask_response = client.get(payload["mask_url"])
                self.assertEqual(mask_response.status_code, 200)

        self.assertEqual(payload["model_status"], "panoptic-click-segment")
        self.assertEqual(payload["model_id"], "facebook/detr-resnet-50-panoptic")
        self.assertEqual(payload["selection_labels"], ["chair"])
        self.assertGreater(payload["mask_pixels"], 0)

    def test_selection_panoptic_precompute_reuses_cached_segmentation(self):
        def fake_compute_panoptic(image, device="auto"):
            segmentation = np.zeros((image.height, image.width), dtype=np.int32)
            segmentation[6:18, 8:18] = 7
            return segmentation, {0: "background", 7: "chair"}, "facebook/detr-resnet-50-panoptic"

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "compute_panoptic_segmentation", side_effect=fake_compute_panoptic) as compute_mock,
            ):
                main_module.SELECTION_PRECOMPUTE_CACHE.clear()
                client = TestClient(main_module.app)
                precompute_response = client.post(
                    "/selection/precompute",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={"model_id": "panoptic-detr"},
                )
                self.assertEqual(precompute_response.status_code, 200, precompute_response.text)
                precompute_payload = precompute_response.json()

                mask_response = client.post(
                    "/selection/precomputed_mask",
                    data={
                        "precompute_id": precompute_payload["precompute_id"],
                        "points_json": json.dumps([{"x": 0.38, "y": 0.5}]),
                    },
                )
                self.assertEqual(mask_response.status_code, 200, mask_response.text)
                payload = mask_response.json()
                for url_field in ("mask_url", "tint_url", "metadata_url"):
                    artifact_response = client.get(payload[url_field])
                    self.assertEqual(artifact_response.status_code, 200, url_field)

        self.assertEqual(compute_mock.call_count, 1)
        self.assertEqual(precompute_payload["model_status"], "panoptic-precomputed")
        self.assertEqual(precompute_payload["segment_count"], 2)
        self.assertEqual(payload["model_status"], "panoptic-precomputed-point")
        self.assertEqual(payload["model_id"], "facebook/detr-resnet-50-panoptic")
        self.assertEqual(payload["selection_labels"], ["chair"])
        self.assertEqual(payload["selected_segment_ids"], [7])
        self.assertGreater(payload["mask_pixels"], 0)
        main_module.SELECTION_PRECOMPUTE_CACHE.clear()

    def test_selection_precomputed_mask_rejects_missing_cache_id(self):
        main_module.SELECTION_PRECOMPUTE_CACHE.clear()
        client = TestClient(main_module.app)
        response = client.post(
            "/selection/precomputed_mask",
            data={
                "precompute_id": "missing",
                "points_json": json.dumps([{"x": 0.5, "y": 0.5}]),
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("precompute session", response.json()["detail"])

    def test_selection_compose_accepts_fetch_urls_and_writes_selected_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"),
                patch.object(main_module, "sam2_selection_mask", side_effect=RuntimeError("checkpoint not cached locally")),
                patch.object(main_module, "panoptic_selection_mask", side_effect=RuntimeError("panoptic checkpoint not cached locally")),
            ):
                client = TestClient(main_module.app)
                first = client.post(
                    "/selection/mask",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "points_json": json.dumps([{"x": 0.38, "y": 0.5}]),
                        "model_id": "sam2.1-hiera-large",
                        "mask_max_dimension": "64",
                    },
                )
                second = client.post(
                    "/selection/mask",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "points_json": json.dumps([{"x": 0.65, "y": 0.5}]),
                        "model_id": "sam2.1-hiera-large",
                        "mask_max_dimension": "64",
                    },
                )
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(second.status_code, 200, second.text)

                response = client.post(
                    "/selection/compose",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={
                        "mask_paths_json": json.dumps(
                            [
                                first.json()["mask"],
                                first.json()["tint_url"].replace("selection_tint.png", "selection_mask.png"),
                                f"http://testserver{second.json()['mask_url']}",
                            ]
                        ),
                        "background_mode": "black",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()

                for url_field in ("selected_image_url", "mask_url", "overlay_url", "tint_url", "metadata_url"):
                    artifact_response = client.get(payload[url_field])
                    self.assertEqual(artifact_response.status_code, 200, url_field)

        self.assertEqual(payload["model_status"], "composed-clicked-masks")
        self.assertEqual(payload["mask_count"], 3)
        self.assertEqual(payload["background_mode"], "black")
        self.assertGreater(payload["mask_pixels"], 0)

    def test_structural_context_research_helper_fills_only_small_holes(self):
        width, height = 96, 72
        source_values = np.zeros((height, width, 3), dtype=np.uint8)
        source_values[..., 0] = np.arange(width, dtype=np.uint8)[None, :] * 2
        source_values[..., 1] = np.arange(height, dtype=np.uint8)[:, None] * 3
        source_values[..., 2] = 80
        source_values[::2, ::2, 2] = 220
        source = Image.fromarray(source_values, mode="RGB")
        mask_values = np.zeros((height, width), dtype=np.uint8)
        mask_values[6:66, 8:88] = 255
        mask_values[28:30, 30:32] = 0
        mask_values[36:48, 54:66] = 0
        selected_image, final_mask_image, infill = main_module.compose_selected_image(
            source,
            Image.fromarray(mask_values, mode="L"),
            background_mode="neutral",
            selection_infill_mode="structural-context",
        )
        selected_values = np.asarray(selected_image.convert("RGB"))
        final_mask = np.asarray(final_mask_image.convert("L")) > 0

        self.assertEqual(infill["mode"], "structural-context")
        self.assertTrue(infill["enabled"])
        self.assertEqual(infill["filled_hole_pixels"], 4)
        self.assertTrue(infill["selected_core_exact"])
        self.assertTrue(infill["selected_pixels_exact"])
        self.assertTrue(final_mask[28:30, 30:32].all())
        self.assertFalse(final_mask[36:48, 54:66].any())
        np.testing.assert_array_equal(selected_values[final_mask], source_values[final_mask])
        self.assertFalse(np.array_equal(selected_values[0, 0], np.array([245, 245, 245])))

    def test_clean_context_research_helper_discards_removed_source_detail(self):
        width, height = 96, 72
        rows, cols = np.indices((height, width))
        source_values = np.zeros((height, width, 3), dtype=np.uint8)
        source_values[..., 0] = np.where((rows + cols) % 2 == 0, 250, 5)
        source_values[..., 1] = np.where(cols % 3 == 0, 240, 15)
        source_values[..., 2] = np.where(rows % 3 == 0, 230, 25)
        source_values[28:30, 30:32] = np.array([0, 255, 0], dtype=np.uint8)
        source = Image.fromarray(source_values, mode="RGB")
        mask_values = np.zeros((height, width), dtype=np.uint8)
        mask_values[6:66, 8:88] = 255
        mask_values[28:30, 30:32] = 0
        mask_values[36:48, 54:66] = 0
        original_selected = mask_values > 0

        selected_image, final_mask_image, infill = main_module.compose_selected_image(
            source,
            Image.fromarray(mask_values, mode="L"),
            background_mode="neutral",
            selection_infill_mode="clean-context",
        )
        selected_values = np.asarray(selected_image.convert("RGB"))
        final_mask = np.asarray(final_mask_image.convert("L")) > 0

        self.assertEqual(infill["mode"], "clean-context")
        self.assertTrue(infill["source_free"])
        self.assertFalse(infill["excluded_source_pixels_used"])
        self.assertEqual(infill["filled_hole_pixels"], 4)
        self.assertEqual(infill["generated_hole_pixels"], 4)
        self.assertEqual(infill["filled_hole_source_pixels_preserved"], 0)
        self.assertTrue(infill["selected_pixels_exact"])
        self.assertTrue(final_mask[28:30, 30:32].all())
        self.assertFalse(final_mask[36:48, 54:66].any())
        np.testing.assert_array_equal(
            selected_values[original_selected],
            source_values[original_selected],
        )
        self.assertFalse(
            np.array_equal(
                selected_values[28:30, 30:32],
                source_values[28:30, 30:32],
            )
        )
        self.assertFalse(np.array_equal(selected_values[0, 0], source_values[0, 0]))

    def test_selection_compose_rejects_removed_and_unknown_infill_modes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            mask_path = output_dir / "selection" / "fixture" / "selection_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (32, 24), 255).save(mask_path)
            with patch.object(main_module, "OUTPUT_DIR", output_dir):
                client = TestClient(main_module.app)
                for mode in (
                    "structural-context",
                    "clean-context",
                    "generative-context",
                    "invented",
                ):
                    with self.subTest(mode=mode):
                        response = client.post(
                            "/selection/compose",
                            files={
                                "file": (
                                    "object.png",
                                    self.png_bytes(),
                                    "image/png",
                                )
                            },
                            data={
                                "mask_paths_json": json.dumps(
                                    ["selection/fixture/selection_mask.png"]
                                ),
                                "selection_infill_mode": mode,
                            },
                        )
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(
                            "Selection infill is not available",
                            response.json()["detail"],
                        )

    def test_selection_compose_rejects_empty_mask_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main_module, "OUTPUT_DIR", Path(temp_dir) / "output"):
                client = TestClient(main_module.app)
                response = client.post(
                    "/selection/compose",
                    files={"file": ("object.png", self.png_bytes(), "image/png")},
                    data={"mask_paths_json": "[]"},
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Select at least one object", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
