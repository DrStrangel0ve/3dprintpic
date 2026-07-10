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
    def png_bytes(self, color=(245, 245, 245), accent=(20, 120, 220)) -> bytes:
        image = Image.new("RGB", (32, 24), color)
        for x in range(8, 18):
            for y in range(6, 18):
                image.putpixel((x, y), accent)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

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

    def test_process_image_emits_output_model_and_diagnostics_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                depth_path = Path(output_dir) / "output_depth_data.npy"
                np.save(depth_path, np.array([[0.1, 0.3], [0.2, 0.6]], dtype=np.float32))
                return str(depth_path)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete_image),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/process_image",
                    files={"file": ("relief.png", b"fake-image-bytes", "image/png")},
                    data={
                        "target_dimension": "-1",
                        "z_scale": "10",
                        "invert": "false",
                        "sigma": "0",
                        "base_border_px": "0",
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

                diagnostics_response = client.get(payload["diagnostics_url"])
                self.assertEqual(diagnostics_response.status_code, 200)
                diagnostics = diagnostics_response.json()
                self.assertEqual(diagnostics["job_id"], payload["job_id"])
                self.assertTrue(diagnostics["stl_positive_volume"])

    def test_process_image_reports_depth_fallback_and_uses_effective_polarity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"

            def fake_complete_image(input_path, **_kwargs):
                return input_path, None

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                output_path = Path(output_dir)
                depth_path = output_path / "output_depth_data.npy"
                np.save(depth_path, np.array([[0.1, 0.3], [0.2, 0.6]], dtype=np.float32))
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
                    files={"file": ("relief.png", b"fake-image-bytes", "image/png")},
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
