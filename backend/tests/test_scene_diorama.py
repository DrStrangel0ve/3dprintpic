import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import backend.main as main_module
from backend.scene_diorama import (
    _facade_detail_field,
    classify_scene_layer,
    normalize_scene_depth,
)


class SceneDioramaTest(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[bytes, Image.Image, Image.Image, np.ndarray]:
        image = Image.new("RGB", (80, 60), (185, 215, 236))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 6, 76, 59), fill=(125, 132, 140))
        for x in range(35, 72, 10):
            for y in range(12, 52, 12):
                draw.rectangle((x, y, x + 5, y + 6), fill=(45, 65, 85))
        draw.ellipse((8, 17, 26, 35), fill=(195, 55, 45))
        draw.rectangle((10, 31, 27, 59), fill=(190, 55, 45))

        person = Image.new("L", image.size, 0)
        person_draw = ImageDraw.Draw(person)
        person_draw.ellipse((8, 17, 26, 35), fill=255)
        person_draw.rectangle((10, 31, 27, 59), fill=255)

        building = Image.new("L", image.size, 0)
        ImageDraw.Draw(building).rectangle((30, 6, 76, 59), fill=255)

        depth = np.full((60, 80), 0.95, dtype=np.float32)
        depth[6:60, 30:77] = 0.8
        depth[17:60, 8:28] = 0.2
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), person, building, depth

    def test_depth_normalization_respects_model_polarity(self):
        close_high = np.array([[1.0, 0.0]], dtype=np.float32)
        normalized = normalize_scene_depth(close_high, far_is_high=False)

        self.assertLess(float(normalized[0, 0]), float(normalized[0, 1]))
        self.assertEqual(classify_scene_layer(np.ones((4, 4), dtype=bool), ["person"]), "subject")
        self.assertEqual(classify_scene_layer(np.ones((4, 4), dtype=bool), ["building-other"]), "facade")

    def test_generated_facade_fill_does_not_reuse_foreground_texture(self):
        photo_detail = np.arange(25, dtype=np.float32).reshape(5, 5)
        visible_facade = np.zeros((5, 5), dtype=bool)
        visible_facade[1, 1:4] = True

        detail = _facade_detail_field(photo_detail, visible_facade)

        np.testing.assert_array_equal(detail[visible_facade], photo_detail[visible_facade])
        self.assertTrue(np.all(detail[~visible_facade] == 0.0))

    def test_scene_endpoint_keeps_people_and_building_as_ordered_layers(self):
        image_bytes, person, building, depth = self.fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            selection_dir = output_root / "selection"
            selection_dir.mkdir(parents=True)
            person.save(selection_dir / "person.png")
            building.save(selection_dir / "building.png")

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                output_path = Path(output_dir)
                depth_path = output_path / "output_depth_data.npy"
                np.save(depth_path, depth)
                (output_path / "output_depth_metadata.json").write_text(
                    json.dumps(
                        {
                            "effective_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                            "stored_depth_normalized": True,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(depth_path)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
            ):
                client = TestClient(main_module.app)
                response = client.post(
                    "/process_scene_diorama",
                    files={"file": ("selfie.png", image_bytes, "image/png")},
                    data={
                        "mask_paths_json": json.dumps(
                            ["selection/person.png", "selection/building.png"]
                        ),
                        "selection_labels_json": json.dumps([["person"], ["building"]]),
                        "depth_model": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                        "max_size_mm": "70",
                        "scene_depth_mm": "28",
                        "base_thickness_mm": "2.4",
                        "facade_detail_mm": "0.6",
                        "subject_depth_mm": "9",
                        "minimum_feature_mm": "0.8",
                        "max_samples": "64",
                    },
                )

                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                roles = {layer["role"]: layer for layer in payload["scene_reconstruction"]["layers"]}
                self.assertIn("subject", roles)
                self.assertIn("facade", roles)
                self.assertLess(roles["subject"]["scene_y_mm"], roles["facade"]["scene_y_mm"])
                self.assertTrue(roles["facade"]["generated_occlusion_fill"])
                self.assertTrue(payload["scene_reconstruction"]["camera_free_view"])
                self.assertEqual(payload["scene_reconstruction"]["novel_view_provider"], "none")
                self.assertEqual(payload["stl_diagnostics"]["runner"], "single-photo-scene-diorama")
                self.assertTrue(payload["stl_diagnostics"]["stl_is_watertight"])
                self.assertTrue(payload["stl_diagnostics"]["stl_is_volume"])
                self.assertTrue(payload["stl_diagnostics"]["stl_single_component"])

                stl_path = output_root / payload["stl_model"]
                mesh = trimesh.load_mesh(stl_path, force="mesh")
                self.assertTrue(mesh.is_watertight)
                self.assertGreater(float(mesh.extents[1]), 20.0)
                self.assertLessEqual(float(mesh.extents[0]), 70.0 + 1e-5)
                self.assertLessEqual(float(mesh.extents[1]), 28.0 + 1e-5)
                self.assertGreaterEqual(float(mesh.bounds[0, 1]), -1e-6)
                self.assertGreaterEqual(float(mesh.bounds[0, 2]), -1e-6)
                self.assertGreater((output_root / payload["scene_model"]).stat().st_size, 1000)

                scene_response = client.get(payload["scene_url"])
                self.assertEqual(scene_response.status_code, 200)
                self.assertEqual(scene_response.headers["content-type"], "model/gltf-binary")
                self.assertEqual(client.get(payload["preview_url"]).status_code, 200)
                self.assertEqual(client.get(payload["diagnostics_url"]).status_code, 200)

    def test_scene_endpoint_rejects_unpaired_selection_labels(self):
        image_bytes, person, _building, _depth = self.fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            output_root.mkdir(parents=True)
            person.save(output_root / "person.png")
            with patch.object(main_module, "OUTPUT_DIR", output_root):
                response = TestClient(main_module.app).post(
                    "/process_scene_diorama",
                    files={"file": ("selfie.png", image_bytes, "image/png")},
                    data={
                        "mask_paths_json": json.dumps(["person.png"]),
                        "selection_labels_json": json.dumps([["person"], ["building"]]),
                    },
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn("same length", response.json()["detail"])

    def test_scene_endpoint_removes_tiny_mask_islands_before_support_generation(self):
        image_bytes, person, _building, depth = self.fixture()
        person_draw = ImageDraw.Draw(person)
        person_draw.rectangle((2, 2, 3, 3), fill=255)
        person_draw.rectangle((5, 8, 6, 9), fill=255)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"
            output_root.mkdir(parents=True)
            person.save(output_root / "person.png")

            def fake_depth_data(_image_path, output_dir, **_kwargs):
                output_path = Path(output_dir)
                depth_path = output_path / "output_depth_data.npy"
                np.save(depth_path, depth)
                return str(depth_path)

            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth_data),
            ):
                response = TestClient(main_module.app).post(
                    "/process_scene_diorama",
                    files={"file": ("portrait.png", image_bytes, "image/png")},
                    data={
                        "mask_paths_json": json.dumps(["person.png"]),
                        "selection_labels_json": json.dumps([["person"]]),
                        "max_size_mm": "70",
                        "scene_depth_mm": "28",
                        "max_samples": "64",
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        layer = response.json()["scene_reconstruction"]["layers"][0]
        self.assertGreaterEqual(layer["mask_components"]["removed_components"], 2)
        self.assertEqual(layer["mask_components"]["kept_components"], 1)
        self.assertEqual(layer["generated_supports"], 0)


if __name__ == "__main__":
    unittest.main()
