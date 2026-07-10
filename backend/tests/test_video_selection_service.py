import unittest

from fastapi.testclient import TestClient

from backend.video_selection_service import DEFAULTS, app


class VideoSelectionServiceTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_model_catalog_exposes_stl_first_groups(self):
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "planner-only")
        self.assertEqual(data["defaults"]["image_to_mesh"], DEFAULTS["image_to_mesh"])
        self.assertIn("selection", data["groups"])
        self.assertIn("video_reconstruction", data["groups"])
        self.assertIn("stl_postprocess", data["groups"])
        self.assertIn("watertightness", data["metrics"])

    def test_photo_plan_routes_to_image_mesh_and_stl_repair(self):
        response = self.client.post(
            "/plan",
            json={
                "input": {"media_type": "photo", "file_name": "chair.png"},
                "models": {"image_to_mesh": "triposg", "stl_postprocess": "trimesh-repair"},
                "print_volume": {"target_dimension_mm": 120},
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "planned")
        self.assertEqual([stage["id"] for stage in data["stages"]], ["object-selection", "image-to-mesh", "stl-postprocess"])
        self.assertEqual(data["stages"][1]["model"]["id"], "triposg")
        self.assertIn("watertight STL", data["next_backend_contract"]["output"])

    def test_video_plan_preserves_camera_and_reconstruction_stages(self):
        response = self.client.post(
            "/plan",
            json={
                "input": {"media_type": "video", "file_name": "turntable.mp4"},
                "models": {
                    "frame_selection": "uniform-frame-sampler",
                    "selection": "sam2.1-hiera-large",
                    "camera_pose": "hloc-lightglue",
                    "video_reconstruction": "vggt-fusion",
                    "stl_postprocess": "trimesh-repair",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(
            [stage["id"] for stage in data["stages"]],
            ["frame-selection", "object-selection", "camera-pose", "video-reconstruction", "stl-postprocess"],
        )
        self.assertEqual(data["stages"][3]["model"]["id"], "vggt-fusion")

    def test_invalid_model_id_returns_valid_options(self):
        response = self.client.post(
            "/plan",
            json={
                "input": {"media_type": "video"},
                "models": {"camera_pose": "not-a-camera-model"},
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported camera_pose model", response.json()["detail"])
        self.assertIn("hloc-lightglue", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
