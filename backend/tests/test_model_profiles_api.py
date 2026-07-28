import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import trimesh

from backend.model_profiles import DEFAULT_MODEL_PROFILE_ID
import backend.video_selection_service as service_module


class ModelProfilesApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service_module.app)

    def test_profile_catalog_separates_recommended_and_candidate_models(self):
        response = self.client.get("/profiles")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["default_profile"], DEFAULT_MODEL_PROFILE_ID)
        profiles = {profile["id"]: profile for profile in payload["profiles"]}
        selected = profiles[DEFAULT_MODEL_PROFILE_ID]
        self.assertEqual(selected["status"], "recommended")
        self.assertEqual(selected["models"]["image_to_mesh"], "triposg")
        features = selected["production_features"]
        self.assertEqual(
            features["photo_relief_depth"]["runtime_id"],
            "depth-anything/Depth-Anything-V2-Large-hf",
        )
        self.assertEqual(features["single_image_mesh"]["runtime_id"], "triposg")
        self.assertEqual(
            features["turntable_reconstruction"]["runtime_id"],
            "multiview-visual-hull",
        )
        self.assertTrue(
            all(feature["reason"] and feature["evidence"] for feature in features.values())
        )
        self.assertEqual(
            selected["routes"]["photo-full-mesh"]["settings"][
                "mesh_max_normalized_face_density_log1p"
            ],
            9.95,
        )
        full_mesh = selected["routes"]["photo-full-mesh"]["settings"]
        self.assertEqual(full_mesh["mesh_repair_preconditioner"], "adaptive-voxel-close")
        self.assertEqual(full_mesh["mesh_repair_voxel_resolution"], 128)
        self.assertEqual(full_mesh["mesh_repair_smoothing_iterations"], 2)
        self.assertFalse(full_mesh["mesh_allow_convex_hull_fallback"])
        self.assertEqual(full_mesh["mesh_target_bbox_mode"], "uniform-max")
        candidates = {candidate["id"]: candidate for candidate in payload["candidates"]}
        self.assertEqual(candidates["pixal3d-detail"]["status"], "provisional")
        self.assertEqual(candidates["trellis2-detail"]["status"], "unbenchmarked")
        self.assertEqual(candidates["hunyuan3d-shape"]["status"], "held")

    def test_plan_uses_profile_models_when_client_omits_manual_choices(self):
        response = self.client.post("/plan", json={"input": {"media_type": "photo"}})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_profile"]["id"], DEFAULT_MODEL_PROFILE_ID)
        stages = {stage["id"]: stage["model"]["id"] for stage in payload["stages"]}
        self.assertEqual(stages["object-selection"], "detr-resnet-50-panoptic")
        self.assertEqual(stages["image-to-mesh"], "triposg")
        self.assertEqual(stages["stl-postprocess"], "trimesh-repair")

    def test_image_runner_resolves_measured_profile_defaults(self):
        observed = {}

        def fake_provider(args):
            observed.update(vars(args))
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(args.output_mesh)
            mesh.export(args.output_stl)
            return args.output_mesh, args.output_stl

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            service_module, "OUTPUT_DIR", Path(temp_dir)
        ), patch.object(service_module, "run_provider_job", side_effect=fake_provider):
            response = self.client.post(
                "/run/image-to-mesh",
                files={"file": ("object.png", b"fake-image", "image/png")},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["model_profile"], DEFAULT_MODEL_PROFILE_ID)
        self.assertEqual(payload["provider_metrics"], {})
        self.assertTrue(payload["provider_metrics_url"].endswith("provider_metrics.json"))
        self.assertEqual(observed["provider"], "triposg")
        self.assertEqual(observed["num_inference_steps"], 50)
        self.assertEqual(observed["guidance_scale"], 7.0)
        self.assertEqual(observed["mesh_repair_preconditioner"], "adaptive-voxel-close")
        self.assertEqual(observed["mesh_repair_voxel_resolution"], 128)
        self.assertEqual(observed["mesh_repair_voxel_fill_method"], "orthographic")
        self.assertEqual(observed["mesh_repair_smoothing_iterations"], 2)
        self.assertFalse(observed["mesh_allow_convex_hull_fallback"])
        self.assertEqual(observed["mesh_target_bbox_mode"], "uniform-max")
        self.assertEqual(observed["mesh_min_bbox_dimension"], 0.0)
        self.assertEqual(observed["mesh_max_bbox_aspect_ratio"], 0.0)
        self.assertEqual(observed["mesh_target_faces"], 10000)
        self.assertEqual(observed["mesh_max_normalized_face_density_log1p"], 9.95)
        self.assertEqual(
            observed["triposg_model_revision"],
            "2c1c516d22d58db486a058d98d31bb6177344e06",
        )
        self.assertEqual(
            observed["triposg_rembg_revision"],
            "2ceba5a5efaec153162aedea169f76caf9b46cf8",
        )

    def test_explicit_runner_values_override_profile_defaults(self):
        observed = {}

        def fake_provider(args):
            observed.update(vars(args))
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(args.output_mesh)
            mesh.export(args.output_stl)
            return args.output_mesh, args.output_stl

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            service_module, "OUTPUT_DIR", Path(temp_dir)
        ), patch.object(service_module, "run_provider_job", side_effect=fake_provider):
            response = self.client.post(
                "/run/image-to-mesh",
                files={"file": ("object.png", b"fake-image", "image/png")},
                data={
                    "mesh_repair_preconditioner": "legacy",
                    "mesh_repair_voxel_resolution": "48",
                    "mesh_repair_voxel_fill_method": "holes",
                    "mesh_repair_smoothing_iterations": "0",
                    "mesh_allow_convex_hull_fallback": "true",
                    "mesh_target_bbox_mode": "exact",
                    "mesh_target_faces": "1200",
                    "mesh_max_normalized_face_density_log1p": "8.4",
                    "num_inference_steps": "32",
                    "guidance_scale": "5.5",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(observed["mesh_repair_preconditioner"], "legacy")
        self.assertEqual(observed["mesh_repair_voxel_resolution"], 48)
        self.assertEqual(observed["mesh_repair_voxel_fill_method"], "holes")
        self.assertEqual(observed["mesh_repair_smoothing_iterations"], 0)
        self.assertTrue(observed["mesh_allow_convex_hull_fallback"])
        self.assertEqual(observed["mesh_target_bbox_mode"], "exact")
        self.assertEqual(observed["mesh_target_faces"], 1200)
        self.assertEqual(observed["mesh_max_normalized_face_density_log1p"], 8.4)
        self.assertEqual(observed["num_inference_steps"], 32)
        self.assertEqual(observed["guidance_scale"], 5.5)

    def test_unknown_profile_is_rejected_with_valid_profile_hint(self):
        response = self.client.post(
            "/run/image-to-mesh",
            files={"file": ("object.png", b"fake-image", "image/png")},
            data={"profile_id": "not-a-profile"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(DEFAULT_MODEL_PROFILE_ID, response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
