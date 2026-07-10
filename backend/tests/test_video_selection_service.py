import unittest
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
import trimesh

import backend.video_selection_service as service_module
from backend.video_selection_service import DEFAULTS, app


class VideoSelectionServiceTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_model_catalog_exposes_stl_first_groups(self):
        response = self.client.get("/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "planner-plus-runner")
        self.assertIn("image-to-mesh", data["runner_modes"])
        self.assertIn("multiview-to-mesh", data["runner_modes"])
        self.assertEqual(data["defaults"]["image_to_mesh"], DEFAULTS["image_to_mesh"])
        self.assertEqual(data["defaults"]["video_reconstruction"], "multiview-visual-hull")
        self.assertIn("selection", data["groups"])
        self.assertIn("panoptic-detr", {model["id"] for model in data["groups"]["selection"]})
        self.assertIn("video_reconstruction", data["groups"])
        self.assertIn("multiview-visual-hull", {model["id"] for model in data["groups"]["video_reconstruction"]})
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

    def test_image_to_mesh_provider_preflight_reports_missing_setup(self):
        env_overrides = {
            "TRIPOSG_DIR": "",
            "TRIPOSR_DIR": "",
            "HUNYUAN3D_DIR": "",
            "SPAR3D_DIR": "",
            "SF3D_DIR": "",
            "IMAGE_TO_MESH_PROVIDER_DIR": "",
            "IMAGE_TO_MESH_PROVIDER_PYTHON": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            response = self.client.get("/providers/image-to-mesh")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["runner"], "image-to-mesh")
        providers = {provider["id"]: provider for provider in data["providers"]}
        self.assertIn(DEFAULTS["image_to_mesh"], providers)
        self.assertNotIn("multiview-visual-hull", providers)
        triposg = providers["triposg"]
        self.assertFalse(triposg["runnable"])
        self.assertEqual(triposg["status"], "missing")
        self.assertTrue(triposg["checks"]["provider_python_found"])
        self.assertIn("TRIPOSG_DIR", triposg["env"]["provider_dir_env_names"])
        self.assertIn("Provider repo is missing", triposg["setup_errors"][0])

    def test_image_to_mesh_provider_preflight_accepts_configured_provider_repo(self):
        with TemporaryDirectory() as provider_dir, patch.dict(os.environ, {"TRIPOSG_DIR": provider_dir}, clear=False):
            scripts_dir = Path(provider_dir) / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "inference_triposg.py").write_text("# smoke entrypoint\n", encoding="utf-8")

            response = self.client.get("/providers/image-to-mesh")

        self.assertEqual(response.status_code, 200)
        providers = {provider["id"]: provider for provider in response.json()["providers"]}
        triposg = providers["triposg"]
        self.assertTrue(triposg["runnable"])
        self.assertEqual(triposg["status"], "available")
        self.assertTrue(triposg["checks"]["provider_dir_resolved"])
        self.assertEqual(triposg["checks"]["entrypoint"], "scripts/inference_triposg.py")
        self.assertTrue(triposg["checks"]["entrypoint_found"])
        self.assertTrue(triposg["env"]["provider_dir_configured"])

    def test_multiview_to_mesh_provider_preflight_exposes_builtin_visual_hull(self):
        response = self.client.get("/providers/multiview-to-mesh")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["runner"], "multiview-to-mesh")
        self.assertEqual(data["default_provider"], "multiview-visual-hull")
        providers = {provider["id"]: provider for provider in data["providers"]}
        visual_hull = providers["multiview-visual-hull"]
        self.assertTrue(visual_hull["runnable"])
        self.assertEqual(visual_hull["status"], "available")
        self.assertTrue(visual_hull["checks"]["builtin_provider"])
        self.assertTrue(visual_hull["checks"]["requires_input_bundle"])

    def test_image_to_mesh_runner_emits_stl_and_diagnostics(self):
        observed_args = {}

        def fake_provider(args):
            observed_args["provider"] = args.provider
            observed_args["provider_dir"] = args.provider_dir
            observed_args["python"] = args.python
            observed_args["timeout"] = args.timeout
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(args.output_mesh)
            mesh.export(args.output_stl)
            return args.output_mesh, args.output_stl

        with patch.dict(
            os.environ,
            {
                "IMAGE_TO_MESH_PROVIDER_DIR": "",
                "IMAGE_TO_MESH_PROVIDER_PYTHON": "",
                "IMAGE_TO_MESH_TIMEOUT_SECONDS": "",
            },
            clear=False,
        ), patch.object(service_module, "run_provider_job", side_effect=fake_provider):
            response = self.client.post(
                "/run/image-to-mesh",
                files={"file": ("object.png", b"fake-image-bytes", "image/png")},
                data={
                    "provider": "triposr",
                    "provider_dir": "C:/should/not/be/used",
                    "provider_python": "C:/should/not/run/python.exe",
                    "timeout": "999999",
                    "provider_device": "cpu",
                    "mesh_repair": "printable",
                    "mesh_target_max_dimension": "96",
                    "mesh_min_bbox_dimension": "12",
                    "mesh_max_bbox_aspect_ratio": "2.25",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "printable")
        self.assertEqual(data["runner"], "image-to-mesh")
        self.assertEqual(data["provider"], "triposr")
        self.assertEqual(observed_args["provider"], "triposr")
        self.assertIsNone(observed_args["provider_dir"])
        self.assertEqual(observed_args["python"], sys.executable)
        self.assertEqual(observed_args["timeout"], 3600)
        self.assertTrue(data["stl_model"].endswith("/output_model.stl"))
        self.assertTrue(data["diagnostics"].endswith("/diagnostics.json"))
        self.assertEqual(data["stl_diagnostics"]["artifact_contract"], "output_model.stl + diagnostics.json")
        self.assertTrue(data["stl_diagnostics"]["stl_is_watertight"])
        self.assertTrue(data["stl_diagnostics"]["stl_is_volume"])
        self.assertTrue(data["stl_passes_hard_checks"])
        self.assertEqual(data["stl_failed_checks"], [])
        self.assertIn("provider_seconds", data["timings"])
        self.assertIn("diagnostics_seconds", data["timings"])

        diagnostics_response = self.client.get(data["diagnostics_url"])
        self.assertEqual(diagnostics_response.status_code, 200)
        self.assertEqual(diagnostics_response.json()["job_id"], data["job_id"])

    def test_image_to_mesh_runner_reports_emitted_stl_when_hard_checks_fail(self):
        def fake_provider(args):
            mesh = trimesh.Trimesh(
                vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                faces=((0, 1, 2),),
                process=False,
            )
            args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(args.output_mesh)
            mesh.export(args.output_stl)
            return args.output_mesh, args.output_stl

        with patch.object(service_module, "run_provider_job", side_effect=fake_provider):
            response = self.client.post(
                "/run/image-to-mesh",
                files={"file": ("object.png", b"fake-image-bytes", "image/png")},
                data={"provider": "triposg", "provider_device": "cpu"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "stl-emitted")
        self.assertFalse(data["stl_passes_hard_checks"])
        self.assertIn("stl_is_watertight", data["stl_failed_checks"])
        self.assertIn("stl_is_volume", data["stl_failed_checks"])

    def test_image_to_mesh_runner_rejects_unknown_provider(self):
        response = self.client.post(
            "/run/image-to-mesh",
            files={"file": ("object.png", b"fake-image-bytes", "image/png")},
            data={"provider": "not-real"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported image_to_mesh provider", response.json()["detail"])

    def test_multiview_to_mesh_runner_emits_stl_and_diagnostics(self):
        observed = {}

        def fake_provider(args):
            observed["provider"] = args.provider
            observed["input_bundle"] = args.input_bundle
            observed["visual_hull_resolution"] = args.visual_hull_resolution
            observed["visual_hull_mask_dilate"] = args.visual_hull_mask_dilate
            bundle = json.loads(Path(args.input_bundle).read_text(encoding="utf-8"))
            observed["bundle"] = bundle
            for view in bundle["views"]:
                self.assertTrue(Path(view["image"]).exists())
                if view.get("mask"):
                    self.assertTrue(Path(view["mask"]).exists())
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(args.output_mesh)
            mesh.export(args.output_stl)
            return args.output_mesh, args.output_stl

        bundle = {
            "sample_id": "turntable_box",
            "views": [
                {"image": "front.png", "mask": "front_mask.png", "camera": {"azimuth_deg": 0}},
                {"image": "side.png", "mask": "side_mask.png", "camera": {"azimuth_deg": 90}},
            ],
        }
        files = [
            ("primary_file", ("front.png", b"primary-bytes", "image/png")),
            ("view_files", ("front.png", b"front-bytes", "image/png")),
            ("view_files", ("side.png", b"side-bytes", "image/png")),
            ("mask_files", ("front_mask.png", b"front-mask-bytes", "image/png")),
            ("mask_files", ("side_mask.png", b"side-mask-bytes", "image/png")),
        ]
        with patch.object(service_module, "run_provider_job", side_effect=fake_provider):
            response = self.client.post(
                "/run/multiview-to-mesh",
                files=files,
                data={
                    "provider": "multiview-visual-hull",
                    "provider_device": "cpu",
                    "bundle_json": json.dumps(bundle),
                    "mesh_repair": "printable",
                    "mesh_target_max_dimension": "96",
                    "mesh_min_bbox_dimension": "12",
                    "visual_hull_resolution": "24",
                    "visual_hull_mask_dilate": "0",
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "printable")
        self.assertEqual(data["runner"], "multiview-to-mesh")
        self.assertEqual(data["provider"], "multiview-visual-hull")
        self.assertEqual(data["view_count"], 2)
        self.assertEqual(observed["provider"], "multiview-visual-hull")
        self.assertEqual(observed["visual_hull_resolution"], 24)
        self.assertEqual(observed["visual_hull_mask_dilate"], 0)
        self.assertEqual(observed["bundle"]["views"][1]["camera"]["azimuth_deg"], 90)
        self.assertTrue(data["input_bundle"].endswith("/multiview_input.json"))
        self.assertTrue(data["stl_model"].endswith("/output_model.stl"))
        self.assertEqual(data["stl_diagnostics"]["artifact_contract"], "output_model.stl + diagnostics.json")
        self.assertEqual(data["stl_diagnostics"]["runner"], "multiview-to-mesh")
        self.assertTrue(data["stl_passes_hard_checks"])
        self.assertEqual(data["stl_failed_checks"], [])

        bundle_response = self.client.get(data["input_bundle_url"])
        self.assertEqual(bundle_response.status_code, 200)
        self.assertEqual(bundle_response.json()["sample_id"], "turntable_box")
        diagnostics_response = self.client.get(data["diagnostics_url"])
        self.assertEqual(diagnostics_response.status_code, 200)
        self.assertEqual(diagnostics_response.json()["job_id"], data["job_id"])

    def test_multiview_to_mesh_runner_rejects_planner_only_provider(self):
        response = self.client.post(
            "/run/multiview-to-mesh",
            files={"primary_file": ("front.png", b"primary-bytes", "image/png")},
            data={"provider": "vggt-fusion"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported multiview_to_mesh provider", response.json()["detail"])

    def test_multiview_to_mesh_runner_rejects_unuploaded_bundle_path(self):
        with TemporaryDirectory() as temp_dir:
            external_path = Path(temp_dir) / "external.png"
            external_path.write_bytes(b"external-bytes")
            response = self.client.post(
                "/run/multiview-to-mesh",
                files={"primary_file": ("front.png", b"primary-bytes", "image/png")},
                data={
                    "provider": "multiview-visual-hull",
                    "bundle_json": json.dumps(
                        {
                            "views": [
                                {
                                    "image": str(external_path),
                                    "camera": {"azimuth_deg": 0},
                                }
                            ]
                        }
                    ),
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not an uploaded or output artifact", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
