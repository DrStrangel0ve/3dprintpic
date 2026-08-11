from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_sam3_selection_fails_closed_without_owner_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "photo.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "read-only HF_TOKEN"):
                    space_runtime.select_object(image_path, 10, 10)

    def test_relief_route_uses_original_pixels_and_subject_lock(self):
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
            selection = {"job_id": "b" * 32}
            payload = {
                "job_id": job_id,
                "stl_model": f"{job_id}/output_model.stl",
                "diagnostics": f"{job_id}/diagnostics.json",
                "stl_diagnostics": {"is_watertight": True, "component_count": 1},
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
                    128,
                    30,
                    320,
                    0.65,
                )
            request_data = post.call_args.kwargs["data"]
            self.assertEqual(request_data["completion_mode"], "none")
            self.assertEqual(request_data["selection_subject_lock"], "true")
            self.assertEqual(request_data["selection_job_id"], selection["job_id"])
            self.assertEqual(request_data["depth_model"], "/models/depth-v2")
            self.assertEqual(result[3]["summary"]["dimensions_mm"], {"x": 128.0, "y": 64.0, "z": 30.0})
            self.assertFalse(result[3]["summary"]["inpainting"])

    def test_model_and_source_revisions_are_immutable(self):
        self.assertEqual(
            space_runtime.PROJECT_REVISION,
            "3808d76bf02d31b9a6724e825bfd325bf6c3d412",
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
