import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import backend.main as main_module


class MainStlContractTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
