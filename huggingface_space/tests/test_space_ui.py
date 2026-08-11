from __future__ import annotations

import unittest

from huggingface_space import app


class SpaceUiTests(unittest.TestCase):
    def test_ui_exposes_all_three_production_outputs(self):
        config = app.demo.get_config_file()
        labels = {
            component.get("props", {}).get("label")
            for component in config.get("components", [])
        }
        self.assertIn("Printable STL", labels)
        self.assertIn("Colored scene", labels)
        self.assertIn("Printable full mesh", labels)
        self.assertIn("Download GLB", labels)

    def test_ui_has_one_model_per_learned_feature(self):
        self.assertIn("depth-anything/Depth-Anything-V2-Large-hf", app.DEPTH_MODEL)
        self.assertEqual(app.SAM3_MODEL, "facebook/sam3")


if __name__ == "__main__":
    unittest.main()
