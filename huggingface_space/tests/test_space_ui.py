from __future__ import annotations

import os
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

    def test_zero_gpu_runtime_uses_writable_xet_cache_and_free_tier_mesh_window(self):
        self.assertEqual(os.environ["HF_XET_CACHE"], str(app.HF_XET_CACHE_DIR))
        self.assertTrue(app.HF_XET_CACHE_DIR.is_dir())
        self.assertEqual(app.FULL_MESH_GPU_DURATION_SECONDS, 150)

    def test_gpu_events_share_one_serial_queue(self):
        gpu_functions = [
            function
            for function in app.demo.fns.values()
            if getattr(function.fn, "__name__", "")
            in {
                "_prepare_hover_selection",
                "_generate_relief_ui",
                "_generate_diorama_ui",
                "_generate_full_mesh_ui",
            }
        ]
        self.assertEqual(len(gpu_functions), 6)
        self.assertTrue(all(function.concurrency_id == "gpu-work" for function in gpu_functions))
        self.assertTrue(all(function.concurrency_limit == 1 for function in gpu_functions))

    def test_hover_selection_is_client_side_after_one_gpu_precompute(self):
        function_names = [
            getattr(function.fn, "__name__", "")
            for function in app.demo.fns.values()
        ]
        self.assertEqual(function_names.count("_prepare_hover_selection"), 3)
        self.assertEqual(function_names.count("_select_from_hover_event"), 3)
        self.assertNotIn("_select_from_click", function_names)
        prepare_functions = [
            function
            for function in app.demo.fns.values()
            if getattr(function.fn, "__name__", "") == "_prepare_hover_selection"
        ]
        self.assertTrue(all(function.trigger_mode == "always_last" for function in prepare_functions))
        cancellation_dependencies = [
            dependency
            for dependency in app.demo.get_config_file().get("dependencies", [])
            if dependency.get("cancels")
        ]
        self.assertEqual(len(cancellation_dependencies), 6)
        self.assertIn('root.addEventListener("pointermove"', app.SELECTION_HOVER_JS)
        self.assertIn("regionAt(current", app.SELECTION_HOVER_JS)
        self.assertIn('fit === "scale-down"', app.SELECTION_HOVER_JS)
        self.assertIn('event.target.closest("button, input', app.SELECTION_HOVER_JS)
        self.assertNotIn("fetch(", app.SELECTION_HOVER_JS)


if __name__ == "__main__":
    unittest.main()
