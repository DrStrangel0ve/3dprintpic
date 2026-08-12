from __future__ import annotations

import os
import unittest
from unittest.mock import patch

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

    def test_relief_defaults_to_local_quality_surface_density(self):
        detail_components = [
            component
            for component in app.demo.get_config_file().get("components", [])
            if component.get("props", {}).get("label") == "Mesh detail"
        ]
        self.assertEqual(len(detail_components), 1)
        self.assertEqual(detail_components[0]["props"]["value"], 512)
        self.assertEqual(
            detail_components[0]["props"]["choices"],
            [
                ("1.5x (384 samples)", 384),
                ("2x (512 samples)", 512),
                ("3x (768 samples)", 768),
                ("4x (900 samples, capped)", 900),
            ],
        )

    def test_relief_defaults_match_local_physical_profile(self):
        components = app.demo.get_config_file().get("components", [])
        values_by_label = {
            component.get("props", {}).get("label"): component.get("props", {}).get("value")
            for component in components
        }
        self.assertEqual(values_by_label["Print size (%)"], 100)
        self.assertEqual(values_by_label["Relief height Z (mm)"], 10)
        self.assertEqual(values_by_label["X width (mm)"], 256)
        self.assertEqual(values_by_label["Y height (mm)"], 256)
        self.assertNotIn("Background depth", values_by_label)

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
        self.assertEqual(function_names.count("_apply_hover_selection"), 3)
        self.assertEqual(function_names.count("_invalidate_hover_selection"), 3)
        self.assertNotIn("_select_from_hover_event", function_names)
        self.assertNotIn("_select_from_click", function_names)
        prepare_functions = [
            function
            for function in app.demo.fns.values()
            if getattr(function.fn, "__name__", "") == "_prepare_hover_selection"
        ]
        self.assertTrue(all(function.trigger_mode == "always_last" for function in prepare_functions))
        dependencies = app.demo.get_config_file().get("dependencies", [])
        cancellation_dependencies = [
            dependency
            for dependency in dependencies
            if dependency.get("cancels")
        ]
        self.assertEqual(len(cancellation_dependencies), 9)
        apply_ids = {
            dependency["id"]
            for dependency in dependencies
            if str(dependency.get("api_name", "")).startswith("_apply_hover_selection")
        }
        draft_cancellations = [
            dependency
            for dependency in cancellation_dependencies
            if set(dependency["cancels"]).issubset(apply_ids)
            and len(dependency["cancels"]) == 1
        ]
        self.assertEqual(len(draft_cancellations), 3)
        invalidations = [
            dependency
            for dependency in dependencies
            if str(dependency.get("api_name", "")).startswith("_invalidate_hover_selection")
        ]
        self.assertEqual(len(invalidations), 3)
        self.assertTrue(all(dependency["queue"] is False for dependency in invalidations))
        self.assertIn('root.addEventListener("pointermove"', app.SELECTION_HOVER_JS)
        self.assertIn("regionAt(current", app.SELECTION_HOVER_JS)
        self.assertIn('fit === "scale-down"', app.SELECTION_HOVER_JS)
        self.assertIn('event.target.closest("button, input', app.SELECTION_HOVER_JS)
        self.assertIn("selectedRegions: new Map()", app.SELECTION_HOVER_JS)
        self.assertIn("picker.selectedRegions.has(regionId)", app.SELECTION_HOVER_JS)
        self.assertIn("window.__sam3UndoSelection", app.SELECTION_HOVER_JS)
        self.assertIn("window.__sam3ClearSelection", app.SELECTION_HOVER_JS)
        self.assertNotIn("fetch(", app.SELECTION_HOVER_JS)

        labels = {
            component.get("props", {}).get("value")
            for component in app.demo.get_config_file().get("components", [])
            if component.get("type") == "button"
        }
        self.assertIn("Undo", labels)
        self.assertIn("Clear", labels)
        self.assertIn("Done selecting", labels)

        with patch.object(
            app,
            "select_precomputed_objects",
            return_value={
                "selected": "selected_image.png",
                "overlay": "selection_overlay.png",
                "labels": ["person"],
                "mask_coverage": 0.25,
            },
        ):
            preview, _selection, _status = app._apply_hover_selection(
                "photo.png",
                {"precompute_id": "test"},
                '{"selections":[{"region_id":1,"x":0.5,"y":0.5}]}',
            )
        self.assertEqual(preview, "selected_image.png")

        cleared = app._invalidate_hover_selection(
            {"region_count": 8},
            '{"selections":[{"region_id":1},{"region_id":2}]}',
        )
        self.assertEqual(cleared[:2], (None, None))
        self.assertIn("2 objects kept", cleared[2])

    def test_packed_mask_state_has_real_ttl(self):
        state_components = [
            component
            for component in app.demo.get_config_file().get("components", [])
            if component.get("type") == "state"
        ]
        ttl_states = [
            component
            for component in state_components
            if component.get("props", {}).get("time_to_live")
            == app.SAM3_PRECOMPUTE_TTL_SECONDS
        ]
        self.assertEqual(len(ttl_states), 3)


if __name__ == "__main__":
    unittest.main()
