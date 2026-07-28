import tempfile
import unittest

from backend.benchmark import run_background_photo_detail_provider_smoke as provider_smoke


class BackgroundPhotoDetailProviderSmokeTests(unittest.TestCase):
    def test_portable_manifest_removes_local_model_paths(self):
        manifest = {
            "model_id": "depth-anything/example",
            "provider_metadata": {
                "model": "C:/Users/example/.cache/model",
                "effective_model": "C:/Users/example/.cache/model",
                "revision": "abc123",
            },
        }

        portable = provider_smoke._portable_inference_manifest(manifest)

        self.assertEqual(
            portable["provider_metadata"]["model"], "depth-anything/example"
        )
        self.assertEqual(
            portable["provider_metadata"]["effective_model"],
            "depth-anything/example",
        )
        self.assertEqual(portable["provider_metadata"]["revision"], "abc123")
        self.assertIn(".cache", manifest["provider_metadata"]["model"])

    def test_detail_telemetry_requires_protected_exact_effective_value(self):
        valid = {
            "enabled": True,
            "requested_detail_mm": 0.60,
            "effective_detail_mm": 0.60,
            "face_protected": True,
            "unprotected_detail_limited": False,
        }
        self.assertTrue(provider_smoke._detail_telemetry_checks(valid, 0.60)["passed"])

        for field, value in (
            ("effective_detail_mm", 0.12),
            ("face_protected", False),
            ("unprotected_detail_limited", True),
            ("enabled", False),
        ):
            with self.subTest(field=field):
                invalid = {**valid, field: value}
                self.assertFalse(
                    provider_smoke._detail_telemetry_checks(invalid, 0.60)["passed"]
                )

    def test_run_rejects_noncanonical_pair_before_loading_assets(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            provider_smoke.run(
                temporary,
                detail_levels_mm=(0.0, 0.12, 0.60),
            )


if __name__ == "__main__":
    unittest.main()
