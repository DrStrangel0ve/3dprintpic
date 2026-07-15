import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from backend.benchmark import run_background_halo_continuity_smoke as halo_smoke


class BackgroundHaloContinuitySmokeTests(unittest.TestCase):
    @staticmethod
    def _fixture():
        rows, cols = np.indices((128, 160), dtype=np.float64)
        protection = ((rows - 64.0) / 24.0) ** 2 + ((cols - 76.0) / 20.0) ** 2 <= 1.0
        distance = distance_transform_edt(~protection) * 0.4
        baseline = 4.0 + 0.008 * cols + 0.004 * rows
        template = 0.12 * np.sin(cols * 1.17) * np.cos(rows * 0.91)
        return baseline, template, protection, distance

    def test_correlated_recovery_without_moat_passes(self):
        baseline, template, protection, distance = self._fixture()
        activation = np.clip((distance - 4.0) / 1.0, 0.0, 1.0)
        candidate = baseline + template * activation

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(metrics["checks"]["passed"])
        self.assertGreater(
            metrics["gain_bands"]["5_8mm"]["source_correlation"], 0.9
        )

    def test_low_frequency_ring_fails_moat_gate(self):
        baseline, template, protection, distance = self._fixture()
        ring = -0.12 * np.exp(-np.square((distance - 3.0) / 1.2))
        candidate = baseline + template * np.clip((distance - 4.0) / 1.0, 0.0, 1.0) + ring

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["passed"])
        self.assertTrue(
            not metrics["checks"]["sector_bin_mean"]
            or not metrics["checks"]["lowpass_amplitude"]
        )

    def test_far_only_detail_fails_recovery_gate(self):
        baseline, template, protection, distance = self._fixture()
        candidate = baseline.copy()
        candidate[distance > 14.0] += template[distance > 14.0]

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["relative_gain_5_8mm"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_delayed_until_ten_mm_recovery_fails_near_gain(self):
        baseline, template, protection, distance = self._fixture()
        activation = np.clip((distance - 10.0) / 1.5, 0.0, 1.0)
        candidate = baseline + template * activation

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["relative_gain_5_8mm"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_localized_sector_moat_fails_sector_gate(self):
        baseline, template, protection, distance = self._fixture()
        rows, cols = np.indices(baseline.shape, dtype=np.float64)
        angle = np.mod(np.arctan2(rows - 64.0, cols - 76.0), 2.0 * np.pi)
        activation = np.clip((distance - 4.0) / 1.0, 0.0, 1.0)
        sector_moat = (
            -0.12
            * np.exp(-np.square((distance - 3.0) / 1.2))
            * ((angle > 0.0) & (angle < np.pi / 3.0))
        )
        candidate = baseline + template * activation + sector_moat

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["sector_bin_mean"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_missing_ring_samples_fail_coverage(self):
        baseline, template, protection, distance = self._fixture()
        candidate = baseline + template * np.clip((distance - 4.0) / 1.0, 0.0, 1.0)
        candidate[(distance > 6.0) & (distance < 7.0)] = np.nan

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["finite_coverage"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_selective_high_energy_omission_fails_coverage(self):
        baseline, template, protection, distance = self._fixture()
        candidate = baseline + template * np.clip((distance - 4.0) / 1.0, 0.0, 1.0)
        near = (distance > 5.0) & (distance <= 8.0)
        threshold = np.percentile(np.abs(template[near]), 99.6)
        candidate[near & (np.abs(template) >= threshold)] = np.nan

        metrics = halo_smoke.halo_continuity_metrics(
            baseline,
            candidate,
            protection,
            template,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["checks"]["finite_coverage"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_metric_rejects_mismatched_grids(self):
        baseline, template, protection, _distance = self._fixture()
        with self.assertRaisesRegex(ValueError, "shared 2D surface grid"):
            halo_smoke.halo_continuity_metrics(
                baseline,
                baseline[:, :-1],
                protection,
                template,
                sample_pitch_mm=0.4,
            )

    def test_metric_is_stable_across_physical_sample_pitch(self):
        results = []
        for pitch in (0.4, 0.2):
            height = int(round(51.2 / pitch))
            width = int(round(64.0 / pitch))
            rows, cols = np.indices((height, width), dtype=np.float64)
            y_mm = rows * pitch
            x_mm = cols * pitch
            protection = (
                ((y_mm - 25.6) / 9.6) ** 2 + ((x_mm - 30.4) / 8.0) ** 2
                <= 1.0
            )
            distance = distance_transform_edt(~protection) * pitch
            baseline = 4.0 + 0.008 * x_mm + 0.004 * y_mm
            template = 0.12 * np.sin(x_mm * 2.9) * np.cos(y_mm * 2.3)
            candidate = baseline + template * np.clip(
                (distance - 4.0) / 1.0, 0.0, 1.0
            )

            metrics = halo_smoke.halo_continuity_metrics(
                baseline,
                candidate,
                protection,
                template,
                sample_pitch_mm=pitch,
            )
            self.assertTrue(metrics["checks"]["passed"])
            results.append(metrics["relative_gain_5_8mm"])

        self.assertLess(abs(results[0] - results[1]), 0.03)

    def test_halo_telemetry_requires_exact_method_and_distances(self):
        valid = {
            "protection_method": "euclidean_inner_guard_smoothstep_to_full_gain",
            "protection_halo_mm": 5.0,
            "protection_zero_guard_mm": 2.0,
        }
        self.assertTrue(halo_smoke._halo_telemetry_checks(valid)["passed"])
        for field, value in (
            ("protection_method", "legacy_square_gaussian"),
            ("protection_halo_mm", 7.0),
            ("protection_zero_guard_mm", 0.0),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    halo_smoke._halo_telemetry_checks(
                        {**valid, field: value}
                    )["passed"]
                )


if __name__ == "__main__":
    unittest.main()
