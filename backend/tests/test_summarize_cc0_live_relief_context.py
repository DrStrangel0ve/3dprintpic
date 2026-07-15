import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from backend.benchmark import summarize_cc0_live_relief_context as summarizer


FINAL_REVISION = "1" * 40
INITIAL_REVISION = "2" * 40


def _response(*, detected: bool, gate: float = 0.4, detail_mm: float = 0.6) -> dict:
    return {
        "job_id": "private-job-id",
        "background_photo_detail_mm": detail_mm,
        "selection_background_depth_ratio": 0.65,
        "z_scale": 30.0,
        "max_xy_size": 96.0,
        "target_dimension": 384,
        "face_refinement": {
            "faces": [
                {
                    "eyewear_deocclusion": {
                        "enabled": detected,
                        "reason": None if detected else "private-reason",
                        "detection": {
                            "method": "broad_eye_band",
                            "enabled": detected,
                            "dark_coverage_ratio": 0.58 if detected else 0.35,
                            "component_coverage_ratio": 0.69 if detected else 0.43,
                            "component_width_ratio": 0.85,
                            "minimum_dark_coverage_ratio": gate,
                            "minimum_component_coverage_ratio": 0.45,
                            "minimum_component_width_ratio": 0.6,
                        },
                    }
                }
            ]
        },
    }


def _quality(*, passed: bool, occluded: bool) -> dict:
    return {
        "detected_faces": 1,
        "refined_faces": 1,
        "selected_detail_regions": 1,
        "exact_subject_depth": {
            "shape_correlation": 0.94,
            "gradient_correlation": 0.67,
            "normalized_rmse": 0.1,
        },
        "appearance": {
            "normal_mean_cosine": 0.99,
            "minimum_lighting_correlation": 0.98,
        },
        "background_depth": {
            "correlation": 0.99,
            "gradient_correlation": 0.98,
            "rms_retention": 1.0,
        },
        "physical_cap": {
            "far_background_cap_violation_mm": 0.0,
            "feasible_attachment_jump_max_mm": 0.8,
            "attachment_constraint_conflicts": 0,
        },
        "boundary_shape": {
            "output_p99_to_limit_ratio": 0.7,
            "output_max_to_limit_ratio": 0.6,
        },
        "topology": {"checks": {"passed": True}},
        "stl_heightfield_agreement": {"passed": True},
        "occlusion_handling": {
            "required": occluded,
            "eyewear_detected": passed if occluded else False,
            "deoccluded_faces": int(passed and occluded),
        },
        "checks": {
            "exact_subject_depth": True,
            "occlusion_deoccluded": passed if occluded else True,
            "passed": passed,
        },
    }


def _row(
    row_id: str,
    *,
    passed: bool,
    response_sha256: str | None = None,
    opacity: float | None = None,
) -> dict:
    occluded = row_id == summarizer.EYEWEAR_ROW_ID
    quality = _quality(passed=passed, occluded=occluded)
    scene = {
        "target_dimension": 384,
        "camera_yaw_deg": -17.0,
        "camera_elevation_deg": 3.0,
    }
    if occluded:
        scene["occluder"] = {"opacity": opacity}
    scene_kind, background_profile, lighting_profile = summarizer.EXPECTED_ROW_RENDER[
        row_id
    ]
    candidate = {"quality": quality}
    if response_sha256 is not None:
        candidate["response_artifact"] = {"sha256": response_sha256}
    return {
        "row_id": row_id,
        "scene": scene,
        "render": {
            "scene_kind": scene_kind,
            "background_profile": background_profile,
            "lighting_profile": lighting_profile,
            "selection_bbox_width_ratio": 0.5,
            "selection_bbox_height_ratio": 0.7,
            "background_depth_span": 0.3,
        },
        "variants": {
            "baseline": {"quality": copy.deepcopy(quality)},
            "candidate": candidate,
        },
        "pair_quality": {
            "background_photo_detail": {
                "source_detail_correlation_intended_background": 0.7,
                "source_aligned_capture_ratio": 0.8,
                "face_interior_p99_change_mm": 0.001,
                "max_attachment_boundary_change_mm": 0.002,
            }
        },
        "checks": {
            "baseline": passed,
            "candidate": passed,
            "input_background_depth_span": True,
            "paired_background_detail": True,
            "passed": passed,
        },
    }


def _summary(*, revision: str, rows: list[dict], passed: bool) -> dict:
    producer = {
        "available": True,
        "clean": True,
        "revision": revision,
        "status": [],
        "files": [
            {"path": path, "sha256": "a" * 64, "private_note": "strip-me"}
            for path in summarizer.EXPECTED_PRODUCER_PATHS
        ],
    }
    provenance = {
        "available": True,
        "clean": True,
        "revision": revision,
        "status": [],
    }
    return {
        "schema_version": 1,
        "run_kind": "cc0_generated_live_relief_context_matrix",
        "privacy": summarizer.EXPECTED_PRIVACY,
        "fixture": {
            "schema_version": 1,
            "license": "CC0-1.0",
            "fixture": {"sha256": summarizer.EXPECTED_FIXTURE_SHA256},
            "source": {
                "repository": "https://github.com/makehumancommunity/makehuman",
                "commit": "3" * 40,
            },
        },
        "checks": {
            "all_behavioral_rows_passed": passed,
            "raw_artifacts_beneath_ignored_output": True,
            "requested_rows_executed_once": True,
            "producer_matches_server_revision": True,
            "producer_unchanged_during_run": True,
            "server_repository_clean": True,
            "server_revision_unchanged": True,
            "server_runtime_matches_checkout": True,
            "passed": passed,
        },
        "matrix": {
            "executed_rows": 4,
            "z_scale": 30.0,
            "max_xy_size": 96.0,
            "baseline_background_photo_detail_mm": 0.0,
            "coverage": {
                "row_ids": list(summarizer.EXPECTED_ROW_IDS),
                "background_profiles": ["deep_shelves", "layered_studio"],
                "lighting_profiles": ["overhead", "side_right", "soft_left"],
                "scene_kinds": ["face", "object"],
            },
        },
        "server_provenance": copy.deepcopy(provenance),
        "server_runtime_provenance": copy.deepcopy(provenance),
        "producer_provenance": copy.deepcopy(producer),
        "final_producer_provenance": copy.deepcopy(producer),
        "rows": rows,
    }


class SummarizeCc0LiveReliefContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.initial_response = self.root / "initial.json"
        self.final_response = self.root / "final.json"
        self.initial_response.write_text(
            json.dumps(_response(detected=False)), encoding="utf-8"
        )
        self.final_response.write_text(
            json.dumps(_response(detected=True)), encoding="utf-8"
        )
        self.initial_summary = self.root / "initial-summary.json"
        self.final_summary = self.root / "final-summary.json"
        initial_rows = []
        final_rows = []
        for row_id in summarizer.EXPECTED_ROW_IDS:
            if row_id == summarizer.EYEWEAR_ROW_ID:
                initial_rows.append(
                    _row(
                        row_id,
                        passed=False,
                        response_sha256=self._sha256(self.initial_response),
                        opacity=0.42,
                    )
                )
                final_rows.append(
                    _row(
                        row_id,
                        passed=True,
                        response_sha256=self._sha256(self.final_response),
                        opacity=0.6,
                    )
                )
            else:
                initial_rows.append(_row(row_id, passed=True))
                final_rows.append(_row(row_id, passed=True))
        self._write_summaries(initial_rows, final_rows)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _eyewear_row(summary: dict) -> dict:
        return next(
            row for row in summary["rows"] if row["row_id"] == summarizer.EYEWEAR_ROW_ID
        )

    def _write_summaries(
        self, initial_rows: list[dict], final_rows: list[dict]
    ) -> None:
        self.initial_summary.write_text(
            json.dumps(
                _summary(revision=INITIAL_REVISION, rows=initial_rows, passed=False)
            ),
            encoding="utf-8",
        )
        self.final_summary.write_text(
            json.dumps(_summary(revision=FINAL_REVISION, rows=final_rows, passed=True)),
            encoding="utf-8",
        )

    def _summarize(self) -> dict:
        return summarizer.summarize(
            self.final_summary,
            self.initial_summary,
            self.final_response,
            self.initial_response,
        )

    def test_generates_privacy_safe_compact_evidence(self):
        compact = self._summarize()
        serialized = json.dumps(compact, sort_keys=True)

        self.assertTrue(compact["checks"]["passed"])
        self.assertEqual(compact["aggregate"]["passed_rows"], 4)
        self.assertEqual(
            [record["path"] for record in compact["producer_files"]],
            list(summarizer.EXPECTED_PRODUCER_PATHS),
        )
        self.assertTrue(
            compact["measured_fixture_correction"]["production_gates_unchanged"]
        )
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("private-job-id", serialized)
        self.assertNotIn("private-reason", serialized)
        self.assertNotIn("private_note", serialized)

    def test_rejects_response_checksum_mismatch(self):
        self.final_response.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self._summarize()

    def test_rejects_non_isolated_initial_failure(self):
        initial = json.loads(self.initial_summary.read_text(encoding="utf-8"))
        row = self._eyewear_row(initial)
        for variant in ("baseline", "candidate"):
            row["variants"][variant]["quality"]["checks"]["exact_subject_depth"] = False
        self.initial_summary.write_text(json.dumps(initial), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not isolated"):
            self._summarize()

    def test_rejects_production_gate_change(self):
        self.final_response.write_text(
            json.dumps(_response(detected=True, gate=0.33)), encoding="utf-8"
        )
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        self._eyewear_row(final)["variants"]["candidate"]["response_artifact"][
            "sha256"
        ] = self._sha256(self.final_response)
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "gates changed"):
            self._summarize()

    def test_rejects_changed_final_producer_provenance(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        final["final_producer_provenance"]["files"][0]["sha256"] = "b" * 64
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "changed during"):
            self._summarize()

    def test_rejects_dirty_initial_provenance(self):
        initial = json.loads(self.initial_summary.read_text(encoding="utf-8"))
        initial["producer_provenance"]["clean"] = False
        initial["final_producer_provenance"]["clean"] = False
        self.initial_summary.write_text(json.dumps(initial), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unavailable or dirty"):
            self._summarize()

    def test_rejects_effective_configuration_change(self):
        self.final_response.write_text(
            json.dumps(_response(detected=True, detail_mm=0.7)), encoding="utf-8"
        )
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        self._eyewear_row(final)["variants"]["candidate"]["response_artifact"][
            "sha256"
        ] = self._sha256(self.final_response)
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "API configuration changed"):
            self._summarize()

    def test_rejects_scene_drift_beyond_opacity(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        self._eyewear_row(final)["scene"]["camera_yaw_deg"] = -10.0
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "beyond eyewear opacity"):
            self._summarize()

    def test_rejects_non_public_fixture(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        final["fixture"]["license"] = "private"
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "fixture provenance"):
            self._summarize()

    def test_inconsistent_row_count_cannot_pass(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        final["matrix"]["executed_rows"] = 5
        initial = json.loads(self.initial_summary.read_text(encoding="utf-8"))
        initial["matrix"]["executed_rows"] = 5
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")
        self.initial_summary.write_text(json.dumps(initial), encoding="utf-8")

        self.assertFalse(self._summarize()["checks"]["passed"])

    def test_rejects_string_boolean(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        final["checks"]["server_repository_clean"] = "false"
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "boolean"):
            self._summarize()

    def test_rejects_stale_final_pass_flag(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        self._eyewear_row(final)["variants"]["candidate"]["quality"]["checks"][
            "exact_subject_depth"
        ] = False
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "failed nested check"):
            self._summarize()

    def test_rejects_extra_initial_row_failure(self):
        initial = json.loads(self.initial_summary.read_text(encoding="utf-8"))
        self._eyewear_row(initial)["checks"]["input_background_depth_span"] = False
        self.initial_summary.write_text(json.dumps(initial), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not isolated to eyewear"):
            self._summarize()

    def test_rejects_private_row_profile(self):
        for path in (self.initial_summary, self.final_summary):
            summary = json.loads(path.read_text(encoding="utf-8"))
            self._eyewear_row(summary)["render"]["background_profile"] = "C:/private"
            path.write_text(json.dumps(summary), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "public row profile"):
            self._summarize()

    def test_rejects_missing_full_run_check(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        del final["checks"]["server_runtime_matches_checkout"]
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "missing required checks"):
            self._summarize()

    def test_rejects_noop_opacity_correction(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        self._eyewear_row(final)["scene"]["occluder"]["opacity"] = 0.42
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "0.42 to 0.60"):
            self._summarize()

    def test_rejects_non_hexadecimal_source_hash(self):
        final = json.loads(self.final_summary.read_text(encoding="utf-8"))
        for key in ("producer_provenance", "final_producer_provenance"):
            final[key]["files"][0]["sha256"] = "z" * 64
        self.final_summary.write_text(json.dumps(final), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "hexadecimal digest"):
            self._summarize()


if __name__ == "__main__":
    unittest.main()
