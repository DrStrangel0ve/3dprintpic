import hashlib
import tempfile
import unittest
from pathlib import Path

from backend.benchmark.summarize_private_live_api_background_replay import (
    EXPECTED_DETAIL_METHOD,
    _face_quality_record,
    _independent_background_checks,
    _independent_cap_checks,
    _photo_detail_record,
    _request_checks,
    _request_record_checks,
    _topology_record,
)
from backend.benchmark.run_private_live_api_background_replay import (
    FORM_FIELDS,
    OMITTED_BACKGROUND_FIELDS,
)


class PrivateLiveApiBackgroundReplayTest(unittest.TestCase):
    def test_request_contract_requires_exact_physical_controls_and_api_defaults(self):
        metadata = {
            "target_dimension": 512,
            "z_scale": 30.0,
            "max_xy_size": 128.0,
            "sigma": 0.35,
            "background_photo_detail_mm": 0.60,
            "selection_background_depth_ratio": 0.65,
            "relief_sample_pitch_mm": 128.0 / 511.0,
        }
        response = {
            "target_dimension": 512,
            "background_photo_detail_mm": 0.60,
            "selection_background_depth_ratio": 0.65,
            "relief_sample_pitch_mm": 128.0 / 511.0,
        }

        self.assertTrue(_request_checks(metadata, response)["passed"])
        response["background_photo_detail_mm"] = 0.12
        self.assertFalse(_request_checks(metadata, response)["passed"])

    def test_photo_detail_requires_selected_euclidean_guard_and_real_signal(self):
        postprocess = {
            "background_photo_detail": {
                "enabled": True,
                "face_protected": True,
                "protection_method": EXPECTED_DETAIL_METHOD,
                "requested_detail_mm": 0.60,
                "effective_detail_mm": 0.60,
                "protection_halo_mm": 5.0,
                "protection_zero_guard_mm": 2.0,
                "input_sample_pitch_mm": 0.25,
                "detail_scale": 0.03,
                "background_coverage_ratio": 0.22,
            }
        }

        self.assertTrue(_photo_detail_record(postprocess)["checks"]["passed"])
        postprocess["background_photo_detail"]["enabled"] = False
        self.assertFalse(_photo_detail_record(postprocess)["checks"]["passed"])

    def test_face_quality_reuses_frozen_appearance_and_component_gates(self):
        postprocess = {
            "surface_appearance_agreement": {
                "face": {
                    "available": True,
                    "candidate_coverage_ratio": 1.0,
                    "normal_mean_cosine": 0.99,
                    "normal_p05_cosine": 0.98,
                    "normal_angle_p95_deg": 8.0,
                    "minimum_lighting_correlation": 0.92,
                    "maximum_lighting_mae": 0.02,
                    "minimum_lighting_rms_retention": 0.95,
                    "maximum_lighting_rms_retention": 1.08,
                    "component_count": 2,
                    "measured_component_count": 2,
                    "unavailable_component_count": 0,
                }
            },
            "face_height_stabilization": {
                "gradient_compression": {
                    "height_span_ratio": 0.99,
                    "correction_span_ratio": 0.5,
                    "output_edge_ratio_p99": 5.0,
                    "output_edge_ratio_max": 14.0,
                    "quality_gates": {
                        "passed": True,
                        "minimum_detail_correlation": 0.85,
                        "minimum_detail_rms_retention": 0.6,
                        "maximum_detail_rms_retention": 2.0,
                        "minimum_height_span_ratio": 0.5,
                        "maximum_height_span_ratio": 1.15,
                        "maximum_correction_span_ratio": 0.9,
                        "maximum_output_edge_p99_ratio": 12.0,
                        "maximum_output_edge_ratio": 24.0,
                    },
                    "detail_preservation": {
                        "correlation": 0.96,
                        "rms_retention": 0.94,
                        "minimum_component_correlation": 0.95,
                        "minimum_component_rms_retention": 0.92,
                    },
                }
            },
        }

        self.assertTrue(_face_quality_record(postprocess)["checks"]["passed"])
        postprocess["surface_appearance_agreement"]["face"][
            "minimum_lighting_correlation"
        ] = 0.79
        self.assertFalse(_face_quality_record(postprocess)["checks"]["passed"])

    def test_face_quality_recomputes_detail_thresholds_instead_of_trusting_boolean(self):
        postprocess = {
            "surface_appearance_agreement": {
                "face": {
                    "available": True,
                    "candidate_coverage_ratio": 1.0,
                    "normal_mean_cosine": 0.99,
                    "normal_p05_cosine": 0.98,
                    "normal_angle_p95_deg": 8.0,
                    "minimum_lighting_correlation": 0.92,
                    "maximum_lighting_mae": 0.02,
                    "minimum_lighting_rms_retention": 0.95,
                    "maximum_lighting_rms_retention": 1.08,
                    "component_count": 1,
                    "measured_component_count": 1,
                    "unavailable_component_count": 0,
                }
            },
            "face_height_stabilization": {
                "gradient_compression": {
                    "height_span_ratio": 0.99,
                    "correction_span_ratio": 0.5,
                    "output_edge_ratio_p99": 5.0,
                    "output_edge_ratio_max": 14.0,
                    "quality_gates": {
                        "passed": True,
                        "minimum_detail_correlation": 0.85,
                        "minimum_detail_rms_retention": 0.6,
                        "maximum_detail_rms_retention": 2.0,
                        "minimum_height_span_ratio": 0.5,
                        "maximum_height_span_ratio": 1.15,
                        "maximum_correction_span_ratio": 0.9,
                        "maximum_output_edge_p99_ratio": 12.0,
                        "maximum_output_edge_ratio": 24.0,
                    },
                    "detail_preservation": {
                        "correlation": 0.2,
                        "rms_retention": 0.94,
                        "minimum_component_correlation": 0.2,
                        "minimum_component_rms_retention": 0.92,
                    },
                }
            },
        }

        self.assertFalse(_face_quality_record(postprocess)["checks"]["passed"])

    def test_topology_fails_closed_for_any_nonprintable_property(self):
        diagnostics = {
            "stl_exists": True,
            "stl_is_watertight": True,
            "stl_is_volume": True,
            "stl_is_manifold": True,
            "stl_winding_consistent": True,
            "stl_component_count": 1,
            "stl_degenerate_face_count": 0,
            "stl_positive_volume": True,
            "stl_bbox_has_volume": True,
            "stl_faces": 100,
            "stl_vertices": 52,
        }

        self.assertTrue(_topology_record(diagnostics)["checks"]["passed"])
        diagnostics["stl_is_watertight"] = False
        self.assertFalse(_topology_record(diagnostics)["checks"]["passed"])

    def test_request_record_proves_background_fields_were_omitted(self):
        self.assertFalse(set(FORM_FIELDS) & set(OMITTED_BACKGROUND_FIELDS))
        with tempfile.TemporaryDirectory() as temporary:
            response_path = Path(temporary) / "response.json"
            response_path.write_bytes(b"{}")
            job_id = "a" * 32
            record = {
                "schema_version": 1,
                "endpoint": "/process_image",
                "http_status": 200,
                "response_sha256": hashlib.sha256(b"{}").hexdigest(),
                "response_job_id": job_id,
                "posted_form_fields": sorted(FORM_FIELDS),
                "omitted_background_fields": list(OMITTED_BACKGROUND_FIELDS),
            }
            self.assertTrue(
                _request_record_checks(record, response_path, job_id)["passed"]
            )
            record["posted_form_fields"].append("background_photo_detail_mm")
            self.assertFalse(
                _request_record_checks(record, response_path, job_id)["passed"]
            )

    def test_cap_recomputes_numeric_invariants(self):
        cap = {
            "emission_passed": True,
            "far_background_cap_passed": True,
            "feasible_attachment_constraints_passed": True,
            "far_background_cap_violation_mm": 0.0,
            "far_background_max_mm": 19.5,
            "far_background_ceiling_mm": 19.5,
            "feasible_attachment_jump_max_mm": 0.8,
            "attachment_step_limit_mm": 0.8,
        }
        self.assertTrue(_independent_cap_checks(cap)["passed"])
        cap["far_background_cap_violation_mm"] = 0.2
        self.assertFalse(_independent_cap_checks(cap)["passed"])

    def test_background_recomputes_numeric_invariants(self):
        stats = {
            "available": True,
            "passed": True,
            "candidate_coverage_ratio": 1.0,
            "minimum_coverage_ratio": 1.0,
            "correlation": 0.99,
            "minimum_correlation": 0.8,
            "rms_retention": 1.0,
            "minimum_rms_retention": 0.5,
            "maximum_rms_retention": 1.5,
            "span_retention": 1.0,
            "minimum_span_retention": 0.5,
            "maximum_span_retention": 1.5,
            "gradient_correlation": 0.9,
            "minimum_gradient_correlation": 0.58,
            "gradient_rms_retention": 1.0,
            "minimum_gradient_rms_retention": 0.25,
            "maximum_gradient_rms_retention": 1.75,
            "mean_shift_mm": 0.0,
            "maximum_mean_shift_mm": 2.0,
            "output_boundary_jump_p99_mm": 1.0,
            "effective_boundary_jump_p99_limit_mm": 6.0,
            "output_boundary_jump_max_mm": 2.0,
            "effective_boundary_jump_max_limit_mm": 12.0,
            "localized_structure": {"passed": True},
            "quality_failures": [],
        }
        self.assertTrue(_independent_background_checks(stats)["passed"])
        stats["correlation"] = 0.1
        self.assertFalse(_independent_background_checks(stats)["passed"])


if __name__ == "__main__":
    unittest.main()
