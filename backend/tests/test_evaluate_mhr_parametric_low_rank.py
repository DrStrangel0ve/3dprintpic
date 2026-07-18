import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark import evaluate_mhr_parametric_low_rank as audit
from backend.benchmark.evaluate_mhr_parametric_low_rank import (
    clamp_search_initial,
    clear_output_artifacts,
    decode_pca_delta,
    fit_geometry_pca,
    fit_pca,
    observed_asset,
    search_score,
    strictly_better,
    validate_training_partition,
    verify_declared_asset,
)


class MHRParametricLowRankTests(unittest.TestCase):
    @staticmethod
    def _declared_asset(root: Path, relative_path: str, payload: bytes) -> dict:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    def test_search_initial_is_clamped_to_declared_bounds(self):
        self.assertEqual(clamp_search_initial(12.0, -10.0, 10.0), 10.0)
        self.assertEqual(clamp_search_initial(-12.0, -10.0, 10.0), -10.0)
        with self.assertRaisesRegex(ValueError, "finite and ordered"):
            clamp_search_initial(0.0, 1.0, -1.0)

    def test_fit_pca_is_centered_and_reports_explained_variance(self):
        values = np.asarray(
            (
                (-2.0, 0.0, 0.0),
                (-1.0, 0.1, 0.0),
                (1.0, -0.1, 0.0),
                (2.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
        basis = fit_pca(values, 1)
        np.testing.assert_allclose(basis["mean"], 0.0, atol=1e-12)
        self.assertEqual(basis["components"].shape, (1, 3))
        self.assertGreater(basis["explained_variance_ratio"], 0.99)

    def test_fit_pca_rejects_invalid_rank_and_nonfinite_values(self):
        values = np.eye(3, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            fit_pca(values, 3)
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite matrix"):
            fit_pca(values, 1)

    def test_geometry_pca_prioritizes_surface_displacement(self):
        values = np.asarray(
            ((-2.0, -0.1), (-1.0, 0.1), (1.0, -0.1), (2.0, 0.1)),
            dtype=np.float64,
        )
        jacobian = np.zeros((2, 1, 3), dtype=np.float64)
        jacobian[0, 0, 0] = 0.1
        jacobian[1, 0, 1] = 100.0
        basis = fit_geometry_pca(
            values,
            jacobian,
            np.ones(1),
            1,
            region_vertex_weights={"feature": np.ones(1)},
        )
        geometry_direction = basis["components"][0] @ jacobian.reshape(2, -1)
        self.assertGreater(
            abs(float(geometry_direction[1])),
            abs(float(geometry_direction[0])),
        )
        self.assertGreater(basis["explained_variance_ratio"], 0.99)
        self.assertGreater(basis["minimum_region_retention"], 0.99)

    def test_decode_pca_delta_is_anchor_relative_and_bounded(self):
        basis = {
            "columns": 2,
            "rank": 1,
            "scales": np.asarray((2.0,)),
            "components": np.asarray(((1.0, 0.0),)),
        }
        decoded = decode_pca_delta(
            np.asarray((0.5, -0.25)),
            basis,
            np.asarray((1.0,)),
            coefficient_limit=1.0,
        )
        np.testing.assert_allclose(decoded, (1.0, -0.25))

    def test_strict_improvement_requires_failures_and_all_metrics(self):
        reference = {
            "combined_part_failures": 9,
            "shape_correlation": 0.80,
            "gradient_correlation": 0.60,
            "normalized_rmse": 0.20,
            "shape_failed_parts": ["left_eye"],
            "affine_failed_parts": ["nose"],
        }
        candidate = {
            "combined_part_failures": 8,
            "shape_correlation": 0.81,
            "gradient_correlation": 0.61,
            "normalized_rmse": 0.19,
            "shape_failed_parts": [],
            "affine_failed_parts": ["nose"],
        }
        self.assertTrue(strictly_better(candidate, reference))
        candidate["gradient_correlation"] = 0.59
        self.assertFalse(strictly_better(candidate, reference))
        candidate["gradient_correlation"] = 0.61
        candidate["shape_failed_parts"] = ["right_eye"]
        self.assertFalse(strictly_better(candidate, reference))

    def test_search_score_prioritizes_valid_strict_candidates(self):
        reference = {
            "combined_part_failures": 9,
            "shape_correlation": 0.80,
            "gradient_correlation": 0.60,
            "normalized_rmse": 0.20,
            "shape_failed_parts": ["left_eye"],
            "affine_failed_parts": ["nose"],
        }
        invalid_fewer_failures = {
            "combined_part_failures": 7,
            "shape_correlation": 0.70,
            "gradient_correlation": 0.50,
            "normalized_rmse": 0.30,
            "shape_failed_parts": ["left_eye", "right_eye"],
            "affine_failed_parts": ["nose"],
        }
        same_failures_regressed = {
            "combined_part_failures": 9,
            "shape_correlation": 0.79,
            "gradient_correlation": 0.59,
            "normalized_rmse": 0.21,
            "shape_failed_parts": ["left_eye"],
            "affine_failed_parts": ["nose"],
        }
        same_failures_improved = {
            "combined_part_failures": 9,
            "shape_correlation": 0.81,
            "gradient_correlation": 0.61,
            "normalized_rmse": 0.19,
            "shape_failed_parts": ["left_eye"],
            "affine_failed_parts": ["nose"],
        }
        valid_strict = {
            "combined_part_failures": 8,
            "shape_correlation": 0.81,
            "gradient_correlation": 0.61,
            "normalized_rmse": 0.19,
            "shape_failed_parts": [],
            "affine_failed_parts": ["nose"],
        }
        new_part_failure = {
            **valid_strict,
            "combined_part_failures": 7,
            "shape_failed_parts": ["right_eye"],
        }
        self.assertLess(
            search_score(valid_strict, reference),
            search_score(invalid_fewer_failures, reference),
        )
        self.assertLess(
            search_score(same_failures_improved, reference),
            search_score(invalid_fewer_failures, reference),
        )
        self.assertLess(
            search_score(same_failures_improved, reference),
            search_score(same_failures_regressed, reference),
        )
        self.assertLess(
            search_score(valid_strict, reference),
            search_score(new_part_failure, reference),
        )

    def test_declared_asset_verification_is_content_addressed_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "asset.bin"
            payload = b"verified input"
            path.write_bytes(payload)
            record = {
                "path": "asset.bin",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            verified = verify_declared_asset(root, record, "fixture")
            self.assertTrue(verified["verified"])
            self.assertEqual(verified["sha256"], record["sha256"])
            observed = observed_asset(path, root, "fixture")
            self.assertEqual(observed["relative_path"], "asset.bin")
            self.assertEqual(observed["sha256"], record["sha256"])
            path.write_bytes(payload + b" changed")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_declared_asset(root, record, "fixture")
            escaping = {**record, "path": "../asset.bin"}
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_declared_asset(root, escaping, "fixture")

    def test_output_cleanup_removes_only_owned_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strict = root / "strict_incumbent_candidate_depth.npy"
            unrelated = root / "keep.txt"
            strict.write_bytes(b"stale")
            unrelated.write_text("keep", encoding="utf-8")
            clear_output_artifacts(root)
            self.assertFalse(strict.exists())
            self.assertTrue(unrelated.exists())

    def test_evaluate_emits_verified_pose_oracle_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            run_root.mkdir()
            output_dir = root / "audit"
            output_dir.mkdir()
            (output_dir / "strict_incumbent_candidate_depth.npy").write_bytes(
                b"stale"
            )

            source_path = run_root / "source.png"
            Image.new("RGB", (8, 8), "white").save(source_path)
            source = self._declared_asset(
                run_root,
                "source.png",
                source_path.read_bytes(),
            )
            selection_path = run_root / "selection.png"
            Image.new("L", (8, 8), 255).save(selection_path)
            selection = self._declared_asset(
                run_root,
                "selection.png",
                selection_path.read_bytes(),
            )
            exact_path = run_root / "exact.npy"
            np.save(exact_path, np.ones((8, 8), dtype=np.float32))
            exact_depth = self._declared_asset(
                run_root,
                "exact.npy",
                exact_path.read_bytes(),
            )
            exact_parts = {}
            for part in (
                "left_eye",
                "right_eye",
                "left_eyebrow",
                "right_eyebrow",
                "nose",
                "mouth",
            ):
                part_path = run_root / "parts" / f"{part}.png"
                part_path.parent.mkdir(exist_ok=True)
                Image.new("L", (8, 8), 255).save(part_path)
                exact_parts[part] = self._declared_asset(
                    run_root,
                    f"parts/{part}.png",
                    part_path.read_bytes(),
                )

            job_id = "candidate-job"
            job_dir = root / job_id
            (job_dir / "face_refinement/face_00_depth").mkdir(parents=True)
            (job_dir / "face_refinement/face_00_parts").mkdir(parents=True)
            metadata = {
                "faces": [
                    {
                        "crop_bbox": [1, 1, 7, 7],
                        "part_masks": {
                            "face_file": "face_refinement/face_00_parts/face.png"
                        },
                    }
                ]
            }
            (job_dir / "output_face_refinement_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            np.save(
                job_dir / "output_depth_data_face_refined.npy",
                np.full((8, 8), 0.5, dtype=np.float32),
            )
            np.save(
                job_dir / "face_refinement/face_00_depth/output_depth_data.npy",
                np.full((6, 6), 0.5, dtype=np.float32),
            )
            Image.new("L", (8, 8), 255).save(
                job_dir / "face_refinement/face_00_parts/face.png"
            )
            request = self._declared_asset(run_root, "request.json", b"{}")
            response_payload = json.dumps({"job_id": job_id}).encode()
            response = self._declared_asset(
                run_root,
                "response.json",
                response_payload,
            )
            summary = {
                "rows": [
                    {
                        "row_id": audit.HARD_ROW_ID,
                        "source": source,
                        "selection_mask": selection,
                        "exact_depth": exact_depth,
                        "exact_face_part_masks": {"files": exact_parts},
                        "scene": {
                            "camera_yaw_deg": 29.0,
                            "camera_elevation_deg": -4.0,
                        },
                        "variants": {
                            "candidate": {
                                "job_id": job_id,
                                "request_artifact": request,
                                "response_artifact": response,
                            }
                        },
                    }
                ]
            }
            (run_root / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            quality = {
                "combined_part_failures": 10,
                "shape_correlation": 0.80,
                "gradient_correlation": 0.60,
                "normalized_rmse": 0.20,
                "shape_failed_parts": ["left_eye"],
                "affine_failed_parts": ["nose"],
            }
            incumbent = {
                **quality,
                "combined_part_failures": 9,
                "shape_correlation": 0.81,
                "gradient_correlation": 0.61,
                "normalized_rmse": 0.19,
            }

            def fake_basis(values, _jacobian, _weights, rank, **_kwargs):
                columns = int(values.shape[1])
                return {
                    "rank": rank,
                    "rows": int(values.shape[0]),
                    "columns": columns,
                    "mean": np.zeros(columns),
                    "scales": np.ones(rank),
                    "components": np.eye(rank, columns),
                    "explained_variance_ratio": 1.0,
                    "geometry_metric_ridge": 0.0,
                    "weighted_vertices": 1,
                    "region_retention": {"face": 1.0},
                    "minimum_region_retention": 1.0,
                }

            def fake_optimizer(objective, bounds, **_kwargs):
                vector = np.zeros(len(bounds), dtype=np.float64)
                vector[-1] = 0.5
                value = objective(vector)
                return SimpleNamespace(
                    success=False,
                    message="bounded-test-budget",
                    fun=value,
                    nit=0,
                    nfev=1,
                    population=np.zeros((1, len(bounds))),
                )

            preflight = {
                "runnable": True,
                "revision": "pinned",
                "expected_revision": "pinned",
                "license": "Apache-2.0",
                "checks": {"source_revision_matches": True},
            }
            family = (
                np.zeros((3, 20)),
                np.zeros((3, 72)),
                np.zeros(20),
                np.zeros(72),
                {"all_basis_rows_are_train_only": True},
            )
            jacobians = (
                np.zeros((20, 1, 3)),
                np.zeros((72, 1, 3)),
                np.ones(1),
                {"face": np.ones(1)},
                {"method": "test", "source_geometry_used": False},
            )
            with (
                patch.object(audit, "_quality", return_value=quality),
                patch.object(
                    audit,
                    "_load_incumbent",
                    return_value=(incumbent, {"sha256": "incumbent"}),
                ),
                patch.object(audit, "_load_training_family", return_value=family),
                patch.object(audit, "preflight_mhr_root", return_value=preflight),
                patch.object(
                    audit,
                    "load_mhr_geometry_runtime",
                    return_value=(object(), np.ones(1), {"device": "cpu"}),
                ),
                patch.object(
                    audit,
                    "mhr_head_displacement_jacobians",
                    return_value=jacobians,
                ),
                patch.object(audit, "fit_geometry_pca", side_effect=fake_basis),
                patch.object(
                    audit,
                    "render_mhr_camera_depth",
                    return_value=(np.full((6, 6), 0.5), {"camera": "test"}),
                ),
                patch.object(
                    audit,
                    "build_small_face_candidate",
                    return_value=(np.full((8, 8), 0.5), {"method": "test"}),
                ),
                patch.object(
                    audit,
                    "differential_evolution",
                    side_effect=fake_optimizer,
                ),
            ):
                evidence = audit.evaluate(
                    run_root,
                    root / "corpus",
                    root / "mhr",
                    root / "incumbent.json",
                    output_dir,
                    identity_rank=1,
                    expression_rank=1,
                    population_multiplier=1,
                    maximum_iterations=0,
                    device="cpu",
                )

            self.assertEqual(evidence["schema_version"], 2)
            self.assertTrue(evidence["checks"]["pose_oracle_diagnostic"])
            self.assertFalse(
                evidence["checks"]["source_geometry_used_only_for_scoring"]
            )
            self.assertTrue(evidence["checks"]["declared_input_assets_verified"])
            self.assertEqual(
                evidence["search"]["known_pose_usage"],
                "oracle-centered-candidate-generation",
            )
            self.assertFalse(evidence["search"]["known_pose_available_in_production"])
            self.assertEqual(
                evidence["implementation"]["sha256"],
                hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
            )
            self.assertIn("baseline_depth", evidence["input_assets"]["consumed_job"])
            self.assertFalse(
                (output_dir / "strict_incumbent_candidate_depth.npy").exists()
            )

    def test_manifest_partition_must_match_summary_train_rows(self):
        summary_rows = [
            {
                "row_id": "identity_000__scene_00",
                "split": "train",
                "identity_group": "identity_000",
                "spec": {"identity_group": "identity_000", "split": "train"},
            },
            {
                "row_id": "identity_030__scene_00",
                "split": "validation",
                "identity_group": "identity_030",
                "spec": {
                    "identity_group": "identity_030",
                    "split": "validation",
                },
            },
        ]
        manifest = [{"row_id": "identity_000__scene_00", "split": "train"}]
        validated = validate_training_partition(summary_rows, manifest)
        self.assertEqual(set(validated), {"identity_000__scene_00"})
        leaked = [{"row_id": "identity_030__scene_00", "split": "train"}]
        with self.assertRaisesRegex(ValueError, "authoritative train rows"):
            validate_training_partition(summary_rows, leaked)
        mismatched_identity = [dict(summary_rows[0])]
        mismatched_identity[0]["identity_group"] = "identity_999"
        with self.assertRaisesRegex(ValueError, "identity or split"):
            validate_training_partition(mismatched_identity, manifest)

if __name__ == "__main__":
    unittest.main()
