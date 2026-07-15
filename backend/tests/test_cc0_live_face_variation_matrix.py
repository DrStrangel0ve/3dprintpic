import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark import run_cc0_live_face_variation_matrix as matrix


REVISION = "1" * 40


def _image_fingerprint(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as loaded:
        image = loaded.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{image.width}x{image.height}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")


class FakeLiveClient:
    def __init__(self, server_output: Path, revision: str = REVISION):
        self.server_output = server_output
        self.revision = revision
        self.calls = []
        self.selection_job_id = "a" * 32
        self.process_index = 0

    @property
    def provenance(self) -> dict:
        return {
            "available": True,
            "revision": self.revision,
            "clean": True,
            "status": [],
            "source": "git-worktree",
        }

    def get(self, endpoint: str):
        self.calls.append(("GET", endpoint, {}))
        return FakeResponse(
            200,
            {
                "status": "ok",
                "output_dir": str(self.server_output),
                "runtime": {"implementation_provenance": self.provenance},
            },
        )

    def post(self, endpoint: str, *, files: dict, data: dict):
        fields = dict(data)
        upload = files["file"][1]
        payload = upload.read()
        self.calls.append(("POST", endpoint, fields))
        if endpoint == "/selection/compose":
            return FakeResponse(
                200,
                {
                    "job_id": self.selection_job_id,
                    "model_status": "composed-clicked-masks",
                    "mask_count": 1,
                    "source_fingerprint": _image_fingerprint(payload),
                },
            )
        if endpoint != "/process_image":
            return FakeResponse(404, {"detail": "missing"})
        self.process_index += 1
        return FakeResponse(
            200,
            {
                "job_id": f"{self.process_index:032x}",
                "requested_target_dimension": int(fields["target_dimension"]),
                "target_dimension": int(fields["target_dimension"]),
                "z_scale": float(fields["z_scale"]),
                "max_xy_size": float(fields["max_xy_size"]),
                "background_photo_detail_mm": float(
                    fields.get("background_photo_detail_mm", 0.60)
                ),
                "selection_background_depth_ratio": float(
                    fields.get("selection_background_depth_ratio", 0.65)
                ),
                "selection_depth_context": {
                    "selection_job_id": fields["selection_job_id"]
                },
                "runtime": {"implementation_provenance": self.provenance},
            },
        )


class CC0LiveFaceVariationMatrixTests(unittest.TestCase):
    def test_recommended_smoke_row_is_first(self):
        first = matrix.DEFAULT_MATRIX[0]
        self.assertEqual(first.row_id, "small_off_axis_yaw_256")
        self.assertEqual(first.target_dimension, 256)
        self.assertNotEqual(first.camera_yaw_deg, 0.0)
        self.assertNotEqual(first.horizontal_offset, 0.0)
        self.assertGreater(first.camera_distance / first.camera_scale, 5.0)

    def test_process_forms_prove_candidate_background_fields_are_omitted(self):
        spec = matrix.DEFAULT_MATRIX[0]
        baseline = matrix._process_form(spec, "a" * 32, baseline=True)
        candidate = matrix._process_form(spec, "a" * 32, baseline=False)

        self.assertEqual(baseline["background_photo_detail_mm"], "0")
        self.assertNotIn("selection_background_depth_ratio", baseline)
        self.assertFalse(set(matrix.BACKGROUND_FIELDS) & set(candidate))
        self.assertEqual(candidate["z_scale"], "30")
        self.assertEqual(candidate["max_xy_size"], "96")

    def test_render_controls_shift_face_and_preserve_bounded_occluder_selection(self):
        size = 64
        silhouette = np.zeros((size, size), dtype=bool)
        silhouette[18:46, 10:22] = True
        rendered = SimpleNamespace(
            silhouette=silhouette,
            rgb=np.full((size, size, 3), 0.5, dtype=np.float32),
            depth=np.full((size, size), 0.4, dtype=np.float32),
        )
        mesh = SimpleNamespace(vertices=np.zeros((3, 3), dtype=np.float64))
        fixture = {
            "profiles": {
                "profile": {
                    "mesh": mesh,
                    "skin_tone": (0.5, 0.4, 0.3),
                }
            },
            "part_weights": {},
            "surface_weights": {},
        }
        spec = matrix.FaceSceneSpec(
            row_id="controlled",
            profile_name="profile",
            target_dimension=size,
            camera_yaw_deg=21.0,
            camera_distance=4.0,
            camera_scale=0.8,
            horizontal_offset=0.125,
            occluder=matrix.OccluderSpec(0.30, 0.30, 0.40, 0.40),
        )

        with (
            patch.object(
                matrix, "make_profile_vertex_colors", return_value=np.zeros((3, 3))
            ),
            patch.object(matrix, "render_mesh", return_value=rendered) as render,
        ):
            source, mask, exact_depth, record = matrix._render_scene_arrays(
                spec,
                fixture,
            )

        camera = render.call_args.args[1]
        config = render.call_args.args[2]
        self.assertEqual(camera.azimuth_deg, 21.0)
        self.assertEqual(config.size, size)
        self.assertEqual(config.camera_distance, 5.0)
        self.assertEqual(record["horizontal_offset_columns"], 8)
        self.assertTrue(np.any(mask[18:46, 18:30]))
        self.assertTrue(np.any(mask[19:26, 19:26]))
        self.assertGreater(record["occluder_pixels"], 0)
        self.assertGreater(record["face_bbox_width_pixels"], 0)
        self.assertEqual(source.shape, (size, size, 3))
        self.assertEqual(exact_depth.shape, (size, size))

    def test_oversized_occluder_is_rejected(self):
        spec = matrix.FaceSceneSpec(
            row_id="bad_occluder",
            profile_name="profile",
            target_dimension=256,
            camera_yaw_deg=0.0,
            camera_distance=3.0,
            occluder=matrix.OccluderSpec(0.0, 0.0, 0.8, 0.8),
        )
        with self.assertRaisesRegex(ValueError, "bounded foreground"):
            matrix._validate_scene_spec(spec)

    def test_one_row_run_records_hashes_fields_status_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "server"
            server_output = repository / "backend" / "output"
            output_dir = server_output / "matrix-test"
            client = FakeLiveClient(server_output.resolve())
            provenance = {
                "available": True,
                "revision": REVISION,
                "clean": True,
                "status": [],
            }
            source = np.full((16, 16, 3), 127, dtype=np.uint8)
            mask_values = np.zeros((16, 16), dtype=np.uint8)
            mask_values[4:12, 5:11] = 255
            render_record = {
                "effective_camera_distance": 6.0,
                "horizontal_offset_columns": 2,
                "visible_face_pixels": 48,
                "visible_face_fraction": 48 / 256,
                "occluder_pixels": 0,
                "face_bbox_xyxy": [5, 4, 11, 12],
                "face_bbox_width_pixels": 6,
                "face_bbox_height_pixels": 8,
                "face_bbox_width_ratio": 6 / 16,
                "face_bbox_height_ratio": 8 / 16,
            }

            with (
                patch.object(
                    matrix, "_clean_server_provenance", return_value=provenance
                ),
                patch.object(
                    matrix,
                    "_producer_provenance",
                    return_value={**provenance, "files": []},
                ),
                patch.object(matrix, "_require_ignored"),
                patch.object(
                    matrix,
                    "load_makehuman_face_fixture",
                    return_value={"manifest": {"license": "CC0-1.0"}},
                ),
                patch.object(
                    matrix,
                    "_render_scene_arrays",
                    return_value=(
                        source,
                        mask_values,
                        np.ones((16, 16), dtype=np.float32),
                        render_record,
                    ),
                ),
                patch.object(
                    matrix,
                    "_score_variant",
                    return_value={"checks": {"passed": True}},
                ),
                patch.object(
                    matrix,
                    "_score_pair",
                    return_value={"checks": {"passed": True}},
                ),
            ):
                summary = matrix.run(
                    repository,
                    output_dir=output_dir,
                    specs=matrix.DEFAULT_MATRIX[:2],
                    limit=1,
                    client=client,
                )

            self.assertTrue(summary["checks"]["passed"])
            self.assertEqual(summary["matrix"]["available_rows"], 2)
            self.assertEqual(summary["matrix"]["executed_rows"], 1)
            self.assertEqual(
                summary["rows"][0]["row_id"], matrix.DEFAULT_MATRIX[0].row_id
            )
            self.assertTrue((output_dir / "summary.json").is_file())
            row_dir = output_dir / "rows" / matrix.DEFAULT_MATRIX[0].row_id
            self.assertTrue((row_dir / "source.png").is_file())
            self.assertTrue((row_dir / "selection_mask.png").is_file())
            self.assertTrue((row_dir / "exact_depth.npy").is_file())
            self.assertRegex(summary["rows"][0]["source"]["sha256"], r"^[a-f0-9]{64}$")

            posts = [call for call in client.calls if call[0] == "POST"]
            self.assertEqual(
                [call[1] for call in posts],
                [
                    "/selection/compose",
                    "/process_image",
                    "/process_image",
                ],
            )
            baseline_fields = posts[1][2]
            candidate_fields = posts[2][2]
            self.assertEqual(baseline_fields["background_photo_detail_mm"], "0")
            self.assertFalse(set(matrix.BACKGROUND_FIELDS) & set(candidate_fields))
            candidate_record = summary["rows"][0]["variants"]["candidate"]["request"]
            self.assertEqual(
                candidate_record["omitted_background_fields"],
                sorted(matrix.BACKGROUND_FIELDS),
            )
            self.assertEqual(candidate_record["http_status"], 200)
            self.assertEqual(
                summary["rows"][0]["variants"]["candidate"]["runtime_provenance"][
                    "revision"
                ],
                REVISION,
            )
            self.assertTrue(summary["rows"][0]["checks"]["passed"])
            self.assertEqual(
                summary["matrix"]["coverage"]["row_ids"],
                [matrix.DEFAULT_MATRIX[0].row_id],
            )

    def test_mask_transform_applies_flip_resize_and_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.png"
            values = np.zeros((2, 4), dtype=np.uint8)
            values[:, :2] = 255
            Image.fromarray(values, mode="L").save(mask_path)
            transform = {
                "input_depth_shape": [2, 4],
                "target_depth_shape": [2, 4],
                "mesh_shape_before_crop": [4, 8],
                "crop_bbox_rc": [1, 2, 3, 6],
                "emitted_shape": [2, 4],
                "flip_x": True,
            }

            transformed = matrix._mask_on_emitted_grid(mask_path, transform, (2, 4))

            self.assertEqual(transformed.shape, (2, 4))
            self.assertFalse(np.any(transformed[:, :2]))
            self.assertTrue(np.all(transformed[:, 2:]))

    def test_variant_quality_fails_without_face_or_selected_detail_refinement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = root / "surface.npy"
            reference = root / "reference.npy"
            stl = root / "model.stl"
            np.save(surface, np.ones((4, 4), dtype=np.float32))
            np.save(reference, np.ones((4, 4), dtype=np.float32))
            stl.write_bytes(b"solid empty\nendsolid empty\n")
            response = {
                "job_id": "b" * 32,
                "face_refinement": {"applied": False},
                "relief_postprocess": {
                    "surface_appearance_agreement": {},
                    "background_depth_preservation": {},
                    "selection_background_physical_cap": {},
                },
                "stl_diagnostics": {},
            }
            with (
                patch.object(
                    matrix,
                    "_job_artifacts",
                    return_value={
                        "surface": surface,
                        "reference_surface": reference,
                        "stl": stl,
                        "refined_depth": surface,
                    },
                ),
                patch.object(
                    matrix,
                    "_stl_heightfield_agreement",
                    return_value={"passed": True},
                ),
                patch.object(
                    matrix,
                    "_exact_face_depth_quality",
                    return_value={"checks": {"passed": True}},
                ),
            ):
                quality = matrix._score_variant(
                    response,
                    server_output=root,
                    exact_depth_path=reference,
                    mask_path=stl,
                    require_occlusion=False,
                )

            self.assertFalse(quality["checks"]["refinement_applied"])
            self.assertFalse(quality["checks"]["validated_human_face_refined"])
            self.assertFalse(quality["checks"]["passed"])

    def test_exact_face_depth_quality_accepts_affine_match_and_rejects_reversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yy, xx = np.mgrid[:32, :32]
            exact = (0.1 * xx + 0.03 * yy + 0.002 * xx * yy).astype(np.float32)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[4:28, 4:28] = 255
            exact_path = root / "exact.npy"
            predicted_path = root / "predicted.npy"
            reversed_path = root / "reversed.npy"
            mask_path = root / "mask.png"
            np.save(exact_path, exact)
            np.save(predicted_path, exact * 2.0 + 3.0)
            np.save(reversed_path, -exact)
            Image.fromarray(mask, mode="L").save(mask_path)

            matched = matrix._exact_face_depth_quality(
                predicted_path, exact_path, mask_path
            )
            reversed_result = matrix._exact_face_depth_quality(
                reversed_path, exact_path, mask_path
            )

            self.assertTrue(matched["checks"]["passed"])
            self.assertFalse(reversed_result["checks"]["positive_orientation"])
            self.assertFalse(reversed_result["checks"]["passed"])

    def test_output_outside_server_ignored_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "server"
            outside = Path(temporary) / "elsewhere"
            with patch.object(matrix, "_require_ignored"):
                with self.assertRaisesRegex(ValueError, "beneath"):
                    matrix._prepare_output_directory(repository, outside)
            self.assertFalse(outside.exists())

    def test_dirty_checkout_and_runtime_revision_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "server"
            with patch.object(
                matrix,
                "_git_output",
                side_effect=[str(repository), REVISION, " M backend/main.py"],
            ):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    matrix._clean_server_provenance(repository)

            server_output = repository / "backend" / "output"
            output_dir = server_output / "matrix-test"
            client = FakeLiveClient(server_output.resolve(), revision="2" * 40)
            clean = {
                "available": True,
                "revision": REVISION,
                "clean": True,
                "status": [],
            }
            with (
                patch.object(matrix, "_clean_server_provenance", return_value=clean),
                patch.object(
                    matrix,
                    "_producer_provenance",
                    return_value={**clean, "files": []},
                ),
                patch.object(matrix, "_require_ignored"),
                patch.object(matrix, "load_makehuman_face_fixture") as fixture_loader,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    matrix.run(
                        repository, output_dir=output_dir, limit=1, client=client
                    )
            fixture_loader.assert_not_called()
            self.assertEqual([call[0] for call in client.calls], ["GET"])


if __name__ == "__main__":
    unittest.main()
