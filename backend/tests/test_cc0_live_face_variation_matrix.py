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
    @staticmethod
    def _complete_part_masks(silhouette: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(silhouette, dtype=bool).copy()
            for name in matrix.FACE_PART_NAMES
        }

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

    def test_varied_context_matrix_covers_mixed_semantics_and_scene_controls(self):
        coverage = matrix._matrix_coverage(matrix.VARIED_CONTEXT_MATRIX)

        self.assertEqual(coverage["scene_kinds"], ["face", "object"])
        self.assertGreaterEqual(len(coverage["background_profiles"]), 2)
        self.assertGreaterEqual(len(coverage["lighting_profiles"]), 3)
        self.assertEqual(len(coverage["occluded_rows"]), 1)
        self.assertTrue(coverage["unique_rows"])

    def test_varied_background_profiles_are_distinct_and_have_depth_span(self):
        shape = (96, 128)
        rendered_depth = np.full(shape, 0.4, dtype=np.float32)
        selection_mask = np.zeros(shape, dtype=bool)
        selection_mask[32:64, 48:80] = True
        selection_rgb = np.full((*shape, 3), 0.6, dtype=np.float32)

        shelves, _ = matrix._compose_scene(
            rendered_depth,
            selection_mask,
            selection_rgb,
            phase=0.7,
            background_profile="deep_shelves",
        )
        studio, _ = matrix._compose_scene(
            rendered_depth,
            selection_mask,
            selection_rgb,
            phase=0.7,
            background_profile="layered_studio",
        )
        background = ~selection_mask
        shelves_span = float(
            np.percentile(shelves[background], 98)
            - np.percentile(shelves[background], 2)
        )
        studio_span = float(
            np.percentile(studio[background], 98) - np.percentile(studio[background], 2)
        )

        self.assertGreater(shelves_span, 0.20)
        self.assertGreater(studio_span, 0.15)
        self.assertGreater(float(np.mean(np.abs(shelves - studio))), 0.03)

    def test_render_controls_shift_face_and_preserve_bounded_occluder_selection(self):
        size = 64
        silhouette = np.zeros((size, size), dtype=bool)
        silhouette[18:46, 10:22] = True
        rendered = SimpleNamespace(
            silhouette=silhouette,
            rgb=np.full((size, size, 3), 0.5, dtype=np.float32),
            depth=np.full((size, size), 0.4, dtype=np.float32),
            part_masks=self._complete_part_masks(silhouette),
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
            source, mask, exact_depth, part_masks, record = matrix._render_scene_arrays(
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
        self.assertEqual(set(part_masks), set(matrix.FACE_PART_NAMES))
        self.assertTrue(all(np.any(values) for values in part_masks.values()))

    def test_procedural_object_render_records_nonface_selection_and_background(self):
        size = 64
        silhouette = np.zeros((size, size), dtype=bool)
        silhouette[14:52, 11:49] = True
        rendered = SimpleNamespace(
            silhouette=silhouette,
            rgb=np.full((size, size, 3), 0.55, dtype=np.float32),
            depth=np.full((size, size), 0.35, dtype=np.float32),
        )
        spec = matrix.ObjectSceneSpec(
            row_id="object",
            procedural_index=5,
            target_dimension=size,
            camera_yaw_deg=27.0,
            camera_elevation_deg=-8.0,
            camera_distance=3.8,
            horizontal_offset=0.0625,
        )

        with (
            patch.object(matrix, "make_procedural_mesh", return_value=object()),
            patch.object(matrix, "render_mesh", return_value=rendered) as render,
        ):
            source, mask, exact_depth, part_masks, record = matrix._render_scene_arrays(
                spec, {}
            )

        camera = render.call_args.args[1]
        self.assertEqual(camera.azimuth_deg, 27.0)
        self.assertEqual(camera.elevation_deg, -8.0)
        self.assertEqual(record["scene_kind"], "object")
        self.assertEqual(record["procedural_index"], 5)
        self.assertGreater(record["visible_selection_pixels"], 0)
        self.assertGreater(record["background_depth_span"], 0.20)
        self.assertEqual(source.shape, (size, size, 3))
        self.assertEqual(mask.shape, (size, size))
        self.assertEqual(exact_depth.shape, (size, size))
        self.assertIsNone(part_masks)

    def test_procedural_object_fixture_is_bit_deterministic(self):
        spec = matrix.OBJECT_SMOKE_MATRIX[0]

        first = matrix._render_scene_arrays(spec, {})
        second = matrix._render_scene_arrays(spec, {})

        for first_array, second_array in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(first_array, second_array)
        self.assertIsNone(first[3])
        self.assertEqual(first[4], second[4])
        mask = first[1] >= 128
        selected_span = float(
            np.percentile(first[2][mask], 98) - np.percentile(first[2][mask], 2)
        )
        self.assertGreater(selected_span, 0.30)
        self.assertGreater(first[4]["background_depth_span"], 0.20)

    def test_object_only_run_skips_makehuman_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "server"
            server_output = repository / "backend" / "output"
            output_dir = server_output / "object-smoke"
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
                "scene_kind": "object",
                "visible_selection_pixels": 48,
                "background_depth_span": 0.2,
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
                    side_effect=AssertionError("object smoke must not load faces"),
                ),
                patch.object(
                    matrix,
                    "_render_scene_arrays",
                    return_value=(
                        source,
                        mask_values,
                        np.ones((16, 16), dtype=np.float32),
                        None,
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
                    specs=matrix.OBJECT_SMOKE_MATRIX,
                    client=client,
                )

            self.assertTrue(summary["checks"]["passed"])
            self.assertEqual(summary["matrix"]["executed_rows"], 1)
            self.assertEqual(summary["matrix"]["coverage"]["scene_kinds"], ["object"])
            self.assertEqual(
                summary["fixture"]["source"],
                "deterministic_generated_procedural_mesh",
            )

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

    def test_eye_band_occluder_is_anchored_to_rendered_eyes(self):
        size = 64
        silhouette = np.zeros((size, size), dtype=bool)
        silhouette[10:54, 8:56] = True
        left_eye = np.zeros_like(silhouette)
        right_eye = np.zeros_like(silhouette)
        left_eye[22:32, 32:45] = True
        right_eye[22:32, 18:31] = True
        rendered_parts = self._complete_part_masks(silhouette)
        rendered_parts.update({"left_eye": left_eye, "right_eye": right_eye})
        rendered = SimpleNamespace(
            silhouette=silhouette,
            rgb=np.full((size, size, 3), 0.7, dtype=np.float32),
            depth=np.full((size, size), 0.4, dtype=np.float32),
            part_masks=rendered_parts,
        )
        fixture = {
            "profiles": {
                "profile": {
                    "mesh": SimpleNamespace(
                        vertices=np.zeros((3, 3), dtype=np.float64)
                    ),
                    "skin_tone": (0.5, 0.4, 0.3),
                }
            },
            "part_weights": {},
            "surface_weights": {},
        }
        spec = matrix.FaceSceneSpec(
            row_id="eye_band",
            profile_name="profile",
            target_dimension=size,
            camera_yaw_deg=0.0,
            camera_distance=4.0,
            occluder=matrix.OccluderSpec(0.39, 0.49, 0.61, 0.62, anchor="eye_band"),
        )

        with (
            patch.object(
                matrix, "make_profile_vertex_colors", return_value=np.zeros((3, 3))
            ),
            patch.object(matrix, "render_mesh", return_value=rendered),
        ):
            source, mask, _exact_depth, part_masks, record = matrix._render_scene_arrays(
                spec, fixture
            )

        top, bottom, left, right = record["occluder_bounds_tblr"]
        self.assertEqual(record["occluder_anchor"], "eye_band")
        self.assertLessEqual(top, 27)
        self.assertGreaterEqual(bottom, 27)
        self.assertLess(left, 18)
        self.assertGreater(right, 45)
        self.assertTrue(np.all(mask[top:bottom, left:right] > 0))
        self.assertTrue(np.all(source[top:bottom, left:right] == (47, 71, 83)))
        self.assertEqual(set(part_masks), set(matrix.FACE_PART_NAMES))

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
                "background_depth_span": 0.2,
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
                        self._complete_part_masks(mask_values > 0),
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
            for name in matrix.FACE_PART_NAMES:
                self.assertTrue(
                    (row_dir / "exact_face_parts" / f"{name}.png").is_file()
                )
            self.assertRegex(summary["rows"][0]["source"]["sha256"], r"^[a-f0-9]{64}$")
            self.assertTrue(summary["rows"][0]["exact_face_part_masks"]["complete"])
            self.assertEqual(
                set(summary["rows"][0]["exact_face_part_masks"]["files"]),
                set(matrix.FACE_PART_NAMES),
            )

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
                patch.object(
                    matrix,
                    "_emitted_face_part_retention",
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

    def test_object_quality_requires_generic_route_and_rejects_false_face(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = root / "surface.npy"
            reference = root / "reference.npy"
            stl = root / "model.stl"
            np.save(surface, np.ones((4, 4), dtype=np.float32))
            np.save(reference, np.ones((4, 4), dtype=np.float32))
            stl.write_bytes(b"solid empty\nendsolid empty\n")
            spec = matrix.ObjectSceneSpec(
                row_id="object",
                procedural_index=5,
                target_dimension=128,
                camera_yaw_deg=0.0,
                camera_distance=3.8,
            )
            base_response = {
                "job_id": "b" * 32,
                "face_refinement": {
                    "applied": True,
                    "detected_faces": 0,
                    "refined_faces": 0,
                    "selection_detail_fallback_regions": 1,
                    "refined_selection_detail_regions": 1,
                    "refined_regions_total": 1,
                },
                "relief_postprocess": {
                    "surface_appearance_agreement": {
                        "selection_nonface": {},
                        "background": {},
                    },
                    "background_depth_preservation": {},
                    "selection_background_physical_cap": {},
                },
                "stl_diagnostics": {},
            }
            artifacts = {
                "surface": surface,
                "reference_surface": reference,
                "stl": stl,
                "refined_depth": surface,
            }
            with (
                patch.object(matrix, "_job_artifacts", return_value=artifacts),
                patch.object(
                    matrix,
                    "_appearance_checks",
                    return_value={"passed": True},
                ),
                patch.object(
                    matrix,
                    "_independent_background_checks",
                    return_value={"passed": True},
                ),
                patch.object(
                    matrix,
                    "_independent_cap_checks",
                    return_value={"passed": True},
                ),
                patch.object(
                    matrix,
                    "_boundary_shape_metrics",
                    return_value={"passed": True},
                ),
                patch.object(
                    matrix,
                    "_topology_record",
                    return_value={"checks": {"passed": True}},
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
                accepted = matrix._score_variant(
                    base_response,
                    server_output=root,
                    exact_depth_path=reference,
                    mask_path=stl,
                    require_occlusion=False,
                    spec=spec,
                )
                false_face_response = json.loads(json.dumps(base_response))
                false_face_response["face_refinement"]["detected_faces"] = 1
                false_face_response["face_refinement"]["refined_faces"] = 1
                rejected = matrix._score_variant(
                    false_face_response,
                    server_output=root,
                    exact_depth_path=reference,
                    mask_path=stl,
                    require_occlusion=False,
                    spec=spec,
                )
                detected_only_response = json.loads(json.dumps(base_response))
                detected_only_response["face_refinement"]["detected_faces"] = 1
                detected_only = matrix._score_variant(
                    detected_only_response,
                    server_output=root,
                    exact_depth_path=reference,
                    mask_path=stl,
                    require_occlusion=False,
                    spec=spec,
                )

            self.assertTrue(accepted["checks"]["passed"])
            self.assertTrue(accepted["checks"]["generic_selection_refined"])
            self.assertTrue(accepted["checks"]["no_false_human_face"])
            self.assertFalse(rejected["checks"]["subject_refinement_route"])
            self.assertFalse(rejected["checks"]["no_false_human_face"])
            self.assertFalse(rejected["checks"]["passed"])
            self.assertFalse(detected_only["checks"]["subject_refinement_route"])
            self.assertFalse(detected_only["checks"]["no_false_human_face"])
            self.assertFalse(detected_only["checks"]["passed"])

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
                predicted_path,
                exact_path,
                mask_path,
                expected_scale_sign=1.0,
                require_face_parts=False,
            )
            reversed_result = matrix._exact_face_depth_quality(
                reversed_path,
                exact_path,
                mask_path,
                expected_scale_sign=1.0,
                require_face_parts=False,
            )

            self.assertTrue(matched["checks"]["passed"])
            self.assertFalse(reversed_result["checks"]["depth_semantics_orientation"])
            self.assertFalse(reversed_result["checks"]["passed"])

    def test_exact_face_parts_reject_one_flattened_eye(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size = 96
            yy, xx = np.mgrid[:size, :size]
            face = ((xx - 48.0) / 34.0) ** 2 + ((yy - 48.0) / 42.0) ** 2 <= 1.0
            signal = (
                0.20
                + 0.002 * xx
                + 0.001 * yy
                + 0.45 * np.exp(-((xx - 48.0) ** 2 + (yy - 49.0) ** 2) / 180.0)
                + 0.06 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
            )
            signal = (signal - signal.min()) / (signal.max() - signal.min())
            exact = (1.0 - signal).astype(np.float32)
            predicted = (exact * 1.7 + 0.2).astype(np.float32)

            def region(x0, y0, x1, y1):
                values = np.zeros((size, size), dtype=bool)
                values[y0:y1, x0:x1] = True
                return values & face

            parts = {
                "left_eye": region(28, 34, 43, 45),
                "right_eye": region(53, 34, 68, 45),
                "left_eyebrow": region(27, 27, 44, 34),
                "right_eyebrow": region(52, 27, 69, 34),
                "nose": region(42, 39, 55, 63),
                "mouth": region(36, 65, 61, 76),
            }
            mask_path = root / "face.png"
            exact_path = root / "exact.npy"
            predicted_path = root / "predicted.npy"
            damaged_path = root / "damaged.npy"
            Image.fromarray(face.astype(np.uint8) * 255, mode="L").save(mask_path)
            np.save(exact_path, exact)
            np.save(predicted_path, predicted)
            damaged = predicted.copy()
            damaged[parts["left_eye"]] = float(np.median(predicted[face]))
            np.save(damaged_path, damaged)
            part_paths = {}
            for name, values in parts.items():
                path = root / f"{name}.png"
                Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)
                part_paths[name] = path

            matched = matrix._exact_face_depth_quality(
                predicted_path,
                exact_path,
                mask_path,
                expected_scale_sign=1.0,
                part_mask_paths=part_paths,
            )
            damaged_result = matrix._exact_face_depth_quality(
                damaged_path,
                exact_path,
                mask_path,
                expected_scale_sign=1.0,
                part_mask_paths=part_paths,
            )

            self.assertTrue(matched["checks"]["passed"])
            self.assertFalse(damaged_result["checks"]["named_part_shape"])
            self.assertIn(
                "left_eye",
                damaged_result["named_part_shape"]["failed_parts"],
            )
            self.assertFalse(damaged_result["checks"]["passed"])

    def test_emitted_face_part_retention_rejects_local_flattening(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size = 64
            yy, xx = np.mgrid[:size, :size]
            face = np.zeros((size, size), dtype=bool)
            face[6:58, 7:57] = True
            reference = (
                3.0
                + 0.03 * xx
                + 0.02 * yy
                + 2.5 * np.exp(-((xx - 32.0) ** 2 + (yy - 32.0) ** 2) / 130.0)
                + 0.4 * np.sin(xx / 4.0) * np.cos(yy / 6.0)
            ).astype(np.float64)
            boxes = {
                "left_eye": (15, 20, 28, 29),
                "right_eye": (36, 20, 49, 29),
                "left_eyebrow": (14, 14, 29, 20),
                "right_eyebrow": (35, 14, 50, 20),
                "nose": (27, 27, 38, 43),
                "mouth": (23, 45, 42, 53),
            }
            mask_path = root / "face.png"
            Image.fromarray(face.astype(np.uint8) * 255, mode="L").save(mask_path)
            part_paths = {}
            part_values = {}
            for name, (x0, y0, x1, y1) in boxes.items():
                values = np.zeros_like(face)
                values[y0:y1, x0:x1] = True
                values &= face
                path = root / f"{name}.png"
                Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)
                part_paths[name] = path
                part_values[name] = values
            transform = {
                "input_depth_shape": [size, size],
                "target_depth_shape": [size, size],
                "mesh_shape_before_crop": [size, size],
                "crop_bbox_rc": [0, 0, size, size],
                "emitted_shape": [size, size],
                "flip_x": False,
            }
            postprocess = {
                "surface_grid_transform": transform,
                "mesh_sample_pitch_mm": 0.5,
            }
            perfect = matrix._emitted_face_part_retention(
                reference,
                reference.copy(),
                mask_path,
                part_paths,
                postprocess,
                required=True,
            )
            damaged = reference.copy()
            damaged[part_values["left_eye"]] = float(np.median(reference[face]))
            rejected = matrix._emitted_face_part_retention(
                reference,
                damaged,
                mask_path,
                part_paths,
                postprocess,
                required=True,
            )

            self.assertTrue(perfect["checks"]["passed"])
            self.assertFalse(rejected["checks"]["passed"])
            self.assertIn("left_eye", rejected["named_part_shape"]["failed_parts"])

    def test_occlusion_gate_accepts_correction_or_measured_consistency(self):
        consistent = matrix._occlusion_handling(
            {
                "eyewear_deoccluded_faces": 0,
                "faces": [
                    {
                        "eyewear_deocclusion": {
                            "enabled": False,
                            "reason": "source_depth_already_consistent",
                            "detection": {"enabled": True},
                        }
                    }
                ],
            },
            required=True,
        )
        corrected = matrix._occlusion_handling(
            {
                "eyewear_deoccluded_faces": 1,
                "faces": [
                    {
                        "eyewear_deocclusion": {
                            "enabled": True,
                            "detection": {"enabled": True},
                        }
                    }
                ],
            },
            required=True,
        )
        missed = matrix._occlusion_handling(
            {"eyewear_deoccluded_faces": 0, "faces": []}, required=True
        )

        self.assertTrue(consistent["passed"])
        self.assertEqual(consistent["already_consistent_faces"], 1)
        self.assertTrue(corrected["passed"])
        self.assertFalse(missed["passed"])

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
