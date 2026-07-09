import csv
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from backend.benchmark.backfill_surface_metrics import backfill_surface_metrics
from backend.benchmark.backfill_lora_provenance import backfill_lora_root, backfill_split_audit
from backend.benchmark import colab_g4_orchestrator, run_completion_benchmark
from backend.benchmark.combine_optimize_runs import combine_runs
from backend.benchmark.compare_optimize_runs import add_score_deltas, compare_run, render_markdown
from backend.benchmark.direct_mesh import direct_mesh_input_path, mesh_is_printable_volume, repair_mesh_for_printable_stl
from backend.benchmark.export_training_pairs import main as export_training_pairs_main
from backend.benchmark.generate_rendered_dataset import attach_multiview_fields
from backend.benchmark.package_colab_inputs import package_inputs
from backend.benchmark.optimize_completion import (
    annotate_per_sample_metrics,
    experiment_metadata,
    load_experiments,
    training_metadata,
    write_experiment_report,
    write_modern_cache_preflight,
    write_selection_decision,
)
from backend.benchmark.preflight_modern_providers import provider_rows
from backend.benchmark.make_artifact_contact_sheet import (
    depth_error_image,
    filter_rows,
    make_contact_sheet,
    reference_artifacts,
    stl_preview_image,
)
from backend.benchmark.metrics import mesh_surface_distance_metrics, surface_distance_metrics
from backend.benchmark.report_run import baseline_delta_rows, paired_baseline_delta_rows, paired_objective_rows, render_report
from backend.benchmark.rank_methods import parse_weights, rank_summary_rows
from backend.benchmark.run_image_to_mesh_provider import main as run_image_to_mesh_provider_main
from backend.benchmark.run_stl_first_smoke import build_experiments as build_stl_first_experiments, build_optimize_command as build_stl_first_optimize_command
from backend.benchmark.run_triposr_repair_smoke import build_experiments, rows_from_csv
from backend.benchmark.run_completion_benchmark import evaluate_sample, run_one, stl_diagnostics, summarize, write_split_audit
from backend.benchmark.select_completion_candidate import evaluate_selection, json_safe
from backend.benchmark.train_inpainting_lora import InpaintPairDataset, collate, dry_run, read_metadata, weighted_mse_loss
from backend.benchmark.weight_training_pairs import weight_metadata_rows
from backend.pic_to_3d import (
    _diagnostic_signed_volume,
    _force_positive_stl_volume,
    _inpaint_torch_dtype,
    _masked_edit_image,
    _qwen_edit_prompt,
    complete_image,
    depth_data_to_3d_model,
)


class StlExportRegressionTests(unittest.TestCase):
    def test_force_positive_stl_volume_flips_negative_winding_once(self):
        import trimesh
        from stl import mesh

        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        reversed_triangles = box.vertices[box.faces][:, [0, 2, 1], :]
        self.assertLess(_diagnostic_signed_volume(reversed_triangles), 0)

        stl_mesh = mesh.Mesh(np.zeros(len(reversed_triangles), dtype=mesh.Mesh.dtype))
        stl_mesh.vectors[:] = reversed_triangles
        _force_positive_stl_volume(stl_mesh)
        fixed_volume = _diagnostic_signed_volume(stl_mesh.vectors)
        _force_positive_stl_volume(stl_mesh)

        self.assertGreater(fixed_volume, 0)
        self.assertAlmostEqual(_diagnostic_signed_volume(stl_mesh.vectors), fixed_volume)

    def test_minimal_depth_export_is_watertight_with_positive_signed_volume(self):
        depth = np.array(
            [
                [0.10, 0.30],
                [0.20, 0.60],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            depth_path = temp_path / "depth.npy"
            stl_path = temp_path / "model.stl"
            np.save(depth_path, depth)

            depth_data_to_3d_model(
                str(depth_path),
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=10,
                invert=False,
                sigma=0,
            )
            diagnostics = stl_diagnostics(stl_path)

        self.assertTrue(diagnostics["stl_exists"])
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertEqual(diagnostics["stl_nonmanifold_edge_count"], 0)
        self.assertEqual(diagnostics["stl_degenerate_face_count"], 0)
        self.assertEqual(diagnostics["stl_degenerate_face_ratio"], 0.0)
        self.assertTrue(diagnostics["stl_winding_consistent"])
        self.assertTrue(diagnostics["stl_positive_volume"])
        self.assertEqual(diagnostics["stl_component_count"], 1)
        self.assertGreater(diagnostics["stl_volume"], 0)
        self.assertEqual(diagnostics["stl_faces"], 12)
        self.assertGreater(diagnostics["stl_z_range"], 0)
        self.assertGreater(diagnostics["stl_bbox_min_dimension"], 0)
        self.assertGreaterEqual(diagnostics["stl_bbox_aspect_ratio"], 1.0)
        self.assertGreater(diagnostics["stl_faces_per_bbox_volume"], 0)
        self.assertGreater(diagnostics["stl_faces_per_bbox_volume_log1p"], 0)

    def test_stl_diagnostics_flags_flat_bbox_as_non_printable(self):
        import warnings

        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "flat.stl"
            mesh = trimesh.Trimesh(
                vertices=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ]
                ),
                faces=np.array([[0, 1, 2], [1, 3, 2]]),
                process=False,
            )
            mesh.export(stl_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                diagnostics = stl_diagnostics(stl_path)

        self.assertEqual(diagnostics["stl_bbox_min_dimension"], 0.0)
        self.assertFalse(diagnostics["stl_bbox_has_volume"])
        self.assertTrue(np.isinf(diagnostics["stl_bbox_aspect_ratio"]))
        self.assertFalse(diagnostics["stl_is_volume"])
        self.assertFalse(diagnostics["stl_is_manifold"])
        self.assertGreater(diagnostics["stl_nonmanifold_edge_count"], 0)

    def test_mesh_surface_distance_is_zero_for_same_mesh(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = Path(temp_dir) / "box.ply"
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(mesh_path)
            metrics = mesh_surface_distance_metrics(mesh_path, mesh_path, max_points=128)

        self.assertAlmostEqual(metrics["chamfer_l1"], 0.0)
        self.assertAlmostEqual(metrics["chamfer_rmse"], 0.0)
        self.assertAlmostEqual(metrics["hausdorff95"], 0.0)
        self.assertGreater(metrics["point_count"], 0)

    def test_mesh_surface_distance_tolerates_different_triangulation(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coarse_path = root / "coarse_box.ply"
            fine_path = root / "fine_box.ply"
            coarse = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            fine = coarse.subdivide().subdivide()
            coarse.export(coarse_path)
            fine.export(fine_path)
            metrics = mesh_surface_distance_metrics(coarse_path, fine_path, max_points=2048)

        self.assertLess(metrics["chamfer_l1"], 0.04)
        self.assertLess(metrics["hausdorff95"], 0.09)

    def test_mesh_surface_distance_applies_reference_camera(self):
        import trimesh
        from backend.benchmark.mesh_rendering import load_mesh, mesh_in_render_frame

        camera = {"azimuth_deg": 37.0, "elevation_deg": 14.0, "roll_deg": 5.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_path = root / "reference_box.ply"
            prediction_path = root / "view_box.ply"
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(reference_path)
            mesh_in_render_frame(load_mesh(reference_path), camera).export(prediction_path)

            framed = mesh_surface_distance_metrics(
                reference_path,
                prediction_path,
                max_points=256,
                reference_camera=camera,
            )
            unframed = mesh_surface_distance_metrics(reference_path, prediction_path, max_points=256)

        self.assertAlmostEqual(framed["chamfer_l1"], 0.0)
        self.assertGreater(unframed["chamfer_l1"], 0.01)

    def test_load_mesh_applies_scene_graph_transforms(self):
        import trimesh
        from backend.benchmark.mesh_rendering import load_mesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "translated.glb"
            scene = trimesh.Scene()
            scene.add_geometry(
                trimesh.creation.box(extents=(1.0, 0.75, 0.5)),
                node_name="translated_box",
                transform=trimesh.transformations.translation_matrix([3.0, 0.0, 0.0]),
            )
            scene.export(scene_path)
            loaded = load_mesh(scene_path)

        center = loaded.bounds.mean(axis=0)
        self.assertGreater(center[0], 2.5)

    def test_source_mesh_oracle_emits_stl_without_depth(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            mesh_path = root / "box.ply"
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(mesh_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "box",
                        "full_image": str(full),
                        "masked_image": str(masked),
                        "mask": str(mask),
                        "mesh": str(mesh_path),
                        "source": "unit",
                        "camera": {"azimuth_deg": 37.0, "elevation_deg": 14.0, "roll_deg": 5.0},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "run"

            with patch.object(
                sys,
                "argv",
                [
                    "run_completion_benchmark",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "source-mesh-oracle",
                    "--limit",
                    "1",
                    "--skip-depth",
                    "--emit-stl",
                    "--mesh-surface-max-points",
                    "128",
                ],
            ):
                run_completion_benchmark.main()

            with (output_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            with (output_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                summary = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["method"], "source-mesh-oracle")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertEqual(rows[0]["stl_is_volume"], "True")
        self.assertEqual(rows[0]["stl_single_component"], "True")
        self.assertIn("output_model.stl", rows[0]["stl_model"])
        self.assertAlmostEqual(float(rows[0]["mesh_surface_chamfer_l1"]), 0.0)
        self.assertEqual(summary[0]["n"], "1")
        self.assertAlmostEqual(float(summary[0]["mesh_surface_chamfer_l1_median"]), 0.0)

    def test_external_image_to_mesh_command_emits_stl_without_depth(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            mesh_path = root / "reference_box.ply"
            script_path = root / "fake_image_to_mesh.py"
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(mesh_path)
            script_path.write_text(
                "\n".join(
                    [
                        "import sys",
                        "import trimesh",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(sys.argv[1])",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(sys.argv[2])",
                    ]
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "box",
                        "full_image": str(full),
                        "masked_image": str(masked),
                        "mask": str(mask),
                        "mesh": str(mesh_path),
                        "source": "unit",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "run"
            command = f'"{sys.executable}" "{script_path}" "{{output_mesh}}" "{{output_stl}}"'

            with patch.object(
                sys,
                "argv",
                [
                    "run_completion_benchmark",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "external-image-to-mesh",
                    "--limit",
                    "1",
                    "--skip-depth",
                    "--emit-stl",
                    "--direct-mesh-command",
                    command,
                    "--direct-mesh-output-ext",
                    "ply",
                    "--mesh-surface-max-points",
                    "128",
                ],
            ):
                run_completion_benchmark.main()

            with (output_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["method"], "external-image-to-mesh")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertIn("output_model.stl", rows[0]["stl_model"])
        self.assertIn("output_mesh.ply", rows[0]["direct_mesh_output_mesh"])
        self.assertAlmostEqual(float(rows[0]["mesh_surface_chamfer_l1"]), 0.0)

    def test_external_multiview_to_mesh_command_receives_bundle(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            view2 = root / "view2.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            mask2 = root / "mask2.png"
            mesh_path = root / "reference_box.ply"
            script_path = root / "fake_multiview_to_mesh.py"
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (90, 130, 170)).save(view2)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            Image.fromarray(np.ones((12, 12), dtype=np.uint8) * 255).save(mask2)
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(mesh_path)
            script_path.write_text(
                "\n".join(
                    [
                        "import json, sys",
                        "from pathlib import Path",
                        "import trimesh",
                        "bundle = json.loads(Path(sys.argv[1]).read_text())",
                        "assert bundle['sample_id'] == 'box'",
                        "assert len(bundle['views']) == 2",
                        "assert bundle['views'][1]['camera']['azimuth_deg'] == 45",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(sys.argv[2])",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(sys.argv[3])",
                    ]
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "box",
                        "full_image": str(full),
                        "masked_image": str(masked),
                        "mask": str(mask),
                        "mesh": str(mesh_path),
                        "source": "unit",
                        "multiview_images": [str(full), str(view2)],
                        "multiview_masks": [str(mask), str(mask2)],
                        "multiview_cameras": [{"azimuth_deg": 0}, {"azimuth_deg": 45}],
                        "multiview_view_ids": ["box_v0", "box_v1"],
                        "multiview_primary_index": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "run"
            command = f'"{sys.executable}" "{script_path}" "{{input_bundle}}" "{{output_mesh}}" "{{output_stl}}"'

            with patch.object(
                sys,
                "argv",
                [
                    "run_completion_benchmark",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "external-multiview-to-mesh",
                    "--limit",
                    "1",
                    "--skip-depth",
                    "--emit-stl",
                    "--direct-mesh-command",
                    command,
                    "--direct-mesh-output-ext",
                    "ply",
                    "--mesh-surface-max-points",
                    "128",
                ],
            ):
                run_completion_benchmark.main()

            with (output_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            bundle_path = Path(rows[0]["direct_mesh_input_bundle"])
            bundle_exists = bundle_path.exists()
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["method"], "external-multiview-to-mesh")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertTrue(bundle_exists)
        self.assertEqual(len(bundle["views"]), 2)
        self.assertEqual(bundle["views"][1]["sample_id"], "box_v1")
        self.assertAlmostEqual(float(rows[0]["mesh_surface_chamfer_l1"]), 0.0)

    def test_direct_mesh_input_can_use_geometry_prefill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            Image.new("RGB", (8, 4), (10, 20, 30)).save(full)
            masked_image = Image.new("RGB", (8, 4), (10, 20, 30))
            masked_image.paste(Image.new("RGB", (4, 4), (255, 255, 255)), (4, 0))
            masked_image.save(masked)
            mask_array = np.zeros((4, 8), dtype=np.uint8)
            mask_array[:, 4:] = 255
            Image.fromarray(mask_array, mode="L").save(mask)

            sample = {
                "id": "prefill",
                "full_image": str(full),
                "masked_image": str(masked),
                "mask": str(mask),
            }
            prefilled = direct_mesh_input_path(sample, "mirror", root / "direct")
            image = Image.open(prefilled).convert("RGB")

        self.assertEqual(prefilled.name, "direct_mesh_input_mirror.png")
        self.assertEqual(image.getpixel((6, 1)), (10, 20, 30))
        self.assertNotEqual(image.getpixel((6, 1)), (255, 255, 255))

    def test_rendered_dataset_attaches_multiview_fields_by_asset(self):
        rows = [
            {
                "id": "asset_v1",
                "asset_key": "asset",
                "view_index": 1,
                "full_image": "v1.png",
                "gt_silhouette": "v1_mask.png",
                "camera": {"azimuth_deg": 90},
            },
            {
                "id": "asset_v0",
                "asset_key": "asset",
                "view_index": 0,
                "full_image": "v0.png",
                "gt_silhouette": "v0_mask.png",
                "camera": {"azimuth_deg": 0},
            },
            {
                "id": "solo",
                "asset_key": "solo",
                "view_index": 0,
                "full_image": "solo.png",
                "camera": {},
            },
        ]

        attach_multiview_fields(rows)

        self.assertEqual(rows[0]["multiview_images"], ["v0.png", "v1.png"])
        self.assertEqual(rows[0]["multiview_masks"], ["v0_mask.png", "v1_mask.png"])
        self.assertEqual(rows[0]["multiview_cameras"][1]["azimuth_deg"], 90)
        self.assertEqual(rows[0]["multiview_primary_index"], 1)
        self.assertNotIn("multiview_images", rows[2])

    def test_package_inputs_rewrites_multiview_path_lists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("full.png", "masked.png", "mask.png", "view0.png", "view1.png", "view0_mask.png", "view1_mask.png"):
                Image.new("RGB", (4, 4), (1, 2, 3)).save(root / name)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "full_image": str(root / "full.png"),
                        "masked_image": str(root / "masked.png"),
                        "mask": str(root / "mask.png"),
                        "multiview_images": [str(root / "view0.png"), str(root / "view1.png")],
                        "multiview_masks": [str(root / "view0_mask.png"), str(root / "view1_mask.png")],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=output,
                extract_root="/content/test_bundle",
                root=root,
            )
            with tarfile.open(output, "r:gz") as tar:
                rewritten = json.loads(tar.extractfile("inputs/manifest.jsonl").read().decode("utf-8").strip())

        self.assertEqual(report["referenced_files"], 7)
        self.assertEqual(len(rewritten["multiview_images"]), 2)
        self.assertTrue(rewritten["multiview_images"][0].startswith("/content/test_bundle/inputs/files/"))
        self.assertTrue(rewritten["multiview_masks"][1].startswith("/content/test_bundle/inputs/files/"))

    def test_image_to_mesh_provider_wrapper_normalizes_repo_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_spar3d"
            provider_dir.mkdir()
            input_image = root / "input.png"
            output_mesh = root / "normalized.glb"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (12, 12), (120, 80, 160)).save(input_image)
            (provider_dir / "run.py").write_text(
                "\n".join(
                    [
                        "import argparse",
                        "from pathlib import Path",
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('input_image')",
                        "parser.add_argument('--output-dir', required=True)",
                        "parser.add_argument('--low-vram-mode', action='store_true')",
                        "parser.add_argument('--remesh_option', default=None)",
                        "parser.add_argument('--device', default=None)",
                        "parser.add_argument('--custom-flag', action='store_true')",
                        "args = parser.parse_args()",
                        "assert args.custom_flag",
                        "Path(args.output_dir).mkdir(parents=True, exist_ok=True)",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(Path(args.output_dir) / 'result.glb')",
                        "trimesh.PointCloud([[0, 0, 0], [1, 1, 1]]).export(Path(args.output_dir) / 'points.ply')",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "spar3d",
                    "--provider-dir",
                    str(provider_dir),
                    "--input-image",
                    str(input_image),
                    "--output-mesh",
                    str(output_mesh),
                    "--output-stl",
                    str(output_stl),
                    "--provider-python",
                    sys.executable,
                    "--low-vram",
                    "--provider-device",
                    "cuda",
                    "--remesh-option",
                    "triangle",
                    "--provider-arg=--custom-flag",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_printable_mesh_repair_falls_back_to_watertight_hull(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            broken_path = root / "open_box.ply"
            repaired_path = root / "repaired.stl"
            broken = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            keep_faces = np.ones(len(broken.faces), dtype=bool)
            keep_faces[-2:] = False
            broken.update_faces(keep_faces)
            broken.export(broken_path)

            repair_mesh_for_printable_stl(broken_path, repaired_path, mode="printable")

            diagnostics = stl_diagnostics(repaired_path)

        self.assertTrue(diagnostics["stl_exists"])
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertEqual(diagnostics["stl_degenerate_face_count"], 0)
        self.assertTrue(diagnostics["stl_single_component"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_printable_mesh_gate_rejects_degenerate_faces(self):
        tetra_faces = np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [1, 3, 2],
                [2, 3, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )
        mesh = SimpleNamespace(
            vertices=np.zeros((4, 3), dtype=np.float64),
            faces=tetra_faces,
            extents=np.array([1.0, 1.0, 1.0]),
            area_faces=np.array([0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float64),
            volume=1.0,
            is_watertight=True,
            is_volume=True,
            is_winding_consistent=True,
            split=lambda only_watertight=False: [object()],
        )

        self.assertFalse(mesh_is_printable_volume(mesh))

    def test_image_to_mesh_provider_wrapper_can_repair_unprintable_mesh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_spar3d"
            provider_dir.mkdir()
            input_image = root / "input.png"
            output_mesh = root / "normalized.ply"
            raw_output_mesh = root / "raw_provider_mesh.ply"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (12, 12), (120, 80, 160)).save(input_image)
            (provider_dir / "run.py").write_text(
                "\n".join(
                    [
                        "import argparse",
                        "from pathlib import Path",
                        "import numpy as np",
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('input_image')",
                        "parser.add_argument('--output-dir', required=True)",
                        "args = parser.parse_args()",
                        "Path(args.output_dir).mkdir(parents=True, exist_ok=True)",
                        "mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))",
                        "keep_faces = np.ones(len(mesh.faces), dtype=bool)",
                        "keep_faces[-2:] = False",
                        "mesh.update_faces(keep_faces)",
                        "mesh.export(Path(args.output_dir) / 'broken.ply')",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "spar3d",
                    "--provider-dir",
                    str(provider_dir),
                    "--input-image",
                    str(input_image),
                    "--output-mesh",
                    str(output_mesh),
                    "--raw-output-mesh",
                    str(raw_output_mesh),
                    "--output-stl",
                    str(output_stl),
                    "--provider-python",
                    sys.executable,
                    "--mesh-repair",
                    "printable",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            raw_output_mesh_exists = raw_output_mesh.exists()
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(raw_output_mesh_exists)
        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertEqual(diagnostics["stl_degenerate_face_count"], 0)
        self.assertTrue(diagnostics["stl_single_component"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_triposr_api_provider_uses_preprocessed_input_without_rembg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_triposr"
            package_dir = provider_dir / "tsr"
            package_dir.mkdir(parents=True)
            input_image = root / "input.png"
            output_mesh = root / "triposr.obj"
            output_stl = root / "triposr.stl"
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "system.py").write_text(
                "\n".join(
                    [
                        "import sys",
                        "import trimesh",
                        "assert 'rembg' in sys.modules",
                        "class Renderer:",
                        "    def __init__(self):",
                        "        self.chunk_size = None",
                        "    def set_chunk_size(self, chunk_size):",
                        "        self.chunk_size = chunk_size",
                        "class FakeMesh:",
                        "    def export(self, path):",
                        "        trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(path)",
                        "class TSR:",
                        "    def __init__(self):",
                        "        self.renderer = Renderer()",
                        "    @classmethod",
                        "    def from_pretrained(cls, model_name, config_name, weight_name):",
                        "        assert model_name == 'unit/triposr'",
                        "        assert config_name == 'config.yaml'",
                        "        assert weight_name == 'model.ckpt'",
                        "        return cls()",
                        "    def to(self, device):",
                        "        self.device = device",
                        "    def __call__(self, images, device):",
                        "        assert device == 'cpu'",
                        "        assert len(images) == 1",
                        "        assert images[0].mode == 'RGB'",
                        "        assert images[0].getpixel((0, 0)) == (191, 63, 63)",
                        "        assert self.renderer.chunk_size == 123",
                        "        return ['scene']",
                        "    def extract_mesh(self, scene_codes, has_vertex_color, resolution):",
                        "        assert scene_codes == ['scene']",
                        "        assert has_vertex_color is True",
                        "        assert resolution == 32",
                        "        return [FakeMesh()]",
                    ]
                ),
                encoding="utf-8",
            )
            rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
            rgba.save(input_image)

            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "triposr-api",
                    "--provider-dir",
                    str(provider_dir),
                    "--input-image",
                    str(input_image),
                    "--output-mesh",
                    str(output_mesh),
                    "--output-stl",
                    str(output_stl),
                    "--provider-device",
                    "cpu",
                    "--chunk-size",
                    "123",
                    "--mc-resolution",
                    "32",
                    "--model-name",
                    "unit/triposr",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_triposr_repair_smoke_config_compares_raw_and_repaired_outputs(self):
        args = SimpleNamespace(
            triposr_python="/content/triposr-venv/bin/python",
            provider_dir="/content/TripoSR",
            provider_device="cuda",
            chunk_size=123,
            mc_resolution=32,
            mesh_repair="printable",
            direct_mesh_timeout=456,
        )

        experiments = build_experiments(args)
        by_name = {experiment["name"]: experiment for experiment in experiments}
        raw_command = by_name["triposr_api_masked_direct_mesh"]["direct_mesh_command"]
        repaired_command = by_name["triposr_api_masked_repaired_direct_mesh"]["direct_mesh_command"]

        self.assertEqual([experiment["name"] for experiment in experiments[:3]], ["masked", "mirror", "biharmonic"])
        self.assertIn("--provider triposr-api", raw_command)
        self.assertIn("--provider-dir /content/TripoSR", raw_command)
        self.assertIn("--chunk-size 123", raw_command)
        self.assertIn("--mc-resolution 32", raw_command)
        self.assertNotIn("--mesh-repair", raw_command)
        self.assertIn("--raw-output-mesh", repaired_command)
        self.assertIn("{output_dir}/output_mesh_raw.obj", repaired_command)
        self.assertIn("--mesh-repair printable", repaired_command)
        self.assertEqual(by_name["triposr_api_masked_repaired_direct_mesh"]["direct_mesh_timeout"], 456)

    def test_stl_first_smoke_builds_direct_and_multiview_candidates(self):
        args = SimpleNamespace(
            include_source_oracle=True,
            include_triposr_api=True,
            include_raw_direct_mesh=False,
            include_hunyuan3d_shape=True,
            multiview_command='python mv.py "{input_bundle}" "{output_mesh}" "{output_stl}"',
            multiview_name="mv_recon",
            multiview_primary_input="masked",
            multiview_output_ext="ply",
            provider_python="python",
            provider_device="cuda",
            triposr_python="/content/triposr-venv/bin/python",
            triposr_dir="/content/TripoSR",
            hunyuan3d_dir="/content/Hunyuan3D",
            chunk_size=256,
            mc_resolution=64,
            mesh_repair="printable",
            direct_mesh_timeout=123,
        )

        experiments = build_stl_first_experiments(args)
        by_name = {experiment["name"]: experiment for experiment in experiments}

        self.assertEqual([experiment["name"] for experiment in experiments[:3]], ["masked", "mirror", "biharmonic"])
        self.assertEqual(by_name["source_mesh_oracle"]["method"], "source-mesh-oracle")
        self.assertIn("--provider triposr-api", by_name["triposr_api_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--provider hunyuan3d-shape", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("{output_dir}/output_mesh_raw.glb", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["mv_recon"]["method"], "external-multiview-to-mesh")
        self.assertIn("{input_bundle}", by_name["mv_recon"]["direct_mesh_command"])

    def test_stl_first_smoke_optimize_command_uses_stl_quality_profile(self):
        args = SimpleNamespace(
            start_index=4,
            limit=2,
            depth_provider="depth-anything-v2",
            depth_model="depth-anything/Depth-Anything-V2-Small-hf",
            device="auto",
            stl_target_dimension=96,
            contact_sheet_max_samples=2,
            continue_on_error=True,
            resume=True,
        )
        experiments = [{"name": "masked"}, {"name": "mv_recon"}]

        command = build_stl_first_optimize_command(
            args,
            Path("manifest.jsonl"),
            Path("experiment"),
            Path("config.json"),
            experiments,
        )

        self.assertIn("--score-profile", command)
        self.assertIn("stl-quality", command)
        self.assertIn("--baseline-method", command)
        self.assertIn("masked", command)
        self.assertIn("--emit-stl", command)
        self.assertIn("masked,mv_recon", command)

    def test_triposr_repair_smoke_reads_missing_or_present_csv_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "summary.csv"
            csv_path.write_text("method,n\nmasked,1\ntriposr,1\n", encoding="utf-8")

            rows = rows_from_csv(csv_path)
            missing_rows = rows_from_csv(root / "missing.csv")

        self.assertEqual(rows[0]["method"], "masked")
        self.assertEqual(rows[1]["n"], "1")
        self.assertEqual(missing_rows, [])

    def test_depth_export_rejects_no_valid_cells(self):
        depth = np.array(
            [
                [0.10, np.nan],
                [np.nan, 0.60],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            depth_path = temp_path / "depth.npy"
            stl_path = temp_path / "model.stl"
            np.save(depth_path, depth)

            with self.assertRaisesRegex(ValueError, "no valid mesh faces"):
                depth_data_to_3d_model(
                    str(depth_path),
                    output_stl_path=str(stl_path),
                    target_dimension=-1,
                    z_scale=10,
                    sigma=0,
                )


class SurfaceMetricRegressionTests(unittest.TestCase):
    def test_surface_distance_metrics_are_zero_for_identical_aligned_depth(self):
        depth = np.array(
            [
                [0.0, 0.1],
                [0.2, 0.3],
            ],
            dtype=np.float32,
        )
        mask = np.ones(depth.shape, dtype=bool)

        metrics = surface_distance_metrics(depth, depth.copy(), mask)

        self.assertEqual(metrics["surface_point_count"], 4)
        self.assertAlmostEqual(metrics["surface_chamfer_l1"], 0.0)
        self.assertAlmostEqual(metrics["surface_chamfer_rmse"], 0.0)
        self.assertAlmostEqual(metrics["surface_rmse"], 0.0)
        self.assertAlmostEqual(metrics["surface_hausdorff95"], 0.0)

    def test_surface_distance_metrics_measure_depth_surface_shift(self):
        reference = np.zeros((2, 2), dtype=np.float32)
        shifted = np.ones((2, 2), dtype=np.float32)
        mask = np.ones(reference.shape, dtype=bool)

        metrics = surface_distance_metrics(reference, shifted, mask)

        self.assertEqual(metrics["surface_point_count"], 4)
        self.assertAlmostEqual(metrics["surface_chamfer_l1"], 1.0)
        self.assertAlmostEqual(metrics["surface_chamfer_rmse"], 1.0)
        self.assertAlmostEqual(metrics["surface_rmse"], 1.0)
        self.assertAlmostEqual(metrics["surface_hausdorff95"], 1.0)

    def test_surface_distance_metrics_return_nonfinite_when_mask_has_no_points(self):
        depth = np.zeros((2, 2), dtype=np.float32)
        mask = np.zeros(depth.shape, dtype=bool)

        metrics = surface_distance_metrics(depth, depth, mask)

        self.assertEqual(metrics["surface_point_count"], 0)
        self.assertTrue(np.isnan(metrics["surface_chamfer_l1"]))
        self.assertTrue(np.isnan(metrics["surface_chamfer_rmse"]))
        self.assertTrue(np.isnan(metrics["surface_rmse"]))
        self.assertTrue(np.isnan(metrics["surface_hausdorff95"]))

    def test_evaluate_sample_writes_surface_metrics_from_depth_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full = np.zeros((4, 4, 3), dtype=np.uint8)
            full[:, :2] = [20, 40, 60]
            full[:, 2:] = [120, 140, 160]
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[:, 2:] = 255
            silhouette = np.full((4, 4), 255, dtype=np.uint8)
            gt_depth = np.tile(np.linspace(0.0, 0.75, num=4, dtype=np.float32), (4, 1))
            pred_depth = gt_depth.copy()
            pred_depth[:, 2:] += 1.0

            full_path = temp_path / "full.png"
            masked_path = temp_path / "masked.png"
            mask_path = temp_path / "mask.png"
            silhouette_path = temp_path / "silhouette.png"
            gt_depth_path = temp_path / "gt_depth.npy"
            pred_depth_path = temp_path / "pred_depth.npy"
            Image.fromarray(full).save(full_path)
            Image.fromarray(full).save(masked_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(silhouette, mode="L").save(silhouette_path)
            np.save(gt_depth_path, gt_depth)
            np.save(pred_depth_path, pred_depth)

            sample = {
                "id": "surface_row",
                "full_image": str(full_path),
                "masked_image": str(masked_path),
                "mask": str(mask_path),
                "gt_depth": str(gt_depth_path),
                "gt_silhouette": str(silhouette_path),
                "completion_mode": "mirror-left-to-right",
            }
            args = SimpleNamespace(
                skip_depth=False,
                depth_provider="mock",
                depth_model="mock",
                device="cpu",
                emit_stl=False,
                prompt="",
                steps=1,
                seed=1,
                guidance=None,
                inpaint_max_dimension=64,
                model_name=None,
                lora_weights=None,
                lora_scale=None,
            )

            with patch(
                "backend.benchmark.run_completion_benchmark.process_image_get_depth_data",
                return_value=str(pred_depth_path),
            ):
                row = evaluate_sample(sample, "mock-method", str(full_path), str(full_path), temp_path / "run", args)
            summary = summarize([row], methods=["mock-method"], attempted_n=1)

        self.assertGreater(row["surface_chamfer_l1"], 0)
        self.assertGreater(row["surface_chamfer_rmse"], 0)
        self.assertGreater(row["object_surface_chamfer_l1"], 0)
        self.assertGreater(row["object_surface_chamfer_rmse"], 0)
        self.assertEqual(row["object_surface_point_count"], 8)
        self.assertIn("object_surface_chamfer_l1_median", summary[0])


class SurfaceBackfillRegressionTests(unittest.TestCase):
    def test_backfill_surface_metrics_infers_cached_depth_data_and_updates_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            run_dir = root / "run"
            method_dir = run_dir / "s1" / "mirror"
            dataset_dir.mkdir()
            method_dir.mkdir(parents=True)

            gt_depth = np.tile(np.linspace(0.0, 0.75, num=4, dtype=np.float32), (4, 1))
            pred_depth = gt_depth.copy()
            pred_depth[:, 2:] += 1.0
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[:, 2:] = 255
            silhouette = np.full((4, 4), 255, dtype=np.uint8)

            gt_depth_path = dataset_dir / "s1_depth.npy"
            pred_depth_path = method_dir / "output_depth_data.npy"
            mask_path = dataset_dir / "s1_mask.png"
            silhouette_path = dataset_dir / "s1_silhouette.png"
            completed_path = method_dir / "completed_visible_preserved.png"
            manifest_path = dataset_dir / "manifest.jsonl"
            np.save(gt_depth_path, gt_depth)
            np.save(pred_depth_path, pred_depth)
            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(silhouette, mode="L").save(silhouette_path)
            Image.new("RGB", (4, 4), (10, 20, 30)).save(completed_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "s1",
                        "gt_depth": str(gt_depth_path),
                        "mask": str(mask_path),
                        "gt_silhouette": str(silhouette_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "split_audit.json").write_text(json.dumps({"manifest": str(manifest_path)}), encoding="utf-8")
            with (run_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "method", "completed_image", "masked_mae"])
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "s1",
                        "method": "mirror",
                        "completed_image": str(completed_path),
                        "masked_mae": "0.2",
                    }
                )
            with (run_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["method", "n", "attempted_n", "error_count", "success_rate", "masked_mae_median"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "base-provider",
                        "n": "0",
                        "attempted_n": "3",
                        "error_count": "0",
                        "success_rate": "1",
                        "masked_mae_median": "",
                    }
                )
                writer.writerow(
                    {
                        "method": "mirror",
                        "n": "1",
                        "attempted_n": "3",
                        "error_count": "2",
                        "success_rate": "0.3333333333",
                        "masked_mae_median": "0.2",
                    }
                )

            result = backfill_surface_metrics(run_dir)
            with (run_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            with (run_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                summary = list(csv.DictReader(csv_file))

        self.assertEqual(result["updated_rows"], 1)
        self.assertGreater(float(rows[0]["surface_chamfer_l1"]), 0)
        self.assertGreater(float(rows[0]["object_surface_chamfer_rmse"]), 0)
        self.assertEqual(rows[0]["depth_data"], str(pred_depth_path))
        self.assertIn("object_surface_chamfer_l1_median", summary[0])
        self.assertGreater(float(summary[0]["object_surface_chamfer_l1_median"]), 0)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["method"], "mirror")
        self.assertEqual(summary[0]["attempted_n"], "3")
        self.assertEqual(summary[0]["error_count"], "2")


class ColabG4OrchestratorRegressionTests(unittest.TestCase):
    def write_manifest(self, root: Path) -> Path:
        manifest_path = root / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(
                {
                    "id": "sample_000",
                    "full_image": str(root / "full.png"),
                    "masked_image": str(root / "masked.png"),
                    "mask": str(root / "mask.png"),
                    "completion_mode": "mirror-left-to-right",
                    "asset_key": "procedural/sample_000",
                    "asset_category": "procedural",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def write_lora_dir(self, root: Path, name: str = "weighted_object-surface_s20") -> Path:
        lora_dir = root / name
        lora_dir.mkdir(parents=True)
        (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"adapter")
        return lora_dir

    def read_command_log(self, output_root: Path) -> list[dict]:
        with (output_root / "command_log.jsonl").open(encoding="utf-8") as log_file:
            return [json.loads(line) for line in log_file]

    def matching_commands(self, commands: list[dict], module_name: str) -> list[dict]:
        return [row for row in commands if module_name in row["command"]]

    def test_resume_eval_combine_uses_existing_lora_without_training_commands(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self.write_manifest(root)
            lora_dir = self.write_lora_dir(root)
            output_root = root / "g4_resume"

            with patch.object(
                sys,
                "argv",
                [
                    "colab_g4_orchestrator",
                    "--use-current-repo",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--stage",
                    "eval",
                    "--stage",
                    "combine",
                    "--manifest",
                    str(manifest),
                    "--existing-lora-weights",
                    str(lora_dir),
                    "--train-steps",
                    "200",
                    "--eval-start",
                    "1",
                    "--eval-start",
                    "2",
                    "--eval-limit",
                    "1",
                    "--allow-missing-split-audit",
                ],
            ):
                colab_g4_orchestrator.main()

            commands = self.read_command_log(output_root)
            eval_config = json.loads((output_root / "configs" / "modern_plus_weighted_lora.json").read_text())
            result = json.loads((output_root / "orchestrator_result.json").read_text())
            expected_manifest_sha = colab_g4_orchestrator.file_sha256(manifest)

        self.assertEqual(len(self.matching_commands(commands, "backend.benchmark.optimize_completion")), 2)
        self.assertEqual(len(self.matching_commands(commands, "backend.benchmark.combine_optimize_runs")), 1)
        first_eval_command = self.matching_commands(commands, "backend.benchmark.optimize_completion")[0]["command"]
        self.assertIn("--max-method-failures", first_eval_command)
        self.assertEqual(first_eval_command[first_eval_command.index("--max-method-failures") + 1], "2")
        self.assertFalse(self.matching_commands(commands, "backend.benchmark.export_training_pairs"))
        self.assertFalse(self.matching_commands(commands, "backend.benchmark.weight_training_pairs"))
        self.assertFalse(self.matching_commands(commands, "backend.benchmark.train_inpainting_lora"))
        self.assertIn(
            "dreamshaper_weighted_lora_object-surface_s20_scale0.75",
            {experiment["name"] for experiment in eval_config},
        )
        self.assertNotIn(
            "dreamshaper_weighted_lora_object-surface_s200_scale0.75",
            {experiment["name"] for experiment in eval_config},
        )
        self.assertEqual(result["training_metadata"], "")
        self.assertEqual(result["weighted_metadata"], "")
        self.assertEqual(result["lora_weights"], str(lora_dir))
        self.assertEqual(len(result["eval_dirs"]), 2)
        self.assertEqual(result["manifest_report"]["sha256"], expected_manifest_sha)
        self.assertTrue(result["source_modern_config"]["exists"])
        self.assertTrue(result["eval_config_report"]["exists"])
        self.assertIn("head", result["repo_state"])
        self.assertIn("duration_seconds", result)

    def test_combine_only_reuses_completed_eval_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self.write_manifest(root)
            lora_dir = self.write_lora_dir(root)
            output_root = root / "g4_combine_only"
            for start in (1, 2):
                eval_dir = output_root / "experiments" / f"modern_weighted_eval_s{start}_n1"
                eval_dir.mkdir(parents=True)
                (eval_dir / "aggregate_summary.csv").write_text("method,n,score\nmirror,1,0\n", encoding="utf-8")

            with patch.object(
                sys,
                "argv",
                [
                    "colab_g4_orchestrator",
                    "--use-current-repo",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--stage",
                    "combine",
                    "--manifest",
                    str(manifest),
                    "--existing-lora-weights",
                    str(lora_dir),
                    "--eval-start",
                    "1",
                    "--eval-start",
                    "2",
                    "--eval-limit",
                    "1",
                ],
            ):
                colab_g4_orchestrator.main()

            commands = self.read_command_log(output_root)
            result = json.loads((output_root / "orchestrator_result.json").read_text())

        self.assertFalse(self.matching_commands(commands, "backend.benchmark.optimize_completion"))
        self.assertEqual(len(self.matching_commands(commands, "backend.benchmark.combine_optimize_runs")), 1)
        self.assertEqual(len(result["eval_dirs"]), 2)
        self.assertTrue(result["combined_dir"].endswith("modern_weighted_object-surface_n2"))

    def test_resume_validation_fails_before_expensive_commands_for_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "g4_missing_manifest"

            with patch.object(
                sys,
                "argv",
                [
                    "colab_g4_orchestrator",
                    "--use-current-repo",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                    "--stage",
                    "eval",
                    "--manifest",
                    str(root / "missing.jsonl"),
                ],
            ):
                with self.assertRaisesRegex(FileNotFoundError, "--manifest does not exist"):
                    colab_g4_orchestrator.main()

            commands = self.read_command_log(output_root)

        self.assertFalse(self.matching_commands(commands, "backend.benchmark.optimize_completion"))

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_lora_dir = Path(temp_dir) / "empty_adapter"
            bad_lora_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "directory has no adapter file"):
                colab_g4_orchestrator.validate_lora_weights(bad_lora_dir)


class CompletionFailureCapRegressionTests(unittest.TestCase):
    def test_max_method_failures_skips_remaining_samples_without_rerunning_method(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"sample_{index:03d}",
                            "full_image": "unused_full.png",
                            "masked_image": "unused_masked.png",
                            "mask": "unused_mask.png",
                            "completion_mode": "mirror-left-to-right",
                        }
                    )
                    + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            output_dir = root / "run"

            with patch.object(
                sys,
                "argv",
                [
                    "run_completion_benchmark",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--methods",
                    "mirror",
                    "--limit",
                    "3",
                    "--skip-depth",
                    "--continue-on-error",
                    "--max-method-failures",
                    "1",
                ],
            ), patch(
                "backend.benchmark.run_completion_benchmark.run_one",
                side_effect=RuntimeError("provider unavailable"),
            ) as run_one_mock:
                run_completion_benchmark.main()

            failures = [
                json.loads(line)
                for line in (output_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with (output_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                summary = list(csv.DictReader(csv_file))

        self.assertEqual(run_one_mock.call_count, 1)
        self.assertEqual([row["error_type"] for row in failures], ["RuntimeError", "MethodFailureLimitExceeded", "MethodFailureLimitExceeded"])
        self.assertEqual([bool(row.get("skipped")) for row in failures], [False, True, True])
        self.assertEqual(summary[0]["attempted_n"], "3")
        self.assertEqual(summary[0]["error_count"], "3")
        self.assertEqual(summary[0]["success_rate"], "0.0")


class ColabInputPackageRegressionTests(unittest.TestCase):
    def test_package_inputs_rewrites_manifest_paths_and_includes_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "backend" / "output" / "completion-benchmark" / "modelnet"
            lora = root / "backend" / "output" / "completion-benchmark" / "lora" / "weighted_surface"
            dataset.mkdir(parents=True)
            lora.mkdir(parents=True)
            files = {
                "full_image": dataset / "sample_full.png",
                "masked_image": dataset / "sample_masked.png",
                "mask": dataset / "sample_mask.png",
                "gt_depth": dataset / "sample_depth.npy",
                "gt_silhouette": dataset / "sample_silhouette.png",
                "mesh": dataset / "sample.ply",
                "asset_path": root / "data" / "modelnet10" / "chair.off",
            }
            for path in files.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"asset")
            (lora / "pytorch_lora_weights.safetensors").write_bytes(b"adapter")
            (lora / "training_report.json").write_text(json.dumps({"final_step": 20}), encoding="utf-8")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        **{field: str(path.relative_to(root)) for field, path in files.items()},
                        "asset_key": "chair/test/chair.off",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "bundle.tar.gz"
            run_script = root / "run_colab_eval.sh"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/modelnet",
                root=root,
                lora_weights=lora,
                include_run_script=True,
                run_script_path=run_script,
                colab_archive_path="/content/bundle.tar.gz",
                run_name="g4_test_eval",
                eval_starts=[40, 50],
                eval_limit=10,
            )
            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())
                rewritten_manifest = json.loads(tar.extractfile("inputs/manifest.jsonl").read().decode("utf-8"))
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")
            local_run_script = run_script.read_text(encoding="utf-8")

        self.assertEqual(report["packaged_rows"], 1)
        self.assertEqual(report["referenced_files"], len(files))
        self.assertTrue(report["rewritten_manifest"].endswith("/inputs/manifest.jsonl"))
        self.assertTrue(report["rewritten_lora_weights"].endswith("/lora/weighted_surface"))
        self.assertEqual(report["run_script_in_archive"], "run_colab_eval.sh")
        self.assertEqual(report["run_script"], str(run_script))
        self.assertEqual(report["max_method_failures"], 2)
        self.assertIn("inputs/files/backend/output/completion-benchmark/modelnet/sample_full.png", names)
        self.assertIn("inputs/files/data/modelnet10/chair.off", names)
        self.assertIn("lora/weighted_surface/pytorch_lora_weights.safetensors", names)
        self.assertIn("lora/weighted_surface/training_report.json", names)
        self.assertIn("run_colab_eval.sh", names)
        self.assertEqual(rewritten_manifest["full_image"], "/content/inputs/modelnet/inputs/files/backend/output/completion-benchmark/modelnet/sample_full.png")
        self.assertEqual(rewritten_manifest["asset_path"], "/content/inputs/modelnet/inputs/files/data/modelnet10/chair.off")
        self.assertEqual(local_run_script, archive_run_script)
        self.assertIn('if [[ ! -d "$REPO_DIR/.git" ]]', archive_run_script)
        self.assertIn('git clone --filter=blob:none "$REPO_REMOTE" "$REPO_DIR"', archive_run_script)
        self.assertIn("git fetch", archive_run_script)
        self.assertIn("EXPECTED_SHA256", archive_run_script)
        self.assertIn("launch_preflight.json", archive_run_script)
        self.assertIn("manifest_rows", archive_run_script)
        self.assertIn("pytorch_lora_weights.safetensors", archive_run_script)
        self.assertIn("training_report.json", archive_run_script)
        self.assertIn("run_colab_eval.log", archive_run_script)
        self.assertIn("results_summary.json", archive_run_script)
        self.assertIn("g4_test_eval_results.tar.gz", archive_run_script)
        self.assertIn('tee "$RUN_LOG"', archive_run_script)
        self.assertIn("run_status", archive_run_script)
        self.assertIn("selection_decisions", archive_run_script)
        self.assertIn("eval_summaries", archive_run_script)
        self.assertIn("combined_summaries", archive_run_script)
        self.assertIn("summarize_benchmark_dir", archive_run_script)
        self.assertIn("--run-name g4_test_eval", archive_run_script)
        self.assertIn("--eval-start 40", archive_run_script)
        self.assertIn("--eval-start 50", archive_run_script)
        self.assertIn("--max-method-failures 2", archive_run_script)
        self.assertIn("--manifest /content/inputs/modelnet/inputs/manifest.jsonl", archive_run_script)
        self.assertIn("--existing-lora-weights /content/inputs/modelnet/lora/weighted_surface", archive_run_script)

    def test_package_inputs_can_write_modern_provider_launcher_without_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            full = dataset / "full.png"
            masked = dataset / "masked.png"
            mask = dataset / "mask.png"
            for path in (full, masked, mask):
                path.write_bytes(b"asset")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "full_image": str(full.relative_to(root)),
                        "masked_image": str(masked.relative_to(root)),
                        "mask": str(mask.relative_to(root)),
                        "category": "chair",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "qwen_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/qwen",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/qwen_bundle.tar.gz",
                run_name="g4_qwen_sanity",
                modern_config="backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json",
                cache_providers=["qwen-image-edit"],
                cache_full=True,
                eval_starts=[40, 50],
                eval_limit=2,
                eval_steps=20,
                eval_inpaint_max_dimension=512,
                depth_provider="depth-anything-v2",
                depth_model="depth-anything/Depth-Anything-V2-Small-hf",
                stl_target_dimension=96,
                score_profile="stl-quality",
                min_paired_n=2,
                allow_missing_split_audit=True,
                contact_sheet_methods="masked,mirror,biharmonic,qwen_edit_s20_s512",
            )
            with tarfile.open(archive, "r:gz") as tar:
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertEqual(report["rewritten_lora_weights"], "")
        self.assertEqual(report["cache_providers"], ["qwen-image-edit"])
        self.assertTrue(report["cache_full"])
        self.assertEqual(report["max_method_failures"], 2)
        self.assertIn("--run-name g4_qwen_sanity", archive_run_script)
        self.assertIn("--modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json", archive_run_script)
        self.assertIn("--cache-provider qwen-image-edit", archive_run_script)
        self.assertIn("--cache-full", archive_run_script)
        self.assertIn("--eval-limit 2", archive_run_script)
        self.assertIn("--eval-steps 20", archive_run_script)
        self.assertIn("--eval-inpaint-max-dimension 512", archive_run_script)
        self.assertIn("--depth-provider depth-anything-v2", archive_run_script)
        self.assertIn("--stl-target-dimension 96", archive_run_script)
        self.assertIn("--score-profile stl-quality", archive_run_script)
        self.assertIn("--min-paired-n 2", archive_run_script)
        self.assertIn("--max-method-failures 2", archive_run_script)
        self.assertIn("--allow-missing-split-audit", archive_run_script)
        self.assertIn("--contact-sheet-methods masked,mirror,biharmonic,qwen_edit_s20_s512", archive_run_script)
        self.assertIn("--manifest /content/inputs/qwen/inputs/manifest.jsonl", archive_run_script)
        self.assertIn("LORA_PATH=''", archive_run_script)
        self.assertIn('if [[ -n "$LORA_PATH" ]]; then', archive_run_script)
        self.assertNotIn("--existing-lora-weights", archive_run_script)


class TrainingProvenanceRegressionTests(unittest.TestCase):
    def write_tiny_manifest(self, root: Path) -> Path:
        manifest_path = root / "manifest.jsonl"
        rows = []
        for index, category in enumerate(["chair", "table"]):
            full = np.full((8, 8, 3), 30 + index * 40, dtype=np.uint8)
            masked = full.copy()
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, 4:] = 255
            masked[mask > 127] = 255
            silhouette = np.full((8, 8), 255, dtype=np.uint8)
            full_path = root / f"{category}_full.png"
            masked_path = root / f"{category}_masked.png"
            mask_path = root / f"{category}_mask.png"
            silhouette_path = root / f"{category}_silhouette.png"
            Image.fromarray(full).save(full_path)
            Image.fromarray(masked).save(masked_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(silhouette, mode="L").save(silhouette_path)
            rows.append(
                {
                    "id": f"{category}_000{index}",
                    "full_image": str(full_path),
                    "masked_image": str(masked_path),
                    "mask": str(mask_path),
                    "gt_silhouette": str(silhouette_path),
                    "completion_mode": "mirror-left-to-right",
                    "source": "mesh_dir",
                    "asset_category": category,
                    "asset_source_split": "train",
                    "asset_key": f"ModelNet10/{category}/train/{category}_{index}.off",
                }
            )
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in rows:
                manifest.write(json.dumps(row) + "\n")
        return manifest_path

    def test_export_pairs_and_lora_dry_run_write_provenance_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_tiny_manifest(root)
            pairs_dir = root / "pairs"
            prompt = "Complete the missing half of the same {category}."
            with patch.object(
                sys,
                "argv",
                [
                    "export_training_pairs",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(pairs_dir),
                    "--prompt",
                    prompt,
                    "--limit",
                    "2",
                ],
            ):
                export_training_pairs_main()

            export_report = json.loads((pairs_dir / "pair_export_report.json").read_text(encoding="utf-8"))
            metadata_path = pairs_dir / "metadata.jsonl"
            rows = read_metadata(metadata_path)
            lora_dir = root / "lora_dry_run"
            dry_run(
                SimpleNamespace(
                    metadata=str(metadata_path),
                    output_dir=str(lora_dir),
                    resolution=8,
                    train_batch_size=2,
                    mask_loss_weight=3.0,
                    seam_loss_weight=4.0,
                    object_loss_weight=1.0,
                ),
                rows,
            )
            training_plan = json.loads((lora_dir / "training_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(export_report["rows"], 2)
        self.assertEqual(export_report["prompt_mode"], "templated")
        self.assertEqual(export_report["object_mask_rows"], 2)
        self.assertIn("metadata_sha256", export_report)
        self.assertEqual(training_plan["pair_export_sha256"], export_report["metadata_sha256"])
        self.assertEqual(training_plan["loss_recipe"], "mask3_seam4_object1")
        self.assertEqual(training_plan["prompt_family"], "category-templated")
        self.assertEqual(training_plan["object_inpaint_nonzero_rows"], 2)
        self.assertEqual(training_plan["sample_weight_non_default_rows"], 0)

    def test_weighted_mse_loss_uses_sample_weights_without_changing_all_one_case(self):
        model_pred = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        target = torch.tensor([[[[1.0]]], [[[3.0]]]], dtype=torch.float32)
        mask = torch.zeros((2, 1, 1, 1), dtype=torch.float32)

        base_loss, _ = weighted_mse_loss(
            model_pred,
            target,
            mask,
            mask_weight=0.0,
            seam_weight=0.0,
        )
        all_one_loss, _ = weighted_mse_loss(
            model_pred,
            target,
            mask,
            mask_weight=0.0,
            seam_weight=0.0,
            sample_weight=torch.ones(2),
        )
        weighted_loss, _ = weighted_mse_loss(
            model_pred,
            target,
            mask,
            mask_weight=0.0,
            seam_weight=0.0,
            sample_weight=torch.tensor([3.0, 1.0]),
        )

        self.assertAlmostEqual(float(base_loss.item()), 5.0)
        self.assertAlmostEqual(float(all_one_loss.item()), float(base_loss.item()))
        self.assertAlmostEqual(float(weighted_loss.item()), 3.0)

    def test_pair_dataset_and_export_preserve_sample_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self.write_tiny_manifest(root)
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["sample_weight"] = 2.5
            manifest_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            pairs_dir = root / "pairs"
            with patch.object(
                sys,
                "argv",
                [
                    "export_training_pairs",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(pairs_dir),
                    "--limit",
                    "2",
                ],
            ):
                export_training_pairs_main()

            metadata_path = pairs_dir / "metadata.jsonl"
            metadata_rows = read_metadata(metadata_path)
            dataset = InpaintPairDataset(metadata_path, metadata_rows, resolution=8)
            batch = collate([dataset[0], dataset[1]])
            export_report = json.loads((pairs_dir / "pair_export_report.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata_rows[0]["sample_weight"], 2.5)
        self.assertNotIn("sample_weight", metadata_rows[1])
        self.assertEqual(batch["sample_weights"].tolist(), [2.5, 1.0])
        self.assertEqual(export_report["sample_weight_rows"], 1)
        self.assertEqual(export_report["sample_weight_max"], 2.5)

    def test_weight_training_pairs_focuses_high_surface_error_rows(self):
        metadata_rows = [
            {"id": "easy"},
            {"id": "hard"},
            {"id": "missing"},
        ]
        benchmark_rows = [
            {"sample_id": "easy", "method": "mirror", "object_surface_chamfer_l1": "0.10"},
            {"sample_id": "hard", "method": "mirror", "object_surface_chamfer_l1": "0.50"},
        ]

        weighted_rows, report = weight_metadata_rows(
            metadata_rows,
            benchmark_rows,
            method="mirror",
            weights={"object_surface_chamfer_l1_median": -4.0},
            score_profile="object-surface",
            base_weight=1.0,
            scale=2.0,
            max_weight=5.0,
        )
        weights_by_id = {row["id"]: row["sample_weight"] for row in weighted_rows}

        self.assertEqual(report["matched_rows"], 2)
        self.assertEqual(report["unmatched_rows"], 1)
        self.assertAlmostEqual(weights_by_id["easy"], 1.0)
        self.assertAlmostEqual(weights_by_id["hard"], 3.0)
        self.assertAlmostEqual(weights_by_id["missing"], 1.0)
        self.assertEqual(weighted_rows[1]["sample_weight_metric_count"], 1)

    def test_split_audit_snapshots_lora_recipe_and_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lora_dir = root / "lora"
            lora_dir.mkdir()
            report_path = lora_dir / "training_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "loss_recipe": "mask3_seam4_object0",
                        "metadata_sha256": "metadata-sha",
                        "pair_export_sha256": "pairs-sha",
                        "prompt_family": "literal",
                        "prompt_template": "Complete the object.",
                        "rows": 4,
                        "asset_count": 2,
                        "asset_keys": ["ModelNet10/chair/train/chair_0.off"],
                        "category_counts": {"chair": 4},
                        "max_train_steps": 20,
                        "final_step": 20,
                        "resolution": 8,
                        "rank": 4,
                        "mask_loss_weight": 3,
                        "seam_loss_weight": 4,
                        "object_loss_weight": 0,
                        "best_loss": 0.5,
                        "adapter_sha256": "adapter-sha",
                    }
                ),
                encoding="utf-8",
            )
            audit = write_split_audit(
                root / "run",
                [
                    {
                        "id": "heldout",
                        "asset_category": "chair",
                        "asset_key": "ModelNet10/chair/test/chair_1.off",
                    }
                ],
                SimpleNamespace(
                    lora_weights=str(lora_dir),
                    manifest="manifest.jsonl",
                    start_index=4,
                    limit=1,
                    prompt="Complete the missing half of the same {category}.",
                ),
            )

        self.assertEqual(audit["train_loss_recipe"], "mask3_seam4_object0")
        self.assertEqual(audit["train_metadata_sha256"], "metadata-sha")
        self.assertEqual(audit["adapter_sha256"], "adapter-sha")
        self.assertFalse(audit["prompt_family_matches_eval"])
        self.assertIn("differs", audit["prompt_family_mismatch_detail"])

    def test_backfill_lora_provenance_updates_old_reports_and_split_audits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_dir = root / "pairs"
            metadata_dir.mkdir()
            lora_root = root / "lora"
            lora_dir = lora_root / "adapter"
            lora_dir.mkdir(parents=True)
            metadata_path = metadata_dir / "metadata.jsonl"
            metadata_rows = [
                {
                    "id": "train_a",
                    "manifest_index": 0,
                    "prompt": "Complete the object.",
                    "prompt_template": "Complete the object.",
                    "category": "chair",
                    "asset_category": "chair",
                    "asset_source_split": "train",
                    "asset_key": "chair/train/chair_0.off",
                    "object_mask": "object_a.png",
                },
                {
                    "id": "train_b",
                    "manifest_index": 1,
                    "prompt": "Complete the object.",
                    "prompt_template": "Complete the object.",
                    "category": "table",
                    "asset_category": "table",
                    "asset_source_split": "train",
                    "asset_key": "table/train/table_0.off",
                    "object_mask": "object_b.png",
                },
            ]
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                for row in metadata_rows:
                    metadata_file.write(json.dumps(row) + "\n")
            (lora_dir / "train_metrics.csv").write_text(
                "step,loss,unweighted_loss,lr\n1,0.5,0.4,0.0001\n2,0.25,0.2,0.0001\n",
                encoding="utf-8",
            )
            adapter_file = lora_dir / "pytorch_lora_weights.safetensors"
            adapter_file.write_bytes(b"fake adapter")
            (lora_dir / "training_report.json").write_text(
                json.dumps(
                    {
                        "metadata": str(metadata_path),
                        "rows": 2,
                        "base_model": "Lykon/dreamshaper-8-inpainting",
                        "resolution": 8,
                        "max_train_steps": 2,
                        "rank": 4,
                        "lora_alpha": 4,
                        "mask_loss_weight": 3,
                        "seam_loss_weight": 4,
                        "object_loss_weight": 0,
                        "asset_keys": ["chair/train/chair_0.off", "table/train/table_0.off"],
                    }
                ),
                encoding="utf-8",
            )
            run_dir = root / "experiments" / "sweep" / "learned"
            run_dir.mkdir(parents=True)
            (run_dir / "per_sample_metrics.csv").write_text(
                "sample_id,method,prompt,masked_mae\ns1,learned,Complete the missing half of the same chair.,0.1\n",
                encoding="utf-8",
            )
            audit_path = run_dir / "split_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "lora_weights": str(lora_dir),
                        "training_report": str(lora_dir / "training_report.json"),
                        "train_eval_asset_overlap_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            lora_rows = backfill_lora_root(lora_root)
            audit_result = backfill_split_audit(audit_path)
            report = json.loads((lora_dir / "training_report.json").read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            pair_report_exists = (metadata_dir / "pair_export_report.json").exists()

        self.assertEqual(len(lora_rows), 1)
        self.assertTrue(lora_rows[0]["updated"])
        self.assertTrue(pair_report_exists)
        self.assertEqual(report["loss_recipe"], "mask3_seam4_object0")
        self.assertEqual(report["final_step"], 2)
        self.assertEqual(report["best_loss"], 0.25)
        self.assertEqual(report["prompt_family"], "literal")
        self.assertIn("adapter_sha256", report)
        self.assertTrue(audit_result["updated"])
        self.assertFalse(audit["prompt_family_matches_eval"])
        self.assertEqual(audit["train_loss_recipe"], "mask3_seam4_object0")

    def test_optimizer_report_renders_training_recipe_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            lora_dir = output_dir / "lora"
            lora_dir.mkdir()
            (lora_dir / "training_report.json").write_text(
                json.dumps(
                    {
                        "base_model": "Lykon/dreamshaper-8-inpainting",
                        "loss_recipe": "mask3_seam4_object1",
                        "metadata_sha256": "metadata-sha",
                        "prompt_family": "category-templated",
                        "rows": 8,
                        "asset_count": 4,
                        "max_train_steps": 20,
                        "final_step": 20,
                        "resolution": 8,
                        "rank": 4,
                        "mask_loss_weight": 3,
                        "seam_loss_weight": 4,
                        "object_loss_weight": 1,
                        "final_loss": 0.7,
                        "best_loss": 0.5,
                        "adapter_sha256": "adapter-sha",
                    }
                ),
                encoding="utf-8",
            )
            aggregate_path = output_dir / "aggregate_summary.csv"
            ranked_path = output_dir / "ranked_experiments.csv"
            aggregate_path.write_text(
                "method,success_rate,error_count,masked_mae_median\nmasked,1,0,0.6\nlearned,1,0,0.2\n",
                encoding="utf-8",
            )
            ranked_path.write_text(
                "method,rank_score,n,attempted_n,success_rate,error_count,masked_mae_median\nlearned,1.0,2,2,1,0,0.2\nmasked,0,2,2,1,0,0.6\n",
                encoding="utf-8",
            )
            for method, value in [("masked", "0.6"), ("learned", "0.2")]:
                method_dir = output_dir / method
                method_dir.mkdir()
                (method_dir / "per_sample_metrics.csv").write_text(
                    f"sample_id,method,masked_mae\ns1,{method},{value}\ns2,{method},{value}\n",
                    encoding="utf-8",
                )
            args = SimpleNamespace(
                manifest="manifest.jsonl",
                limit=2,
                baseline_method="masked",
                score_mode="baseline-delta",
                weight=["masked_mae_median=-4"],
                paired_bootstrap_samples=0,
                paired_bootstrap_seed=1234,
                prompt="Complete {category}",
                steps=20,
                guidance=7.5,
                seed=1234,
                inpaint_max_dimension=256,
                model_name="",
                lora_weights="",
                lora_scale="",
                start_index=0,
                skip_depth=True,
                emit_stl=False,
            )
            report_path = write_experiment_report(
                args,
                [
                    {"name": "masked", "method": "masked"},
                    {
                        "name": "learned",
                        "method": "dreamshaper-inpaint",
                        "lora_weights": str(lora_dir),
                        "lora_scale": 0.5,
                    },
                ],
                output_dir,
                aggregate_path,
                ranked_path,
            )
            markdown = report_path.read_text(encoding="utf-8")
            metadata = training_metadata({"lora_weights": str(lora_dir)})

        self.assertEqual(metadata["train_loss_recipe"], "mask3_seam4_object1")
        self.assertIn("## Training Recipes", markdown)
        self.assertIn("mask3_seam4_object1", markdown)
        self.assertIn("adapter-sha", markdown)


class OptimizeCompletionRegressionTests(unittest.TestCase):
    def test_default_experiments_use_provider_ids_for_deterministic_methods(self):
        experiments = load_experiments(None)
        names = {experiment["name"] for experiment in experiments}

        self.assertIn("mirror-seam-repair", names)
        self.assertNotIn("mirror_seam_repair", names)

    def test_per_sample_annotation_preserves_rendered_prompt_and_labels_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            metrics_path = run_dir / "per_sample_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["sample_id", "method", "prompt", "lora_scale", "masked_mae"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "mesh_dir_table_0001_v00",
                        "method": "dreamshaper-inpaint",
                        "prompt": "Complete the missing half of the same table.",
                        "lora_scale": "0.5",
                        "masked_mae": "0.12",
                    }
                )

            annotate_per_sample_metrics(
                {
                    "name": "dreamshaper_lora_scale050",
                    "method": "dreamshaper-inpaint",
                    "prompt": "Complete the missing half of the same {category}.",
                    "lora_scale": 0.5,
                },
                run_dir,
                default_start_index=40,
            )
            annotate_per_sample_metrics(
                {
                    "name": "dreamshaper_lora_scale050",
                    "method": "dreamshaper-inpaint",
                    "prompt": "Complete the missing half of the same {category}.",
                    "lora_scale": 0.5,
                },
                run_dir,
                default_start_index=40,
            )

            with metrics_path.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "dreamshaper_lora_scale050")
        self.assertEqual(rows[0]["base_method"], "dreamshaper-inpaint")
        self.assertEqual(rows[0]["start_index"], "40")
        self.assertEqual(rows[0]["prompt"], "Complete the missing half of the same table.")
        self.assertEqual(rows[0]["lora_scale"], "0.5")
        self.assertEqual(rows[0]["masked_mae"], "0.12")
        self.assertEqual(fieldnames.count("base_method"), 1)
        self.assertLess(fieldnames.index("method"), fieldnames.index("base_method"))

    def test_experiment_metadata_records_inherited_modern_edit_fill(self):
        modern = {
            "name": "qwen_mirror_prefill",
            "method": "qwen-image-edit",
        }
        deterministic = {
            "name": "mirror",
            "method": "mirror",
        }

        self.assertEqual(experiment_metadata(modern, default_edit_mask_fill="mirror")["edit_mask_fill"], "mirror")
        self.assertEqual(experiment_metadata(modern, default_edit_mask_fill="input")["edit_mask_fill"], "")
        self.assertEqual(experiment_metadata(deterministic, default_edit_mask_fill="mirror")["edit_mask_fill"], "")

    def test_modern_cache_preflight_fails_fast_for_missing_files(self):
        def fake_planner(provider, full, revision, local_dir, model_name=None):
            return {
                "provider": provider,
                "model": model_name,
                "full": full,
                "cache_status": {
                    "revision": revision or "main",
                    "local_dir": local_dir or "",
                    "complete": False,
                    "missing_file_count": 2,
                    "missing_size_gb": 6.076,
                    "missing_files": [
                        {"path": "text_encoder_2/model.fp16.safetensors", "size": 1389382884},
                        {"path": "unet/diffusion_pytorch_model.fp16.safetensors", "size": 5135178560},
                    ],
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            args = SimpleNamespace(
                model_name=None,
                modern_cache_full=False,
                modern_cache_revision=None,
                modern_cache_local_dir=None,
            )

            with self.assertRaisesRegex(RuntimeError, "sdxl-inpaint.*missing 2 files"):
                write_modern_cache_preflight(
                    args,
                    [
                        {"name": "mirror", "method": "mirror"},
                        {
                            "name": "sdxl_candidate",
                            "method": "sdxl-inpaint",
                            "model_name": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        },
                    ],
                    output_dir,
                    planner=fake_planner,
                )

            report = json.loads((output_dir / "modern_cache_preflight.json").read_text(encoding="utf-8"))

        self.assertTrue(report["require_complete"])
        self.assertEqual(report["plans"][0]["provider"], "sdxl-inpaint")
        self.assertEqual(report["plans"][0]["experiment_names"], ["sdxl_candidate"])
        self.assertFalse(report["plans"][0]["cache_status"]["complete"])

    def test_write_selection_decision_emits_json_and_markdown_for_sweep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "mirror").mkdir()
            (output_dir / "mirror" / "split_audit.json").write_text(
                json.dumps({"train_eval_asset_overlap_count": 0}),
                encoding="utf-8",
            )
            aggregate_rows = [
                {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
                {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.20"},
            ]
            per_sample_rows = [
                {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
                {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
                {"sample_id": "b", "method": "mirror", "masked_mae": "0.10"},
            ]
            args = SimpleNamespace(
                baseline_method="masked",
                candidate_method=None,
                current_method="mirror",
                weight=["masked_mae_median=-4"],
                min_success_rate=1.0,
                min_paired_n=2,
                min_win_rate=0.8,
                min_ci95_low=0.0,
                min_score_margin=0.0,
                max_train_eval_overlap=0,
                allow_missing_split_audit=False,
                paired_bootstrap_samples=0,
                paired_bootstrap_seed=1234,
            )

            output_json, output_md = write_selection_decision(args, output_dir, aggregate_rows, per_sample_rows)
            decision = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(decision["decision"], "keep_current")
        self.assertEqual(decision["candidate_method"], "mirror")
        self.assertIn("split_audit_present", {check["name"] for check in decision["checks"]})
        self.assertIn("## Gate Checks", markdown)

    def test_write_selection_decision_emits_paired_current_section_for_incumbent_challenge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "learned").mkdir()
            (output_dir / "learned" / "split_audit.json").write_text(
                json.dumps({"train_eval_asset_overlap_count": 0}),
                encoding="utf-8",
            )
            aggregate_rows = [
                {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
                {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.30"},
                {"method": "learned", "success_rate": "1.0", "masked_mae_median": "0.10"},
            ]
            per_sample_rows = [
                {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
                {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                {"sample_id": "a", "method": "learned", "masked_mae": "0.25"},
                {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
                {"sample_id": "b", "method": "mirror", "masked_mae": "0.20"},
                {"sample_id": "b", "method": "learned", "masked_mae": "0.25"},
            ]
            args = SimpleNamespace(
                baseline_method="masked",
                candidate_method="learned",
                current_method="mirror",
                weight=["masked_mae_median=-4"],
                min_success_rate=1.0,
                min_paired_n=2,
                min_win_rate=0.8,
                min_ci95_low=0.0,
                min_score_margin=0.0,
                min_stl_watertight=1.0,
                min_stl_positive_volume=1.0,
                max_train_eval_overlap=0,
                allow_missing_split_audit=False,
                paired_bootstrap_samples=0,
                paired_bootstrap_seed=1234,
            )

            output_json, output_md = write_selection_decision(args, output_dir, aggregate_rows, per_sample_rows)
            decision = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("paired_objective_vs_current", decision)
        self.assertLess(decision["paired_objective_vs_current"]["mean_improvement"], 0)
        self.assertIn("## Paired Objective vs Current", markdown)
        self.assertIn("paired_win_rate_vs_current", {check["name"] for check in decision["failed_checks"]})

    def test_write_selection_decision_can_use_object_surface_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "learned").mkdir()
            (output_dir / "learned" / "split_audit.json").write_text(
                json.dumps({"train_eval_asset_overlap_count": 0}),
                encoding="utf-8",
            )
            aggregate_rows = [
                {
                    "method": "masked",
                    "success_rate": "1.0",
                    "masked_mae_median": "0.50",
                    "object_surface_chamfer_l1_median": "0.50",
                    "stl_is_watertight_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                },
                {
                    "method": "mirror",
                    "success_rate": "1.0",
                    "masked_mae_median": "0.10",
                    "object_surface_chamfer_l1_median": "0.60",
                    "stl_is_watertight_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                },
                {
                    "method": "learned",
                    "success_rate": "1.0",
                    "masked_mae_median": "0.80",
                    "object_surface_chamfer_l1_median": "0.20",
                    "stl_is_watertight_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                },
            ]
            per_sample_rows = []
            for sample_id in ("a", "b"):
                per_sample_rows.extend(
                    [
                        {
                            "sample_id": sample_id,
                            "method": "masked",
                            "masked_mae": "0.50",
                            "object_surface_chamfer_l1": "0.50",
                            "stl_is_watertight": "True",
                            "stl_positive_volume": "True",
                        },
                        {
                            "sample_id": sample_id,
                            "method": "mirror",
                            "masked_mae": "0.10",
                            "object_surface_chamfer_l1": "0.60",
                            "stl_is_watertight": "True",
                            "stl_positive_volume": "True",
                        },
                        {
                            "sample_id": sample_id,
                            "method": "learned",
                            "masked_mae": "0.80",
                            "object_surface_chamfer_l1": "0.20",
                            "stl_is_watertight": "True",
                            "stl_positive_volume": "True",
                        },
                    ]
                )
            args = SimpleNamespace(
                baseline_method="masked",
                score_profile="object-surface",
                candidate_method=None,
                current_method="mirror",
                weight=[],
                min_success_rate=1.0,
                min_paired_n=2,
                min_win_rate=0.8,
                min_ci95_low=0.0,
                min_score_margin=0.0,
                min_stl_watertight=1.0,
                min_stl_positive_volume=1.0,
                max_train_eval_overlap=0,
                allow_missing_split_audit=False,
                paired_bootstrap_samples=0,
                paired_bootstrap_seed=1234,
            )

            output_json, output_md = write_selection_decision(args, output_dir, aggregate_rows, per_sample_rows)
            decision = json.loads(output_json.read_text(encoding="utf-8"))
            markdown = output_md.read_text(encoding="utf-8")

        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["candidate_method"], "learned")
        self.assertEqual(decision["score_profile"], "object-surface")
        self.assertIn("- Score profile: `object-surface`", markdown)
        self.assertGreater(decision["candidate_rank_score"], decision["current_rank_score"])


class ModernProviderPreflightRegressionTests(unittest.TestCase):
    def test_provider_rows_reports_auth_override_and_import_readiness(self):
        class FakeApi:
            def model_info(self, model_id, files_metadata=True):
                tags_by_model = {
                    "black-forest-labs/FLUX.1-Fill-dev": [
                        "license:other",
                        "diffusers:FluxFillPipeline",
                    ],
                    "Qwen/Qwen-Image-Edit": [
                        "license:apache-2.0",
                        "diffusers:QwenImageEditPipeline",
                    ],
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": [
                        "license:openrail++",
                        "diffusers:StableDiffusionXLInpaintPipeline",
                    ],
                }
                return SimpleNamespace(
                    private=False,
                    gated="auto" if model_id == "black-forest-labs/FLUX.1-Fill-dev" else False,
                    disabled=False,
                    siblings=[SimpleNamespace(size=1024), SimpleNamespace(size=2048)],
                    tags=tags_by_model[model_id],
                )

        def fake_model_index(model_id):
            if model_id == "black-forest-labs/FLUX.1-Fill-dev":
                return "", "GatedRepoError: 401 Client Error"
            if model_id == "Qwen/Qwen-Image-Edit":
                return "QwenImageEditPipeline", ""
            return "StableDiffusionXLInpaintPipeline", ""

        def fake_import(class_name):
            if class_name == "StableDiffusionXLInpaintPipeline":
                return False, "missing"
            return True, ""

        rows = provider_rows(
            ["flux-fill", "qwen-image-inpaint", "sdxl-inpaint"],
            api=FakeApi(),
            model_index_lookup=fake_model_index,
            pipeline_import_lookup=fake_import,
        )
        by_provider = {row["provider"]: row for row in rows}

        self.assertEqual(by_provider["flux-fill"]["readiness"], "auth-required")
        self.assertTrue(by_provider["flux-fill"]["class_matches_advertised"])
        self.assertEqual(by_provider["qwen-image-inpaint"]["readiness"], "pipeline-override")
        self.assertFalse(by_provider["qwen-image-inpaint"]["class_matches_advertised"])
        self.assertEqual(by_provider["sdxl-inpaint"]["readiness"], "pipeline-missing")
        self.assertFalse(by_provider["sdxl-inpaint"]["pipeline_import_available"])


class CompareOptimizeRunsRegressionTests(unittest.TestCase):
    def write_run(self, root: Path, candidate_method: str, prompt_match: bool, candidate_mae: float) -> Path:
        run_dir = root / candidate_method
        run_dir.mkdir()
        aggregate_path = run_dir / "aggregate_summary.csv"
        aggregate_path.write_text(
            "\n".join(
                [
                    "method,success_rate,error_count,masked_mae_median,lora_weights,lora_scale,stl_is_watertight_median,stl_positive_volume_median",
                    "masked,1,0,0.60,,,1,1",
                    "mirror,1,0,0.20,,,1,1",
                    f"{candidate_method},1,0,{candidate_mae},backend/output/lora/adapter,0.5,1,1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        for method, mae in [("masked", 0.60), ("mirror", 0.20), (candidate_method, candidate_mae)]:
            method_dir = run_dir / method
            method_dir.mkdir()
            (method_dir / "per_sample_metrics.csv").write_text(
                f"sample_id,method,masked_mae\ns1,{method},{mae}\ns2,{method},{mae}\n",
                encoding="utf-8",
            )
        (run_dir / candidate_method / "split_audit.json").write_text(
            json.dumps(
                {
                    "lora_weights": "backend/output/lora/adapter",
                    "training_report": "backend/output/lora/adapter/training_report.json",
                    "train_eval_asset_overlap_count": 0,
                    "train_loss_recipe": "mask3_seam4_object0",
                    "adapter_sha256": "adapter-sha",
                    "train_prompt_family": "literal",
                    "eval_prompt_family": "literal",
                    "prompt_family_matches_eval": prompt_match,
                    "prompt_family_mismatch_detail": "" if prompt_match else "train_prompt_template differs from eval_prompt_template",
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_compare_runs_ranks_each_run_and_surfaces_prompt_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_run = self.write_run(root, "old_lora", prompt_match=False, candidate_mae=0.10)
            new_run = self.write_run(root, "new_lora", prompt_match=True, candidate_mae=0.15)

            rows = add_score_deltas(
                [
                    compare_run(
                        "old",
                        old_run,
                        "old_lora",
                        baseline_method="masked",
                        current_method="mirror",
                        weights=parse_weights(["masked_mae_median=-4"]),
                        min_success_rate=1.0,
                        min_paired_n=2,
                        min_win_rate=0.8,
                        min_ci95_low=0.0,
                        min_score_margin=0.0,
                        min_stl_watertight=1.0,
                        min_stl_positive_volume=1.0,
                        max_train_eval_overlap=0,
                        allow_missing_split_audit=False,
                        paired_bootstrap_samples=0,
                        paired_bootstrap_seed=1234,
                    ),
                    compare_run(
                        "new",
                        new_run,
                        "new_lora",
                        baseline_method="masked",
                        current_method="mirror",
                        weights=parse_weights(["masked_mae_median=-4"]),
                        min_success_rate=1.0,
                        min_paired_n=2,
                        min_win_rate=0.8,
                        min_ci95_low=0.0,
                        min_score_margin=0.0,
                        min_stl_watertight=1.0,
                        min_stl_positive_volume=1.0,
                        max_train_eval_overlap=0,
                        allow_missing_split_audit=False,
                        paired_bootstrap_samples=0,
                        paired_bootstrap_seed=1234,
                    ),
                ]
            )
            markdown = render_markdown(
                rows,
                SimpleNamespace(baseline_method="masked", current_method="mirror"),
            )

        self.assertEqual(rows[0]["run_label"], "old")
        self.assertEqual(rows[0]["decision"], "hold")
        self.assertIn("prompt_family_matches_eval", rows[0]["failed_checks"])
        self.assertEqual(rows[1]["prompt_family_matches_eval"], True)
        self.assertLess(float(rows[1]["rank_score_delta_vs_first"]), 0)
        self.assertIn("Delta vs First", markdown)
        self.assertIn("new_lora", markdown)


class CombineOptimizeRunsRegressionTests(unittest.TestCase):
    def write_optimize_run(self, root: Path, name: str, sample_id: str, mirror_surface: float) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "aggregate_summary.csv").write_text(
            "\n".join(
                [
                    "method,n,attempted_n,error_count,success_rate,object_surface_chamfer_l1_median",
                    "masked,1,1,0,1,0.50",
                    f"mirror,1,1,0,1,{mirror_surface}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        for method, value in (("masked", "0.50"), ("mirror", str(mirror_surface))):
            method_dir = run_dir / method
            method_dir.mkdir()
            (method_dir / "per_sample_metrics.csv").write_text(
                "\n".join(
                    [
                        "sample_id,method,object_surface_chamfer_l1",
                        f"{sample_id},{method},{value}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return run_dir

    def test_combine_runs_prefixes_sample_ids_and_recomputes_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_a = self.write_optimize_run(root, "run_a", "same_id", 0.20)
            run_b = self.write_optimize_run(root, "run_b", "same_id", 0.40)

            per_sample_rows, summary_rows = combine_runs([("a", run_a), ("b", run_b)])

        sample_ids = {row["sample_id"] for row in per_sample_rows}
        mirror = next(row for row in summary_rows if row["method"] == "mirror")

        self.assertEqual(sample_ids, {"a::same_id", "b::same_id"})
        self.assertEqual(mirror["n"], 2)
        self.assertEqual(mirror["attempted_n"], 2)
        self.assertAlmostEqual(mirror["object_surface_chamfer_l1_median"], 0.30)


class CompletionMethodRegressionTests(unittest.TestCase):
    def test_large_modern_providers_use_bfloat16_on_cuda(self):
        self.assertEqual(_inpaint_torch_dtype("flux-fill", "cuda"), torch.bfloat16)
        self.assertEqual(_inpaint_torch_dtype("qwen-image-inpaint", "cuda"), torch.bfloat16)
        self.assertEqual(_inpaint_torch_dtype("qwen-image-edit", "cuda"), torch.bfloat16)
        self.assertEqual(_inpaint_torch_dtype("sdxl-inpaint", "cuda"), torch.float16)
        self.assertEqual(_inpaint_torch_dtype("dreamshaper-inpaint", "cpu"), torch.float32)

    def test_masked_edit_image_can_mark_missing_region_without_touching_visible_pixels(self):
        image = Image.new("RGB", (8, 4), (10, 20, 30))
        mask = Image.new("L", (8, 4), 0)
        mask_data = np.zeros((4, 8), dtype=np.uint8)
        mask_data[:, 4:] = 255
        mask = Image.fromarray(mask_data, mode="L")

        edited = np.asarray(_masked_edit_image(image, mask, "checker"))

        self.assertTrue(np.all(edited[:, :4] == [10, 20, 30]))
        self.assertFalse(np.all(edited[:, 4:] == [10, 20, 30]))

    def test_masked_edit_image_can_prefill_missing_region_from_geometry(self):
        left = np.zeros((4, 4, 3), dtype=np.uint8)
        left[:, 0] = [10, 20, 30]
        left[:, 1] = [40, 50, 60]
        left[:, 2] = [70, 80, 90]
        left[:, 3] = [100, 110, 120]
        image_data = np.concatenate([left, np.full((4, 4, 3), 255, dtype=np.uint8)], axis=1)
        image = Image.fromarray(image_data)
        mask = Image.fromarray(np.pad(np.full((4, 4), 255, dtype=np.uint8), ((0, 0), (4, 0))), mode="L")

        mirrored = np.asarray(_masked_edit_image(image, mask, "mirror"))
        biharmonic = np.asarray(_masked_edit_image(image, mask, "biharmonic"))

        self.assertTrue(np.array_equal(mirrored[:, :4], image_data[:, :4]))
        self.assertTrue(np.array_equal(mirrored[:, 4], image_data[:, 3]))
        self.assertTrue(np.array_equal(mirrored[:, 7], image_data[:, 0]))
        self.assertTrue(np.array_equal(biharmonic[:, :4], image_data[:, :4]))
        self.assertFalse(np.all(biharmonic[:, 4:] == 255))

    def test_qwen_edit_prompt_matches_edit_fill_cue(self):
        base_prompt = "Complete the object."

        self.assertIn("white blank region", _qwen_edit_prompt(base_prompt, "white"))
        self.assertIn("checkerboard region", _qwen_edit_prompt(base_prompt, "checker"))
        self.assertIn("mirrored prefilled half", _qwen_edit_prompt(base_prompt, "mirror"))
        self.assertIn("smooth prefilled half", _qwen_edit_prompt(base_prompt, "biharmonic"))
        self.assertNotIn("white blank region", _qwen_edit_prompt(base_prompt, "mirror"))
        self.assertNotIn("white blank region", _qwen_edit_prompt(base_prompt, "biharmonic"))

    def test_mirror_seam_repair_provider_runs_without_visible_pixel_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full = np.zeros((8, 8, 3), dtype=np.uint8)
            full[:, :4] = [30, 60, 120]
            full[:, 4:] = [120, 60, 30]
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, 4:] = 255
            masked = full.copy()
            masked[mask > 127] = 255

            full_path = temp_path / "full.png"
            masked_path = temp_path / "masked.png"
            mask_path = temp_path / "mask.png"
            Image.fromarray(full).save(full_path)
            Image.fromarray(masked).save(masked_path)
            Image.fromarray(mask, mode="L").save(mask_path)

            completed_path, applied_mode = complete_image(
                str(masked_path),
                output_dir=str(temp_path / "provider"),
                mode="mirror-left-to-right",
                provider="mirror-seam-repair",
            )
            row = run_one(
                {
                    "id": "tiny_repair",
                    "full_image": str(full_path),
                    "masked_image": str(masked_path),
                    "mask": str(mask_path),
                    "completion_mode": "mirror-left-to-right",
                },
                "mirror-seam-repair",
                temp_path / "run",
                SimpleNamespace(skip_depth=True),
            )
            provider_output_exists = Path(completed_path).exists()
            benchmark_output_exists = Path(row["completed_image"]).exists()

        self.assertTrue(provider_output_exists)
        self.assertEqual(applied_mode, "mirror-left-to-right:seam-repair")
        self.assertEqual(row["method"], "mirror-seam-repair")
        self.assertEqual(row["visible_mae"], 0)
        self.assertTrue(benchmark_output_exists)

    def test_masked_method_scores_blank_half_without_special_evaluation_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full = np.zeros((4, 4, 3), dtype=np.uint8)
            full[:, :2] = [20, 40, 60]
            full[:, 2:] = [120, 140, 160]
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[:, 2:] = 255
            masked = full.copy()
            masked[mask > 127] = 255

            full_path = temp_path / "full.png"
            masked_path = temp_path / "masked.png"
            mask_path = temp_path / "mask.png"
            Image.fromarray(full).save(full_path)
            Image.fromarray(masked).save(masked_path)
            Image.fromarray(mask, mode="L").save(mask_path)

            row = run_one(
                {
                    "id": "tiny_masked",
                    "full_image": str(full_path),
                    "masked_image": str(masked_path),
                    "mask": str(mask_path),
                    "completion_mode": "mirror-left-to-right",
                },
                "masked",
                temp_path / "run",
                SimpleNamespace(skip_depth=True),
            )
            self.assertTrue(Path(row["raw_completed_image"]).exists())
            self.assertTrue(Path(row["completed_image"]).exists())

        self.assertEqual(row["method"], "masked")
        self.assertEqual(row["full_image"], str(full_path))
        self.assertEqual(row["masked_image"], str(masked_path))
        self.assertEqual(row["mask"], str(mask_path))
        self.assertGreater(row["masked_mae"], 0)
        self.assertEqual(row["visible_mae"], 0)


class ArtifactContactSheetRegressionTests(unittest.TestCase):
    def test_reference_artifacts_fall_back_to_manifest_when_row_paths_are_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_full = root / "valid_full.png"
            valid_masked = root / "valid_masked.png"
            valid_mask = root / "valid_mask.png"
            for path in (valid_full, valid_masked, valid_mask):
                Image.new("RGB", (4, 4), (10, 20, 30)).save(path)
            manifest_path = root / "manifest.jsonl"
            row = {
                "sample_id": "s1",
                "_run_dir": str(root / "run"),
                "full_image": str(root / "missing_full.png"),
                "masked_image": str(root / "missing_masked.png"),
                "mask": str(root / "missing_mask.png"),
            }
            manifest = {
                "s1": {
                    "full_image": str(valid_full),
                    "masked_image": str(valid_masked),
                    "mask": str(valid_mask),
                }
            }

            artifacts = reference_artifacts(row, manifest, manifest_path)

        self.assertEqual(artifacts["full_image"], valid_full)
        self.assertEqual(artifacts["masked_image"], valid_masked)
        self.assertEqual(artifacts["mask"], valid_mask)

    def test_filter_rows_accepts_base_method_names_for_named_sweeps(self):
        rows = [
            {
                "sample_id": "s1",
                "method": "dreamshaper_lora_scale050",
                "_method": "dreamshaper_lora_scale050",
                "base_method": "dreamshaper-inpaint",
                "_experiment_name": "dreamshaper_lora_scale050",
            },
            {
                "sample_id": "s1",
                "method": "mirror",
                "_method": "mirror",
                "base_method": "mirror",
                "_experiment_name": "mirror",
            },
        ]

        filtered = filter_rows(rows, methods=["dreamshaper-inpaint"], samples=None, max_samples=6)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["method"], "dreamshaper_lora_scale050")

    def test_contact_sheet_renders_run_with_manifest_references_and_depth_preview(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            run_dir = root / "run"
            dataset.mkdir()
            run_dir.mkdir()

            full = Image.new("RGB", (16, 16), (80, 120, 160))
            masked = Image.new("RGB", (16, 16), (255, 255, 255))
            mask = Image.new("L", (16, 16), 255)
            silhouette = Image.new("L", (16, 16), 255)
            gt_depth = np.tile(np.linspace(0.0, 1.0, num=16, dtype=np.float32), (16, 1))
            full_path = dataset / "s1_full.png"
            masked_path = dataset / "s1_masked.png"
            mask_path = dataset / "s1_mask.png"
            silhouette_path = dataset / "s1_silhouette.png"
            gt_depth_path = dataset / "s1_depth.npy"
            full.save(full_path)
            masked.save(masked_path)
            mask.save(mask_path)
            silhouette.save(silhouette_path)
            np.save(gt_depth_path, gt_depth)

            manifest_path = dataset / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "s1",
                        "full_image": str(full_path),
                        "masked_image": str(masked_path),
                        "mask": str(mask_path),
                        "gt_depth": str(gt_depth_path),
                        "gt_silhouette": str(silhouette_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "split_audit.json").write_text(
                json.dumps({"manifest": str(manifest_path)}),
                encoding="utf-8",
            )

            fieldnames = [
                "sample_id",
                "method",
                "full_image",
                "masked_image",
                "mask",
                "raw_completed_image",
                "completed_image",
                "depth_preview",
                "depth_data",
                "masked_mae",
                "object_depth_mae",
                "stl_model",
                "stl_is_watertight",
                "stl_positive_volume",
            ]
            rows = []
            for method, color, mae in [
                ("masked", (230, 230, 230), "0.5"),
                ("mirror", (80, 180, 100), "0.2"),
            ]:
                method_dir = run_dir / "s1" / method
                method_dir.mkdir(parents=True)
                raw_path = method_dir / "completed_input.png"
                completed_path = method_dir / "completed_visible_preserved.png"
                depth_path = method_dir / "output_depth_preview.png"
                depth_data_path = method_dir / "output_depth_data.npy"
                stl_path = method_dir / "output_model.stl"
                Image.new("RGB", (16, 16), color).save(raw_path)
                Image.new("RGB", (16, 16), color).save(completed_path)
                Image.new("RGB", (16, 16), (30, 30, 180)).save(depth_path)
                pred_depth = gt_depth.copy()
                pred_depth[:, 8:] += 0.2 if method == "masked" else 0.05
                np.save(depth_data_path, pred_depth)
                trimesh.creation.box(extents=(1.0, 1.0, 0.25)).export(stl_path)
                rows.append(
                    {
                        "sample_id": "s1",
                        "method": method,
                        "full_image": str(full_path),
                        "masked_image": str(masked_path),
                        "mask": str(mask_path),
                        "raw_completed_image": str(raw_path),
                        "completed_image": str(completed_path),
                        "depth_preview": str(depth_path),
                        "depth_data": str(depth_data_path),
                        "masked_mae": mae,
                        "object_depth_mae": "0.1",
                        "stl_model": str(stl_path),
                        "stl_is_watertight": "True",
                        "stl_positive_volume": "True",
                    }
                )

            with (run_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            output_path = root / "sheet.png"
            result = make_contact_sheet(run_dir, output_path=output_path, max_samples=1, thumb_size=64)
            output_exists = output_path.exists()
            with Image.open(output_path) as rendered:
                rendered_width = rendered.width
                rendered_height = rendered.height

        self.assertEqual(result["row_count"], 2)
        self.assertTrue(output_exists)
        self.assertGreater(rendered_width, 700)
        self.assertGreater(rendered_height, 200)

    def test_depth_error_image_renders_aligned_hidden_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_depth = np.tile(np.linspace(0.0, 1.0, num=8, dtype=np.float32), (8, 1))
            pred_depth = gt_depth.copy()
            pred_depth[:, 4:] += np.linspace(0.1, 0.4, num=4, dtype=np.float32)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, 4:] = 255
            silhouette = np.full((8, 8), 255, dtype=np.uint8)

            gt_depth_path = root / "gt_depth.npy"
            pred_depth_path = root / "output_depth_data.npy"
            mask_path = root / "mask.png"
            silhouette_path = root / "silhouette.png"
            np.save(gt_depth_path, gt_depth)
            np.save(pred_depth_path, pred_depth)
            Image.fromarray(mask, mode="L").save(mask_path)
            Image.fromarray(silhouette, mode="L").save(silhouette_path)
            row = {
                "sample_id": "s1",
                "_run_dir": str(root),
                "mask": str(mask_path),
                "gt_depth": str(gt_depth_path),
                "gt_silhouette": str(silhouette_path),
            }

            image = depth_error_image(row, {}, None, pred_depth_path)
            colors = image.getcolors(maxcolors=4096)

        self.assertEqual(image.size, (8, 8))
        self.assertIsNotNone(colors)
        self.assertGreater(len(colors), 2)

    def test_depth_error_image_returns_none_for_corrupt_depth_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_depth_path = root / "gt_depth.npy"
            bad_depth_path = root / "output_depth_data.npy"
            mask_path = root / "mask.png"
            np.save(gt_depth_path, np.zeros((4, 4), dtype=np.float32))
            bad_depth_path.write_text("not a numpy array", encoding="utf-8")
            Image.fromarray(np.full((4, 4), 255, dtype=np.uint8), mode="L").save(mask_path)
            row = {
                "sample_id": "s1",
                "_run_dir": str(root),
                "mask": str(mask_path),
                "gt_depth": str(gt_depth_path),
            }

            image = depth_error_image(row, {}, None, bad_depth_path)

        self.assertIsNone(image)

    def test_stl_preview_image_renders_valid_mesh(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            stl_path = Path(temp_dir) / "box.stl"
            trimesh.creation.box(extents=(1.0, 0.7, 0.25)).export(stl_path)

            preview = stl_preview_image(stl_path, 64)
            colors = preview.getcolors(maxcolors=4096)

        self.assertEqual(preview.size, (64, 64))
        self.assertIsNotNone(colors)
        self.assertGreater(len(colors), 2)

    def test_report_run_contact_sheet_cli_writes_png_and_links_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            image_path = run_dir / "image.png"
            Image.new("RGB", (12, 12), (120, 80, 40)).save(image_path)
            fieldnames = [
                "sample_id",
                "method",
                "full_image",
                "masked_image",
                "mask",
                "raw_completed_image",
                "completed_image",
                "masked_mae",
            ]
            with (run_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "s1",
                        "method": "masked",
                        "full_image": str(image_path),
                        "masked_image": str(image_path),
                        "mask": str(image_path),
                        "raw_completed_image": str(image_path),
                        "completed_image": str(image_path),
                        "masked_mae": "0.5",
                    }
                )
            with (run_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["method", "n", "attempted_n", "success_rate", "error_count", "masked_mae_median"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method": "masked",
                        "n": "1",
                        "attempted_n": "1",
                        "success_rate": "1",
                        "error_count": "0",
                        "masked_mae_median": "0.5",
                    }
                )

            report_path = run_dir / "report.md"
            sheet_path = run_dir / "sheet.png"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.benchmark.report_run",
                    str(run_dir),
                    "--output",
                    str(report_path),
                    "--contact-sheet",
                    "--contact-sheet-output",
                    str(sheet_path),
                    "--contact-sheet-max-samples",
                    "1",
                    "--contact-sheet-thumb-size",
                    "48",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
            )
            report = report_path.read_text(encoding="utf-8")
            sheet_exists = sheet_path.exists()

        self.assertTrue(sheet_exists)
        self.assertIn("Contact sheet", report)

    def test_optimize_completion_contact_sheet_cli_writes_png_and_links_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            output_dir = root / "experiment"
            dataset_dir.mkdir()
            full = np.zeros((8, 8, 3), dtype=np.uint8)
            full[:, :4] = [20, 40, 60]
            full[:, 4:] = [120, 140, 160]
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:, 4:] = 255
            masked = full.copy()
            masked[mask > 127] = 255
            full_path = dataset_dir / "full.png"
            masked_path = dataset_dir / "masked.png"
            mask_path = dataset_dir / "mask.png"
            Image.fromarray(full).save(full_path)
            Image.fromarray(masked).save(masked_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            manifest_path = dataset_dir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "s1",
                        "full_image": str(full_path),
                        "masked_image": str(masked_path),
                        "mask": str(mask_path),
                        "completion_mode": "mirror-left-to-right",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(json.dumps([{"name": "masked", "method": "masked"}]), encoding="utf-8")
            sheet_path = output_dir / "sheet.png"

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.benchmark.optimize_completion",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--config",
                    str(config_path),
                    "--limit",
                    "1",
                    "--skip-depth",
                    "--contact-sheet",
                    "--contact-sheet-output",
                    str(sheet_path),
                    "--contact-sheet-max-samples",
                    "1",
                    "--contact-sheet-thumb-size",
                    "48",
                    "--resume",
                    "--continue-on-error",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
            )
            report = (output_dir / "experiment_report.md").read_text(encoding="utf-8")
            sheet_exists = sheet_path.exists()

        self.assertTrue(sheet_exists)
        self.assertIn("Contact sheet", report)


class ReportRegressionTests(unittest.TestCase):
    def test_baseline_deltas_use_positive_improvement_for_error_drops_and_quality_gains(self):
        rows = [
            {
                "method": "masked",
                "masked_mae_median": "0.50",
                "masked_psnr_median": "8.0",
                "object_surface_chamfer_l1_median": "0.40",
                "silhouette_iou_masked_median": "0.20",
            },
            {
                "method": "mirror",
                "masked_mae_median": "0.20",
                "masked_psnr_median": "12.0",
                "object_surface_chamfer_l1_median": "0.10",
                "silhouette_iou_masked_median": "0.35",
            },
        ]

        deltas = {
            (row["method"], row["metric"]): row
            for row in baseline_delta_rows(rows, baseline_method="masked")
        }

        self.assertAlmostEqual(deltas[("mirror", "Masked MAE")]["delta"], -0.30)
        self.assertAlmostEqual(deltas[("mirror", "Masked MAE")]["improvement"], 0.30)
        self.assertAlmostEqual(deltas[("mirror", "Masked PSNR")]["delta"], 4.0)
        self.assertAlmostEqual(deltas[("mirror", "Masked PSNR")]["improvement"], 4.0)
        self.assertAlmostEqual(deltas[("mirror", "Object Surface Chamfer")]["delta"], -0.30)
        self.assertAlmostEqual(deltas[("mirror", "Object Surface Chamfer")]["improvement"], 0.30)
        self.assertAlmostEqual(deltas[("mirror", "Silhouette IoU")]["delta"], 0.15)
        self.assertAlmostEqual(deltas[("mirror", "Silhouette IoU")]["improvement"], 0.15)

    def test_paired_baseline_deltas_compare_same_sample_ids_and_track_ties(self):
        rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50", "masked_psnr": "8.0"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20", "masked_psnr": "8.0"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.30", "masked_psnr": "9.0"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.40", "masked_psnr": "10.0"},
            {"sample_id": "d", "method": "masked", "masked_mae": "0.70", "masked_psnr": "nan"},
            {"sample_id": "d", "method": "mirror", "masked_mae": "0.60", "masked_psnr": "nan"},
            {"sample_id": "c", "method": "mirror", "masked_mae": "0.10", "masked_psnr": "11.0"},
        ]

        deltas = {
            (row["method"], row["metric"]): row
            for row in paired_baseline_delta_rows(rows, baseline_method="masked")
        }
        mae = deltas[("mirror", "Masked MAE")]
        psnr = deltas[("mirror", "Masked PSNR")]

        self.assertEqual(mae["baseline_n"], 3)
        self.assertEqual(mae["method_n"], 4)
        self.assertEqual(mae["common_n"], 3)
        self.assertEqual(mae["paired_n"], 3)
        self.assertEqual(mae["skipped_nonfinite"], 0)
        self.assertEqual(mae["win_count"], 2)
        self.assertEqual(mae["tie_count"], 0)
        self.assertAlmostEqual(mae["win_rate"], 2 / 3)
        self.assertAlmostEqual(mae["median_delta"], -0.10)
        self.assertAlmostEqual(mae["median_improvement"], 0.10)
        self.assertEqual(psnr["common_n"], 3)
        self.assertEqual(psnr["paired_n"], 2)
        self.assertEqual(psnr["skipped_nonfinite"], 1)
        self.assertEqual(psnr["win_count"], 1)
        self.assertEqual(psnr["tie_count"], 1)
        self.assertAlmostEqual(psnr["median_improvement"], 0.5)

    def test_paired_objective_rows_sum_weighted_sample_improvements_and_boolean_metrics(self):
        rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50", "stl_is_watertight": "False"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20", "stl_is_watertight": "True"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.40", "stl_is_watertight": "True"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.50", "stl_is_watertight": "True"},
            {"sample_id": "c", "method": "mirror", "masked_mae": "0.10", "stl_is_watertight": "True"},
        ]

        objective_rows = paired_objective_rows(
            rows,
            {"masked_mae_median": -4.0, "stl_is_watertight_median": 2.0},
            baseline_method="masked",
            bootstrap_samples=0,
        )

        self.assertEqual(len(objective_rows), 1)
        mirror = objective_rows[0]
        self.assertEqual(mirror["method"], "mirror")
        self.assertEqual(mirror["baseline_n"], 2)
        self.assertEqual(mirror["method_n"], 3)
        self.assertEqual(mirror["common_n"], 2)
        self.assertEqual(mirror["paired_n"], 2)
        self.assertEqual(mirror["win_count"], 1)
        self.assertEqual(mirror["tie_count"], 0)
        self.assertAlmostEqual(mirror["win_rate"], 0.5)
        self.assertAlmostEqual(mirror["median_improvement"], 1.4)
        self.assertAlmostEqual(mirror["mean_improvement"], 1.4)
        self.assertAlmostEqual(mirror["ci95_low"], 1.4)
        self.assertAlmostEqual(mirror["ci95_high"], 1.4)
        self.assertAlmostEqual(mirror["mean_metric_count"], 2.0)

    def test_render_report_emits_paired_objective_confidence_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.md"
            render_report(
                run_dir=Path(temp_dir),
                output_path=output_path,
                summary_rows=[
                    {
                        "method": "masked",
                        "n": "1",
                        "attempted_n": "1",
                        "success_rate": "1",
                        "error_count": "0",
                        "masked_mae_median": "0.50",
                    },
                    {
                        "method": "mirror",
                        "n": "1",
                        "attempted_n": "1",
                        "success_rate": "1",
                        "error_count": "0",
                        "masked_mae_median": "0.20",
                    },
                ],
                per_sample_rows=[
                    {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
                    {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                ],
                failures=[],
                split_audit={},
                ranked_rows=[
                    {
                        "method": "mirror",
                        "rank_score": "1.2",
                        "n": "1",
                        "attempted_n": "1",
                        "success_rate": "1",
                        "error_count": "0",
                        "masked_mae_median": "0.20",
                    },
                    {
                        "method": "masked",
                        "rank_score": "0",
                        "n": "1",
                        "attempted_n": "1",
                        "success_rate": "1",
                        "error_count": "0",
                        "masked_mae_median": "0.50",
                    },
                ],
                used_metrics=["masked_mae_median"],
                top=10,
                examples=0,
                baseline_method="masked",
                score_mode="baseline-delta",
                weights={"masked_mae_median": -4.0},
                paired_bootstrap_samples=0,
                paired_bootstrap_seed=1234,
            )

            report = output_path.read_text(encoding="utf-8")

        self.assertIn("## Paired Objective Confidence: `masked`", report)
        self.assertIn("raw sign-normalized improvement", report)
        self.assertIn("| mirror | 1 | 1 | 1 | 1 | 0 | 1/1 | 1 | 0 | 1.2 | 1.2 | 1.2 | 1.2 | 1 |", report)


class RankMethodRegressionTests(unittest.TestCase):
    def test_baseline_delta_score_is_stable_when_candidates_are_added(self):
        weights = {
            "masked_mae_median": -4.0,
            "masked_psnr_median": 0.15,
        }
        rows = [
            {"method": "masked", "masked_mae_median": "0.60", "masked_psnr_median": "8.0"},
            {"method": "mirror", "masked_mae_median": "0.20", "masked_psnr_median": "12.0"},
        ]
        expanded_rows = rows + [
            {"method": "excellent", "masked_mae_median": "0.10", "masked_psnr_median": "13.0"},
        ]

        normalized, _ = rank_summary_rows(rows, weights, score_mode="normalized")
        expanded_normalized, _ = rank_summary_rows(expanded_rows, weights, score_mode="normalized")
        ranked, used_metrics = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        expanded_ranked, expanded_used_metrics = rank_summary_rows(
            expanded_rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        normalized_scores = {row["method"]: row["rank_score"] for row in normalized}
        expanded_normalized_scores = {row["method"]: row["rank_score"] for row in expanded_normalized}
        scores = {row["method"]: row["rank_score"] for row in ranked}
        expanded_scores = {row["method"]: row["rank_score"] for row in expanded_ranked}

        self.assertNotAlmostEqual(expanded_normalized_scores["mirror"], normalized_scores["mirror"])
        self.assertEqual(used_metrics, ["masked_mae_median", "masked_psnr_median"])
        self.assertEqual(expanded_used_metrics, used_metrics)
        self.assertAlmostEqual(scores["masked"], 0.0)
        self.assertAlmostEqual(scores["mirror"], 2.2)
        self.assertAlmostEqual(expanded_scores["mirror"], scores["mirror"])
        self.assertGreater(expanded_scores["excellent"], expanded_scores["mirror"])

    def test_baseline_delta_score_requires_baseline_method(self):
        with self.assertRaisesRegex(ValueError, "Baseline method `masked` was not found"):
            rank_summary_rows(
                [{"method": "mirror", "masked_mae_median": "0.20"}],
                {"masked_mae_median": -4.0},
                score_mode="baseline-delta",
                baseline_method="masked",
            )

    def test_default_objective_uses_surface_metrics_when_present(self):
        rows = [
            {
                "method": "masked",
                "object_surface_chamfer_l1_median": "0.50",
                "object_surface_chamfer_rmse_median": "0.55",
                "object_surface_hausdorff95_median": "0.70",
            },
            {
                "method": "mirror",
                "object_surface_chamfer_l1_median": "0.20",
                "object_surface_chamfer_rmse_median": "0.25",
                "object_surface_hausdorff95_median": "0.30",
            },
        ]

        ranked, used_metrics = rank_summary_rows(
            rows,
            parse_weights([]),
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        scores = {row["method"]: row["rank_score"] for row in ranked}

        self.assertIn("object_surface_chamfer_l1_median", used_metrics)
        self.assertIn("object_surface_chamfer_rmse_median", used_metrics)
        self.assertIn("object_surface_hausdorff95_median", used_metrics)
        self.assertGreater(scores["mirror"], scores["masked"])

    def test_object_surface_profile_ignores_rgb_only_metrics(self):
        weights = parse_weights([], profile="object-surface")
        rows = [
            {
                "method": "masked",
                "masked_mae_median": "0.50",
                "object_surface_chamfer_l1_median": "0.50",
                "stl_is_watertight_median": "1.0",
                "stl_positive_volume_median": "1.0",
            },
            {
                "method": "rgb_good_surface_bad",
                "masked_mae_median": "0.10",
                "object_surface_chamfer_l1_median": "0.60",
                "stl_is_watertight_median": "1.0",
                "stl_positive_volume_median": "1.0",
            },
            {
                "method": "surface_good_rgb_bad",
                "masked_mae_median": "0.90",
                "object_surface_chamfer_l1_median": "0.20",
                "stl_is_watertight_median": "1.0",
                "stl_positive_volume_median": "1.0",
            },
        ]

        ranked, used_metrics = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        scores = {row["method"]: row["rank_score"] for row in ranked}

        self.assertIn("object_surface_chamfer_l1_median", used_metrics)
        self.assertNotIn("masked_mae_median", used_metrics)
        self.assertGreater(scores["surface_good_rgb_bad"], scores["rgb_good_surface_bad"])
        self.assertGreater(scores["surface_good_rgb_bad"], scores["masked"])

    def test_stl_quality_profile_uses_printability_metrics(self):
        weights = parse_weights([], profile="stl-quality")
        rows = [
            {
                "method": "masked",
                "mesh_surface_chamfer_l1_median": "0.80",
                "mesh_surface_chamfer_rmse_median": "0.90",
                "mesh_surface_hausdorff95_median": "1.20",
                "object_surface_chamfer_l1_median": "0.01",
                "stl_is_watertight_median": "0.0",
                "stl_is_volume_median": "0.0",
                "stl_winding_consistent_median": "0.0",
                "stl_positive_volume_median": "0.0",
                "stl_single_component_median": "0.0",
                "stl_component_count_median": "3",
                "stl_component_excess_median": "2",
                "stl_component_excess_log1p_median": "1.1",
                "stl_bbox_has_volume_median": "0.0",
                "stl_bbox_aspect_ratio_median": "12",
                "stl_faces_per_bbox_volume_median": "1000",
                "stl_faces_per_bbox_volume_log1p_median": "6.9",
            },
            {
                "method": "direct_mesh_good",
                "mesh_surface_chamfer_l1_median": "0.20",
                "mesh_surface_chamfer_rmse_median": "0.25",
                "mesh_surface_hausdorff95_median": "0.35",
                "object_surface_chamfer_l1_median": "0.99",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_winding_consistent_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_component_count_median": "1",
                "stl_component_excess_median": "0",
                "stl_component_excess_log1p_median": "0",
                "stl_bbox_has_volume_median": "1.0",
                "stl_bbox_aspect_ratio_median": "2",
                "stl_faces_per_bbox_volume_median": "20",
                "stl_faces_per_bbox_volume_log1p_median": "3.0",
            },
        ]

        ranked, used_metrics = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        scores = {row["method"]: row["rank_score"] for row in ranked}

        self.assertIn("mesh_surface_chamfer_l1_median", used_metrics)
        self.assertIn("mesh_surface_chamfer_rmse_median", used_metrics)
        self.assertIn("mesh_surface_hausdorff95_median", used_metrics)
        self.assertNotIn("object_surface_chamfer_l1_median", used_metrics)
        self.assertIn("stl_is_volume_median", used_metrics)
        self.assertIn("stl_winding_consistent_median", used_metrics)
        self.assertIn("stl_single_component_median", used_metrics)
        self.assertIn("stl_component_excess_log1p_median", used_metrics)
        self.assertIn("stl_bbox_has_volume_median", used_metrics)
        self.assertIn("stl_faces_per_bbox_volume_log1p_median", used_metrics)
        self.assertNotIn("stl_component_count_median", used_metrics)
        self.assertNotIn("stl_component_excess_median", used_metrics)
        self.assertNotIn("stl_faces_per_bbox_volume_median", used_metrics)
        self.assertGreater(scores["direct_mesh_good"], scores["masked"])


class SelectionRegressionTests(unittest.TestCase):
    def test_selection_promotes_candidate_when_confidence_and_quality_gates_pass(self):
        summary_rows = [
            {
                "method": "masked",
                "success_rate": "1.0",
                "masked_mae_median": "0.50",
                "stl_is_watertight_median": "1.0",
                "stl_positive_volume_median": "1.0",
            },
            {
                "method": "mirror",
                "success_rate": "1.0",
                "masked_mae_median": "0.20",
                "stl_is_watertight_median": "1.0",
                "stl_positive_volume_median": "1.0",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "promote")
        self.assertEqual(decision["candidate_method"], "mirror")
        self.assertFalse(decision["failed_checks"])
        self.assertAlmostEqual(decision["candidate_rank_score"], 1.2)
        self.assertAlmostEqual(decision["paired_objective"]["ci95_low"], 1.6)

    def test_selection_holds_candidate_that_does_not_beat_current_method(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.20"},
            {"method": "learned", "success_rate": "1.0", "masked_mae_median": "0.30"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "a", "method": "learned", "masked_mae": "0.30"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "b", "method": "learned", "masked_mae": "0.30"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="learned",
            current_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("score_margin_vs_current", {check["name"] for check in decision["failed_checks"]})
        self.assertLess(decision["candidate_rank_score"], decision["current_rank_score"])

    def test_selection_holds_candidate_without_paired_confidence_vs_current(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.30"},
            {"method": "learned", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "a", "method": "learned", "masked_mae": "0.25"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "b", "method": "learned", "masked_mae": "0.25"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="learned",
            current_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed_checks = {check["name"] for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], decision["current_rank_score"])
        self.assertIn("paired_win_rate_vs_current", failed_checks)
        self.assertIn("paired_ci95_low_vs_current", failed_checks)
        self.assertLess(decision["paired_objective_vs_current"]["mean_improvement"], 0)

    def test_selection_holds_lora_candidate_without_training_report(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {
                "method": "learned",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "lora_weights": "backend/output/completion-benchmark/lora/missing-report",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "learned", "masked_mae": "0.10"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "b", "method": "learned", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="learned",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            min_win_rate=1.0,
            min_ci95_low=0.0,
            split_audit={
                "lora_weights": "backend/output/completion-benchmark/lora/missing-report",
                "train_eval_asset_overlap_count": 0,
            },
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("training_report_present", {check["name"] for check in decision["failed_checks"]})
        self.assertEqual(
            decision["training_provenance"]["lora_weights"],
            "backend/output/completion-benchmark/lora/missing-report",
        )

    def test_selection_holds_lora_candidate_with_prompt_family_mismatch(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {
                "method": "learned",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "lora_weights": "backend/output/completion-benchmark/lora/adapter",
                "training_report": "backend/output/completion-benchmark/lora/adapter/training_report.json",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "learned", "masked_mae": "0.10"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "b", "method": "learned", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="learned",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            min_win_rate=1.0,
            min_ci95_low=0.0,
            split_audit={
                "lora_weights": "backend/output/completion-benchmark/lora/adapter",
                "training_report": "backend/output/completion-benchmark/lora/adapter/training_report.json",
                "train_eval_asset_overlap_count": 0,
                "prompt_family_matches_eval": False,
                "prompt_family_mismatch_detail": "train_prompt_template differs from eval_prompt_template",
            },
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("prompt_family_matches_eval", {check["name"] for check in decision["failed_checks"]})

    def test_selection_promotes_candidate_that_beats_current_paired_gate(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.30"},
            {"method": "learned", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.30"},
            {"sample_id": "a", "method": "learned", "masked_mae": "0.10"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.40"},
            {"sample_id": "b", "method": "learned", "masked_mae": "0.20"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="learned",
            current_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "promote")
        self.assertFalse(decision["failed_checks"])
        self.assertGreater(decision["paired_objective_vs_current"]["ci95_low"], 0)

    def test_selection_keeps_current_when_top_candidate_is_current_method(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
            {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.20"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "mirror", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="mirror",
            current_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "keep_current")
        self.assertFalse(decision["failed_checks"])
        self.assertNotIn("score_margin_vs_current", {check["name"] for check in decision["checks"]})

    def test_selection_holds_when_current_method_is_missing(self):
        decision = evaluate_selection(
            [
                {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
                {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.20"},
            ],
            [
                {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
                {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
                {"sample_id": "b", "method": "mirror", "masked_mae": "0.10"},
            ],
            baseline_method="masked",
            candidate_method="mirror",
            current_method="missing",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("current_method_present", {check["name"] for check in decision["failed_checks"]})

    def test_selection_requires_split_audit_by_default(self):
        decision = evaluate_selection(
            [
                {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
                {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.20"},
            ],
            [
                {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
                {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
                {"sample_id": "b", "method": "mirror", "masked_mae": "0.10"},
            ],
            baseline_method="masked",
            candidate_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("split_audit_present", {check["name"] for check in decision["failed_checks"]})

    def test_selection_json_safe_replaces_nonfinite_values(self):
        decision = evaluate_selection(
            [
                {"method": "masked", "masked_mae_median": "0.50"},
                {"method": "mirror", "masked_mae_median": "0.20"},
            ],
            [
                {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
                {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
            ],
            baseline_method="masked",
            candidate_method="mirror",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        encoded = json.dumps(json_safe(decision), allow_nan=False)
        self.assertIn('"value": null', encoded)


if __name__ == "__main__":
    unittest.main()
