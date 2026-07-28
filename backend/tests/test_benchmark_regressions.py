import csv
import hashlib
import io
import json
import math
import os
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
from backend.benchmark import (
    colab_g4_orchestrator,
    direct_mesh,
    run_completion_benchmark,
    run_image_to_mesh_provider,
)
from backend.benchmark.combine_optimize_runs import combine_runs, write_selection as write_combined_selection
from backend.benchmark.compare_optimize_runs import add_score_deltas, compare_run, render_markdown
from backend.benchmark.direct_mesh import (
    bbox_extent_comparison_metrics,
    direct_mesh_input_path,
    max_faces_for_normalized_bbox_complexity,
    mesh_is_printable_volume,
    postprocess_mesh_for_stl,
    repair_mesh_for_printable_stl,
    run_direct_mesh,
)
from backend.benchmark.explain_rank_score import contribution_rows, explain, markdown_report, summarize_contributions
from backend.benchmark.explain_paired_objective import (
    explain as explain_paired_objective,
    paired_sample_rows,
)
from backend.benchmark.extract_colab_output_summary import extract_colab_output, parse_colab_output
from backend.benchmark.export_training_pairs import main as export_training_pairs_main
from backend.benchmark.generate_rendered_dataset import attach_multiview_fields, generate_dataset
from backend.benchmark.ingest_stl_results import (
    discover_result_runs,
    render_markdown as render_stl_ingest_markdown,
    summarize_inputs as summarize_stl_inputs,
)
from backend.benchmark.package_colab_inputs import (
    build_compact_results_archive,
    build_fetch_colab_launcher,
    build_inline_colab_launcher,
    package_inputs,
    parse_colab_env,
)
from backend.benchmark.optimize_completion import (
    annotate_per_sample_metrics,
    experiment_metadata,
    infer_selection_candidate,
    load_experiments,
    run_experiment,
    training_metadata,
    write_image_to_mesh_provider_preflight,
    write_experiment_report,
    write_modern_cache_preflight,
    write_selection_decision,
)
from backend.benchmark.preflight_image_to_mesh_providers import parse_provider_command, provider_preflight_row
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
from backend.benchmark.rank_methods import parse_weights, rank_summary_rows, with_derived_metrics
from backend.benchmark.run_image_to_mesh_provider import main as run_image_to_mesh_provider_main
from backend.benchmark.run_stl_first_smoke import (
    build_experiments as build_stl_first_experiments,
    build_optimize_command as build_stl_first_optimize_command,
    method_failure_rows,
    shell_token as stl_first_shell_token,
    write_config_only as write_stl_first_config_only,
    write_architecture_report as write_stl_first_architecture_report,
)
from backend.benchmark.run_triposr_repair_smoke import build_experiments, rows_from_csv
from backend.benchmark.run_completion_benchmark import evaluate_direct_mesh_sample, evaluate_sample, run_one, stl_diagnostics, summarize, write_split_audit
from backend.benchmark.select_completion_candidate import evaluate_selection, json_safe
from backend.benchmark.train_inpainting_lora import InpaintPairDataset, collate, dry_run, read_metadata, weighted_mse_loss
from backend.benchmark.triposg_models import (
    DEFAULT_TRIPOSG_MODEL_REVISION,
    DEFAULT_TRIPOSG_REMBG_REVISION,
)
from backend.benchmark.weight_training_pairs import weight_metadata_rows
from backend.stl_diagnostics import mesh_diagnostics
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
    def stl_first_args(self, **overrides):
        defaults = {
            "include_source_oracle": True,
            "include_triposr_api": False,
            "include_raw_direct_mesh": False,
            "triposr_direct_inputs": ["masked"],
            "include_hunyuan3d_shape": False,
            "include_triposg": False,
            "include_source_multiview_oracle": False,
            "include_visual_hull_multiview": False,
            "multiview_command": None,
            "multiview_name": "mv_recon",
            "multiview_primary_input": "masked",
            "multiview_output_ext": "ply",
            "visual_hull_resolution": 40,
            "visual_hull_grid_extent": 1.7,
            "visual_hull_ortho_scale": 2.0,
            "visual_hull_mask_dilate": 2,
            "provider_python": "python",
            "provider_device": "cuda",
            "triposr_python": "/content/triposr-venv/bin/python",
            "triposr_dir": "/content/TripoSR",
            "hunyuan3d_dir": "/content/Hunyuan3D",
            "triposg_python": "/content/triposg-venv/bin/python",
            "triposg_dir": "/content/TripoSG",
            "triposg_direct_inputs": ["masked"],
            "hunyuan_num_inference_steps": 24,
            "hunyuan_guidance_scale": 4.0,
            "hunyuan_octree_resolution": 192,
            "hunyuan_num_chunks": 4096,
            "hunyuan_low_vram": True,
            "triposg_num_inference_steps": 8,
            "triposg_guidance_scale": 3.5,
            "triposg_seed": None,
            "chunk_size": 256,
            "mc_resolution": 64,
            "mesh_repair": "printable",
            "mesh_target_max_dimension": 0.0,
            "mesh_min_bbox_dimension": 0.0,
            "mesh_max_bbox_aspect_ratio": 0.0,
            "mesh_target_bbox_source": "none",
            "direct_mesh_reference_method": "mirror",
            "mesh_target_faces": 0,
            "mesh_max_normalized_face_density_log1p": 0.0,
            "direct_mesh_timeout": 123,
            "write_config": "",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

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
        self.assertGreater(diagnostics["stl_faces_per_normalized_bbox_volume"], 0)
        self.assertGreater(diagnostics["stl_faces_per_normalized_bbox_volume_log1p"], 0)

    def test_stl_diagnostics_scale_free_complexity_ignores_coordinate_units(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            unit_path = root / "unit_box.stl"
            large_path = root / "large_box.stl"
            trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(unit_path)
            trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(large_path)

            unit = stl_diagnostics(unit_path)
            large = stl_diagnostics(large_path)

        self.assertNotAlmostEqual(
            unit["stl_faces_per_bbox_volume_log1p"],
            large["stl_faces_per_bbox_volume_log1p"],
        )
        self.assertAlmostEqual(
            unit["stl_faces_per_normalized_bbox_volume_log1p"],
            large["stl_faces_per_normalized_bbox_volume_log1p"],
        )

    def test_mesh_diagnostics_supports_raw_prefix_and_normalized_volume_change(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.glb"
            trimesh.creation.box(extents=(2.0, 3.0, 4.0)).export(raw_path)
            diagnostics = mesh_diagnostics(raw_path, prefix="raw_mesh")

        self.assertTrue(diagnostics["raw_mesh_exists"])
        self.assertAlmostEqual(diagnostics["raw_mesh_volume_fill_ratio"], 1.0)
        self.assertFalse(diagnostics["raw_mesh_self_intersection_supported"])
        self.assertTrue(math.isnan(diagnostics["raw_mesh_self_intersection_count"]))

    def test_mesh_diagnostics_fill_reliability_and_surface_proxy_fixtures(self):
        import warnings

        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            closed = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
            reversed_mesh = closed.copy()
            reversed_mesh.invert()
            open_mesh = closed.copy()
            open_mesh.update_faces(open_mesh.face_normals[:, 0] < 0.9)
            open_mesh.remove_unreferenced_vertices()
            thin = trimesh.creation.box(extents=(4.0, 0.1, 4.0))
            tiny = closed.copy()
            tiny.apply_scale(1e-5)
            second = closed.copy()
            second.apply_translation((5.0, 0.0, 0.0))
            fragmented = trimesh.util.concatenate((closed.copy(), second))
            reversed_second = closed.copy()
            reversed_second.invert()
            reversed_second.apply_translation((5.0, 0.0, 0.0))
            fragmented_mixed_winding = trimesh.util.concatenate(
                (closed.copy(), reversed_second)
            )
            touching_second = closed.copy()
            touching_second.apply_translation((2.0, 3.0, 4.0))
            point_touching = trimesh.util.concatenate((closed.copy(), touching_second))
            overlapping_second = closed.copy()
            overlapping_second.apply_translation((0.2, 0.0, 0.0))
            overlapping = trimesh.util.concatenate((closed.copy(), overlapping_second))
            meshes = {
                "closed": closed,
                "reversed": reversed_mesh,
                "open": open_mesh,
                "thin": thin,
                "tiny": tiny,
                "fragmented": fragmented,
                "fragmented_mixed": fragmented_mixed_winding,
                "point_touching": point_touching,
                "overlapping": overlapping,
            }
            diagnostics = {}
            for name, mesh in meshes.items():
                path = root / f"{name}.ply"
                mesh.export(path)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    diagnostics[name] = mesh_diagnostics(
                        path,
                        prefix=name,
                        include_surface_fill_proxy=True,
                    )
            closed_path = root / "closed.ply"
            with patch("backend.stl_diagnostics.MAX_SIGNED_VOLUME_RELIABILITY_FACES", 1):
                diagnostics["unknown"] = mesh_diagnostics(
                    closed_path,
                    prefix="unknown",
                    include_topology=False,
                    include_surface_fill_proxy=True,
                )
            open_path = root / "open.ply"
            with patch("backend.stl_diagnostics.MAX_SURFACE_FILL_PROXY_FACES", 1):
                diagnostics["proxy_limited"] = mesh_diagnostics(
                    open_path,
                    prefix="proxy_limited",
                    include_surface_fill_proxy=True,
                )

        self.assertTrue(diagnostics["closed"]["closed_volume_fill_ratio_reliable"])
        self.assertIn(
            ":positive",
            diagnostics["closed"]["closed_volume_fill_ratio_reliability_reason"],
        )
        self.assertTrue(diagnostics["reversed"]["reversed_volume_fill_ratio_reliable"])
        self.assertIn(
            ":reversed",
            diagnostics["reversed"]["reversed_volume_fill_ratio_reliability_reason"],
        )
        self.assertFalse(diagnostics["open"]["open_volume_fill_ratio_reliable"])
        self.assertEqual(
            diagnostics["open"]["open_volume_fill_ratio_reliability_reason"],
            "not-watertight",
        )
        self.assertTrue(diagnostics["thin"]["thin_volume_fill_ratio_reliable"])
        self.assertTrue(diagnostics["tiny"]["tiny_volume_fill_ratio_reliable"])
        self.assertFalse(diagnostics["fragmented"]["fragmented_volume_fill_ratio_reliable"])
        self.assertEqual(
            diagnostics["fragmented"]["fragmented_volume_fill_ratio_reliability_reason"],
            "component-count:2",
        )
        self.assertFalse(
            diagnostics["fragmented_mixed"][
                "fragmented_mixed_volume_fill_ratio_reliable"
            ]
        )
        self.assertEqual(
            diagnostics["point_touching"][
                "point_touching_volume_fill_ratio_reliability_reason"
            ],
            "component-count:2",
        )
        self.assertEqual(
            diagnostics["unknown"]["unknown_volume_fill_ratio_reliability_status"],
            "unknown",
        )
        self.assertFalse(
            diagnostics["unknown"]["unknown_volume_fill_ratio_topology_assessed"]
        )
        self.assertEqual(
            diagnostics["unknown"]["unknown_volume_fill_ratio_reliability_reason"],
            "topology-not-assessed-face-limit",
        )
        for name in (
            "open",
            "fragmented",
            "fragmented_mixed",
            "point_touching",
            "overlapping",
            "unknown",
        ):
            self.assertTrue(diagnostics[name][f"{name}_surface_fill_ratio_supported"])
            self.assertGreater(diagnostics[name][f"{name}_surface_fill_ratio"], 0.0)
            self.assertLessEqual(diagnostics[name][f"{name}_surface_fill_ratio"], 1.0)
        for name in ("closed", "reversed", "thin", "tiny"):
            self.assertNotIn(f"{name}_surface_fill_ratio", diagnostics[name])
        self.assertAlmostEqual(
            diagnostics["open"]["open_surface_fill_ratio"],
            5.0 / 6.0,
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics["fragmented"]["fragmented_surface_fill_ratio"],
            diagnostics["fragmented_mixed"]["fragmented_mixed_surface_fill_ratio"],
            places=6,
        )
        self.assertEqual(diagnostics["overlapping"]["overlapping_surface_fill_ratio"], 1.0)
        self.assertGreater(
            diagnostics["overlapping"]["overlapping_surface_fill_ratio_comparison"],
            1.0,
        )
        self.assertTrue(
            diagnostics["overlapping"]["overlapping_surface_fill_proxy_clipped"]
        )
        self.assertFalse(
            diagnostics["proxy_limited"][
                "proxy_limited_surface_fill_ratio_supported"
            ]
        )
        self.assertIn(
            "bounded face limit",
            diagnostics["proxy_limited"]["proxy_limited_surface_fill_ratio_error"],
        )

    def test_derived_fill_metric_preserves_legacy_replay(self):
        row = with_derived_metrics(
            {
                "repair_volume_fill_ratio_relative_change_abs": "0.25",
                "repair_volume_fill_ratio_relative_change_abs_median": "0.5",
            }
        )

        self.assertEqual(row["repair_fill_ratio_relative_change_abs"], 0.25)
        self.assertEqual(row["repair_fill_ratio_relative_change_abs_median"], 0.5)
        self.assertEqual(row["repair_fill_ratio_metric"], "legacy-signed-volume-fallback")

        unsupported_v2 = with_derived_metrics(
            {
                "repair_fill_ratio_supported": False,
                "repair_volume_fill_ratio_relative_change_abs": "0.1",
            }
        )
        self.assertNotIn("repair_fill_ratio_relative_change_abs", unsupported_v2)

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
        self.assertEqual(rows[0]["stl_mode"], "source-mesh-oracle")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertEqual(rows[0]["stl_is_volume"], "True")
        self.assertEqual(rows[0]["stl_single_component"], "True")
        self.assertIn("output_model.stl", rows[0]["stl_model"])
        self.assertAlmostEqual(float(rows[0]["mesh_surface_chamfer_l1"]), 0.0)
        self.assertEqual(summary[0]["n"], "1")
        self.assertAlmostEqual(float(summary[0]["mesh_surface_chamfer_l1_median"]), 0.0)

    def test_source_mesh_oracle_can_repair_unprintable_source_mesh(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            mesh_path = root / "open_box.ply"
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            keep_faces = np.ones(len(mesh.faces), dtype=bool)
            keep_faces[-2:] = False
            mesh.update_faces(keep_faces)
            mesh.export(mesh_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "open_box",
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
                    "--source-mesh-repair",
                    "printable",
                    "--mesh-surface-max-points",
                    "128",
                ],
            ):
                run_completion_benchmark.main()

            with (output_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            repaired_mesh = Path(rows[0]["direct_mesh_output_mesh"])
            repaired_mesh_exists = repaired_mesh.exists()

        self.assertEqual(rows[0]["method"], "source-mesh-oracle")
        self.assertEqual(rows[0]["source_mesh_repair"], "printable")
        self.assertEqual(repaired_mesh.name, "source_mesh_repaired.ply")
        self.assertTrue(repaired_mesh_exists)
        self.assertEqual(rows[0]["stl_is_watertight"], "True")
        self.assertEqual(rows[0]["stl_is_volume"], "True")
        self.assertEqual(rows[0]["stl_is_manifold"], "True")
        self.assertEqual(rows[0]["stl_single_component"], "True")
        self.assertEqual(rows[0]["stl_positive_volume"], "True")

    def test_run_completion_benchmark_limit_zero_writes_empty_outputs_without_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "sample_0", "masked_image": "missing.png"}),
                        json.dumps({"id": "sample_1", "masked_image": "missing.png"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "run"

            with patch.object(run_completion_benchmark, "run_one", side_effect=AssertionError("run_one should not execute")):
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
                        "masked",
                        "--limit",
                        "0",
                    ],
                ):
                    run_completion_benchmark.main()

            with (output_dir / "split_audit.json").open(encoding="utf-8") as audit_file:
                audit = json.load(audit_file)
            with (output_dir / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            with (output_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                summary = list(csv.DictReader(csv_file))

        self.assertEqual(audit["limit"], 0)
        self.assertEqual(audit["eval_n"], 0)
        self.assertEqual(rows, [])
        self.assertEqual(summary[0]["method"], "masked")
        self.assertEqual(summary[0]["attempted_n"], "0")
        self.assertEqual(summary[0]["n"], "0")

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
                        "assert sys.argv[3] == '96,72,48', sys.argv[3]",
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
            command = f'"{sys.executable}" "{script_path}" "{{output_mesh}}" "{{output_stl}}" "{{source_bbox_extents}}"'

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
                    "--stl-target-dimension",
                    "96",
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
            with (output_dir / "summary_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                summary_rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["method"], "external-image-to-mesh")
        self.assertEqual(rows[0]["stl_mode"], "single-image-mesh")
        self.assertEqual(rows[0]["direct_mesh_bbox_source"], "source")
        self.assertEqual(rows[0]["oracle_diagnostic"], "True")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertIn("output_model.stl", rows[0]["stl_model"])
        self.assertIn("output_mesh.ply", rows[0]["direct_mesh_output_mesh"])
        self.assertAlmostEqual(float(rows[0]["mesh_surface_chamfer_l1"]), 0.0)
        self.assertEqual(summary_rows[0]["direct_mesh_bbox_source"], "source")
        self.assertEqual(summary_rows[0]["oracle_diagnostic"], "True")

    def test_external_image_to_mesh_command_can_use_reference_mirror_bbox(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            mesh_path = root / "reference_box.ply"
            script_path = root / "fake_image_to_mesh.py"
            sweep_dir = root / "sweep"
            mirror_stl = sweep_dir / "mirror" / "box" / "mirror" / "output_model.stl"
            mirror_stl.parent.mkdir(parents=True)
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(mesh_path)
            trimesh.creation.box(extents=(96.0, 48.0, 24.0)).export(mirror_stl)
            script_path.write_text(
                "\n".join(
                    [
                        "import sys",
                        "import trimesh",
                        "assert sys.argv[3] == '96,48,24', sys.argv[3]",
                        "assert sys.argv[4] == '96,48,24', sys.argv[4]",
                        "assert sys.argv[5].replace('\\\\', '/').endswith('mirror/output_model.stl'), sys.argv[5]",
                        "assert sys.argv[6] == 'mirror', sys.argv[6]",
                        "assert sys.argv[7] == '96,48,24', sys.argv[7]",
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
            command = (
                f'"{sys.executable}" "{script_path}" "{{output_mesh}}" "{{output_stl}}" '
                '"{mirror_bbox_extents}" "{reference_bbox_extents}" "{reference_stl}" "{reference_method}" '
                '"{inferred_bbox_extents}"'
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_completion_benchmark",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(sweep_dir / "candidate"),
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
                    "--direct-mesh-reference-output-dir",
                    str(sweep_dir),
                    "--direct-mesh-reference-method",
                    "mirror",
                    "--mesh-surface-max-points",
                    "128",
                ],
            ):
                run_completion_benchmark.main()

            with (sweep_dir / "candidate" / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["method"], "external-image-to-mesh")
        self.assertEqual(rows[0]["stl_exists"], "True")
        self.assertIn("output_model.stl", rows[0]["stl_model"])
        self.assertGreater(float(rows[0]["inferred_bbox_centered_iou"]), 0.0)

    def test_external_image_to_mesh_inferred_bbox_fails_closed_without_mirror(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            Image.new("RGB", (12, 12), (80, 120, 160)).save(full)
            Image.new("RGB", (12, 12), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(mask)
            sample = {
                "id": "missing-mirror",
                "full_image": str(full),
                "masked_image": str(masked),
                "mask": str(mask),
            }
            args = SimpleNamespace(
                direct_mesh_input="full",
                direct_mesh_output_ext="ply",
                direct_mesh_command='provider --bbox "{inferred_bbox_extents}"',
                direct_mesh_reference_output_dir=str(root / "reference"),
                direct_mesh_reference_method="mirror",
                stl_target_dimension=96.0,
            )

            with self.assertRaisesRegex(RuntimeError, "valid inferred bbox"):
                run_direct_mesh(
                    sample,
                    "external-image-to-mesh",
                    root / "candidate",
                    args,
                )

    def test_bbox_extent_comparison_metrics_separate_shape_from_absolute_scale(self):
        metrics = bbox_extent_comparison_metrics(
            (96.0, 72.0, 48.0),
            (48.0, 36.0, 24.0),
        )

        self.assertAlmostEqual(metrics["inferred_bbox_shape_log_mae"], 0.0)
        self.assertAlmostEqual(metrics["inferred_bbox_shape_relative_mae"], 0.0)
        self.assertAlmostEqual(metrics["inferred_bbox_centered_iou"], 1.0)

    def test_standalone_reference_to_source_oracle_bbox_is_diagnostic(self):
        args = SimpleNamespace(
            prompt=None,
            steps=None,
            guidance=None,
            seed=None,
            inpaint_max_dimension=None,
            edit_mask_fill="input",
            model_name=None,
            lora_weights=None,
            lora_scale=None,
            source_mesh_repair="none",
            direct_mesh_command='provider --mesh-target-bbox-extents "{reference_bbox_extents}"',
            direct_mesh_reference_method="source_mesh_oracle",
        )

        key = run_completion_benchmark.experiment_key(
            {"id": "sample"},
            "external-image-to-mesh",
            args,
        )

        self.assertEqual(key["direct_mesh_bbox_source"], "reference")
        self.assertEqual(key["direct_mesh_reference_method"], "source_mesh_oracle")
        self.assertTrue(key["oracle_diagnostic"])

    def test_resume_filters_stale_source_bbox_rows_before_inferred_rerun(self):
        common = {
            "prompt": None,
            "steps": None,
            "guidance": None,
            "seed": None,
            "inpaint_max_dimension": None,
            "edit_mask_fill": "input",
            "model_name": None,
            "lora_weights": None,
            "lora_scale": None,
            "source_mesh_repair": "none",
            "direct_mesh_reference_method": "mirror",
            "direct_mesh_input": "biharmonic",
            "direct_mesh_output_ext": "glb",
            "direct_mesh_reference_output_dir": None,
            "stl_target_dimension": 96,
            "mesh_surface_max_points": 128,
        }
        source_args = SimpleNamespace(
            **common,
            direct_mesh_command=(
                'provider --bbox "{source_bbox_x},{source_bbox_y},{source_bbox_z}"'
            ),
        )
        inferred_args = SimpleNamespace(
            **common,
            direct_mesh_command='provider --bbox "{inferred_bbox_extents}"',
        )
        source_row = run_completion_benchmark.experiment_key(
            {"id": "sample"},
            "external-image-to-mesh",
            source_args,
        )
        inferred_key = run_completion_benchmark.experiment_key(
            {"id": "sample"},
            "external-image-to-mesh",
            inferred_args,
        )

        filtered = run_completion_benchmark.filter_resume_rows(
            [source_row],
            {run_completion_benchmark.key_tuple(inferred_key)},
        )

        self.assertEqual(source_row["direct_mesh_bbox_source"], "source")
        self.assertTrue(source_row["oracle_diagnostic"])
        self.assertNotEqual(source_row["direct_mesh_config_hash"], inferred_key["direct_mesh_config_hash"])
        self.assertEqual(filtered, [])

    def test_optimize_direct_mesh_experiment_passes_reference_sweep_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "sweep"
            captured = []
            args = SimpleNamespace(
                manifest=str(root / "manifest.jsonl"),
                limit=1,
                start_index=0,
                depth_provider="depth-anything-v2",
                depth_model="depth-anything/Depth-Anything-V2-Small-hf",
                device="cpu",
                skip_depth=True,
                emit_stl=True,
                stl_no_invert=False,
                resume=True,
                continue_on_error=True,
                prompt="",
                steps=1,
                guidance=None,
                seed=123,
                inpaint_max_dimension=64,
                edit_mask_fill="input",
                model_name=None,
                lora_weights=None,
                lora_scale=None,
                stl_target_dimension=96,
                stl_z_scale=50.0,
                stl_sigma=4.0,
                mesh_surface_max_points=128,
                direct_mesh_input="masked",
                direct_mesh_command='python fake.py "{input_image}" "{output_mesh}"',
                direct_mesh_output_ext="ply",
                direct_mesh_timeout=456,
                direct_mesh_reference_output_dir=None,
                direct_mesh_reference_method="mirror",
                source_mesh_repair="none",
                max_method_failures=0,
            )

            with patch("backend.benchmark.optimize_completion.run", side_effect=lambda command: captured.append(command)):
                summary_path = run_experiment(
                    args,
                    {"name": "direct_candidate", "method": "external-image-to-mesh"},
                    output_dir,
                )

        self.assertEqual(summary_path, output_dir / "direct_candidate" / "summary_metrics.csv")
        self.assertEqual(len(captured), 1)
        command = captured[0]
        self.assertIn("--direct-mesh-reference-output-dir", command)
        self.assertEqual(command[command.index("--direct-mesh-reference-output-dir") + 1], str(output_dir))
        self.assertIn("--direct-mesh-reference-method", command)
        self.assertEqual(command[command.index("--direct-mesh-reference-method") + 1], "mirror")

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
        self.assertEqual(bundle["source_mesh"], str(mesh_path))
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

    def test_procedural_rendered_dataset_can_emit_multiview_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = generate_dataset(
                output_dir=Path(temp_dir),
                source="procedural",
                count=4,
                size=32,
                seed=123,
                views_per_asset=2,
            )
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["asset_key"], rows[1]["asset_key"])
        self.assertEqual(len(rows[0]["multiview_images"]), 2)
        self.assertEqual(rows[1]["multiview_primary_index"], 1)
        self.assertNotEqual(rows[0]["camera"], rows[1]["camera"])

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
            provider_metrics = json.loads(
                (root / "provider_metrics.json").read_text(encoding="utf-8")
            )

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])
        self.assertEqual(provider_metrics["status"], "ok")
        self.assertFalse(provider_metrics["provider_cache_hit"])
        self.assertGreaterEqual(provider_metrics["provider_inference_runtime_seconds"], 0.0)
        self.assertEqual(provider_metrics["provider_peak_cuda_vram_gib"], None)
        self.assertFalse(provider_metrics["provider_peak_cuda_vram_supported"])

    def test_triposg_provider_wrapper_uses_module_cli(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_triposg"
            scripts_dir = provider_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            input_image = root / "input.png"
            output_mesh = root / "normalized.glb"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (12, 12), (120, 80, 160)).save(input_image)
            (scripts_dir / "__init__.py").write_text("", encoding="utf-8")
            (scripts_dir / "inference_triposg.py").write_text(
                "\n".join(
                    [
                        "import argparse, json, sys",
                        "from pathlib import Path",
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--image-input', required=True)",
                        "parser.add_argument('--output-path', required=True)",
                        "parser.add_argument('--num-inference-steps', type=int, default=50)",
                        "parser.add_argument('--guidance-scale', type=float, default=7.0)",
                        "parser.add_argument('--faces', type=int, default=40000)",
                        "parser.add_argument('--seed', type=int, default=None)",
                        "args = parser.parse_args()",
                        "assert '--device' not in sys.argv",
                        "assert args.num_inference_steps == 7",
                        "assert args.guidance_scale == 3.5",
                        "assert args.faces == 123",
                        "assert args.seed == 9",
                        "Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(args.output_path)",
                        "Path(args.output_path).with_name('args.json').write_text(json.dumps(vars(args)))",
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
                    "triposg",
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
                    "--provider-device",
                    "cuda",
                    "--num-inference-steps",
                    "7",
                    "--guidance-scale",
                    "3.5",
                    "--seed",
                    "9",
                    "--mesh-target-faces",
                    "123",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            raw_args = json.loads((root / "triposg_raw" / "args.json").read_text(encoding="utf-8"))
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertEqual(raw_args["image_input"], str(input_image))
        self.assertEqual(raw_args["faces"], 123)
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

    def test_voxel_close_repair_preserves_concavity_under_face_budget(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            broken_path = root / "open_torus.ply"
            repaired_path = root / "repaired.stl"
            torus = trimesh.creation.torus(
                major_radius=1.0,
                minor_radius=0.32,
                major_sections=64,
                minor_sections=32,
            )
            keep_faces = np.ones(len(torus.faces), dtype=bool)
            centers = np.asarray(torus.triangles_center)
            keep_faces[(centers[:, 0] > 1.15) & (centers[:, 2] > 0.0)] = False
            broken = torus.submesh([np.flatnonzero(keep_faces)], append=True, repair=False)
            broken.export(broken_path)
            repair_metrics = {}

            with patch.object(
                direct_mesh,
                "_clean_mesh",
                side_effect=AssertionError("printable voxel mesh should bypass generic cleanup"),
            ):
                repair_mesh_for_printable_stl(
                    broken_path,
                    repaired_path,
                    mode="printable",
                    target_faces=5_000,
                    preconditioner="voxel-close",
                    voxel_resolution=96,
                    voxel_fill_method="orthographic",
                    metrics=repair_metrics,
                )

            repaired = trimesh.load_mesh(repaired_path, force="mesh")
            diagnostics = stl_diagnostics(repaired_path)

        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertTrue(diagnostics["stl_single_component"])
        self.assertLessEqual(len(repaired.faces), 5_000)
        self.assertLess(abs(float(repaired.volume) - float(torus.volume)), 0.2)
        self.assertLess(float(repaired.volume), float(torus.convex_hull.volume) * 0.8)
        self.assertFalse(repair_metrics["repair_convex_hull_used"])
        self.assertTrue(repair_metrics["repair_preclean_printable"])
        self.assertTrue(repair_metrics["repair_cleaning_skipped"])
        self.assertNotIn("repair_cleaned_printable", repair_metrics)
        self.assertGreater(repair_metrics["repair_precondition_voxel_faces"], 5_000)
        self.assertLessEqual(repair_metrics["repair_simplified_faces"], 5_000)
        self.assertEqual(repair_metrics["repair_simplify_placement"], "optimal")
        self.assertEqual(repair_metrics["repair_simplification_target_faces"], 5_000)
        self.assertTrue(repair_metrics["repair_simplification_applied"])
        self.assertTrue(repair_metrics["repair_simplification_audit_available"])
        self.assertGreater(
            repair_metrics["repair_simplification_audit_runtime_seconds"],
            0.0,
        )
        self.assertIn(
            "repair_simplification_surface_chamfer_l1_normalized",
            repair_metrics,
        )
        self.assertIn(
            "repair_simplification_surface_hausdorff95_normalized",
            repair_metrics,
        )
        self.assertNotIn("repair_simplification_geometry_preserved", repair_metrics)
        self.assertNotIn(
            "repair_simplification_vertex_displacement_max_normalized",
            repair_metrics,
        )

    def test_topology_preserving_simplification_supports_endpoint_placement(self):
        import trimesh
        from scipy.spatial import cKDTree

        torus = trimesh.creation.torus(
            major_radius=1.0,
            minor_radius=0.32,
            major_sections=48,
            minor_sections=24,
        )
        voxel_mesh = direct_mesh._voxel_close_mesh(torus, 64, "orthographic")
        simplified = direct_mesh._simplify_preserving_topology(
            voxel_mesh,
            2_000,
            placement="endpoint",
            strict=True,
        )
        nearest_source_distance, _ = cKDTree(voxel_mesh.vertices).query(
            simplified.vertices,
            k=1,
        )

        self.assertLessEqual(len(simplified.faces), 2_000)
        self.assertLessEqual(float(np.max(nearest_source_distance)), 1e-12)
        with self.assertRaisesRegex(ValueError, "simplification placement"):
            direct_mesh._simplify_preserving_topology(
                voxel_mesh,
                2_000,
                placement="centroid",
                strict=True,
            )

    def test_component_close_relaxes_open_boundaries_only_after_strict_stall(self):
        import trimesh

        side = 41
        vertices = np.asarray(
            [
                (x / (side - 1), y / (side - 1), 0.0)
                for y in range(side)
                for x in range(side)
            ],
            dtype=np.float64,
        )
        faces = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                b = a + 1
                c = a + side
                d = c + 1
                faces.extend(((a, b, d), (a, d, c)))
        open_grid = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )

        with self.assertRaisesRegex(RuntimeError, "could not reach"):
            direct_mesh._simplify_preserving_topology(
                open_grid,
                100,
                placement="optimal",
                strict=True,
            )

        repair_metrics = {}
        simplified = direct_mesh._simplify_preserving_topology(
            open_grid,
            100,
            placement="optimal",
            strict=True,
            allow_boundary_relaxation=True,
            metrics=repair_metrics,
        )

        self.assertLessEqual(len(simplified.faces), 100)
        self.assertGreater(
            repair_metrics["repair_simplification_topology_preserving_faces"],
            100,
        )
        self.assertGreater(
            repair_metrics["repair_simplification_boundary_preserving_faces"],
            100,
        )
        self.assertLessEqual(
            repair_metrics["repair_simplification_boundary_relaxed_faces"],
            100,
        )
        self.assertTrue(
            repair_metrics["repair_simplification_boundary_relaxation_used"]
        )
        self.assertTrue(
            repair_metrics["repair_simplification_topology_relaxation_attempted"]
        )
        self.assertFalse(
            repair_metrics["repair_simplification_topology_relaxation_used"]
        )
        self.assertTrue(
            repair_metrics["repair_simplification_boundary_relaxation_attempted"]
        )

    def test_self_intersection_audit_uses_pymeshlab_when_available(self):
        import trimesh

        vertices = np.asarray(
            [
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -0.5, -1.0),
                (0.0, -0.5, 1.0),
                (0.0, 1.0, 0.5),
            ],
            dtype=np.float64,
        )
        intersecting = trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
            process=False,
        )

        audit = direct_mesh._self_intersection_audit(intersecting, "probe")

        self.assertTrue(audit["probe_self_intersection_supported"])
        self.assertEqual(audit["probe_self_intersection_count"], 2)

    def test_voxel_close_repair_retriangulates_single_degenerate_face_before_cleanup(self):
        import trimesh

        voxel_mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
        original_audit = direct_mesh._printability_audit

        def audit_with_decimation_artifact(mesh, prefix):
            audit = original_audit(mesh, prefix)
            if prefix == "repair_preclean":
                audit["repair_preclean_printable"] = False
                audit["repair_preclean_degenerate_face_count"] = 1
            return audit

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "repaired.stl"
            repair_metrics = {}
            with (
                patch.object(direct_mesh, "load_mesh", return_value=voxel_mesh.copy()),
                patch.object(direct_mesh, "_voxel_close_mesh", return_value=voxel_mesh.copy()),
                patch.object(
                    direct_mesh,
                    "_printability_audit",
                    side_effect=audit_with_decimation_artifact,
                ),
                patch.object(
                    direct_mesh,
                    "_retriangulate_marching_cubes_mesh",
                    return_value=voxel_mesh.copy(),
                ) as retriangulate,
                patch.object(
                    direct_mesh,
                    "_clean_mesh",
                    side_effect=AssertionError(
                        "retriangulated printable mesh should bypass generic cleanup"
                    ),
                ),
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "input.glb",
                    output_path,
                    mode="printable",
                    preconditioner="voxel-close",
                    metrics=repair_metrics,
                )

            diagnostics = stl_diagnostics(output_path)

        retriangulate.assert_called_once()
        self.assertFalse(repair_metrics["repair_preclean_printable"])
        self.assertEqual(repair_metrics["repair_preclean_degenerate_face_count"], 1)
        self.assertTrue(repair_metrics["repair_preclean_retriangulated_printable"])
        self.assertTrue(repair_metrics["repair_preclean_retriangulated_geometry_preserved"])
        self.assertTrue(repair_metrics["repair_preclean_retriangulation_attempted"])
        self.assertTrue(repair_metrics["repair_preclean_retriangulation_accepted"])
        self.assertTrue(repair_metrics["repair_cleaning_skipped"])
        self.assertFalse(repair_metrics["repair_convex_hull_used"])
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])

    def test_marching_cubes_retriangulation_repairs_collinear_face_without_shape_drift(self):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2)
        a, b, c = map(int, mesh.faces[0])
        mesh.vertices[c] = (mesh.vertices[a] + mesh.vertices[b]) / 2.0
        before_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        before_faces = np.asarray(mesh.faces, dtype=np.int64).copy()
        before_volume = float(mesh.volume)

        before_audit = direct_mesh._printability_audit(mesh, "before")
        cleaned_audit = direct_mesh._printability_audit(
            direct_mesh._clean_mesh(mesh),
            "cleaned",
        )
        retriangulated = direct_mesh._retriangulate_marching_cubes_mesh(mesh)
        after_audit = direct_mesh._printability_audit(retriangulated, "after")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "retriangulated.stl"
            retriangulated.export(output_path)
            round_trip = stl_diagnostics(output_path)

        self.assertFalse(before_audit["before_printable"])
        self.assertEqual(before_audit["before_degenerate_face_count"], 1)
        self.assertEqual(before_audit["before_nonmanifold_edge_count"], 0)
        self.assertFalse(cleaned_audit["cleaned_printable"])
        self.assertEqual(cleaned_audit["cleaned_nonmanifold_edge_count"], 3)
        self.assertTrue(after_audit["after_printable"])
        self.assertEqual(after_audit["after_degenerate_face_count"], 0)
        self.assertEqual(after_audit["after_nonmanifold_edge_count"], 0)
        self.assertEqual(len(retriangulated.vertices), len(before_vertices) - 1)
        self.assertEqual(len(retriangulated.faces), len(before_faces) - 2)
        self.assertFalse(np.array_equal(retriangulated.faces, before_faces))
        self.assertLess(
            abs(float(retriangulated.volume) - before_volume) / abs(before_volume),
            direct_mesh.RETRIANGULATION_MAX_VOLUME_RELATIVE_CHANGE,
        )
        self.assertTrue(round_trip["stl_is_watertight"])
        self.assertTrue(round_trip["stl_is_volume"])
        self.assertTrue(round_trip["stl_is_manifold"])
        self.assertEqual(round_trip["stl_degenerate_face_count"], 0)

        geometry_audit = direct_mesh._retriangulation_geometry_audit(
            mesh,
            retriangulated,
            "retriangulated",
        )
        self.assertTrue(geometry_audit["retriangulated_geometry_preserved"])
        self.assertLessEqual(
            geometry_audit["retriangulated_vertex_displacement_max_normalized"],
            direct_mesh.RETRIANGULATION_MAX_VERTEX_DISPLACEMENT_NORMALIZED,
        )
        self.assertLessEqual(
            geometry_audit["retriangulated_volume_relative_change_abs"],
            direct_mesh.RETRIANGULATION_MAX_VOLUME_RELATIVE_CHANGE,
        )

    def test_marching_cubes_retriangulation_keeps_failed_local_collapse_bounded(self):
        import trimesh

        torus = trimesh.creation.torus(
            major_radius=1.0,
            minor_radius=0.32,
            major_sections=96,
            minor_sections=48,
        )
        voxel_mesh = direct_mesh._voxel_close_mesh(torus, 96, "orthographic")
        simplified = direct_mesh._simplify_preserving_topology(
            voxel_mesh,
            3_000,
            strict=True,
        )
        a, b, c = map(int, simplified.faces[0])
        simplified.vertices[c] = (simplified.vertices[a] + simplified.vertices[b]) / 2.0

        before_audit = direct_mesh._printability_audit(simplified, "before")
        retriangulated = direct_mesh._retriangulate_marching_cubes_mesh(simplified)
        after_audit = direct_mesh._printability_audit(retriangulated, "after")
        geometry_audit = direct_mesh._retriangulation_geometry_audit(
            simplified,
            retriangulated,
            "retriangulated",
        )

        self.assertEqual(before_audit["before_degenerate_face_count"], 1)
        self.assertFalse(after_audit["after_printable"])
        self.assertTrue(geometry_audit["retriangulated_geometry_preserved"])
        self.assertLessEqual(
            geometry_audit["retriangulated_face_count_relative_change_abs"],
            direct_mesh.RETRIANGULATION_MAX_FACE_COUNT_RELATIVE_CHANGE,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repair_metrics = {}
            printable = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            with (
                patch.object(direct_mesh, "load_mesh", return_value=simplified.copy()),
                patch.object(direct_mesh, "_voxel_close_mesh", return_value=simplified.copy()),
                patch.object(direct_mesh, "_clean_mesh", return_value=printable) as clean_mesh,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "input.glb",
                    Path(temp_dir) / "repaired.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    metrics=repair_metrics,
                )

        clean_mesh.assert_called_once()
        self.assertTrue(repair_metrics["repair_preclean_retriangulation_attempted"])
        self.assertTrue(repair_metrics["repair_preclean_retriangulated_geometry_preserved"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulated_printable"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulation_accepted"])

    def test_retriangulation_geometry_audit_rejects_local_vertex_displacement(self):
        import trimesh

        reference = trimesh.creation.icosphere(subdivisions=2)
        a, b, c = map(int, reference.faces[0])
        reference.vertices[c] = (reference.vertices[a] + reference.vertices[b]) / 2.0
        candidate = direct_mesh._retriangulate_marching_cubes_mesh(reference)
        candidate.vertices[10] += np.array([0.05, 0.0, 0.0]) * np.linalg.norm(reference.extents)

        geometry_audit = direct_mesh._retriangulation_geometry_audit(
            reference,
            candidate,
            "retriangulated",
        )

        self.assertFalse(geometry_audit["retriangulated_geometry_preserved"])
        self.assertGreater(
            geometry_audit["retriangulated_vertex_displacement_max_normalized"],
            direct_mesh.RETRIANGULATION_MAX_VERTEX_DISPLACEMENT_NORMALIZED,
        )

    def test_marching_cubes_retriangulation_rejects_edit_budget_overshoot(self):
        import pymeshlab
        import trimesh

        reference = trimesh.creation.icosphere(subdivisions=2)
        a, b, c = map(int, reference.faces[0])
        reference.vertices[c] = (reference.vertices[a] + reference.vertices[b]) / 2.0
        bounded = direct_mesh._retriangulate_marching_cubes_mesh(reference)

        class FakeMesh:
            def vertex_matrix(self):
                return np.asarray(bounded.vertices, dtype=np.float64)

            def face_matrix(self):
                return np.asarray(bounded.faces[:-1], dtype=np.int32)

        class FakeMeshSet:
            def add_mesh(self, _mesh):
                pass

            def apply_filter(self, *_args, **_kwargs):
                pass

            def current_mesh(self):
                return FakeMesh()

        with patch.object(pymeshlab, "MeshSet", return_value=FakeMeshSet()):
            with self.assertRaisesRegex(RuntimeError, "exceeded its local edit budget"):
                direct_mesh._retriangulate_marching_cubes_mesh(reference)

    def test_voxel_close_repair_falls_back_when_retriangulation_changes_geometry(self):
        import trimesh

        voxel_mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
        distorted = trimesh.creation.box(extents=(0.8, 0.75, 0.5))
        original_audit = direct_mesh._printability_audit

        def audit_with_decimation_artifact(mesh, prefix):
            audit = original_audit(mesh, prefix)
            if prefix == "repair_preclean":
                audit["repair_preclean_printable"] = False
                audit["repair_preclean_degenerate_face_count"] = 1
            return audit

        with tempfile.TemporaryDirectory() as temp_dir:
            repair_metrics = {}
            with (
                patch.object(direct_mesh, "load_mesh", return_value=voxel_mesh.copy()),
                patch.object(direct_mesh, "_voxel_close_mesh", return_value=voxel_mesh.copy()),
                patch.object(
                    direct_mesh,
                    "_printability_audit",
                    side_effect=audit_with_decimation_artifact,
                ),
                patch.object(
                    direct_mesh,
                    "_retriangulate_marching_cubes_mesh",
                    return_value=distorted,
                ),
                patch.object(direct_mesh, "_clean_mesh", return_value=voxel_mesh.copy()) as clean_mesh,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "input.glb",
                    Path(temp_dir) / "repaired.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    metrics=repair_metrics,
                )

        clean_mesh.assert_called_once()
        self.assertTrue(repair_metrics["repair_preclean_retriangulation_attempted"])
        self.assertTrue(repair_metrics["repair_preclean_retriangulated_printable"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulated_geometry_preserved"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulation_accepted"])
        self.assertFalse(repair_metrics["repair_cleaning_skipped"])

    def test_voxel_close_repair_does_not_retriangulate_nonmanifold_preclean_mesh(self):
        import trimesh

        voxel_mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
        printable = voxel_mesh.copy()
        original_audit = direct_mesh._printability_audit

        def audit_with_sofa_failure(mesh, prefix):
            audit = original_audit(mesh, prefix)
            if prefix == "repair_preclean":
                audit["repair_preclean_printable"] = False
                audit["repair_preclean_component_count"] = 3
                audit["repair_preclean_nonmanifold_edge_count"] = 6
            return audit

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "repaired.stl"
            repair_metrics = {}
            with (
                patch.object(direct_mesh, "load_mesh", return_value=voxel_mesh.copy()),
                patch.object(direct_mesh, "_voxel_close_mesh", return_value=voxel_mesh.copy()),
                patch.object(
                    direct_mesh,
                    "_printability_audit",
                    side_effect=audit_with_sofa_failure,
                ),
                patch.object(
                    direct_mesh,
                    "_retriangulate_marching_cubes_mesh",
                    side_effect=AssertionError("nonmanifold mesh must not enter targeted retriangulation"),
                ),
                patch.object(direct_mesh, "_clean_mesh", return_value=printable) as clean_mesh,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "input.glb",
                    output_path,
                    mode="printable",
                    preconditioner="voxel-close",
                    metrics=repair_metrics,
                )

        clean_mesh.assert_called_once()
        self.assertFalse(repair_metrics["repair_preclean_retriangulation_attempted"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulation_accepted"])
        self.assertFalse(repair_metrics["repair_cleaning_skipped"])
        self.assertFalse(repair_metrics["repair_convex_hull_used"])

    def test_voxel_close_repair_falls_back_when_retriangulation_fails(self):
        import trimesh

        voxel_mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
        original_audit = direct_mesh._printability_audit

        def audit_with_decimation_artifact(mesh, prefix):
            audit = original_audit(mesh, prefix)
            if prefix == "repair_preclean":
                audit["repair_preclean_printable"] = False
                audit["repair_preclean_degenerate_face_count"] = 1
            return audit

        with tempfile.TemporaryDirectory() as temp_dir:
            repair_metrics = {}
            with (
                patch.object(direct_mesh, "load_mesh", return_value=voxel_mesh.copy()),
                patch.object(direct_mesh, "_voxel_close_mesh", return_value=voxel_mesh.copy()),
                patch.object(
                    direct_mesh,
                    "_printability_audit",
                    side_effect=audit_with_decimation_artifact,
                ),
                patch.object(
                    direct_mesh,
                    "_retriangulate_marching_cubes_mesh",
                    side_effect=RuntimeError("filter unavailable"),
                ),
                patch.object(direct_mesh, "_clean_mesh", return_value=voxel_mesh.copy()) as clean_mesh,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "input.glb",
                    Path(temp_dir) / "repaired.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    metrics=repair_metrics,
                )

        clean_mesh.assert_called_once()
        self.assertTrue(repair_metrics["repair_preclean_retriangulation_attempted"])
        self.assertFalse(repair_metrics["repair_preclean_retriangulation_accepted"])
        self.assertFalse(repair_metrics["repair_cleaning_skipped"])
        self.assertFalse(repair_metrics["repair_convex_hull_used"])

    def test_component_area_filter_drops_only_configured_surface_fragments(self):
        import trimesh

        main = trimesh.creation.box(extents=(2.0, 1.5, 1.0))
        small = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
        small.apply_translation((4.0, 0.0, 0.0))
        fragmented = trimesh.util.concatenate((main, small))

        kept = direct_mesh._filter_face_components_by_area(fragmented, 0.0)
        filtered = direct_mesh._filter_face_components_by_area(fragmented, 0.01)

        self.assertEqual(len(kept.faces), len(fragmented.faces))
        self.assertEqual(len(filtered.faces), len(main.faces))
        np.testing.assert_allclose(filtered.extents, main.extents)

    def test_component_area_filter_rejects_unbounded_raw_adjacency(self):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2)
        with (
            patch.object(direct_mesh, "MAX_MESH_REPAIR_COMPONENT_FILTER_FACES", 100),
            self.assertRaisesRegex(RuntimeError, "bounded face limit"),
        ):
            direct_mesh._filter_face_components_by_area(mesh, 0.01)

    def test_component_close_filters_fragments_and_fills_only_small_holes(self):
        import trimesh

        box = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
        open_box = trimesh.Trimesh(
            vertices=np.asarray(box.vertices).copy(),
            faces=np.delete(np.asarray(box.faces), 0, axis=0),
            process=False,
        )
        fragment = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
        fragment.apply_translation((3.0, 0.0, 0.0))
        fragmented = trimesh.util.concatenate((open_box, fragment))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "fragmented.ply"
            output_path = root / "component_closed.stl"
            fragmented.export(input_path)
            repair_metrics = {}

            repair_mesh_for_printable_stl(
                input_path,
                output_path,
                mode="basic",
                preconditioner="component-close",
                component_area_ratio=0.01,
                hole_face_addition_ratio=0.02,
                target_faces=100,
                metrics=repair_metrics,
            )
            diagnostics = stl_diagnostics(output_path)

        self.assertEqual(repair_metrics["repair_component_filter_input_components"], 2)
        self.assertEqual(repair_metrics["repair_component_filter_output_components"], 1)
        self.assertEqual(repair_metrics["repair_bounded_hole_fill_max_boundary_edges"], 4)
        self.assertEqual(repair_metrics["repair_bounded_hole_fill_faces_added"], 1)
        self.assertEqual(repair_metrics["repair_simplification_requested_target_faces"], 100)
        self.assertEqual(repair_metrics["repair_simplification_reserved_hole_faces"], 2)
        self.assertEqual(repair_metrics["repair_simplification_target_faces"], 98)
        self.assertFalse(repair_metrics["repair_convex_hull_used"])
        self.assertLessEqual(diagnostics["stl_faces"], 100)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertTrue(diagnostics["stl_single_component"])

    def test_component_close_rejects_hole_fill_above_configured_budget(self):
        import trimesh

        box = trimesh.creation.box()
        open_box = trimesh.Trimesh(
            vertices=np.asarray(box.vertices).copy(),
            faces=np.delete(np.asarray(box.faces), 0, axis=0),
            process=False,
        )

        with self.assertRaisesRegex(RuntimeError, "face-addition budget"):
            direct_mesh._bounded_fill_small_holes(open_box, 0.0)

    def test_component_close_uses_optimal_topology_preserving_simplification(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "dense.ply"
            output_path = root / "simplified.stl"
            trimesh.creation.icosphere(subdivisions=3).export(input_path)
            repair_metrics = {}

            with patch.object(
                direct_mesh,
                "_simplify_preserving_topology",
                wraps=direct_mesh._simplify_preserving_topology,
            ) as simplify:
                repair_mesh_for_printable_stl(
                    input_path,
                    output_path,
                    mode="basic",
                    preconditioner="component-close",
                    component_area_ratio=0.01,
                    hole_face_addition_ratio=0.02,
                    target_faces=100,
                    simplify_placement="optimal",
                    metrics=repair_metrics,
                )

            simplified = direct_mesh.load_mesh(output_path)

        self.assertEqual(simplify.call_args.kwargs["placement"], "optimal")
        self.assertTrue(simplify.call_args.kwargs["allow_boundary_relaxation"])
        self.assertTrue(repair_metrics["repair_simplification_applied"])
        self.assertLessEqual(len(simplified.faces), 100)
        self.assertLessEqual(direct_mesh.normalized_bbox_complexity_log1p(simplified), 9.95)
        self.assertTrue(mesh_is_printable_volume(simplified))

    def test_component_close_basic_never_uses_hull_for_retained_components(self):
        import trimesh

        first = trimesh.creation.box()
        second = trimesh.creation.box()
        second.apply_translation((2.0, 0.0, 0.0))
        disconnected = trimesh.util.concatenate((first, second))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "disconnected.ply"
            output_path = root / "no_hull.stl"
            disconnected.export(input_path)
            repair_metrics = {}

            with patch.object(
                direct_mesh,
                "_convex_hull_mesh",
                side_effect=AssertionError("component-close basic must not use a hull"),
            ) as hull:
                repair_mesh_for_printable_stl(
                    input_path,
                    output_path,
                    mode="basic",
                    preconditioner="component-close",
                    component_area_ratio=0.01,
                    hole_face_addition_ratio=0.02,
                    metrics=repair_metrics,
                )
            diagnostics = stl_diagnostics(output_path)

        hull.assert_not_called()
        self.assertFalse(repair_metrics["repair_convex_hull_used"])
        self.assertEqual(diagnostics["stl_component_count"], 2)
        self.assertFalse(diagnostics["stl_single_component"])

    def test_postprocess_preserves_printable_topology_during_decimation(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "torus.ply"
            output_path = root / "decimated.stl"
            trimesh.creation.torus(
                major_radius=1.0,
                minor_radius=0.32,
                major_sections=64,
                minor_sections=32,
            ).export(input_path)

            postprocess_mesh_for_stl(
                input_path,
                output_path,
                target_faces=20,
                preserve_printability=True,
                simplify_placement="endpoint",
            )
            diagnostics = stl_diagnostics(output_path)

        self.assertLessEqual(diagnostics["stl_faces"], 20)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])
        self.assertTrue(diagnostics["stl_single_component"])

    def test_voxel_close_repair_rejects_invalid_precondition_parameters(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh_path = root / "box.ply"
            trimesh.creation.box().export(mesh_path)

            with self.assertRaisesRegex(ValueError, "area ratio"):
                repair_mesh_for_printable_stl(
                    mesh_path,
                    root / "bad_ratio.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    component_area_ratio=1.01,
                )
            with self.assertRaisesRegex(ValueError, "at least 16"):
                repair_mesh_for_printable_stl(
                    mesh_path,
                    root / "bad_resolution.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    voxel_resolution=8,
                )
            with self.assertRaisesRegex(ValueError, "at most 384"):
                repair_mesh_for_printable_stl(
                    mesh_path,
                    root / "too_dense.stl",
                    mode="printable",
                    preconditioner="voxel-close",
                    voxel_resolution=512,
                )

    def test_printable_mesh_repair_preconditions_dense_mesh_to_face_budget(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dense_path = root / "dense_sphere.ply"
            repaired_path = root / "repaired.stl"
            dense = trimesh.creation.icosphere(subdivisions=4)
            dense.export(dense_path)

            with patch.object(direct_mesh, "_clean_mesh", wraps=direct_mesh._clean_mesh) as clean_mesh:
                repair_mesh_for_printable_stl(
                    dense_path,
                    repaired_path,
                    mode="printable",
                    target_faces=512,
                )
                repair_input_faces = len(clean_mesh.call_args.args[0].faces)

            diagnostics = stl_diagnostics(repaired_path)

        self.assertGreater(len(dense.faces), 512)
        self.assertLessEqual(repair_input_faces, 512)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_single_component"])

    def test_printable_mesh_repair_preconditions_fragmented_glb_before_component_split(self):
        import trimesh
        import warnings

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fragmented_path = root / "fragmented.glb"
            repaired_path = root / "repaired.stl"
            components = []
            for index in range(32):
                component = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
                component.apply_translation((float(index % 8) * 2.0, float(index // 8) * 2.0, 0.0))
                components.append(component)
            fragmented = trimesh.util.concatenate(components)
            fragmented.export(fragmented_path)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with patch.object(direct_mesh, "_clean_mesh", wraps=direct_mesh._clean_mesh) as clean_mesh:
                    repair_mesh_for_printable_stl(
                        fragmented_path,
                        repaired_path,
                        mode="printable",
                        target_faces=64,
                    )
                    repair_input_faces = len(clean_mesh.call_args.args[0].faces)

            diagnostics = stl_diagnostics(repaired_path)

        self.assertEqual(len(fragmented.split(only_watertight=False)), 32)
        self.assertLessEqual(repair_input_faces, 64)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_single_component"])

    def test_fast_component_filter_prefers_enclosed_volume_over_surface_area(self):
        import trimesh

        plate = trimesh.creation.box(extents=(10.0, 10.0, 0.01))
        plate.apply_translation((20.0, 0.0, 0.0))
        solid = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
        fragmented = trimesh.util.concatenate((plate, solid))

        filtered = direct_mesh._largest_face_component(fragmented)

        self.assertGreater(float(plate.area), float(solid.area))
        self.assertGreater(float(solid.volume), float(plate.volume) * 60.0)
        self.assertAlmostEqual(abs(float(filtered.volume)), float(solid.volume), places=6)
        np.testing.assert_allclose(filtered.extents, solid.extents)

    def test_printable_mesh_repair_accepts_bounded_residual_before_component_cleanup(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "repaired.stl"
            residual = SimpleNamespace(faces=np.zeros((103_036, 3), dtype=np.int64))
            printable = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            with (
                patch.object(direct_mesh, "load_mesh", return_value=residual),
                patch.object(direct_mesh, "_simplify_to_face_count", return_value=residual) as simplify,
                patch.object(direct_mesh, "_clean_mesh", return_value=printable),
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "dense.glb",
                    output_path,
                    mode="printable",
                    target_faces=128,
                )

            diagnostics = stl_diagnostics(output_path)

        simplify.assert_called_once_with(residual, 128, strict=True)
        self.assertGreater(len(residual.faces), 128)
        self.assertLessEqual(len(residual.faces), direct_mesh.MIN_SAFE_TOPOLOGY_REPAIR_FACES)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])

    def test_printable_mesh_repair_rejects_residual_above_safe_topology_limit(self):
        residual = SimpleNamespace(
            faces=np.zeros(
                (direct_mesh.MIN_SAFE_TOPOLOGY_REPAIR_FACES + 1, 3),
                dtype=np.int64,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(direct_mesh, "load_mesh", return_value=residual),
                patch.object(direct_mesh, "_simplify_to_face_count", return_value=residual),
                self.assertRaisesRegex(RuntimeError, "safe topology-repair limit"),
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "dense.glb",
                    Path(temp_dir) / "repaired.stl",
                    mode="printable",
                    target_faces=128,
                )

    def test_printable_mesh_repair_filters_fragmented_residual_before_rejecting(self):
        import trimesh

        fragments = []
        for index in range(40):
            fragment = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
            fragment.apply_translation((float(index) * 2.0, 0.0, 0.0))
            fragments.append(fragment)
        residual = trimesh.util.concatenate(fragments)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "repaired.stl"
            with (
                patch.object(direct_mesh, "load_mesh", return_value=residual),
                patch.object(
                    direct_mesh,
                    "_simplify_to_face_count",
                    return_value=residual,
                ) as simplify,
                patch.object(direct_mesh, "MIN_SAFE_TOPOLOGY_REPAIR_FACES", 64),
                patch.object(direct_mesh, "MAX_FAST_COMPONENT_FILTER_FACES", 1_000),
                patch.object(direct_mesh, "_clean_mesh", wraps=direct_mesh._clean_mesh) as clean_mesh,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "fragmented.glb",
                    output_path,
                    mode="printable",
                    target_faces=16,
                )

            diagnostics = stl_diagnostics(output_path)

        simplify.assert_called_once_with(residual, 16, strict=True)
        self.assertEqual(len(clean_mesh.call_args.args[0].faces), 12)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])

    def test_printable_mesh_repair_preserves_simplifier_failure_cause(self):
        def fail_simplification(*, face_count):
            raise ValueError(f"cannot simplify to {face_count}")

        mesh = SimpleNamespace(
            faces=np.zeros((1_024, 3), dtype=np.int64),
            simplify_quadric_decimation=fail_simplification,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(direct_mesh, "load_mesh", return_value=mesh),
                self.assertRaisesRegex(RuntimeError, "failed while simplifying") as raised,
            ):
                repair_mesh_for_printable_stl(
                    Path(temp_dir) / "dense.glb",
                    Path(temp_dir) / "repaired.stl",
                    mode="printable",
                    target_faces=128,
                )

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_printable_mesh_repair_derives_precondition_budget_from_actual_bbox(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dense_path = root / "dense_sphere.ply"
            repaired_path = root / "repaired.stl"
            dense = trimesh.creation.icosphere(subdivisions=4)
            dense.export(dense_path)

            with patch.object(direct_mesh, "_clean_mesh", wraps=direct_mesh._clean_mesh) as clean_mesh:
                repair_mesh_for_printable_stl(
                    dense_path,
                    repaired_path,
                    mode="printable",
                    max_normalized_face_density_log1p=8.0,
                )
                repair_input_faces = len(clean_mesh.call_args.args[0].faces)

            diagnostics = stl_diagnostics(repaired_path)

        expected_target = max_faces_for_normalized_bbox_complexity(dense.extents, 8.0)
        self.assertLessEqual(repair_input_faces, expected_target)
        self.assertTrue(diagnostics["stl_is_watertight"])
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

    def test_mesh_postprocess_scales_and_compacts_bbox_for_stl_objective(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_mesh = root / "skinny_box.ply"
            output_mesh = root / "scaled_compact.stl"
            trimesh.creation.box(extents=(2.0, 1.0, 0.1)).export(input_mesh)

            postprocess_mesh_for_stl(
                input_mesh,
                output_mesh,
                target_max_dimension=96.0,
                min_bbox_dimension=12.0,
            )

            diagnostics = stl_diagnostics(output_mesh)

        self.assertAlmostEqual(diagnostics["stl_bbox_max_dimension"], 96.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_min_dimension"], 12.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_aspect_ratio"], 8.0, places=4)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_mesh_postprocess_can_clamp_bbox_aspect_for_stl_objective(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_mesh = root / "skinny_box.ply"
            output_mesh = root / "aspect_clamped.stl"
            trimesh.creation.box(extents=(2.0, 1.0, 0.1)).export(input_mesh)

            postprocess_mesh_for_stl(
                input_mesh,
                output_mesh,
                target_max_dimension=96.0,
                min_bbox_dimension=12.0,
                max_bbox_aspect_ratio=4.0,
            )

            diagnostics = stl_diagnostics(output_mesh)

        self.assertAlmostEqual(diagnostics["stl_bbox_max_dimension"], 96.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_min_dimension"], 24.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_aspect_ratio"], 4.0, places=4)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_mesh_postprocess_can_match_target_bbox_extents_for_stl_objective(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_mesh = root / "skinny_box.ply"
            output_mesh = root / "target_extents.stl"
            trimesh.creation.box(extents=(2.0, 1.0, 0.1)).export(input_mesh)

            postprocess_mesh_for_stl(
                input_mesh,
                output_mesh,
                target_bbox_extents=(96.0, 48.0, 24.0),
            )

            diagnostics = stl_diagnostics(output_mesh)

        self.assertAlmostEqual(diagnostics["stl_bbox_x"], 96.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_y"], 48.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_z"], 24.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_aspect_ratio"], 4.0, places=4)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_mesh_postprocess_adapts_faces_to_scale_free_complexity_limit(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_mesh = root / "dense_sphere.ply"
            output_mesh = root / "complexity_capped.stl"
            dense_mesh = trimesh.creation.icosphere(subdivisions=4)
            dense_mesh.export(input_mesh)

            postprocess_mesh_for_stl(
                input_mesh,
                output_mesh,
                target_bbox_extents=(96.0, 72.0, 48.0),
                target_faces=40000,
                max_normalized_face_density_log1p=8.0,
            )

            diagnostics = stl_diagnostics(output_mesh)

        self.assertLess(diagnostics["stl_faces"], len(dense_mesh.faces))
        self.assertLessEqual(diagnostics["stl_faces_per_normalized_bbox_volume_log1p"], 8.01)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_mesh_postprocess_rejects_output_when_decimation_shrinks_normalized_bbox(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_mesh = root / "dense_sphere.ply"
            output_mesh = root / "too_thin.stl"
            trimesh.creation.icosphere(subdivisions=4).export(input_mesh)

            with self.assertRaisesRegex(RuntimeError, "still exceeds the adaptive scale-free complexity limit"):
                postprocess_mesh_for_stl(
                    input_mesh,
                    output_mesh,
                    target_bbox_extents=(96.0, 96.0, 0.96),
                    target_faces=40000,
                    max_normalized_face_density_log1p=8.0,
                )

    def test_provider_derives_native_face_target_from_exact_bbox_before_inference(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            provider_mesh = root / "provider.ply"
            output_mesh = root / "output.ply"
            output_stl = root / "output.stl"
            Image.new("RGB", (8, 8), "white").save(input_image)
            trimesh.creation.icosphere(subdivisions=4).export(provider_mesh)
            observed_targets = []

            def fake_provider(args):
                observed_targets.append(args.mesh_target_faces)
                return provider_mesh

            args = SimpleNamespace(
                provider="triposg",
                input_image=input_image,
                input_bundle=None,
                output_mesh=output_mesh,
                output_stl=output_stl,
                raw_output_mesh=None,
                mesh_repair="none",
                mesh_target_max_dimension=0.0,
                mesh_min_bbox_dimension=0.0,
                mesh_max_bbox_aspect_ratio=0.0,
                mesh_target_bbox_extents=(96.0, 48.0, 24.0),
                mesh_target_faces=40000,
                mesh_max_normalized_face_density_log1p=8.0,
            )

            with patch.object(run_image_to_mesh_provider, "run_cli_provider", side_effect=fake_provider):
                run_image_to_mesh_provider.run_provider(args)

            diagnostics = stl_diagnostics(output_stl)

        expected_target = max_faces_for_normalized_bbox_complexity((96.0, 48.0, 24.0), 8.0)
        self.assertEqual(observed_targets, [expected_target])
        self.assertLessEqual(diagnostics["stl_faces_per_normalized_bbox_volume_log1p"], 8.01)

    def test_provider_rejects_invalid_repair_options_before_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = SimpleNamespace(
                provider="triposg",
                input_image=root / "input.png",
                input_bundle=None,
                output_mesh=root / "output.ply",
                output_stl=None,
                raw_output_mesh=None,
                mesh_repair="printable",
                mesh_repair_preconditioner="voxel-close",
                mesh_repair_component_area_ratio=0.01,
                mesh_repair_voxel_resolution=0,
                mesh_repair_voxel_fill_method="orthographic",
            )

            with (
                patch.object(run_image_to_mesh_provider, "run_cli_provider") as provider,
                self.assertRaisesRegex(ValueError, "between 16 and 384"),
            ):
                run_image_to_mesh_provider.run_provider(args)

        provider.assert_not_called()

    def test_provider_passes_adaptive_face_target_into_printable_repair(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            provider_mesh = root / "provider.ply"
            output_mesh = root / "output.ply"
            output_stl = root / "output.stl"
            Image.new("RGB", (8, 8), "white").save(input_image)
            trimesh.creation.icosphere(subdivisions=4).export(provider_mesh)
            args = SimpleNamespace(
                provider="triposg",
                input_image=input_image,
                input_bundle=None,
                output_mesh=output_mesh,
                output_stl=output_stl,
                raw_output_mesh=None,
                mesh_repair="printable",
                mesh_repair_preconditioner="legacy",
                mesh_repair_component_area_ratio=0.01,
                mesh_repair_hole_face_addition_ratio=0.03,
                mesh_repair_voxel_resolution=256,
                mesh_repair_voxel_fill_method="orthographic",
                mesh_repair_simplify_placement="endpoint",
                mesh_target_max_dimension=0.0,
                mesh_min_bbox_dimension=0.0,
                mesh_max_bbox_aspect_ratio=0.0,
                mesh_target_bbox_extents=(96.0, 48.0, 24.0),
                mesh_target_faces=40000,
                mesh_max_normalized_face_density_log1p=8.0,
            )

            with (
                patch.object(run_image_to_mesh_provider, "run_cli_provider", return_value=provider_mesh),
                patch.object(
                    run_image_to_mesh_provider,
                    "repair_mesh_for_printable_stl",
                    wraps=repair_mesh_for_printable_stl,
                ) as repair_mesh,
            ):
                run_image_to_mesh_provider.run_provider(args)

            diagnostics = stl_diagnostics(output_stl)

        expected_target = max_faces_for_normalized_bbox_complexity((96.0, 48.0, 24.0), 8.0)
        self.assertEqual(repair_mesh.call_args.kwargs["target_faces"], expected_target)
        self.assertEqual(repair_mesh.call_args.kwargs["preconditioner"], "legacy")
        self.assertEqual(repair_mesh.call_args.kwargs["component_area_ratio"], 0.01)
        self.assertEqual(repair_mesh.call_args.kwargs["hole_face_addition_ratio"], 0.03)
        self.assertEqual(repair_mesh.call_args.kwargs["voxel_resolution"], 256)
        self.assertEqual(repair_mesh.call_args.kwargs["voxel_fill_method"], "orthographic")
        self.assertEqual(repair_mesh.call_args.kwargs["simplify_placement"], "endpoint")
        self.assertEqual(
            args._provider_metrics["provider_mesh_repair_simplify_placement"],
            "endpoint",
        )
        self.assertEqual(
            args._provider_metrics["provider_mesh_repair_hole_face_addition_ratio"],
            0.03,
        )
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_component_close_provider_keeps_postprocess_fail_closed_without_hull(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            provider_mesh = root / "provider.ply"
            output_mesh = root / "output.ply"
            output_stl = root / "output.stl"
            Image.new("RGB", (8, 8), "white").save(input_image)
            trimesh.creation.box().export(provider_mesh)
            args = SimpleNamespace(
                provider="triposg",
                input_image=input_image,
                input_bundle=None,
                output_mesh=output_mesh,
                output_stl=output_stl,
                raw_output_mesh=None,
                mesh_repair="basic",
                mesh_repair_preconditioner="component-close",
                mesh_repair_component_area_ratio=0.01,
                mesh_repair_hole_face_addition_ratio=0.02,
                mesh_repair_voxel_resolution=192,
                mesh_repair_voxel_fill_method="orthographic",
                mesh_repair_simplify_placement="optimal",
                mesh_target_max_dimension=0.0,
                mesh_min_bbox_dimension=0.0,
                mesh_max_bbox_aspect_ratio=0.0,
                mesh_target_bbox_extents=(96.0, 48.0, 24.0),
                mesh_target_faces=40000,
                mesh_max_normalized_face_density_log1p=8.0,
            )

            with (
                patch.object(
                    run_image_to_mesh_provider,
                    "run_cli_provider",
                    return_value=provider_mesh,
                ),
                patch.object(
                    run_image_to_mesh_provider,
                    "postprocess_mesh_for_stl",
                    wraps=postprocess_mesh_for_stl,
                ) as postprocess,
            ):
                run_image_to_mesh_provider.run_provider(args)

            diagnostics = stl_diagnostics(output_stl)

        self.assertTrue(postprocess.call_args.kwargs["preserve_printability"])
        self.assertFalse(args._provider_metrics["repair_convex_hull_used"])
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_is_manifold"])

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

    def test_image_to_mesh_provider_wrapper_can_scale_and_compact_provider_mesh(self):
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
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('input_image')",
                        "parser.add_argument('--output-dir', required=True)",
                        "args = parser.parse_args()",
                        "Path(args.output_dir).mkdir(parents=True, exist_ok=True)",
                        "trimesh.creation.box(extents=(2.0, 1.0, 0.1)).export(Path(args.output_dir) / 'result.ply')",
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
                    "--mesh-target-max-dimension",
                    "96",
                    "--mesh-min-bbox-dimension",
                    "12",
                    "--mesh-max-bbox-aspect-ratio",
                    "4",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            raw_diagnostics = stl_diagnostics(raw_output_mesh)
            raw_output_mesh_exists = raw_output_mesh.exists()

        self.assertTrue(raw_output_mesh_exists)
        self.assertAlmostEqual(raw_diagnostics["stl_bbox_max_dimension"], 2.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_max_dimension"], 96.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_min_dimension"], 24.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_aspect_ratio"], 4.0, places=4)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_positive_volume"])

    def test_image_to_mesh_provider_wrapper_can_match_target_bbox_extents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_spar3d"
            provider_dir.mkdir()
            input_image = root / "input.png"
            output_mesh = root / "normalized.ply"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (12, 12), (120, 80, 160)).save(input_image)
            (provider_dir / "run.py").write_text(
                "\n".join(
                    [
                        "import argparse",
                        "import sys",
                        "from pathlib import Path",
                        "import trimesh",
                        "assert '--mesh-target-bbox-extents' not in sys.argv",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('input_image')",
                        "parser.add_argument('--output-dir', required=True)",
                        "args = parser.parse_args()",
                        "Path(args.output_dir).mkdir(parents=True, exist_ok=True)",
                        "trimesh.creation.box(extents=(2.0, 1.0, 0.1)).export(Path(args.output_dir) / 'result.ply')",
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
                    "--mesh-target-bbox-extents",
                    "96,48,24",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)

        self.assertAlmostEqual(diagnostics["stl_bbox_x"], 96.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_y"], 48.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_z"], 24.0, places=4)
        self.assertAlmostEqual(diagnostics["stl_bbox_aspect_ratio"], 4.0, places=4)
        self.assertTrue(diagnostics["stl_is_watertight"])
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

    def test_hunyuan3d_shape_provider_exports_mesh_from_repo_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_hunyuan"
            package_dir = provider_dir / "hy3dshape" / "hy3dshape"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "pipelines.py").write_text(
                "import trimesh\n"
                "\n"
                "class Hunyuan3DDiTFlowMatchingPipeline:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, model_name, device='cuda', dtype=None):\n"
                "        assert model_name == 'unit/hunyuan'\n"
                "        assert str(device) == 'cpu'\n"
                "        assert 'float32' in str(dtype)\n"
                "        return cls()\n"
                "\n"
                "    def __call__(self, image, **kwargs):\n"
                "        assert image.endswith('input.png')\n"
                "        assert kwargs['num_inference_steps'] == 12\n"
                "        assert kwargs['guidance_scale'] == 4.5\n"
                "        assert kwargs['octree_resolution'] == 128\n"
                "        assert kwargs['num_chunks'] == 256\n"
                "        assert kwargs['enable_pbar'] is False\n"
                "        return [trimesh.creation.box(extents=(1.0, 0.75, 0.5))]\n",
                encoding="utf-8",
            )
            input_image = root / "input.png"
            output_mesh = root / "hunyuan.glb"
            output_stl = root / "hunyuan.stl"
            Image.new("RGB", (8, 8), (127, 127, 127)).save(input_image)

            for module_name in list(sys.modules):
                if module_name == "hy3dshape" or module_name.startswith("hy3dshape."):
                    sys.modules.pop(module_name, None)
            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "hunyuan3d-shape",
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
                    "--model-name",
                    "unit/hunyuan",
                    "--num-inference-steps",
                    "12",
                    "--guidance-scale",
                    "4.5",
                    "--octree-resolution",
                    "128",
                    "--num-chunks",
                    "256",
                    "--disable-progress",
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

    def test_external_direct_mesh_command_failure_writes_stdout_stderr_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.png"
            masked = root / "masked.png"
            mask = root / "mask.png"
            output_dir = root / "candidate"
            script_path = root / "fail_provider.py"
            Image.new("RGB", (8, 8), (127, 127, 127)).save(full)
            Image.new("RGB", (8, 8), (255, 255, 255)).save(masked)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask)
            script_path.write_text(
                "\n".join(
                    [
                        "import sys",
                        "print('provider stdout marker')",
                        "print('provider stderr marker', file=sys.stderr)",
                        "raise SystemExit(7)",
                    ]
                ),
                encoding="utf-8",
            )
            sample = {
                "id": "box",
                "full_image": str(full),
                "masked_image": str(masked),
                "mask": str(mask),
            }
            args = SimpleNamespace(
                direct_mesh_input="masked",
                direct_mesh_output_ext="ply",
                direct_mesh_timeout=30,
                direct_mesh_command=f'"{sys.executable}" "{script_path}"',
                direct_mesh_reference_output_dir=None,
                direct_mesh_reference_method="mirror",
                stl_target_dimension=96,
            )
            output_dir.mkdir()
            stale_mesh = output_dir / "output_mesh.ply"
            stale_stl = output_dir / "output_model.stl"
            stale_metrics = output_dir / "provider_metrics.json"
            stale_mesh.write_bytes(b"stale mesh")
            stale_stl.write_bytes(b"stale stl")
            stale_metrics.write_text('{"status":"stale"}', encoding="utf-8")

            with self.assertRaises(RuntimeError) as raised:
                run_direct_mesh(sample, "external-image-to-mesh", output_dir, args)

            stdout_log = output_dir / "external_command.stdout.log"
            stderr_log = output_dir / "external_command.stderr.log"
            message = str(raised.exception)
            stdout_exists = stdout_log.exists()
            stderr_exists = stderr_log.exists()
            stdout_text = stdout_log.read_text(encoding="utf-8")
            stderr_text = stderr_log.read_text(encoding="utf-8")
            command_metrics = json.loads(
                (output_dir / "direct_mesh_command_metrics.json").read_text(encoding="utf-8")
            )
            stale_artifacts_exist = any(
                path.exists() for path in (stale_mesh, stale_stl, stale_metrics)
            )

        self.assertTrue(stdout_exists)
        self.assertTrue(stderr_exists)
        self.assertIn("provider stdout marker", stdout_text)
        self.assertIn("provider stderr marker", stderr_text)
        self.assertIn("exit status 7", message)
        self.assertIn("provider stdout marker", message)
        self.assertIn("provider stderr marker", message)
        self.assertEqual(command_metrics["direct_mesh_command_status"], "failed")
        self.assertGreaterEqual(command_metrics["direct_mesh_command_runtime_seconds"], 0.0)
        self.assertFalse(stale_artifacts_exist)

    def test_direct_mesh_metrics_ingest_provider_runtime_and_raw_geometry(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            output_mesh = root / "output.glb"
            raw_mesh = root / "output_raw.glb"
            output_stl = root / "output.stl"
            Image.new("RGB", (8, 8), (120, 130, 140)).save(input_image)
            trimesh.creation.icosphere(subdivisions=1).export(raw_mesh)
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(output_mesh)
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(output_stl)
            (root / "provider_metrics.json").write_text(
                json.dumps(
                    {
                        "provider": "trellis2",
                        "provider_cache_hit": False,
                        "provider_inference_runtime_seconds": 12.5,
                        "provider_invocation_runtime_seconds": 13.0,
                        "repair_runtime_seconds": 1.25,
                        "provider_peak_cuda_vram_gib": None,
                        "provider_peak_cuda_vram_supported": False,
                        "provider_raw_output_mesh": str(raw_mesh),
                        "provider_final_output_mesh": str(output_mesh),
                        "provider_mesh_repair": "printable",
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            (root / "direct_mesh_command_metrics.json").write_text(
                json.dumps(
                    {
                        "direct_mesh_command_runtime_seconds": 14.5,
                        "direct_mesh_command_status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "id": "box",
                "full_image": str(input_image),
                "masked_image": str(input_image),
                "mask": str(input_image),
            }
            args = SimpleNamespace(
                direct_mesh_command="provider-wrapper",
                direct_mesh_input="biharmonic",
                direct_mesh_output_ext="glb",
                direct_mesh_reference_method="mirror",
                direct_mesh_reference_output_dir=None,
                source_mesh_repair="none",
                stl_target_dimension=96,
                mesh_surface_max_points=64,
                emit_stl=True,
            )
            with patch.object(
                run_completion_benchmark,
                "run_direct_mesh",
                return_value=(input_image, output_mesh, output_stl, None),
            ):
                row = evaluate_direct_mesh_sample(
                    sample,
                    "external-image-to-mesh",
                    root,
                    args,
                )
        self.assertEqual(row["provider_inference_runtime_seconds"], 12.5)
        self.assertEqual(row["direct_mesh_command_runtime_seconds"], 14.5)
        self.assertTrue(row["raw_mesh_exists"])
        self.assertGreater(row["repair_volume_fill_ratio_change"], 0.0)
        self.assertGreater(row["repair_volume_fill_ratio_relative_change"], 0.0)
        self.assertEqual(
            row["repair_volume_fill_ratio_relative_change_abs"],
            abs(row["repair_volume_fill_ratio_relative_change"]),
        )
        self.assertFalse(row["provider_peak_cuda_vram_supported"])
        self.assertIsNone(row["provider_peak_cuda_vram_gib"])

    def test_direct_mesh_repair_uses_surface_proxy_for_open_raw_mesh(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            raw_mesh = root / "raw-open.ply"
            output_mesh = root / "normalized.glb"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (8, 8), (120, 130, 140)).save(input_image)
            closed = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
            open_mesh = closed.copy()
            open_mesh.update_faces(open_mesh.face_normals[:, 0] < 0.9)
            open_mesh.remove_unreferenced_vertices()
            open_mesh.export(raw_mesh)
            closed.export(output_mesh)
            closed.export(output_stl)
            (root / "provider_metrics.json").write_text(
                json.dumps(
                    {
                        "provider": "fixture",
                        "provider_raw_output_mesh": str(raw_mesh),
                        "provider_final_output_mesh": str(output_mesh),
                        "provider_mesh_repair": "printable",
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "id": "open-box",
                "full_image": str(input_image),
                "masked_image": str(input_image),
                "mask": str(input_image),
            }
            args = SimpleNamespace(
                direct_mesh_command="provider-wrapper",
                direct_mesh_input="biharmonic",
                direct_mesh_output_ext="glb",
                direct_mesh_reference_method="mirror",
                direct_mesh_reference_output_dir=None,
                source_mesh_repair="none",
                stl_target_dimension=96,
                mesh_surface_max_points=64,
                emit_stl=True,
            )
            with patch.object(
                run_completion_benchmark,
                "run_direct_mesh",
                return_value=(input_image, output_mesh, output_stl, None),
            ):
                row = evaluate_direct_mesh_sample(
                    sample,
                    "external-image-to-mesh",
                    root,
                    args,
                )

            overlap_raw = root / "raw-overlap.ply"
            overlap_second = closed.copy()
            overlap_second.apply_translation((0.2, 0.0, 0.0))
            trimesh.util.concatenate((closed.copy(), overlap_second)).export(overlap_raw)
            (root / "provider_metrics.json").write_text(
                json.dumps(
                    {
                        "provider": "fixture",
                        "provider_raw_output_mesh": str(overlap_raw),
                        "provider_final_output_mesh": str(output_mesh),
                        "provider_mesh_repair": "printable",
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                run_completion_benchmark,
                "run_direct_mesh",
                return_value=(input_image, output_mesh, output_stl, None),
            ):
                overlap_row = evaluate_direct_mesh_sample(
                    sample,
                    "external-image-to-mesh",
                    root,
                    args,
                )

        self.assertFalse(row["raw_mesh_volume_fill_ratio_reliable"])
        self.assertTrue(row["raw_mesh_surface_fill_ratio_supported"])
        self.assertTrue(row["stl_surface_fill_ratio_supported"])
        self.assertEqual(
            row["repair_fill_ratio_metric"],
            "surface-component-unsigned-tetrahedra",
        )
        self.assertTrue(row["repair_fill_ratio_surface_proxy_used"])
        self.assertTrue(row["repair_fill_ratio_supported"])
        self.assertGreater(row["repair_fill_ratio_relative_change_abs"], 0.0)
        self.assertAlmostEqual(row["repair_fill_ratio_relative_change_abs"], 0.2, places=6)
        self.assertLessEqual(row["repair_fill_ratio_relative_change_abs"], 0.5)
        self.assertNotEqual(
            row["repair_fill_ratio_relative_change_abs"],
            row["repair_volume_fill_ratio_relative_change_abs"],
        )
        self.assertEqual(overlap_row["raw_mesh_surface_fill_ratio"], 1.0)
        self.assertGreater(overlap_row["raw_mesh_surface_fill_ratio_comparison"], 1.0)
        self.assertGreater(overlap_row["repair_fill_ratio_relative_change_abs"], 0.4)

    def test_single_image_mesh_scores_views_held_out_from_primary_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = [root / "front.png", root / "back.png"]
            masks = [root / "front-mask.png", root / "back-mask.png"]
            for image_path in images:
                Image.new("RGB", (8, 8), (128, 128, 128)).save(image_path)
            for mask_path in masks:
                Image.new("L", (8, 8), 255).save(mask_path)
            sample = {
                "multiview_images": [str(path) for path in images],
                "multiview_masks": [str(path) for path in masks],
                "multiview_cameras": [{"view": "front"}, {"view": "back"}],
            }
            rendered = SimpleNamespace(silhouette=np.ones((16, 16), dtype=bool))
            with (
                patch.object(run_completion_benchmark, "load_mesh", return_value=object()),
                patch.object(run_completion_benchmark, "render_mesh", return_value=rendered) as render,
            ):
                metrics = run_completion_benchmark.heldout_multiview_mesh_metrics(
                    sample,
                    root / "mesh.glb",
                    {},
                    primary_view_index=0,
                    render_size=16,
                )

        self.assertTrue(metrics["heldout_view_agreement_supported"])
        self.assertEqual(metrics["heldout_view_selected_count"], 1)
        self.assertEqual(metrics["heldout_view_count"], 1)
        self.assertEqual(metrics["heldout_view_silhouette_iou_mean"], 1.0)
        self.assertEqual(render.call_args.kwargs["camera"], {"view": "back"})

    def test_direct_mesh_metrics_include_raw_geometry_without_mesh_repair(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_image = root / "input.png"
            raw_mesh = root / "raw.glb"
            output_mesh = root / "normalized.glb"
            output_stl = root / "normalized.stl"
            Image.new("RGB", (8, 8), (120, 130, 140)).save(input_image)
            trimesh.creation.icosphere(subdivisions=1).export(raw_mesh)
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(output_mesh)
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(output_stl)
            (root / "provider_metrics.json").write_text(
                json.dumps(
                    {
                        "provider": "pixal3d",
                        "provider_raw_output_mesh": str(raw_mesh),
                        "provider_final_output_mesh": str(output_mesh),
                        "provider_mesh_repair": "none",
                        "status": "ok",
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "id": "box",
                "full_image": str(input_image),
                "masked_image": str(input_image),
                "mask": str(input_image),
            }
            args = SimpleNamespace(
                direct_mesh_command="provider-wrapper",
                direct_mesh_input="biharmonic",
                direct_mesh_output_ext="glb",
                direct_mesh_reference_method="mirror",
                direct_mesh_reference_output_dir=None,
                source_mesh_repair="none",
                stl_target_dimension=96,
                mesh_surface_max_points=64,
                emit_stl=True,
            )
            with patch.object(
                run_completion_benchmark,
                "run_direct_mesh",
                return_value=(input_image, output_mesh, output_stl, None),
            ):
                row = evaluate_direct_mesh_sample(sample, "external-image-to-mesh", root, args)

        self.assertTrue(row["raw_mesh_exists"])
        self.assertEqual(row["provider_mesh_repair"], "none")
        self.assertNotIn("repair_volume_fill_ratio_change", row)

    def test_hunyuan3d_shape_provider_low_vram_accepts_offload_without_device_arg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_hunyuan"
            package_dir = provider_dir / "hy3dshape" / "hy3dshape"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "pipelines.py").write_text(
                "import trimesh\n"
                "\n"
                "class Hunyuan3DDiTFlowMatchingPipeline:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, model_name, device='cuda', dtype=None):\n"
                "        return cls()\n"
                "\n"
                "    def enable_model_cpu_offload(self):\n"
                "        self.offload = True\n"
                "\n"
                "    def __call__(self, image, **kwargs):\n"
                "        assert getattr(self, 'offload', False) is True\n"
                "        return [trimesh.creation.box(extents=(1.0, 0.75, 0.5))]\n",
                encoding="utf-8",
            )
            input_image = root / "input.png"
            output_mesh = root / "hunyuan.glb"
            output_stl = root / "hunyuan.stl"
            Image.new("RGB", (8, 8), (127, 127, 127)).save(input_image)

            for module_name in list(sys.modules):
                if module_name == "hy3dshape" or module_name.startswith("hy3dshape."):
                    sys.modules.pop(module_name, None)
            with patch("torch.cuda.is_available", return_value=True):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "run_image_to_mesh_provider",
                        "--provider",
                        "hunyuan3d-shape",
                        "--provider-dir",
                        str(provider_dir),
                        "--input-image",
                        str(input_image),
                        "--output-mesh",
                        str(output_mesh),
                        "--output-stl",
                        str(output_stl),
                        "--provider-device",
                        "cuda",
                        "--low-vram",
                    ],
                ):
                    run_image_to_mesh_provider_main()

            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)

    def test_source_mesh_bundle_oracle_exports_stl_from_multiview_bundle(self):
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_mesh = root / "source_box.ply"
            input_image = root / "primary.png"
            bundle_path = root / "multiview_input.json"
            output_mesh = root / "oracle.ply"
            output_stl = root / "oracle.stl"
            trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(source_mesh)
            Image.new("RGB", (8, 8), (127, 127, 127)).save(input_image)
            bundle_path.write_text(
                json.dumps(
                    {
                        "sample_id": "box",
                        "source_mesh": str(source_mesh),
                        "primary_image": str(input_image),
                        "views": [{"image": str(input_image), "camera": {"azimuth_deg": 0}}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "source-mesh-bundle-oracle",
                    "--input-image",
                    str(input_image),
                    "--input-bundle",
                    str(bundle_path),
                    "--output-mesh",
                    str(output_mesh),
                    "--output-stl",
                    str(output_stl),
                    "--mesh-repair",
                    "printable",
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

    def test_multiview_visual_hull_provider_exports_printable_stl(self):
        import trimesh
        from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mesh = trimesh.creation.box(extents=(1.0, 0.75, 0.5))
            config = RenderConfig(size=48)
            views = []
            for index, camera in enumerate(
                [
                    CameraSpec(azimuth_deg=0.0, elevation_deg=0.0),
                    CameraSpec(azimuth_deg=90.0, elevation_deg=0.0),
                    CameraSpec(azimuth_deg=0.0, elevation_deg=90.0),
                ]
            ):
                result = render_mesh(mesh, camera=camera, config=config, base_color=(120, 150, 180))
                image_path = root / f"view{index}.png"
                mask_path = root / f"view{index}_mask.png"
                Image.fromarray(np.clip(result.rgb * 255, 0, 255).astype(np.uint8)).save(image_path)
                Image.fromarray(result.silhouette.astype(np.uint8) * 255).save(mask_path)
                views.append(
                    {
                        "index": index,
                        "sample_id": f"box_v{index}",
                        "image": str(image_path),
                        "mask": str(mask_path),
                        "camera": camera.to_dict(),
                    }
                )

            bundle_path = root / "multiview_input.json"
            output_mesh = root / "visual_hull.ply"
            output_stl = root / "visual_hull.stl"
            bundle_path.write_text(
                json.dumps(
                    {
                        "sample_id": "box",
                        "primary_image": views[0]["image"],
                        "views": views,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "multiview-visual-hull",
                    "--input-image",
                    views[0]["image"],
                    "--input-bundle",
                    str(bundle_path),
                    "--output-mesh",
                    str(output_mesh),
                    "--output-stl",
                    str(output_stl),
                    "--visual-hull-resolution",
                    "24",
                    "--visual-hull-mask-dilate",
                    "0",
                    "--mesh-repair",
                    "printable",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])
        self.assertTrue(diagnostics["stl_is_volume"])
        self.assertTrue(diagnostics["stl_positive_volume"])
        self.assertTrue(diagnostics["stl_bbox_has_volume"])

    def test_hunyuan3d_shape_provider_retries_after_partial_model_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_root = root / ".cache" / "hy3dgen" / "tencent" / "Hunyuan3D-2.1"
            model_dir = cache_root / "hunyuan3d-dit-v2-1"
            download_dir = cache_root / ".cache" / "huggingface" / "download" / "hunyuan3d-dit-v2-1"
            model_dir.mkdir(parents=True)
            download_dir.mkdir(parents=True)
            (model_dir / "config.yaml").write_text("partial", encoding="utf-8")
            (download_dir / "model.fp16.ckpt.lock").write_text("", encoding="utf-8")
            provider_dir = root / "fake_hunyuan"
            package_dir = provider_dir / "hy3dshape" / "hy3dshape"
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            missing_model = model_dir / "model.fp16.ckpt"
            (package_dir / "pipelines.py").write_text(
                "import pathlib\n"
                "import trimesh\n"
                f"CACHE_MODEL = pathlib.Path({str(missing_model)!r})\n"
                "CALLS_FILE = CACHE_MODEL.parents[1] / 'calls.txt'\n"
                "\n"
                "class Hunyuan3DDiTFlowMatchingPipeline:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, model_name, device='cuda', dtype=None):\n"
                "        calls = int(CALLS_FILE.read_text() or '0') if CALLS_FILE.exists() else 0\n"
                "        CALLS_FILE.write_text(str(calls + 1))\n"
                "        if calls == 0:\n"
                "            raise FileNotFoundError(f'Model file {CACHE_MODEL} not found')\n"
                "        return cls()\n"
                "\n"
                "    def __call__(self, image, **kwargs):\n"
                "        return [trimesh.creation.box(extents=(1.0, 0.75, 0.5))]\n",
                encoding="utf-8",
            )
            input_image = root / "input.png"
            output_mesh = root / "hunyuan.glb"
            output_stl = root / "hunyuan.stl"
            Image.new("RGB", (8, 8), (127, 127, 127)).save(input_image)

            for module_name in list(sys.modules):
                if module_name == "hy3dshape" or module_name.startswith("hy3dshape."):
                    sys.modules.pop(module_name, None)
            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "hunyuan3d-shape",
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
                    "--model-name",
                    "unit/hunyuan",
                ],
            ):
                run_image_to_mesh_provider_main()

            diagnostics = stl_diagnostics(output_stl)
            calls = (cache_root / "calls.txt").read_text(encoding="utf-8")
            model_dir_exists = model_dir.exists()
            download_dir_exists = download_dir.exists()
            output_mesh_exists = output_mesh.exists()
            output_stl_exists = output_stl.exists()

        self.assertEqual(calls, "2")
        self.assertFalse(model_dir_exists)
        self.assertFalse(download_dir_exists)
        self.assertTrue(output_mesh_exists)
        self.assertTrue(output_stl_exists)
        self.assertTrue(diagnostics["stl_is_watertight"])

    def test_hunyuan3d_shape_provider_prefetch_only_downloads_without_image_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "fake_hunyuan"
            utils_dir = provider_dir / "hy3dshape" / "hy3dshape" / "utils"
            utils_dir.mkdir(parents=True)
            (provider_dir / "hy3dshape" / "hy3dshape" / "__init__.py").write_text("", encoding="utf-8")
            (utils_dir / "__init__.py").write_text("", encoding="utf-8")
            model_root = root / ".cache" / "hy3dgen"
            (utils_dir / "utils.py").write_text(
                "from pathlib import Path\n"
                f"MODEL_ROOT = Path({str(model_root)!r})\n"
                "\n"
                "def smart_load_model(model_path, subfolder, use_safetensors, variant):\n"
                "    model_dir = MODEL_ROOT / model_path / subfolder\n"
                "    model_dir.mkdir(parents=True, exist_ok=True)\n"
                "    config = model_dir / 'config.yaml'\n"
                "    ckpt = model_dir / 'model.fp16.ckpt'\n"
                "    config.write_text('ok')\n"
                "    ckpt.write_text('weights')\n"
                "    return str(config), str(ckpt)\n",
                encoding="utf-8",
            )

            for module_name in list(sys.modules):
                if module_name == "hy3dshape" or module_name.startswith("hy3dshape."):
                    sys.modules.pop(module_name, None)
            with patch.object(
                sys,
                "argv",
                [
                    "run_image_to_mesh_provider",
                    "--provider",
                    "hunyuan3d-shape",
                    "--provider-dir",
                    str(provider_dir),
                    "--model-name",
                    "unit/hunyuan",
                    "--prefetch-only",
                ],
            ):
                run_image_to_mesh_provider_main()

            config_exists = (model_root / "unit" / "hunyuan" / "hunyuan3d-dit-v2-1" / "config.yaml").exists()
            ckpt_exists = (model_root / "unit" / "hunyuan" / "hunyuan3d-dit-v2-1" / "model.fp16.ckpt").exists()

        self.assertTrue(config_exists)
        self.assertTrue(ckpt_exists)

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
            triposr_direct_inputs=["masked", "mirror", "biharmonic"],
            include_hunyuan3d_shape=True,
            include_triposg=True,
            include_source_multiview_oracle=True,
            include_visual_hull_multiview=True,
            multiview_command='python mv.py "{input_bundle}" "{output_mesh}" "{output_stl}"',
            multiview_name="mv_recon",
            multiview_primary_input="masked",
            multiview_output_ext="ply",
            visual_hull_resolution=40,
            visual_hull_grid_extent=1.7,
            visual_hull_ortho_scale=2.0,
            visual_hull_mask_dilate=2,
            provider_python="python",
            provider_device="cuda",
            triposr_python="/content/triposr-venv/bin/python",
            triposr_dir="/content/TripoSR",
            hunyuan3d_dir="/content/Hunyuan3D",
            triposg_python="/content/triposg-venv/bin/python",
            triposg_dir="/content/TripoSG",
            triposg_direct_inputs=["masked", "mirror", "biharmonic"],
            hunyuan_num_inference_steps=24,
            hunyuan_guidance_scale=4.0,
            hunyuan_octree_resolution=192,
            hunyuan_num_chunks=4096,
            hunyuan_low_vram=True,
            triposg_num_inference_steps=8,
            triposg_guidance_scale=3.5,
            triposg_seed=99,
            chunk_size=256,
            mc_resolution=64,
            mesh_repair="printable",
            mesh_target_max_dimension=96.0,
            mesh_min_bbox_dimension=12.0,
            mesh_max_bbox_aspect_ratio=2.25,
            mesh_target_faces=512,
            direct_mesh_timeout=123,
        )

        experiments = build_stl_first_experiments(args)
        by_name = {experiment["name"]: experiment for experiment in experiments}

        self.assertEqual([experiment["name"] for experiment in experiments[:3]], ["masked", "mirror", "biharmonic"])
        self.assertEqual(by_name["masked"]["stl_mode"], "depth-relief")
        self.assertEqual(by_name["source_mesh_oracle"]["method"], "source-mesh-oracle")
        self.assertEqual(by_name["source_mesh_oracle"]["stl_mode"], "source-mesh-oracle")
        self.assertEqual(by_name["source_mesh_oracle"]["source_mesh_repair"], "printable")
        self.assertEqual(by_name["triposr_api_masked_repaired_direct_mesh"]["stl_mode"], "single-image-mesh")
        self.assertIn("--provider triposr-api", by_name["triposr_api_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_input"], "mirror")
        self.assertIn("--mesh-repair printable", by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-target-max-dimension 96.0", by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-min-bbox-dimension 12.0", by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-max-bbox-aspect-ratio 2.25", by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-target-faces 512", by_name["triposr_api_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["triposr_api_biharmonic_prefill_repaired_direct_mesh"]["direct_mesh_input"], "biharmonic")
        self.assertIn("--mesh-repair printable", by_name["triposr_api_biharmonic_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--provider hunyuan3d-shape", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("{output_dir}/output_mesh_raw.glb", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-target-max-dimension 96.0", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--num-inference-steps 24", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--guidance-scale 4.0", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--octree-resolution 192", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--num-chunks 4096", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--disable-progress", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--low-vram", by_name["hunyuan3d_shape_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["triposg_masked_repaired_direct_mesh"]["stl_mode"], "single-image-mesh")
        self.assertIn("--provider triposg", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--provider-dir /content/TripoSG", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--num-inference-steps 8", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--guidance-scale 3.5", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--seed 99", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-target-faces 512", by_name["triposg_masked_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["triposg_mirror_prefill_repaired_direct_mesh"]["direct_mesh_input"], "mirror")
        self.assertIn("--provider triposg", by_name["triposg_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-target-max-dimension 96.0", by_name["triposg_mirror_prefill_repaired_direct_mesh"]["direct_mesh_command"])
        self.assertEqual(
            by_name["triposg_biharmonic_prefill_repaired_direct_mesh"]["direct_mesh_input"],
            "biharmonic",
        )
        self.assertIn(
            "--mesh-target-faces 512",
            by_name["triposg_biharmonic_prefill_repaired_direct_mesh"]["direct_mesh_command"],
        )
        self.assertEqual(by_name["source_mesh_bundle_multiview_oracle"]["method"], "external-multiview-to-mesh")
        self.assertEqual(by_name["source_mesh_bundle_multiview_oracle"]["stl_mode"], "multiview-mesh")
        self.assertIn("--provider source-mesh-bundle-oracle", by_name["source_mesh_bundle_multiview_oracle"]["direct_mesh_command"])
        self.assertIn("--input-bundle \"{input_bundle}\"", by_name["source_mesh_bundle_multiview_oracle"]["direct_mesh_command"])
        self.assertEqual(by_name["visual_hull_multiview_repaired_mesh"]["method"], "external-multiview-to-mesh")
        self.assertEqual(by_name["visual_hull_multiview_repaired_mesh"]["stl_mode"], "multiview-mesh")
        self.assertIn("--provider multiview-visual-hull", by_name["visual_hull_multiview_repaired_mesh"]["direct_mesh_command"])
        self.assertIn("--input-bundle \"{input_bundle}\"", by_name["visual_hull_multiview_repaired_mesh"]["direct_mesh_command"])
        self.assertIn("--visual-hull-resolution 40", by_name["visual_hull_multiview_repaired_mesh"]["direct_mesh_command"])
        self.assertIn("--visual-hull-mask-dilate 2", by_name["visual_hull_multiview_repaired_mesh"]["direct_mesh_command"])
        self.assertIn("--mesh-repair printable", by_name["visual_hull_multiview_repaired_mesh"]["direct_mesh_command"])
        self.assertEqual(by_name["mv_recon"]["method"], "external-multiview-to-mesh")
        self.assertEqual(by_name["mv_recon"]["stl_mode"], "multiview-mesh")
        self.assertIn("{input_bundle}", by_name["mv_recon"]["direct_mesh_command"])

    def test_stl_first_smoke_can_target_reference_bbox_for_direct_mesh_candidates(self):
        args = SimpleNamespace(
            include_source_oracle=False,
            include_triposr_api=True,
            include_raw_direct_mesh=True,
            triposr_direct_inputs=["masked", "mirror"],
            include_hunyuan3d_shape=True,
            include_triposg=True,
            include_source_multiview_oracle=False,
            include_visual_hull_multiview=False,
            multiview_command=None,
            multiview_primary_input="masked",
            provider_python="python",
            provider_device="cuda",
            triposr_python="/content/triposr-venv/bin/python",
            triposr_dir="/content/TripoSR",
            hunyuan3d_dir="/content/Hunyuan3D",
            triposg_python="/content/triposg-venv/bin/python",
            triposg_dir="/content/TripoSG",
            triposg_direct_inputs=["biharmonic"],
            hunyuan_num_inference_steps=24,
            hunyuan_guidance_scale=4.0,
            hunyuan_octree_resolution=192,
            hunyuan_num_chunks=4096,
            hunyuan_low_vram=False,
            triposg_num_inference_steps=8,
            triposg_guidance_scale=3.5,
            triposg_seed=None,
            chunk_size=256,
            mc_resolution=64,
            mesh_repair="printable",
            mesh_target_max_dimension=96.0,
            mesh_min_bbox_dimension=0.0,
            mesh_max_bbox_aspect_ratio=0.0,
            mesh_target_bbox_source="reference",
            direct_mesh_reference_method="mirror",
            mesh_target_faces=40000,
            direct_mesh_timeout=123,
        )

        experiments = build_stl_first_experiments(args)
        by_name = {experiment["name"]: experiment for experiment in experiments}
        expected_direct_names = [
            "triposr_api_masked_stl_reference_bbox_direct_mesh",
            "triposr_api_masked_repaired_stl_reference_bbox_direct_mesh",
            "triposr_api_mirror_prefill_repaired_stl_reference_bbox_direct_mesh",
            "hunyuan3d_shape_masked_repaired_stl_reference_bbox_direct_mesh",
            "triposg_biharmonic_prefill_repaired_stl_reference_bbox_direct_mesh",
        ]

        for name in expected_direct_names:
            self.assertIn(name, by_name)
            experiment = by_name[name]
            self.assertEqual(experiment["direct_mesh_reference_method"], "mirror")
            self.assertIn(
                '--mesh-target-bbox-extents "{reference_bbox_extents}"',
                experiment["direct_mesh_command"],
            )

        self.assertNotIn("triposr_api_masked_repaired_direct_mesh", by_name)
        self.assertIn("--mesh-target-faces 40000", by_name[expected_direct_names[-1]]["direct_mesh_command"])

    def test_stl_first_smoke_marks_source_oracle_reference_bbox_as_diagnostic(self):
        args = self.stl_first_args(
            include_triposg=True,
            triposg_direct_inputs=["biharmonic"],
            mesh_target_bbox_source="reference",
            direct_mesh_reference_method="source_mesh_oracle",
        )

        experiments = build_stl_first_experiments(args)
        candidate = next(
            experiment
            for experiment in experiments
            if experiment["name"] == "triposg_biharmonic_prefill_repaired_stl_reference_bbox_direct_mesh"
        )

        self.assertEqual(candidate["direct_mesh_bbox_source"], "reference")
        self.assertTrue(candidate["oracle_diagnostic"])

    def test_stl_first_smoke_can_write_generated_triposg_bbox_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "triposg_generated.json"
            args = self.stl_first_args(
                write_config=str(config_path),
                include_triposg=True,
                triposg_direct_inputs=["masked", "mirror", "biharmonic"],
                triposg_num_inference_steps=50,
                triposg_guidance_scale=7.0,
                mesh_target_bbox_source="inferred",
                direct_mesh_reference_method="mirror",
                mesh_target_faces=40000,
                mesh_max_normalized_face_density_log1p=9.95,
                direct_mesh_timeout=3600,
            )

            report = write_stl_first_config_only(args)
            experiments = json.loads(config_path.read_text(encoding="utf-8"))

        by_name = {experiment["name"]: experiment for experiment in experiments}
        self.assertEqual(report["config"], str(config_path.resolve()))
        self.assertEqual(
            report["experiments"],
            [
                "masked",
                "mirror",
                "biharmonic",
                "source_mesh_oracle",
                "triposg_masked_repaired_stl_inferred_bbox_direct_mesh",
                "triposg_mirror_prefill_repaired_stl_inferred_bbox_direct_mesh",
                "triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh",
            ],
        )
        self.assertEqual(report["experiment_count"], 7)
        candidate = by_name["triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"]
        self.assertEqual(candidate["direct_mesh_bbox_source"], "inferred")
        self.assertEqual(candidate["direct_mesh_reference_method"], "mirror")
        self.assertFalse(candidate["oracle_diagnostic"])
        self.assertIn('--mesh-target-bbox-extents "{inferred_bbox_extents}"', candidate["direct_mesh_command"])
        self.assertIn("--mesh-target-faces 40000", candidate["direct_mesh_command"])
        self.assertIn(
            "--mesh-max-normalized-face-density-log1p 9.95",
            candidate["direct_mesh_command"],
        )

    def test_stl_first_smoke_shell_token_matches_current_platform(self):
        token = stl_first_shell_token(Path("C:/Program Files/Python/python.exe"))

        if sys.platform == "win32":
            self.assertTrue(token.startswith('"'))
            self.assertTrue(token.endswith('"'))
            self.assertNotIn("'", token)
        else:
            self.assertEqual(token, "'C:/Program Files/Python/python.exe'")

    def test_stl_first_smoke_flags_candidate_failures(self):
        failures = method_failure_rows(
            [
                {"method": "ok", "error_count": "0", "logged_failure_count": "0", "last_error": ""},
                {"method": "bad", "error_count": "1", "logged_failure_count": "0", "last_error": "boom"},
            ]
        )

        self.assertEqual([failure["method"] for failure in failures], ["bad"])

    def test_stl_first_smoke_writes_architecture_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            rows = [
                {
                    "method": "masked",
                    "base_method": "masked",
                    "stl_mode": "depth-relief",
                    "n": "2",
                    "attempted_n": "2",
                    "success_rate": "1.0",
                    "error_count": "0",
                    "mesh_surface_chamfer_l1_median": "0.40",
                    "mesh_surface_hausdorff95_median": "0.50",
                    "silhouette_iou_masked_median": "0.50",
                    "stl_is_watertight_median": "1.0",
                    "stl_is_volume_median": "1.0",
                    "stl_is_manifold_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                    "stl_single_component_median": "1.0",
                    "stl_faces_per_bbox_volume_log1p_median": "5.0",
                },
                {
                    "method": "mirror",
                    "base_method": "mirror",
                    "stl_mode": "depth-relief",
                    "n": "2",
                    "attempted_n": "2",
                    "success_rate": "1.0",
                    "error_count": "0",
                    "mesh_surface_chamfer_l1_median": "0.24",
                    "mesh_surface_hausdorff95_median": "0.32",
                    "silhouette_iou_masked_median": "0.70",
                    "stl_is_watertight_median": "1.0",
                    "stl_is_volume_median": "1.0",
                    "stl_is_manifold_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                    "stl_single_component_median": "1.0",
                    "stl_faces_per_bbox_volume_log1p_median": "5.1",
                },
                {
                    "method": "triposr_repaired",
                    "base_method": "external-image-to-mesh",
                    "stl_mode": "single-image-mesh",
                    "heldout_view_silhouette_iou_mean_median": "0.80",
                    "heldout_view_silhouette_iou_min_median": "0.70",
                    "n": "2",
                    "attempted_n": "2",
                    "success_rate": "1.0",
                    "error_count": "0",
                    "mesh_surface_chamfer_l1_median": "0.12",
                    "mesh_surface_hausdorff95_median": "0.18",
                    "stl_is_watertight_median": "1.0",
                    "stl_is_volume_median": "1.0",
                    "stl_is_manifold_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                    "stl_single_component_median": "1.0",
                    "stl_faces_per_bbox_volume_log1p_median": "4.9",
                },
                {
                    "method": "vggt_multiview_repaired",
                    "base_method": "external-multiview-to-mesh",
                    "stl_mode": "multiview-mesh",
                    "heldout_view_silhouette_iou_mean_median": "0.60",
                    "heldout_view_silhouette_iou_min_median": "0.50",
                    "n": "2",
                    "attempted_n": "2",
                    "success_rate": "1.0",
                    "error_count": "0",
                    "mesh_surface_chamfer_l1_median": "0.18",
                    "mesh_surface_hausdorff95_median": "0.24",
                    "stl_is_watertight_median": "1.0",
                    "stl_is_volume_median": "1.0",
                    "stl_is_manifold_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                    "stl_single_component_median": "1.0",
                    "stl_faces_per_bbox_volume_log1p_median": "5.2",
                },
                {
                    "method": "source_mesh_oracle",
                    "base_method": "source-mesh-oracle",
                    "stl_mode": "source-mesh-oracle",
                    "n": "2",
                    "attempted_n": "2",
                    "success_rate": "1.0",
                    "error_count": "0",
                    "mesh_surface_chamfer_l1_median": "0.0",
                    "mesh_surface_hausdorff95_median": "0.0",
                    "stl_is_watertight_median": "1.0",
                    "stl_is_volume_median": "1.0",
                    "stl_is_manifold_median": "1.0",
                    "stl_positive_volume_median": "1.0",
                    "stl_single_component_median": "1.0",
                    "stl_faces_per_bbox_volume_log1p_median": "4.0",
                },
            ]
            for row in rows:
                row.update(
                    {
                        "stl_exists_median": "1.0",
                        "stl_winding_consistent_median": "1.0",
                        "stl_bbox_has_volume_median": "1.0",
                        "stl_nonmanifold_edge_count_log1p_median": "0.0",
                        "stl_degenerate_face_ratio_median": "0.0",
                        "stl_component_excess_log1p_median": "0.0",
                        "stl_bbox_aspect_ratio_median": "2.0",
                        "heldout_view_silhouette_iou_mean_median": row.get(
                            "heldout_view_silhouette_iou_mean_median",
                            "",
                        ),
                        "heldout_view_silhouette_iou_min_median": row.get(
                            "heldout_view_silhouette_iou_min_median",
                            "",
                        ),
                        "repair_convex_hull_used_mean": (
                            "0.0" if "repaired" in row["method"] else ""
                        ),
                        "repair_volume_fill_ratio_relative_change_abs_median": (
                            "0.1" if "repaired" in row["method"] else ""
                        ),
                    }
                )
            with (experiment_dir / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            with (experiment_dir / "per_sample_metrics.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "sample_id",
                        "method",
                        "repair_volume_fill_ratio_relative_change_abs",
                    ],
                )
                writer.writeheader()
                for sample_id in ("a", "b"):
                    for row in rows:
                        writer.writerow(
                            {
                                "sample_id": sample_id,
                                "method": row["method"],
                                "repair_volume_fill_ratio_relative_change_abs": (
                                    "0.1" if "repaired" in row["method"] else ""
                                ),
                            }
                        )

            report = write_stl_first_architecture_report(experiment_dir, root, label="unit")

            markdown = (root / "stl_first_architecture_report.md").read_text(encoding="utf-8")

        leaders = {row["stl_mode"]: row["method"] for row in report["best_by_stl_mode"]}
        self.assertEqual(report["deployable_winner"]["method"], "triposr_repaired")
        self.assertEqual(report["promotion_eligible_winner"]["method"], "triposr_repaired")
        self.assertEqual(report["oracle_diagnostic_winner"]["method"], "source_mesh_oracle")
        self.assertEqual(report["architecture_replacement_decision"]["decision"], "promote-challenger")
        self.assertEqual(
            report["architecture_replacement_decision"]["recommended_method"],
            "triposr_repaired",
        )
        self.assertEqual(leaders["depth-relief"], "mirror")
        self.assertEqual(leaders["single-image-mesh"], "triposr_repaired")
        self.assertEqual(leaders["multiview-mesh"], "vggt_multiview_repaired")
        self.assertTrue(Path(report["report_json"]).name.endswith(".json"))
        self.assertIn("Architecture Decision", markdown)
        self.assertIn("Architecture Leaders", markdown)

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

    def test_stl_first_smoke_optimize_command_can_emit_selection_decision(self):
        args = SimpleNamespace(
            start_index=4,
            limit=10,
            depth_provider="depth-anything-v2",
            depth_model="depth-anything/Depth-Anything-V2-Small-hf",
            device="auto",
            stl_target_dimension=96,
            contact_sheet_max_samples=2,
            continue_on_error=True,
            resume=True,
            select_candidate=True,
            candidate_method="triposr_api_masked_repaired_direct_mesh",
            current_method="mirror",
            min_paired_n=10,
            min_win_rate=0.8,
            min_ci95_low=0.0,
            min_score_margin=0.0,
            min_stl_watertight=1.0,
            min_stl_is_volume=1.0,
            min_stl_is_manifold=1.0,
            min_stl_winding_consistent=1.0,
            min_stl_positive_volume=1.0,
            min_stl_single_component=1.0,
            min_stl_bbox_has_volume=1.0,
            max_stl_nonmanifold_edge_count_log1p=0.0,
            max_stl_degenerate_face_ratio=0.0,
            max_stl_component_excess_log1p=0.0,
            max_stl_bbox_aspect_ratio=10.0,
            max_stl_faces_per_bbox_volume_log1p=10.0,
            max_mesh_surface_chamfer_ratio_vs_current=1.05,
            max_mesh_surface_hausdorff95_ratio_vs_current=1.08,
            allow_missing_split_audit=True,
        )
        experiments = [{"name": "masked"}, {"name": "mirror"}, {"name": args.candidate_method}]

        command = build_stl_first_optimize_command(
            args,
            Path("manifest.jsonl"),
            Path("experiment"),
            Path("config.json"),
            experiments,
        )

        self.assertIn("--select-candidate", command)
        self.assertIn("--candidate-method", command)
        self.assertIn(args.candidate_method, command)
        self.assertIn("--current-method", command)
        self.assertIn("mirror", command)
        self.assertIn("--min-paired-n", command)
        self.assertIn("10", command)
        self.assertIn("--min-stl-is-manifold", command)
        self.assertIn("--min-stl-single-component", command)
        self.assertIn("--max-stl-nonmanifold-edge-count-log1p", command)
        self.assertIn("--max-stl-bbox-aspect-ratio", command)
        self.assertEqual(
            command[command.index("--max-mesh-surface-chamfer-ratio-vs-current") + 1],
            "1.05",
        )
        self.assertEqual(
            command[command.index("--max-mesh-surface-hausdorff95-ratio-vs-current") + 1],
            "1.08",
        )
        self.assertIn("--allow-missing-split-audit", command)

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
                    base_border_px=0,
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
                "multiview_primary_index": 0,
            }
            args = SimpleNamespace(
                skip_depth=False,
                depth_provider="mock",
                depth_model="mock",
                device="cpu",
                emit_stl=True,
                stl_target_dimension=96,
                stl_z_scale=50.0,
                stl_no_invert=False,
                stl_sigma=4.0,
                mesh_surface_max_points=64,
                prompt="",
                steps=1,
                seed=1,
                guidance=None,
                inpaint_max_dimension=64,
                model_name=None,
                lora_weights=None,
                lora_scale=None,
            )

            with (
                patch(
                    "backend.benchmark.run_completion_benchmark.process_image_get_depth_data",
                    return_value=str(pred_depth_path),
                ),
                patch.object(run_completion_benchmark, "depth_data_to_3d_model"),
                patch.object(
                    run_completion_benchmark,
                    "stl_and_mesh_metrics",
                    return_value={"stl_exists": True},
                ),
                patch.object(
                    run_completion_benchmark,
                    "heldout_multiview_mesh_metrics",
                    return_value={"heldout_view_silhouette_iou_mean": 0.75},
                ) as heldout,
            ):
                row = evaluate_sample(sample, "mock-method", str(full_path), str(full_path), temp_path / "run", args)
            summary = summarize([row], methods=["mock-method"], attempted_n=1)

        self.assertGreater(row["surface_chamfer_l1"], 0)
        self.assertGreater(row["surface_chamfer_rmse"], 0)
        self.assertGreater(row["object_surface_chamfer_l1"], 0)
        self.assertGreater(row["object_surface_chamfer_rmse"], 0)
        self.assertEqual(row["object_surface_point_count"], 8)
        self.assertEqual(row["heldout_view_silhouette_iou_mean"], 0.75)
        self.assertEqual(heldout.call_args.kwargs["primary_view_index"], 0)
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
                    "--candidate-method",
                    "triposr_api_masked_repaired_direct_mesh",
                    "--current-method",
                    "mirror",
                    "--require-image-to-mesh-providers",
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
        combine_command = self.matching_commands(commands, "backend.benchmark.combine_optimize_runs")[0]["command"]
        self.assertIn("--max-method-failures", first_eval_command)
        self.assertEqual(first_eval_command[first_eval_command.index("--max-method-failures") + 1], "2")
        self.assertIn("--candidate-method", first_eval_command)
        self.assertEqual(
            first_eval_command[first_eval_command.index("--candidate-method") + 1],
            "triposr_api_masked_repaired_direct_mesh",
        )
        self.assertIn("--current-method", first_eval_command)
        self.assertEqual(first_eval_command[first_eval_command.index("--current-method") + 1], "mirror")
        self.assertIn("--require-image-to-mesh-providers", first_eval_command)
        for command in (first_eval_command, combine_command):
            self.assertEqual(
                command[command.index("--max-mesh-surface-chamfer-ratio-vs-current") + 1],
                "1.1",
            )
            self.assertEqual(
                command[
                    command.index("--max-mesh-surface-hausdorff95-ratio-vs-current")
                    + 1
                ],
                "1.1",
            )
        self.assertIn("--candidate-method", combine_command)
        self.assertEqual(
            combine_command[combine_command.index("--candidate-method") + 1],
            "triposr_api_masked_repaired_direct_mesh",
        )
        self.assertIn("--current-method", combine_command)
        self.assertEqual(combine_command[combine_command.index("--current-method") + 1], "mirror")
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
                candidate_method="dreamshaper_weighted_lora",
                current_method="mirror",
                require_image_to_mesh_providers=True,
                colab_require_gpu_name_regex="RTX PRO 6000|Blackwell",
                colab_min_gpu_memory_gb=90,
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
        self.assertTrue(report["require_image_to_mesh_providers"])
        self.assertEqual(report["candidate_method"], "dreamshaper_weighted_lora")
        self.assertEqual(report["current_method"], "mirror")
        self.assertEqual(report["max_mesh_surface_chamfer_ratio_vs_current"], 1.1)
        self.assertEqual(report["max_mesh_surface_hausdorff95_ratio_vs_current"], 1.1)
        self.assertEqual(report["colab_require_gpu_name_regex"], "RTX PRO 6000|Blackwell")
        self.assertEqual(report["colab_min_gpu_memory_gb"], 90)
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
        self.assertIn("GPU_PREFLIGHT_PATH=/content/inputs/modelnet/gpu_preflight.json", archive_run_script)
        self.assertIn("COLAB_REQUIRE_GPU_NAME_REGEX='RTX PRO 6000|Blackwell'", archive_run_script)
        self.assertIn("COLAB_MIN_GPU_MEMORY_GB=90", archive_run_script)
        self.assertIn("--query-gpu=name,memory.total", archive_run_script)
        self.assertIn("gpu_preflight.json", archive_run_script)
        self.assertIn("'gpu_preflight': os.environ['GPU_PREFLIGHT_PATH']", archive_run_script)
        self.assertIn("summarize_gpu_preflight", archive_run_script)
        self.assertIn("'gpu_preflight_summary': gpu_preflight_summary", archive_run_script)
        self.assertIn("'failure_stage': 'gpu_preflight'", archive_run_script)
        self.assertIn("'gpu_preflight_failed' if gpu_preflight_summary.get('ok') is False", archive_run_script)
        self.assertIn("'gpu_preflight_errors'", archive_run_script)
        self.assertIn("add_if_exists(tar, gpu_preflight, 'gpu_preflight.json')", archive_run_script)
        self.assertIn("launch_preflight.json", archive_run_script)
        self.assertIn("manifest_rows", archive_run_script)
        self.assertIn("pytorch_lora_weights.safetensors", archive_run_script)
        self.assertIn("training_report.json", archive_run_script)
        self.assertIn("--max-mesh-surface-chamfer-ratio-vs-current 1.1", archive_run_script)
        self.assertIn("--max-mesh-surface-hausdorff95-ratio-vs-current 1.1", archive_run_script)
        self.assertIn("run_colab_eval.log", archive_run_script)
        self.assertIn("results_summary.json", archive_run_script)
        self.assertIn("g4_test_eval_results.tar.gz", archive_run_script)
        self.assertIn("g4_test_eval_results_compact.tar.gz", archive_run_script)
        self.assertIn("RESULTS_COMPACT_ARCHIVE", archive_run_script)
        self.assertIn("'results_compact_archive': str(compact_archive_path)", archive_run_script)
        self.assertIn("build_compact_results_archive(", archive_run_script)
        self.assertIn("(summary_path, 'results_summary.json')", archive_run_script)
        self.assertIn('echo "Compact results archive: $RESULTS_COMPACT_ARCHIVE"', archive_run_script)
        self.assertIn("STL_INGEST_DIR", archive_run_script)
        self.assertIn("g4_test_eval_stl_first_ingest", archive_run_script)
        self.assertIn("'backend.benchmark.ingest_stl_results'", archive_run_script)
        self.assertIn("'--score-mode'", archive_run_script)
        self.assertIn("'baseline-delta'", archive_run_script)
        self.assertIn("'--baseline-method'", archive_run_script)
        self.assertIn("'masked'", archive_run_script)
        self.assertIn("'stl_first_ingest_report': stl_ingest_report", archive_run_script)
        self.assertIn("'architecture_replacement_decision'", archive_run_script)
        self.assertIn("'recommended_method': decision.get('recommended_method', '')", archive_run_script)
        self.assertIn("'recommended_stl_mode': decision.get('recommended_stl_mode', '')", archive_run_script)
        self.assertIn("'score_delta_vs_depth_relief': decision.get('score_delta_vs_depth_relief')", archive_run_script)
        self.assertIn("(decision.get('depth_relief_baseline') or {}).get('method', '')", archive_run_script)
        self.assertIn("(decision.get('promotion_eligible_challenger') or {}).get('method', '')", archive_run_script)
        self.assertIn("(decision.get('score_leading_challenger') or {}).get('method', '')", archive_run_script)
        self.assertIn("add_if_exists(tar, stl_ingest_json, 'stl_first_ingest_report.json')", archive_run_script)
        self.assertIn("add_if_exists(tar, stl_ingest_md, 'stl_first_ingest_report.md')", archive_run_script)
        self.assertIn("add_if_exists(tar, stl_ingest_dir / 'stl_first_ingest.log', 'stl_first_ingest.log')", archive_run_script)
        self.assertIn('tee "$RUN_LOG"', archive_run_script)
        self.assertIn("run_status", archive_run_script)
        self.assertIn("selection_decisions", archive_run_script)
        self.assertIn("eval_summaries", archive_run_script)
        self.assertIn("combined_summaries", archive_run_script)
        self.assertIn("summarize_benchmark_dir", archive_run_script)
        self.assertIn("image_to_mesh_provider_preflight_exists", archive_run_script)
        self.assertIn("image_to_mesh_provider_readiness", archive_run_script)
        self.assertIn("'methods': compact_methods(ranked_rows or aggregate_rows)", archive_run_script)
        self.assertIn("'stl_mode'", archive_run_script)
        self.assertIn("'mesh_surface_chamfer_l1_median'", archive_run_script)
        self.assertIn("'mesh_surface_hausdorff95_median'", archive_run_script)
        self.assertIn("'stl_is_watertight_median'", archive_run_script)
        self.assertIn("'stl_is_volume_median'", archive_run_script)
        self.assertIn("'stl_is_manifold_median'", archive_run_script)
        self.assertIn("'stl_positive_volume_median'", archive_run_script)
        self.assertIn("'stl_single_component_median'", archive_run_script)
        self.assertIn("'stl_bbox_aspect_ratio_median'", archive_run_script)
        self.assertIn("'stl_faces_per_bbox_volume_log1p_median'", archive_run_script)
        self.assertIn("--run-name g4_test_eval", archive_run_script)
        self.assertIn("--eval-start 40", archive_run_script)
        self.assertIn("--eval-start 50", archive_run_script)
        self.assertIn("--max-method-failures 2", archive_run_script)
        self.assertIn("--require-image-to-mesh-providers", archive_run_script)
        self.assertIn("--candidate-method dreamshaper_weighted_lora", archive_run_script)
        self.assertIn("--current-method mirror", archive_run_script)
        self.assertIn("--manifest /content/inputs/modelnet/inputs/manifest.jsonl", archive_run_script)
        self.assertIn("--existing-lora-weights /content/inputs/modelnet/lora/weighted_surface", archive_run_script)

    def test_package_inputs_can_write_inline_colab_launcher(self):
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "inline_bundle.tar.gz"
            launcher = root / "launch_inline_colab.py"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/inline",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/inline_bundle.tar.gz",
                inline_colab_launcher_path=launcher,
                inline_colab_chunk_size=64,
                run_name="g4_inline_test",
                eval_starts=[0],
                eval_limit=1,
            )
            launcher_text = launcher.read_text(encoding="utf-8")

        self.assertEqual(report["inline_colab_launcher"], str(launcher))
        self.assertGreater(report["inline_colab_chunk_count"], 1)
        self.assertEqual(report["inline_colab_chunk_size"], 64)
        self.assertGreater(report["output_size"], 0)
        self.assertIn("inline_colab_launcher_sha256", report)
        self.assertIn("ARCHIVE_B64_CHUNKS", launcher_text)
        self.assertIn("/content/inline_bundle.tar.gz", launcher_text)
        self.assertIn(report["output_sha256"], launcher_text)
        self.assertIn("EXPECTED_SIZE", launcher_text)
        self.assertIn("base64.b64decode", launcher_text)
        self.assertIn("run_colab_eval.sh", launcher_text)
        self.assertIn("EXPECTED_SHA256", launcher_text)
        self.assertIn("env['EXTRACT_ROOT'] = str(EXTRACT_ROOT)", launcher_text)
        self.assertIn("completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)", launcher_text)
        self.assertIn("---RESULTS_SUMMARY_JSON---", launcher_text)
        self.assertIn("---RESULT_ARCHIVES_JSON---", launcher_text)
        self.assertIn("'results_compact_archive'", launcher_text)
        self.assertIn("if not archive_candidates:", launcher_text)
        self.assertIn("completed.check_returncode()", launcher_text)

    def test_package_inputs_rejects_inline_launcher_without_run_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            full = dataset / "full.png"
            full.write_bytes(b"asset")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "sample", "full_image": str(full.relative_to(root))}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--inline-colab-launcher requires --include-run-script"):
                package_inputs(
                    manifest=manifest,
                    output=root / "bundle.tar.gz",
                    extract_root="/content/inputs/inline",
                    root=root,
                    inline_colab_launcher_path=root / "launch_inline_colab.py",
                )

    def test_build_inline_colab_launcher_materializes_stub_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "stub_bundle.tar.gz"
            stub_script = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "echo stub\n"
            )
            with tarfile.open(archive, "w:gz") as tar:
                encoded = stub_script.encode("utf-8")
                info = tarfile.TarInfo("run_colab_eval.sh")
                info.mode = 0o755
                info.size = len(encoded)
                tar.addfile(info, io.BytesIO(encoded))
            expected_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            launcher_text, meta = build_inline_colab_launcher(
                archive_path=archive,
                colab_archive_path=str(root / "content" / "stub_bundle.tar.gz"),
                extract_root=str(root / "extract"),
                expected_sha256=expected_sha,
                expected_size=archive.stat().st_size,
                chunk_size=32,
                colab_env={"TRIPOSG_SETUP_ONLY": "1"},
            )
            materialized_archive = root / "content" / "stub_bundle.tar.gz"
            extracted_script = root / "extract" / "run_colab_eval.sh"

            with patch("subprocess.run") as run_mock:
                exec(compile(launcher_text, str(root / "launch_inline_colab.py"), "exec"), {"__name__": "__main__"})

            self.assertGreater(meta["inline_colab_chunk_count"], 1)
            self.assertEqual(hashlib.sha256(materialized_archive.read_bytes()).hexdigest(), expected_sha)
            self.assertEqual(extracted_script.read_text(encoding="utf-8"), stub_script)
            run_mock.assert_called_once()
            args, kwargs = run_mock.call_args
            self.assertEqual(args[0], ["bash", str(extracted_script), str(materialized_archive)])
            self.assertFalse(kwargs["check"])
            self.assertEqual(kwargs["env"]["EXTRACT_ROOT"], str(root / "extract"))
            self.assertEqual(kwargs["env"]["EXPECTED_SHA256"], expected_sha)
            self.assertEqual(kwargs["env"]["TRIPOSG_SETUP_ONLY"], "1")
            run_mock.return_value.check_returncode.assert_called_once()

    def test_inline_colab_launcher_rejects_non_positive_chunk_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "stub_bundle.tar.gz"
            archive.write_bytes(b"payload")

            with self.assertRaisesRegex(ValueError, "--inline-colab-chunk-size must be positive"):
                build_inline_colab_launcher(
                    archive_path=archive,
                    colab_archive_path="/content/stub_bundle.tar.gz",
                    extract_root="/content/stub",
                    expected_sha256="abc",
                    expected_size=archive.stat().st_size,
                    chunk_size=0,
                )

    def test_package_inputs_can_write_fetch_colab_launcher(self):
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "fetch_bundle.tar.gz"
            launcher = root / "fetch_colab.py"
            notebook = root / "fetch_colab.ipynb"
            payload_url = "https://raw.githubusercontent.com/example/repo/commit/fetch_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/fetch",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/fetch_bundle.tar.gz",
                fetch_colab_launcher_path=launcher,
                fetch_colab_notebook_path=notebook,
                fetch_colab_payload_url=payload_url,
                colab_env={"TRIPOSG_SETUP_ONLY": "1"},
                run_name="g4_fetch_test",
                eval_starts=[0],
                eval_limit=1,
            )
            launcher_text = launcher.read_text(encoding="utf-8")
            notebook_bytes = notebook.read_bytes()
            notebook_json = json.loads(notebook_bytes.decode("utf-8"))
            notebook_source = "".join(notebook_json["cells"][1]["source"])

        self.assertEqual(report["fetch_colab_launcher"], str(launcher))
        self.assertEqual(report["fetch_colab_notebook"], str(notebook))
        self.assertEqual(report["fetch_colab_payload_url"], payload_url)
        self.assertEqual(report["fetch_colab_expected_size"], report["output_size"])
        self.assertEqual(report["colab_env"], {"TRIPOSG_SETUP_ONLY": "1"})
        self.assertIn("fetch_colab_launcher_sha256", report)
        self.assertIn("fetch_colab_notebook_sha256", report)
        self.assertEqual(report["fetch_colab_notebook_sha256"], hashlib.sha256(notebook_bytes).hexdigest())
        self.assertNotIn(b"\r\n", notebook_bytes)
        self.assertIn(payload_url, launcher_text)
        self.assertIn(report["output_sha256"], launcher_text)
        self.assertEqual(notebook_json["nbformat"], 4)
        self.assertEqual(notebook_json["metadata"]["accelerator"], "GPU")
        self.assertEqual(notebook_json["metadata"]["kernelspec"]["name"], "python3")
        self.assertIn(payload_url, notebook_source)
        self.assertIn(report["output_sha256"], notebook_source)
        self.assertIn("completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)", notebook_source)
        self.assertIn("---RESULTS_SUMMARY_JSON---", notebook_source)
        self.assertIn("---RESULT_ARCHIVES_JSON---", notebook_source)
        self.assertIn("'results_compact_archive'", notebook_source)
        self.assertIn("completed.check_returncode()", notebook_source)
        self.assertIn("urllib.request.urlopen(PAYLOAD_URL)", launcher_text)
        self.assertIn("EXPECTED_SIZE", launcher_text)
        self.assertIn('"TRIPOSG_SETUP_ONLY": "1"', launcher_text)
        self.assertIn("env['EXTRACT_ROOT'] = str(EXTRACT_ROOT)", launcher_text)
        self.assertIn("env['EXPECTED_SHA256'] = EXPECTED_SHA256", launcher_text)
        self.assertIn("for key, value in LAUNCH_ENV.items():", launcher_text)
        self.assertIn("completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)", launcher_text)
        self.assertIn("---RESULTS_SUMMARY_JSON---", launcher_text)
        self.assertIn("---RESULT_ARCHIVES_JSON---", launcher_text)
        self.assertIn("'results_compact_archive'", launcher_text)
        self.assertIn("completed.check_returncode()", launcher_text)

    def test_package_inputs_writes_deterministic_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "image.png"
            mask = root / "mask.png"
            manifest = root / "manifest.jsonl"
            image.write_bytes(b"image")
            mask.write_bytes(b"mask")
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "full_image": str(image.relative_to(root)),
                        "masked_image": str(image.relative_to(root)),
                        "mask": str(mask.relative_to(root)),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "bundle.tar.gz"
            first = package_inputs(
                manifest=manifest,
                output=output,
                extract_root="/content/inputs/deterministic",
                root=root,
                include_run_script=True,
                run_name="deterministic_test",
                eval_starts=[0],
                eval_limit=1,
            )
            os.utime(image, (1_900_000_000, 1_900_000_000))
            os.utime(mask, (1_900_000_000, 1_900_000_000))
            second = package_inputs(
                manifest=manifest,
                output=output,
                extract_root="/content/inputs/deterministic",
                root=root,
                include_run_script=True,
                run_name="deterministic_test",
                eval_starts=[0],
                eval_limit=1,
            )

        self.assertEqual(first["output_size"], second["output_size"])
        self.assertEqual(first["output_sha256"], second["output_sha256"])

    def test_build_fetch_colab_launcher_materializes_stub_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "stub_bundle.tar.gz"
            stub_script = (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "echo fetch-stub\n"
            )
            with tarfile.open(archive, "w:gz") as tar:
                encoded = stub_script.encode("utf-8")
                info = tarfile.TarInfo("run_colab_eval.sh")
                info.mode = 0o755
                info.size = len(encoded)
                tar.addfile(info, io.BytesIO(encoded))
            archive_bytes = archive.read_bytes()
            expected_sha = hashlib.sha256(archive_bytes).hexdigest()
            payload_url = "https://example.invalid/stub_bundle.tar.gz"
            launcher_text = build_fetch_colab_launcher(
                payload_url=payload_url,
                colab_archive_path=str(root / "content" / "stub_bundle.tar.gz"),
                extract_root=str(root / "extract"),
                expected_sha256=expected_sha,
                expected_size=archive.stat().st_size,
                colab_env={"TRIPOSG_SETUP_ONLY": "1", "HUNYUAN3D_PREFETCH": "0"},
            )
            materialized_archive = root / "content" / "stub_bundle.tar.gz"
            extracted_script = root / "extract" / "run_colab_eval.sh"

            with patch("urllib.request.urlopen", return_value=io.BytesIO(archive_bytes)) as urlopen_mock:
                with patch("subprocess.run") as run_mock:
                    exec(compile(launcher_text, str(root / "fetch_colab.py"), "exec"), {"__name__": "__main__"})

            urlopen_mock.assert_called_once_with(payload_url)
            self.assertEqual(hashlib.sha256(materialized_archive.read_bytes()).hexdigest(), expected_sha)
            self.assertEqual(extracted_script.read_text(encoding="utf-8"), stub_script)
            run_mock.assert_called_once()
            args, kwargs = run_mock.call_args
            self.assertEqual(args[0], ["bash", str(extracted_script), str(materialized_archive)])
            self.assertFalse(kwargs["check"])
            self.assertEqual(kwargs["env"]["EXTRACT_ROOT"], str(root / "extract"))
            self.assertEqual(kwargs["env"]["EXPECTED_SHA256"], expected_sha)
            self.assertEqual(kwargs["env"]["TRIPOSG_SETUP_ONLY"], "1")
            self.assertEqual(kwargs["env"]["HUNYUAN3D_PREFETCH"], "0")
            run_mock.return_value.check_returncode.assert_called_once()

    def test_parse_colab_env_validates_key_value_pairs(self):
        self.assertEqual(
            parse_colab_env(["TRIPOSG_SETUP_ONLY=1", "HUNYUAN3D_PREFETCH=0"]),
            {"TRIPOSG_SETUP_ONLY": "1", "HUNYUAN3D_PREFETCH": "0"},
        )
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            parse_colab_env(["TRIPOSG_SETUP_ONLY"])
        with self.assertRaisesRegex(ValueError, "valid environment variable"):
            parse_colab_env(["BAD-NAME=1"])

    def test_fetch_colab_launcher_requires_url_and_run_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            full = dataset / "full.png"
            full.write_bytes(b"asset")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "sample", "full_image": str(full.relative_to(root))}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "--fetch-colab-launcher requires --include-run-script"):
                package_inputs(
                    manifest=manifest,
                    output=root / "bundle.tar.gz",
                    extract_root="/content/inputs/fetch",
                    root=root,
                    fetch_colab_launcher_path=root / "fetch_colab.py",
                    fetch_colab_payload_url="https://example.invalid/bundle.tar.gz",
                )
            with self.assertRaisesRegex(ValueError, "--fetch-colab-payload-url is required"):
                package_inputs(
                    manifest=manifest,
                    output=root / "bundle.tar.gz",
                    extract_root="/content/inputs/fetch",
                    root=root,
                    include_run_script=True,
                    fetch_colab_launcher_path=root / "fetch_colab.py",
                )
            with self.assertRaisesRegex(ValueError, "--fetch-colab-notebook requires --include-run-script"):
                package_inputs(
                    manifest=manifest,
                    output=root / "bundle.tar.gz",
                    extract_root="/content/inputs/fetch",
                    root=root,
                    fetch_colab_notebook_path=root / "fetch_colab.ipynb",
                    fetch_colab_payload_url="https://example.invalid/bundle.tar.gz",
                )

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
                skip_cache=True,
                cache_full=True,
                eval_starts=[40, 50],
                eval_limit=2,
                eval_steps=20,
                eval_inpaint_max_dimension=512,
                depth_provider="depth-anything-v2",
                depth_model="depth-anything/Depth-Anything-V2-Small-hf",
                stl_target_dimension=96,
                score_profile="stl-quality",
                candidate_method="triposr_api_masked_repaired_direct_mesh",
                current_method="mirror",
                min_paired_n=2,
                allow_missing_split_audit=True,
                contact_sheet_methods="masked,mirror,biharmonic,qwen_edit_s20_s512",
            )
            with tarfile.open(archive, "r:gz") as tar:
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertEqual(report["rewritten_lora_weights"], "")
        self.assertEqual(report["cache_providers"], ["qwen-image-edit"])
        self.assertTrue(report["skip_cache"])
        self.assertTrue(report["cache_full"])
        self.assertEqual(report["max_method_failures"], 2)
        self.assertIn("--run-name g4_qwen_sanity", archive_run_script)
        self.assertIn("--stage cache --stage eval --stage combine", archive_run_script)
        self.assertIn("python -m pip install -U pip", archive_run_script)
        self.assertIn("python -m pip install -r backend/requirements-cuda.txt", archive_run_script)
        self.assertIn('if [[ "${COLAB_SKIP_BACKEND_INSTALL:-0}" == "1" ]]; then', archive_run_script)
        self.assertIn("Skipping backend pip install because COLAB_SKIP_BACKEND_INSTALL=1", archive_run_script)
        self.assertIn('mkdir -p "$EXTRACT_ROOT" "$(dirname "$RUN_LOG")" "$(dirname "$RESULTS_SUMMARY")" "$(dirname "$RESULTS_ARCHIVE")"', archive_run_script)
        self.assertIn("set +e\n(\nset -euo pipefail\n", archive_run_script)
        self.assertIn('export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"', archive_run_script)
        self.assertIn(
            'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
            archive_run_script,
        )
        self.assertIn('export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"', archive_run_script)
        self.assertIn('export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"', archive_run_script)
        self.assertIn(') 2>&1 | tee "$RUN_LOG"\nrun_status="${PIPESTATUS[0]}"', archive_run_script)
        self.assertIn('export RUN_STATUS="$run_status"\ncd "$REPO_DIR"\npython - <<\'PY\'', archive_run_script)
        self.assertIn("'repair_preclean_retriangulation_accepted_mean'", archive_run_script)
        self.assertIn(
            "'repair_preclean_retriangulated_geometry_preserved_mean'",
            archive_run_script,
        )
        self.assertIn(
            "'repair_preclean_retriangulated_vertex_displacement_max_normalized_median'",
            archive_run_script,
        )
        self.assertNotIn('backend.benchmark.colab_g4_orchestrator --use-current-repo --run-name g4_qwen_sanity 2>&1 | tee "$RUN_LOG"', archive_run_script)
        self.assertIn("--modern-config backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_qwen_edit_g4_depth_stl.json", archive_run_script)
        self.assertIn("--cache-provider qwen-image-edit", archive_run_script)
        self.assertIn("--skip-cache", archive_run_script)
        self.assertIn("--cache-full", archive_run_script)
        self.assertIn("--eval-limit 2", archive_run_script)
        self.assertIn("--eval-steps 20", archive_run_script)
        self.assertIn("--eval-inpaint-max-dimension 512", archive_run_script)
        self.assertIn("--depth-provider depth-anything-v2", archive_run_script)
        self.assertIn("--stl-target-dimension 96", archive_run_script)
        self.assertIn("--score-profile stl-quality", archive_run_script)
        self.assertEqual(report["candidate_method"], "triposr_api_masked_repaired_direct_mesh")
        self.assertEqual(report["current_method"], "mirror")
        self.assertIn("--candidate-method triposr_api_masked_repaired_direct_mesh", archive_run_script)
        self.assertIn("--current-method mirror", archive_run_script)
        self.assertIn("--min-paired-n 2", archive_run_script)
        self.assertIn("--max-method-failures 2", archive_run_script)
        self.assertIn("--allow-missing-split-audit", archive_run_script)
        self.assertIn("--contact-sheet-methods masked,mirror,biharmonic,qwen_edit_s20_s512", archive_run_script)
        self.assertIn("--manifest /content/inputs/qwen/inputs/manifest.jsonl", archive_run_script)
        self.assertIn("LORA_PATH=''", archive_run_script)
        self.assertIn('if [[ -n "$LORA_PATH" ]]; then', archive_run_script)
        self.assertNotIn("--existing-lora-weights", archive_run_script)

    def test_package_inputs_can_embed_triposr_provider_setup(self):
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "triposr_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/triposr",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/triposr_bundle.tar.gz",
                run_name="g4_triposr_setup",
                modern_config="backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposr_mirror_bbox_candidate.json",
                skip_cache=True,
                eval_starts=[0],
                eval_limit=1,
                score_profile="stl-quality",
                candidate_method="triposr_api_masked_repaired_stl_mirror_bbox_direct_mesh",
                current_method="mirror",
                include_triposr_setup=True,
            )
            with tarfile.open(archive, "r:gz") as tar:
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertTrue(report["include_triposr_setup"])
        self.assertIn("https://github.com/VAST-AI-Research/TripoSR", archive_run_script)
        self.assertIn('TRIPOSR_VENV="${TRIPOSR_VENV:-/content/triposr-venv}"', archive_run_script)
        self.assertIn("python -m virtualenv --system-site-packages", archive_run_script)
        self.assertIn('export PYTHONPATH="$TRIPOSR_DIR:${PYTHONPATH:-}"', archive_run_script)
        self.assertIn("import torchmcubes", archive_run_script)
        self.assertIn("transformers==4.35.0", archive_run_script)
        self.assertIn("git+https://github.com/tatsy/torchmcubes.git", archive_run_script)
        self.assertIn("--candidate-method triposr_api_masked_repaired_stl_mirror_bbox_direct_mesh", archive_run_script)

    def test_package_inputs_can_embed_triposg_provider_setup(self):
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "triposg_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/triposg",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/triposg_bundle.tar.gz",
                run_name="g4_triposg_setup",
                modern_config="backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_triposg_direct_mesh_smoke.json",
                skip_cache=True,
                eval_starts=[0],
                eval_limit=1,
                score_profile="stl-quality",
                candidate_method="triposg_masked_repaired_direct_mesh",
                current_method="mirror",
                colab_require_gpu_name_regex="RTX PRO 6000|Blackwell",
                colab_min_gpu_memory_gb=90,
                include_triposg_setup=True,
            )
            with tarfile.open(archive, "r:gz") as tar:
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertTrue(report["include_triposg_setup"])
        self.assertEqual(
            report["triposg_model_snapshots"]["triposg"]["revision"],
            DEFAULT_TRIPOSG_MODEL_REVISION,
        )
        self.assertEqual(
            report["triposg_model_snapshots"]["rembg"]["revision"],
            DEFAULT_TRIPOSG_REMBG_REVISION,
        )
        self.assertEqual(report["colab_require_gpu_name_regex"], "RTX PRO 6000|Blackwell")
        self.assertEqual(report["colab_min_gpu_memory_gb"], 90)
        self.assertLess(archive_run_script.index("COLAB_REQUIRE_GPU_NAME_REGEX='RTX PRO 6000|Blackwell'"), archive_run_script.index('TRIPOSG_DIR="${TRIPOSG_DIR:-/content/TripoSG}"'))
        self.assertLess(archive_run_script.index("--query-gpu=name,memory.total"), archive_run_script.index('TRIPOSG_DIR="${TRIPOSG_DIR:-/content/TripoSG}"'))
        self.assertIn("https://github.com/VAST-AI-Research/TripoSG", archive_run_script)
        self.assertIn("TRIPOSG_REF=", archive_run_script)
        self.assertIn("fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c", archive_run_script)
        self.assertIn('git -C "$TRIPOSG_DIR" checkout "$TRIPOSG_REF"', archive_run_script)
        self.assertIn('git -C "$TRIPOSG_DIR" reset --hard "$TRIPOSG_REF"', archive_run_script)
        self.assertIn('TRIPOSG_VENV="${TRIPOSG_VENV:-/content/triposg-venv}"', archive_run_script)
        self.assertIn('export PYTHONPATH="$TRIPOSG_DIR:${PYTHONPATH:-}"', archive_run_script)
        self.assertIn("importlib.import_module(name)", archive_run_script)
        self.assertIn("'diso'", archive_run_script)
        self.assertIn("non-flash decoder patch applied", archive_run_script)
        self.assertIn("TRIPOSG_USE_FLASH_DECODER", archive_run_script)
        self.assertIn("use_flash_decoder=os.environ.get", archive_run_script)
        self.assertIn("from diso import DiffDMC", archive_run_script)
        self.assertIn("TRIPOSG_INSTALL_DISO", archive_run_script)
        self.assertIn("diso==0.1.4", archive_run_script)
        self.assertIn("packaging ninja", archive_run_script)
        self.assertIn("fast-simplification", archive_run_script)
        self.assertIn("pip install -v --no-cache-dir --no-build-isolation --no-binary=:all: diso==0.1.4", archive_run_script)
        self.assertIn("triposg.pipelines.pipeline_triposg", archive_run_script)
        self.assertIn("/tmp/triposg_requirements_colab.txt", archive_run_script)
        self.assertIn("numpy==2.0.2", archive_run_script)
        self.assertIn("TripoSG setup checkpoint: Python deps importable", archive_run_script)
        self.assertIn(
            'PYTHONPATH="$TRIPOSG_DIR${PYTHONPATH:+:$PYTHONPATH}"',
            archive_run_script,
        )
        self.assertIn(
            '"$TRIPOSG_DIR/scripts/inference_triposg.py" --help',
            archive_run_script,
        )
        self.assertIn("TripoSG setup checkpoint: CLI imports ok", archive_run_script)
        self.assertIn("TRIPOSG_PREFETCH", archive_run_script)
        self.assertIn("torch.cuda.is_available()", archive_run_script)
        self.assertIn("VAST-AI/TripoSG", archive_run_script)
        self.assertIn("briaai/RMBG-1.4", archive_run_script)
        self.assertIn(DEFAULT_TRIPOSG_MODEL_REVISION, archive_run_script)
        self.assertIn(DEFAULT_TRIPOSG_REMBG_REVISION, archive_run_script)
        self.assertIn("snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)", archive_run_script)
        self.assertIn("python -m backend.benchmark.patch_triposg_sources", archive_run_script)
        self.assertIn("export TRIPOSG_HF_LOCAL_ONLY=1", archive_run_script)
        self.assertIn("TRIPOSG_SETUP_ONLY", archive_run_script)
        self.assertIn("--candidate-method triposg_masked_repaired_direct_mesh", archive_run_script)

    def test_package_inputs_can_embed_hunyuan3d_provider_setup(self):
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
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "hunyuan_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/hunyuan",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/hunyuan_bundle.tar.gz",
                run_name="g4_hunyuan_setup",
                modern_config="backend/benchmark/experiment_configs/modelnet10_60_balanced_stl_quality_hunyuan3d_shape_candidate.json",
                skip_cache=True,
                eval_starts=[0],
                eval_limit=1,
                score_profile="stl-quality",
                candidate_method="hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh",
                current_method="mirror",
                colab_require_gpu_name_regex="RTX PRO 6000|Blackwell",
                colab_min_gpu_memory_gb=90,
                include_hunyuan3d_setup=True,
            )
            with tarfile.open(archive, "r:gz") as tar:
                archive_run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertTrue(report["include_hunyuan3d_setup"])
        self.assertEqual(report["colab_require_gpu_name_regex"], "RTX PRO 6000|Blackwell")
        self.assertEqual(report["colab_min_gpu_memory_gb"], 90)
        self.assertLess(archive_run_script.index("COLAB_REQUIRE_GPU_NAME_REGEX='RTX PRO 6000|Blackwell'"), archive_run_script.index('HUNYUAN3D_DIR="${HUNYUAN3D_DIR:-/content/Hunyuan3D-2.1}"'))
        self.assertLess(archive_run_script.index("--query-gpu=name,memory.total"), archive_run_script.index('HUNYUAN3D_DIR="${HUNYUAN3D_DIR:-/content/Hunyuan3D-2.1}"'))
        self.assertIn("https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1", archive_run_script)
        self.assertIn(
            'HUNYUAN3D_REF="${HUNYUAN3D_REF:-82920d643c0dc2f7bfd7255f45f62d386edfe60c}"',
            archive_run_script,
        )
        self.assertIn("git clone --filter=blob:none --sparse", archive_run_script)
        self.assertIn('git -C "$HUNYUAN3D_DIR" checkout "$HUNYUAN3D_REF"', archive_run_script)
        self.assertIn('git -C "$HUNYUAN3D_DIR" fetch origin "$HUNYUAN3D_REF" || true', archive_run_script)
        self.assertIn(
            'git -C "$HUNYUAN3D_DIR" sparse-checkout set hy3dshape/hy3dshape hy3dshape/configs',
            archive_run_script,
        )
        self.assertIn('HUNYUAN3D_VENV="${HUNYUAN3D_VENV:-/content/hunyuan3d-venv}"', archive_run_script)
        self.assertIn('HUNYUAN3D_DEPS="${HUNYUAN3D_DEPS:-/content/hunyuan3d-deps}"', archive_run_script)
        self.assertIn('HUNYUAN3D_VENV_BACKEND="${HUNYUAN3D_VENV_BACKEND:-target}"', archive_run_script)
        self.assertIn("Creating Hunyuan3D Python env wrapper with target deps", archive_run_script)
        self.assertIn('HUNYUAN3D_WRAPPER_TMP="$HUNYUAN3D_VENV/bin/python.target.$$"', archive_run_script)
        self.assertIn('cat > "$HUNYUAN3D_WRAPPER_TMP"', archive_run_script)
        self.assertIn('export PYTHONPATH="$HUNYUAN3D_DEPS:$HUNYUAN3D_DIR/hy3dshape:$HUNYUAN3D_DIR:${PYTHONPATH:-}"', archive_run_script)
        self.assertIn('exec python "$@"', archive_run_script)
        self.assertIn('mv -f "$HUNYUAN3D_WRAPPER_TMP" "$HUNYUAN3D_VENV/bin/python"', archive_run_script)
        self.assertIn("Hunyuan3D setup checkpoint: target wrapper ready", archive_run_script)
        self.assertIn(
            'HUNYUAN3D_PIP_INSTALL=(python -m pip install --upgrade --target "$HUNYUAN3D_DEPS" --no-deps)',
            archive_run_script,
        )
        self.assertIn("Creating Hunyuan3D Python env with stdlib venv", archive_run_script)
        self.assertIn("python -m venv --system-site-packages", archive_run_script)
        self.assertIn("Creating Hunyuan3D Python env with virtualenv", archive_run_script)
        self.assertIn("python -m virtualenv --system-site-packages", archive_run_script)
        self.assertIn('export PYTHONPATH="$HUNYUAN3D_DIR/hy3dshape:$HUNYUAN3D_DIR:${PYTHONPATH:-}"', archive_run_script)
        self.assertIn("import importlib.util", archive_run_script)
        self.assertIn("except ModuleNotFoundError:", archive_run_script)
        self.assertIn("'hy3dshape.pipelines'", archive_run_script)
        self.assertIn("'pymeshlab'", archive_run_script)
        self.assertIn("missing Hunyuan3D Python deps: ", archive_run_script)
        self.assertIn("Installing Hunyuan3D Python deps: core diffusers stack", archive_run_script)
        self.assertIn("Installing Hunyuan3D Python deps: geometry and image stack", archive_run_script)
        self.assertIn("Installing Hunyuan3D Python deps: model helpers", archive_run_script)
        self.assertIn("Installing Hunyuan3D Python deps: pymeshlab", archive_run_script)
        self.assertIn('"${HUNYUAN3D_PIP_INSTALL[@]}" timm torchdiffeq', archive_run_script)
        self.assertIn('"${HUNYUAN3D_PIP_INSTALL[@]}" pymeshlab==2023.12.post3', archive_run_script)
        self.assertIn("Hunyuan3D setup checkpoint: Python deps importable", archive_run_script)
        self.assertIn("HUNYUAN3D_PREFETCH:-1", archive_run_script)
        self.assertIn("--prefetch-only", archive_run_script)
        self.assertIn("Hunyuan3D setup checkpoint: shape weights prefetched", archive_run_script)
        self.assertIn("COLAB_PROVIDER_SETUP_ONLY", archive_run_script)
        self.assertIn("HUNYUAN3D_SETUP_ONLY", archive_run_script)
        self.assertIn("TRIPOSR_SETUP_ONLY", archive_run_script)
        self.assertIn("TRIPOSG_SETUP_ONLY", archive_run_script)
        self.assertIn("Provider setup only requested; skipping benchmark stages", archive_run_script)
        self.assertIn("'provider_setup_only'", archive_run_script)
        self.assertIn("diffusers==0.30.0", archive_run_script)
        self.assertIn("transformers==4.46.0", archive_run_script)
        self.assertIn("--candidate-method hunyuan3d_shape_masked_repaired_stl_mirror_bbox_direct_mesh", archive_run_script)


class ColabOutputSummaryExtractionTests(unittest.TestCase):
    def sample_output(self) -> str:
        return """
Launching /content/inputs/run_colab_eval.sh
---RESULTS_SUMMARY_JSON---
{
  "provider_setup_only": false,
  "run_name": "g4_stl_first_triposg_prefill_s40_n10",
  "run_status": 0,
  "eval_summaries": [
    {
      "top_method": "mirror",
      "selection_decision_exists": true
    }
  ]
}
---RESULT_ARCHIVES_JSON---
[
  {
    "bytes": 2905172,
    "path": "/content/g4_stl_first_triposg_prefill_s40_n10_results.tar.gz",
    "sha256": "abc123"
  }
]
CompletedProcess(args=['bash'], returncode=0)
"""

    def test_parse_colab_output_reads_latest_marked_json_blocks(self):
        parsed = parse_colab_output("stale\n" + self.sample_output())

        self.assertTrue(parsed["markers_found"]["results_summary"])
        self.assertTrue(parsed["markers_found"]["result_archives"])
        self.assertEqual(parsed["results_summary"]["run_status"], 0)
        self.assertEqual(parsed["results_summary"]["run_name"], "g4_stl_first_triposg_prefill_s40_n10")
        self.assertEqual(parsed["result_archives"][0]["bytes"], 2905172)
        self.assertEqual(parsed["result_archives"][0]["sha256"], "abc123")

    def test_extract_colab_output_writes_summary_archives_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_text = root / "colab_output.txt"
            output_text.write_text(self.sample_output(), encoding="utf-8")

            report = extract_colab_output(output_text, root / "extracted")
            summary = json.loads((root / "extracted" / "results_summary.json").read_text(encoding="utf-8"))
            archives = json.loads((root / "extracted" / "result_archives.json").read_text(encoding="utf-8"))
            report_json = json.loads((root / "extracted" / "colab_output_extract_report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["run_status"], 0)
        self.assertEqual(report["archive_count"], 1)
        self.assertEqual(summary["run_name"], "g4_stl_first_triposg_prefill_s40_n10")
        self.assertEqual(archives[0]["path"], "/content/g4_stl_first_triposg_prefill_s40_n10_results.tar.gz")
        self.assertEqual(report_json["archive_paths"], ["/content/g4_stl_first_triposg_prefill_s40_n10_results.tar.gz"])


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

    def test_source_bbox_configs_are_automatically_oracle_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "bbox_provenance.json"
            config_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "source_probe",
                            "method": "external-image-to-mesh",
                            "stl_mode": "single-image-mesh",
                            "direct_mesh_command": 'provider --mesh-target-bbox-extents "{source_bbox_extents}"',
                        },
                        {
                            "name": "inferred_candidate",
                            "method": "external-image-to-mesh",
                            "stl_mode": "single-image-mesh",
                            "direct_mesh_command": 'provider --mesh-target-bbox-extents "{inferred_bbox_extents}"',
                        },
                    ]
                ),
                encoding="utf-8",
            )

            experiments = load_experiments(str(config_path))

        by_name = {experiment["name"]: experiment for experiment in experiments}
        self.assertEqual(by_name["source_probe"]["direct_mesh_bbox_source"], "source")
        self.assertTrue(by_name["source_probe"]["oracle_diagnostic"])
        self.assertEqual(by_name["inferred_candidate"]["direct_mesh_bbox_source"], "inferred")
        self.assertFalse(by_name["inferred_candidate"].get("oracle_diagnostic", False))

    def test_bbox_provenance_rejects_metadata_that_hides_source_placeholder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "conflicting_bbox_provenance.json"
            config_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "mislabelled_source_probe",
                            "method": "external-image-to-mesh",
                            "stl_mode": "single-image-mesh",
                            "direct_mesh_bbox_source": "inferred",
                            "direct_mesh_command": 'provider --mesh-target-bbox-extents "{source_bbox_extents}"',
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "conflicts with hidden-source placeholder"):
                load_experiments(str(config_path))

    def test_bbox_provenance_detects_axis_placeholders_and_transitive_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "transitive_bbox_provenance.json"
            config_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "calibration_a",
                            "method": "external-image-to-mesh",
                            "direct_mesh_command": (
                                'provider --bbox "{source_bbox_x},{source_bbox_y},{source_bbox_z}"'
                            ),
                        },
                        {
                            "name": "reference_candidate",
                            "method": "external-image-to-mesh",
                            "direct_mesh_reference_method": "calibration_a",
                            "direct_mesh_command": (
                                'provider --bbox "{reference_bbox_extents}"'
                            ),
                        },
                    ]
                ),
                encoding="utf-8",
            )

            experiments = load_experiments(str(config_path))

        by_name = {experiment["name"]: experiment for experiment in experiments}
        self.assertEqual(by_name["calibration_a"]["direct_mesh_bbox_source"], "source")
        self.assertTrue(by_name["calibration_a"]["oracle_diagnostic"])
        self.assertEqual(by_name["reference_candidate"]["direct_mesh_bbox_source"], "reference")
        self.assertTrue(by_name["reference_candidate"]["oracle_diagnostic"])

    def test_optimize_auto_selection_skips_oracle_score_leader(self):
        args = SimpleNamespace(
            candidate_method=None,
            baseline_method="masked",
            weight=["masked_mae_median=-4"],
            score_profile="default",
        )
        rows = [
            {"method": "masked", "masked_mae_median": "0.50"},
            {
                "method": "source_probe",
                "masked_mae_median": "0.01",
                "oracle_diagnostic": True,
            },
            {"method": "mirror", "masked_mae_median": "0.10"},
        ]

        self.assertEqual(infer_selection_candidate(args, rows), "mirror")

    def test_triposg_bbox_tuning_config_uses_mirror_bbox_placeholder(self):
        config_path = (
            Path(__file__).resolve().parents[2]
            / "backend"
            / "benchmark"
            / "experiment_configs"
            / "modelnet10_60_balanced_stl_quality_triposg_bbox_tuning_candidates.json"
        )

        self.assertTrue(config_path.exists(), f"Missing experiment config: {config_path}")
        experiments = load_experiments(str(config_path))
        by_name = {experiment["name"]: experiment for experiment in experiments}

        expected_baselines = {
            "masked": "masked",
            "mirror": "mirror",
            "biharmonic": "biharmonic",
            "source_mesh_oracle": "source-mesh-oracle",
        }
        for name, method in expected_baselines.items():
            self.assertIn(name, by_name)
            self.assertEqual(by_name[name]["method"], method)

        expected_direct_inputs = {
            "triposg_masked_repaired_stl_mirror_bbox_direct_mesh": "masked",
            "triposg_mirror_prefill_repaired_stl_mirror_bbox_direct_mesh": "mirror",
            "triposg_biharmonic_prefill_repaired_stl_mirror_bbox_direct_mesh": "biharmonic",
        }
        for name, direct_input in expected_direct_inputs.items():
            self.assertIn(name, by_name)
            experiment = by_name[name]
            command = experiment["direct_mesh_command"]
            self.assertEqual(experiment["method"], "external-image-to-mesh")
            self.assertEqual(experiment["stl_mode"], "single-image-mesh")
            self.assertEqual(experiment["direct_mesh_input"], direct_input)
            self.assertIn("--provider triposg", command)
            self.assertIn('--mesh-target-bbox-extents "{mirror_bbox_extents}"', command)
            self.assertIn("--mesh-repair printable", command)
            self.assertIn("--mesh-target-faces 40000", command)

        for name in [
            "triposg_mirror_prefill_repaired_stl_mirror_bbox_direct_mesh",
            "triposg_biharmonic_prefill_repaired_stl_mirror_bbox_direct_mesh",
        ]:
            self.assertEqual(by_name[name]["direct_mesh_reference_method"], "mirror")

        aspect_clamped = by_name[
            "triposg_biharmonic_prefill_repaired_stl_aspect_clamped2p25_direct_mesh"
        ]
        self.assertEqual(aspect_clamped["direct_mesh_input"], "biharmonic")
        self.assertIn("--mesh-max-bbox-aspect-ratio 2.25", aspect_clamped["direct_mesh_command"])

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

    def test_experiment_metadata_records_stl_mode(self):
        self.assertEqual(
            experiment_metadata({"name": "mirror", "method": "mirror"}, default_emit_stl=True)["stl_mode"],
            "depth-relief",
        )
        self.assertEqual(
            experiment_metadata({"name": "triposr", "method": "external-image-to-mesh"})["stl_mode"],
            "single-image-mesh",
        )
        self.assertEqual(
            experiment_metadata({"name": "mv", "method": "external-multiview-to-mesh"})["stl_mode"],
            "multiview-mesh",
        )

    def test_load_experiments_rejects_unknown_stl_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "bad_stl_mode.json"
            config_path.write_text(
                json.dumps([{"name": "bad", "method": "mirror", "stl_mode": "preview-only"}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown stl_mode"):
                load_experiments(str(config_path))

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

    def test_image_to_mesh_provider_preflight_accepts_configured_triposg_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "TripoSG"
            scripts_dir = provider_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "inference_triposg.py").write_text("# provider entrypoint\n", encoding="utf-8")
            command = (
                f'"{sys.executable}" -m backend.benchmark.run_image_to_mesh_provider '
                f'--provider triposg --provider-dir "{provider_dir}" --input-image "{{input_image}}" '
                f'--output-mesh "{{output_mesh}}" --output-stl "{{output_stl}}"'
            )

            parsed = parse_provider_command(command)
            self.assertIsNotNone(parsed)
            row = provider_preflight_row(parsed, experiment_names=["triposg_candidate"])

        self.assertEqual(parsed["provider"], "triposg")
        self.assertTrue(row["runnable"])
        self.assertEqual(row["readiness"], "ready")
        self.assertEqual(row["experiment_names"], ["triposg_candidate"])
        self.assertTrue(row["checks"]["entrypoint_found"])
        self.assertEqual(row["checks"]["entrypoint"], "scripts/inference_triposg.py")

    def test_image_to_mesh_provider_preflight_accepts_builtin_visual_hull(self):
        command = (
            f'"{sys.executable}" -m backend.benchmark.run_image_to_mesh_provider '
            '--provider multiview-visual-hull --input-image "{input_image}" '
            '--input-bundle "{input_bundle}" --output-mesh "{output_mesh}" --output-stl "{output_stl}"'
        )

        parsed = parse_provider_command(command)
        self.assertIsNotNone(parsed)
        row = provider_preflight_row(parsed, experiment_names=["visual_hull"])

        self.assertEqual(parsed["provider"], "multiview-visual-hull")
        self.assertEqual(row["readiness"], "ready")
        self.assertTrue(row["runnable"])
        self.assertEqual(row["experiment_names"], ["visual_hull"])
        self.assertTrue(row["checks"]["provider_dir_resolved"])

    def test_image_to_mesh_provider_preflight_report_can_fail_fast(self):
        def fake_preflight(_experiments):
            return [
                {
                    "provider": "triposg",
                    "readiness": "missing",
                    "runnable": False,
                    "experiment_names": ["triposg_candidate"],
                    "setup_errors": ["Provider repo is missing. Configure one of: TRIPOSG_DIR."],
                    "checks": {"provider_dir_resolved": False},
                }
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            args = SimpleNamespace(require_image_to_mesh_providers=True)
            experiments = [
                {"name": "mirror", "method": "mirror"},
                {
                    "name": "triposg_candidate",
                    "method": "external-image-to-mesh",
                    "direct_mesh_command": (
                        f'"{sys.executable}" -m backend.benchmark.run_image_to_mesh_provider '
                        '--provider triposg --input-image "{input_image}" --output-mesh "{output_mesh}" '
                        '--output-stl "{output_stl}"'
                    ),
                },
            ]

            with self.assertRaisesRegex(RuntimeError, "triposg.*Provider repo is missing"):
                write_image_to_mesh_provider_preflight(args, experiments, output_dir, preflight=fake_preflight)

            report = json.loads((output_dir / "image_to_mesh_provider_preflight.json").read_text(encoding="utf-8"))

        self.assertTrue(report["require_runnable"])
        self.assertEqual(report["rows"][0]["provider"], "triposg")
        self.assertFalse(report["rows"][0]["runnable"])
        self.assertEqual(report["rows"][0]["experiment_names"], ["triposg_candidate"])

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

    def test_combined_selection_applies_surface_ratio_thresholds(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.6"},
            {"method": "current", "success_rate": "1.0", "masked_mae_median": "0.3"},
            {"method": "candidate", "success_rate": "1.0", "masked_mae_median": "0.1"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.6"},
            {
                "sample_id": "a",
                "method": "current",
                "masked_mae": "0.3",
                "mesh_surface_chamfer_l1": "0.10",
                "mesh_surface_hausdorff95": "0.20",
            },
            {
                "sample_id": "a",
                "method": "candidate",
                "masked_mae": "0.1",
                "mesh_surface_chamfer_l1": "0.12",
                "mesh_surface_hausdorff95": "0.23",
            },
        ]
        args = SimpleNamespace(
            baseline_method="masked",
            candidate_method="candidate",
            current_method="current",
            min_success_rate=1.0,
            min_paired_n=1,
            min_win_rate=0.0,
            min_ci95_low=-1.0,
            min_score_margin=0.0,
            min_stl_watertight=1.0,
            min_stl_positive_volume=1.0,
            max_mesh_surface_chamfer_ratio_vs_current=1.1,
            max_mesh_surface_hausdorff95_ratio_vs_current=1.1,
            paired_bootstrap_samples=0,
            paired_bootstrap_seed=1234,
            score_profile="default",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            output_json, _ = write_combined_selection(
                args,
                output_dir,
                summary_rows,
                per_sample_rows,
                {"masked_mae_median": -4.0},
            )
            decision = json.loads(output_json.read_text(encoding="utf-8"))

        failed = {check["name"] for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertIn("paired_mesh_surface_chamfer_ratio_vs_current", failed)
        self.assertIn("paired_mesh_surface_hausdorff95_ratio_vs_current", failed)


class StlResultIngestRegressionTests(unittest.TestCase):
    def test_result_discovery_ignores_method_summaries_nested_under_aggregate_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            method_dir = run_dir / "triposg"
            method_dir.mkdir(parents=True)
            (run_dir / "aggregate_summary.csv").write_text("method,success_rate\nmasked,1\n", encoding="utf-8")
            (method_dir / "summary_metrics.csv").write_text(
                "method,success_rate\nexternal-image-to-mesh,1\n",
                encoding="utf-8",
            )

            discovered = discover_result_runs(root)

        self.assertEqual(discovered, [run_dir])

    def write_stl_result_run(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True)
        summary_rows = [
            {
                "method": "masked",
                "base_method": "masked",
                "stl_mode": "depth-relief",
                "n": "2",
                "attempted_n": "2",
                "success_rate": "1.0",
                "error_count": "0",
                "mesh_surface_chamfer_l1_median": "0.40",
                "mesh_surface_hausdorff95_median": "0.50",
                "silhouette_iou_masked_median": "0.70",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_faces_per_bbox_volume_log1p_median": "5.0",
            },
            {
                "method": "mirror",
                "base_method": "mirror",
                "stl_mode": "depth-relief",
                "n": "2",
                "attempted_n": "2",
                "success_rate": "1.0",
                "error_count": "0",
                "mesh_surface_chamfer_l1_median": "0.22",
                "mesh_surface_hausdorff95_median": "0.33",
                "silhouette_iou_masked_median": "0.74",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_faces_per_bbox_volume_log1p_median": "5.1",
            },
            {
                "method": "hunyuan3d_shape_repaired",
                "base_method": "external-image-to-mesh",
                "stl_mode": "single-image-mesh",
                "n": "2",
                "attempted_n": "2",
                "success_rate": "1.0",
                "error_count": "0",
                "mesh_surface_chamfer_l1_median": "0.12",
                "mesh_surface_hausdorff95_median": "0.20",
                "silhouette_iou_masked_median": "0.81",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_faces_per_bbox_volume_log1p_median": "5.4",
            },
            {
                "method": "vggt_multiview_repaired",
                "base_method": "external-multiview-to-mesh",
                "stl_mode": "multiview-mesh",
                "n": "2",
                "attempted_n": "2",
                "success_rate": "1.0",
                "error_count": "0",
                "mesh_surface_chamfer_l1_median": "0.18",
                "mesh_surface_hausdorff95_median": "0.24",
                "silhouette_iou_masked_median": "0.78",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_faces_per_bbox_volume_log1p_median": "5.2",
            },
            {
                "method": "source_mesh_oracle",
                "base_method": "source-mesh-oracle",
                "stl_mode": "source-mesh-oracle",
                "n": "2",
                "attempted_n": "2",
                "success_rate": "1.0",
                "error_count": "0",
                "mesh_surface_chamfer_l1_median": "0.00",
                "mesh_surface_hausdorff95_median": "0.00",
                "silhouette_iou_masked_median": "1.0",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_faces_per_bbox_volume_log1p_median": "4.0",
            },
        ]
        for row in summary_rows:
            row.update(
                {
                    "stl_exists_median": "1.0",
                    "stl_winding_consistent_median": "1.0",
                    "stl_bbox_has_volume_median": "1.0",
                    "stl_nonmanifold_edge_count_log1p_median": "0.0",
                    "stl_degenerate_face_ratio_median": "0.0",
                    "stl_component_excess_log1p_median": "0.0",
                    "stl_bbox_aspect_ratio_median": "2.0",
                    "repair_convex_hull_used_mean": (
                        "0.0" if "repaired" in row["method"] else ""
                    ),
                    "repair_volume_fill_ratio_relative_change_abs_median": (
                        "0.1" if "repaired" in row["method"] else ""
                    ),
                }
            )
        fieldnames = list(summary_rows[0].keys())
        with (run_dir / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        per_sample_fieldnames = [
            "sample_id",
            "method",
            "mesh_surface_chamfer_l1",
            "stl_exists",
            "stl_is_watertight",
            "stl_is_volume",
            "stl_is_manifold",
            "stl_winding_consistent",
            "stl_positive_volume",
            "stl_single_component",
            "stl_bbox_has_volume",
            "stl_nonmanifold_edge_count_log1p",
            "stl_degenerate_face_ratio",
            "stl_component_excess_log1p",
            "stl_bbox_aspect_ratio",
            "stl_faces_per_bbox_volume_log1p",
            "repair_volume_fill_ratio_relative_change_abs",
        ]
        with (run_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=per_sample_fieldnames)
            writer.writeheader()
            for sample_id in ("a", "b"):
                for row in summary_rows:
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "method": row["method"],
                            "mesh_surface_chamfer_l1": row["mesh_surface_chamfer_l1_median"],
                            "stl_exists": row["stl_exists_median"],
                            "stl_is_watertight": row["stl_is_watertight_median"],
                            "stl_is_volume": row["stl_is_volume_median"],
                            "stl_is_manifold": row["stl_is_manifold_median"],
                            "stl_winding_consistent": row["stl_winding_consistent_median"],
                            "stl_positive_volume": row["stl_positive_volume_median"],
                            "stl_single_component": row["stl_single_component_median"],
                            "stl_bbox_has_volume": row["stl_bbox_has_volume_median"],
                            "stl_nonmanifold_edge_count_log1p": row["stl_nonmanifold_edge_count_log1p_median"],
                            "stl_degenerate_face_ratio": row["stl_degenerate_face_ratio_median"],
                            "stl_component_excess_log1p": row["stl_component_excess_log1p_median"],
                            "stl_bbox_aspect_ratio": row["stl_bbox_aspect_ratio_median"],
                            "stl_faces_per_bbox_volume_log1p": row["stl_faces_per_bbox_volume_log1p_median"],
                            "repair_volume_fill_ratio_relative_change_abs": (
                                "0.1" if "repaired" in row["method"] else ""
                            ),
                        }
                    )
        return run_dir

    def test_compact_results_archive_is_ingestable_and_excludes_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "compact_run"
            eval_dir = self.write_stl_result_run(output_root / "experiments" / "eval")
            (eval_dir / "artifact_contact_sheet.png").write_bytes(b"contact-sheet")
            (eval_dir / "provider_metrics.json").write_text('{"provider": "stub"}\n', encoding="utf-8")
            (eval_dir / "provider.log").write_text("provider ok\n", encoding="utf-8")
            for name in ("output_model.stl", "output_mesh.glb", "depth.npy", "preview.png", "notes.txt"):
                (eval_dir / name).write_bytes(b"excluded")
            results_summary = root / "results_summary.json"
            results_summary.write_text('{"run_status": 0}\n', encoding="utf-8")
            archive = root / "compact_results.tar.gz"

            metadata = build_compact_results_archive(
                output_root=output_root,
                archive_path=archive,
                extra_files=[(results_summary, "results_summary.json")],
            )
            with tarfile.open(archive, "r:gz") as tar:
                names = set(tar.getnames())
            archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
            ingest_report = summarize_stl_inputs(
                [str(archive)],
                output_dir=root / "ingested",
                top=5,
            )

        prefix = "output/compact_run/experiments/eval"
        self.assertEqual(metadata["archive_files"], len(names))
        self.assertEqual(metadata["archive_sha256"], archive_sha256)
        self.assertIn("results_summary.json", names)
        self.assertIn(f"{prefix}/aggregate_summary.csv", names)
        self.assertIn(f"{prefix}/per_sample_metrics.csv", names)
        self.assertIn(f"{prefix}/artifact_contact_sheet.png", names)
        self.assertIn(f"{prefix}/provider_metrics.json", names)
        self.assertIn(f"{prefix}/provider.log", names)
        for name in ("output_model.stl", "output_mesh.glb", "depth.npy", "preview.png", "notes.txt"):
            self.assertNotIn(f"{prefix}/{name}", names)
        self.assertEqual(len(ingest_report["runs"]), 1)
        self.assertEqual(ingest_report["runs"][0]["deployable_winner"]["method"], "hunyuan3d_shape_repaired")

    def test_stl_result_ingest_preserves_failed_provider_repair_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            metrics_path = (
                run_dir
                / "trellis_component_close"
                / "sample_a"
                / "external-image-to-mesh"
                / "provider_metrics.json"
            )
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "provider": "trellis2",
                        "provider_cache_hit": True,
                        "repair_simplification_target_faces": 100,
                        "repair_simplification_topology_preserving_faces": 160,
                        "repair_simplification_topology_relaxation_attempted": True,
                        "repair_simplification_boundary_preserving_faces": 160,
                        "repair_simplification_boundary_relaxation_attempted": True,
                        "repair_simplification_boundary_relaxed_faces": 120,
                        "repair_output_self_intersection_supported": True,
                        "repair_output_self_intersection_count": 2,
                        "error_type": "RuntimeError",
                        "error": "repair did not pass the final guard",
                    }
                ),
                encoding="utf-8",
            )

            report = summarize_stl_inputs(
                [str(run_dir)],
                output_dir=root / "ingested",
                top=5,
            )
            markdown = render_stl_ingest_markdown(report)

        failed = report["runs"][0]["failed_provider_diagnostics"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["method"], "trellis_component_close")
        self.assertEqual(failed[0]["sample_id"], "sample_a")
        self.assertEqual(failed[0]["repair_simplification_target_faces"], 100)
        self.assertEqual(failed[0]["repair_output_self_intersection_count"], 2)
        self.assertIn("Failed Provider Diagnostics", markdown)
        self.assertIn("repair did not pass the final guard", markdown)

    def test_stl_result_ingest_reports_deployable_winner_by_architecture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=5)
            markdown = render_stl_ingest_markdown(report)

        run = report["runs"][0]
        leaders = {row["stl_mode"]: row["method"] for row in run["best_by_stl_mode"]}
        decision = run["architecture_replacement_decision"]

        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["promotion_eligible_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["oracle_diagnostic_winner"]["method"], "source_mesh_oracle")
        self.assertEqual(leaders["depth-relief"], "mirror")
        self.assertEqual(leaders["single-image-mesh"], "hunyuan3d_shape_repaired")
        self.assertEqual(leaders["multiview-mesh"], "vggt_multiview_repaired")
        self.assertEqual(decision["decision"], "promote-challenger")
        self.assertEqual(decision["recommended_method"], "hunyuan3d_shape_repaired")
        self.assertEqual(decision["recommended_stl_mode"], "single-image-mesh")
        self.assertEqual(decision["depth_relief_baseline"]["method"], "mirror")
        self.assertEqual(decision["promotion_eligible_challenger"]["method"], "hunyuan3d_shape_repaired")
        self.assertGreater(decision["score_delta_vs_depth_relief"], 0)
        self.assertIn("Source-mesh oracle rows, including source-mesh bundle oracles", markdown)
        self.assertIn("Architecture Decision", markdown)
        self.assertIn("promote-challenger", markdown)
        self.assertIn("Architecture Leaders", markdown)

    def test_stl_result_ingest_treats_source_mesh_bundle_oracle_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            summary_path = run_dir / "aggregate_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            bundle_oracle = dict(next(row for row in rows if row["method"] == "source_mesh_oracle"))
            bundle_oracle.update(
                {
                    "method": "source-mesh-bundle-oracle",
                    "base_method": "external-multiview-to-mesh",
                    "stl_mode": "multiview-mesh",
                }
            )
            rows.append(bundle_oracle)
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=6)

        run = report["runs"][0]
        leaders = {row["stl_mode"]: row["method"] for row in run["best_by_stl_mode"]}
        bundle_row = next(row for row in run["ranked_methods"] if row["method"] == "source-mesh-bundle-oracle")

        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["promotion_eligible_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["oracle_diagnostic_winner"]["method"], "source_mesh_oracle")
        self.assertEqual(leaders["multiview-mesh"], "vggt_multiview_repaired")
        self.assertTrue(bundle_row["oracle_diagnostic"])
        self.assertFalse(bundle_row["promotion_eligible"])
        self.assertIn("deployable_stl_mode", bundle_row["failed_promotion_gates"])

    def test_stl_result_ingest_treats_source_bbox_calibration_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            summary_path = run_dir / "aggregate_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            for row in rows:
                row["direct_mesh_bbox_source"] = ""
                row["oracle_diagnostic"] = ""
            source_bbox_probe = dict(next(row for row in rows if row["method"] == "hunyuan3d_shape_repaired"))
            source_bbox_probe.update(
                {
                    "method": "triposr_source_bbox_probe",
                    "direct_mesh_bbox_source": "source",
                    "oracle_diagnostic": "True",
                    "mesh_surface_chamfer_l1_median": "0.001",
                    "mesh_surface_hausdorff95_median": "0.002",
                }
            )
            rows.append(source_bbox_probe)
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=6)

        run = report["runs"][0]
        probe_row = next(row for row in run["ranked_methods"] if row["method"] == "triposr_source_bbox_probe")
        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertTrue(probe_row["oracle_diagnostic"])
        self.assertFalse(probe_row["promotion_eligible"])
        self.assertIn("deployable_stl_mode", probe_row["failed_promotion_gates"])

    def test_stl_result_ingest_separates_score_leader_from_promotion_winner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            summary_path = run_dir / "aggregate_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            bad_direct = next(row for row in rows if row["method"] == "hunyuan3d_shape_repaired")
            bad_direct["mesh_surface_chamfer_l1_median"] = "0.01"
            bad_direct["mesh_surface_hausdorff95_median"] = "0.02"
            bad_direct["stl_degenerate_face_ratio_median"] = "0.01"
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=5)
            markdown = render_stl_ingest_markdown(report)

        run = report["runs"][0]
        blocked_direct = next(row for row in run["ranked_methods"] if row["method"] == "hunyuan3d_shape_repaired")
        decision = run["architecture_replacement_decision"]
        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["promotion_eligible_winner"]["method"], "vggt_multiview_repaired")
        self.assertEqual(decision["decision"], "promote-challenger")
        self.assertEqual(decision["recommended_method"], "vggt_multiview_repaired")
        self.assertEqual(decision["score_leading_challenger"]["method"], "hunyuan3d_shape_repaired")
        self.assertFalse(blocked_direct["promotion_eligible"])
        self.assertIn("Degenerate Face Ratio", blocked_direct["failed_promotion_gates"])
        self.assertIn("Promotion-eligible winner", markdown)

    def test_stl_result_ingest_keeps_depth_relief_when_all_challengers_fail_gates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            summary_path = run_dir / "aggregate_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            for row in rows:
                if row["stl_mode"] in {"single-image-mesh", "multiview-mesh"}:
                    row["mesh_surface_chamfer_l1_median"] = "0.01"
                    row["mesh_surface_hausdorff95_median"] = "0.02"
                    row["stl_degenerate_face_ratio_median"] = "0.01"
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=5)
            markdown = render_stl_ingest_markdown(report)

        run = report["runs"][0]
        decision = run["architecture_replacement_decision"]

        self.assertEqual(run["promotion_eligible_winner"]["method"], "mirror")
        self.assertEqual(decision["decision"], "keep-depth-relief")
        self.assertEqual(decision["recommended_method"], "mirror")
        self.assertEqual(decision["recommended_stl_mode"], "depth-relief")
        self.assertEqual(decision["depth_relief_baseline"]["method"], "mirror")
        self.assertEqual(decision["promotion_eligible_challenger"], {})
        self.assertEqual(decision["score_leading_challenger"]["method"], "hunyuan3d_shape_repaired")
        self.assertIn("blocked by STL promotion gates", decision["reason"])
        self.assertIn("keep-depth-relief", markdown)

    def test_stl_result_ingest_blocks_promotion_on_per_sample_stl_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            metrics_path = run_dir / "per_sample_metrics.csv"
            with metrics_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            failing = next(
                row
                for row in rows
                if row["method"] == "hunyuan3d_shape_repaired" and row["sample_id"] == "b"
            )
            failing["stl_is_watertight"] = "0.0"
            failing["stl_is_manifold"] = "0.0"
            with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=5)
            markdown = render_stl_ingest_markdown(report)

        run = report["runs"][0]
        blocked_direct = next(row for row in run["ranked_methods"] if row["method"] == "hunyuan3d_shape_repaired")
        sample_gate = next(
            row
            for row in run["gate_failures"]
            if row["method"] == "hunyuan3d_shape_repaired" and row["gate"] == "Sample Manifold"
        )
        hotspot = next(row for row in run["sample_failure_hotspots"] if row["sample_id"] == "b")
        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["promotion_eligible_winner"]["method"], "vggt_multiview_repaired")
        self.assertFalse(blocked_direct["promotion_eligible"])
        self.assertIn("Sample Manifold", blocked_direct["failed_promotion_gates"])
        self.assertEqual(sample_gate["failed_sample_count"], 1)
        self.assertEqual(sample_gate["failed_samples"], ["b"])
        self.assertEqual(hotspot["method_count"], 1)
        self.assertEqual(hotspot["gate_count"], 2)
        self.assertEqual(hotspot["methods"], ["hunyuan3d_shape_repaired"])
        self.assertEqual(hotspot["gates"], ["Sample Manifold", "Sample Watertight"])
        self.assertIn("Promotion Gate Failures", markdown)
        self.assertIn("Sample Failure Hotspots", markdown)
        self.assertIn("failed_samples=b", markdown)

    def test_stl_result_ingest_blocks_destructive_repair_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = self.write_stl_result_run(root / "run")
            summary_path = run_dir / "aggregate_summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                summary_rows = list(csv.DictReader(csv_file))
            destructive = next(row for row in summary_rows if row["method"] == "hunyuan3d_shape_repaired")
            destructive["repair_convex_hull_used_mean"] = "0.5"
            destructive["repair_volume_fill_ratio_relative_change_abs_median"] = ""
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

            metrics_path = run_dir / "per_sample_metrics.csv"
            with metrics_path.open(newline="", encoding="utf-8") as csv_file:
                sample_rows = list(csv.DictReader(csv_file))
            for row in sample_rows:
                row["repair_volume_fill_ratio_relative_change"] = ""
                if row["method"] == "hunyuan3d_shape_repaired":
                    row["repair_volume_fill_ratio_relative_change_abs"] = ""
                    row["repair_volume_fill_ratio_relative_change"] = (
                        "0.2" if row["sample_id"] == "a" else "-5.0"
                    )
            with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(sample_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sample_rows)

            report = summarize_stl_inputs([str(run_dir)], output_dir=root / "ingested", top=5)

        run = report["runs"][0]
        destructive_row = next(
            row for row in run["ranked_methods"] if row["method"] == "hunyuan3d_shape_repaired"
        )
        failed = {
            row["gate"]
            for row in run["gate_failures"]
            if row["method"] == "hunyuan3d_shape_repaired"
        }
        self.assertEqual(run["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(run["promotion_eligible_winner"]["method"], "vggt_multiview_repaired")
        self.assertFalse(destructive_row["promotion_eligible"])
        self.assertIn("Convex Hull Fallback", failed)
        self.assertIn("Sample Repair Fill-Ratio Drift Maximum", failed)
        self.assertIn("Sample Repair Fill-Ratio Drift Coverage", failed)

    def test_stl_result_ingest_extracts_colab_style_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_root = root / "archive_root"
            run_dir = self.write_stl_result_run(archive_root / "output" / "g4_run" / "combined" / "eval")
            (archive_root / "results_summary.json").write_text(
                json.dumps({"run_name": "g4_run", "combined_summaries": [{"path": str(run_dir)}]}),
                encoding="utf-8",
            )
            archive_path = root / "g4_run_results.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(archive_root / "results_summary.json", arcname="results_summary.json")
                tar.add(archive_root / "output", arcname="output")

            report = summarize_stl_inputs([f"g4={archive_path}"], output_dir=root / "ingested", top=3)

        self.assertEqual(report["run_count"], 1)
        self.assertTrue(report["inputs"][0]["extracted_to"])
        self.assertIn("ingested", report["runs"][0]["run_dir"])
        self.assertEqual(report["runs"][0]["deployable_winner"]["method"], "hunyuan3d_shape_repaired")
        self.assertEqual(report["runs"][0]["promotion_eligible_winner"]["method"], "hunyuan3d_shape_repaired")


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

    def test_paired_objective_explainer_lists_sample_losses_against_current(self):
        rows = [
            {"sample_id": "win", "method": "masked", "masked_mae": "0.60", "stl_faces_per_bbox_volume_log1p": "1.0"},
            {"sample_id": "win", "method": "mirror", "masked_mae": "0.30", "stl_faces_per_bbox_volume_log1p": "1.0"},
            {"sample_id": "win", "method": "direct", "masked_mae": "0.10", "stl_faces_per_bbox_volume_log1p": "0.5"},
            {"sample_id": "loss", "method": "masked", "masked_mae": "0.60", "stl_faces_per_bbox_volume_log1p": "1.0"},
            {"sample_id": "loss", "method": "mirror", "masked_mae": "0.20", "stl_faces_per_bbox_volume_log1p": "1.0"},
            {"sample_id": "loss", "method": "direct", "masked_mae": "0.50", "stl_faces_per_bbox_volume_log1p": "0.5"},
        ]

        sample_rows, contribution_rows = paired_sample_rows(
            rows,
            candidate_method="direct",
            baseline_method="masked",
            current_method="mirror",
            weights={"masked_mae_median": -4.0, "stl_faces_per_bbox_volume_log1p_median": -0.25},
            top_n=2,
        )

        self.assertEqual([row["sample_id"] for row in sample_rows], ["loss", "win"])
        self.assertLess(sample_rows[0]["score_vs_current"], 0)
        self.assertFalse(sample_rows[0]["win_vs_current"])
        self.assertIn("masked_mae", sample_rows[0]["top_hurts_vs_current"])
        self.assertGreater(sample_rows[1]["score_vs_current"], 0)
        self.assertTrue(sample_rows[1]["win_vs_current"])
        self.assertEqual(
            {(row["sample_id"], row["comparison_method"]) for row in contribution_rows},
            {("win", "masked"), ("win", "mirror"), ("loss", "masked"), ("loss", "mirror")},
        )

    def test_paired_objective_explainer_writes_outputs_from_run_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (root / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["sample_id", "method", "masked_mae"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
                        {"sample_id": "a", "method": "mirror", "masked_mae": "0.20"},
                        {"sample_id": "a", "method": "direct", "masked_mae": "0.10"},
                        {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
                        {"sample_id": "b", "method": "mirror", "masked_mae": "0.20"},
                        {"sample_id": "b", "method": "direct", "masked_mae": "0.30"},
                    ]
                )

            summary, sample_rows, contribution_rows = explain_paired_objective(
                root,
                candidate_method="direct",
                baseline_method="masked",
                current_method="mirror",
                score_profile="default",
                weight_overrides=["masked_mae_median=-4.0"],
            )

        self.assertEqual(summary["paired_n"], 2)
        self.assertEqual(summary["wins_vs_current"], 1)
        self.assertEqual(len(sample_rows), 2)
        self.assertEqual(len(contribution_rows), 4)

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
                "stl_faces_per_normalized_bbox_volume_log1p_median": "6.9",
            },
            {
                "method": "direct_mesh_good",
                "heldout_view_silhouette_iou_mean_median": "0.75",
                "heldout_view_silhouette_iou_min_median": "0.50",
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
                "stl_faces_per_normalized_bbox_volume_log1p_median": "3.0",
            },
        ]

        ranked, used_metrics = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        scores = {row["method"]: row["rank_score"] for row in ranked}

        self.assertIn("heldout_view_silhouette_iou_mean_median", used_metrics)
        self.assertIn("heldout_view_silhouette_iou_min_median", used_metrics)
        self.assertNotIn("mesh_surface_chamfer_l1_median", used_metrics)
        self.assertNotIn("mesh_surface_chamfer_rmse_median", used_metrics)
        self.assertNotIn("mesh_surface_hausdorff95_median", used_metrics)
        self.assertNotIn("object_surface_chamfer_l1_median", used_metrics)
        self.assertIn("stl_is_volume_median", used_metrics)
        self.assertIn("stl_winding_consistent_median", used_metrics)
        self.assertIn("stl_single_component_median", used_metrics)
        self.assertIn("stl_component_excess_log1p_median", used_metrics)
        self.assertIn("stl_bbox_has_volume_median", used_metrics)
        self.assertNotIn("stl_faces_per_normalized_bbox_volume_log1p_median", used_metrics)
        self.assertNotIn("stl_component_count_median", used_metrics)
        self.assertNotIn("stl_component_excess_median", used_metrics)
        self.assertNotIn("stl_faces_per_bbox_volume_median", used_metrics)
        self.assertNotIn("stl_faces_per_bbox_volume_log1p_median", used_metrics)
        self.assertGreater(scores["direct_mesh_good"], scores["masked"])

    def test_stl_quality_profile_derives_scale_free_complexity_for_old_summaries(self):
        weights = {"stl_faces_per_normalized_bbox_volume_log1p_median": -1.0}
        rows = [
            {
                "method": "unit_box",
                "stl_faces_median": "12",
                "stl_bbox_volume_median": "1",
                "stl_bbox_max_dimension_median": "1",
            },
            {
                "method": "scaled_box",
                "stl_faces_median": "12",
                "stl_bbox_volume_median": "1000",
                "stl_bbox_max_dimension_median": "10",
            },
        ]

        ranked, used_metrics = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="unit_box",
        )
        complexity = {
            row["method"]: float(row["stl_faces_per_normalized_bbox_volume_log1p_median"])
            for row in ranked
        }

        self.assertIn("stl_faces_per_normalized_bbox_volume_log1p_median", used_metrics)
        self.assertAlmostEqual(complexity["unit_box"], complexity["scaled_box"])
        self.assertTrue(all(abs(row["rank_score"]) < 1e-12 for row in ranked))

    def test_score_explainer_matches_baseline_delta_and_shows_metric_tradeoffs(self):
        weights = {
            "mesh_surface_chamfer_l1_median": -4.0,
            "stl_bbox_aspect_ratio_median": -0.5,
        }
        rows = [
            {
                "method": "masked",
                "mesh_surface_chamfer_l1_median": "0.20",
                "stl_bbox_aspect_ratio_median": "2.0",
            },
            {
                "method": "direct_mesh",
                "mesh_surface_chamfer_l1_median": "0.10",
                "stl_bbox_aspect_ratio_median": "8.0",
            },
        ]

        ranked, _ = rank_summary_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        contributions, used_metrics = contribution_rows(
            rows,
            weights,
            score_mode="baseline-delta",
            baseline_method="masked",
        )
        summary = summarize_contributions(contributions)
        score_by_method = {row["method"]: row["rank_score"] for row in ranked}
        explained_score_by_method = {row["method"]: row["rank_score"] for row in summary}
        direct_contributions = {
            row["metric"]: row["contribution"] for row in contributions if row["method"] == "direct_mesh"
        }

        self.assertEqual(used_metrics, ["mesh_surface_chamfer_l1_median", "stl_bbox_aspect_ratio_median"])
        self.assertAlmostEqual(explained_score_by_method["direct_mesh"], score_by_method["direct_mesh"])
        self.assertAlmostEqual(direct_contributions["mesh_surface_chamfer_l1_median"], 0.4)
        self.assertAlmostEqual(direct_contributions["stl_bbox_aspect_ratio_median"], -3.0)
        self.assertLess(explained_score_by_method["direct_mesh"], 0)

        report = markdown_report(
            summary=summary,
            contributions=contributions,
            used_metrics=used_metrics,
            score_profile="stl-quality",
            score_mode="baseline-delta",
            baseline_method="masked",
            focus_methods=["direct_mesh"],
            top_n=3,
        )

        self.assertIn("## `direct_mesh`", report)
        self.assertIn("mesh_surface_chamfer_l1_median", report)
        self.assertIn("stl_bbox_aspect_ratio_median", report)

    def test_score_explainer_accepts_utf8_sig_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.csv"
            summary_path.write_text(
                "method,masked_mae_median\nmasked,0.5\nmirror,0.2\n",
                encoding="utf-8-sig",
            )

            summary, _, used_metrics = explain(
                summary_path,
                score_profile="default",
                weight_overrides=["masked_mae_median=-4.0"],
                score_mode="baseline-delta",
                baseline_method="masked",
            )
            scores = {row["method"]: row["rank_score"] for row in summary}

        self.assertEqual(used_metrics, ["masked_mae_median"])
        self.assertAlmostEqual(scores["mirror"], 1.2)


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

    def test_selection_never_promotes_hidden_source_bbox_probe(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
            {
                "method": "triposr_source_bbox_probe",
                "success_rate": "1.0",
                "masked_mae_median": "0.05",
                "direct_mesh_bbox_source": "source",
                "oracle_diagnostic": "True",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "a", "method": "triposr_source_bbox_probe", "masked_mae": "0.05"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "triposr_source_bbox_probe", "masked_mae": "0.04"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="triposr_source_bbox_probe",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], 0)
        self.assertIn("deployable_candidate", {check["name"] for check in decision["failed_checks"]})

    def test_selection_default_skips_higher_scoring_oracle_probe(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
            {
                "method": "source_bbox_oracle",
                "success_rate": "1.0",
                "masked_mae_median": "0.01",
                "direct_mesh_bbox_source": "source",
            },
            {"method": "mirror", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "a", "method": "source_bbox_oracle", "masked_mae": "0.01"},
            {"sample_id": "a", "method": "mirror", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["candidate_method"], "mirror")
        self.assertEqual(decision["decision"], "promote")

    def test_selection_default_skips_higher_scoring_destructive_repair(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {
                "method": "hunyuan_repaired",
                "success_rate": "1.0",
                "masked_mae_median": "0.01",
                "repair_convex_hull_used_mean": "1.0",
                "repair_volume_fill_ratio_relative_change_abs_median": "0.1",
            },
            {
                "method": "triposg_repaired",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "repair_convex_hull_used_mean": "0.0",
                "repair_volume_fill_ratio_relative_change_abs_median": "0.1",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {
                "sample_id": "a",
                "method": "hunyuan_repaired",
                "masked_mae": "0.01",
                "repair_volume_fill_ratio_relative_change_abs": "0.1",
            },
            {
                "sample_id": "a",
                "method": "triposg_repaired",
                "masked_mae": "0.10",
                "repair_volume_fill_ratio_relative_change_abs": "0.1",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["candidate_method"], "triposg_repaired")
        self.assertEqual(decision["decision"], "promote")

    def test_selection_fails_closed_when_repaired_candidate_lacks_audits(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "triposg_repaired", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "triposg_repaired", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="triposg_repaired",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed = {check["name"] for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertIn("repair_convex_hull_fallback_rate", failed)
        self.assertIn("repair_volume_fill_ratio_relative_change_abs", failed)
        self.assertIn("per_sample_repair_volume_fill_ratio_relative_change_abs", failed)

    def test_selection_does_not_fallback_when_v2_fill_proxy_is_unsupported(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {
                "method": "candidate_repaired",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "repair_convex_hull_used_mean": "0.0",
                "repair_fill_ratio_supported_mean": "0.0",
                "repair_volume_fill_ratio_relative_change_abs_median": "0.1",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {
                "sample_id": "a",
                "method": "candidate_repaired",
                "masked_mae": "0.10",
                "repair_fill_ratio_supported": "False",
                "repair_volume_fill_ratio_relative_change_abs": "0.1",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="candidate_repaired",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed = {check["name"] for check in decision["failed_checks"]}
        candidate = next(
            row for row in decision["ranked_methods"] if row["method"] == "candidate_repaired"
        )
        self.assertEqual(decision["decision"], "hold")
        self.assertNotIn("repair_fill_ratio_relative_change_abs_median", candidate)
        self.assertIn("repair_volume_fill_ratio_relative_change_abs", failed)
        self.assertIn("per_sample_repair_volume_fill_ratio_relative_change_abs", failed)

    def test_enabled_surface_gates_fail_when_metrics_are_absent(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "current", "success_rate": "1.0", "masked_mae_median": "0.30"},
            {"method": "candidate", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "current", "masked_mae": "0.30"},
            {"sample_id": "a", "method": "candidate", "masked_mae": "0.10"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="candidate",
            current_method="current",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            max_mesh_surface_chamfer_ratio_vs_current=1.1,
            max_mesh_surface_hausdorff95_ratio_vs_current=1.1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed = {check["name"]: check for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        for name in (
            "paired_mesh_surface_chamfer_ratio_vs_current",
            "paired_mesh_surface_hausdorff95_ratio_vs_current",
        ):
            self.assertIn(name, failed)
            self.assertIn("metric_absent", failed[name]["detail"])

    def test_selection_rejects_reference_alias_to_source_oracle(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.50"},
            {
                "method": "reference_probe",
                "success_rate": "1.0",
                "masked_mae_median": "0.01",
                "direct_mesh_bbox_source": "reference",
                "direct_mesh_reference_method": "source_mesh_oracle",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {"sample_id": "a", "method": "reference_probe", "masked_mae": "0.01"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="reference_probe",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "hold")
        self.assertIn("deployable_candidate", {check["name"] for check in decision["failed_checks"]})

    def test_selection_uses_scale_free_complexity_gate_for_stl_candidates(self):
        stl_pass_fields = {
            "success_rate": "1.0",
            "stl_is_watertight_median": "1.0",
            "stl_is_volume_median": "1.0",
            "stl_is_manifold_median": "1.0",
            "stl_winding_consistent_median": "1.0",
            "stl_positive_volume_median": "1.0",
            "stl_single_component_median": "1.0",
            "stl_bbox_has_volume_median": "1.0",
            "stl_nonmanifold_edge_count_log1p_median": "0.0",
            "stl_degenerate_face_ratio_median": "0.0",
            "stl_component_excess_log1p_median": "0.0",
            "stl_bbox_aspect_ratio_median": "1.5",
        }
        summary_rows = [
            {
                "method": "masked",
                "masked_mae_median": "0.50",
                **stl_pass_fields,
            },
            {
                "method": "direct",
                "masked_mae_median": "0.10",
                "stl_faces_per_bbox_volume_log1p_median": "12.0",
                "stl_faces_per_normalized_bbox_volume_log1p_median": "9.0",
                **stl_pass_fields,
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.50"},
            {
                "sample_id": "a",
                "method": "direct",
                "masked_mae": "0.10",
                "stl_faces_per_bbox_volume_log1p": "12.0",
                "stl_faces_per_normalized_bbox_volume_log1p": "9.0",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="direct",
            weights={"masked_mae_median": -4.0},
            min_paired_n=1,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        self.assertEqual(decision["decision"], "promote")
        self.assertNotIn("stl_face_density", {check["name"] for check in decision["checks"]})
        self.assertTrue(
            next(check for check in decision["checks"] if check["name"] == "stl_scale_free_complexity")["passed"]
        )

    def test_selection_holds_nonprintable_stl_candidate_despite_score_win(self):
        summary_rows = [
            {
                "method": "masked",
                "success_rate": "1.0",
                "masked_mae_median": "0.60",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "1.0",
                "stl_winding_consistent_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "1.0",
                "stl_bbox_has_volume_median": "1.0",
                "stl_nonmanifold_edge_count_log1p_median": "0.0",
                "stl_degenerate_face_ratio_median": "0.0",
                "stl_component_excess_log1p_median": "0.0",
                "stl_bbox_aspect_ratio_median": "1.5",
                "stl_faces_per_bbox_volume_log1p_median": "4.0",
            },
            {
                "method": "direct_mesh",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "stl_is_watertight_median": "1.0",
                "stl_is_volume_median": "1.0",
                "stl_is_manifold_median": "0.0",
                "stl_winding_consistent_median": "1.0",
                "stl_positive_volume_median": "1.0",
                "stl_single_component_median": "0.0",
                "stl_bbox_has_volume_median": "1.0",
                "stl_nonmanifold_edge_count_log1p_median": "2.0",
                "stl_degenerate_face_ratio_median": "0.01",
                "stl_component_excess_log1p_median": "1.1",
                "stl_bbox_aspect_ratio_median": "1.5",
                "stl_faces_per_bbox_volume_log1p_median": "4.0",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "a", "method": "direct_mesh", "masked_mae": "0.10"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70"},
            {"sample_id": "b", "method": "direct_mesh", "masked_mae": "0.05"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="direct_mesh",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed_checks = {check["name"] for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], 0)
        self.assertIn("stl_manifold", failed_checks)
        self.assertIn("stl_single_component", failed_checks)
        self.assertIn("stl_nonmanifold_edges", failed_checks)
        self.assertIn("stl_degenerate_face_ratio", failed_checks)
        self.assertIn("stl_component_excess", failed_checks)

    def test_selection_holds_high_scoring_destructive_repair(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {
                "method": "direct_mesh",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "repair_convex_hull_used_mean": "0.5",
                "repair_volume_fill_ratio_relative_change_median": "-0.4",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70"},
            {
                "sample_id": "a",
                "method": "direct_mesh",
                "masked_mae": "0.10",
                "repair_volume_fill_ratio_relative_change": "0.2",
            },
            {
                "sample_id": "b",
                "method": "direct_mesh",
                "masked_mae": "0.05",
                "repair_volume_fill_ratio_relative_change": "-5.0",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="direct_mesh",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed_checks = {check["name"]: check for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], 0)
        self.assertIn("repair_convex_hull_fallback_rate", failed_checks)
        self.assertAlmostEqual(
            failed_checks["repair_volume_fill_ratio_relative_change_abs"]["value"],
            2.6,
        )
        self.assertIn("per_sample_repair_volume_fill_ratio_relative_change_abs", failed_checks)
        self.assertIn("repair_volume_fill_ratio_relative_change_abs_coverage", failed_checks)
        self.assertIn(
            "failed_samples=b",
            failed_checks["per_sample_repair_volume_fill_ratio_relative_change_abs"]["detail"],
        )

    def test_selection_holds_candidate_with_per_sample_stl_failure(self):
        summary_rows = [
            {
                "method": "masked",
                "success_rate": "1.0",
                "masked_mae_median": "0.60",
                "stl_is_manifold_median": "1.0",
            },
            {
                "method": "direct_mesh",
                "success_rate": "1.0",
                "masked_mae_median": "0.10",
                "stl_is_manifold_median": "1.0",
            },
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60", "stl_is_manifold": "1.0"},
            {"sample_id": "a", "method": "direct_mesh", "masked_mae": "0.10", "stl_is_manifold": "1.0"},
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70", "stl_is_manifold": "1.0"},
            {"sample_id": "b", "method": "direct_mesh", "masked_mae": "0.05", "stl_is_manifold": "0.0"},
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="direct_mesh",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed_checks = {check["name"]: check for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], 0)
        self.assertIn("per_sample_stl_manifold", failed_checks)
        self.assertIn("failed_samples=b", failed_checks["per_sample_stl_manifold"]["detail"])

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

    def test_selection_holds_heldout_view_regression_vs_current(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "current", "success_rate": "1.0", "masked_mae_median": "0.30"},
            {"method": "candidate", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {
                "sample_id": "a",
                "method": "current",
                "masked_mae": "0.30",
                "heldout_view_silhouette_iou_mean": "0.80",
                "mesh_surface_chamfer_l1": "0.10",
                "mesh_surface_hausdorff95": "0.20",
            },
            {
                "sample_id": "a",
                "method": "candidate",
                "masked_mae": "0.10",
                "heldout_view_silhouette_iou_mean": "0.60",
                "mesh_surface_chamfer_l1": "0.15",
                "mesh_surface_hausdorff95": "0.28",
            },
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70"},
            {
                "sample_id": "b",
                "method": "current",
                "masked_mae": "0.40",
                "heldout_view_silhouette_iou_mean": "0.70",
                "mesh_surface_chamfer_l1": "0.20",
                "mesh_surface_hausdorff95": "0.30",
            },
            {
                "sample_id": "b",
                "method": "candidate",
                "masked_mae": "0.20",
                "heldout_view_silhouette_iou_mean": "0.50",
                "mesh_surface_chamfer_l1": "0.21",
                "mesh_surface_hausdorff95": "0.32",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="candidate",
            current_method="current",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        failed_checks = {check["name"]: check for check in decision["failed_checks"]}
        self.assertEqual(decision["decision"], "hold")
        self.assertGreater(decision["candidate_rank_score"], decision["current_rank_score"])
        self.assertAlmostEqual(
            failed_checks["paired_heldout_view_silhouette_iou_mean_ratio_vs_current"]["value"],
            1.4,
        )
        self.assertNotIn("paired_mesh_surface_chamfer_ratio_vs_current", failed_checks)
        self.assertNotIn("paired_mesh_surface_hausdorff95_ratio_vs_current", failed_checks)

    def test_selection_promotes_when_heldout_view_agreement_stays_within_ratio(self):
        summary_rows = [
            {"method": "masked", "success_rate": "1.0", "masked_mae_median": "0.60"},
            {"method": "current", "success_rate": "1.0", "masked_mae_median": "0.30"},
            {"method": "candidate", "success_rate": "1.0", "masked_mae_median": "0.10"},
        ]
        per_sample_rows = [
            {"sample_id": "a", "method": "masked", "masked_mae": "0.60"},
            {
                "sample_id": "a",
                "method": "current",
                "masked_mae": "0.30",
                "heldout_view_silhouette_iou_mean": "0.80",
                "mesh_surface_chamfer_l1": "0.10",
                "mesh_surface_hausdorff95": "0.20",
            },
            {
                "sample_id": "a",
                "method": "candidate",
                "masked_mae": "0.10",
                "heldout_view_silhouette_iou_mean": "0.78",
                "mesh_surface_chamfer_l1": "0.105",
                "mesh_surface_hausdorff95": "0.21",
            },
            {"sample_id": "b", "method": "masked", "masked_mae": "0.70"},
            {
                "sample_id": "b",
                "method": "current",
                "masked_mae": "0.40",
                "heldout_view_silhouette_iou_mean": "0.70",
                "mesh_surface_chamfer_l1": "0.20",
                "mesh_surface_hausdorff95": "0.30",
            },
            {
                "sample_id": "b",
                "method": "candidate",
                "masked_mae": "0.20",
                "heldout_view_silhouette_iou_mean": "0.68",
                "mesh_surface_chamfer_l1": "0.21",
                "mesh_surface_hausdorff95": "0.315",
            },
        ]

        decision = evaluate_selection(
            summary_rows,
            per_sample_rows,
            baseline_method="masked",
            candidate_method="candidate",
            current_method="current",
            weights={"masked_mae_median": -4.0},
            min_paired_n=2,
            require_split_audit=False,
            bootstrap_samples=0,
        )

        checks = {check["name"]: check for check in decision["checks"]}
        self.assertEqual(decision["decision"], "promote")
        self.assertTrue(checks["paired_heldout_view_silhouette_iou_mean_ratio_vs_current"]["passed"])
        self.assertNotIn("paired_mesh_surface_chamfer_ratio_vs_current", checks)
        self.assertNotIn("paired_mesh_surface_hausdorff95_ratio_vs_current", checks)

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
