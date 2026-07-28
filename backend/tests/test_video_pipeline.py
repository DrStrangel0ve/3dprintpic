import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.run_image_to_mesh_provider import run_provider
from backend.stl_diagnostics import stl_diagnostics
from backend.video_pipeline import (
    SAM2_VIDEO_REVISIONS,
    VideoDecodeError,
    VideoPipelineError,
    evaluate_mask_sequence,
    prepare_turntable_video,
    probe_video,
    segment_turntable_frame,
    video_model_preflight,
)


def write_turntable_fixture(path: Path, *, frame_count: int = 24, size: int = 96) -> list[np.ndarray]:
    import cv2
    import trimesh

    mesh = trimesh.creation.box(extents=(1.1, 0.8, 0.55))
    config = RenderConfig(size=size, background_rgb=(0.91, 0.93, 0.96))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (size, size))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG writer is unavailable")
    silhouettes = []
    try:
        for index in range(frame_count):
            camera = CameraSpec(azimuth_deg=index * 360.0 / frame_count, elevation_deg=8.0)
            rendered = render_mesh(mesh, camera=camera, config=config, base_color=(80, 145, 205))
            rgb = np.clip(rendered.rgb * 255.0, 0, 255).astype(np.uint8)
            writer.write(rgb[:, :, ::-1])
            silhouettes.append(rendered.silhouette.astype(bool))
    finally:
        writer.release()
    return silhouettes


class VideoPipelineTest(unittest.TestCase):
    def test_prepare_turntable_video_decodes_segments_and_writes_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "turntable.avi"
            ground_truth = write_turntable_fixture(video_path)

            prepared = prepare_turntable_video(
                video_path,
                root / "prepared",
                selected_frame_count=8,
                frame_selection="uniform-frame-sampler",
                segmentation_provider="turntable-grabcut",
                object_point=(0.5, 0.5),
                frame_max_side=128,
            )

            self.assertTrue(prepared.bundle_path.exists())
            self.assertTrue(prepared.report_path.exists())
            self.assertEqual(len(prepared.frame_paths), 8)
            self.assertEqual(len(prepared.mask_paths), 8)
            self.assertTrue(prepared.report["segmentation"]["quality"]["passes_hard_checks"])
            source_indices = prepared.report["sampling"]["selected_source_indices"]
            self.assertEqual(source_indices, sorted(set(source_indices)))
            self.assertGreater(source_indices[-1] - source_indices[0], 15)

            ious = []
            for source_index, mask_path in zip(source_indices, prepared.mask_paths, strict=True):
                predicted = np.asarray(Image.open(mask_path).convert("L")) > 127
                expected = ground_truth[source_index]
                intersection = np.count_nonzero(predicted & expected)
                union = np.count_nonzero(predicted | expected)
                ious.append(intersection / float(max(1, union)))
            self.assertGreater(float(np.median(ious)), 0.95)
            self.assertGreater(float(np.min(ious)), 0.90)

            bundle = json.loads(prepared.bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["source_kind"], "turntable-video")
            self.assertEqual(len(bundle["views"]), 8)
            self.assertTrue(all(Path(view["image"]).exists() for view in bundle["views"]))
            self.assertTrue(all(Path(view["mask"]).exists() for view in bundle["views"]))
            azimuths = [view["camera"]["azimuth_deg"] for view in bundle["views"]]
            self.assertGreater(max(azimuths) - min(azimuths), 240.0)

    def test_prepared_turntable_bundle_emits_printable_visual_hull_stl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "turntable.avi"
            write_turntable_fixture(video_path, frame_count=20)
            prepared = prepare_turntable_video(
                video_path,
                root / "prepared",
                selected_frame_count=8,
                segmentation_provider="turntable-grabcut",
                frame_max_side=128,
            )
            output_mesh = root / "output_mesh.ply"
            output_stl = root / "output_model.stl"
            args = SimpleNamespace(
                provider="multiview-visual-hull",
                input_image=prepared.frame_paths[0],
                input_bundle=prepared.bundle_path,
                output_mesh=output_mesh,
                output_stl=output_stl,
                raw_output_mesh=root / "output_mesh_raw.ply",
                mesh_repair="printable",
                mesh_target_max_dimension=96.0,
                mesh_min_bbox_dimension=12.0,
                mesh_max_bbox_aspect_ratio=0.0,
                mesh_target_bbox_extents=None,
                mesh_target_faces=40000,
                mesh_max_normalized_face_density_log1p=0.0,
                visual_hull_resolution=24,
                visual_hull_grid_extent=1.9,
                visual_hull_ortho_scale=2.0,
                visual_hull_mask_dilate=1,
            )

            mesh_path, stl_path = run_provider(args)
            diagnostics = stl_diagnostics(stl_path)

            self.assertTrue(mesh_path.exists())
            self.assertTrue(stl_path.exists())
            self.assertTrue(diagnostics["stl_is_watertight"])
            self.assertTrue(diagnostics["stl_is_volume"])
            self.assertTrue(diagnostics["stl_positive_volume"])
            self.assertTrue(diagnostics["stl_single_component"])

    def test_mask_quality_rejects_empty_full_and_identity_jump(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        full = np.ones((64, 64), dtype=np.uint8)
        left = np.zeros((64, 64), dtype=np.uint8)
        left[20:44, 2:18] = 1
        right = np.zeros((64, 64), dtype=np.uint8)
        right[20:44, 46:62] = 1

        report = evaluate_mask_sequence([empty, full, left, right])

        self.assertFalse(report["passes_hard_checks"])
        joined = " ".join(report["failed_checks"])
        self.assertIn("mask_not_empty", joined)
        self.assertIn("mask_not_full_frame", joined)
        self.assertIn("mask_centroid_stability", joined)
        self.assertIn("mask_temporal_overlap", joined)

    def test_turntable_temporal_prior_does_not_grow_on_identical_frames(self):
        import cv2

        frame = np.full((256, 256, 3), 238, dtype=np.uint8)
        cv2.rectangle(frame, (68, 82), (187, 173), (170, 105, 45), thickness=-1)
        masks = []
        previous = None
        for _ in range(4):
            previous = segment_turntable_frame(frame, point=(0.5, 0.5), previous_mask=previous)
            masks.append(previous)

        self.assertTrue(all(np.array_equal(masks[0], mask) for mask in masks[1:]))

    def test_corrupt_video_is_rejected_before_model_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.mp4"
            path.write_bytes(b"not a video")

            with self.assertRaisesRegex(VideoDecodeError, "could not be opened"):
                probe_video(path)

    def test_invalid_frame_selection_and_object_point_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "turntable.avi"
            write_turntable_fixture(video_path, frame_count=8)

            with self.assertRaisesRegex(VideoPipelineError, "Unsupported frame selection"):
                prepare_turntable_video(video_path, root / "bad-mode", frame_selection="every-pretty-frame")
            with self.assertRaisesRegex(VideoPipelineError, "normalized values"):
                prepare_turntable_video(video_path, root / "bad-point", object_point=(1.2, 0.5))

    def test_model_preflight_never_claims_unattached_vggt_stl_runner(self):
        preflight = video_model_preflight()
        geometry = {row["id"]: row for row in preflight["geometry"]}
        segmentation = {row["id"]: row for row in preflight["segmentation"]}

        self.assertTrue(geometry["multiview-visual-hull"]["runnable"])
        self.assertFalse(geometry["vggt-omega"]["runnable"])
        self.assertFalse(geometry["vggt-omega"]["checks"]["stl_extractor_attached"])
        self.assertIn("sam3.1-video", segmentation)
        self.assertEqual(
            segmentation["sam2.1-hiera-tiny-video"]["checks"]["model_revision"],
            SAM2_VIDEO_REVISIONS["sam2.1-hiera-tiny-video"],
        )


if __name__ == "__main__":
    unittest.main()
