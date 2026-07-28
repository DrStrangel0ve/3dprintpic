import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.video_pipeline import PreparedVideo
from backend.video_subject_relief import analyze_subject_motion, select_best_subject_view


def write_mask(path: Path, bounds: tuple[int, int, int, int], size: int = 100) -> None:
    mask = np.zeros((size, size), dtype=np.uint8)
    x0, y0, x1, y1 = bounds
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(path)


def write_frame(path: Path, size: int = 100) -> None:
    rows, cols = np.indices((size, size))
    pattern = ((rows // 3 + cols // 3) % 2) * 180 + 40
    rgb = np.stack((pattern, np.roll(pattern, 1, axis=0), np.roll(pattern, 1, axis=1)), axis=-1)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


class VideoSubjectReliefTest(unittest.TestCase):
    def test_motion_profile_recognizes_monotonic_approach(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index, bounds in enumerate(
                ((44, 38, 56, 58), (39, 28, 61, 68), (31, 14, 69, 86))
            ):
                path = root / f"mask_{index}.png"
                write_mask(path, bounds)
                paths.append(path)

            report = analyze_subject_motion(tuple(paths), [0, 10, 20])

        self.assertEqual(report["classification"], "approach")
        self.assertGreater(report["start_end_coverage_ratio"], 5.0)
        self.assertEqual(report["increasing_step_fraction"], 1.0)
        self.assertLess(report["max_centroid_jump"], 0.05)

    def test_best_view_prefers_large_fully_visible_subject_over_clipped_last_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_paths = []
            mask_paths = []
            bounds = (
                (43, 35, 57, 60),
                (35, 20, 65, 80),
                (30, 20, 70, 100),
            )
            for index, box in enumerate(bounds):
                frame_path = root / f"frame_{index}.png"
                mask_path = root / f"mask_{index}.png"
                write_frame(frame_path)
                write_mask(mask_path, box)
                frame_paths.append(frame_path)
                mask_paths.append(mask_path)
            prepared = PreparedVideo(
                bundle_path=root / "bundle.json",
                report_path=root / "report.json",
                frame_paths=tuple(frame_paths),
                mask_paths=tuple(mask_paths),
                bundle={},
                report={"sampling": {"selected_source_indices": [0, 10, 20]}},
            )
            motion = analyze_subject_motion(tuple(mask_paths), [0, 10, 20])

            selection = select_best_subject_view(prepared, motion)

        self.assertEqual(selection["selected"]["position"], 1)
        self.assertGreater(
            selection["candidates"][1]["score_components"]["full_visibility"],
            selection["candidates"][2]["score_components"]["full_visibility"],
        )


if __name__ == "__main__":
    unittest.main()
