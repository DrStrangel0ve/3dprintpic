import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from backend.benchmark.train_face_surface_adapter import (
    CachedSurface,
    _candidate_full_surface,
    _face_feather,
    _failed_part_names,
    _infer_variable_crop_depths,
    _per_row_non_regression,
    _strictly_improves,
    build_surface_adapter,
    evaluate_exact_surfaces,
    positive_affine_fit,
    remove_affine_residual,
    surface_training_loss,
)
from backend.face_depth_refinement import FACE_PART_NAMES


def test_surface_adapter_is_initially_an_exact_noop():
    model = build_surface_adapter(7)
    values = torch.randn(2, 7, 32, 32)

    residual = model(values)

    assert residual.shape == (2, 1, 32, 32)
    assert torch.count_nonzero(residual) == 0


def test_unavailable_or_malformed_part_metrics_fail_all_parts_closed():
    expected = list(FACE_PART_NAMES)

    assert _failed_part_names(None) == expected
    assert _failed_part_names({"available": False, "failed_parts": []}) == expected
    assert _failed_part_names({"available": True}) == expected
    assert _failed_part_names(
        {"available": True, "passed": False, "failed_parts": []}
    ) == expected
    assert _failed_part_names(
        {"available": True, "passed": False, "failed_parts": ["nose", "mouth"]}
    ) == [name for name in FACE_PART_NAMES if name in {"nose", "mouth"}]


def test_per_row_non_regression_rejects_unpaired_or_duplicate_rows():
    candidate = {"rows": [{"row_id": "a", "combined_part_failures": 0}]}
    baseline = {"rows": [{"row_id": "b", "combined_part_failures": 0}]}

    with np.testing.assert_raises_regex(ValueError, "identical unique rows"):
        _per_row_non_regression(candidate, baseline)


def test_exact_surface_evaluation_records_unavailable_metric_as_hold():
    item = CachedSurface(
        row={
            "row_id": "unavailable",
            "identity_group": "synthetic",
            "expression": "neutral",
            "render": {"face_bbox_height_pixels": 75},
            "exact_depth": {"path": "exact.npy"},
            "selection_mask": {"path": "face.png"},
            "exact_face_parts": {
                name: {"path": f"{name}.png"} for name in FACE_PART_NAMES
            },
        },
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        baseline=np.zeros((8, 8), dtype=np.float32),
        target=np.zeros((8, 8), dtype=np.float32),
        face=np.ones((8, 8), dtype=bool),
        feather=np.ones((8, 8), dtype=np.float32),
        parts=np.ones((len(FACE_PART_NAMES), 8, 8), dtype=bool),
        bbox=(0, 0, 8, 8),
        source_shape=(8, 8),
    )
    with tempfile.TemporaryDirectory() as temporary, mock.patch(
        "backend.benchmark.train_face_surface_adapter._exact_face_depth_quality",
        return_value={
            "available": False,
            "reason": "insufficient_candidate_samples",
        },
    ):
        summary = evaluate_exact_surfaces(
            Path(temporary),
            [item],
            [np.zeros((8, 8), dtype=np.float32)],
            alpha=0.0,
            output_dir=Path(temporary) / "output",
        )

    assert summary["unavailable_row_count"] == 1
    assert summary["global_check_failure_counts"] == {"metric_unavailable": 1}
    assert summary["combined_part_failures"] == 2 * len(FACE_PART_NAMES)


def test_strict_selector_allows_paired_global_failures_but_no_regression():
    baseline = {
        "unavailable_row_count": 0,
        "global_failure_row_count": 2,
        "global_check_failure_counts": {"normalized_rmse": 2},
        "combined_part_failures": 10,
        "median_shape_correlation": 0.9,
        "median_gradient_correlation": 0.8,
        "median_normalized_rmse": 0.1,
        "shape_check_failure_counts": {},
    }
    candidate = {
        **baseline,
        "combined_part_failures": 9,
        "median_shape_correlation": 0.91,
        "median_gradient_correlation": 0.81,
        "median_normalized_rmse": 0.09,
    }

    assert _strictly_improves(candidate, baseline)
    candidate["global_check_failure_counts"] = {"normalized_rmse": 3}
    candidate["global_failure_row_count"] = 3
    assert not _strictly_improves(candidate, baseline)
    candidate["global_check_failure_counts"] = {"normalized_rmse": 2}
    candidate["global_failure_row_count"] = 2
    candidate["unavailable_row_count"] = 1
    assert not _strictly_improves(candidate, baseline)


def test_positive_affine_fit_recovers_target_shape():
    prediction = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
    target = prediction * 3.5 - 0.7
    mask = torch.ones_like(prediction)

    fitted, scale, shift = positive_affine_fit(prediction, target, mask)

    assert torch.allclose(fitted, target, atol=1e-5)
    assert torch.allclose(scale, torch.tensor([[[[3.5]]]]), atol=1e-5)
    assert torch.allclose(shift, torch.tensor([[[[-0.7]]]]), atol=1e-5)


def test_surface_training_loss_is_finite_and_backpropagates():
    baseline = torch.linspace(0.0, 1.0, 32 * 32).reshape(1, 1, 32, 32)
    target = baseline + 0.05 * torch.sin(baseline * 20.0)
    face = torch.zeros_like(baseline)
    face[..., 4:28, 5:27] = 1.0
    feather = face.clone()
    parts = torch.zeros_like(face)
    parts[..., 10:22, 10:22] = 1.0
    residual = torch.zeros_like(baseline, requires_grad=True)

    loss, details = surface_training_loss(
        residual,
        baseline,
        target,
        face,
        feather,
        parts,
        torch.ones((1, 1, 1, 1)),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert residual.grad is not None
    assert torch.all(torch.isfinite(residual.grad))
    assert set(details) == {
        "value",
        "gradient",
        "multiscale",
        "normal",
        "residual_regularization",
        "fit_scale_median",
        "fit_shift_median",
    }


def test_face_feather_is_zero_at_boundary_and_positive_inside():
    face = np.zeros((32, 32), dtype=bool)
    face[4:28, 5:27] = True

    feather = _face_feather(face)

    assert feather.shape == face.shape
    assert feather[4, 10] == 0.0
    assert feather[16, 16] > 0.9
    assert np.all(feather[~face] == 0.0)


def test_remove_affine_residual_keeps_only_non_affine_shape():
    yy, xx = np.indices((24, 24), dtype=np.float32)
    baseline = 0.2 + xx * 0.01 + yy * 0.002
    bump = np.exp(-((xx - 12.0) ** 2 + (yy - 13.0) ** 2) / 18.0)
    residual = 0.4 * baseline - 0.3 + bump
    face = np.zeros((24, 24), dtype=bool)
    face[2:22, 3:21] = True

    cleaned, stats = remove_affine_residual(residual, baseline, face)
    design = np.column_stack((baseline[face], np.ones(np.count_nonzero(face))))
    fitted_scale, fitted_offset = np.linalg.lstsq(
        design,
        cleaned[face],
        rcond=None,
    )[0]

    assert stats["method"] == "remove-affine-component-relative-to-local-depth"
    assert abs(float(fitted_scale)) < 1e-5
    assert abs(float(fitted_offset)) < 1e-5
    assert float(np.max(cleaned) - np.min(cleaned)) > 0.5


def test_candidate_surface_maps_network_crop_into_source_grid():
    class Item:
        baseline = np.full((8, 8), 0.25, dtype=np.float32)
        feather = np.ones((8, 8), dtype=np.float32)
        bbox = (3, 4, 11, 12)
        source_shape = (16, 18)

    candidate = _candidate_full_surface(
        Item(),
        np.full((8, 8), 0.2, dtype=np.float32),
        0.5,
    )

    assert candidate.shape == (16, 18)
    assert np.allclose(candidate[4:12, 3:11], 0.35)
    outside = candidate.copy()
    outside[4:12, 3:11] = 0.0
    assert np.count_nonzero(outside) == 0


def test_variable_crop_inference_groups_processor_shapes_and_restores_order():
    class Processor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            width, height = images.size
            output_width = 12 if width > height else 8
            return {
                "pixel_values": torch.full(
                    (1, 3, 8, output_width),
                    float(width),
                )
            }

    class Model:
        def __init__(self):
            self.batch_shapes = []

        def __call__(self, *, pixel_values):
            self.batch_shapes.append(tuple(pixel_values.shape))
            return type(
                "DepthOutput",
                (),
                {"predicted_depth": pixel_values.mean(dim=1)},
            )()

    images = [
        Image.new("RGB", (16, 16)),
        Image.new("RGB", (24, 16)),
        Image.new("RGB", (20, 20)),
    ]
    model = Model()

    predictions = _infer_variable_crop_depths(
        Processor(),
        model,
        images,
        device="cpu",
        dtype=torch.float32,
    )

    assert model.batch_shapes == [(2, 3, 8, 8), (1, 3, 8, 12)]
    assert [tuple(values.shape) for values in predictions] == [
        (8, 8),
        (8, 12),
        (8, 8),
    ]
    assert torch.all(predictions[0] == 16.0)
    assert torch.all(predictions[1] == 24.0)
    assert torch.all(predictions[2] == 20.0)
