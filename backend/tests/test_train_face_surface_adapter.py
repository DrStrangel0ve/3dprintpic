import numpy as np
import torch

from backend.benchmark.train_face_surface_adapter import (
    _candidate_full_surface,
    _face_feather,
    build_surface_adapter,
    positive_affine_fit,
    remove_affine_residual,
    surface_training_loss,
)


def test_surface_adapter_is_initially_an_exact_noop():
    model = build_surface_adapter(7)
    values = torch.randn(2, 7, 32, 32)

    residual = model(values)

    assert residual.shape == (2, 1, 32, 32)
    assert torch.count_nonzero(residual) == 0


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
