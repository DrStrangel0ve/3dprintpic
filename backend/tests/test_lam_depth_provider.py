from pathlib import Path

import numpy as np
import pytest

from backend.benchmark import lam_depth_provider as provider


def test_source_to_render_maps_preserve_half_pixel_resize_contract():
    map_x, map_y = provider.source_to_render_maps(
        (4, 8),
        [0, 0, 8, 4],
        [0, 0, 1024, 1024],
    )

    assert map_x.shape == (4, 8)
    assert map_y.shape == (4, 8)
    assert map_x[0, 0] == pytest.approx(31.5)
    assert map_x[0, -1] == pytest.approx(479.5)
    assert map_y[0, 0] == pytest.approx(63.5)
    assert map_y[-1, 0] == pytest.approx(447.5)


def test_warp_lam_depth_to_source_is_identity_for_matching_affine():
    y, x = np.mgrid[0:provider.LAM_RENDER_SIZE, 0:provider.LAM_RENDER_SIZE]
    depth = (1.0 + x * 0.001 + y * 0.002).astype(np.float32)
    mask = np.ones_like(depth)

    warped, warped_mask = provider.warp_lam_depth_to_source(
        depth,
        mask,
        (provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE),
        [0, 0, provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE],
        [0, 0, provider.LAM_TRACKER_SIZE, provider.LAM_TRACKER_SIZE],
    )

    np.testing.assert_allclose(warped, depth, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(warped_mask, mask)


def test_warp_lam_depth_warps_weighted_numerator_before_division():
    y, x = np.mgrid[0:provider.LAM_RENDER_SIZE, 0:provider.LAM_RENDER_SIZE]
    expected = (0.7 + x * 0.0001 + y * 0.0002).astype(np.float32)
    mask = (0.1 + x / (provider.LAM_RENDER_SIZE * 2)).astype(np.float32)
    mask = np.broadcast_to(mask, expected.shape).copy()
    weighted = expected * mask

    warped, warped_mask = provider.warp_lam_depth_to_source(
        weighted,
        mask,
        (provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE),
        [0, 0, provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE],
        [0, 0, provider.LAM_TRACKER_SIZE, provider.LAM_TRACKER_SIZE],
    )

    np.testing.assert_allclose(warped, expected, rtol=1e-6)
    np.testing.assert_array_equal(warped_mask, mask)


def test_warp_lam_depth_to_source_rejects_wrong_render_shape():
    with pytest.raises(ValueError, match="Unexpected LAM depth shape"):
        provider.warp_lam_depth_to_source(
            np.ones((32, 32), dtype=np.float32),
            np.ones((32, 32), dtype=np.float32),
            (32, 32),
            [0, 0, 32, 32],
            [0, 0, 1024, 1024],
        )


def test_runtime_environment_is_scoped_to_bundle(tmp_path: Path):
    environment = provider._runtime_environment(tmp_path)
    research_root = tmp_path.parents[2]

    assert environment["HF_HOME"] == str(tmp_path / "models")
    assert environment["TORCH_HOME"] == str(tmp_path / "models")
    assert environment["MODELSCOPE_CACHE"] == str(tmp_path)
    assert environment["XFORMERS_FORCE_DISABLE_TRITON"] == "1"
    assert str(tmp_path / ".glut" / "Scripts") in environment["PATH"]
    assert environment["CUDA_HOME"] == str(research_root / "CUDA128/toolkit")
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        research_root / "LAM/torch_extensions_cuda128"
    )
    assert environment["PYTHONPATH"] == str(
        research_root / "LAM/diff-gaussian-rasterization"
    )
    assert environment["PYTHONHOME"] == ""


def test_validate_lam_camera_depth_accepts_positive_varying_depth():
    y, x = np.mgrid[0:provider.LAM_RENDER_SIZE, 0:provider.LAM_RENDER_SIZE]
    render_depth = (1.0 + x * 0.001 + y * 0.002).astype(np.float32)
    render_mask = np.ones_like(render_depth)
    source_depth = render_depth[::2, ::2]
    source_mask = np.ones_like(source_depth)
    selection = np.ones_like(source_depth)

    result = provider.validate_lam_camera_depth(
        render_depth,
        render_mask,
        source_depth,
        source_mask,
        selection_mask=selection,
    )

    assert result["render_valid_pixels"] == provider.LAM_RENDER_SIZE**2
    assert result["source_valid_pixels"] == source_depth.size
    assert result["selection_overlap_ratio"] == pytest.approx(1.0)


def test_normalize_lam_comp_depth_recovers_expected_camera_depth():
    y, x = np.mgrid[0:provider.LAM_RENDER_SIZE, 0:provider.LAM_RENDER_SIZE]
    expected = (0.7 + x * 0.0001 + y * 0.0002).astype(np.float32)
    mask = np.full_like(expected, 0.5)
    mask[:8] = 0.005
    weighted = expected * mask

    normalized = provider.normalize_lam_comp_depth(weighted, mask)

    assert np.all(np.isnan(normalized[:8]))
    np.testing.assert_allclose(normalized[8:], expected[8:], rtol=1e-6)


def test_normalize_lam_comp_depth_rejects_invalid_mask():
    shape = (provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE)
    weighted = np.ones(shape, dtype=np.float32)
    invalid_mask = np.ones(shape, dtype=np.float32)
    invalid_mask[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        provider.normalize_lam_comp_depth(weighted, invalid_mask)

    with pytest.raises(ValueError, match="shapes differ"):
        provider.normalize_lam_comp_depth(
            weighted,
            np.ones((64, 64), dtype=np.float32),
        )


def test_validate_lam_camera_depth_rejects_empty_depth():
    render_depth = np.zeros(
        (provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE),
        dtype=np.float32,
    )
    render_mask = np.ones_like(render_depth)

    with pytest.raises(ValueError, match="too few valid pixels"):
        provider.validate_lam_camera_depth(
            render_depth,
            render_mask,
            np.zeros((64, 64), dtype=np.float32),
            np.ones((64, 64), dtype=np.float32),
        )


def test_validate_lam_camera_depth_rejects_mask_outside_unit_interval():
    render_depth = np.ones(
        (provider.LAM_RENDER_SIZE, provider.LAM_RENDER_SIZE),
        dtype=np.float32,
    )
    render_mask = np.full_like(render_depth, 1.1)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        provider.validate_lam_camera_depth(
            render_depth,
            render_mask,
            np.ones((64, 64), dtype=np.float32),
            np.ones((64, 64), dtype=np.float32),
        )


def test_write_json_rejects_nonfinite_payload(tmp_path: Path):
    with pytest.raises(ValueError, match="Out of range float values"):
        provider._write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_eager_torch_compile_uses_documented_disable_flag():
    class FakeTorch:
        def __init__(self):
            self.calls = []

        def compile(self, model=None, **kwargs):
            self.calls.append(kwargs)
            if model is None:
                return lambda function: function
            return model

    fake_torch = FakeTorch()
    record = provider._install_eager_torch_compile(fake_torch)

    @fake_torch.compile
    def identity(value):
        return value

    assert identity(3) == 3
    assert fake_torch.calls == [{"disable": True}]
    assert record["method"] == "torch.compile-disable-true"
    assert record["source_modified"] is False


def test_lam_model_and_bundle_are_immutable_and_research_only():
    assert provider.LAM_MODEL_SIZE_BYTES == 2_356_556_212
    assert len(provider.LAM_MODEL_SHA256) == 64
    assert provider.LAM_WINDOWS_BUNDLE_SIZE_BYTES == 8_993_781_762
    assert len(provider.LAM_WINDOWS_BUNDLE_SHA256) == 64
    assert provider.LAM_SOURCE_REVISION != provider.LAM_WINDOWS_SOURCE_REVISION
    assert len(provider.LAM_ASSET_SPECS) == 5
    assert set(provider.LAM_ASSET_PROVENANCE) == set(provider.LAM_ASSET_SPECS)
    assert all(
        record["production_eligible"] is False
        for record in provider.LAM_ASSET_PROVENANCE.values()
    )
    assert len(provider.LAM_UNTRACKED_RUNTIME_SPECS) == 4
    assert len(provider.LAM_RUNTIME_VERSION_SPECS) == 10
    assert len(provider.LAM_RUNTIME_BINARY_SPECS) == 10
    assert len(provider.CUDA_REDISTRIBUTABLE_SPECS) == 6
    assert len(provider.DIFF_GAUSSIAN_SOURCE_REVISION) == 40
    assert len(provider.DIFF_GAUSSIAN_EXTENSION_SHA256) == 64
