"""Core contracts for a pixel-space face-depth rectified-flow challenger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence


STATIC_CONDITIONING_CHANNELS = 10
MODEL_INPUT_CHANNELS = 1 + STATIC_CONDITIONING_CHANNELS
MODEL_OUTPUT_CHANNELS = 1
FACE_SUPPORT_STATIC_CHANNEL = 5


@dataclass(frozen=True)
class RectifiedFlowConfig:
    sample_size: int = 96
    block_out_channels: tuple[int, ...] = (32, 64, 96)
    layers_per_block: int = 1
    norm_num_groups: int = 8
    attention_head_dim: int = 8
    noise_sigma: float = 0.15
    inference_steps: int = 8
    inference_seeds: tuple[int, ...] = (20260719, 20260720, 20260721, 20260722)

    def validated(self) -> "RectifiedFlowConfig":
        if not 32 <= int(self.sample_size) <= 256:
            raise ValueError("Rectified-flow sample size must be in [32, 256]")
        if not 2 <= len(self.block_out_channels) <= 5:
            raise ValueError("Rectified-flow U-Net must contain two to five levels")
        divisor = 2 ** (len(self.block_out_channels) - 1)
        if int(self.sample_size) % divisor:
            raise ValueError("Rectified-flow sample size must fit the U-Net scale")
        if not 1 <= int(self.layers_per_block) <= 3:
            raise ValueError("Rectified-flow layers per block must be in [1, 3]")
        if int(self.norm_num_groups) <= 0 or any(
            int(channels) % int(self.norm_num_groups)
            for channels in self.block_out_channels
        ):
            raise ValueError("Every U-Net channel width must fit norm groups")
        if not 1 <= int(self.attention_head_dim) <= min(self.block_out_channels):
            raise ValueError("Rectified-flow attention head dimension is invalid")
        if not math.isfinite(float(self.noise_sigma)) or not 0.0 < float(
            self.noise_sigma
        ) <= 1.0:
            raise ValueError("Rectified-flow noise sigma must be in (0, 1]")
        if not 1 <= int(self.inference_steps) <= 64:
            raise ValueError("Rectified-flow inference steps must be in [1, 64]")
        if not self.inference_seeds or len(set(self.inference_seeds)) != len(
            self.inference_seeds
        ):
            raise ValueError("Rectified-flow inference seeds must be unique")
        return self

    def provenance(self) -> dict:
        self.validated()
        return {
            **asdict(self),
            "model_input_channels": MODEL_INPUT_CHANNELS,
            "model_output_channels": MODEL_OUTPUT_CHANNELS,
            "representation": "floating-relative-camera-z-residual",
            "flow_path": "noise-to-camera-z-residual",
            "latent_vae": False,
        }


@dataclass(frozen=True)
class GeometryLossWeights:
    flow: float = 1.0
    camera_z: float = 0.50
    gradient: float = 0.25
    normal: float = 0.15
    laplacian: float = 0.10
    boundary: float = 0.10
    ordering: float = 0.05

    def validated(self) -> "GeometryLossWeights":
        values = asdict(self)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values.values()):
            raise ValueError("Rectified-flow loss weights must be finite and nonnegative")
        if not any(float(value) > 0.0 for value in values.values()):
            raise ValueError("Rectified-flow loss must contain a positive term")
        return self


def build_unet(config: RectifiedFlowConfig):
    """Build an uninitialized Diffusers U-Net without importing model weights."""

    from diffusers import UNet2DModel

    config = config.validated()
    levels = len(config.block_out_channels)
    down_blocks = ["DownBlock2D"] * levels
    up_blocks = ["UpBlock2D"] * levels
    down_blocks[-1] = "AttnDownBlock2D"
    up_blocks[0] = "AttnUpBlock2D"
    return UNet2DModel(
        sample_size=int(config.sample_size),
        in_channels=MODEL_INPUT_CHANNELS,
        out_channels=MODEL_OUTPUT_CHANNELS,
        layers_per_block=int(config.layers_per_block),
        block_out_channels=tuple(int(value) for value in config.block_out_channels),
        down_block_types=tuple(down_blocks),
        up_block_types=tuple(up_blocks),
        norm_num_groups=int(config.norm_num_groups),
        attention_head_dim=int(config.attention_head_dim),
    )


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def normalized_camera_rays(
    height: int,
    width: int,
    fov_y_degrees: float,
    *,
    device=None,
    dtype=None,
):
    """Return OpenCV-style unit camera rays as [1, 3, H, W]."""

    import torch

    height = int(height)
    width = int(width)
    fov = float(fov_y_degrees)
    if height < 2 or width < 2:
        raise ValueError("Camera-ray grid must be at least 2x2")
    if not math.isfinite(fov) or not 5.0 <= fov <= 120.0:
        raise ValueError("Camera vertical field of view must be in [5, 120] degrees")
    dtype = dtype or torch.float32
    focal = 0.5 * height / math.tan(math.radians(fov) * 0.5)
    columns = torch.arange(width, device=device, dtype=dtype)
    rows = torch.arange(height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, columns, indexing="ij")
    xx = (xx - 0.5 * (width - 1)) / focal
    yy = (yy - 0.5 * (height - 1)) / focal
    rays = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)
    rays = rays / torch.linalg.vector_norm(rays, dim=0, keepdim=True).clamp_min(1e-8)
    return rays.unsqueeze(0)


def camera_rays_from_intrinsics(
    height: int,
    width: int,
    intrinsics: Sequence[Sequence[float]],
    crop_xyxy: Sequence[int],
    *,
    device=None,
    dtype=None,
):
    """Return unit rays for pixels inverse-warped through a source-image crop."""

    import torch

    height = int(height)
    width = int(width)
    if height < 2 or width < 2:
        raise ValueError("Camera-ray grid must be at least 2x2")
    matrix = torch.as_tensor(intrinsics, dtype=torch.float64)
    if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
        raise ValueError("Camera intrinsics must be one finite 3x3 matrix")
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive")
    if len(crop_xyxy) != 4:
        raise ValueError("Camera crop must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (float(value) for value in crop_xyxy)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("Camera crop must be finite")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Camera crop must have positive area")
    dtype = dtype or torch.float32
    columns = torch.arange(width, device=device, dtype=dtype) + 0.5
    rows = torch.arange(height, device=device, dtype=dtype) + 0.5
    source_x = x0 + columns * ((x1 - x0) / width)
    source_y = y0 + rows * ((y1 - y0) / height)
    yy, xx = torch.meshgrid(source_y, source_x, indexing="ij")
    rays = torch.stack(
        (
            (xx - cx) / fx,
            (yy - cy) / fy,
            torch.ones_like(xx),
        ),
        dim=0,
    )
    rays = rays / torch.linalg.vector_norm(rays, dim=0, keepdim=True).clamp_min(1e-8)
    return rays.unsqueeze(0)


def _require_image_tensor(name: str, values, channels: int):
    import torch

    if not torch.is_tensor(values) or values.ndim != 4 or values.shape[1] != channels:
        raise ValueError(f"{name} must have shape [B, {channels}, H, W]")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def build_static_conditioning(
    rgb,
    coarse_depth,
    coarse_validity,
    face_support,
    camera_rays,
    focal_confidence,
):
    """Pack the ten non-stochastic conditioning channels."""

    import torch

    _require_image_tensor("RGB conditioning", rgb, 3)
    _require_image_tensor("Coarse depth", coarse_depth, 1)
    _require_image_tensor("Coarse validity", coarse_validity, 1)
    _require_image_tensor("Face support", face_support, 1)
    _require_image_tensor("Camera rays", camera_rays, 3)
    shape = rgb.shape[0], rgb.shape[2], rgb.shape[3]
    for name, values in (
        ("coarse depth", coarse_depth),
        ("coarse validity", coarse_validity),
        ("face support", face_support),
        ("camera rays", camera_rays),
    ):
        if (values.shape[0], values.shape[2], values.shape[3]) != shape:
            raise ValueError(f"{name} does not match RGB batch and spatial shape")
    confidence = torch.as_tensor(
        focal_confidence,
        device=rgb.device,
        dtype=rgb.dtype,
    )
    if confidence.ndim == 0:
        confidence = confidence.expand(rgb.shape[0])
    if confidence.shape != (rgb.shape[0],) or not torch.isfinite(confidence).all():
        raise ValueError("Focal confidence must be one finite value per batch row")
    confidence = confidence[:, None, None, None].expand(
        -1,
        1,
        rgb.shape[2],
        rgb.shape[3],
    )
    packed = torch.cat(
        (
            rgb,
            coarse_depth,
            coarse_validity,
            face_support,
            camera_rays,
            confidence,
        ),
        dim=1,
    )
    if packed.shape[1] != STATIC_CONDITIONING_CHANNELS:
        raise AssertionError("Static rectified-flow conditioning width changed")
    return packed


def flow_training_state(target_residual, time, noise, noise_sigma: float):
    """Interpolate from bounded noise at t=0 to target residual at t=1."""

    import torch

    _require_image_tensor("Target residual", target_residual, 1)
    _require_image_tensor("Flow noise", noise, 1)
    if noise.shape != target_residual.shape:
        raise ValueError("Flow noise and target residual shapes must match")
    time = torch.as_tensor(time, device=target_residual.device, dtype=target_residual.dtype)
    if time.ndim == 1:
        time = time[:, None, None, None]
    if time.shape != (target_residual.shape[0], 1, 1, 1):
        raise ValueError("Flow time must contain one value per batch row")
    if not torch.isfinite(time).all() or torch.any((time < 0.0) | (time > 1.0)):
        raise ValueError("Flow time must be finite in [0, 1]")
    sigma = float(noise_sigma)
    if not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
        raise ValueError("Flow noise sigma must be in (0, 1]")
    source = sigma * noise
    state = (1.0 - time) * source + time * target_residual
    velocity = target_residual - source
    return state, velocity


def flow_endpoint(state, velocity, time):
    import torch

    time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
    if time.ndim == 1:
        time = time[:, None, None, None]
    return state + (1.0 - time) * velocity


def _masked_mean(values, mask):
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _equal_part_mean(values, part_masks: Sequence):
    active = [
        _masked_mean(values, mask)
        for mask in part_masks
        if bool(mask.to(dtype=values.dtype).sum().detach() > 0.0)
    ]
    if not active:
        return values.new_tensor(0.0)
    return sum(active) / len(active)


def _balanced_face_part_mean(values, face_mask, part_masks: Sequence):
    active_parts = [
        mask
        for mask in part_masks
        if bool(mask.to(dtype=values.dtype).sum().detach() > 0.0)
    ]
    whole_face = _masked_mean(values, face_mask)
    if not active_parts:
        return whole_face
    return 0.5 * whole_face + 0.5 * _equal_part_mean(values, active_parts)


def _axis_gradient(values):
    return values[..., :, 1:] - values[..., :, :-1], values[..., 1:, :] - values[..., :-1, :]


def _laplacian(values):
    import torch.nn.functional as functional

    return (
        -4.0 * values
        + functional.pad(values[..., :, 1:], (0, 1, 0, 0), mode="replicate")
        + functional.pad(values[..., :, :-1], (1, 0, 0, 0), mode="replicate")
        + functional.pad(values[..., 1:, :], (0, 0, 0, 1), mode="replicate")
        + functional.pad(values[..., :-1, :], (0, 0, 1, 0), mode="replicate")
    )


def _surface_normals(depth, camera_rays=None):
    import torch
    import torch.nn.functional as functional

    if camera_rays is None:
        columns = torch.arange(depth.shape[3], device=depth.device, dtype=depth.dtype)
        rows = torch.arange(depth.shape[2], device=depth.device, dtype=depth.dtype)
        yy, xx = torch.meshgrid(rows, columns, indexing="ij")
        xx = xx[None, None].expand(depth.shape[0], -1, -1, -1)
        yy = yy[None, None].expand(depth.shape[0], -1, -1, -1)
        points = torch.cat((xx, yy, depth), dim=1)
    else:
        if camera_rays.shape != (depth.shape[0], 3, depth.shape[2], depth.shape[3]):
            raise ValueError("Camera rays must match the depth batch and spatial shape")
        if not torch.isfinite(camera_rays).all():
            raise ValueError("Camera rays must be finite")
        ray_z = camera_rays[:, 2:3]
        if torch.any(ray_z <= 1e-6):
            raise ValueError("Camera rays must point forward")
        points = camera_rays * (depth / ray_z)
    dx, dy = _axis_gradient(points)
    dx = functional.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = functional.pad(dy, (0, 0, 0, 1), mode="replicate")
    normals = torch.linalg.cross(dy, dx, dim=1)
    return normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp_min(1e-8)


def geometric_flow_loss(
    predicted_velocity,
    target_velocity,
    state,
    time,
    target_residual,
    face_mask,
    part_masks: Sequence,
    boundary_mask,
    *,
    coarse_depth=None,
    camera_rays=None,
    weights: GeometryLossWeights = GeometryLossWeights(),
):
    """Compute flow and endpoint geometry losses with equal six-part influence."""

    import torch

    weights = weights.validated()
    for name, values in (
        ("predicted velocity", predicted_velocity),
        ("target velocity", target_velocity),
        ("flow state", state),
        ("target residual", target_residual),
        ("face mask", face_mask),
        ("boundary mask", boundary_mask),
    ):
        _require_image_tensor(name, values, 1)
    expected_shape = target_residual.shape
    for name, values in (
        ("predicted velocity", predicted_velocity),
        ("target velocity", target_velocity),
        ("flow state", state),
        ("face mask", face_mask),
        ("boundary mask", boundary_mask),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must match the target residual shape")
    if any(mask.shape != face_mask.shape for mask in part_masks):
        raise ValueError("Every face-part mask must match the face support")
    if coarse_depth is None:
        coarse_depth = torch.zeros_like(target_residual)
    else:
        _require_image_tensor("coarse depth", coarse_depth, 1)
        if coarse_depth.shape != expected_shape:
            raise ValueError("coarse depth must match the target residual shape")
    raw_endpoint = flow_endpoint(state, predicted_velocity, time)
    endpoint = torch.where(face_mask > 0.5, raw_endpoint, torch.zeros_like(raw_endpoint))
    predicted_depth = coarse_depth + endpoint
    target_depth = coarse_depth + target_residual
    value_error = torch.abs(endpoint - target_residual)
    # Both challengers optimize the emitted endpoint. The flow parameterization
    # still predicts velocity, but receives no extra supervision unavailable to
    # the direct control.
    flow_loss = _balanced_face_part_mean(value_error, face_mask, part_masks)
    camera_z_loss = _balanced_face_part_mean(value_error, face_mask, part_masks)

    pred_dx, pred_dy = _axis_gradient(predicted_depth)
    target_dx, target_dy = _axis_gradient(target_depth)
    face_x = face_mask[..., :, 1:] * face_mask[..., :, :-1]
    face_y = face_mask[..., 1:, :] * face_mask[..., :-1, :]
    part_x = [mask[..., :, 1:] * mask[..., :, :-1] for mask in part_masks]
    part_y = [mask[..., 1:, :] * mask[..., :-1, :] for mask in part_masks]
    gradient_loss = 0.5 * (
        _balanced_face_part_mean(torch.abs(pred_dx - target_dx), face_x, part_x)
        + _balanced_face_part_mean(torch.abs(pred_dy - target_dy), face_y, part_y)
    )

    predicted_normals = _surface_normals(predicted_depth, camera_rays)
    target_normals = _surface_normals(target_depth, camera_rays)
    normal_error = 1.0 - torch.sum(predicted_normals * target_normals, dim=1, keepdim=True)
    normal_loss = _balanced_face_part_mean(normal_error, face_mask, part_masks)
    laplacian_loss = _balanced_face_part_mean(
        torch.abs(_laplacian(predicted_depth) - _laplacian(target_depth)),
        face_mask,
        part_masks,
    )
    boundary_loss = _masked_mean(torch.abs(endpoint), boundary_mask)
    ordering_x = torch.relu(-(pred_dx * target_dx)) * torch.abs(target_dx)
    ordering_y = torch.relu(-(pred_dy * target_dy)) * torch.abs(target_dy)
    ordering_loss = 0.5 * (
        _balanced_face_part_mean(ordering_x, face_x, part_x)
        + _balanced_face_part_mean(ordering_y, face_y, part_y)
    )
    components = {
        "flow": flow_loss,
        "camera_z": camera_z_loss,
        "gradient": gradient_loss,
        "normal": normal_loss,
        "laplacian": laplacian_loss,
        "boundary": boundary_loss,
        "ordering": ordering_loss,
    }
    total = sum(
        float(getattr(weights, name)) * value for name, value in components.items()
    )
    return total, components, endpoint


def deterministic_geometry_loss(
    predicted_residual,
    target_residual,
    face_mask,
    part_masks: Sequence,
    boundary_mask,
    *,
    coarse_depth=None,
    camera_rays=None,
    weights: GeometryLossWeights = GeometryLossWeights(),
):
    """Architecture-matched direct-regression control for the flow challenger."""

    zero_time = predicted_residual.new_zeros(predicted_residual.shape[0])
    zero_state = predicted_residual.new_zeros(predicted_residual.shape)
    target_velocity = target_residual
    return geometric_flow_loss(
        predicted_residual,
        target_velocity,
        zero_state,
        zero_time,
        target_residual,
        face_mask,
        part_masks,
        boundary_mask,
        coarse_depth=coarse_depth,
        camera_rays=camera_rays,
        weights=weights,
    )


def _model_velocity(model, state, static_conditioning, time):
    import torch

    sample = torch.cat((state, static_conditioning), dim=1)
    timestep = torch.as_tensor(time, device=state.device, dtype=state.dtype)
    if timestep.ndim == 0:
        timestep = timestep.expand(state.shape[0])
    return model(sample, timestep * 1000.0).sample


def project_flow_state_to_support(state, static_conditioning):
    """Remove dynamic state outside support while retaining scene conditioning."""

    if static_conditioning.ndim != 4 or static_conditioning.shape[1] != STATIC_CONDITIONING_CHANNELS:
        raise ValueError("Static conditioning has the wrong shape")
    if state.ndim != 4 or state.shape[1] != 1:
        raise ValueError("Flow state must have shape [B, 1, H, W]")
    if state.shape[0] != static_conditioning.shape[0] or state.shape[2:] != static_conditioning.shape[2:]:
        raise ValueError("Flow state and conditioning shapes must match")
    support = static_conditioning[
        :,
        FACE_SUPPORT_STATIC_CHANNEL : FACE_SUPPORT_STATIC_CHANNEL + 1,
    ]
    return state * (support > 0.5).to(dtype=state.dtype)


def sample_rectified_flow(
    model,
    static_conditioning,
    *,
    seeds: Iterable[int],
    steps: int,
    noise_sigma: float,
):
    """Return fixed-seed Euler samples as [S, B, 1, H, W]."""

    import torch

    if static_conditioning.ndim != 4 or static_conditioning.shape[1] != STATIC_CONDITIONING_CHANNELS:
        raise ValueError("Static conditioning has the wrong shape")
    steps = int(steps)
    if not 1 <= steps <= 64:
        raise ValueError("Flow sampling steps must be in [1, 64]")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("Flow sampling seeds must be unique")
    samples = []
    for seed in seed_values:
        generator = torch.Generator(device=static_conditioning.device)
        generator.manual_seed(seed)
        state = float(noise_sigma) * torch.randn(
            (
                static_conditioning.shape[0],
                1,
                static_conditioning.shape[2],
                static_conditioning.shape[3],
            ),
            generator=generator,
            device=static_conditioning.device,
            dtype=static_conditioning.dtype,
        )
        state = project_flow_state_to_support(state, static_conditioning)
        for index in range(steps):
            time = state.new_full((state.shape[0],), index / steps)
            state = project_flow_state_to_support(
                state
                + _model_velocity(
                    model,
                    state,
                    static_conditioning,
                    time,
                )
                / steps,
                static_conditioning,
            )
        samples.append(state)
    return torch.stack(samples, dim=0)


def posterior_medoid(samples, support):
    """Select the sample with minimum supported pairwise L1 distance."""

    import torch

    if samples.ndim != 5 or samples.shape[2] != 1:
        raise ValueError("Flow samples must have shape [S, B, 1, H, W]")
    _require_image_tensor("Posterior support", support, 1)
    if samples.shape[1:] != support.shape:
        raise ValueError("Posterior support does not match flow samples")
    weights = support.to(dtype=samples.dtype)
    distances = []
    for candidate in samples:
        difference = torch.abs(samples - candidate.unsqueeze(0))
        distance = (difference * weights.unsqueeze(0)).sum(dim=(2, 3, 4))
        distance = distance / weights.sum(dim=(1, 2, 3)).clamp_min(1.0)
        distances.append(distance.sum(dim=0))
    scores = torch.stack(distances, dim=0)
    indices = torch.argmin(scores, dim=0)
    batch_indices = torch.arange(samples.shape[1], device=samples.device)
    return samples[indices, batch_indices], indices


def median_absolute_deviation(samples):
    import torch

    if samples.ndim != 5:
        raise ValueError("Flow samples must have shape [S, B, C, H, W]")
    median = torch.median(samples, dim=0).values
    return torch.median(torch.abs(samples - median.unsqueeze(0)), dim=0).values


def apply_supported_residual(coarse_depth, residual, support):
    """Apply a correction while preserving every unsupported value bit-exactly."""

    import torch

    for name, values in (
        ("coarse depth", coarse_depth),
        ("residual", residual),
        ("support", support),
    ):
        _require_image_tensor(name, values, 1)
    if coarse_depth.shape != residual.shape or coarse_depth.shape != support.shape:
        raise ValueError("Supported residual inputs must have matching shapes")
    return torch.where(support > 0.5, coarse_depth + residual, coarse_depth)
