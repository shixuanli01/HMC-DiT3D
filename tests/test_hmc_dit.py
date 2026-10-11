"""Tests for the minimal HMC-conditioned DiT backbone."""

from __future__ import annotations

import pytest
import torch

from hmc_dit3d.hmc.config import HMCConfig, HMCEncoderConfig
from hmc_dit3d.hmc.encoder import HMCConditionOutput
from hmc_dit3d.models.hmc_dit import (
    HMCConditionedBottleneckDiT,
    HMCConditionedBottleneckDiTBackbone,
    HMCConditionedBottleneckDiTPointCloud,
    HMCConditionedBottleneckDiTVoxel,
    HMCConditionedDiT,
    HMCConditionedDiTBackbone,
    HMCConditionedDiTBlock,
    HMCConditionedDiTConfig,
    HMCConditionedDiTPointCloud,
    HMCConditionedDiTVoxel,
    HMCModelConfigurationError,
    HMCModelInputError,
    PointCloudVoxelizer,
)


def test_hmc_dit_block_supports_forward_and_backward() -> None:
    """The DiT block should preserve shapes and propagate gradients."""
    block = HMCConditionedDiTBlock(model_dim=32, num_heads=4, mlp_ratio=2.0)
    x = torch.randn(2, 64, 32, requires_grad=True)
    condition = torch.randn(2, 32, requires_grad=True)
    hmc_tokens = torch.randn(2, 20, 32, requires_grad=True)

    output = block(x, condition, hmc_tokens)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 64, 32)
    assert x.grad is not None
    assert condition.grad is not None
    assert hmc_tokens.grad is not None
    assert torch.isfinite(output).all()


def test_hmc_dit_backbone_supports_forward_and_backward() -> None:
    """The backbone should consume `(g, C)` and return projected tokens."""
    config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=128,
        num_classes=3,
    )
    model = HMCConditionedDiTBackbone(config)
    x_tokens = torch.randn(2, 64, 24, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    hmc_conditions = HMCConditionOutput(
        global_embedding=torch.randn(2, 32, requires_grad=True),
        condition_tokens=torch.randn(2, 20, 32, requires_grad=True),
    )

    output = model(x_tokens, timesteps, labels, hmc_conditions)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 64, 24)
    assert x_tokens.grad is not None
    assert hmc_conditions.global_embedding.grad is not None
    assert hmc_conditions.condition_tokens.grad is not None
    assert torch.isfinite(output).all()


def test_hmc_dit_backbone_rejects_excessive_token_count() -> None:
    """The backbone should fail fast when token count exceeds max_tokens."""
    config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=1,
        num_heads=4,
        max_tokens=16,
        num_classes=2,
    )
    model = HMCConditionedDiTBackbone(config)
    x_tokens = torch.randn(2, 32, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.int64)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    hmc_conditions = HMCConditionOutput(
        global_embedding=torch.randn(2, 32),
        condition_tokens=torch.randn(2, 8, 32),
    )

    with pytest.raises(HMCModelInputError):
        model(x_tokens, timesteps, labels, hmc_conditions)


@pytest.mark.parametrize("bad_labels", [torch.tensor([0, 3]), torch.tensor([-1, 1])])
def test_hmc_dit_backbone_rejects_invalid_label_ranges(
    bad_labels: torch.Tensor,
) -> None:
    """The backbone should reject labels outside `[0, num_classes - 1]`."""
    config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=1,
        num_heads=4,
        max_tokens=16,
        num_classes=3,
    )
    model = HMCConditionedDiTBackbone(config)
    x_tokens = torch.randn(2, 8, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.int64)
    hmc_conditions = HMCConditionOutput(
        global_embedding=torch.randn(2, 32),
        condition_tokens=torch.randn(2, 8, 32),
    )

    with pytest.raises(HMCModelInputError):
        model(x_tokens, timesteps, bad_labels, hmc_conditions)


def test_hmc_dit_backbone_zero_initializes_output_projection() -> None:
    """The output projection should follow DiT-style zero initialization."""
    config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=1,
        num_heads=4,
        max_tokens=16,
        num_classes=2,
    )
    model = HMCConditionedDiTBackbone(config)

    assert torch.count_nonzero(model.output_projection.weight) == 0
    assert torch.count_nonzero(model.output_projection.bias) == 0


def test_hmc_conditioned_dit_supports_end_to_end_forward_and_backward() -> None:
    """The wrapper should connect HMC encoding and the DiT backbone end to end."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=128,
        num_classes=3,
    )
    model = HMCConditionedDiT(hmc_config, encoder_config, backbone_config)
    x_tokens = torch.randn(2, 64, 24, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(x_tokens, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 64, 24)
    assert x_tokens.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None


def test_hmc_conditioned_dit_requires_matching_condition_dimensions() -> None:
    """The wrapper should reject encoder/backbone dimension mismatches."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=32,
        condition_dim=64,
        depth=1,
        num_heads=4,
        max_tokens=32,
        num_classes=2,
    )

    with pytest.raises(HMCModelConfigurationError):
        HMCConditionedDiT(hmc_config, encoder_config, backbone_config)


def test_hmc_bottleneck_backbone_supports_forward_and_backward() -> None:
    """The 3.2 bottleneck backbone should support gradients end to end."""
    config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=128,
        num_classes=3,
    )
    model = HMCConditionedBottleneckDiTBackbone(
        config=config,
        num_bottleneck_latents=8,
        resampler_depth=2,
    )
    x_tokens = torch.randn(2, 64, 24, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    hmc_conditions = HMCConditionOutput(
        global_embedding=torch.randn(2, 32, requires_grad=True),
        condition_tokens=torch.randn(2, 20, 32, requires_grad=True),
    )

    output = model(x_tokens, timesteps, labels, hmc_conditions)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 64, 24)
    assert x_tokens.grad is not None
    assert hmc_conditions.global_embedding.grad is not None
    assert hmc_conditions.condition_tokens.grad is not None
    assert torch.isfinite(output).all()


def test_hmc_conditioned_bottleneck_dit_rejects_invalid_latent_counts() -> None:
    """The 3.2 backbone should reject invalid bottleneck settings."""
    config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=1,
        num_heads=4,
        max_tokens=16,
        num_classes=2,
    )

    with pytest.raises(HMCModelConfigurationError):
        HMCConditionedBottleneckDiTBackbone(
            config=config,
            num_bottleneck_latents=0,
            resampler_depth=1,
        )


def test_hmc_bottleneck_wrapper_supports_end_to_end_forward_and_backward() -> None:
    """The 3.2 wrapper should connect the HMC encoder and bottleneck path."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=128,
        num_classes=3,
    )
    model = HMCConditionedBottleneckDiT(
        hmc_config=hmc_config,
        encoder_config=encoder_config,
        backbone_config=backbone_config,
        num_bottleneck_latents=8,
        resampler_depth=2,
    )
    x_tokens = torch.randn(2, 64, 24, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(x_tokens, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 64, 24)
    assert x_tokens.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None


def test_hmc_conditioned_voxel_dit_supports_voxel_to_voxel_forward() -> None:
    """The voxel wrapper should preserve dense voxel shapes end to end."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=64,
        num_classes=3,
        out_dim=24,
    )
    model = HMCConditionedDiTVoxel(
        hmc_config=hmc_config,
        encoder_config=encoder_config,
        backbone_config=backbone_config,
        voxel_size=8,
        patch_size=2,
        in_channels=3,
    )
    voxels = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(voxels, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 3, 8, 8, 8)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))
    assert voxels.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None
    assert not model.token_model.backbone.position_embedding.requires_grad


def test_hmc_conditioned_bottleneck_voxel_dit_supports_voxel_to_voxel_forward() -> None:
    """The 3.2 voxel wrapper should preserve dense voxel shapes end to end."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=64,
        num_classes=3,
        out_dim=24,
    )
    model = HMCConditionedBottleneckDiTVoxel(
        hmc_config=hmc_config,
        encoder_config=encoder_config,
        backbone_config=backbone_config,
        voxel_size=8,
        patch_size=2,
        in_channels=3,
        num_bottleneck_latents=8,
        resampler_depth=2,
    )
    voxels = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(voxels, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 3, 8, 8, 8)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))
    assert voxels.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None
    assert not model.token_model.backbone.position_embedding.requires_grad


def test_hmc_conditioned_voxel_dit_validates_patch_interface_contract() -> None:
    """The voxel wrapper should reject incompatible backbone patch outputs."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=1,
        num_heads=4,
        max_tokens=64,
        num_classes=2,
        out_dim=16,
    )

    with pytest.raises(HMCModelConfigurationError):
        HMCConditionedDiTVoxel(
            hmc_config=hmc_config,
            encoder_config=encoder_config,
            backbone_config=backbone_config,
            voxel_size=8,
            patch_size=2,
            in_channels=3,
        )


def test_hmc_dit_backbone_matches_baseline_init_scales() -> None:
    """Key embedding weights should follow DiT-3D-style initialization scales."""
    torch.manual_seed(7)
    config = HMCConditionedDiTConfig(
        input_dim=24,
        model_dim=96,
        condition_dim=96,
        depth=2,
        num_heads=4,
        max_tokens=128,
        num_classes=3,
    )
    model = HMCConditionedDiTBackbone(config)

    label_std = float(model.label_embedder.embedding_table.weight.std().item())
    time_std_0 = float(model.timestep_embedder.mlp[0].weight.std().item())
    time_std_2 = float(model.timestep_embedder.mlp[2].weight.std().item())

    assert 0.005 < label_std < 0.05
    assert 0.005 < time_std_0 < 0.05
    assert 0.005 < time_std_2 < 0.05


def test_point_cloud_voxelizer_returns_expected_shapes() -> None:
    """The pure PyTorch voxelizer should produce dense voxels and voxel coords."""
    voxelizer = PointCloudVoxelizer(resolution=8, normalize=True, eps=0.0)
    points = torch.randn(2, 3, 128)

    voxels, norm_coords = voxelizer(points, points)

    assert voxels.shape == (2, 3, 8, 8, 8)
    assert norm_coords.shape == (2, 3, 128)
    assert torch.isfinite(voxels).all()
    assert torch.isfinite(norm_coords).all()


def test_point_cloud_voxelizer_maps_unit_interval_to_closed_voxel_range() -> None:
    """Check `[0, 1] -> [0, resolution - 1]` mapping without early saturation."""
    voxelizer = PointCloudVoxelizer(resolution=8, normalize=True, eps=0.0)
    points = torch.tensor(
        [[[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )

    _, norm_coords = voxelizer(points, points)

    expected = torch.tensor(
        [[[7.0, 0.0], [3.5, 3.5], [3.5, 3.5]]],
        dtype=torch.float32,
    )
    assert torch.allclose(norm_coords, expected)


def test_hmc_conditioned_point_cloud_dit_supports_point_to_point_forward() -> None:
    """The point-cloud wrapper should preserve point tensor shapes end to end."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=64,
        num_classes=3,
        out_dim=24,
    )
    model = HMCConditionedDiTPointCloud(
        hmc_config=hmc_config,
        encoder_config=encoder_config,
        backbone_config=backbone_config,
        voxel_size=8,
        patch_size=2,
        in_channels=3,
    )
    points = torch.randn(2, 3, 128, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(points, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 3, 128)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))
    assert points.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None


def test_hmc_bottleneck_point_cloud_wrapper_supports_forward() -> None:
    """The 3.2 point-cloud wrapper should preserve point tensor shapes."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    backbone_config = HMCConditionedDiTConfig(
        input_dim=16,
        model_dim=32,
        condition_dim=32,
        depth=2,
        num_heads=4,
        max_tokens=64,
        num_classes=3,
        out_dim=24,
    )
    model = HMCConditionedBottleneckDiTPointCloud(
        hmc_config=hmc_config,
        encoder_config=encoder_config,
        backbone_config=backbone_config,
        voxel_size=8,
        patch_size=2,
        in_channels=3,
        num_bottleneck_latents=8,
        resampler_depth=2,
    )
    points = torch.randn(2, 3, 128, requires_grad=True)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    labels = torch.tensor([0, 2], dtype=torch.int64)
    descriptor = torch.randn(2, hmc_config.descriptor_dim, requires_grad=True)
    sequences = [
        torch.randn(2, (2**2) ** 3, requires_grad=True),
        torch.randn(2, (2**3) ** 3, requires_grad=True),
    ]

    output = model(points, timesteps, labels, descriptor, sequences)
    loss = output.sum()
    loss.backward()

    assert output.shape == (2, 3, 128)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))
    assert points.grad is not None
    assert descriptor.grad is not None
    assert sequences[0].grad is not None
    assert sequences[1].grad is not None
