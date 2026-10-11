"""Tests for Gaussian diffusion utilities."""

from __future__ import annotations

import pytest
import torch

from hmc_dit3d.train.diffusion import (
    DiffusionConfig,
    DiffusionConfigError,
    GaussianDiffusion,
    get_beta_schedule,
)


def test_get_beta_schedule_returns_expected_shape() -> None:
    """The beta schedule should match the configured timestep count."""
    config = DiffusionConfig(
        schedule_type="warm0.1",
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        num_timesteps=100,
    )

    betas = get_beta_schedule(config)

    assert betas.shape == (100,)
    assert betas[0] == config.beta_start
    assert betas[-1] == config.beta_end


def test_gaussian_diffusion_q_sample_preserves_shape() -> None:
    """q_sample should return noisy tensors with the same shape as inputs."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=50,
        )
    )
    x_start = torch.randn(2, 3, 128)
    timesteps = torch.tensor([1, 7], dtype=torch.int64)
    noise = torch.randn_like(x_start)

    x_t = diffusion.q_sample(x_start, timesteps, noise=noise)

    assert x_t.shape == x_start.shape
    assert torch.isfinite(x_t).all()


def test_gaussian_diffusion_p_losses_returns_per_sample_values() -> None:
    """p_losses should compute one MSE loss per sample for epsilon prediction."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=50,
        )
    )
    x_start = torch.randn(2, 3, 128)
    timesteps = torch.tensor([3, 11], dtype=torch.int64)

    losses, x_t, noise = diffusion.p_losses(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        x_start=x_start,
        timesteps=timesteps,
    )

    assert losses.shape == (2,)
    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape
    assert torch.isfinite(losses).all()


def test_gaussian_diffusion_p_sample_preserves_shape_and_finiteness() -> None:
    """One reverse diffusion step should preserve shape and remain finite."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=20,
        )
    )
    x_t = torch.randn(2, 3, 32)
    timesteps = torch.tensor([7, 3], dtype=torch.int64)

    x_prev = diffusion.p_sample(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        x_t=x_t,
        timesteps=timesteps,
    )

    assert x_prev.shape == x_t.shape
    assert torch.isfinite(x_prev).all()


def test_gaussian_diffusion_p_sample_is_deterministic_at_t0() -> None:
    """The final reverse step should not inject extra Gaussian noise at `t=0`."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=20,
        )
    )
    x_t = torch.randn(2, 3, 32)
    timesteps = torch.zeros(2, dtype=torch.int64)

    first = diffusion.p_sample(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        x_t=x_t,
        timesteps=timesteps,
    )
    second = diffusion.p_sample(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        x_t=x_t,
        timesteps=timesteps,
    )

    assert torch.allclose(first, second)


def test_gaussian_diffusion_p_sample_loop_returns_requested_shape() -> None:
    """Check that ancestral sampling returns the requested finite tensor shape."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=4,
        )
    )

    samples = diffusion.p_sample_loop(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        shape=(2, 3, 16),
        device="cpu",
    )

    assert samples.shape == (2, 3, 16)
    assert torch.isfinite(samples).all()


def test_gaussian_diffusion_p_sample_loop_collect_timesteps_stride() -> None:
    """Collect noise and checkpoints at stride-aligned steps."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=20,
        )
    )
    final, traj = diffusion.p_sample_loop_collect_timesteps(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        shape=(1, 3, 8),
        device="cpu",
        collect_stride=10,
    )
    assert final.shape == (1, 3, 8)
    assert torch.isfinite(final).all()
    expected_keys = {20, 10, 0}
    assert set(traj.keys()) == expected_keys
    for _k, tensor in traj.items():
        assert tensor.shape == (1, 8, 3)
        assert tensor.device.type == "cpu"


def test_gaussian_diffusion_p_sample_loop_collect_timesteps_collect_at() -> None:
    """Explicit collect_at should save only requested labels."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=20,
        )
    )
    final, traj = diffusion.p_sample_loop_collect_timesteps(
        denoise_fn=lambda x, t: torch.zeros_like(x),
        shape=(1, 3, 8),
        device="cpu",
        collect_at=[20, 15, 6, 0],
    )
    assert final.shape == (1, 3, 8)
    assert set(traj.keys()) == {20, 15, 6, 0}


def test_gaussian_diffusion_collect_timesteps_rejects_stride_and_at_together() -> None:
    """Passing both collect_stride and collect_at must fail."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=10,
        )
    )
    with pytest.raises(DiffusionConfigError, match="exactly one"):
        diffusion.p_sample_loop_collect_timesteps(
            denoise_fn=lambda x, t: torch.zeros_like(x),
            shape=(1, 3, 4),
            device="cpu",
            collect_stride=2,
            collect_at=[10, 0],
        )


@pytest.mark.parametrize(
    ("timesteps", "message_fragment"),
    [
        (torch.tensor([[1, 2]], dtype=torch.int64), r"shape \(B,\)"),
        (torch.tensor([50, 0], dtype=torch.int64), "invalid values"),
        (torch.tensor([-1, 0], dtype=torch.int64), "invalid values"),
        (torch.tensor([0.5, 1.5], dtype=torch.float32), "integer dtype"),
    ],
)
def test_gaussian_diffusion_rejects_invalid_timesteps(
    timesteps: torch.Tensor, message_fragment: str
) -> None:
    """Invalid timestep tensors should fail with explicit config errors."""
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            schedule_type="linear",
            beta_start=1.0e-4,
            beta_end=2.0e-2,
            num_timesteps=50,
        )
    )
    x_start = torch.randn(2, 3, 128)

    with pytest.raises(DiffusionConfigError, match=message_fragment):
        diffusion.q_sample(x_start, timesteps)
