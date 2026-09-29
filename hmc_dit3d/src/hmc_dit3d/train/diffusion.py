"""Diffusion utilities for HMC-DiT3D training."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)

VALID_SCHEDULE_TYPES = ("linear", "warm0.1", "warm0.2", "warm0.5")


class DiffusionConfigError(RuntimeError):
    """Raised when diffusion settings are invalid."""


@dataclass(slots=True, frozen=True)
class DiffusionConfig:
    """Configuration for Gaussian diffusion training.

    Args:
        schedule_type: Beta schedule type.
        beta_start: Initial beta value.
        beta_end: Final beta value.
        num_timesteps: Number of diffusion steps.
    """

    schedule_type: str = "linear"
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    num_timesteps: int = 1000

    def __post_init__(self) -> None:
        """Validate diffusion hyperparameters."""
        if self.schedule_type not in VALID_SCHEDULE_TYPES:
            message = (
                f"schedule_type must be one of {VALID_SCHEDULE_TYPES!r}, "
                f"got {self.schedule_type!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if not 0.0 < self.beta_start <= 1.0:
            message = "beta_start must be in (0, 1]."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if not 0.0 < self.beta_end <= 1.0:
            message = "beta_end must be in (0, 1]."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if self.beta_start > self.beta_end:
            message = "beta_start must be less than or equal to beta_end."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if self.num_timesteps <= 0:
            message = "num_timesteps must be positive."
            LOGGER.error(message)
            raise DiffusionConfigError(message)


def get_beta_schedule(config: DiffusionConfig) -> np.ndarray:
    """Create a beta schedule matching baseline training options.

    Args:
        config: Diffusion configuration.

    Returns:
        np.ndarray: Beta schedule with shape `(T,)`.
    """
    if config.schedule_type == "linear":
        return np.linspace(config.beta_start, config.beta_end, config.num_timesteps)

    warm_fraction = {
        "warm0.1": 0.1,
        "warm0.2": 0.2,
        "warm0.5": 0.5,
    }[config.schedule_type]
    betas = np.full(config.num_timesteps, config.beta_end, dtype=np.float64)
    warm_steps = max(int(config.num_timesteps * warm_fraction), 1)
    betas[:warm_steps] = np.linspace(
        config.beta_start,
        config.beta_end,
        warm_steps,
        dtype=np.float64,
    )
    return betas


class GaussianDiffusion:
    """Minimal Gaussian diffusion training helper."""

    def __init__(self, config: DiffusionConfig) -> None:
        """Initialize diffusion coefficients from a config.

        Args:
            config: Diffusion configuration.
        """
        self.config = config
        betas = get_beta_schedule(config).astype(np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.num_timesteps = config.num_timesteps
        self.betas = torch.from_numpy(betas).to(torch.float32)
        self.alphas = torch.from_numpy(alphas).to(torch.float32)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).to(torch.float32)
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).to(
            torch.float32
        )
        self.sqrt_alphas_cumprod = torch.from_numpy(np.sqrt(alphas_cumprod)).to(
            torch.float32
        )
        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 - alphas_cumprod)
        ).to(torch.float32)
        self.sqrt_recip_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / alphas_cumprod)
        ).to(torch.float32)
        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / alphas_cumprod - 1.0)
        ).to(torch.float32)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.from_numpy(posterior_variance).to(torch.float32)
        self.posterior_mean_coef1 = torch.from_numpy(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(torch.float32)
        self.posterior_mean_coef2 = torch.from_numpy(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(torch.float32)

    @staticmethod
    def _extract(
        coefficients: Tensor,
        timesteps: Tensor,
        x_shape: torch.Size,
    ) -> Tensor:
        """Gather timestep-dependent coefficients and reshape for broadcasting."""
        gathered = torch.gather(coefficients, 0, timesteps)
        return gathered.view(timesteps.shape[0], *([1] * (len(x_shape) - 1)))

    def _validate_timesteps(self, x_start: Tensor, timesteps: Tensor) -> Tensor:
        """Validate timestep tensors before coefficient lookup.

        Args:
            x_start: Clean tensor used to infer the batch size and target device.
            timesteps: Candidate timestep tensor.

        Returns:
            Tensor: Validated timestep tensor on the same device as `x_start`.
        """
        timesteps = torch.as_tensor(timesteps, device=x_start.device)
        if timesteps.ndim != 1:
            message = f"timesteps must have shape (B,). Got {tuple(timesteps.shape)!r}."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if timesteps.shape[0] != x_start.shape[0]:
            message = (
                "timesteps batch size must match x_start batch size. "
                f"Got {timesteps.shape[0]} and {x_start.shape[0]}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if timesteps.dtype not in (torch.int32, torch.int64):
            message = f"timesteps must use an integer dtype. Got {timesteps.dtype}."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        timesteps = timesteps.to(torch.int64)
        if ((timesteps < 0) | (timesteps >= self.num_timesteps)).any():
            invalid_values = (
                timesteps[(timesteps < 0) | (timesteps >= self.num_timesteps)]
                .detach()
                .cpu()
            )
            message = (
                f"timesteps must be in [0, {self.num_timesteps - 1}]. "
                f"Got invalid values {invalid_values.tolist()!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        return timesteps

    def q_sample(
        self,
        x_start: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Diffuse clean data with Gaussian noise.

        Args:
            x_start: Clean tensor with shape `(B, C, N)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            noise: Optional noise tensor. Defaults to standard Gaussian noise.

        Returns:
            Tensor: Noisy tensor `x_t`.
        """
        timesteps = self._validate_timesteps(x_start, timesteps)
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            message = (
                f"noise shape {tuple(noise.shape)!r} must match x_start "
                f"{tuple(x_start.shape)!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        return (
            self._extract(
                self.sqrt_alphas_cumprod.to(x_start.device),
                timesteps,
                x_start.shape,
            )
            * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device),
                timesteps,
                x_start.shape,
            )
            * noise
        )

    def min_snr_weights(self, timesteps: Tensor, gamma: float) -> Tensor:
        """Return Min-SNR weights for an epsilon-prediction objective."""
        if gamma <= 0.0:
            message = f"Min-SNR gamma must be positive, got {gamma}."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        timesteps = torch.as_tensor(timesteps)
        if timesteps.ndim != 1:
            message = f"timesteps must have shape (B,), got {tuple(timesteps.shape)!r}."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if timesteps.dtype not in (torch.int32, torch.int64):
            message = f"timesteps must use an integer dtype. Got {timesteps.dtype}."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        alpha_bar = torch.gather(
            self.alphas_cumprod.to(timesteps.device),
            0,
            timesteps.to(torch.int64),
        )
        snr = alpha_bar / torch.clamp(
            1.0 - alpha_bar,
            min=torch.finfo(alpha_bar.dtype).eps,
        )
        return torch.clamp(snr, max=float(gamma)) / torch.clamp(
            snr,
            min=torch.finfo(snr.dtype).eps,
        )

    def p_losses(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        x_start: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-sample DDPM noise-prediction losses.

        Args:
            denoise_fn: Callable mapping `(x_t, timesteps) -> predicted_noise`.
            x_start: Clean tensor with shape `(B, C, N)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            noise: Optional fixed noise tensor.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                Per-sample losses, noisy input `x_t`, and target noise.
        """
        timesteps = self._validate_timesteps(x_start, timesteps)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        predicted_noise = denoise_fn(x_t, timesteps)
        if predicted_noise.shape != x_start.shape:
            message = (
                "denoise_fn output must match x_start shape. "
                f"Expected {tuple(x_start.shape)!r}, "
                f"got {tuple(predicted_noise.shape)!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        losses = (noise - predicted_noise).pow(2).mean(dim=(1, 2))
        return losses, x_t, noise

    def predict_xstart_from_eps(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        eps: Tensor,
    ) -> Tensor:
        """Recover `x_0` from noisy samples and predicted noise.

        Args:
            x_t: Noisy tensor with shape `(B, C, N)` or a compatible broadcastable
                shape.
            timesteps: Diffusion timesteps with shape `(B,)`.
            eps: Predicted noise tensor with the same shape as `x_t`.

        Returns:
            Tensor: Predicted clean sample `x_0`.
        """
        timesteps = self._validate_timesteps(x_t, timesteps)
        if eps.shape != x_t.shape:
            message = (
                f"eps shape {tuple(eps.shape)!r} must match x_t {tuple(x_t.shape)!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        return (
            self._extract(
                self.sqrt_recip_alphas_cumprod.to(x_t.device),
                timesteps,
                x_t.shape,
            )
            * x_t
            - self._extract(
                self.sqrt_recipm1_alphas_cumprod.to(x_t.device),
                timesteps,
                x_t.shape,
            )
            * eps
        )

    def p_mean_variance(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        x_t: Tensor,
        timesteps: Tensor,
        clip_denoised: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the DDPM posterior mean/variance from predicted noise.

        Args:
            denoise_fn: Callable mapping `(x_t, timesteps) -> predicted_noise`.
            x_t: Noisy tensor with shape `(B, C, N)` or an equivalent layout.
            timesteps: Diffusion timesteps with shape `(B,)`.
            clip_denoised: Whether to clip predicted `x_0` into `[-1, 1]`.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                Posterior mean, posterior variance, and predicted clean sample.
        """
        timesteps = self._validate_timesteps(x_t, timesteps)
        predicted_noise = denoise_fn(x_t, timesteps)
        if predicted_noise.shape != x_t.shape:
            message = (
                "denoise_fn output must match x_t shape during sampling. "
                f"Expected {tuple(x_t.shape)!r}, got {tuple(predicted_noise.shape)!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        predicted_xstart = self.predict_xstart_from_eps(x_t, timesteps, predicted_noise)
        if clip_denoised:
            predicted_xstart = predicted_xstart.clamp(-1.0, 1.0)
        posterior_mean = (
            self._extract(
                self.posterior_mean_coef1.to(x_t.device),
                timesteps,
                x_t.shape,
            )
            * predicted_xstart
            + self._extract(
                self.posterior_mean_coef2.to(x_t.device), timesteps, x_t.shape
            )
            * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance.to(x_t.device),
            timesteps,
            x_t.shape,
        )
        return posterior_mean, posterior_variance, predicted_xstart

    def p_sample(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        x_t: Tensor,
        timesteps: Tensor,
        clip_denoised: bool = False,
    ) -> Tensor:
        """Sample one reverse-diffusion step from `x_t` to `x_{t-1}`.

        Args:
            denoise_fn: Callable mapping `(x_t, timesteps) -> predicted_noise`.
            x_t: Noisy tensor with shape `(B, C, N)` or an equivalent layout.
            timesteps: Diffusion timesteps with shape `(B,)`.
            clip_denoised: Whether to clip predicted `x_0` into `[-1, 1]`.

        Returns:
            Tensor: Reverse-sampled tensor `x_{t-1}`.
        """
        timesteps = self._validate_timesteps(x_t, timesteps)
        posterior_mean, posterior_variance, _ = self.p_mean_variance(
            denoise_fn=denoise_fn,
            x_t=x_t,
            timesteps=timesteps,
            clip_denoised=clip_denoised,
        )
        nonzero_mask = (
            (timesteps != 0)
            .to(x_t.dtype)
            .view(timesteps.shape[0], *([1] * (x_t.ndim - 1)))
        )
        noise = torch.randn_like(x_t)
        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    def p_sample_loop(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        device: torch.device | str,
        clip_denoised: bool = False,
    ) -> Tensor:
        """Run ancestral DDPM sampling from pure Gaussian noise.

        Args:
            denoise_fn: Callable mapping `(x_t, timesteps) -> predicted_noise`.
            shape: Target sample shape, for example `(B, C, N)`.
            device: Target torch device for the sampling loop.
            clip_denoised: Whether to clip predicted `x_0` into `[-1, 1]`.

        Returns:
            Tensor: Final sampled tensor with the requested `shape`.
        """
        if len(shape) == 0 or shape[0] <= 0:
            message = (
                f"shape must start with a positive batch dimension, got {shape!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        device = torch.device(device)
        x_t = torch.randn(shape, device=device, dtype=torch.float32)
        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (shape[0],),
                step,
                device=device,
                dtype=torch.int64,
            )
            x_t = self.p_sample(
                denoise_fn=denoise_fn,
                x_t=x_t,
                timesteps=timesteps,
                clip_denoised=clip_denoised,
            )
        return x_t

    def p_sample_loop_collect_timesteps(
        self,
        denoise_fn: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        device: torch.device | str,
        clip_denoised: bool = False,
        *,
        collect_stride: int | None = None,
        collect_at: Sequence[int] | None = None,
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Like `p_sample_loop`, but also record intermediate ``x`` states.

        Saves tensors in **point-cloud layout** ``(B, N, 3)`` (same as training batch).

        Provide **exactly one** of ``collect_stride`` or ``collect_at``.

        Keys:
            - ``num_timesteps`` (e.g. 1000): optional ``x_T`` before the reverse loop,
              included when that integer appears in ``collect_at`` or when using stride
              mode (stride mode always saves ``x_T``).
            - ``t`` in ``0 .. num_timesteps-1``: state **after** finishing the denoising
              update at DDPM index ``t``.

        Args:
            denoise_fn: Callable mapping `(x_t, timesteps) -> predicted_noise`.
            shape: Target sample shape, for example `(B, C, N)`.
            device: Target torch device for the sampling loop.
            clip_denoised: Whether to clip predicted `x_0` into `[-1, 1]`.
            collect_stride: If set, save ``x_T`` and every ``t`` with
                ``t % stride == 0``.
            collect_at: If set, save only these labels: use ``num_timesteps``
                for initial noise, and loop indices ``0 .. num_timesteps-1``
                after each matching step.

        Returns:
            tuple[Tensor, dict[int, Tensor]]: Final ``x_0`` tensor (same layout as
            ``p_sample_loop``) and CPU tensors keyed by timestep label.
        """
        if (collect_stride is None) == (collect_at is None):
            message = "Specify exactly one of collect_stride or collect_at."
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        if len(shape) == 0 or shape[0] <= 0:
            message = (
                f"shape must start with a positive batch dimension, got {shape!r}."
            )
            LOGGER.error(message)
            raise DiffusionConfigError(message)
        device = torch.device(device)
        x_t = torch.randn(shape, device=device, dtype=torch.float32)

        if collect_at is not None:
            wanted = frozenset(int(t) for t in collect_at)
            if not wanted:
                message = "collect_at must be non-empty."
                LOGGER.error(message)
                raise DiffusionConfigError(message)
            nt = int(self.num_timesteps)
            invalid = sorted(
                t for t in wanted if t != nt and not (0 <= t < self.num_timesteps)
            )
            if invalid:
                message = (
                    f"collect_at values must be {nt} (initial noise) or "
                    f"integers in [0, {self.num_timesteps - 1}], "
                    f"got extra: {invalid!r}."
                )
                LOGGER.error(message)
                raise DiffusionConfigError(message)
            trajectory: dict[int, Tensor] = {}
            if nt in wanted:
                trajectory[nt] = x_t.transpose(1, 2).detach().cpu().clone()
        else:
            assert collect_stride is not None
            if collect_stride <= 0:
                message = f"collect_stride must be positive, got {collect_stride}."
                LOGGER.error(message)
                raise DiffusionConfigError(message)
            wanted = frozenset([int(self.num_timesteps)]).union(
                t for t in range(self.num_timesteps) if t % collect_stride == 0
            )
            trajectory = {
                int(self.num_timesteps): x_t.transpose(1, 2).detach().cpu().clone()
            }

        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (shape[0],),
                step,
                device=device,
                dtype=torch.int64,
            )
            x_t = self.p_sample(
                denoise_fn=denoise_fn,
                x_t=x_t,
                timesteps=timesteps,
                clip_denoised=clip_denoised,
            )
            if int(step) in wanted:
                trajectory[int(step)] = x_t.transpose(1, 2).detach().cpu().clone()
        return x_t, trajectory
