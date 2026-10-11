"""Train-bank latent priors for diverse HMC condition generation.

The condition VAE is trained with a standard-normal prior, but its aggregated
posterior can remain strongly multi-modal.  This module fits a lightweight
diagonal Gaussian mixture to posterior means from the training bank and
provides balanced sampling plus K-center candidate selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class LatentGMMError(RuntimeError):
    """Raised when fitting or sampling a latent mixture fails."""


@dataclass(slots=True, frozen=True)
class DiagonalGaussianMixture:
    """Parameters of a diagonal-covariance Gaussian mixture in raw latent space."""

    weights: Tensor
    means: Tensor
    variances: Tensor
    data_mean: Tensor
    data_std: Tensor
    iterations: int
    log_likelihood: float

    @property
    def num_components(self) -> int:
        """Return the number of mixture components."""
        return int(self.means.shape[0])

    @property
    def latent_dim(self) -> int:
        """Return the latent dimensionality."""
        return int(self.means.shape[1])


def _validate_latents(latents: Tensor) -> None:
    if latents.ndim != 2 or min(latents.shape) <= 0:
        raise LatentGMMError(
            f"latents must have non-empty shape (N, D), got {tuple(latents.shape)}."
        )
    if not torch.isfinite(latents).all():
        raise LatentGMMError("latents contain non-finite values.")


def _kmeans_plus_plus(
    data: Tensor,
    num_components: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Choose deterministic-seed K-means++ centers from standardized data."""
    num_samples = int(data.shape[0])
    first = int(
        torch.randint(
            num_samples,
            (1,),
            generator=generator,
            device=data.device,
        ).item()
    )
    chosen = [first]
    min_distance = (data - data[first]).square().sum(dim=1)
    for _ in range(1, num_components):
        total = min_distance.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            remaining = torch.ones(num_samples, dtype=torch.bool, device=data.device)
            remaining[torch.tensor(chosen, device=data.device)] = False
            candidates = remaining.nonzero(as_tuple=False).flatten()
            next_index = int(candidates[0].item())
        else:
            next_index = int(
                torch.multinomial(
                    min_distance / total,
                    1,
                    generator=generator,
                ).item()
            )
        chosen.append(next_index)
        distance = (data - data[next_index]).square().sum(dim=1)
        min_distance = torch.minimum(min_distance, distance)
    return data[torch.tensor(chosen, device=data.device)].clone()


def fit_diagonal_gmm(
    latents: Tensor,
    num_components: int,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    covariance_floor: float = 1e-3,
    seed: int = 0,
) -> DiagonalGaussianMixture:
    """Fit a diagonal GMM to latent vectors with EM.

    Fitting happens in globally standardized coordinates for numerical
    stability. Returned component parameters are transformed back to the raw
    VAE latent space, while ``data_mean`` and ``data_std`` retain the whitening
    transform used later by K-center selection.
    """
    _validate_latents(latents)
    num_samples, latent_dim = latents.shape
    if not 1 <= num_components <= num_samples:
        raise LatentGMMError(
            "num_components must be in [1, N], got "
            f"{num_components} for N={num_samples}."
        )
    if max_iterations <= 0:
        raise LatentGMMError("max_iterations must be positive.")
    if covariance_floor <= 0.0:
        raise LatentGMMError("covariance_floor must be positive.")

    data_mean = latents.mean(dim=0)
    data_std = latents.std(dim=0, unbiased=False).clamp_min(1e-6)
    data = (latents - data_mean) / data_std
    generator = torch.Generator(device=data.device).manual_seed(seed)
    means = _kmeans_plus_plus(
        data,
        num_components,
        generator=generator,
    )
    variances = torch.ones(
        num_components,
        latent_dim,
        dtype=data.dtype,
        device=data.device,
    )
    weights = torch.full(
        (num_components,),
        1.0 / num_components,
        dtype=data.dtype,
        device=data.device,
    )

    previous = None
    final_likelihood = float("-inf")
    completed_iterations = 0
    log_two_pi = math.log(2.0 * math.pi)
    for iteration in range(max_iterations):
        log_determinant = torch.log(variances).sum(dim=1)
        quadratic = (
            (data[:, None, :] - means[None, :, :]).square() / variances[None, :, :]
        ).sum(dim=2)
        log_probability = -0.5 * (
            latent_dim * log_two_pi + log_determinant[None, :] + quadratic
        )
        log_joint = log_probability + weights.clamp_min(1e-12).log()[None, :]
        log_normalizer = torch.logsumexp(log_joint, dim=1)
        responsibilities = torch.exp(log_joint - log_normalizer[:, None])
        likelihood = log_normalizer.mean()

        counts = responsibilities.sum(dim=0).clamp_min(1e-6)
        weights = counts / counts.sum()
        means = responsibilities.transpose(0, 1) @ data / counts[:, None]
        second_moment = (
            responsibilities.transpose(0, 1) @ data.square() / counts[:, None]
        )
        variances = (second_moment - means.square()).clamp_min(covariance_floor)

        completed_iterations = iteration + 1
        final_likelihood = float(likelihood.detach())
        if previous is not None and abs(final_likelihood - previous) <= tolerance:
            break
        previous = final_likelihood

    raw_means = means * data_std + data_mean
    raw_variances = variances * data_std.square()
    return DiagonalGaussianMixture(
        weights=weights.detach(),
        means=raw_means.detach(),
        variances=raw_variances.detach(),
        data_mean=data_mean.detach(),
        data_std=data_std.detach(),
        iterations=completed_iterations,
        log_likelihood=final_likelihood,
    )


def sample_diagonal_gmm(
    mixture: DiagonalGaussianMixture,
    num_samples: int,
    *,
    balanced: bool = True,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample latent vectors and return their component assignments."""
    if num_samples <= 0:
        raise LatentGMMError("num_samples must be positive.")
    if temperature < 0.0:
        raise LatentGMMError("temperature must be non-negative.")
    device = mixture.means.device
    components = mixture.num_components
    if balanced:
        repeats, remainder = divmod(num_samples, components)
        assignments = torch.arange(components, device=device).repeat_interleave(repeats)
        if remainder:
            order = torch.randperm(
                components,
                generator=generator,
                device=device,
            )
            assignments = torch.cat([assignments, order[:remainder]])
        shuffle = torch.randperm(
            num_samples,
            generator=generator,
            device=device,
        )
        assignments = assignments[shuffle]
    else:
        assignments = torch.multinomial(
            mixture.weights,
            num_samples,
            replacement=True,
            generator=generator,
        )
    noise = torch.randn(
        num_samples,
        mixture.latent_dim,
        dtype=mixture.means.dtype,
        device=device,
        generator=generator,
    )
    samples = mixture.means[assignments] + (
        float(temperature) * mixture.variances[assignments].sqrt() * noise
    )
    return samples, assignments


def assign_diagonal_gmm(
    mixture: DiagonalGaussianMixture,
    latents: Tensor,
) -> Tensor:
    """Assign latent vectors to their maximum-posterior mixture component."""
    _validate_latents(latents)
    if latents.shape[1] != mixture.latent_dim:
        raise LatentGMMError(
            f"Expected latent dim {mixture.latent_dim}, got {latents.shape[1]}."
        )
    log_determinant = torch.log(mixture.variances).sum(dim=1)
    quadratic = (
        (latents[:, None, :] - mixture.means[None, :, :]).square()
        / mixture.variances[None, :, :]
    ).sum(dim=2)
    log_probability = -0.5 * (
        mixture.latent_dim * math.log(2.0 * math.pi)
        + log_determinant[None, :]
        + quadratic
    )
    log_joint = log_probability + mixture.weights.clamp_min(1e-12).log()[None, :]
    return log_joint.argmax(dim=1)


def kcenter_select(
    candidates: Tensor,
    num_select: int,
    *,
    center: Tensor | None = None,
    scale: Tensor | None = None,
    initial_indices: Tensor | None = None,
) -> Tensor:
    """Select a diverse candidate subset with greedy farthest-first traversal."""
    _validate_latents(candidates)
    num_candidates = int(candidates.shape[0])
    if not 1 <= num_select <= num_candidates:
        raise LatentGMMError(
            f"num_select must be in [1, {num_candidates}], got {num_select}."
        )
    if center is None:
        center = candidates.mean(dim=0)
    if scale is None:
        scale = candidates.std(dim=0, unbiased=False)
    normalized = (candidates - center) / scale.clamp_min(1e-6)

    selected: list[int] = []
    if initial_indices is not None:
        for index in initial_indices.flatten().tolist():
            value = int(index)
            if not 0 <= value < num_candidates:
                raise LatentGMMError(f"initial index {value} is out of range.")
            if value not in selected:
                selected.append(value)
            if len(selected) == num_select:
                break
    if not selected:
        # Starting near the global center avoids using an extreme tail as the
        # anchor while remaining deterministic.
        selected.append(int(normalized.square().sum(dim=1).argmin().item()))

    selected_mask = torch.zeros(
        num_candidates,
        dtype=torch.bool,
        device=candidates.device,
    )
    selected_mask[torch.tensor(selected, device=candidates.device)] = True
    chosen = normalized[torch.tensor(selected, device=candidates.device)]
    minimum_distance = torch.cdist(normalized, chosen).square().amin(dim=1)
    minimum_distance[selected_mask] = -1.0
    while len(selected) < num_select:
        index = int(minimum_distance.argmax().item())
        selected.append(index)
        selected_mask[index] = True
        distance = (normalized - normalized[index]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected_mask] = -1.0
    return torch.tensor(selected, dtype=torch.int64, device=candidates.device)


def component_center_indices(
    candidates: Tensor,
    assignments: Tensor,
    mixture: DiagonalGaussianMixture,
) -> Tensor:
    """Return the candidate nearest each represented component mean."""
    if assignments.shape != (candidates.shape[0],):
        raise LatentGMMError("assignments must align with candidates.")
    selected: list[int] = []
    for component in range(mixture.num_components):
        indices = (assignments == component).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        scale = mixture.variances[component].sqrt().clamp_min(1e-6)
        distance = (
            ((candidates[indices] - mixture.means[component]) / scale)
            .square()
            .sum(dim=1)
        )
        selected.append(int(indices[distance.argmin()].item()))
    return torch.tensor(selected, dtype=torch.int64, device=candidates.device)
