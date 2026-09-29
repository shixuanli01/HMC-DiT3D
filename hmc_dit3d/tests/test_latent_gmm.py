"""Tests for the train-bank latent GMM prior."""

from __future__ import annotations

import torch

from hmc_dit3d.hmc.latent_gmm import (
    assign_diagonal_gmm,
    component_center_indices,
    fit_diagonal_gmm,
    kcenter_select,
    sample_diagonal_gmm,
)


def test_gmm_fit_and_balanced_sampling_are_finite() -> None:
    """A two-mode latent cloud should yield finite, balanced prior samples."""
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(128, 4, generator=generator) * 0.2 - 2.0
    right = torch.randn(128, 4, generator=generator) * 0.3 + 2.0
    mixture = fit_diagonal_gmm(
        torch.cat([left, right]),
        2,
        max_iterations=50,
        seed=5,
    )
    samples, assignments = sample_diagonal_gmm(
        mixture,
        20,
        balanced=True,
        generator=torch.Generator().manual_seed(7),
    )

    assert samples.shape == (20, 4)
    assert torch.isfinite(samples).all()
    assert assignments.bincount(minlength=2).tolist() == [10, 10]
    assert torch.sort(mixture.means[:, 0]).values.tolist()[0] < -1.5
    assert torch.sort(mixture.means[:, 0]).values.tolist()[1] > 1.5
    assigned = assign_diagonal_gmm(mixture, torch.stack([left.mean(0), right.mean(0)]))
    assert assigned.unique().numel() == 2


def test_kcenter_preserves_component_anchors_and_spreads_points() -> None:
    """K-center should retain component anchors and add distant candidates."""
    training = torch.tensor(
        [[-2.0, 0.0], [-1.8, 0.1], [0.0, 0.0], [0.2, -0.1], [2.0, 0.0], [2.2, 0.1]]
    )
    mixture = fit_diagonal_gmm(training, 3, max_iterations=20, seed=3)
    candidates, assignments = sample_diagonal_gmm(
        mixture,
        12,
        balanced=True,
        temperature=0.5,
        generator=torch.Generator().manual_seed(13),
    )
    anchors = component_center_indices(candidates, assignments, mixture)
    selected = kcenter_select(
        candidates,
        4,
        center=mixture.data_mean,
        scale=mixture.data_std,
        initial_indices=anchors,
    )

    assert selected.unique().numel() == 4
    assert set(anchors.tolist()).issubset(set(selected.tolist()))
    assert assignments[selected].unique().numel() == 3
