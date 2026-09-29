"""Tests for TopoDiT batch-8 metric helpers."""

from __future__ import annotations

import torch

from hmc_dit3d.metrics.topodit_batch8 import coverage, knn_accuracy


def test_coverage_counts_unique_reference_matches() -> None:
    reference_to_sample = torch.tensor(
        [
            [0.0, 9.0, 9.0],
            [9.0, 0.0, 1.0],
            [8.0, 7.0, 6.0],
        ]
    )
    assert coverage(reference_to_sample) == 2.0 / 3.0


def test_knn_accuracy_is_one_for_separated_sets() -> None:
    within = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cross = torch.full((2, 2), 10.0)
    assert knn_accuracy(within, cross, within) == 1.0
