"""Tests for minimal point-cloud evaluation metrics."""

from __future__ import annotations

import math

import torch

from hmc_dit3d.metrics.pointcloud import (
    chamfer_distance,
    compute_minimal_pointcloud_metrics,
    jsd_between_point_cloud_sets,
)


def test_chamfer_distance_is_zero_for_identical_clouds() -> None:
    """Chamfer distance should be zero for identical aligned point clouds."""
    point_clouds = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.3, 0.2]],
            [[-0.2, 0.0, 0.1], [0.1, -0.1, 0.0], [0.2, 0.2, -0.2]],
        ],
        dtype=torch.float32,
    )

    distances = chamfer_distance(point_clouds, point_clouds, device="cpu")

    assert torch.allclose(distances, torch.zeros_like(distances))


def test_chamfer_distance_matches_squared_distance_definition() -> None:
    """Chamfer distance should follow the squared-distance baseline convention."""
    sample = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    reference = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)

    distance = chamfer_distance(sample, reference, device="cpu")

    assert torch.allclose(distance, torch.tensor([8.0]))


def test_jsd_is_zero_for_identical_sets() -> None:
    """JSD should vanish when the two occupancy distributions match exactly."""
    point_clouds = torch.tensor(
        [
            [[-0.2, 0.0, 0.1], [0.1, -0.1, 0.0], [0.2, 0.2, -0.2]],
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.3, 0.2]],
        ],
        dtype=torch.float32,
    )

    jsd = jsd_between_point_cloud_sets(
        point_clouds.numpy(),
        point_clouds.numpy(),
        resolution=8,
    )

    assert jsd == 0.0


def test_compute_minimal_pointcloud_metrics_returns_expected_keys() -> None:
    """The minimal metric bundle should return finite comparison scalars."""
    references = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 0.3, 0.2]],
            [[-0.2, 0.0, 0.1], [0.1, -0.1, 0.0], [0.2, 0.2, -0.2]],
        ],
        dtype=torch.float32,
    )
    samples = references + 0.01

    metrics = compute_minimal_pointcloud_metrics(
        samples,
        references,
        batch_size=1,
        device="cpu",
        jsd_resolution=8,
    )

    assert set(metrics) == {
        "lgan_mmd-CD",
        "lgan_cov-CD",
        "lgan_mmd_smp-CD",
        "1-NN-CD-acc",
        "lgan_mmd-EMD",
        "lgan_cov-EMD",
        "lgan_mmd_smp-EMD",
        "1-NN-EMD-acc",
        "JSD",
    }
    for value in metrics.values():
        assert math.isfinite(value)
    assert metrics["lgan_mmd-CD"] >= 0.0
    assert metrics["lgan_cov-CD"] >= 0.0
    assert 0.0 <= metrics["1-NN-CD-acc"] <= 1.0
    assert metrics["lgan_mmd-EMD"] >= 0.0
    assert metrics["lgan_cov-EMD"] >= 0.0
    assert 0.0 <= metrics["1-NN-EMD-acc"] <= 1.0
