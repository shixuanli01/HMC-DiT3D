"""Minimal point-cloud metrics aligned with DiT-3D evaluation terminology."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

LOGGER = logging.getLogger(__name__)


class PointCloudMetricError(RuntimeError):
    """Raised when point-cloud metric inputs are invalid."""


def _as_point_cloud_tensor(
    point_clouds: Tensor | np.ndarray,
    *,
    device: torch.device | str,
) -> Tensor:
    """Convert point-cloud batches into validated float tensors.

    Args:
        point_clouds: Point clouds with shape `(B, N, 3)`.
        device: Target torch device.

    Returns:
        Tensor: Validated tensor on the target device.
    """
    tensor = torch.as_tensor(point_clouds, device=device, dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        message = (
            f"point_clouds must have shape (B, N, 3). Got {tuple(tensor.shape)!r}."
        )
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        message = "point_clouds must contain at least one cloud and one point."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if not torch.isfinite(tensor).all():
        message = "point_clouds contains NaN or Inf values."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    return tensor


def chamfer_distance(
    sample_pcs: Tensor | np.ndarray,
    ref_pcs: Tensor | np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute per-sample Chamfer distance for aligned point-cloud batches.

    Args:
        sample_pcs: Sampled point clouds with shape `(B, N, 3)`.
        ref_pcs: Reference point clouds with shape `(B, M, 3)`.
        device: Target torch device.

    Returns:
        Tensor: Chamfer distances with shape `(B,)`.
    """
    sample_tensor = _as_point_cloud_tensor(sample_pcs, device=device)
    ref_tensor = _as_point_cloud_tensor(ref_pcs, device=device)
    if sample_tensor.shape[0] != ref_tensor.shape[0]:
        message = (
            "sample_pcs and ref_pcs must share batch size. "
            f"Got {sample_tensor.shape[0]} and {ref_tensor.shape[0]}."
        )
        LOGGER.error(message)
        raise PointCloudMetricError(message)

    # Keep the squared-distance Chamfer definition used by the DiT-3D baseline.
    distances = torch.cdist(sample_tensor, ref_tensor).square()
    left = distances.min(dim=2).values.mean(dim=1)
    right = distances.min(dim=1).values.mean(dim=1)
    return left + right


def pairwise_chamfer_distance_matrix(
    sample_pcs: Tensor | np.ndarray,
    ref_pcs: Tensor | np.ndarray,
    *,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute the full pairwise Chamfer matrix between two point-cloud sets.

    Args:
        sample_pcs: Sampled point clouds with shape `(S, N, 3)`.
        ref_pcs: Reference point clouds with shape `(R, M, 3)`.
        batch_size: Number of reference clouds processed per inner loop.
        device: Target torch device.

    Returns:
        Tensor: Pairwise Chamfer matrix with shape `(S, R)` on CPU.
    """
    if batch_size <= 0:
        message = f"batch_size must be positive, got {batch_size}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)

    sample_tensor = _as_point_cloud_tensor(sample_pcs, device="cpu")
    ref_tensor = _as_point_cloud_tensor(ref_pcs, device="cpu")
    all_rows: list[Tensor] = []

    # Follow the baseline strategy (per sample x small reference batches) to reduce
    # peak memory from N^2 distance matrices.
    for sample_index in range(sample_tensor.shape[0]):
        sample_cloud = sample_tensor[sample_index].to(device)
        row_chunks: list[Tensor] = []
        for ref_start in range(0, ref_tensor.shape[0], batch_size):
            ref_batch = ref_tensor[ref_start : ref_start + batch_size].to(device)
            expanded_sample = sample_cloud.unsqueeze(0).expand(
                ref_batch.shape[0],
                -1,
                -1,
            )
            chunk = chamfer_distance(expanded_sample, ref_batch, device=device).cpu()
            row_chunks.append(chunk)
        all_rows.append(torch.cat(row_chunks, dim=0).unsqueeze(0))
    return torch.cat(all_rows, dim=0)


def _downsample_point_clouds(point_clouds: Tensor, target_count: int) -> Tensor:
    """Deterministically downsample point clouds to a fixed point count."""
    if target_count <= 0:
        message = f"target_count must be positive, got {target_count}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    point_count = point_clouds.shape[1]
    if point_count <= target_count:
        return point_clouds
    # Deterministic linspace indexing keeps runs reproducible across devices.
    indices = torch.linspace(
        0,
        point_count - 1,
        steps=target_count,
        dtype=torch.int64,
    )
    return point_clouds.index_select(dim=1, index=indices)


def _sinkhorn_emd_distance(
    sample_batch: Tensor,
    ref_batch: Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> Tensor:
    """Approximate EMD for aligned batches using Sinkhorn transport."""
    if epsilon <= 0.0:
        message = f"epsilon must be positive, got {epsilon}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if iterations <= 0:
        message = f"iterations must be positive, got {iterations}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if sample_batch.shape[0] != ref_batch.shape[0]:
        message = (
            "sample_batch and ref_batch must share batch size. "
            f"Got {sample_batch.shape[0]} and {ref_batch.shape[0]}."
        )
        LOGGER.error(message)
        raise PointCloudMetricError(message)

    pairwise_cost = torch.cdist(sample_batch, ref_batch, p=2)
    kernel = torch.exp(-pairwise_cost / epsilon).clamp_min(1.0e-12)
    batch_size, sample_count, ref_count = kernel.shape
    sample_mass = torch.full(
        (batch_size, sample_count),
        fill_value=1.0 / float(sample_count),
        dtype=pairwise_cost.dtype,
        device=pairwise_cost.device,
    )
    ref_mass = torch.full(
        (batch_size, ref_count),
        fill_value=1.0 / float(ref_count),
        dtype=pairwise_cost.dtype,
        device=pairwise_cost.device,
    )
    u = sample_mass.clone()
    v = ref_mass.clone()
    for _ in range(iterations):
        kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1.0e-12)
        u = sample_mass / kv
        ktu = (
            torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1))
            .squeeze(-1)
            .clamp_min(1.0e-12)
        )
        v = ref_mass / ktu
    transport = u.unsqueeze(-1) * kernel * v.unsqueeze(1)
    return (transport * pairwise_cost).sum(dim=(1, 2))


def pairwise_emd_distance_matrix(
    sample_pcs: Tensor | np.ndarray,
    ref_pcs: Tensor | np.ndarray,
    *,
    point_count: int = 128,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iterations: int = 50,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute a pairwise approximate EMD matrix with Sinkhorn transport.

    The original DiT-3D code relies on a custom CUDA EMD operator. To keep this
    project portable on Python 3.12 / PyTorch 2.x, we compute EMD-style metrics
    with a deterministic Sinkhorn approximation on downsampled point clouds.
    """
    if batch_size <= 0:
        message = f"batch_size must be positive, got {batch_size}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)

    sample_tensor = _as_point_cloud_tensor(sample_pcs, device="cpu")
    ref_tensor = _as_point_cloud_tensor(ref_pcs, device="cpu")
    sample_tensor = _downsample_point_clouds(sample_tensor, target_count=point_count)
    ref_tensor = _downsample_point_clouds(ref_tensor, target_count=point_count)
    all_rows: list[Tensor] = []

    for sample_index in range(sample_tensor.shape[0]):
        sample_cloud = sample_tensor[sample_index].to(device)
        row_chunks: list[Tensor] = []
        for ref_start in range(0, ref_tensor.shape[0], batch_size):
            ref_batch = ref_tensor[ref_start : ref_start + batch_size].to(device)
            expanded_sample = sample_cloud.unsqueeze(0).expand(
                ref_batch.shape[0],
                -1,
                -1,
            )
            chunk = _sinkhorn_emd_distance(
                expanded_sample,
                ref_batch,
                epsilon=sinkhorn_epsilon,
                iterations=sinkhorn_iterations,
            ).cpu()
            row_chunks.append(chunk)
        all_rows.append(torch.cat(row_chunks, dim=0).unsqueeze(0))
    return torch.cat(all_rows, dim=0)


def _lgan_mmd_cov(all_dist: Tensor) -> dict[str, float]:
    """Compute MMD/COV metrics from a pairwise distance matrix."""
    min_val_from_sample, min_index = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    return {
        "lgan_mmd-CD": float(min_val.mean().item()),
        "lgan_cov-CD": float(min_index.unique().numel() / all_dist.shape[1]),
        "lgan_mmd_smp-CD": float(min_val_from_sample.mean().item()),
    }


def _one_nn_accuracy(
    ref_ref: Tensor,
    ref_sample: Tensor,
    sample_sample: Tensor,
) -> float:
    """Compute 1-NN accuracy from block distance matrices."""
    ref_count = ref_ref.shape[0]
    sample_count = sample_sample.shape[0]
    labels = torch.cat(
        [
            torch.ones(ref_count, dtype=torch.int64),
            torch.zeros(sample_count, dtype=torch.int64),
        ]
    )
    combined = torch.cat(
        [
            torch.cat([ref_ref, ref_sample], dim=1),
            torch.cat([ref_sample.transpose(0, 1), sample_sample], dim=1),
        ],
        dim=0,
    )
    combined.fill_diagonal_(torch.inf)
    nearest = combined.argmin(dim=1)
    predictions = labels[nearest]
    return float((predictions == labels).float().mean().item())


def _as_numpy_point_clouds(point_clouds: Tensor | np.ndarray) -> np.ndarray:
    """Convert point-cloud batches into validated numpy arrays."""
    array = np.asarray(point_clouds, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        message = f"point_clouds must have shape (B, N, 3). Got {array.shape!r}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        message = "point_clouds must contain at least one cloud and one point."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if not np.isfinite(array).all():
        message = "point_clouds contains NaN or Inf values."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    return array


def _occupancy_counts(point_clouds: np.ndarray, resolution: int) -> np.ndarray:
    """Estimate occupancy-grid counts used for JSD evaluation.

    Args:
        point_clouds: Point clouds with shape `(B, N, 3)`.
        resolution: Occupancy grid resolution.

    Returns:
        np.ndarray: Flattened occupancy counts with shape `(resolution**3,)`.
    """
    if resolution <= 1:
        message = f"resolution must be greater than 1, got {resolution}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)

    clipped = np.clip(point_clouds, -0.5, 0.5)
    indices = np.rint((clipped + 0.5) * (resolution - 1)).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)
    flat_indices = (
        indices[..., 0] * resolution * resolution
        + indices[..., 1] * resolution
        + indices[..., 2]
    )
    counters = np.zeros(resolution**3, dtype=np.float64)
    for cloud_indices in flat_indices:
        counters += np.bincount(cloud_indices, minlength=resolution**3)
    return counters


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence with log base 2."""
    if p.shape != q.shape:
        message = f"p and q must share shape, got {p.shape!r} and {q.shape!r}."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    if np.any(p < 0) or np.any(q < 0):
        message = "p and q must be non-negative."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0.0 or q_sum <= 0.0:
        message = "p and q must have positive total mass."
        LOGGER.error(message)
        raise PointCloudMetricError(message)
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0.0) & (b > 0.0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def jsd_between_point_cloud_sets(
    sample_pcs: Tensor | np.ndarray,
    ref_pcs: Tensor | np.ndarray,
    *,
    resolution: int = 28,
) -> float:
    """Compute occupancy-grid JSD between two point-cloud sets."""
    sample_array = _as_numpy_point_clouds(sample_pcs)
    ref_array = _as_numpy_point_clouds(ref_pcs)
    sample_counts = _occupancy_counts(sample_array, resolution)
    ref_counts = _occupancy_counts(ref_array, resolution)
    return jensen_shannon_divergence(sample_counts, ref_counts)


def compute_minimal_pointcloud_metrics(
    sample_pcs: Tensor | np.ndarray,
    ref_pcs: Tensor | np.ndarray,
    *,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    jsd_resolution: int = 28,
    emd_point_count: int = 128,
    emd_sinkhorn_epsilon: float = 0.1,
    emd_sinkhorn_iterations: int = 50,
    emd_batch_size: int = 8,
) -> dict[str, Any]:
    """Compute the minimal metric set needed for early HMC-DiT3D comparisons.

    The current implementation focuses on metrics that are stable in the new
    Python 3.12 / PyTorch 2.x stack without relying on legacy CUDA extensions.

    Args:
        sample_pcs: Sampled point clouds with shape `(S, N, 3)`.
        ref_pcs: Reference point clouds with shape `(R, N, 3)`.
        batch_size: Inner-loop batch size for pairwise CD evaluation.
        device: Target torch device.
        jsd_resolution: Occupancy-grid resolution used by JSD.
        emd_point_count: Per-cloud point count used for EMD approximation.
        emd_sinkhorn_epsilon: Entropic regularization for Sinkhorn EMD.
        emd_sinkhorn_iterations: Sinkhorn update iterations for EMD.
        emd_batch_size: Inner-loop batch size for pairwise EMD evaluation.

    Returns:
        dict[str, Any]: Metric dictionary aligned to baseline naming where possible.
    """
    sample_tensor = _as_point_cloud_tensor(sample_pcs, device="cpu")
    ref_tensor = _as_point_cloud_tensor(ref_pcs, device="cpu")
    ref_sample_cd = pairwise_chamfer_distance_matrix(
        ref_tensor,
        sample_tensor,
        batch_size=batch_size,
        device=device,
    )
    ref_ref_cd = pairwise_chamfer_distance_matrix(
        ref_tensor,
        ref_tensor,
        batch_size=batch_size,
        device=device,
    )
    sample_sample_cd = pairwise_chamfer_distance_matrix(
        sample_tensor,
        sample_tensor,
        batch_size=batch_size,
        device=device,
    )
    ref_sample_emd = pairwise_emd_distance_matrix(
        ref_tensor,
        sample_tensor,
        point_count=emd_point_count,
        sinkhorn_epsilon=emd_sinkhorn_epsilon,
        sinkhorn_iterations=emd_sinkhorn_iterations,
        batch_size=emd_batch_size,
        device=device,
    )
    ref_ref_emd = pairwise_emd_distance_matrix(
        ref_tensor,
        ref_tensor,
        point_count=emd_point_count,
        sinkhorn_epsilon=emd_sinkhorn_epsilon,
        sinkhorn_iterations=emd_sinkhorn_iterations,
        batch_size=emd_batch_size,
        device=device,
    )
    sample_sample_emd = pairwise_emd_distance_matrix(
        sample_tensor,
        sample_tensor,
        point_count=emd_point_count,
        sinkhorn_epsilon=emd_sinkhorn_epsilon,
        sinkhorn_iterations=emd_sinkhorn_iterations,
        batch_size=emd_batch_size,
        device=device,
    )

    results = _lgan_mmd_cov(ref_sample_cd.transpose(0, 1))
    results["1-NN-CD-acc"] = _one_nn_accuracy(
        ref_ref_cd,
        ref_sample_cd,
        sample_sample_cd,
    )
    emd_results = _lgan_mmd_cov(ref_sample_emd.transpose(0, 1))
    results["lgan_mmd-EMD"] = emd_results["lgan_mmd-CD"]
    results["lgan_cov-EMD"] = emd_results["lgan_cov-CD"]
    results["lgan_mmd_smp-EMD"] = emd_results["lgan_mmd_smp-CD"]
    results["1-NN-EMD-acc"] = _one_nn_accuracy(
        ref_ref_emd,
        ref_sample_emd,
        sample_sample_emd,
    )
    results["JSD"] = jsd_between_point_cloud_sets(
        sample_tensor.detach().numpy(),
        ref_tensor.detach().numpy(),
        resolution=jsd_resolution,
    )
    return results
