"""Evaluate sample payloads under TopoDiT-3D's metric definitions (CD side + JSD).

Reimplements, in pure PyTorch/NumPy, the exact math of
`TopoDiT-3D/metrics/evaluation_metrics.py`:
  - pairwise CD: squared nearest-neighbor distances, `dl.mean + dr.mean`
  - lgan MMD/COV from the pairwise matrix
  - 1-NNA via the GAN-metrics kNN classifier (k=1)
  - JSD on a 28^3 occupancy grid clipped to the unit sphere

EMD metrics are skipped: TopoDiT uses a compiled CUDA approxmatch kernel that
cannot be replicated exactly without their build environment.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch


def pairwise_squared_cd(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    device: torch.device,
    ref_chunk: int = 24,
) -> torch.Tensor:
    """Return the (Na, Nb) matrix of TopoDiT-style squared Chamfer distances."""
    result = torch.empty(a.shape[0], b.shape[0])
    for i in range(a.shape[0]):
        x = a[i].to(device)  # (P, 3)
        rows = []
        for j_start in range(0, b.shape[0], ref_chunk):
            y = b[j_start : j_start + ref_chunk].to(device)  # (C, P, 3)
            dists = torch.cdist(x.unsqueeze(0).expand(y.shape[0], -1, -1), y) ** 2
            rows.append(
                (
                    dists.min(dim=2).values.mean(dim=1)
                    + dists.min(dim=1).values.mean(dim=1)
                ).cpu()
            )
        result[i] = torch.cat(rows)
    return result


def knn_acc(mxx: torch.Tensor, mxy: torch.Tensor, myy: torch.Tensor) -> float:
    """1-NN two-sample classifier accuracy (GAN-metrics convention, k=1)."""
    n0, n1 = mxx.shape[0], myy.shape[0]
    label = torch.cat([torch.ones(n0), torch.zeros(n1)])
    m = torch.cat(
        [torch.cat([mxx, mxy], dim=1), torch.cat([mxy.t(), myy], dim=1)], dim=0
    )
    m = m + torch.diag(torch.full((n0 + n1,), float("inf")))
    idx = m.topk(1, dim=0, largest=False).indices[0]
    pred = label.index_select(0, idx) >= 0.5
    return float((pred.float() == label).float().mean())


def lgan_mmd_cov(all_dist: torch.Tensor) -> dict[str, float]:
    """MMD/COV from an (N_sample, N_ref) distance matrix."""
    min_from_sample, min_idx = all_dist.min(dim=1)
    min_per_ref = all_dist.min(dim=0).values
    return {
        "lgan_mmd": float(min_per_ref.mean()),
        "lgan_cov": float(min_idx.unique().numel()) / float(all_dist.shape[1]),
        "lgan_mmd_smp": float(min_from_sample.mean()),
    }


def entropy_of_occupancy_grid(
    clouds: np.ndarray, resolution: int, device: torch.device
) -> np.ndarray:
    """Return per-cell activation frequencies (TopoDiT `grid_var`)."""
    spacing = 1.0 / float(resolution - 1)
    grid = np.stack(
        np.meshgrid(*[np.arange(resolution) * spacing - 0.5] * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    keep = np.linalg.norm(grid, axis=1) <= 0.5
    grid = grid[keep]

    grid_t = torch.from_numpy(grid.astype(np.float32)).to(device)
    counters = np.zeros(grid.shape[0])
    for cloud in clouds:
        cloud_t = torch.from_numpy(cloud.astype(np.float32)).to(device)
        indices = torch.cdist(cloud_t, grid_t).argmin(dim=1).cpu().numpy()
        for cell in np.unique(indices):
            counters[cell] += 1
    return counters / float(len(clouds))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete JSD used by latent_3d_points (natural log entropy variant)."""
    p1 = p / np.sum(p)
    q1 = q / np.sum(q)
    m = 0.5 * (p1 + q1)

    def _entropy(x: np.ndarray) -> float:
        x = x[x > 0]
        return float(-np.sum(x * np.log(x)))

    return _entropy(m) - 0.5 * (_entropy(p1) + _entropy(q1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", required=True, help=".pt payload with samples/references"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ref-chunk", type=int, default=24)
    args = parser.parse_args()

    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    samples = payload["samples"].float()
    references = payload["references"].float()
    device = torch.device(args.device)
    print(f"samples {tuple(samples.shape)} references {tuple(references.shape)}")

    m_rs = pairwise_squared_cd(
        references, samples, device=device, ref_chunk=args.ref_chunk
    )
    m_rr = pairwise_squared_cd(
        references, references, device=device, ref_chunk=args.ref_chunk
    )
    m_ss = pairwise_squared_cd(
        samples, samples, device=device, ref_chunk=args.ref_chunk
    )

    res = {f"{k}-CD": v for k, v in lgan_mmd_cov(m_rs.t()).items()}
    res["1-NNA-CD"] = knn_acc(m_rr, m_rs, m_ss)
    res["JSD"] = jensen_shannon_divergence(
        entropy_of_occupancy_grid(samples.numpy(), 28, device),
        entropy_of_occupancy_grid(references.numpy(), 28, device),
    )
    for key, value in sorted(res.items()):
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
