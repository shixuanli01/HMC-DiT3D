"""Evaluation entrypoints for saved HMC-DiT3D sample payloads."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from hmc_dit3d.metrics.pointcloud import compute_minimal_pointcloud_metrics

LOGGER = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when saved-sample evaluation fails."""


def load_sample_payload(sample_path: str | Path) -> dict[str, Any]:
    """Load a saved sample payload from disk.

    Args:
        sample_path: `.pt` path produced by `train.sample`.

    Returns:
        dict[str, Any]: Loaded payload.
    """
    resolved_path = Path(sample_path).expanduser().resolve()
    if not resolved_path.exists():
        message = f"Sample payload does not exist: {resolved_path!s}."
        LOGGER.error(message)
        raise EvaluationError(message)
    try:
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(resolved_path, map_location="cpu")
    if not isinstance(payload, dict):
        message = "Sample payload must deserialize to a dictionary."
        LOGGER.error(message)
        raise EvaluationError(message)
    required_keys = {"samples", "references"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        message = f"Sample payload is missing required keys: {sorted(missing_keys)!r}."
        LOGGER.error(message)
        raise EvaluationError(message)
    return payload


def evaluate_saved_samples(
    sample_path: str | Path,
    *,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    jsd_resolution: int = 28,
    emd_point_count: int = 128,
    emd_sinkhorn_epsilon: float = 0.1,
    emd_sinkhorn_iterations: int = 50,
    emd_batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate a saved sample payload against its reference point clouds.

    Args:
        sample_path: `.pt` path produced by `train.sample`.
        batch_size: Inner-loop batch size used by pairwise CD evaluation.
        device: Target torch device.
        jsd_resolution: Occupancy-grid resolution used by JSD.
        emd_point_count: Per-cloud point count used for EMD approximation.
        emd_sinkhorn_epsilon: Entropic regularization for Sinkhorn EMD.
        emd_sinkhorn_iterations: Sinkhorn update iterations for EMD.
        emd_batch_size: Inner-loop batch size used by pairwise EMD evaluation.

    Returns:
        dict[str, float]: Minimal evaluation metrics.
    """
    payload = load_sample_payload(sample_path)
    metrics = compute_minimal_pointcloud_metrics(
        payload["samples"],
        payload["references"],
        batch_size=batch_size,
        device=device,
        jsd_resolution=jsd_resolution,
        emd_point_count=emd_point_count,
        emd_sinkhorn_epsilon=emd_sinkhorn_epsilon,
        emd_sinkhorn_iterations=emd_sinkhorn_iterations,
        emd_batch_size=emd_batch_size,
    )
    return {key: float(value) for key, value in metrics.items()}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for saved-sample evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved HMC-DiT3D sample payloads.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Saved .pt payload produced by train.sample.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Pairwise CD evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used during metric computation.",
    )
    parser.add_argument(
        "--jsd-resolution",
        type=int,
        default=28,
        help="Occupancy grid resolution used by JSD.",
    )
    parser.add_argument(
        "--emd-point-count",
        type=int,
        default=128,
        help="Per-cloud point count used for EMD approximation.",
    )
    parser.add_argument(
        "--emd-sinkhorn-epsilon",
        type=float,
        default=0.1,
        help="Entropic regularization for Sinkhorn EMD.",
    )
    parser.add_argument(
        "--emd-sinkhorn-iterations",
        type=int,
        default=50,
        help="Sinkhorn update iterations for EMD.",
    )
    parser.add_argument(
        "--emd-batch-size",
        type=int,
        default=8,
        help="Pairwise EMD evaluation batch size.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for saved-sample evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    metrics = evaluate_saved_samples(
        args.samples,
        batch_size=args.batch_size,
        device=args.device,
        jsd_resolution=args.jsd_resolution,
        emd_point_count=args.emd_point_count,
        emd_sinkhorn_epsilon=args.emd_sinkhorn_epsilon,
        emd_sinkhorn_iterations=args.emd_sinkhorn_iterations,
        emd_batch_size=args.emd_batch_size,
    )
    LOGGER.info(
        "Evaluation metrics:\n%s",
        json.dumps(metrics, indent=2, sort_keys=True),
    )


if __name__ == "__main__":
    main()
