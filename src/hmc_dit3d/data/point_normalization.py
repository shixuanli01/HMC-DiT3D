"""Global point-cloud normalization statistics (PVD / TopoDiT-3D convention).

The released HMC-DiT3D checkpoints were trained on raw ShapeNetCore.v2.PC15k
coordinates whose global standard deviation is about 0.16, while the linear
beta schedule assumes unit-variance data. This module computes the per-axis
global mean and the scalar global standard deviation of the train split, so
the denoiser can be trained on ``(x - mean) / std`` and samples mapped back
with ``x * std + mean``. A scalar std keeps aspect ratios, so bbox-normalized
HMC conditions, banks, and condition VAEs are unaffected.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID

LOGGER = logging.getLogger(__name__)


class PointNormalizationError(RuntimeError):
    """Raised when normalization statistics are invalid or inconsistent."""


@dataclass(frozen=True, slots=True)
class PointNormalization:
    """Global per-axis mean and scalar std used to whiten point clouds.

    Args:
        mean: Per-axis global mean with shape ``(3,)``.
        std: Scalar global standard deviation over all coordinates.
        source: Free-form description of how the statistics were computed.
    """

    mean: tuple[float, float, float]
    std: float
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.mean) != 3 or not all(np.isfinite(self.mean)):
            message = f"point normalization mean must be 3 finite values: {self.mean!r}"
            LOGGER.error(message)
            raise PointNormalizationError(message)
        if not np.isfinite(self.std) or self.std <= 0.0:
            message = f"point normalization std must be positive: {self.std!r}"
            LOGGER.error(message)
            raise PointNormalizationError(message)

    def _mean_tensor(self, like: Tensor) -> Tensor:
        shape = [1] * like.ndim
        # Points are stored as (B, N, 3) or (B, 3, N); place the axis of size 3.
        axis = like.ndim - 1 if like.shape[-1] == 3 else 1
        shape[axis] = 3
        return torch.tensor(self.mean, dtype=like.dtype, device=like.device).view(shape)

    def normalize(self, points: Tensor) -> Tensor:
        """Map raw coordinates to the whitened frame ``(x - mean) / std``."""
        return (points - self._mean_tensor(points)) / float(self.std)

    def denormalize(self, points: Tensor) -> Tensor:
        """Map whitened coordinates back to raw ShapeNet coordinates."""
        return points * float(self.std) + self._mean_tensor(points)

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a checkpoint-friendly dictionary."""
        return {
            "mean": [float(v) for v in self.mean],
            "std": float(self.std),
            "source": str(self.source),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PointNormalization | None:
        """Rebuild from ``to_dict`` output; ``None`` when nothing is stored."""
        if payload is None:
            return None
        if (
            not isinstance(payload, dict)
            or "mean" not in payload
            or "std" not in payload
        ):
            message = f"Invalid point normalization payload: {payload!r}"
            LOGGER.error(message)
            raise PointNormalizationError(message)
        mean = tuple(float(v) for v in payload["mean"])
        return cls(
            mean=mean, std=float(payload["std"]), source=str(payload.get("source", ""))
        )  # type: ignore[arg-type]

    def is_close(self, other: PointNormalization, *, rtol: float = 1.0e-4) -> bool:
        """Whether two statistics agree up to a relative tolerance."""
        return bool(
            np.allclose(self.mean, other.mean, rtol=rtol, atol=1.0e-6)
            and np.isclose(self.std, other.std, rtol=rtol)
        )


def iter_split_files(
    root_dir: str | Path, categories: Iterable[str], split: str
) -> list[Path]:
    """List the ``.npy`` files of ``split`` for ``categories`` in sorted order."""
    root = Path(root_dir).expanduser().resolve()
    files: list[Path] = []
    for category in categories:
        synset_id = CATEGORY_TO_SYNSETID[category]
        split_dir = root / synset_id / split
        if not split_dir.is_dir():
            message = f"Split directory does not exist: {split_dir!s}"
            LOGGER.error(message)
            raise PointNormalizationError(message)
        files.extend(sorted(split_dir.glob("*.npy")))
    if not files:
        message = (
            f"No point-cloud files found for {list(categories)!r} / {split!r} "
            f"under {root!s}"
        )
        LOGGER.error(message)
        raise PointNormalizationError(message)
    return files


def compute_point_normalization(
    root_dir: str | Path,
    categories: Iterable[str],
    *,
    split: str = "train",
) -> PointNormalization:
    """Compute global per-axis mean and scalar std over every point of ``split``.

    All stored points of each shape (15k in ShapeNetCore.v2.PC15k) are used,
    matching PVD's ``normalize across the dataset`` branch. Accumulation is
    done in float64 so the result does not depend on file order.
    """
    categories = tuple(categories)
    files = iter_split_files(root_dir, categories, split)
    count = 0
    axis_sum = np.zeros(3, dtype=np.float64)
    axis_sq_sum = np.zeros(3, dtype=np.float64)
    for path in files:
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] != 3:
            message = (
                f"Point cloud must have shape (N, 3): {path!s} -> {points.shape!r}"
            )
            LOGGER.error(message)
            raise PointNormalizationError(message)
        points = points.astype(np.float64, copy=False)
        count += points.shape[0]
        axis_sum += points.sum(axis=0)
        axis_sq_sum += np.square(points).sum(axis=0)
    mean = axis_sum / count
    # PVD convention: std of the flattened coordinate array, i.e. pooled over
    # the 3 axes with the pooled (not per-axis) mean.
    pooled_mean = axis_sum.sum() / (3 * count)
    total_var = (axis_sq_sum.sum() / (3 * count)) - pooled_mean**2
    std = float(np.sqrt(max(total_var, 0.0)))
    stats = PointNormalization(
        mean=(float(mean[0]), float(mean[1]), float(mean[2])),
        std=std,
        source=f"{split}:{','.join(categories)}:{len(files)} shapes:{count} points",
    )
    LOGGER.info(
        "Point normalization from %s: mean=%s std=%.6f",
        stats.source,
        [round(v, 6) for v in stats.mean],
        stats.std,
    )
    return stats
