"""Hilbert-Multifractal feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from hmc_dit3d.hmc.config import HMCComputationError, HMCConfig, NormalizationMode

LOGGER = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True, frozen=True)
class HMCScaleSummary:
    """Per-scale HMC statistics.

    Args:
        scale: Dyadic scale `s`.
        resolution: Voxel grid resolution `2**s`.
        epsilon: Effective voxel side length.
        measures: Voxel measure tensor with shape `(R, R, R)`.
        sequence: Hilbert-serialized measure sequence with shape `(R**3,)`.
        partition_values: Partition function values for all q-orders.
    """

    scale: int
    resolution: int
    epsilon: float
    measures: FloatArray
    sequence: FloatArray
    partition_values: FloatArray


@dataclass(slots=True, frozen=True)
class HMCResult:
    """Final HMC extraction result for one point cloud.

    Args:
        normalized_points: Normalized point cloud in `[0, 1]^3`.
        scales: Per-scale summaries.
        q_orders: q-orders used in multifractal statistics.
        tau: Estimated mass exponents.
        generalized_dimensions: Generalized dimensions, with `D(1)` replaced by
            the information dimension estimate when `q=1`.
        information_dimension: Estimated `D(1)`.
        alpha: Optional spectrum abscissa.
        spectrum: Optional `f(alpha)` features.
        descriptor: Final global descriptor fed to the HMC condition encoder.
    """

    normalized_points: FloatArray
    scales: tuple[HMCScaleSummary, ...]
    q_orders: FloatArray
    tau: FloatArray
    generalized_dimensions: FloatArray
    information_dimension: float
    alpha: FloatArray | None
    spectrum: FloatArray | None
    descriptor: FloatArray

    @property
    def sequences(self) -> tuple[FloatArray, ...]:
        """Return Hilbert sequences ordered by scale."""
        return tuple(scale.sequence for scale in self.scales)


class HMCFeatureExtractor:
    """Extract HMC descriptors from point clouds."""

    def __init__(self, config: HMCConfig) -> None:
        """Create an extractor with cached Hilbert permutations.

        Args:
            config: HMC extraction settings.
        """
        self.config = config
        self._hilbert_permutations: dict[int, IntArray] = {}

    def extract(self, points: NDArray[np.floating[Any]] | torch.Tensor) -> HMCResult:
        """Extract HMC features from a point cloud.

        Args:
            points: Input point cloud with shape `(N, 3)`.

        Returns:
            HMCResult: Computed HMC features and descriptors.
        """
        normalized_points = self._normalize_points(self._as_numpy_points(points))
        scale_summaries = tuple(
            self._extract_single_scale(normalized_points, scale)
            for scale in self.config.scales
        )
        partition_matrix = np.stack(
            [scale.partition_values for scale in scale_summaries],
            axis=0,
        )
        log_epsilons = np.log(
            np.asarray([scale.epsilon for scale in scale_summaries], dtype=np.float64)
        )
        # `tau` and D(1) are fits across *all* scales; omitting a scale would break
        # the multifractal / entropy slopes. (Neural HMC may only use the finest
        # Hilbert sequence for token conditioning.)
        tau = self._estimate_tau(log_epsilons, partition_matrix)
        generalized_dimensions = self._estimate_generalized_dimensions(tau)
        information_dimension = self._estimate_information_dimension(scale_summaries)
        q_orders = np.asarray(self.config.q_orders, dtype=np.float64)
        q_is_one = np.isclose(q_orders, 1.0)
        if q_is_one.any():
            generalized_dimensions[q_is_one] = information_dimension

        alpha: FloatArray | None = None
        spectrum: FloatArray | None = None
        if self.config.use_spectrum:
            alpha, spectrum = self._estimate_spectrum(q_orders, tau)

        descriptor_parts = [generalized_dimensions.astype(np.float64, copy=False)]
        if spectrum is not None:
            descriptor_parts.append(spectrum.astype(np.float64, copy=False))
        descriptor = np.concatenate(descriptor_parts, axis=0)

        return HMCResult(
            normalized_points=normalized_points,
            scales=scale_summaries,
            q_orders=q_orders,
            tau=tau,
            generalized_dimensions=generalized_dimensions,
            information_dimension=float(information_dimension),
            alpha=alpha,
            spectrum=spectrum,
            descriptor=descriptor,
        )

    def _extract_single_scale(
        self,
        normalized_points: FloatArray,
        scale: int,
    ) -> HMCScaleSummary:
        """Extract HMC statistics for one dyadic scale.

        Args:
            normalized_points: Point cloud already mapped to `[0, 1]^3`.
            scale: Dyadic scale `s`.

        Returns:
            HMCScaleSummary: Per-scale measurements.
        """
        resolution = 2**scale
        epsilon = 1.0 / float(resolution)
        measures = self._voxelize(normalized_points, resolution)
        sequence = self._hilbert_serialize(measures, scale)
        partition_values = np.asarray(
            [self._partition_value(sequence, q) for q in self.config.q_orders],
            dtype=np.float64,
        )
        return HMCScaleSummary(
            scale=scale,
            resolution=resolution,
            epsilon=epsilon,
            measures=measures,
            sequence=sequence,
            partition_values=partition_values,
        )

    def _as_numpy_points(
        self,
        points: NDArray[np.floating[Any]] | torch.Tensor,
    ) -> FloatArray:
        """Convert the input point cloud to a finite float64 numpy array.

        Args:
            points: Point cloud with shape `(N, 3)`.

        Returns:
            FloatArray: Numpy point cloud.
        """
        if isinstance(points, torch.Tensor):
            points_np = points.detach().cpu().numpy()
        else:
            points_np = np.asarray(points)

        if points_np.ndim != 2 or points_np.shape[1] != 3:
            message = f"Expected point cloud shape (N, 3), got {points_np.shape!r}."
            LOGGER.error(message)
            raise HMCComputationError(message)
        if points_np.shape[0] == 0:
            message = "Point cloud is empty."
            LOGGER.error(message)
            raise HMCComputationError(message)
        if not np.isfinite(points_np).all():
            message = "Point cloud contains NaN or Inf values."
            LOGGER.error(message)
            raise HMCComputationError(message)
        return points_np.astype(np.float64, copy=False)

    def _normalize_points(self, points: FloatArray) -> FloatArray:
        """Normalize a point cloud into `[0, 1]^3`.

        Args:
            points: Input point cloud with shape `(N, 3)`.

        Returns:
            FloatArray: Normalized point cloud.
        """
        mode = self.config.normalization_mode
        if mode == NormalizationMode.MIN_MAX:
            mins = points.min(axis=0)
            spans = points.max(axis=0) - mins
            if np.any(spans <= 0.0):
                message = "min_max normalization received a degenerate axis span."
                LOGGER.error(message)
                raise HMCComputationError(message)
            normalized = (points - mins) / spans
        elif mode == NormalizationMode.BBOX:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            center = (mins + maxs) * 0.5
            edge = float(np.max(maxs - mins))
            if edge <= 0.0:
                message = "bbox normalization received a degenerate bounding box."
                LOGGER.error(message)
                raise HMCComputationError(message)
            normalized = (points - center) / edge + 0.5
        elif mode == NormalizationMode.UNIT_SPHERE:
            center = points.mean(axis=0)
            centered = points - center
            radius = float(np.max(np.linalg.norm(centered, axis=1)))
            if radius <= 0.0:
                message = "unit_sphere normalization received zero radius."
                LOGGER.error(message)
                raise HMCComputationError(message)
            normalized = centered / (2.0 * radius) + 0.5
        else:
            message = f"Unsupported normalization mode: {mode!s}."
            LOGGER.error(message)
            raise HMCComputationError(message)

        # Key detail: Hilbert ordering and voxelization assume points are in
        # `[0, 1)`; otherwise boundary points can spill into the next cell.
        return np.clip(normalized, 0.0, np.nextafter(1.0, 0.0))

    def _voxelize(self, points: FloatArray, resolution: int) -> FloatArray:
        """Build a voxel measure tensor from normalized points.

        Args:
            points: Normalized point cloud in `[0, 1)^3`.
            resolution: Voxel resolution.

        Returns:
            FloatArray: Measure tensor with shape `(R, R, R)`.
        """
        indices = np.floor(points * resolution).astype(np.int64)
        indices = np.clip(indices, 0, resolution - 1)
        flat_indices = np.ravel_multi_index(indices.T, dims=(resolution,) * 3)
        counts = np.bincount(flat_indices, minlength=resolution**3).astype(np.float64)
        measures = counts / (counts.sum() + self.config.delta)
        return measures.reshape(resolution, resolution, resolution)

    def _hilbert_serialize(self, measures: FloatArray, scale: int) -> FloatArray:
        """Serialize a voxel tensor into Hilbert order.

        Args:
            measures: Measure tensor with shape `(R, R, R)`.
            scale: Dyadic scale `s`.

        Returns:
            FloatArray: Hilbert sequence with shape `(R**3,)`.
        """
        permutation = self._hilbert_permutation(scale)
        flat_measures = measures.reshape(-1)
        return flat_measures[permutation]

    def _hilbert_permutation(self, scale: int) -> IntArray:
        """Return the linear-index permutation induced by a 3D Hilbert curve.

        Args:
            scale: Dyadic scale `s`.

        Returns:
            IntArray: Flat indices ordered by Hilbert distance.
        """
        if scale in self._hilbert_permutations:
            return self._hilbert_permutations[scale]

        try:
            from hilbertcurve.hilbertcurve import HilbertCurve
        except ImportError as exc:
            message = (
                "hilbertcurve is required for HMC serialization. "
                "Install it with `python -m pip install -e .[dev]`."
            )
            LOGGER.error(message)
            raise HMCComputationError(message) from exc

        resolution = 2**scale
        curve = HilbertCurve(scale, 3)
        total_voxels = resolution**3
        if total_voxels >= 2**22:
            LOGGER.warning(
                "Large Hilbert permutation requested at scale=%d (voxels=%d). "
                "Consider offline caching for production experiments.",
                scale,
                total_voxels,
            )
        permutation = np.fromiter(
            (
                int(
                    np.ravel_multi_index(
                        tuple(curve.point_from_distance(distance)),
                        dims=(resolution,) * 3,
                    )
                )
                for distance in range(total_voxels)
            ),
            dtype=np.int64,
            count=total_voxels,
        )
        self._hilbert_permutations[scale] = permutation
        return permutation

    def _partition_value(self, sequence: FloatArray, q: float) -> float:
        """Compute one partition function value `Z_s(q)`.

        Args:
            sequence: Hilbert sequence for one scale.
            q: Moment order.

        Returns:
            float: Partition function value.
        """
        occupied = sequence[sequence > 0.0]
        if occupied.size == 0:
            message = "Encountered an empty measure sequence."
            LOGGER.error(message)
            raise HMCComputationError(message)
        if np.isclose(q, 0.0):
            return float(occupied.size)
        if q < 0.0:
            # For finite point clouds, negative moments including empty cells can
            # collapse into stabilizer-dominated values. Compute on non-empty
            # support only to avoid epsilon-driven artifacts.
            return float(np.power(occupied, q).sum())
        return float(np.power(occupied, q).sum())

    def _estimate_tau(
        self,
        log_epsilons: FloatArray,
        partition_matrix: FloatArray,
    ) -> FloatArray:
        """Estimate `tau(q)` from log-log slopes across scales.

        Args:
            log_epsilons: `log(epsilon_s)` with shape `(S,)`.
            partition_matrix: Partition values with shape `(S, |Q|)`.

        Returns:
            FloatArray: `tau(q)` values with shape `(|Q|,)`.
        """
        safe_partition = np.clip(
            partition_matrix,
            self.config.empty_box_epsilon,
            None,
        )
        tau = np.empty(safe_partition.shape[1], dtype=np.float64)
        for q_index in range(safe_partition.shape[1]):
            tau[q_index] = np.polyfit(
                log_epsilons,
                np.log(safe_partition[:, q_index]),
                deg=1,
            )[0]
        return tau

    def _estimate_generalized_dimensions(self, tau: FloatArray) -> FloatArray:
        """Convert `tau(q)` into generalized dimensions.

        Args:
            tau: Mass exponents with shape `(|Q|,)`.

        Returns:
            FloatArray: Generalized dimensions with shape `(|Q|,)`.
        """
        q_orders = np.asarray(self.config.q_orders, dtype=np.float64)
        generalized_dimensions = np.empty_like(tau)
        q_is_one = np.isclose(q_orders, 1.0)
        generalized_dimensions[~q_is_one] = tau[~q_is_one] / (q_orders[~q_is_one] - 1.0)
        generalized_dimensions[q_is_one] = np.nan
        return generalized_dimensions

    def _estimate_information_dimension(
        self,
        scale_summaries: tuple[HMCScaleSummary, ...],
    ) -> float:
        """Estimate the information dimension `D(1)`.

        Args:
            scale_summaries: Per-scale HMC measurements.

        Returns:
            float: Estimated information dimension.
        """
        log_epsilons = np.log(
            np.asarray([scale.epsilon for scale in scale_summaries], dtype=np.float64)
        )
        entropy_terms = np.asarray(
            [self._entropy_term(scale.sequence) for scale in scale_summaries],
            dtype=np.float64,
        )
        return float(np.polyfit(log_epsilons, entropy_terms, deg=1)[0])

    def _entropy_term(self, sequence: FloatArray) -> float:
        """Compute the entropy term `sum(mu log mu)` for one scale.

        Args:
            sequence: Hilbert sequence for one scale.

        Returns:
            float: Entropy term.
        """
        occupied = sequence[sequence > 0.0]
        return float(np.sum(occupied * np.log(occupied)))

    def _estimate_spectrum(
        self,
        q_orders: FloatArray,
        tau: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Estimate optional multifractal spectrum features.

        Args:
            q_orders: q-orders with shape `(|Q|,)`.
            tau: Mass exponents with shape `(|Q|,)`.

        Returns:
            tuple[FloatArray, FloatArray]: `(alpha, f(alpha))`.
        """
        if q_orders.size < 2:
            message = "Spectrum estimation requires at least two q-orders."
            LOGGER.error(message)
            raise HMCComputationError(message)
        alpha = np.gradient(tau, q_orders)
        f_alpha = q_orders * alpha - tau
        if self.config.spectrum_bins <= 0:
            return alpha.astype(np.float64), f_alpha.astype(np.float64)

        order = np.argsort(alpha)
        alpha_sorted = alpha[order]
        f_sorted = f_alpha[order]
        alpha_grid = np.linspace(
            alpha_sorted[0],
            alpha_sorted[-1],
            self.config.spectrum_bins,
        )
        binned_spectrum = np.interp(alpha_grid, alpha_sorted, f_sorted)
        return alpha_grid.astype(np.float64), binned_spectrum.astype(np.float64)
