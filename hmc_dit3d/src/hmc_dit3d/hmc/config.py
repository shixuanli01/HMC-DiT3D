"""Configuration objects for HMC extraction and encoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

LOGGER = logging.getLogger(__name__)


class HMCError(RuntimeError):
    """Base exception for HMC-related failures."""


class HMCConfigurationError(HMCError):
    """Raised when HMC configuration is invalid."""


class HMCComputationError(HMCError):
    """Raised when HMC features cannot be computed."""


class NormalizationMode(StrEnum):
    """Point cloud normalization modes.

    `min_max`:
        Per-axis min-max normalization to `[0, 1]^3`.
    `bbox`:
        Preserve aspect ratio with the longest bounding-box edge, then map to
        `[0, 1]^3`.
    `unit_sphere`:
        Mean-center, normalize by maximal radius, then map to `[0, 1]^3`.
    """

    MIN_MAX = "min_max"
    BBOX = "bbox"
    UNIT_SPHERE = "unit_sphere"


@dataclass(slots=True, frozen=True)
class HMCConfig:
    """Numerical settings for HMC feature extraction.

    Args:
        scales: Dyadic scales `s`, each scale uses resolution `2**s`.
        q_orders: Multifractal moments used to compute `tau(q)` and `D(q)`.
        normalization_mode: Point-cloud normalization strategy.
        delta: Small positive value used in measure normalization.
        empty_box_epsilon: Stabilizer used when negative moments see empty boxes.
        use_spectrum: Whether to append spectrum features to the descriptor.
        spectrum_bins: Optional fixed-size spectrum feature dimension.
    """

    scales: tuple[int, ...] = (2, 3, 4)
    q_orders: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    normalization_mode: NormalizationMode = NormalizationMode.BBOX
    delta: float = 1.0e-8
    empty_box_epsilon: float = 1.0e-12
    use_spectrum: bool = False
    spectrum_bins: int = 0

    def __post_init__(self) -> None:
        """Validate the HMC numerical configuration."""
        object.__setattr__(self, "scales", tuple(int(scale) for scale in self.scales))
        object.__setattr__(self, "q_orders", tuple(float(q) for q in self.q_orders))
        try:
            normalization_mode = NormalizationMode(self.normalization_mode)
        except ValueError as exc:
            message = (
                "normalization_mode must be one of "
                f"{[mode.value for mode in NormalizationMode]!r}."
            )
            LOGGER.error(message)
            raise HMCConfigurationError(message) from exc
        object.__setattr__(self, "normalization_mode", normalization_mode)

        if len(self.scales) < 2:
            message = "At least two scales are required to estimate tau(q)."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if tuple(sorted(set(self.scales))) != self.scales:
            message = "Scales must be unique and sorted in ascending order."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if any(scale <= 0 for scale in self.scales):
            message = "Scales must be positive integers."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if len(self.q_orders) == 0:
            message = "At least one q-order is required."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if tuple(sorted(set(self.q_orders))) != self.q_orders:
            message = "q_orders must be unique and sorted in ascending order."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.delta <= 0.0:
            message = "delta must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.empty_box_epsilon <= 0.0:
            message = "empty_box_epsilon must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.spectrum_bins < 0:
            message = "spectrum_bins must be greater than or equal to zero."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.use_spectrum and len(self.q_orders) < 2:
            message = "use_spectrum=True requires at least two q-orders."
            LOGGER.error(message)
            raise HMCConfigurationError(message)

    @property
    def descriptor_dim(self) -> int:
        """Return the descriptor feature dimension."""
        return len(self.q_orders) + self.spectrum_feature_dim

    @property
    def spectrum_feature_dim(self) -> int:
        """Return the appended spectrum feature dimension."""
        if not self.use_spectrum:
            return 0
        if self.spectrum_bins > 0:
            return self.spectrum_bins
        return len(self.q_orders)


@dataclass(slots=True, frozen=True)
class HMCEncoderConfig:
    """Network settings for the HMC condition encoder.

    Args:
        model_dim: Token and global embedding dimension.
        num_heads: Attention heads in the condition encoder.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability used inside the encoder.
        window_length: Hilbert window length `L` from the method specification.
    """

    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    window_length: int = 64

    def __post_init__(self) -> None:
        """Validate encoder hyperparameters."""
        if self.model_dim <= 0:
            message = "model_dim must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.num_heads <= 0:
            message = "num_heads must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.model_dim % self.num_heads != 0:
            message = "model_dim must be divisible by num_heads."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.num_layers <= 0:
            message = "num_layers must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if not 0.0 <= self.dropout < 1.0:
            message = "dropout must be in [0, 1)."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if self.window_length <= 0:
            message = "window_length must be positive."
            LOGGER.error(message)
            raise HMCConfigurationError(message)
