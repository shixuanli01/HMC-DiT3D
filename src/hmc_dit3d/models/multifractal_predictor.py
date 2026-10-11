"""Auxiliary predictor for the multifractal descriptor loss."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

LOGGER = logging.getLogger(__name__)


class MultifractalPredictorError(RuntimeError):
    """Raised when the multifractal descriptor predictor is misconfigured."""


class MultifractalDescriptorPredictor(nn.Module):
    """Predict HMC descriptor targets from generated point clouds.

    This auxiliary head follows a compact PointNet-style design so it can be
    trained jointly with the diffusion model when `L_mf` is enabled.
    """

    def __init__(
        self,
        input_channels: int,
        descriptor_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize the descriptor predictor.

        Args:
            input_channels: Point feature channels, typically `3`.
            descriptor_dim: Target HMC descriptor dimension.
            hidden_dim: Internal hidden width used by the predictor.
        """
        super().__init__()
        if input_channels <= 0:
            message = "input_channels must be positive."
            LOGGER.error(message)
            raise MultifractalPredictorError(message)
        if descriptor_dim <= 0:
            message = "descriptor_dim must be positive."
            LOGGER.error(message)
            raise MultifractalPredictorError(message)
        if hidden_dim <= 0:
            message = "hidden_dim must be positive."
            LOGGER.error(message)
            raise MultifractalPredictorError(message)

        self.input_channels = input_channels
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize the predictor with stable linear defaults."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d | nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, points: Tensor) -> Tensor:
        """Predict descriptors from point clouds.

        Args:
            points: Point tensor with shape `(B, C, N)`.

        Returns:
            Tensor: Predicted descriptor tensor with shape `(B, descriptor_dim)`.
        """
        if points.ndim != 3:
            message = f"points must have shape (B, C, N), got {tuple(points.shape)!r}."
            LOGGER.error(message)
            raise MultifractalPredictorError(message)
        if points.shape[1] != self.input_channels:
            message = (
                f"points channel count must be {self.input_channels}, "
                f"got {points.shape[1]}."
            )
            LOGGER.error(message)
            raise MultifractalPredictorError(message)
        features = self.point_mlp(points.to(torch.float32))
        pooled = torch.cat(
            [
                features.amax(dim=2),
                features.mean(dim=2),
            ],
            dim=1,
        )
        return self.readout(pooled)
