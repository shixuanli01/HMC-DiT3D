"""HMC condition encoder."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hmc_dit3d.hmc.config import HMCConfig, HMCConfigurationError, HMCEncoderConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class HMCConditionOutput:
    """Encoded HMC conditions.

    Args:
        global_embedding: Global descriptor embedding `g` with shape `(B, d)`.
        condition_tokens: Condition tokens `C` with shape `(B, M, d)`.
    """

    global_embedding: Tensor
    condition_tokens: Tensor


class HMCConditionEncoder(nn.Module):
    """Encode HMC descriptors and Hilbert sequences into `(g, C)`."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
    ) -> None:
        """Initialize the HMC encoder.

        Args:
            hmc_config: Numerical HMC configuration.
            encoder_config: Neural encoder hyperparameters.
        """
        super().__init__()
        self.hmc_config = hmc_config
        self.encoder_config = encoder_config

        window_length = encoder_config.window_length
        model_dim = encoder_config.model_dim

        self.window_projector = nn.Sequential(
            nn.LayerNorm(window_length),
            nn.Linear(window_length, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.valid_fraction_projector = nn.Linear(1, model_dim, bias=False)
        self.scale_embedding = nn.Embedding(len(hmc_config.scales), model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=encoder_config.num_heads,
            dim_feedforward=model_dim * 4,
            dropout=encoder_config.dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_config.num_layers,
        )
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(hmc_config.descriptor_dim),
            nn.Linear(hmc_config.descriptor_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def _window_position_encoding(
        self,
        num_windows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build sinusoidal position encodings for window tokens.

        Args:
            num_windows: Number of windows at the current scale.
            device: Target torch device.
            dtype: Target floating-point dtype.

        Returns:
            Tensor: Position encodings with shape `(1, num_windows, model_dim)`.
        """
        positions = torch.arange(num_windows, device=device, dtype=dtype).unsqueeze(1)
        half_dim = self.encoder_config.model_dim // 2
        frequency_denominator = max(half_dim, 1)
        frequencies = torch.exp(
            -torch.log(torch.tensor(10_000.0, device=device, dtype=dtype))
            * torch.arange(half_dim, device=device, dtype=dtype)
            / frequency_denominator
        )
        angles = positions * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.encoder_config.model_dim % 2 != 0:
            embeddings = torch.cat(
                [embeddings, torch.zeros(num_windows, 1, device=device, dtype=dtype)],
                dim=1,
            )
        return embeddings.unsqueeze(0)

    def forward(
        self,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> HMCConditionOutput:
        """Encode global descriptor and Hilbert sequences.

        Args:
            descriptor: Global descriptor tensor with shape `(B, D)`.
            sequences: One tensor per scale, each of shape `(B, L_s)`.

        Returns:
            HMCConditionOutput: Global embedding `g` and condition tokens `C`.
        """
        target_device = self.scale_embedding.weight.device
        target_dtype = self.scale_embedding.weight.dtype
        descriptor = torch.as_tensor(
            descriptor,
            device=target_device,
            dtype=target_dtype,
        )
        if descriptor.ndim != 2:
            message = (
                f"Descriptor must have shape (B, D), got {tuple(descriptor.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if descriptor.shape[-1] != self.hmc_config.descriptor_dim:
            message = (
                "Descriptor dimension does not match HMCConfig. "
                f"Expected {self.hmc_config.descriptor_dim}, "
                f"got {descriptor.shape[-1]}."
            )
            LOGGER.error(message)
            raise HMCConfigurationError(message)
        if len(sequences) != len(self.hmc_config.scales):
            message = (
                "Number of Hilbert sequences does not match the configured scales. "
                f"Expected {len(self.hmc_config.scales)}, got {len(sequences)}."
            )
            LOGGER.error(message)
            raise HMCConfigurationError(message)

        token_batches: list[Tensor] = []
        batch_size = descriptor.shape[0]
        for scale_index, sequence in enumerate(sequences):
            scale = self.hmc_config.scales[scale_index]
            expected_length = (2**scale) ** 3
            sequence = torch.as_tensor(
                sequence,
                device=target_device,
                dtype=target_dtype,
            )
            if sequence.ndim != 2 or sequence.shape[0] != batch_size:
                message = (
                    "Each sequence must have shape (B, L_s). "
                    f"Scale {scale_index} got {tuple(sequence.shape)!r}."
                )
                LOGGER.error(message)
                raise HMCConfigurationError(message)
            if sequence.shape[1] != expected_length:
                message = (
                    "Each sequence length must match its Hilbert resolution. "
                    f"Scale {scale} expects length {expected_length}, "
                    f"got {sequence.shape[1]}."
                )
                LOGGER.error(message)
                raise HMCConfigurationError(message)
            windows, valid_fractions = self._windowize(sequence)
            projected = self.window_projector(windows)
            projected = projected + self.valid_fraction_projector(valid_fractions)
            projected = projected + self._window_position_encoding(
                num_windows=projected.shape[1],
                device=target_device,
                dtype=target_dtype,
            )
            scale_bias = self.scale_embedding.weight[scale_index].view(1, 1, -1)
            token_batches.append(projected + scale_bias)

        condition_tokens = torch.cat(token_batches, dim=1)
        condition_tokens = self.transformer(condition_tokens)
        condition_tokens = self.output_norm(condition_tokens)
        global_embedding = self.global_mlp(descriptor)
        return HMCConditionOutput(
            global_embedding=global_embedding,
            condition_tokens=condition_tokens,
        )

    def _windowize(self, sequence: Tensor) -> tuple[Tensor, Tensor]:
        """Split one Hilbert sequence into fixed-length windows.

        Args:
            sequence: Hilbert sequence with shape `(B, L_s)`.

        Returns:
            tuple[Tensor, Tensor]:
                Window tensor with shape `(B, M_s, window_length)` and valid
                fractions with shape `(B, M_s, 1)`.
        """
        window_length = self.encoder_config.window_length
        batch_size, sequence_length = sequence.shape
        num_windows = (sequence_length + window_length - 1) // window_length
        remainder = sequence_length % window_length
        if remainder != 0:
            pad_width = window_length - remainder
            # Key detail: zero-padding at the tail keeps per-scale windows aligned
            # without dropping trailing information.
            sequence = torch.nn.functional.pad(sequence, (0, pad_width), value=0.0)
        windows = sequence.view(batch_size, num_windows, window_length)
        valid_counts = [window_length] * num_windows
        if remainder != 0:
            valid_counts[-1] = remainder
        valid_fractions = torch.tensor(
            valid_counts,
            device=sequence.device,
            dtype=sequence.dtype,
        ).view(1, num_windows, 1) / float(window_length)
        valid_fractions = valid_fractions.expand(batch_size, -1, -1)
        return windows, valid_fractions
