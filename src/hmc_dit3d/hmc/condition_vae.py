"""VAE prior over raw HMC conditions (descriptor + Hilbert measure sequences).

The VAE is trained on the same raw features stored in an offline HMC condition
bank: one multifractal descriptor per shape plus one normalized Hilbert measure
sequence per dyadic scale. At inference time, sampling ``z ~ N(0, I)`` and
decoding yields synthetic HMC conditions, so generation no longer depends on
any ground-truth point cloud (train or test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


class HMCConditionVAEError(RuntimeError):
    """Raised when HMC condition-VAE construction or sampling fails."""


@dataclass(slots=True, frozen=True)
class HMCConditionVAEOutput:
    """Forward outputs of :class:`HMCConditionVAE`.

    Args:
        descriptor: Reconstructed descriptor in normalized space `(B, D)`.
        sequence_logits: Per-scale reconstruction logits, each `(B, L_s)`.
        mu: Posterior means `(B, Z)`.
        logvar: Posterior log-variances `(B, Z)`.
    """

    descriptor: Tensor
    sequence_logits: tuple[Tensor, ...]
    mu: Tensor
    logvar: Tensor


class HMCConditionVAE(nn.Module):
    """MLP VAE over concatenated HMC descriptors and Hilbert sequences.

    Sequences are probability measures over Hilbert-ordered voxels (each scale
    sums to ~1). The decoder emits per-scale logits followed by a softmax so
    generated sequences remain valid measures. Descriptors are modeled in
    z-scored space; normalization statistics are stored as buffers.
    """

    def __init__(
        self,
        descriptor_dim: int,
        sequence_lengths: tuple[int, ...],
        *,
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (2048, 1024, 512),
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0 or len(sequence_lengths) == 0:
            message = "descriptor_dim must be positive and sequence_lengths non-empty."
            LOGGER.error(message)
            raise HMCConditionVAEError(message)
        self.descriptor_dim = int(descriptor_dim)
        self.sequence_lengths = tuple(int(length) for length in sequence_lengths)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)

        input_dim = self.descriptor_dim + sum(self.sequence_lengths)
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for width in self.hidden_dims:
            encoder_layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(previous, self.latent_dim)
        self.fc_logvar = nn.Linear(previous, self.latent_dim)

        decoder_layers: list[nn.Module] = []
        previous = self.latent_dim
        for width in reversed(self.hidden_dims):
            decoder_layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        self.decoder = nn.Sequential(*decoder_layers)
        self.descriptor_head = nn.Linear(previous, self.descriptor_dim)
        self.sequence_heads = nn.ModuleList(
            nn.Linear(previous, length) for length in self.sequence_lengths
        )

        self.register_buffer("descriptor_mean", torch.zeros(self.descriptor_dim))
        self.register_buffer("descriptor_std", torch.ones(self.descriptor_dim))

    def set_descriptor_stats(self, mean: Tensor, std: Tensor) -> None:
        """Store descriptor normalization statistics computed from data."""
        self.descriptor_mean.copy_(mean.reshape(-1).to(self.descriptor_mean))
        self.descriptor_std.copy_(
            std.reshape(-1).clamp_min(1e-8).to(self.descriptor_std)
        )

    def _flatten_inputs(self, descriptor: Tensor, sequences: list[Tensor]) -> Tensor:
        """Normalize and concatenate inputs into a single feature vector."""
        if len(sequences) != len(self.sequence_lengths):
            message = (
                f"Expected {len(self.sequence_lengths)} sequences, "
                f"got {len(sequences)}."
            )
            LOGGER.error(message)
            raise HMCConditionVAEError(message)
        descriptor_norm = (descriptor - self.descriptor_mean) / self.descriptor_std
        # Scale measures by sequence length so every input dimension is O(1).
        scaled_sequences = [
            sequence * float(length)
            for sequence, length in zip(sequences, self.sequence_lengths, strict=True)
        ]
        return torch.cat([descriptor_norm, *scaled_sequences], dim=-1)

    def encode(
        self,
        descriptor: Tensor,
        sequences: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Encode raw HMC conditions into posterior parameters."""
        hidden = self.encode_features(descriptor, sequences)
        # Clamp keeps exp(logvar) finite and avoids KL blow-ups late in training.
        return self.fc_mu(hidden), self.fc_logvar(hidden).clamp(-10.0, 10.0)

    def encode_features(self, descriptor: Tensor, sequences: list[Tensor]) -> Tensor:
        """Return the pre-latent encoder representation of raw conditions."""
        return self.encoder(self._flatten_inputs(descriptor, sequences))

    def decode(self, latent: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Decode latents into normalized descriptors and sequence logits."""
        hidden = self.decoder(latent)
        descriptor = self.descriptor_head(hidden)
        logits = tuple(head(hidden) for head in self.sequence_heads)
        return descriptor, logits

    @torch.no_grad()
    def decode_conditions(
        self,
        latent: Tensor,
        *,
        sequence_threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Decode latent vectors into raw, valid HMC conditions."""
        descriptor_norm, sequence_logits = self.decode(latent)
        descriptors = descriptor_norm * self.descriptor_std + self.descriptor_mean
        sequences: list[Tensor] = []
        for logits, length in zip(sequence_logits, self.sequence_lengths, strict=True):
            measures = torch.softmax(logits, dim=-1)
            if sequence_threshold > 0.0:
                cutoff = float(sequence_threshold) / float(length)
                measures = torch.where(
                    measures >= cutoff, measures, torch.zeros_like(measures)
                )
                measures = measures / measures.sum(dim=-1, keepdim=True).clamp_min(
                    1e-12
                )
            sequences.append(measures)
        return {"descriptors": descriptors, "sequences": sequences}

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Draw a posterior sample with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(
        self,
        descriptor: Tensor,
        sequences: list[Tensor],
    ) -> HMCConditionVAEOutput:
        """Run the full encode-sample-decode pass."""
        mu, logvar = self.encode(descriptor, sequences)
        latent = self.reparameterize(mu, logvar)
        descriptor_recon, sequence_logits = self.decode(latent)
        return HMCConditionVAEOutput(
            descriptor=descriptor_recon,
            sequence_logits=sequence_logits,
            mu=mu,
            logvar=logvar,
        )

    def loss(
        self,
        output: HMCConditionVAEOutput,
        descriptor: Tensor,
        sequences: list[Tensor],
        *,
        beta: float,
        descriptor_weight: float = 1.0,
        sequence_weights: tuple[float, ...] | None = None,
    ) -> dict[str, Tensor]:
        """Compute ELBO-style training losses.

        Sequence reconstruction uses cross-entropy between the target measure
        and the softmax head (equal to KL(target || pred) up to a constant).
        """
        descriptor_target = (descriptor - self.descriptor_mean) / self.descriptor_std
        loss_descriptor = F.mse_loss(output.descriptor, descriptor_target)
        loss_sequences = torch.zeros((), device=descriptor.device)
        if sequence_weights is None:
            weights = tuple(1.0 for _ in sequences)
        else:
            if len(sequence_weights) != len(sequences):
                message = "sequence_weights length must match the number of sequences."
                LOGGER.error(message)
                raise HMCConditionVAEError(message)
            weights = sequence_weights
        for logits, target, weight in zip(
            output.sequence_logits, sequences, weights, strict=True
        ):
            loss_sequences = loss_sequences + weight * (
                -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            )
        loss_kl = (
            -0.5
            * (1.0 + output.logvar - output.mu.pow(2) - output.logvar.exp())
            .sum(dim=-1)
            .mean()
        )
        loss_total = (
            descriptor_weight * loss_descriptor + loss_sequences + beta * loss_kl
        )
        return {
            "loss": loss_total,
            "loss_descriptor": loss_descriptor,
            "loss_sequences": loss_sequences,
            "loss_kl": loss_kl,
        }

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        *,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        sequence_threshold: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Sample synthetic HMC conditions from the learned prior.

        Args:
            num_samples: Number of conditions to draw.
            device: Target device for returned tensors.
            temperature: Std multiplier on the latent prior.
            sequence_threshold: Relative sparsification threshold. Measure
                entries below ``threshold / L_s`` are zeroed and each sequence
                is renormalized, mimicking the exact zeros of real Hilbert
                measures.
            generator: Optional torch generator for reproducibility.

        Returns:
            dict[str, Any]: ``descriptors`` `(B, D)` in raw descriptor space and
            ``sequences`` as a list of `(B, L_s)` measures.
        """
        if num_samples <= 0:
            message = f"num_samples must be positive, got {num_samples}."
            LOGGER.error(message)
            raise HMCConditionVAEError(message)
        resolved_device = torch.device(device)
        latent = torch.randn(
            num_samples,
            self.latent_dim,
            device=resolved_device,
            generator=generator,
        ) * float(temperature)
        return self.decode_conditions(
            latent,
            sequence_threshold=sequence_threshold,
        )


def save_condition_vae_checkpoint(
    model: HMCConditionVAE,
    output_path: str | Path,
    *,
    hmc_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a condition-VAE checkpoint with rebuild metadata."""
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "descriptor_dim": model.descriptor_dim,
        "sequence_lengths": model.sequence_lengths,
        "latent_dim": model.latent_dim,
        "hidden_dims": model.hidden_dims,
        "hmc_config": hmc_config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, resolved_path)
    return resolved_path


def load_condition_vae_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[HMCConditionVAE, dict[str, Any]]:
    """Load a condition VAE and its metadata from disk."""
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.exists():
        message = f"Condition-VAE checkpoint does not exist: {resolved_path!s}."
        LOGGER.error(message)
        raise HMCConditionVAEError(message)
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    model = HMCConditionVAE(
        descriptor_dim=int(payload["descriptor_dim"]),
        sequence_lengths=tuple(payload["sequence_lengths"]),
        latent_dim=int(payload["latent_dim"]),
        hidden_dims=tuple(payload["hidden_dims"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, payload
