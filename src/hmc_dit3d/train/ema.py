"""Exponential moving average utilities for diffusion training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


class ModelEMA:
    """Maintain a detached exponential moving average of a model state."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update the moving average from the current model parameters."""
        current_state = model.state_dict()
        if current_state.keys() != self.shadow.keys():
            raise ValueError("EMA and model state dictionaries do not match.")
        update_rate = 1.0 - self.decay
        for name, current in current_state.items():
            averaged = self.shadow[name]
            current = current.detach().to(device=averaged.device)
            if averaged.is_floating_point() or averaged.is_complex():
                averaged.lerp_(current.to(dtype=averaged.dtype), update_rate)
            else:
                averaged.copy_(current)

    def state_dict(self) -> dict[str, Tensor]:
        """Return the averaged model state for checkpoint serialization."""
        return {name: value.detach() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Restore a previously saved averaged model state."""
        if state_dict.keys() != self.shadow.keys():
            raise ValueError("Saved EMA and model state dictionaries do not match.")
        for name, value in state_dict.items():
            target = self.shadow[name]
            target.copy_(
                torch.as_tensor(
                    value,
                    device=target.device,
                    dtype=target.dtype,
                )
            )
