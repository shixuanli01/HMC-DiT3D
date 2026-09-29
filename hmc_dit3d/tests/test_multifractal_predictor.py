"""Tests for the auxiliary multifractal descriptor predictor."""

from __future__ import annotations

import pytest
import torch

from hmc_dit3d.models.multifractal_predictor import (
    MultifractalDescriptorPredictor,
    MultifractalPredictorError,
)


def test_multifractal_predictor_returns_expected_shape() -> None:
    """The auxiliary predictor should map point clouds to descriptor vectors."""
    predictor = MultifractalDescriptorPredictor(
        input_channels=3,
        descriptor_dim=5,
        hidden_dim=16,
    )

    descriptor = predictor(torch.randn(2, 3, 32))

    assert descriptor.shape == (2, 5)
    assert torch.isfinite(descriptor).all()


@pytest.mark.parametrize(
    ("points", "message_fragment"),
    [
        (torch.randn(2, 32), r"shape \(B, C, N\)"),
        (torch.randn(2, 4, 32), "channel count"),
    ],
)
def test_multifractal_predictor_rejects_invalid_inputs(
    points: torch.Tensor, message_fragment: str
) -> None:
    """The auxiliary predictor should fail fast on malformed tensors."""
    predictor = MultifractalDescriptorPredictor(
        input_channels=3,
        descriptor_dim=5,
        hidden_dim=16,
    )

    with pytest.raises(MultifractalPredictorError, match=message_fragment):
        predictor(points)
