"""Tests for HMC condition encoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hmc_dit3d.hmc.config import HMCConfig, HMCConfigurationError, HMCEncoderConfig
from hmc_dit3d.hmc.encoder import HMCConditionEncoder
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor


def test_hmc_encoder_produces_expected_shapes() -> None:
    """The encoder should map descriptors and sequences to `(g, C)`."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
        use_spectrum=True,
        spectrum_bins=4,
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=2,
        window_length=16,
    )
    encoder = HMCConditionEncoder(hmc_config, encoder_config)

    batch_size = 3
    descriptor = torch.randn(batch_size, hmc_config.descriptor_dim)
    sequences = [
        torch.randn(batch_size, (2**2) ** 3),
        torch.randn(batch_size, (2**3) ** 3),
    ]

    output = encoder(descriptor, sequences)

    expected_tokens = ((64 + 15) // 16) + ((512 + 15) // 16)
    assert output.global_embedding.shape == (batch_size, 32)
    assert output.condition_tokens.shape == (batch_size, expected_tokens, 32)
    assert torch.isfinite(output.global_embedding).all()
    assert torch.isfinite(output.condition_tokens).all()


def test_extractor_output_can_be_fed_into_encoder_directly() -> None:
    """Extractor outputs should be accepted by the encoder without dtype errors."""
    points = np.random.default_rng(13).normal(size=(320, 3))
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    extractor = HMCFeatureExtractor(hmc_config)
    encoder = HMCConditionEncoder(hmc_config, encoder_config)

    result = extractor.extract(points)
    output = encoder(
        np.expand_dims(result.descriptor, axis=0),
        [np.expand_dims(sequence, axis=0) for sequence in result.sequences],
    )

    assert output.global_embedding.dtype == torch.float32
    assert output.condition_tokens.dtype == torch.float32
    assert output.global_embedding.shape == (1, 32)


@pytest.mark.parametrize("bad_length", [63, 0])
def test_hmc_encoder_rejects_invalid_hilbert_sequence_lengths(
    bad_length: int,
) -> None:
    """The encoder should fail fast on malformed per-scale sequence lengths."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        window_length=16,
    )
    encoder = HMCConditionEncoder(hmc_config, encoder_config)
    descriptor = torch.randn(2, hmc_config.descriptor_dim)
    sequences = [
        torch.randn(2, bad_length),
        torch.randn(2, (2**3) ** 3),
    ]

    with pytest.raises(HMCConfigurationError):
        encoder(descriptor, sequences)


def test_hmc_encoder_is_sensitive_to_window_order() -> None:
    """Window-level Hilbert order should affect the encoded condition tokens."""
    hmc_config = HMCConfig(
        scales=(2, 3),
        q_orders=(0.0, 1.0, 2.0),
    )
    encoder_config = HMCEncoderConfig(
        model_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        window_length=16,
    )
    encoder = HMCConditionEncoder(hmc_config, encoder_config).eval()

    torch.manual_seed(7)
    descriptor = torch.randn(1, hmc_config.descriptor_dim)
    sequence_scale_2 = torch.randn(1, (2**2) ** 3)
    sequence_scale_3 = torch.randn(1, (2**3) ** 3)
    permuted_scale_2 = sequence_scale_2.view(1, -1, 16)[:, [3, 1, 2, 0], :].reshape(
        1, -1
    )
    permuted_scale_3 = sequence_scale_3.view(1, -1, 16)[:, torch.arange(31, -1, -1), :]
    permuted_scale_3 = permuted_scale_3.reshape(1, -1)

    original = encoder(descriptor, [sequence_scale_2, sequence_scale_3])
    permuted = encoder(descriptor, [permuted_scale_2, permuted_scale_3])

    assert torch.allclose(original.global_embedding, permuted.global_embedding)
    assert not torch.allclose(original.condition_tokens, permuted.condition_tokens)
