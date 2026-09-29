"""Tests for HMC feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from hmc_dit3d.hmc.config import HMCConfig, HMCConfigurationError, NormalizationMode
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor


def _make_point_cloud(seed: int = 7, num_points: int = 256) -> np.ndarray:
    """Create a stable synthetic point cloud for testing.

    Args:
        seed: RNG seed.
        num_points: Number of points.

    Returns:
        np.ndarray: Point cloud with shape `(num_points, 3)`.
    """
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(
        loc=(0.2, 0.4, 0.6),
        scale=0.05,
        size=(num_points // 2, 3),
    )
    cluster_b = rng.normal(
        loc=(0.8, 0.6, 0.3),
        scale=0.04,
        size=(num_points // 2, 3),
    )
    return np.concatenate([cluster_a, cluster_b], axis=0)


def test_hmc_extractor_returns_finite_statistics() -> None:
    """HMC extraction should return finite statistics and correct sequence lengths."""
    config = HMCConfig(
        scales=(1, 2, 3),
        q_orders=(-1.0, 0.0, 1.0, 2.0),
        normalization_mode=NormalizationMode.BBOX,
    )
    extractor = HMCFeatureExtractor(config)

    result = extractor.extract(_make_point_cloud())

    assert len(result.scales) == 3
    assert result.scales[0].sequence.shape == (8,)
    assert result.scales[1].sequence.shape == (64,)
    assert result.scales[2].sequence.shape == (512,)
    assert result.descriptor.shape == (config.descriptor_dim,)
    assert np.isfinite(result.descriptor).all()
    assert np.isfinite(result.tau).all()
    assert np.isfinite(result.generalized_dimensions).all()
    assert np.isfinite(result.information_dimension)
    assert np.all((0.0 <= result.normalized_points) & (result.normalized_points < 1.0))


def test_hmc_extractor_appends_spectrum_features_when_enabled() -> None:
    """Spectrum features should enlarge the descriptor when enabled."""
    config = HMCConfig(
        scales=(2, 3, 4),
        q_orders=(0.0, 0.5, 1.0, 2.0),
        normalization_mode=NormalizationMode.UNIT_SPHERE,
        use_spectrum=True,
        spectrum_bins=5,
    )
    extractor = HMCFeatureExtractor(config)

    result = extractor.extract(_make_point_cloud(seed=11, num_points=384))

    assert result.alpha is not None
    assert result.spectrum is not None
    assert result.spectrum.shape == (5,)
    assert result.descriptor.shape == (config.descriptor_dim,)
    assert np.isfinite(result.spectrum).all()


def test_hmc_config_accepts_yaml_style_lists_and_strings() -> None:
    """YAML-like config values should be accepted by HMCConfig."""
    raw_config = yaml.safe_load(
        """
        scales: [2, 3, 4]
        q_orders: [0.0, 0.5, 1.0, 2.0]
        normalization_mode: bbox
        delta: 1.0e-8
        empty_box_epsilon: 1.0e-12
        use_spectrum: false
        spectrum_bins: 0
        """
    )

    config = HMCConfig(**raw_config)

    assert config.scales == (2, 3, 4)
    assert config.q_orders == (0.0, 0.5, 1.0, 2.0)
    assert config.normalization_mode == NormalizationMode.BBOX


def test_hmc_spectrum_requires_at_least_two_q_orders() -> None:
    """Spectrum mode should fail fast on invalid q-order settings."""
    with pytest.raises(HMCConfigurationError):
        HMCConfig(
            scales=(2, 3),
            q_orders=(1.0,),
            use_spectrum=True,
        )
