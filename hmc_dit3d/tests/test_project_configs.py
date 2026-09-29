"""Project-level configuration regression tests."""

from __future__ import annotations

from pathlib import Path

from hmc_dit3d.train.config import load_experiment_config


def test_committed_experiment_configs_load() -> None:
    """Committed experiment configs should remain parseable."""
    project_root = Path(__file__).resolve().parents[1]
    config_names = sorted(
        path.name
        for path in (project_root / "configs").glob("*.yaml")
        if path.name != "hmc_base.yaml"
    )

    for config_name in config_names:
        config = load_experiment_config(project_root / "configs" / config_name)
        assert len(config.data.categories) >= 1
        assert config.model.depth >= 1
        assert config.train.grad_accum_steps >= 1
        if "h100_formal" in config_name:
            assert config.train.resolved_save_every >= 1
            assert config.train.validation_split == "val"
