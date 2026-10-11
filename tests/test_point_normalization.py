"""Tests for global point normalization (scheme 3: normalized DiT retraining)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from hmc_dit3d.data.point_normalization import (
    PointNormalization,
    PointNormalizationError,
    compute_point_normalization,
)
from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.train.sample import point_normalization_from_checkpoint
from hmc_dit3d.train.train import run_training


def _write_sample(
    root_dir: Path, category: str, split: str, name: str, seed: int
) -> None:
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    points = (rng.normal(size=(15_000, 3)) * 0.16 + np.array([0.0, 0.1, -0.05])).astype(
        np.float32
    )
    np.save(sample_dir / f"{name}.npy", points)


def test_normalize_roundtrip_for_both_layouts() -> None:
    stats = PointNormalization(mean=(0.1, -0.2, 0.3), std=0.5)
    bnc = torch.randn(2, 64, 3)
    bcn = bnc.transpose(1, 2)
    assert torch.allclose(stats.denormalize(stats.normalize(bnc)), bnc, atol=1e-6)
    assert torch.allclose(stats.denormalize(stats.normalize(bcn)), bcn, atol=1e-6)
    # Both layouts must whiten the same coordinates.
    assert torch.allclose(stats.normalize(bnc).transpose(1, 2), stats.normalize(bcn))
    whitened = stats.normalize(torch.tensor([[[0.1, -0.2, 0.3]]]))
    assert torch.allclose(whitened, torch.zeros(1, 1, 3))


def test_to_dict_from_dict_and_validation() -> None:
    stats = PointNormalization(mean=(0.1, -0.2, 0.3), std=0.5, source="unit")
    rebuilt = PointNormalization.from_dict(stats.to_dict())
    assert rebuilt == stats
    assert PointNormalization.from_dict(None) is None
    with pytest.raises(PointNormalizationError):
        PointNormalization(mean=(0.0, 0.0, 0.0), std=0.0)
    with pytest.raises(PointNormalizationError):
        PointNormalization.from_dict({"mean": [0, 0, 0]})


def test_compute_matches_numpy_global_statistics(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for index in range(3):
        _write_sample(root, "chair", "train", f"chair_{index}", seed=index)
    _write_sample(root, "chair", "val", "chair_val", seed=99)
    stats = compute_point_normalization(root, ["chair"], split="train")
    synset = CATEGORY_TO_SYNSETID["chair"]
    all_points = np.concatenate(
        [np.load(p) for p in sorted((root / synset / "train").glob("*.npy"))]
    ).astype(np.float64)
    assert np.allclose(stats.mean, all_points.mean(axis=0), atol=1e-6)
    assert np.isclose(stats.std, all_points.reshape(-1).std(), rtol=1e-6)
    assert "3 shapes" in stats.source


def _raw_config(root: Path, results: Path, normalize: bool) -> dict[str, object]:
    return {
        "experiment_name": "norm_train",
        "hmc": {
            "scales": [2, 3],
            "q_orders": [0.0, 1.0, 2.0],
            "normalization_mode": "bbox",
            "delta": 1.0e-8,
            "empty_box_epsilon": 1.0e-12,
            "use_spectrum": False,
            "spectrum_bins": 0,
        },
        "encoder": {
            "model_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "window_length": 16,
        },
        "diffusion": {
            "schedule_type": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
            "num_timesteps": 1000,
        },
        "data": {
            "root_dir": str(root),
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
            "normalize_points": normalize,
        },
        "model": {
            "voxel_size": 8,
            "patch_size": 2,
            "in_channels": 3,
            "out_channels": 3,
            "input_dim": 16,
            "model_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "class_dropout_prob": 0.1,
        },
        "train": {
            "seed": 7,
            "results_dir": str(results),
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 2,
            "log_every": 1,
            "checkpoint_name": "final.pt",
            "checkpoint_every": 1,
            "validation_split": "val",
            "validation_every": 1,
            "grad_clip": None,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }


def test_run_training_stores_point_normalization_in_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_sample(root, "chair", "train", "chair_a", seed=1)
    _write_sample(root, "chair", "train", "chair_b", seed=2)
    _write_sample(root, "chair", "val", "chair_val", seed=3)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(_raw_config(root, tmp_path / "results", normalize=True)),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert config.data.normalize_points is True
    run_training(config)

    expected = compute_point_normalization(root, ["chair"], split="train")
    output_dir = config.train.results_dir / config.experiment_name
    for name in ("final.pt", "epoch_1.pt", "latest.pt", "best_train.pt", "best_val.pt"):
        payload = torch.load(output_dir / name, map_location="cpu", weights_only=False)
        stored = point_normalization_from_checkpoint(payload)
        assert stored is not None and stored.is_close(expected), name


def test_run_training_without_normalization_stores_none(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_sample(root, "chair", "train", "chair_a", seed=1)
    _write_sample(root, "chair", "val", "chair_val", seed=3)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(_raw_config(root, tmp_path / "results", normalize=False)),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    run_training(config)
    payload = torch.load(
        config.train.results_dir / config.experiment_name / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["point_normalization"] is None
    assert point_normalization_from_checkpoint(payload) is None
