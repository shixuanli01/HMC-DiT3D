"""Tests for smoke training configuration and runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID
from hmc_dit3d.hmc.condition_vae import HMCConditionVAE
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.train.smoke import (
    apply_condition_vae_reconstruction,
    run_smoke_training,
)


def _write_sample(root_dir: Path, category: str, split: str, name: str) -> None:
    """Create one synthetic ShapeNet sample for smoke training tests."""
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    points = np.linspace(0.0, 1.0, 45_000, dtype=np.float32).reshape(15_000, 3)
    np.save(sample_dir / f"{name}.npy", points)


def test_condition_vae_reconstruction_replaces_selected_conditions() -> None:
    """Frozen-VAE augmentation should preserve condition shapes and measures."""
    vae = HMCConditionVAE(
        descriptor_dim=3,
        sequence_lengths=(4, 8),
        latent_dim=2,
        hidden_dims=(8,),
    )
    descriptors = torch.randn(3, 3)
    sequences = [
        torch.softmax(torch.randn(3, 4), dim=1),
        torch.softmax(torch.randn(3, 8), dim=1),
    ]

    reconstructed_descriptors, reconstructed_sequences, mask = (
        apply_condition_vae_reconstruction(
            descriptors,
            sequences,
            vae,
            1.0,
            posterior_temperature=0.0,
            deterministic=True,
        )
    )

    assert mask.tolist() == [True, True, True]
    assert reconstructed_descriptors.shape == descriptors.shape
    assert not torch.equal(reconstructed_descriptors, descriptors)
    for reconstructed, original in zip(
        reconstructed_sequences,
        sequences,
        strict=True,
    ):
        assert reconstructed.shape == original.shape
        assert torch.allclose(reconstructed.sum(dim=1), torch.ones(3))


def test_load_experiment_config_resolves_relative_paths(tmp_path: Path) -> None:
    """Config loading should resolve dataset and results paths against the config."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "smoke.yaml"
    raw_config = {
        "experiment_name": "smoke_test",
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
            "root_dir": "./dataset",
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
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
            "results_dir": "./results",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke.pt",
            "grad_clip": None,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_experiment_config(config_path)

    assert config.data.root_dir == dataset_root.resolve()
    assert config.train.results_dir == (tmp_path / "results").resolve()
    assert config.data.categories == ("chair",)


def test_run_smoke_training_saves_checkpoint(tmp_path: Path) -> None:
    """The smoke runner should train one step and save a checkpoint."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "smoke.yaml"
    raw_config = {
        "experiment_name": "smoke_train",
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
            "root_dir": "./dataset",
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
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
            "results_dir": "./results",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke.pt",
            "grad_clip": None,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_experiment_config(config_path)
    metrics = run_smoke_training(config)

    checkpoint_path = config.train.results_dir / config.experiment_name / "smoke.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert metrics["steps"] == 1.0
    assert metrics["loss"] >= 0.0
    assert metrics["grad_norm"] >= 0.0
    assert checkpoint["step"] == 1
    assert checkpoint["scaler_state"] is None


def test_run_smoke_training_with_l_mf_saves_auxiliary_state(tmp_path: Path) -> None:
    """Smoke training should honor `use_l_mf` and persist the auxiliary head."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "smoke_lmf.yaml"
    raw_config = {
        "experiment_name": "smoke_train_lmf",
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
            "root_dir": "./dataset",
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
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
            "results_dir": "./results",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke_lmf.pt",
            "grad_clip": None,
            "use_l_mf": True,
            "l_mf_weight": 0.25,
            "l_mf_hidden_dim": 32,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_experiment_config(config_path)
    metrics = run_smoke_training(config)

    checkpoint_path = config.train.results_dir / config.experiment_name / "smoke_lmf.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert metrics["loss_mf"] >= 0.0
    assert "multifractal_predictor" in checkpoint["auxiliary_state"]


def test_run_smoke_training_with_bottleneck_variant_saves_checkpoint(
    tmp_path: Path,
) -> None:
    """Smoke training should support the 3.2 bottleneck variant."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "smoke_bottleneck.yaml"
    raw_config = {
        "experiment_name": "smoke_bottleneck",
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
            "root_dir": "./dataset",
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
        },
        "model": {
            "model_variant": "bottleneck",
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
            "num_bottleneck_latents": 8,
            "resampler_depth": 1,
        },
        "train": {
            "seed": 7,
            "results_dir": "./results",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke_bottleneck.pt",
            "grad_clip": None,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_experiment_config(config_path)
    metrics = run_smoke_training(config)

    checkpoint_path = (
        config.train.results_dir / config.experiment_name / "smoke_bottleneck.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert config.model.model_variant == "bottleneck"
    assert checkpoint_path.exists()
    assert metrics["steps"] == 1.0
    assert metrics["loss"] >= 0.0
    assert checkpoint["step"] == 1


def test_run_smoke_training_with_fair_training_options(tmp_path: Path) -> None:
    """Full-point HMC, Min-SNR, dropout, and EMA should train together."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "smoke_fair.yaml"
    raw_config = {
        "experiment_name": "smoke_fair",
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
            "root_dir": "./dataset",
            "categories": ["chair"],
            "split": "train",
            "sample_size": 128,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": True,
            "drop_last": False,
            "hmc_point_source": "full",
        },
        "model": {
            "model_variant": "bottleneck",
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
            "num_bottleneck_latents": 8,
            "resampler_depth": 1,
        },
        "train": {
            "seed": 7,
            "results_dir": "./results",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke_fair.pt",
            "grad_clip": 1.0,
            "min_snr_gamma": 5.0,
            "hmc_dropout_prob": 0.5,
            "use_ema": True,
            "ema_decay": 0.999,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    config = load_experiment_config(config_path)
    metrics = run_smoke_training(config)

    checkpoint_path = (
        config.train.results_dir / config.experiment_name / "smoke_fair.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert 0.0 < metrics["snr_weight"] <= 1.0
    assert 0.0 <= metrics["hmc_drop_fraction"] <= 1.0
    assert isinstance(checkpoint["ema_model_state"], dict)
    assert checkpoint["ema_model_state"].keys() == checkpoint["model_state"].keys()
