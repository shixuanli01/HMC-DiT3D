"""Tests for checkpoint-based sampling and saved-sample evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from hmc_dit3d.data.hmc_condition_bank import (
    build_hmc_condition_bank,
    save_hmc_condition_bank,
)
from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.train.evaluate import evaluate_saved_samples
from hmc_dit3d.train.sample import (
    generate_bank_conditioned_samples,
    generate_reference_conditioned_samples,
    save_sample_payload,
)
from hmc_dit3d.train.smoke import run_smoke_training


def _write_sample(root_dir: Path, category: str, split: str, name: str) -> None:
    """Create one synthetic ShapeNet sample for sampling tests."""
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    points = np.linspace(0.0, 1.0, 45_000, dtype=np.float32).reshape(15_000, 3)
    np.save(sample_dir / f"{name}.npy", points)


def _write_config(config_path: Path, dataset_root: Path) -> None:
    """Create a compact CPU-only config for training and sampling tests."""
    raw_config = {
        "experiment_name": "sampling_test",
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
            "num_timesteps": 4,
        },
        "data": {
            "root_dir": str(dataset_root.resolve()),
            "categories": ["chair"],
            "split": "train",
            "sample_size": 64,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "random_subsample": False,
            "drop_last": False,
            "hmc_point_source": "full",
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
            "results_dir": str((config_path.parent / "results").resolve()),
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_name": "smoke.pt",
            "grad_clip": None,
            "min_snr_gamma": 5.0,
            "hmc_dropout_prob": 0.15,
            "use_ema": True,
            "ema_decay": 0.999,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")


def test_reference_conditioned_sampling_and_saved_sample_evaluation(
    tmp_path: Path,
) -> None:
    """Sampling from a trained checkpoint should produce a finite saved payload."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "sampling.yaml"
    _write_config(config_path, dataset_root)

    config = load_experiment_config(config_path)
    run_smoke_training(config)
    checkpoint_path = config.train.results_dir / config.experiment_name / "smoke.pt"

    payload = generate_reference_conditioned_samples(
        config,
        checkpoint_path,
        split="train",
        limit=2,
        hmc_guidance_scale=1.5,
        device=torch.device("cpu"),
    )

    assert payload["samples"].shape == (2, 64, 3)
    assert payload["references"].shape == (2, 64, 3)
    assert payload["labels"].shape == (2,)
    assert payload["hmc_guidance_scale"] == 1.5
    assert torch.isfinite(payload["samples"]).all()
    assert torch.isfinite(payload["references"]).all()

    sample_path = tmp_path / "samples.pt"
    save_sample_payload(payload, sample_path)
    metrics = evaluate_saved_samples(
        sample_path,
        batch_size=1,
        device="cpu",
        jsd_resolution=8,
    )

    assert sample_path.exists()
    assert set(metrics) == {
        "lgan_mmd-CD",
        "lgan_cov-CD",
        "lgan_mmd_smp-CD",
        "1-NN-CD-acc",
        "lgan_mmd-EMD",
        "lgan_cov-EMD",
        "lgan_mmd_smp-EMD",
        "1-NN-EMD-acc",
        "JSD",
    }
    for value in metrics.values():
        assert np.isfinite(value)


def test_bank_conditioned_sampling_and_saved_sample_evaluation(tmp_path: Path) -> None:
    """Bank-conditioned sampling should produce finite samples and references."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    _write_sample(dataset_root, "chair", "test", "chair_test")
    config_path = tmp_path / "sampling.yaml"
    _write_config(config_path, dataset_root)

    config = load_experiment_config(config_path)
    run_smoke_training(config)
    checkpoint_path = config.train.results_dir / config.experiment_name / "smoke.pt"
    bank = build_hmc_condition_bank(
        root_dir=dataset_root,
        categories=config.data.categories,
        split="train",
        hmc_config=config.hmc,
        limit=2,
    )
    bank_path = tmp_path / "condition_bank.pt"
    save_hmc_condition_bank(bank, bank_path)

    payload = generate_bank_conditioned_samples(
        config,
        checkpoint_path,
        bank_path,
        limit=2,
        bank_categories=("chair",),
        device=torch.device("cpu"),
    )

    assert payload["samples"].shape == (2, 64, 3)
    assert payload["references"].shape == (2, 64, 3)
    assert payload["labels"].shape == (2,)
    assert payload["condition_source"] == "bank"
    assert payload["bank_indices"].shape == (2,)
    assert torch.isfinite(payload["samples"]).all()
    assert torch.isfinite(payload["references"]).all()

    sample_path = tmp_path / "bank_samples.pt"
    save_sample_payload(payload, sample_path)
    metrics = evaluate_saved_samples(
        sample_path,
        batch_size=1,
        device="cpu",
        jsd_resolution=8,
    )

    assert sample_path.exists()
    assert set(metrics) == {
        "lgan_mmd-CD",
        "lgan_cov-CD",
        "lgan_mmd_smp-CD",
        "1-NN-CD-acc",
        "lgan_mmd-EMD",
        "lgan_cov-EMD",
        "lgan_mmd_smp-EMD",
        "1-NN-EMD-acc",
        "JSD",
    }
    for value in metrics.values():
        assert np.isfinite(value)
