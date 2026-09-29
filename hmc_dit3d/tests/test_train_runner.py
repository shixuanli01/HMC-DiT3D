"""Tests for the formal HMC-DiT3D training runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.train.train import run_training


def _write_sample(root_dir: Path, category: str, split: str, name: str) -> None:
    """Create one synthetic ShapeNet sample for training tests."""
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    points = np.linspace(0.0, 1.0, 45_000, dtype=np.float32).reshape(15_000, 3)
    np.save(sample_dir / f"{name}.npy", points)


def _write_config(config_path: Path, raw_config: dict[str, object]) -> None:
    """Write one YAML experiment config for training tests."""
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")


def _build_raw_config(
    *,
    experiment_name: str,
    max_steps: int | None,
    epochs: int,
    batch_size: int,
    checkpoint_name: str,
    checkpoint_every: int | None,
    save_every: int | None = None,
    resume_checkpoint: str | None = None,
    use_l_mf: bool = False,
    l_mf_weight: float = 0.0,
    grad_accum_steps: int = 1,
    validation_split: str | None = None,
) -> dict[str, object]:
    """Build a minimal formal-training config dictionary."""
    return {
        "experiment_name": experiment_name,
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
            "batch_size": batch_size,
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
            "epochs": epochs,
            "max_steps": max_steps,
            "log_every": 1,
            "checkpoint_name": checkpoint_name,
            "latest_checkpoint_name": "latest.pt",
            "best_train_checkpoint_name": "best_train.pt",
            "best_val_checkpoint_name": "best_val.pt",
            "resume_checkpoint": resume_checkpoint,
            "checkpoint_every": checkpoint_every,
            "save_every": save_every,
            "metrics_name": "train_metrics.jsonl",
            "evolution_name": "evolution.jsonl",
            "validation_split": validation_split,
            "validation_every": 1,
            "validation_max_batches": None,
            "use_l_mf": use_l_mf,
            "l_mf_weight": l_mf_weight,
            "l_mf_hidden_dim": 32,
            "grad_clip": None,
            "grad_accum_steps": grad_accum_steps,
            "use_amp": False,
            "prefer_cuda": False,
        },
    }


def test_load_experiment_config_resolves_resume_checkpoint(tmp_path: Path) -> None:
    """Resume checkpoints should resolve relative to the YAML config path."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    results_dir = tmp_path / "results" / "resume_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = results_dir / "epoch_1.pt"
    torch.save(
        {
            "epoch": 0,
            "step": 1,
            "model_state": {},
            "optimizer_state": {},
            "scaler_state": None,
            "metrics": {"loss": 1.0, "grad_norm": 2.0},
        },
        resume_checkpoint,
    )
    config_path = tmp_path / "train.yaml"
    raw_config = _build_raw_config(
        experiment_name="resume_test",
        max_steps=1,
        epochs=2,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        resume_checkpoint="./results/resume_test/epoch_1.pt",
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)

    assert config.data.root_dir == dataset_root.resolve()
    assert config.train.resume_checkpoint == resume_checkpoint.resolve()


def test_run_training_saves_epoch_and_final_checkpoints(tmp_path: Path) -> None:
    """Formal training should save JSONL metrics plus epoch and final checkpoints."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "train.yaml"
    raw_config = _build_raw_config(
        experiment_name="formal_train",
        max_steps=2,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)
    metrics = run_training(config)

    output_dir = config.train.results_dir / config.experiment_name
    epoch_checkpoint = output_dir / "epoch_1.pt"
    final_checkpoint = output_dir / "final.pt"
    metrics_path = output_dir / "train_metrics.jsonl"
    evolution_path = output_dir / "evolution.jsonl"

    assert metrics["steps"] == 2.0
    assert epoch_checkpoint.exists()
    assert final_checkpoint.exists()
    assert metrics_path.exists()
    assert evolution_path.exists()

    final_payload = torch.load(final_checkpoint, map_location="cpu")
    assert final_payload["step"] == 2
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    evolution_rows = [
        json.loads(line)
        for line in evolution_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(metric_rows) == 2
    assert metric_rows[-1]["step"] == 2
    assert len(evolution_rows) == 2
    assert evolution_rows[0]["record_type"] == "epoch"
    assert evolution_rows[-1]["record_type"] == "final"
    assert evolution_rows[-1]["checkpoint"] == "final.pt"
    assert evolution_rows[0]["latest_checkpoint"] == "latest.pt"


def test_run_training_saves_best_train_and_best_val(tmp_path: Path) -> None:
    """Formal training should save latest, best-train, and best-val checkpoints."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    _write_sample(dataset_root, "chair", "val", "chair_val_a")
    config_path = tmp_path / "train_with_val.yaml"
    raw_config = _build_raw_config(
        experiment_name="formal_train_with_val",
        max_steps=2,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        save_every=1,
        validation_split="val",
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)
    run_training(config)

    output_dir = config.train.results_dir / config.experiment_name
    latest_checkpoint = output_dir / "latest.pt"
    best_train_checkpoint = output_dir / "best_train.pt"
    best_val_checkpoint = output_dir / "best_val.pt"
    evolution_path = output_dir / "evolution.jsonl"
    evolution_rows = [
        json.loads(line)
        for line in evolution_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert latest_checkpoint.exists()
    assert best_train_checkpoint.exists()
    assert best_val_checkpoint.exists()
    assert evolution_rows[0]["best_train_checkpoint"] == "best_train.pt"
    assert evolution_rows[0]["best_val_checkpoint"] == "best_val.pt"
    assert evolution_rows[0]["val_loss"] is not None


def test_run_training_resumes_from_epoch_checkpoint(tmp_path: Path) -> None:
    """Formal training should resume from an epoch checkpoint and continue steps."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "train.yaml"

    initial_raw_config = _build_raw_config(
        experiment_name="resume_train",
        max_steps=1,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
    )
    _write_config(config_path, initial_raw_config)
    initial_config = load_experiment_config(config_path)
    run_training(initial_config)

    resume_raw_config = _build_raw_config(
        experiment_name="resume_train",
        max_steps=2,
        epochs=2,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        resume_checkpoint="./results/resume_train/epoch_1.pt",
    )
    _write_config(config_path, resume_raw_config)
    resumed_config = load_experiment_config(config_path)
    metrics = run_training(resumed_config)

    final_checkpoint = (
        resumed_config.train.results_dir / resumed_config.experiment_name / "final.pt"
    )
    final_payload = torch.load(final_checkpoint, map_location="cpu")

    assert metrics["steps"] == 2.0
    assert metrics["epoch"] == 2.0
    assert final_payload["step"] == 2
    assert final_payload["epoch"] == 1


def test_run_training_with_l_mf_saves_auxiliary_state(tmp_path: Path) -> None:
    """Formal training should persist the auxiliary predictor when `L_mf` is used."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "train_lmf.yaml"
    raw_config = _build_raw_config(
        experiment_name="formal_train_lmf",
        max_steps=1,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        use_l_mf=True,
        l_mf_weight=0.25,
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)
    metrics = run_training(config)

    final_checkpoint = (
        config.train.results_dir / config.experiment_name / config.train.checkpoint_name
    )
    final_payload = torch.load(final_checkpoint, map_location="cpu")
    metrics_path = (
        config.train.results_dir / config.experiment_name / "train_metrics.jsonl"
    )
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert metrics["loss_mf"] >= 0.0
    assert "multifractal_predictor" in final_payload["auxiliary_state"]
    assert metric_rows[0]["loss_mf"] >= 0.0


def test_run_training_with_l_mf_resumes_auxiliary_state(tmp_path: Path) -> None:
    """`L_mf` resume should restore the auxiliary predictor and continue steps."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "train_lmf_resume.yaml"

    initial_raw_config = _build_raw_config(
        experiment_name="formal_train_lmf_resume",
        max_steps=1,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        use_l_mf=True,
        l_mf_weight=0.25,
    )
    _write_config(config_path, initial_raw_config)
    initial_config = load_experiment_config(config_path)
    run_training(initial_config)

    resume_raw_config = _build_raw_config(
        experiment_name="formal_train_lmf_resume",
        max_steps=2,
        epochs=2,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        resume_checkpoint="./results/formal_train_lmf_resume/epoch_1.pt",
        use_l_mf=True,
        l_mf_weight=0.25,
    )
    _write_config(config_path, resume_raw_config)
    resumed_config = load_experiment_config(config_path)
    metrics = run_training(resumed_config)

    final_checkpoint = (
        resumed_config.train.results_dir
        / resumed_config.experiment_name
        / resumed_config.train.checkpoint_name
    )
    final_payload = torch.load(final_checkpoint, map_location="cpu")

    assert metrics["steps"] == 2.0
    assert metrics["loss_mf"] >= 0.0
    assert "multifractal_predictor" in final_payload["auxiliary_state"]


def test_run_training_with_gradient_accumulation(tmp_path: Path) -> None:
    """Formal training should support gradient accumulation without breaking IO."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    config_path = tmp_path / "train_accum.yaml"
    raw_config = _build_raw_config(
        experiment_name="formal_train_accum",
        max_steps=2,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=1,
        grad_accum_steps=2,
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)
    metrics = run_training(config)

    final_checkpoint = (
        config.train.results_dir / config.experiment_name / config.train.checkpoint_name
    )
    metrics_path = (
        config.train.results_dir / config.experiment_name / "train_metrics.jsonl"
    )
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert metrics["steps"] == 2.0
    assert final_checkpoint.exists()
    assert len(metric_rows) == 2


def test_run_training_with_fair_options_saves_ema(tmp_path: Path) -> None:
    """Formal training should combine full HMC, Min-SNR, dropout, and EMA."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    config_path = tmp_path / "train_fair.yaml"
    raw_config = _build_raw_config(
        experiment_name="formal_train_fair",
        max_steps=1,
        epochs=1,
        batch_size=1,
        checkpoint_name="final.pt",
        checkpoint_every=None,
    )
    data_config = raw_config["data"]
    train_config = raw_config["train"]
    assert isinstance(data_config, dict)
    assert isinstance(train_config, dict)
    data_config["hmc_point_source"] = "full"
    train_config.update(
        {
            "min_snr_gamma": 5.0,
            "hmc_dropout_prob": 0.5,
            "use_ema": True,
            "ema_decay": 0.999,
        }
    )
    _write_config(config_path, raw_config)

    config = load_experiment_config(config_path)
    metrics = run_training(config)

    final_checkpoint = (
        config.train.results_dir / config.experiment_name / config.train.checkpoint_name
    )
    payload = torch.load(final_checkpoint, map_location="cpu")
    assert 0.0 < metrics["snr_weight"] <= 1.0
    assert isinstance(payload["ema_model_state"], dict)
    assert payload["ema_model_state"].keys() == payload["model_state"].keys()
