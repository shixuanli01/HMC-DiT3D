"""Tests for offline HMC condition-bank utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hmc_dit3d.data.hmc_condition_bank import (
    build_hmc_condition_bank,
    load_hmc_condition_bank,
    sample_hmc_condition_bank,
    save_hmc_condition_bank,
)
from hmc_dit3d.data.shapenet_pc15k import CATEGORY_TO_SYNSETID
from hmc_dit3d.hmc.config import HMCConfig


def _write_sample(root_dir: Path, category: str, split: str, name: str) -> None:
    """Create one synthetic ShapeNet sample for condition-bank tests."""
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    points = np.linspace(0.0, 1.0, 45_000, dtype=np.float32).reshape(15_000, 3)
    np.save(sample_dir / f"{name}.npy", points)


def test_build_hmc_condition_bank_collects_expected_shapes(tmp_path: Path) -> None:
    """Building a bank should collect one descriptor and one sequence per sample."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    _write_sample(dataset_root, "airplane", "train", "airplane_a")
    config = HMCConfig(scales=(2, 3), q_orders=(0.0, 1.0, 2.0))

    bank = build_hmc_condition_bank(
        root_dir=dataset_root,
        categories=("chair", "airplane"),
        split="train",
        hmc_config=config,
    )

    assert len(bank) == 3
    assert bank.descriptors.shape == (3, config.descriptor_dim)
    assert bank.sequences[0].shape == (3, (2**2) ** 3)
    assert bank.sequences[1].shape == (3, (2**3) ** 3)
    assert bank.labels.tolist() == [0, 0, 1]


def test_hmc_condition_bank_roundtrip_is_lossless(tmp_path: Path) -> None:
    """Saving and loading a bank should preserve tensors and metadata."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "airplane", "train", "airplane_a")
    config = HMCConfig(scales=(2, 3), q_orders=(0.0, 1.0, 2.0))
    bank = build_hmc_condition_bank(
        root_dir=dataset_root,
        categories=("chair", "airplane"),
        split="train",
        hmc_config=config,
    )
    bank_path = tmp_path / "bank.pt"

    save_hmc_condition_bank(bank, bank_path)
    loaded = load_hmc_condition_bank(bank_path)

    assert bank_path.exists()
    assert loaded.categories == bank.categories
    assert loaded.model_ids == bank.model_ids
    assert loaded.source_paths == bank.source_paths
    assert torch.equal(loaded.labels, bank.labels)
    assert torch.allclose(loaded.descriptors, bank.descriptors)
    assert torch.allclose(loaded.sequences[0], bank.sequences[0])
    assert torch.allclose(loaded.sequences[1], bank.sequences[1])


def test_sample_hmc_condition_bank_is_reproducible_and_filterable(
    tmp_path: Path,
) -> None:
    """Sampling should respect category filters and a fixed torch generator."""
    dataset_root = tmp_path / "dataset"
    _write_sample(dataset_root, "chair", "train", "chair_a")
    _write_sample(dataset_root, "chair", "train", "chair_b")
    _write_sample(dataset_root, "airplane", "train", "airplane_a")
    config = HMCConfig(scales=(2, 3), q_orders=(0.0, 1.0, 2.0))
    bank = build_hmc_condition_bank(
        root_dir=dataset_root,
        categories=("chair", "airplane"),
        split="train",
        hmc_config=config,
    )
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)

    first = sample_hmc_condition_bank(
        bank,
        num_samples=2,
        categories=("airplane",),
        replacement=True,
        generator=generator_a,
        device="cpu",
    )
    second = sample_hmc_condition_bank(
        bank,
        num_samples=2,
        categories=("airplane",),
        replacement=True,
        generator=generator_b,
        device="cpu",
    )

    assert first["categories"] == ["airplane", "airplane"]
    assert second["categories"] == ["airplane", "airplane"]
    assert torch.equal(first["indices"], second["indices"])
    assert torch.allclose(first["descriptors"], second["descriptors"])
    assert torch.equal(first["labels"], second["labels"])
