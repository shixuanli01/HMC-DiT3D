"""Tests for ShapeNetCore.v2.PC15k dataset loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hmc_dit3d.data.shapenet_pc15k import (
    CATEGORY_TO_SYNSETID,
    ShapeNetDataError,
    ShapeNetPC15KDataset,
)


def _write_sample(root_dir: Path, category: str, split: str, name: str) -> None:
    """Create one synthetic ShapeNet sample."""
    synset_id = CATEGORY_TO_SYNSETID[category]
    sample_dir = root_dir / synset_id / split
    sample_dir.mkdir(parents=True, exist_ok=True)
    points = np.linspace(0.0, 1.0, 45_000, dtype=np.float32).reshape(15_000, 3)
    np.save(sample_dir / f"{name}.npy", points)


def test_shapenet_dataset_loads_points_and_metadata(tmp_path: Path) -> None:
    """The dataset should load requested categories and sample the requested size."""
    _write_sample(tmp_path, "chair", "train", "chair_a")
    _write_sample(tmp_path, "chair", "train", "chair_b")

    dataset = ShapeNetPC15KDataset(
        root_dir=tmp_path,
        categories=["chair"],
        split="train",
        sample_size=128,
        random_subsample=False,
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["points"].shape == (128, 3)
    assert sample["label"] == 0
    assert sample["category"] == "chair"
    assert sample["synset_id"] == CATEGORY_TO_SYNSETID["chair"]
    assert sample["model_id"] in {"chair_a", "chair_b"}
    assert isinstance(sample["path"], str)


def test_shapenet_dataset_rejects_unknown_categories(tmp_path: Path) -> None:
    """Unknown ShapeNet categories should fail fast."""
    with pytest.raises(ShapeNetDataError):
        ShapeNetPC15KDataset(
            root_dir=tmp_path,
            categories=["unknown_category"],
            split="train",
            sample_size=128,
        )
