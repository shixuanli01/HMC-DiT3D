"""Offline HMC condition-bank utilities."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from hmc_dit3d.data.shapenet_pc15k import ShapeNetPC15KDataset
from hmc_dit3d.hmc.config import HMCConfig
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor

LOGGER = logging.getLogger(__name__)


class HMCConditionBankError(RuntimeError):
    """Raised when HMC condition-bank construction or sampling fails."""


@dataclass(slots=True, frozen=True)
class HMCConditionBank:
    """Offline bank of HMC conditions derived from ShapeNet point clouds.

    Args:
        hmc_config: HMC extraction settings used to build the bank.
        split: Dataset split the bank was built from.
        categories: Requested category order.
        labels: Per-entry numeric category labels with shape `(B,)`.
        descriptors: Per-entry HMC global descriptors with shape `(B, D_hmc)`.
        sequences: One Hilbert sequence tensor per configured scale, each with
            shape `(B, L_s)`.
        model_ids: ShapeNet model ids stored in bank order.
        source_paths: Absolute source `.npy` paths stored in bank order.
    """

    hmc_config: HMCConfig
    split: str
    categories: tuple[str, ...]
    labels: Tensor
    descriptors: Tensor
    sequences: tuple[Tensor, ...]
    model_ids: tuple[str, ...]
    source_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shape contracts for the saved bank."""
        entry_count = self.labels.shape[0]
        if self.labels.ndim != 1:
            message = f"labels must have shape (B,), got {tuple(self.labels.shape)!r}."
            LOGGER.error(message)
            raise HMCConditionBankError(message)
        if self.descriptors.shape != (entry_count, self.hmc_config.descriptor_dim):
            message = (
                "descriptors must have shape "
                f"({entry_count}, {self.hmc_config.descriptor_dim}), got "
                f"{tuple(self.descriptors.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCConditionBankError(message)
        if len(self.sequences) != len(self.hmc_config.scales):
            message = (
                "sequences must contain one tensor per HMC scale. "
                f"Expected {len(self.hmc_config.scales)}, got {len(self.sequences)}."
            )
            LOGGER.error(message)
            raise HMCConditionBankError(message)
        for scale_index, (scale, sequence_tensor) in enumerate(
            zip(self.hmc_config.scales, self.sequences, strict=True)
        ):
            expected_shape = (entry_count, (2**scale) ** 3)
            if sequence_tensor.shape != expected_shape:
                message = (
                    f"sequences[{scale_index}] must have shape {expected_shape!r}, got "
                    f"{tuple(sequence_tensor.shape)!r}."
                )
                LOGGER.error(message)
                raise HMCConditionBankError(message)
        if len(self.model_ids) != entry_count or len(self.source_paths) != entry_count:
            message = "model_ids and source_paths must align with the entry count."
            LOGGER.error(message)
            raise HMCConditionBankError(message)

    def __len__(self) -> int:
        """Return the number of stored HMC condition entries."""
        return int(self.labels.shape[0])


def _serialize_hmc_config(config: HMCConfig) -> dict[str, Any]:
    """Convert `HMCConfig` into a torch-safe primitive mapping."""
    return {
        "scales": config.scales,
        "q_orders": config.q_orders,
        "normalization_mode": config.normalization_mode.value,
        "delta": config.delta,
        "empty_box_epsilon": config.empty_box_epsilon,
        "use_spectrum": config.use_spectrum,
        "spectrum_bins": config.spectrum_bins,
    }


def build_hmc_condition_bank(
    root_dir: str | Path,
    categories: Sequence[str],
    split: str,
    hmc_config: HMCConfig,
    *,
    limit: int | None = None,
) -> HMCConditionBank:
    """Build an offline HMC condition bank from ShapeNet point clouds.

    Args:
        root_dir: ShapeNetCore.v2.PC15k root directory.
        categories: Requested category names.
        split: Dataset split used to build the bank.
        hmc_config: HMC extraction settings.
        limit: Optional maximum number of bank entries.

    Returns:
        HMCConditionBank: Built bank stored in CPU tensors.
    """
    if limit is not None and limit <= 0:
        message = f"limit must be positive when provided, got {limit}."
        LOGGER.error(message)
        raise HMCConditionBankError(message)

    dataset = ShapeNetPC15KDataset(
        root_dir=root_dir,
        categories=tuple(categories),
        split=split,
        sample_size=1,
        random_subsample=False,
        return_full_points=True,
    )
    extractor = HMCFeatureExtractor(hmc_config)
    descriptors: list[Tensor] = []
    sequence_buckets: list[list[Tensor]] = [[] for _ in hmc_config.scales]
    labels: list[int] = []
    model_ids: list[str] = []
    source_paths: list[str] = []

    for index in range(len(dataset)):
        sample = dataset[index]
        full_points = torch.as_tensor(sample["full_points"], dtype=torch.float32)
        result = extractor.extract(full_points)
        descriptors.append(torch.from_numpy(result.descriptor).to(torch.float32))
        for sequence_index, sequence in enumerate(result.sequences):
            sequence_buckets[sequence_index].append(
                torch.from_numpy(sequence).to(torch.float32)
            )
        labels.append(int(sample["label"]))
        model_ids.append(str(sample["model_id"]))
        source_paths.append(str(sample["path"]))
        if limit is not None and len(labels) >= limit:
            break

    if len(labels) == 0:
        message = "No HMC conditions were extracted for the requested bank."
        LOGGER.error(message)
        raise HMCConditionBankError(message)

    return HMCConditionBank(
        hmc_config=hmc_config,
        split=split,
        categories=tuple(categories),
        labels=torch.tensor(labels, dtype=torch.int64),
        descriptors=torch.stack(descriptors, dim=0),
        sequences=tuple(torch.stack(bucket, dim=0) for bucket in sequence_buckets),
        model_ids=tuple(model_ids),
        source_paths=tuple(source_paths),
    )


def save_hmc_condition_bank(bank: HMCConditionBank, output_path: str | Path) -> Path:
    """Persist an HMC condition bank to disk.

    Args:
        bank: Condition bank to save.
        output_path: Target `.pt` path.

    Returns:
        Path: Resolved save path.
    """
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hmc_config": _serialize_hmc_config(bank.hmc_config),
            "split": bank.split,
            "categories": bank.categories,
            "labels": bank.labels,
            "descriptors": bank.descriptors,
            "sequences": bank.sequences,
            "model_ids": bank.model_ids,
            "source_paths": bank.source_paths,
        },
        resolved_path,
    )
    return resolved_path


def load_hmc_condition_bank(bank_path: str | Path) -> HMCConditionBank:
    """Load an HMC condition bank from disk.

    Args:
        bank_path: Saved `.pt` bank path.

    Returns:
        HMCConditionBank: Loaded condition bank.
    """
    resolved_path = Path(bank_path).expanduser().resolve()
    if not resolved_path.exists():
        message = f"HMC condition bank does not exist: {resolved_path!s}."
        LOGGER.error(message)
        raise HMCConditionBankError(message)
    payload = torch.load(resolved_path, map_location="cpu", weights_only=True)
    required_keys = {
        "hmc_config",
        "split",
        "categories",
        "labels",
        "descriptors",
        "sequences",
        "model_ids",
        "source_paths",
    }
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        message = f"HMC condition bank is missing keys: {sorted(missing_keys)!r}."
        LOGGER.error(message)
        raise HMCConditionBankError(message)
    return HMCConditionBank(
        hmc_config=HMCConfig(**payload["hmc_config"]),
        split=str(payload["split"]),
        categories=tuple(payload["categories"]),
        labels=torch.as_tensor(payload["labels"], dtype=torch.int64),
        descriptors=torch.as_tensor(payload["descriptors"], dtype=torch.float32),
        sequences=tuple(
            torch.as_tensor(sequence, dtype=torch.float32)
            for sequence in payload["sequences"]
        ),
        model_ids=tuple(str(model_id) for model_id in payload["model_ids"]),
        source_paths=tuple(str(path) for path in payload["source_paths"]),
    )


def sample_hmc_condition_bank(
    bank: HMCConditionBank,
    num_samples: int,
    *,
    categories: Sequence[str] | None = None,
    replacement: bool = True,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Sample HMC condition entries from a saved bank.

    Args:
        bank: Offline condition bank.
        num_samples: Number of entries to sample.
        categories: Optional category subset filter.
        replacement: Whether sampling may repeat entries.
        generator: Optional torch generator for reproducibility.
        device: Target device for returned tensors.

    Returns:
        dict[str, Any]: Sampled labels, descriptors, sequences, and metadata.
    """
    if num_samples <= 0:
        message = f"num_samples must be positive, got {num_samples}."
        LOGGER.error(message)
        raise HMCConditionBankError(message)

    eligible_indices = torch.arange(len(bank), dtype=torch.int64)
    if categories is not None:
        category_set = {str(category) for category in categories}
        category_labels = {
            index
            for index, category in enumerate(bank.categories)
            if category in category_set
        }
        eligible_mask = torch.tensor(
            [int(label.item()) in category_labels for label in bank.labels],
            dtype=torch.bool,
        )
        eligible_indices = eligible_indices[eligible_mask]
    if eligible_indices.numel() == 0:
        message = "No HMC condition-bank entries match the requested filter."
        LOGGER.error(message)
        raise HMCConditionBankError(message)
    if not replacement and num_samples > eligible_indices.numel():
        message = (
            "num_samples exceeds the number of eligible entries while "
            "replacement=False."
        )
        LOGGER.error(message)
        raise HMCConditionBankError(message)

    if replacement:
        draw_indices = torch.randint(
            0,
            eligible_indices.numel(),
            (num_samples,),
            generator=generator,
        )
        bank_indices = eligible_indices[draw_indices]
    else:
        permutation = torch.randperm(
            eligible_indices.numel(),
            generator=generator,
        )[:num_samples]
        bank_indices = eligible_indices[permutation]

    resolved_device = torch.device(device)
    selected_indices = bank_indices.tolist()
    sampled_labels = bank.labels[bank_indices].to(resolved_device)
    sampled_descriptors = bank.descriptors[bank_indices].to(resolved_device)
    sampled_sequences = [
        sequence[bank_indices].to(resolved_device) for sequence in bank.sequences
    ]
    sampled_categories = [
        bank.categories[int(label.item())] for label in sampled_labels.cpu()
    ]
    sampled_model_ids = [bank.model_ids[index] for index in selected_indices]
    sampled_source_paths = [bank.source_paths[index] for index in selected_indices]
    return {
        "indices": bank_indices.to(resolved_device),
        "labels": sampled_labels,
        "descriptors": sampled_descriptors,
        "sequences": sampled_sequences,
        "categories": sampled_categories,
        "model_ids": sampled_model_ids,
        "source_paths": sampled_source_paths,
    }
