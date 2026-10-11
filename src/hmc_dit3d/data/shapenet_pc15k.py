"""ShapeNetCore.v2.PC15k dataset utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

SYNSETID_TO_CATEGORY: Final[dict[str, str]] = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
}
CATEGORY_TO_SYNSETID: Final[dict[str, str]] = {
    category: synset_id for synset_id, category in SYNSETID_TO_CATEGORY.items()
}
VALID_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


class ShapeNetDataError(RuntimeError):
    """Raised when ShapeNetCore.v2.PC15k data is invalid or unavailable."""


@dataclass(slots=True, frozen=True)
class ShapeNetSample:
    """Metadata for one ShapeNet point-cloud file.

    Args:
        category: Human-readable category name.
        synset_id: ShapeNet synset identifier.
        model_id: ShapeNet model identifier without suffix.
        path: Absolute path to the `.npy` file.
    """

    category: str
    synset_id: str
    model_id: str
    path: Path


class ShapeNetPC15KDataset(Dataset[dict[str, object]]):
    """Load PointFlow-style ShapeNetCore.v2.PC15k point clouds."""

    def __init__(
        self,
        root_dir: str | Path,
        categories: tuple[str, ...] | list[str],
        split: str = "train",
        sample_size: int = 2048,
        random_subsample: bool = True,
        return_full_points: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Dataset root containing synset folders.
            categories: Category names such as `chair`, `airplane`, `car`.
            split: One of `train`, `val`, `test`.
            sample_size: Number of points returned per sample.
            random_subsample: Whether to subsample points randomly each access.
            return_full_points: Whether to include the full 15k points in outputs.
        """
        super().__init__()
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.categories = tuple(categories)
        self.split = split
        self.sample_size = sample_size
        self.random_subsample = random_subsample
        self.return_full_points = return_full_points
        self.category_to_label = {
            category: label for label, category in enumerate(self.categories)
        }

        self._validate_init_args()
        self.samples = self._collect_samples()

    def _validate_init_args(self) -> None:
        """Validate dataset constructor arguments."""
        if not self.root_dir.exists():
            message = f"ShapeNet root directory does not exist: {self.root_dir!s}."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if not self.root_dir.is_dir():
            message = f"ShapeNet root path is not a directory: {self.root_dir!s}."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if self.split not in VALID_SPLITS:
            message = f"split must be one of {VALID_SPLITS!r}, got {self.split!r}."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if len(self.categories) == 0:
            message = "categories must contain at least one category."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        unknown_categories = [
            category
            for category in self.categories
            if category not in CATEGORY_TO_SYNSETID
        ]
        if unknown_categories:
            message = f"Unknown ShapeNet categories requested: {unknown_categories!r}."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if len(set(self.categories)) != len(self.categories):
            message = "categories must be unique."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if self.sample_size <= 0:
            message = "sample_size must be positive."
            LOGGER.error(message)
            raise ShapeNetDataError(message)

    def _collect_samples(self) -> tuple[ShapeNetSample, ...]:
        """Collect all `.npy` point-cloud samples for the requested categories."""
        samples: list[ShapeNetSample] = []
        for category in self.categories:
            synset_id = CATEGORY_TO_SYNSETID[category]
            split_dir = self.root_dir / synset_id / self.split
            if not split_dir.exists():
                message = f"Missing split directory: {split_dir!s}."
                LOGGER.error(message)
                raise ShapeNetDataError(message)
            if not split_dir.is_dir():
                message = f"Split path is not a directory: {split_dir!s}."
                LOGGER.error(message)
                raise ShapeNetDataError(message)
            for path in sorted(split_dir.glob("*.npy")):
                samples.append(
                    ShapeNetSample(
                        category=category,
                        synset_id=synset_id,
                        model_id=path.stem,
                        path=path.resolve(),
                    )
                )
        if len(samples) == 0:
            message = (
                "No ShapeNet samples were found for the requested categories and split."
            )
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        return tuple(samples)

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.samples)

    def _load_points(self, path: Path) -> np.ndarray:
        """Load and validate one raw point cloud.

        Args:
            path: Absolute `.npy` file path.

        Returns:
            np.ndarray: Array with shape `(15000, 3)` and dtype `float32`.
        """
        points = np.load(path)
        if points.shape != (15000, 3):
            message = (
                "ShapeNet point cloud must have shape (15000, 3). "
                f"Got {points.shape!r} from {path!s}."
            )
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if not np.isfinite(points).all():
            message = f"ShapeNet point cloud contains NaN or Inf: {path!s}."
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        if self.sample_size > points.shape[0]:
            message = (
                f"sample_size={self.sample_size} exceeds the available points "
                f"{points.shape[0]} for {path!s}."
            )
            LOGGER.error(message)
            raise ShapeNetDataError(message)
        return points.astype(np.float32, copy=False)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """Subsample points according to dataset settings.

        Args:
            points: Full point cloud with shape `(15000, 3)`.

        Returns:
            np.ndarray: Sampled point cloud with shape `(sample_size, 3)`.
        """
        if self.sample_size == points.shape[0]:
            return points
        if self.random_subsample:
            indices = torch.randperm(points.shape[0])[: self.sample_size].numpy()
        else:
            indices = np.arange(self.sample_size)
        return np.ascontiguousarray(points[indices])

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one sampled point cloud and its metadata.

        Args:
            index: Dataset index.

        Returns:
            dict[str, object]: Sample payload.
        """
        sample = self.samples[index]
        full_points = self._load_points(sample.path)
        sampled_points = self._sample_points(full_points)
        output: dict[str, object] = {
            "index": index,
            "points": torch.from_numpy(sampled_points),
            "label": self.category_to_label[sample.category],
            "category": sample.category,
            "synset_id": sample.synset_id,
            "model_id": sample.model_id,
            "path": str(sample.path),
        }
        if self.return_full_points:
            output["full_points"] = torch.from_numpy(np.ascontiguousarray(full_points))
        return output
