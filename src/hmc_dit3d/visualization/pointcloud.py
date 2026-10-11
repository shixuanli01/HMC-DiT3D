"""Point-cloud rendering helpers for saved HMC-DiT3D samples."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

LOGGER = logging.getLogger(__name__)

REFERENCE_COLOR = "#1f77b4"
GENERATED_COLOR = "#d62728"


class VisualizationError(RuntimeError):
    """Raised when point-cloud visualization input is invalid."""


def _to_numpy_points(points: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert one point cloud to a validated NumPy array.

    Args:
        points: Point cloud with shape `(N, 3)`.

    Returns:
        np.ndarray: Float32 point cloud with shape `(N, 3)`.
    """
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        message = f"Point cloud must have shape (N, 3), got {array.shape!r}."
        LOGGER.error(message)
        raise VisualizationError(message)
    if not np.isfinite(array).all():
        message = "Point cloud contains NaN or Inf values."
        LOGGER.error(message)
        raise VisualizationError(message)
    return array


def _compute_plot_limit(rows: list[dict[str, torch.Tensor]], num_samples: int) -> float:
    """Compute one shared symmetric axis limit for all rendered subplots.

    Args:
        rows: Sample payload rows.
        num_samples: Number of samples visualized per row.

    Returns:
        float: Shared half-range for axis limits.
    """
    clouds: list[np.ndarray] = []
    for row in rows:
        sample_count = min(num_samples, int(row["samples"].shape[0]))
        for sample_index in range(sample_count):
            clouds.append(_to_numpy_points(row["references"][sample_index]))
            clouds.append(_to_numpy_points(row["samples"][sample_index]))
    if not clouds:
        message = "No point clouds are available for visualization."
        LOGGER.error(message)
        raise VisualizationError(message)
    stacked = np.concatenate(clouds, axis=0)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    limit = float(np.abs(centered).max())
    return max(limit * 1.05, 1.0e-3)


def _render_single_cloud(
    axis: Axes,
    points: np.ndarray,
    *,
    title: str,
    color: str,
    limit: float,
    elev: float,
    azim: float,
    point_size: float,
) -> None:
    """Render one point cloud onto a Matplotlib 3D axis.

    Args:
        axis: Target Matplotlib axis.
        points: Point cloud with shape `(N, 3)`.
        title: Subplot title.
        color: Scatter color.
        limit: Shared symmetric axis range.
        elev: Camera elevation angle.
        azim: Camera azimuth angle.
        point_size: Scatter marker size.
    """
    centered = points - points.mean(axis=0, keepdims=True)
    axis.scatter(
        centered[:, 0],
        centered[:, 2],
        centered[:, 1],
        c=color,
        s=point_size,
        linewidths=0.0,
        alpha=0.9,
    )
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.view_init(elev=elev, azim=azim)
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
    try:
        axis.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        # Matplotlib versions without `set_box_aspect` can still render correctly.
        pass


def render_sample_comparison_figure(
    rows: list[dict[str, torch.Tensor | str]],
    output_path: str | Path,
    *,
    num_samples: int = 4,
    elev: float = 30.0,
    azim: float = 225.0,
    point_size: float = 1.5,
) -> Path:
    """Render multiple saved-sample payloads into one comparison image.

    Each row corresponds to one method or condition source. Columns are arranged
    as `(reference_i, generated_i)` pairs so qualitative differences are easy to
    inspect side by side.

    Args:
        rows: Visualization rows containing `name`, `samples`, and `references`.
        output_path: Output `.png` path.
        num_samples: Number of sample pairs shown per row.
        elev: Camera elevation angle.
        azim: Camera azimuth angle.
        point_size: Scatter marker size.

    Returns:
        Path: Resolved image path.
    """
    if len(rows) == 0:
        message = "At least one visualization row is required."
        LOGGER.error(message)
        raise VisualizationError(message)
    if num_samples <= 0:
        message = "num_samples must be positive."
        LOGGER.error(message)
        raise VisualizationError(message)
    if point_size <= 0.0:
        message = "point_size must be positive."
        LOGGER.error(message)
        raise VisualizationError(message)

    typed_rows: list[dict[str, torch.Tensor]] = []
    row_names: list[str] = []
    for row in rows:
        name = str(row["name"])
        samples = torch.as_tensor(row["samples"]).detach().cpu()
        references = torch.as_tensor(row["references"]).detach().cpu()
        if samples.ndim != 3 or samples.shape[-1] != 3:
            message = (
                "samples must have shape (B, N, 3), "
                f"got {tuple(samples.shape)!r} for row {name!r}."
            )
            LOGGER.error(message)
            raise VisualizationError(message)
        if references.ndim != 3 or references.shape[-1] != 3:
            message = (
                "references must have shape (B, N, 3), "
                f"got {tuple(references.shape)!r} for row {name!r}."
            )
            LOGGER.error(message)
            raise VisualizationError(message)
        if samples.shape[0] != references.shape[0]:
            message = (
                "samples and references must share batch size. "
                f"Got {samples.shape[0]} and {references.shape[0]} for row {name!r}."
            )
            LOGGER.error(message)
            raise VisualizationError(message)
        if samples.shape[0] == 0:
            message = f"Visualization row {name!r} has no samples."
            LOGGER.error(message)
            raise VisualizationError(message)
        typed_rows.append({"samples": samples, "references": references})
        row_names.append(name)

    limit = _compute_plot_limit(typed_rows, num_samples)
    columns = num_samples * 2
    figure = plt.figure(figsize=(columns * 2.6, len(typed_rows) * 2.8))

    for row_index, (row_name, row_payload) in enumerate(
        zip(row_names, typed_rows, strict=True)
    ):
        visible_samples = min(num_samples, int(row_payload["samples"].shape[0]))
        for sample_index in range(visible_samples):
            reference_axis = figure.add_subplot(
                len(typed_rows),
                columns,
                row_index * columns + sample_index * 2 + 1,
                projection=Axes3D.name,
            )
            generated_axis = figure.add_subplot(
                len(typed_rows),
                columns,
                row_index * columns + sample_index * 2 + 2,
                projection=Axes3D.name,
            )
            prefix = f"{row_name}\n" if sample_index == 0 else ""
            _render_single_cloud(
                reference_axis,
                _to_numpy_points(row_payload["references"][sample_index]),
                title=f"{prefix}ref {sample_index + 1}",
                color=REFERENCE_COLOR,
                limit=limit,
                elev=elev,
                azim=azim,
                point_size=point_size,
            )
            _render_single_cloud(
                generated_axis,
                _to_numpy_points(row_payload["samples"][sample_index]),
                title=f"gen {sample_index + 1}",
                color=GENERATED_COLOR,
                limit=limit,
                elev=elev,
                azim=azim,
                point_size=point_size,
            )
        for sample_index in range(visible_samples, num_samples):
            for column_offset in (1, 2):
                axis = figure.add_subplot(
                    len(typed_rows),
                    columns,
                    row_index * columns + sample_index * 2 + column_offset,
                )
                axis.axis("off")

    figure.tight_layout()
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return resolved_path
