"""Tests for sample visualization utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from hmc_dit3d.visualization.pointcloud import render_sample_comparison_figure


def _build_cloud(offset: float) -> torch.Tensor:
    """Create one tiny synthetic point cloud for rendering tests."""
    return torch.tensor(
        [
            [offset + 0.0, 0.0, 0.0],
            [offset + 0.1, 0.1, 0.0],
            [offset + 0.0, 0.0, 0.1],
            [offset - 0.1, -0.1, 0.0],
        ],
        dtype=torch.float32,
    )


def test_render_sample_comparison_figure_writes_png(tmp_path: Path) -> None:
    """Rendering should export a non-empty PNG for valid sample payloads."""
    rows = [
        {
            "name": "baseline-ref",
            "references": torch.stack([_build_cloud(0.0), _build_cloud(0.2)], dim=0),
            "samples": torch.stack([_build_cloud(0.05), _build_cloud(0.25)], dim=0),
        },
        {
            "name": "lmf-ref",
            "references": torch.stack([_build_cloud(-0.1), _build_cloud(0.3)], dim=0),
            "samples": torch.stack([_build_cloud(-0.05), _build_cloud(0.35)], dim=0),
        },
    ]
    output_path = tmp_path / "comparison.png"

    resolved_path = render_sample_comparison_figure(
        rows,
        output_path,
        num_samples=2,
        point_size=8.0,
    )

    assert resolved_path == output_path.resolve()
    assert output_path.exists()
    assert output_path.stat().st_size > 0
