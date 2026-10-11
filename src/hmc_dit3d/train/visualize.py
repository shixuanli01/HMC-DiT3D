"""Visualization CLI for saved HMC-DiT3D sample payloads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hmc_dit3d.train.evaluate import load_sample_payload
from hmc_dit3d.visualization.pointcloud import render_sample_comparison_figure

LOGGER = logging.getLogger(__name__)


class VisualizationCliError(RuntimeError):
    """Raised when visualization CLI arguments are invalid."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sample visualization.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Render saved HMC-DiT3D sample payloads into comparison figures.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        type=Path,
        required=True,
        help="One or more saved .pt payloads produced by train.sample.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional row labels matching --samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .png path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of sample pairs shown per row.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=30.0,
        help="Camera elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=225.0,
        help="Camera azimuth angle.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Scatter marker size.",
    )
    return parser.parse_args()


def _resolve_row_labels(
    sample_paths: list[Path],
    labels: list[str] | None,
) -> list[str]:
    """Resolve row labels from explicit labels or payload metadata.

    Args:
        sample_paths: Sample payload paths.
        labels: Optional explicit row labels.

    Returns:
        list[str]: Row labels aligned with `sample_paths`.
    """
    if labels is None:
        return [path.stem for path in sample_paths]
    if len(labels) != len(sample_paths):
        message = "--labels must have the same length as --samples."
        LOGGER.error(message)
        raise VisualizationCliError(message)
    return [str(label) for label in labels]


def main() -> None:
    """CLI entrypoint for saved-sample visualization."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    row_labels = _resolve_row_labels(args.samples, args.labels)
    rows: list[dict[str, torch.Tensor | str]] = []
    for sample_path, row_label in zip(args.samples, row_labels, strict=True):
        payload = load_sample_payload(sample_path)
        rows.append(
            {
                "name": row_label,
                "samples": torch.as_tensor(payload["samples"]),
                "references": torch.as_tensor(payload["references"]),
            }
        )
    output_path = render_sample_comparison_figure(
        rows,
        args.output,
        num_samples=args.num_samples,
        elev=args.elev,
        azim=args.azim,
        point_size=args.point_size,
    )
    LOGGER.info("Saved visualization to %s", output_path)


if __name__ == "__main__":
    main()
