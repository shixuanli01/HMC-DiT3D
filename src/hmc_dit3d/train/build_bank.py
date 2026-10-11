"""CLI helpers for building offline HMC condition banks from experiment configs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hmc_dit3d.data.hmc_condition_bank import (
    build_hmc_condition_bank,
    save_hmc_condition_bank,
)
from hmc_dit3d.train.config import load_experiment_config

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for bank construction."""
    parser = argparse.ArgumentParser(
        description="Build an offline HMC condition bank from a training config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path used to define dataset and HMC extraction settings.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split used to build the HMC bank.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt path for the saved HMC bank.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of entries extracted into the bank.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for building offline HMC condition banks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_experiment_config(args.config)
    bank = build_hmc_condition_bank(
        root_dir=config.data.root_dir,
        categories=config.data.categories,
        split=args.split,
        hmc_config=config.hmc,
        limit=args.limit,
    )
    output_path = save_hmc_condition_bank(bank, args.output)
    LOGGER.info(
        "Saved HMC condition bank with %d entries to %s",
        len(bank),
        output_path,
    )


if __name__ == "__main__":
    main()
