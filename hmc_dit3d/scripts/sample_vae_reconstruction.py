"""Bank-conditioned sampling with conditions passed through the VAE bottleneck.

Reproduces the exact bank evaluation protocol (same bank, same seed, same
paired references) except that each drawn HMC condition is first encoded to
the VAE posterior mean and decoded back. Comparing against the plain bank run
isolates the effect of inserting the VAE into the conditioning path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hmc_dit3d.data.hmc_condition_bank import (
    load_hmc_condition_bank,
    sample_hmc_condition_bank,
)
from hmc_dit3d.hmc.condition_vae import load_condition_vae_checkpoint
from hmc_dit3d.train.config import load_experiment_config
from hmc_dit3d.train.sample import (
    _generate_batch_samples,
    _load_reference_points_from_paths,
    load_sampling_checkpoint,
    save_sample_payload,
)
from hmc_dit3d.utils.runtime import detect_device, set_global_seed

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--sequence-threshold",
        type=float,
        default=0.1,
        help="Relative sparsification threshold applied to decoded sequences.",
    )
    parser.add_argument(
        "--topk-occupancy",
        action="store_true",
        help=(
            "Keep only the top-K measure entries per scale, with K equal to the "
            "mean occupied-voxel count in the evaluation bank."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    set_global_seed(config.train.seed)
    device = detect_device(config.train.prefer_cuda)
    model, diffusion, checkpoint = load_sampling_checkpoint(
        config, args.checkpoint, device
    )
    vae, _ = load_condition_vae_checkpoint(args.vae, device=device)

    bank = load_hmc_condition_bank(args.bank)
    generator = torch.Generator().manual_seed(config.train.seed)
    conditions = sample_hmc_condition_bank(
        bank,
        num_samples=args.limit,
        replacement=True,
        generator=generator,
        device=device,
    )

    with torch.no_grad():
        mu, _ = vae.encode(conditions["descriptors"], conditions["sequences"])
        descriptor_norm, sequence_logits = vae.decode(mu)
        descriptors = descriptor_norm * vae.descriptor_std + vae.descriptor_mean
        topk_ks: list[int] | None = None
        if args.topk_occupancy:
            topk_ks = [
                max(1, int(round(float((sequence > 0).float().sum(dim=-1).mean()))))
                for sequence in bank.sequences
            ]
            LOGGER.info("Using top-k occupancy matching with K=%s", topk_ks)

        sequences = []
        for scale_index, (logits, length) in enumerate(
            zip(sequence_logits, vae.sequence_lengths, strict=True)
        ):
            measures = torch.softmax(logits, dim=-1)
            if topk_ks is not None:
                k = min(topk_ks[scale_index], length)
                values, indices = torch.topk(measures, k=k, dim=-1)
                sparse = torch.zeros_like(measures)
                sparse.scatter_(dim=-1, index=indices, src=values)
                measures = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            elif args.sequence_threshold > 0:
                cutoff = args.sequence_threshold / float(length)
                measures = torch.where(
                    measures >= cutoff, measures, torch.zeros_like(measures)
                )
                measures = measures / measures.sum(dim=-1, keepdim=True).clamp_min(
                    1e-12
                )
            sequences.append(measures)

        samples = _generate_batch_samples(
            diffusion,
            model,
            conditions["labels"],
            descriptors,
            sequences,
            point_count=config.data.sample_size,
            device=device,
            clip_denoised=False,
        )

    references = _load_reference_points_from_paths(
        conditions["source_paths"], config.data.sample_size
    )
    payload = {
        "samples": samples,
        "references": references,
        "labels": conditions["labels"].cpu(),
        "categories": conditions["categories"],
        "model_ids": conditions["model_ids"],
        "source_paths": conditions["source_paths"],
        "bank_path": str(args.bank),
        "vae_path": str(args.vae),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "condition_source": "vae_reconstruction",
        "config_path": str(args.config),
    }
    output = save_sample_payload(payload, args.output)
    LOGGER.info("Saved %d VAE-reconstruction samples to %s", args.limit, output)


if __name__ == "__main__":
    main()
