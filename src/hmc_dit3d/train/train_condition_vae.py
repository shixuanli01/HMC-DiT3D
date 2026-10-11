"""Train a VAE prior over raw HMC conditions from an offline condition bank."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from hmc_dit3d.data.hmc_condition_bank import load_hmc_condition_bank
from hmc_dit3d.hmc.condition_vae import (
    HMCConditionVAE,
    save_condition_vae_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for condition-VAE training."""
    parser = argparse.ArgumentParser(
        description="Train a VAE prior over raw HMC conditions.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        required=True,
        help="Offline HMC condition bank (.pt) used as the training set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt path for the trained condition VAE.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="2048,1024,512",
        help="Comma-separated encoder widths (decoder mirrors them).",
    )
    parser.add_argument("--beta", type=float, default=0.02, help="KL weight.")
    parser.add_argument("--descriptor-weight", type=float, default=1.0)
    parser.add_argument(
        "--sequence-weights",
        type=str,
        default=None,
        help="Optional comma-separated per-scale CE weights, e.g. 1,1,2.",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for condition-VAE training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    bank = load_hmc_condition_bank(args.bank)
    descriptors = bank.descriptors.to(device)
    sequences = [sequence.to(device) for sequence in bank.sequences]
    LOGGER.info(
        "Loaded bank with %d entries | descriptor_dim=%d | sequence_lengths=%s",
        len(bank),
        descriptors.shape[1],
        [sequence.shape[1] for sequence in sequences],
    )

    model = HMCConditionVAE(
        descriptor_dim=descriptors.shape[1],
        sequence_lengths=tuple(sequence.shape[1] for sequence in sequences),
        latent_dim=args.latent_dim,
        hidden_dims=tuple(int(w) for w in args.hidden_dims.split(",")),
    ).to(device)
    model.set_descriptor_stats(
        descriptors.mean(dim=0),
        descriptors.std(dim=0),
    )

    dataset = TensorDataset(torch.arange(len(bank)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * max(1, len(loader)),
    )

    model.train()
    step = 0
    for epoch in range(args.epochs):
        epoch_losses: dict[str, float] = {}
        for (batch_indices,) in loader:
            batch_descriptors = descriptors[batch_indices]
            batch_sequences = [sequence[batch_indices] for sequence in sequences]
            output = model(batch_descriptors, batch_sequences)
            sequence_weights = None
            if args.sequence_weights:
                sequence_weights = tuple(
                    float(x) for x in args.sequence_weights.split(",")
                )
            losses = model.loss(
                output,
                batch_descriptors,
                batch_sequences,
                beta=args.beta,
                descriptor_weight=args.descriptor_weight,
                sequence_weights=sequence_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + float(value.detach())
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            summary = " | ".join(
                f"{key}={value / len(loader):.6f}"
                for key, value in sorted(epoch_losses.items())
            )
            LOGGER.info("epoch=%d step=%d | %s", epoch + 1, step, summary)

    model.eval()
    with torch.no_grad():
        prior_samples = model.sample(512, device=device)
        prior_descriptors = prior_samples["descriptors"]
        LOGGER.info(
            "Prior descriptor mean=%s std=%s | data mean=%s std=%s",
            [round(v, 4) for v in prior_descriptors.mean(dim=0).tolist()],
            [round(v, 4) for v in prior_descriptors.std(dim=0).tolist()],
            [round(v, 4) for v in descriptors.mean(dim=0).tolist()],
            [round(v, 4) for v in descriptors.std(dim=0).tolist()],
        )
        for scale_index, generated in enumerate(prior_samples["sequences"]):
            data_sequence = sequences[scale_index]
            data_occupancy = (data_sequence > 0).float().sum(dim=-1).mean()
            # Effective support size via exponential of entropy (perplexity).
            generated_entropy = -(generated.clamp_min(1e-12).log() * generated).sum(
                dim=-1
            )
            data_entropy = -(data_sequence.clamp_min(1e-12).log() * data_sequence).sum(
                dim=-1
            ) * (data_sequence > 0).any(dim=-1)
            LOGGER.info(
                "scale[%d]: data occupancy=%.1f | data perplexity=%.1f | "
                "prior perplexity=%.1f",
                scale_index,
                float(data_occupancy),
                float(data_entropy.exp().mean()),
                float(generated_entropy.exp().mean()),
            )

    from hmc_dit3d.data.hmc_condition_bank import _serialize_hmc_config

    output_path = save_condition_vae_checkpoint(
        model,
        args.output,
        hmc_config=_serialize_hmc_config(bank.hmc_config),
        extra={
            "bank_path": str(Path(args.bank).expanduser().resolve()),
            "bank_split": bank.split,
            "bank_categories": bank.categories,
            "bank_size": len(bank),
            "beta": args.beta,
            "epochs": args.epochs,
            "seed": args.seed,
        },
    )
    LOGGER.info("Saved condition VAE to %s", output_path)


if __name__ == "__main__":
    main()
