"""Generate 664 samples for the validation protocol with a chosen HMC condition source.

Condition sources:
  vae        sample conditions from the category's condition VAE (N(0, I) prior)   -> "Ours-VAE"
  trainbank  draw real train-split conditions from the HMC condition bank            -> "Ours-Bank"

The reference clouds stored in the payload are test-split shapes; the validation-protocol
scorer (scripts/evaluate_val_protocol.py) replaces them with the val split, so they only
serve as provenance. `--category` is for multi-class checkpoints: it restricts the bank
entries, the class label and the references to one category.

Example:
  python scripts/generate.py --config configs/retrain_airplane_norm_bs512.yaml \
      --checkpoint ../checkpoints/dit/airplane_norm_epoch8000.pt --mode trainbank \
      --bank results/banks/airplane_train.pt --seed 0 --fast --out results/airplane_bank_seed0.pt
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

import hmc_dit3d.train.sample as S
from hmc_dit3d.data.hmc_condition_bank import (
    _serialize_hmc_config,
    load_hmc_condition_bank,
)
from hmc_dit3d.data.shapenet_pc15k import ShapeNetPC15KDataset
from hmc_dit3d.train.config import load_experiment_config

NUM_SAMPLES = 664


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="training YAML of the checkpoint")
    p.add_argument("--checkpoint", required=True, help="DiT checkpoint (.pt)")
    p.add_argument("--mode", choices=["vae", "trainbank"], required=True)
    p.add_argument(
        "--bank",
        required=True,
        help="train-split HMC bank (.pt), see hmc_dit3d.train.build_bank",
    )
    p.add_argument(
        "--vae", default=None, help="condition VAE checkpoint (required for --mode vae)"
    )
    p.add_argument("--out", required=True, help="output payload (.pt)")
    p.add_argument(
        "--seed", type=int, default=0, help="overrides train.seed of the config"
    )
    p.add_argument("--gen-bs", type=int, default=166, help="generation batch size")
    p.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="HMC guidance scale (1.0 = plain conditional sampling)",
    )
    p.add_argument(
        "--temperature", type=float, default=1.2, help="VAE prior temperature"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="VAE sequence sparsification threshold",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="fp16 autocast + non-deterministic kernels (~3x faster; validated against fp32)",
    )
    p.add_argument(
        "--category",
        default=None,
        help="multi-class checkpoints: restrict to this category",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.mode == "vae" and not a.vae:
        raise SystemExit("--vae is required for --mode vae")
    cfg = load_experiment_config(Path(a.config))
    cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=a.seed))

    if a.fast:
        _seed, _load = S.set_global_seed, S.load_sampling_checkpoint

        def seed_fast(seed, *x, **k):
            _seed(seed, deterministic=False)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        def load_fast(*x, **k):
            model, diffusion, ckpt = _load(*x, **k)
            forward = model.forward

            def fast_forward(*xx, **kk):
                with torch.autocast("cuda", dtype=torch.float16):
                    return forward(*xx, **kk).float()

            model.forward = fast_forward
            return model, diffusion, ckpt

        S.set_global_seed, S.load_sampling_checkpoint = seed_fast, load_fast

    if a.category is not None:
        label = list(cfg.data.categories).index(a.category)

        class CategoryDataset(ShapeNetPC15KDataset):
            """References from one category, labels remapped to the checkpoint's class index."""

            def __init__(self, *x, **k):
                k["categories"] = [a.category]
                super().__init__(*x, **k)

            def __getitem__(self, i):
                d = super().__getitem__(i)
                d["label"] = label
                return d

        S.ShapeNetPC15KDataset = CategoryDataset

    bank = load_hmc_condition_bank(a.bank)

    class BankConditions:
        """Stands in for the VAE: returns real train-split conditions instead of sampling."""

        def sample(self, n, device, **kw):
            if a.category is not None:
                pool = torch.nonzero(
                    bank.labels == list(bank.categories).index(a.category)
                ).flatten()
            else:
                pool = torch.arange(len(bank))
            idx = pool[
                torch.randperm(
                    len(pool),
                    generator=torch.Generator().manual_seed(cfg.train.seed + 1),
                )[:n]
            ]
            return {
                "descriptors": bank.descriptors[idx].to(device),
                "sequences": [x[idx].to(device) for x in bank.sequences],
            }

    if a.mode == "trainbank":
        # no VAE is loaded in bank mode; the stub satisfies the HMC-config check in the sampler
        S.load_condition_vae_checkpoint = lambda *x, **k: (
            BankConditions(),
            {"hmc_config": _serialize_hmc_config(cfg.hmc)},
        )

    payload = S.generate_vae_conditioned_samples(
        cfg,
        Path(a.checkpoint),
        Path(a.vae or "/dev/null"),
        split="test",
        limit=NUM_SAMPLES,
        vae_temperature=a.temperature,
        vae_sequence_threshold=a.threshold,
        generation_batch_size=a.gen_bs,
        hmc_guidance_scale=a.guidance,
    )
    payload.update(
        {
            "condition_mode": a.mode,
            "fast": a.fast,
            "seed": cfg.train.seed,
            "category": a.category,
        }
    )
    S.save_sample_payload(payload, Path(a.out))
    print("saved", a.out)


if __name__ == "__main__":
    main()
