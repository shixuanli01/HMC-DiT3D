"""Hyperparameter sweep for HMC condition VAE on one category.

Trains several VAE variants, evaluates paired reconstruction (paper protocol)
and unpaired prior sampling, and prints a compact comparison table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_metrics(samples: Path) -> dict[str, float]:
    out = subprocess.check_output(
        [
            "micromamba",
            "run",
            "-n",
            "hmc-dit3d-py312",
            "python",
            "-m",
            "hmc_dit3d.train.evaluate",
            "--samples",
            str(samples),
            "--batch-size",
            "8",
            "--device",
            "cuda",
            "--jsd-resolution",
            "8",
        ],
        text=True,
    )
    # evaluate logs a JSON block; recover key lines
    metrics: dict[str, float] = {}
    for line in out.splitlines():
        line = line.strip().rstrip(",")
        for key in (
            "1-NN-CD-acc",
            "1-NN-EMD-acc",
            "lgan_cov-CD",
            "lgan_cov-EMD",
            "lgan_mmd-CD",
            "lgan_mmd-EMD",
            "JSD",
        ):
            if line.startswith(f'"{key}"'):
                metrics[key] = float(line.split(":")[1].strip().rstrip(","))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=("airplane", "car"))
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cat = args.category
    cfg = f"configs/train_{cat}_h100_formal_bottleneck.yaml"
    bank_full = (
        f"results/train_{cat}_h100_formal_bottleneck/vae/{cat}_train_bank_full.pt"
    )
    eval_bank = (
        f"results/train_{cat}_h100_formal_bottleneck/test_eval/"
        f"{cat}_train_bank_eval128.pt"
    )
    out_dir = Path(f"results/train_{cat}_h100_formal_bottleneck/vae/tune")
    out_dir.mkdir(parents=True, exist_ok=True)

    # (tag, beta, latent, hidden, epochs, recon_thr)
    configs = [
        ("b0.01_z128_h3k", 0.01, 128, "3072,1536,768", 3000, 0.0),
        ("b0.005_z128_h3k", 0.005, 128, "3072,1536,768", 3000, 0.0),
        ("b0.01_z256_h4k", 0.01, 256, "4096,2048,1024", 3000, 0.0),
        ("b0.02_z128_h3k", 0.02, 128, "3072,1536,768", 3000, 0.0),
        ("b0.005_z64_h2k", 0.005, 64, "2048,1024,512", 3000, 0.0),
        # keep a thr=0.1 variant of the best-looking capacity for prior sampling later
        ("b0.01_z128_h3k_thr01", 0.01, 128, "3072,1536,768", 3000, 0.1),
    ]

    results = []
    py = ["micromamba", "run", "-n", "hmc-dit3d-py312", "python"]

    for tag, beta, z, hidden, epochs, thr in configs:
        vae_path = out_dir / f"{cat}_condition_vae_{tag}.pt"
        # thr is sampling-only; share weights for thr01 with base when possible
        train_tag = tag.replace("_thr01", "")
        train_path = out_dir / f"{cat}_condition_vae_{train_tag}.pt"
        if not train_path.exists() or tag == train_tag:
            run(
                py
                + [
                    "-m",
                    "hmc_dit3d.train.train_condition_vae",
                    "--bank",
                    bank_full,
                    "--output",
                    str(train_path),
                    "--epochs",
                    str(epochs),
                    "--beta",
                    str(beta),
                    "--latent-dim",
                    str(z),
                    "--hidden-dims",
                    hidden,
                    "--device",
                    "cuda",
                    "--log-every",
                    "500",
                ]
            )
        if tag != train_tag:
            # symlink / copy pointer: reuse trained weights
            if not vae_path.exists():
                vae_path.write_bytes(train_path.read_bytes())
        else:
            vae_path = train_path

        recon_out = out_dir / f"vae_recon_{tag}.pt"
        run(
            py
            + [
                "scripts/sample_vae_reconstruction.py",
                "--config",
                cfg,
                "--checkpoint",
                args.checkpoint,
                "--bank",
                eval_bank,
                "--vae",
                str(vae_path if tag == train_tag else train_path),
                "--limit",
                "64",
                "--sequence-threshold",
                str(thr),
                "--output",
                str(recon_out),
            ]
        )
        recon_m = eval_metrics(recon_out)

        prior_out = out_dir / f"vae_prior_{tag}.pt"
        run(
            py
            + [
                "-m",
                "hmc_dit3d.train.sample",
                "--config",
                cfg,
                "--checkpoint",
                args.checkpoint,
                "--condition-source",
                "vae",
                "--vae",
                str(vae_path if tag == train_tag else train_path),
                "--split",
                "test",
                "--limit",
                "64",
                "--vae-sequence-threshold",
                str(thr),
                "--vae-temperature",
                "1.0",
                "--output",
                str(prior_out),
            ]
        )
        prior_m = eval_metrics(prior_out)

        row = {
            "tag": tag,
            "beta": beta,
            "latent": z,
            "hidden": hidden,
            "thr": thr,
            "recon": recon_m,
            "prior": prior_m,
        }
        results.append(row)
        print(
            f"RESULT {cat} {tag} | recon 1NN-CD={recon_m.get('1-NN-CD-acc')} "
            f"1NN-EMD={recon_m.get('1-NN-EMD-acc')} "
            f"COV-CD={recon_m.get('lgan_cov-CD')} | prior "
            f"1NN-CD={prior_m.get('1-NN-CD-acc')} "
            f"COV-CD={prior_m.get('lgan_cov-CD')}",
            flush=True,
        )

    summary_path = out_dir / f"{cat}_tune_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
