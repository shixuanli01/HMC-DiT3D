"""Evaluate VAE-prior and TopoDiT generation over multiple random seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from hmc_dit3d.metrics.pointcloud import compute_minimal_pointcloud_metrics

CATEGORIES: dict[str, dict[str, Any]] = {
    "airplane": {
        "config": "configs/train_airplane_h100_formal_bottleneck.yaml",
        "checkpoint": "../checkpoints/dit/airplane.pt",
        "vae": "../checkpoints/vae/airplane.pt",
        "train_bank": (
            "results/train_airplane_h100_formal_bottleneck/vae/"
            "airplane_train_bank_full.pt"
        ),
        "topodit": (
            "../TopoDiT-3D/checkpoints/output/test_S4_airplane/syn/samples.pth"
        ),
    },
    "car": {
        "config": "configs/train_car_h100_formal_bottleneck.yaml",
        "checkpoint": "../checkpoints/dit/car.pt",
        "vae": "../checkpoints/vae/car.pt",
        "train_bank": (
            "results/train_car_h100_formal_bottleneck/vae/car_train_bank_full.pt"
        ),
        "topodit": ("../TopoDiT-3D/checkpoints/output/test_S4_car/syn/samples.pth"),
    },
    "chair": {
        "config": "configs/train_chair_h100_formal_bottleneck.yaml",
        "checkpoint": "../checkpoints/dit/chair.pt",
        "vae": "../checkpoints/vae/chair.pt",
        "train_bank": (
            "results/train_chair_h100_formal_bottleneck/vae/chair_train_bank_full.pt"
        ),
        "topodit": ("../TopoDiT-3D/checkpoints/output/test_S4_chair/syn/samples.pth"),
    },
}

METRIC_KEYS = (
    "1-NN-CD-acc",
    "1-NN-EMD-acc",
    "lgan_cov-CD",
    "lgan_cov-EMD",
    "lgan_mmd-CD",
    "lgan_mmd-EMD",
    "JSD",
)


def evaluate(samples: torch.Tensor, references: torch.Tensor) -> dict[str, float]:
    """Evaluate one 64-sample run with the common project metric pipeline."""
    metrics = compute_minimal_pointcloud_metrics(
        samples.float(),
        references.float(),
        batch_size=8,
        device="cuda",
        jsd_resolution=8,
        emd_point_count=128,
        emd_sinkhorn_epsilon=0.1,
        emd_sinkhorn_iterations=50,
        emd_batch_size=8,
    )
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute population mean and standard deviation across seeds."""
    output: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=tuple(CATEGORIES),
        default=list(CATEGORIES),
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--vae-prior", choices=("normal", "gmm"), default="normal")
    parser.add_argument("--vae-temperature", type=float, default=1.2)
    parser.add_argument("--vae-gmm-components", type=int, default=32)
    parser.add_argument("--vae-gmm-iterations", type=int, default=100)
    parser.add_argument("--vae-gmm-candidate-multiplier", type=int, default=8)
    parser.add_argument(
        "--vae-gmm-candidate-source",
        choices=("sampled", "train"),
        default="sampled",
    )
    parser.add_argument("--vae-gmm-stratified-batch-size", type=int, default=None)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate VAE-prior payloads without running the metric pipelines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vae_prior_multiseed"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    for category in args.categories:
        paths = CATEGORIES[category]
        category_dir = args.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        if not args.generate_only:
            topo_payload = torch.load(
                paths["topodit"], map_location="cpu", weights_only=False
            )
            topo_samples = (
                topo_payload["samples"]
                if isinstance(topo_payload, dict)
                else topo_payload
            ).float()

        vae_rows: list[dict[str, float]] = []
        topo_rows: list[dict[str, float]] = []
        for seed in args.seeds:
            config_data = yaml.safe_load(Path(paths["config"]).read_text())
            config_data["train"]["seed"] = seed
            config_data["data"]["root_dir"] = str(
                Path("../ShapeNetCore.v2.PC15k").resolve()
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as config_file:
                yaml.safe_dump(config_data, config_file, sort_keys=False)
                runtime_config = Path(config_file.name)

            vae_output = category_dir / f"vae_prior_seed{seed}.pt"
            try:
                subprocess.run(
                    [
                        "micromamba",
                        "run",
                        "-n",
                        "hmc-dit3d-py312",
                        "python",
                        "-m",
                        "hmc_dit3d.train.sample",
                        "--config",
                        str(runtime_config),
                        "--checkpoint",
                        paths["checkpoint"],
                        "--condition-source",
                        "vae",
                        "--vae",
                        paths["vae"],
                        "--vae-prior",
                        args.vae_prior,
                        *(
                            ["--vae-prior-bank", paths["train_bank"]]
                            if args.vae_prior == "gmm"
                            else []
                        ),
                        "--vae-gmm-components",
                        str(args.vae_gmm_components),
                        "--vae-gmm-iterations",
                        str(args.vae_gmm_iterations),
                        "--vae-gmm-candidate-multiplier",
                        str(args.vae_gmm_candidate_multiplier),
                        "--vae-gmm-candidate-source",
                        args.vae_gmm_candidate_source,
                        *(
                            [
                                "--vae-gmm-stratified-batch-size",
                                str(args.vae_gmm_stratified_batch_size),
                            ]
                            if args.vae_gmm_stratified_batch_size is not None
                            else []
                        ),
                        "--split",
                        "test",
                        "--limit",
                        str(args.limit),
                        *(
                            [
                                "--generation-batch-size",
                                str(args.generation_batch_size),
                            ]
                            if args.generation_batch_size is not None
                            else []
                        ),
                        "--vae-sequence-threshold",
                        "0.1",
                        "--vae-temperature",
                        str(args.vae_temperature),
                        "--output",
                        str(vae_output),
                    ],
                    check=True,
                )
            finally:
                runtime_config.unlink(missing_ok=True)

            if args.generate_only:
                print(f"Generated {vae_output}", flush=True)
                continue

            vae_payload = torch.load(vae_output, map_location="cpu", weights_only=False)
            references = vae_payload["references"].float()
            vae_metrics = evaluate(vae_payload["samples"], references)

            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(topo_samples.shape[0], generator=generator)[
                : args.limit
            ]
            selected_topo = topo_samples[indices]
            topo_output = category_dir / f"topodit_seed{seed}.pt"
            torch.save(
                {
                    "samples": selected_topo,
                    "references": references,
                    "seed": seed,
                },
                topo_output,
            )
            topo_metrics = evaluate(selected_topo, references)
            vae_rows.append(vae_metrics)
            topo_rows.append(topo_metrics)
            print(
                f"{category} seed={seed}: "
                f"VAE COV-CD={vae_metrics['lgan_cov-CD'] * 100:.2f}, "
                f"TopoDiT COV-CD={topo_metrics['lgan_cov-CD'] * 100:.2f}",
                flush=True,
            )

        if args.generate_only:
            continue

        all_results[category] = {
            "seeds": args.seeds,
            "vae_prior": {
                "runs": vae_rows,
                "summary": summarize(vae_rows),
            },
            "topodit": {
                "runs": topo_rows,
                "summary": summarize(topo_rows),
            },
        }

    if not args.generate_only:
        output_path = args.output_dir / "summary.json"
        output_path.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"Saved summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
