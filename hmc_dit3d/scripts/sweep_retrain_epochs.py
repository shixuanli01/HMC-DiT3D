"""Evaluate every epoch_N.pt of the bs512 retrain runs under the formal N=664 protocol.

Same generation/metric commands as run_multiseed_n664.py (VAE prior, generation
batch 8, TopoDiT batch-8 metrics with CUDA approxmatch EMD), but distributed over
GPUs with one worker per GPU. Resumable: finished metric files are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
CONFIGS = {
    c: f"retrain_{c}_h100_formal_bottleneck_bs512.yaml"
    for c in ("chair", "airplane", "car")
}


def build_tasks(args) -> list[dict]:
    tasks = []
    for epoch in args.epochs:
        for category in args.categories:
            if epoch == 0:  # original released checkpoint as baseline
                ckpt = REPO / "checkpoints/dit" / f"{category}.pt"
                tag = "orig"
            else:
                ckpt = (
                    PROJECT
                    / "results"
                    / f"retrain_{category}_h100_formal_bottleneck_bs512"
                    / f"epoch_{epoch}.pt"
                )
                tag = f"epoch{epoch}"
            tasks.append({"category": category, "tag": tag, "checkpoint": ckpt})
    return tasks


def run_task(task: dict, gpu: int, args) -> None:
    category, tag = task["category"], task["tag"]
    out_dir = args.output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_seed{args.seed}"
    sample_path = out_dir / f"{stem}.pt"
    metrics_path = out_dir / f"metrics_{stem}.json"
    log_path = out_dir / f"{stem}.log"
    if metrics_path.is_file():
        print(f"[skip] {category} {tag}", flush=True)
        return

    config = yaml.safe_load((PROJECT / "configs" / CONFIGS[category]).read_text())
    config["train"]["seed"] = args.seed
    config["data"]["root_dir"] = str(args.data_root)
    runtime_config = args.output_dir / "runtime_configs" / f"{category}_{stem}.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.emd_extension_dir), str(PROJECT / "src"), env.get("PYTHONPATH", "")]
    )
    commands = []
    if not sample_path.is_file():
        commands.append([
            sys.executable, "-m", "hmc_dit3d.train.sample",
            "--config", str(runtime_config),
            "--checkpoint", str(task["checkpoint"]),
            "--condition-source", "vae",
            "--vae", str(REPO / "checkpoints/vae" / f"{category}.pt"),
            "--split", "test",
            "--limit", str(args.limit),
            "--generation-batch-size", "8",
            "--vae-temperature", str(args.vae_temperature),
            "--vae-sequence-threshold", str(args.vae_sequence_threshold),
            "--output", str(sample_path),
        ])
    commands.append([
        sys.executable, "-m", "hmc_dit3d.metrics.topodit_batch8",
        "--samples", str(sample_path),
        "--output", str(metrics_path),
        "--batch-size", "8",
        "--expected-count", str(args.limit),
        "--seed", str(args.seed),
        "--device", "cuda",
    ])
    print(f"[start] gpu{gpu} {category} {tag}", flush=True)
    with log_path.open("a") as log:
        for command in commands:
            subprocess.run(command, cwd=PROJECT, env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
    metrics = json.loads(metrics_path.read_text())["metrics"]
    print(f"[done] {category} {tag} {json.dumps(metrics)}", flush=True)


def summarize(args) -> None:
    rows = []
    for category in args.categories:
        for path in sorted((args.output_dir / category).glob("metrics_*.json")):
            metrics = json.loads(path.read_text())["metrics"]
            rows.append({"category": category, "run": path.stem[len("metrics_"):], **metrics})
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=list(CONFIGS))
    parser.add_argument("--epochs", type=int, nargs="+",
                        default=[0] + list(range(1000, 10001, 1000)),
                        help="0 = original released checkpoint")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=664)
    parser.add_argument("--vae-temperature", type=float, default=1.2)
    parser.add_argument("--vae-sequence-threshold", type=float, default=0.1)
    parser.add_argument("--data-root", type=Path, default=REPO / "ShapeNetCore.v2.PC15k")
    parser.add_argument("--emd-extension-dir", type=Path,
                        default=Path(os.environ.get("EMD_EXTENSION_DIR", "")))
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "results/retrain_epoch_sweep_n664_batch8")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.emd_extension_dir = args.emd_extension_dir.resolve()
    if not (args.emd_extension_dir.is_dir()
            and any(args.emd_extension_dir.glob("emd_cuda*.so"))):
        raise FileNotFoundError("set --emd-extension-dir to the emd_cuda build dir")

    tasks: queue.Queue = queue.Queue()
    for task in build_tasks(args):
        tasks.put(task)
    failures = []

    def worker(gpu: int) -> None:
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                run_task(task, gpu, args)
            except Exception as exc:  # keep other tasks going
                failures.append((task["category"], task["tag"], repr(exc)))
                print(f"[fail] {task['category']} {task['tag']}: {exc}", flush=True)
            summarize(args)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summarize(args)
    print(f"[finished] failures={failures}", flush=True)


if __name__ == "__main__":
    main()
