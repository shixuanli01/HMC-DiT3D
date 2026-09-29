"""Run a resumable two-phase N=664 multi-seed generation/evaluation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

CATEGORIES = {
    "chair": "train_chair_h100_formal_bottleneck.yaml",
    "airplane": "train_airplane_h100_formal_bottleneck.yaml",
    "car": "train_car_h100_formal_bottleneck.yaml",
}
METRIC_KEYS = ("1-NNA-CD", "1-NNA-EMD", "COV-CD", "COV-EMD")


def parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed list while preserving order."""
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise ValueError("seed list must not be empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seed list contains duplicates: {text}")
    return seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_sample(path: Path, expected_count: int, generation_batch_size: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload["samples"]
        references = payload["references"]
        return bool(
            tuple(samples.shape) == (expected_count, 2048, 3)
            and samples.shape == references.shape
            and torch.isfinite(samples).all()
            and torch.isfinite(references).all()
            and payload.get("condition_source") == "vae"
            and payload.get("generation_batch_size") == generation_batch_size
        )
    except Exception:
        return False


def valid_metrics(path: Path, seed: int, expected_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload["protocol"]["seed"] == seed
            and payload["protocol"]["sample_count"] == expected_count
            and all(key in payload["metrics"] for key in METRIC_KEYS)
        )
    except Exception:
        return False


def preserve_invalid(path: Path) -> None:
    if path.exists():
        destination = path.with_name(f"{path.name}.invalid-{int(time.time())}")
        path.replace(destination)
        print(f"Preserved invalid output as {destination}", flush=True)


def run_with_retries(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    retries: int,
) -> None:
    for attempt in range(1, retries + 1):
        print(
            json.dumps({"event": "command", "attempt": attempt, "argv": command}),
            flush=True,
        )
        try:
            subprocess.run(command, cwd=cwd, env=env, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            print(f"Command failed; retrying ({attempt}/{retries}).", flush=True)
            time.sleep(10)


def update_summary(output_dir: Path, categories: list[str], seeds: list[int]) -> None:
    runs: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_runs: dict[str, Any] = {}
        for seed in seeds:
            metrics_path = output_dir / category / f"metrics_seed{seed}.json"
            if metrics_path.is_file():
                record = json.loads(metrics_path.read_text(encoding="utf-8"))
                category_runs[str(seed)] = record["metrics"]
        runs[category] = category_runs

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for category, category_runs in runs.items():
        aggregate[category] = {}
        for key in METRIC_KEYS:
            values = torch.tensor(
                [row[key] for row in category_runs.values()], dtype=torch.float64
            )
            if values.numel():
                aggregate[category][key] = {
                    "mean": float(values.mean()),
                    "std_population": float(values.std(unbiased=False)),
                    "count": int(values.numel()),
                }
    atomic_json(
        output_dir / "summary.json",
        {"requested_seeds": seeds, "runs": runs, "aggregate": aggregate},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seeds", default="0,1,2,3,4")
    parser.add_argument("--second-seeds", default="5,6,7,8,9,10,11,12,13,42")
    parser.add_argument(
        "--categories", nargs="+", choices=tuple(CATEGORIES), default=list(CATEGORIES)
    )
    parser.add_argument("--limit", type=int, default=664)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--vae-temperature", type=float, default=1.2)
    parser.add_argument("--vae-sequence-threshold", type=float, default=0.1)
    parser.add_argument("--task-retries", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--emd-extension-dir",
        type=Path,
        default=(
            Path(os.environ["EMD_EXTENSION_DIR"])
            if "EMD_EXTENSION_DIR" in os.environ
            else None
        ),
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    repo_root = project_dir.parent
    data_root = (args.data_root or repo_root / "ShapeNetCore.v2.PC15k").resolve()
    checkpoint_root = (args.checkpoint_root or repo_root / "checkpoints").resolve()
    output_dir = (
        args.output_dir or project_dir / "results/vae_prior_multiseed_n664_batch8"
    ).resolve()
    emd_extension_dir = args.emd_extension_dir
    if emd_extension_dir is None or not emd_extension_dir.is_dir():
        raise FileNotFoundError(
            "Pass --emd-extension-dir or set EMD_EXTENSION_DIR to the directory "
            "containing the compiled emd_cuda extension."
        )
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset not found: {data_root}")

    first_seeds = parse_seeds(args.first_seeds)
    second_seeds = parse_seeds(args.second_seeds)
    all_seeds = first_seeds + second_seeds
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("first and second seed phases must be disjoint")
    phases = (("first5", first_seeds), ("next10", second_seeds))
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, Any]] = {}
    for category in args.categories:
        checkpoint = checkpoint_root / "dit" / f"{category}.pt"
        vae = checkpoint_root / "vae" / f"{category}.pt"
        if not checkpoint.is_file() or not vae.is_file():
            raise FileNotFoundError(
                f"missing checkpoint pair for {category}: {checkpoint}, {vae}"
            )
        artifacts[category] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "vae": str(vae),
            "vae_sha256": sha256(vae),
        }

    manifest = {
        "protocol": {
            "first_seeds": first_seeds,
            "second_seeds": second_seeds,
            "categories": args.categories,
            "sample_count": args.limit,
            "generation_batch_size": args.generation_batch_size,
            "metric_batch_size": 8,
            "vae_temperature": args.vae_temperature,
            "vae_sequence_threshold": args.vae_sequence_threshold,
        },
        "data_root": str(data_root),
        "artifacts": artifacts,
        "emd_extension_dir": str(emd_extension_dir.resolve()),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)

    state_path = output_dir / "run_state.json"
    state: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "completed": [],
        "current": None,
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["started_at_unix"] = previous.get("started_at_unix", time.time())
        state["completed"] = previous.get("completed", [])
    atomic_json(state_path, state)

    environment = os.environ.copy()
    source_dir = project_dir / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(emd_extension_dir.resolve()),
            str(source_dir),
            environment.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)

    for phase_name, seeds in phases:
        for seed in seeds:
            for category in args.categories:
                category_dir = output_dir / category
                config_dir = output_dir / "runtime_configs"
                category_dir.mkdir(parents=True, exist_ok=True)
                config_dir.mkdir(parents=True, exist_ok=True)
                sample_path = category_dir / f"vae_prior_seed{seed}.pt"
                metrics_path = category_dir / f"metrics_seed{seed}.json"
                task_id = f"{phase_name}:{category}:seed{seed}"

                state["current"] = {"task": task_id, "stage": "generation"}
                atomic_json(state_path, state)
                sample_is_valid = valid_sample(
                    sample_path, args.limit, args.generation_batch_size
                )
                if not sample_is_valid:
                    preserve_invalid(sample_path)
                    source_config = project_dir / "configs" / CATEGORIES[category]
                    runtime_config = config_dir / f"{category}_seed{seed}.yaml"
                    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
                    config["train"]["seed"] = seed
                    config["data"]["root_dir"] = str(data_root)
                    runtime_config.write_text(
                        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                    )
                    generation_command = [
                        sys.executable,
                        "-m",
                        "hmc_dit3d.train.sample",
                        "--config",
                        str(runtime_config),
                        "--checkpoint",
                        artifacts[category]["checkpoint"],
                        "--condition-source",
                        "vae",
                        "--vae",
                        artifacts[category]["vae"],
                        "--split",
                        "test",
                        "--limit",
                        str(args.limit),
                        "--generation-batch-size",
                        str(args.generation_batch_size),
                        "--vae-temperature",
                        str(args.vae_temperature),
                        "--vae-sequence-threshold",
                        str(args.vae_sequence_threshold),
                        "--output",
                        str(sample_path),
                    ]
                    run_with_retries(
                        generation_command,
                        cwd=project_dir,
                        env=environment,
                        retries=args.task_retries,
                    )
                    if not valid_sample(
                        sample_path, args.limit, args.generation_batch_size
                    ):
                        raise RuntimeError(
                            f"generated payload failed validation: {sample_path}"
                        )
                else:
                    print(f"Skipping valid sample payload {sample_path}", flush=True)

                state["current"] = {"task": task_id, "stage": "metrics"}
                atomic_json(state_path, state)
                if not valid_metrics(metrics_path, seed, args.limit):
                    preserve_invalid(metrics_path)
                    metric_command = [
                        sys.executable,
                        "-m",
                        "hmc_dit3d.metrics.topodit_batch8",
                        "--samples",
                        str(sample_path),
                        "--output",
                        str(metrics_path),
                        "--batch-size",
                        "8",
                        "--expected-count",
                        str(args.limit),
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda",
                    ]
                    run_with_retries(
                        metric_command,
                        cwd=project_dir,
                        env=environment,
                        retries=args.task_retries,
                    )
                    if not valid_metrics(metrics_path, seed, args.limit):
                        raise RuntimeError(
                            f"metric output failed validation: {metrics_path}"
                        )
                else:
                    print(f"Skipping valid metric payload {metrics_path}", flush=True)

                if task_id not in state["completed"]:
                    state["completed"].append(task_id)
                state["current"] = None
                atomic_json(state_path, state)
                update_summary(output_dir, args.categories, all_seeds)
                print(
                    json.dumps({"event": "task_complete", "task": task_id}),
                    flush=True,
                )

        atomic_json(
            output_dir / f"{phase_name}_complete.json",
            {"phase": phase_name, "seeds": seeds, "completed_at_unix": time.time()},
        )

    state["status"] = "complete"
    state["current"] = None
    state["completed_at_unix"] = time.time()
    atomic_json(state_path, state)
    update_summary(output_dir, args.categories, all_seeds)
    print(
        json.dumps({"event": "queue_complete", "output": str(output_dir)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
