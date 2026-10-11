"""TopoDiT-style metrics averaged over independent 8-vs-8 groups.

The EMD path uses TopoDiT's CUDA ``approxmatch`` extension.  The extension is
imported lazily so the pure PyTorch helpers remain testable without a compiler.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

METRIC_KEYS = ("1-NNA-CD", "1-NNA-EMD", "COV-CD", "COV-EMD")


def knn_accuracy(mxx: Tensor, mxy: Tensor, myy: Tensor) -> float:
    """Return TopoDiT/GAN-metrics 1-nearest-neighbour accuracy."""
    n0, n1 = mxx.shape[0], myy.shape[0]
    labels = torch.cat((torch.ones(n0), torch.zeros(n1)))
    matrix = torch.cat(
        (torch.cat((mxx, mxy), dim=1), torch.cat((mxy.t(), myy), dim=1)),
        dim=0,
    )
    matrix = matrix + torch.diag(torch.full((n0 + n1,), float("inf")))
    indices = matrix.topk(1, dim=0, largest=False).indices[0]
    predictions = labels.index_select(0, indices) >= 0.5
    return float((predictions.float() == labels).float().mean())


def coverage(reference_to_sample: Tensor) -> float:
    """Return TopoDiT coverage from a reference-by-sample distance matrix."""
    nearest_reference = reference_to_sample.t().min(dim=1).indices
    return float(nearest_reference.unique().numel()) / float(
        reference_to_sample.shape[0]
    )


def _emd_cuda() -> Any:
    try:
        return importlib.import_module("emd_cuda")
    except ImportError as exc:
        message = (
            "TopoDiT's compiled emd_cuda extension is required for exact EMD. "
            "Build metrics/PyTorchEMD from TopoDiT-3D and prepend the directory "
            "containing emd_cuda*.so to PYTHONPATH."
        )
        raise RuntimeError(message) from exc


def approximate_emd(x: Tensor, y: Tensor) -> Tensor:
    """Return TopoDiT's CUDA approxmatch EMD for aligned batches."""
    extension = _emd_cuda()
    x = x.contiguous()
    y = y.contiguous()
    match = extension.approxmatch_forward(x, y)
    cost = extension.matchcost_forward(x, y, match)
    return cost / x.shape[1]


def pairwise_cd_emd(
    a: Tensor,
    b: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return pairwise squared Chamfer and TopoDiT EMD matrices."""
    n_a, n_b = a.shape[0], b.shape[0]
    cds = torch.empty(n_a, n_b)
    emds = torch.empty(n_a, n_b)
    b_device = b.to(device).contiguous()
    for index in range(n_a):
        x = a[index].to(device).view(1, -1, 3)
        x = x.expand(n_b, -1, -1).contiguous()
        distances = torch.cdist(x, b_device).square()
        cds[index] = (
            distances.min(dim=2).values.mean(dim=1)
            + distances.min(dim=1).values.mean(dim=1)
        ).cpu()
        emds[index] = approximate_emd(x, b_device).cpu()
    return cds, emds


def evaluate_group(
    sample: Tensor,
    reference: Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one independent sample/reference group."""
    m_rs_cd, m_rs_emd = pairwise_cd_emd(reference, sample, device=device)
    m_rr_cd, m_rr_emd = pairwise_cd_emd(reference, reference, device=device)
    m_ss_cd, m_ss_emd = pairwise_cd_emd(sample, sample, device=device)
    return {
        "1-NNA-CD": 100.0 * knn_accuracy(m_rr_cd, m_rs_cd, m_ss_cd),
        "1-NNA-EMD": 100.0 * knn_accuracy(m_rr_emd, m_rs_emd, m_ss_emd),
        "COV-CD": 100.0 * coverage(m_rs_cd),
        "COV-EMD": 100.0 * coverage(m_rs_emd),
    }


def evaluate_payload(
    path: Path,
    *,
    device: torch.device,
    batch_size: int,
    expected_count: int | None = None,
    max_batches: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a saved sample payload and return batch rows plus their mean."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    samples = payload["samples"].float()
    references = payload["references"].float()
    if samples.shape != references.shape:
        raise ValueError(f"shape mismatch: {samples.shape} vs {references.shape}")
    if expected_count is not None and samples.shape[0] != expected_count:
        raise ValueError(
            f"expected {expected_count} samples, found {samples.shape[0]} in {path}"
        )
    if samples.shape[0] % batch_size:
        raise ValueError(
            f"sample count {samples.shape[0]} is not divisible by "
            f"batch size {batch_size}"
        )

    num_batches = samples.shape[0] // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    rows: list[dict[str, float]] = []
    started = time.monotonic()
    with torch.no_grad():
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            stop = start + batch_size
            row = evaluate_group(samples[start:stop], references[start:stop], device)
            rows.append(row)
            should_report = (
                batch_index == 0
                or (batch_index + 1) % 10 == 0
                or batch_index + 1 == num_batches
            )
            if should_report:
                print(
                    json.dumps(
                        {
                            "event": "metric_progress",
                            "file": str(path),
                            "batch": batch_index + 1,
                            "num_batches": num_batches,
                            **row,
                        }
                    ),
                    flush=True,
                )

    metrics = {
        key: sum(row[key] for row in rows) / len(rows) for key in METRIC_KEYS
    }
    return {
        "protocol": {
            "sample_count": samples.shape[0],
            "evaluated_sample_count": num_batches * batch_size,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "aggregation": "independent groups followed by arithmetic mean",
            "cd": "TopoDiT squared Chamfer",
            "emd": "TopoDiT PyTorchEMD CUDA approxmatch",
            "seed": seed,
        },
        "source": {
            "payload": str(path.resolve()),
            "condition_source": payload.get("condition_source"),
            "generation_batch_size": payload.get("generation_batch_size"),
            "checkpoint_path": payload.get("checkpoint_path"),
            "checkpoint_epoch": payload.get("checkpoint_epoch"),
            "checkpoint_step": payload.get("checkpoint_step"),
            "vae_path": payload.get("vae_path"),
            "vae_temperature": payload.get("vae_temperature"),
            "vae_sequence_threshold": payload.get("vae_sequence_threshold"),
        },
        "metrics": metrics,
        "batches": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_payload(
        args.samples,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        expected_count=args.expected_count,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    _write_json(args.output, result)
    print(json.dumps({"event": "metric_complete", **result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
