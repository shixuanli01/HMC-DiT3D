"""Minimal smoke-training entry for HMC-DiT3D."""

from __future__ import annotations

import argparse
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, default_collate

from hmc_dit3d.data.shapenet_pc15k import ShapeNetPC15KDataset
from hmc_dit3d.hmc.condition_vae import HMCConditionVAE
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor
from hmc_dit3d.models.hmc_dit import (
    HMCConditionedBottleneckDiTPointCloud,
    HMCConditionedDiTPointCloud,
)
from hmc_dit3d.models.multifractal_predictor import (
    MultifractalDescriptorPredictor,
)
from hmc_dit3d.train.config import ExperimentConfig, load_experiment_config
from hmc_dit3d.train.diffusion import GaussianDiffusion
from hmc_dit3d.train.ema import ModelEMA
from hmc_dit3d.utils.runtime import detect_device, set_global_seed

LOGGER = logging.getLogger(__name__)


class SmokeTrainingError(RuntimeError):
    """Raised when the smoke training pipeline fails."""


@dataclass(slots=True, frozen=True)
class TrainingCheckpointState:
    """Loaded training state from a saved checkpoint.

    Args:
        epoch: Zero-based epoch index stored in the checkpoint.
        step: Global optimization step stored in the checkpoint.
        metrics: Latest scalar metrics carried by the checkpoint.
    """

    epoch: int
    step: int
    metrics: dict[str, float]
    auxiliary_state: dict[str, dict[str, Any]]


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader workers for reproducible point subsampling.

    Args:
        worker_id: Worker index assigned by the dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


class HMCBatchCollator:
    """Collate a batch and extract its HMC features inside dataloader workers.

    HMC extraction is deterministic given the points, so moving it off the
    training process yields tensors identical to `prepare_hmc_batch`.
    """

    def __init__(self, extractor: HMCFeatureExtractor, point_source: str) -> None:
        self.extractor = extractor
        self.point_source = point_source

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = default_collate(samples)
        hmc_points = select_hmc_condition_points(batch, self.point_source)
        descriptors, sequences = extract_hmc_features_cpu(hmc_points, self.extractor)
        batch["hmc_descriptors"] = descriptors
        batch["hmc_sequences"] = sequences
        return batch


def _loader_worker_kwargs(num_workers: int) -> dict[str, Any]:
    """Keep workers (and their Hilbert caches) alive across epochs."""
    if num_workers <= 0:
        return {}
    return {"persistent_workers": True, "prefetch_factor": 4}


def build_dataloader(
    config: ExperimentConfig,
    extractor: HMCFeatureExtractor | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create the ShapeNet dataloader for smoke training.

    Args:
        config: Full experiment configuration.
        extractor: Optional HMC extractor; when given, HMC features are
            computed in the dataloader workers.

    Returns:
        DataLoader[dict[str, Any]]: Configured dataloader.
    """
    dataset = ShapeNetPC15KDataset(
        root_dir=config.data.root_dir,
        categories=config.data.categories,
        split=config.data.split,
        sample_size=config.data.sample_size,
        random_subsample=config.data.random_subsample,
        return_full_points=config.data.hmc_point_source == "full",
    )
    generator = torch.Generator()
    generator.manual_seed(config.train.seed)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
        generator=generator,
        worker_init_fn=_seed_worker,
        collate_fn=None
        if extractor is None
        else HMCBatchCollator(extractor, config.data.hmc_point_source),
        **_loader_worker_kwargs(config.data.num_workers),
    )


def build_eval_dataloader(
    config: ExperimentConfig,
    *,
    split: str,
    extractor: HMCFeatureExtractor | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a deterministic dataloader for validation or test evaluation.

    Args:
        config: Full experiment configuration.
        split: Dataset split used for evaluation.
        extractor: Optional HMC extractor; when given, HMC features are
            computed in the dataloader workers.

    Returns:
        DataLoader[dict[str, Any]]: Deterministic evaluation dataloader.
    """
    dataset = ShapeNetPC15KDataset(
        root_dir=config.data.root_dir,
        categories=config.data.categories,
        split=split,
        sample_size=config.data.sample_size,
        random_subsample=False,
        return_full_points=config.data.hmc_point_source == "full",
    )
    generator = torch.Generator()
    generator.manual_seed(config.train.seed)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_worker,
        collate_fn=None
        if extractor is None
        else HMCBatchCollator(extractor, config.data.hmc_point_source),
    )


def build_model(config: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    """Construct the point-cloud HMC-conditioned DiT model.

    Args:
        config: Full experiment configuration.
        device: Target torch device.

    Returns:
        torch.nn.Module: Smoke-training model.
    """
    backbone_config = config.model.build_backbone_config(
        num_classes=len(config.data.categories)
    )
    # Route by configured variant (3.1 vs 3.2) to avoid duplicating runner logic.
    if config.model.model_variant == "cross_attention":
        model = HMCConditionedDiTPointCloud(
            hmc_config=config.hmc,
            encoder_config=config.encoder,
            backbone_config=backbone_config,
            voxel_size=config.model.voxel_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
        )
    else:
        model = HMCConditionedBottleneckDiTPointCloud(
            hmc_config=config.hmc,
            encoder_config=config.encoder,
            backbone_config=backbone_config,
            voxel_size=config.model.voxel_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            num_bottleneck_latents=config.model.num_bottleneck_latents,
            resampler_depth=config.model.resampler_depth,
            resampler_mlp_ratio=config.model.resampler_mlp_ratio,
        )
    return model.to(device)


def build_multifractal_predictor(
    config: ExperimentConfig,
    device: torch.device,
) -> MultifractalDescriptorPredictor:
    """Construct the optional descriptor predictor used by `L_mf`.

    Args:
        config: Full experiment configuration.
        device: Target torch device.

    Returns:
        MultifractalDescriptorPredictor: Auxiliary descriptor regressor.
    """
    predictor = MultifractalDescriptorPredictor(
        input_channels=config.model.resolved_out_channels,
        descriptor_dim=config.hmc.descriptor_dim,
        hidden_dim=config.train.l_mf_hidden_dim,
    )
    return predictor.to(device)


def prepare_hmc_batch(
    points_bnc: Tensor,
    extractor: HMCFeatureExtractor,
    device: torch.device,
) -> tuple[Tensor, list[Tensor]]:
    """Extract batched HMC descriptors and sequences from point clouds.

    Args:
        points_bnc: Point tensor with shape `(B, N, 3)`.
        extractor: HMC feature extractor.
        device: Target torch device.

    Returns:
        tuple[Tensor, list[Tensor]]:
            Descriptor tensor `(B, D_hmc)` and one sequence tensor per scale.
    """
    if points_bnc.ndim != 3 or points_bnc.shape[-1] != 3:
        message = (
            f"points_bnc must have shape (B, N, 3), got {tuple(points_bnc.shape)!r}."
        )
        LOGGER.error(message)
        raise SmokeTrainingError(message)

    descriptor_tensor, sequence_tensors = extract_hmc_features_cpu(
        points_bnc,
        extractor,
    )
    return descriptor_tensor.to(device), [
        sequence.to(device) for sequence in sequence_tensors
    ]


def extract_hmc_features_cpu(
    points_bnc: Tensor,
    extractor: HMCFeatureExtractor,
) -> tuple[Tensor, list[Tensor]]:
    """Extract batched HMC descriptors and sequences as CPU float32 tensors."""
    descriptors: list[Tensor] = []
    sequence_buckets: list[list[Tensor]] = [[] for _ in extractor.config.scales]
    for sample in points_bnc.detach().cpu().numpy():
        result = extractor.extract(sample)
        descriptors.append(torch.from_numpy(result.descriptor).to(torch.float32))
        for sequence_index, sequence in enumerate(result.sequences):
            sequence_buckets[sequence_index].append(
                torch.from_numpy(sequence).to(torch.float32)
            )
    return torch.stack(descriptors, dim=0), [
        torch.stack(sequence_bucket, dim=0) for sequence_bucket in sequence_buckets
    ]


def resolve_hmc_batch(
    batch: dict[str, Any],
    extractor: HMCFeatureExtractor,
    device: torch.device,
    point_source: str,
) -> tuple[Tensor, list[Tensor]]:
    """Return HMC features, reusing ones precomputed by `HMCBatchCollator`."""
    if "hmc_descriptors" in batch:
        return batch["hmc_descriptors"].to(device, non_blocking=True), [
            sequence.to(device, non_blocking=True)
            for sequence in batch["hmc_sequences"]
        ]
    hmc_points = select_hmc_condition_points(batch, point_source)
    return prepare_hmc_batch(hmc_points, extractor, device)


def select_hmc_condition_points(
    batch: dict[str, Any],
    point_source: str,
) -> Tensor:
    """Select sampled or full-resolution points for HMC extraction."""
    if point_source == "sampled":
        key = "points"
    elif point_source == "full":
        key = "full_points"
    else:
        raise SmokeTrainingError(f"Unsupported HMC point source {point_source!r}.")
    if key not in batch:
        raise SmokeTrainingError(
            f"Batch is missing {key!r} required by HMC point source {point_source!r}."
        )
    points = torch.as_tensor(batch[key], dtype=torch.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise SmokeTrainingError(
            f"Batch {key} must have shape (B, N, 3), got {tuple(points.shape)!r}."
        )
    return points


def apply_hmc_dropout(
    descriptors: Tensor,
    sequences: list[Tensor],
    dropout_prob: float,
) -> tuple[Tensor, list[Tensor], Tensor]:
    """Replace complete per-sample HMC conditions with zeros."""
    batch_size = descriptors.shape[0]
    drop_mask = torch.zeros(
        batch_size,
        device=descriptors.device,
        dtype=torch.bool,
    )
    if dropout_prob <= 0.0:
        return descriptors, sequences, drop_mask
    drop_mask = torch.rand(batch_size, device=descriptors.device) < dropout_prob
    keep = (~drop_mask).to(descriptors.dtype).unsqueeze(1)
    return (
        descriptors * keep,
        [sequence * keep for sequence in sequences],
        drop_mask,
    )


@torch.no_grad()
def apply_condition_vae_reconstruction(
    descriptors: Tensor,
    sequences: list[Tensor],
    condition_vae: HMCConditionVAE | None,
    reconstruction_prob: float,
    *,
    sequence_threshold: float = 0.0,
    posterior_temperature: float = 1.0,
    deterministic: bool = False,
) -> tuple[Tensor, list[Tensor], Tensor]:
    """Replace selected exact HMC conditions with frozen-VAE reconstructions."""
    batch_size = int(descriptors.shape[0])
    reconstruction_mask = torch.zeros(
        batch_size,
        device=descriptors.device,
        dtype=torch.bool,
    )
    if condition_vae is None or reconstruction_prob <= 0.0:
        return descriptors, sequences, reconstruction_mask
    if not 0.0 <= reconstruction_prob <= 1.0:
        raise SmokeTrainingError("reconstruction_prob must be in [0, 1].")
    if condition_vae.descriptor_dim != descriptors.shape[1]:
        raise SmokeTrainingError(
            "Condition VAE descriptor dimension does not match HMC descriptors."
        )
    if condition_vae.sequence_lengths != tuple(x.shape[1] for x in sequences):
        raise SmokeTrainingError(
            "Condition VAE sequence lengths do not match HMC sequences."
        )

    mu, logvar = condition_vae.encode(descriptors, sequences)
    if deterministic or posterior_temperature == 0.0:
        latent = mu
    else:
        latent = mu + (
            float(posterior_temperature)
            * torch.exp(0.5 * logvar)
            * torch.randn_like(mu)
        )
    reconstructed = condition_vae.decode_conditions(
        latent,
        sequence_threshold=sequence_threshold,
    )
    if reconstruction_prob >= 1.0:
        reconstruction_mask.fill_(True)
    else:
        reconstruction_mask = (
            torch.rand(batch_size, device=descriptors.device) < reconstruction_prob
        )
    choose = reconstruction_mask.to(descriptors.dtype).unsqueeze(1)
    mixed_descriptors = (
        descriptors * (1.0 - choose) + reconstructed["descriptors"] * choose
    )
    mixed_sequences = [
        original * (1.0 - choose) + replacement * choose
        for original, replacement in zip(
            sequences,
            reconstructed["sequences"],
            strict=True,
        )
    ]
    return mixed_descriptors, mixed_sequences, reconstruction_mask


def _compute_grad_norm(parameters: list[Tensor]) -> float:
    """Compute the global L2 gradient norm of a parameter list."""
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        return 0.0
    # One host sync instead of one per parameter.
    squared_norms = torch.stack([grad.double().pow(2).sum() for grad in grads])
    return float(squared_norms.sum().item()) ** 0.5


def _collect_trainable_parameters(
    model: torch.nn.Module,
    multifractal_predictor: MultifractalDescriptorPredictor | None = None,
) -> list[Tensor]:
    """Collect trainable parameters from the main model and optional auxiliaries.

    Args:
        model: Primary denoising model.
        multifractal_predictor: Optional auxiliary descriptor predictor.

    Returns:
        list[Tensor]: Flat list of trainable parameters.
    """
    parameters = list(model.parameters())
    if multifractal_predictor is not None:
        parameters.extend(multifractal_predictor.parameters())
    return [parameter for parameter in parameters if parameter.requires_grad]


def train_one_step(
    batch: dict[str, Any],
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    extractor: HMCFeatureExtractor,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    grad_clip: float | None,
    grad_accum_steps: int = 1,
    optimizer_step: bool = True,
    multifractal_predictor: MultifractalDescriptorPredictor | None = None,
    l_mf_weight: float = 0.0,
    hmc_point_source: str = "sampled",
    hmc_dropout_prob: float = 0.0,
    condition_vae: HMCConditionVAE | None = None,
    condition_vae_reconstruction_prob: float = 0.0,
    condition_vae_sequence_threshold: float = 0.0,
    condition_vae_posterior_temperature: float = 1.0,
    min_snr_gamma: float | None = None,
) -> dict[str, float]:
    """Run one smoke-training step.

    Args:
        batch: Batched sample dictionary from the dataloader.
        model: Point-cloud HMC-conditioned DiT model.
        extractor: HMC feature extractor.
        optimizer: AdamW optimizer.
        scaler: Optional CUDA gradient scaler for mixed precision stability.
        device: Target torch device.
        use_amp: Whether CUDA autocast is enabled.
        grad_clip: Optional gradient clipping threshold.
        grad_accum_steps: Number of micro-batches contributing to one
            optimizer step.
        optimizer_step: Whether this call should advance the optimizer.
        multifractal_predictor: Optional auxiliary descriptor predictor.
        l_mf_weight: Weight for the auxiliary descriptor loss.
        hmc_point_source: Whether HMC uses sampled targets or full 15k points.
        hmc_dropout_prob: Probability of dropping each complete HMC condition.
        condition_vae: Optional frozen VAE used to reconstruct conditions.
        condition_vae_reconstruction_prob: Per-sample reconstruction probability.
        condition_vae_sequence_threshold: Sparsification threshold for decoded
            measure sequences.
        condition_vae_posterior_temperature: Posterior noise multiplier.
        min_snr_gamma: Optional Min-SNR weighting gamma.

    Returns:
        dict[str, float]: Scalar metrics for the step.
    """
    points = torch.as_tensor(batch["points"], dtype=torch.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        message = (
            f"Batch points must have shape (B, N, 3). Got {tuple(points.shape)!r}."
        )
        LOGGER.error(message)
        raise SmokeTrainingError(message)
    labels = torch.as_tensor(batch["label"], device=device, dtype=torch.int64)
    descriptors, sequences = resolve_hmc_batch(
        batch,
        extractor,
        device,
        hmc_point_source,
    )
    target_descriptors = descriptors
    descriptors, sequences, vae_reconstruction_mask = (
        apply_condition_vae_reconstruction(
            descriptors,
            sequences,
            condition_vae,
            condition_vae_reconstruction_prob,
            sequence_threshold=condition_vae_sequence_threshold,
            posterior_temperature=condition_vae_posterior_temperature,
        )
    )
    descriptors, sequences, hmc_drop_mask = apply_hmc_dropout(
        descriptors,
        sequences,
        hmc_dropout_prob,
    )
    inputs = points.transpose(1, 2).to(device)
    timesteps = torch.randint(
        0,
        diffusion.num_timesteps,
        (inputs.shape[0],),
        device=device,
    )
    noise = torch.randn_like(inputs)

    enable_multifractal_loss = multifractal_predictor is not None and l_mf_weight > 0.0
    # `L_mf` backpropagates from x_hat to the denoising backbone. Under the
    # current CUDA AMP path this mixed-precision chain can cause NaN gradients
    # in early conv layers, so this branch is forced to FP32.
    amp_enabled = use_amp and device.type == "cuda" and not enable_multifractal_loss
    autocast_context = torch.amp.autocast(
        device_type=device.type,
        enabled=amp_enabled,
    )
    with autocast_context if device.type == "cuda" else nullcontext():
        x_t = diffusion.q_sample(
            x_start=inputs,
            timesteps=timesteps,
            noise=noise,
        )
        predicted_noise = model(
            x_t,
            timesteps,
            labels,
            descriptors,
            sequences,
        )
        per_sample_diff_loss = (noise - predicted_noise).pow(2).mean(dim=(1, 2))
        if min_snr_gamma is None:
            snr_weights = torch.ones_like(per_sample_diff_loss)
        else:
            snr_weights = diffusion.min_snr_weights(
                timesteps,
                min_snr_gamma,
            ).to(per_sample_diff_loss.dtype)
        diff_loss = (per_sample_diff_loss * snr_weights).mean()
    total_loss = diff_loss
    loss_mf = torch.zeros((), device=device, dtype=torch.float32)
    if enable_multifractal_loss:
        predicted_xstart = diffusion.predict_xstart_from_eps(
            x_t.float(),
            timesteps,
            predicted_noise.float(),
        )
        predicted_descriptor = multifractal_predictor(predicted_xstart)
        loss_mf = F.mse_loss(predicted_descriptor, target_descriptors.float())
        total_loss = total_loss + l_mf_weight * loss_mf

    scaled_total_loss = total_loss / float(grad_accum_steps)

    # CUDA AMP should be paired with GradScaler to avoid float16 underflow.
    if scaler is not None and amp_enabled:
        scaler.scale(scaled_total_loss).backward()
        if optimizer_step:
            scaler.unscale_(optimizer)
    else:
        scaled_total_loss.backward()
    trainable_parameters = _collect_trainable_parameters(model, multifractal_predictor)
    grad_norm = float("nan")
    if optimizer_step:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
        grad_norm = _compute_grad_norm(trainable_parameters)
        if scaler is not None and amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "loss": float(total_loss.detach().item()),
        "loss_diff": float(diff_loss.detach().item()),
        "loss_mf": float(loss_mf.detach().item()),
        "grad_norm": grad_norm,
        "hmc_drop_fraction": float(hmc_drop_mask.float().mean().item()),
        "vae_reconstruction_fraction": float(
            vae_reconstruction_mask.float().mean().item()
        ),
        "snr_weight": float(snr_weights.detach().mean().item()),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: HMCConditionedDiTPointCloud,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    auxiliary_state: dict[str, dict[str, Any]] | None = None,
    ema_model_state: dict[str, Tensor] | None = None,
) -> None:
    """Save a smoke-training checkpoint.

    Args:
        checkpoint_path: Output checkpoint path.
        model: Training model.
        optimizer: Optimizer state.
        scaler: Optional AMP gradient scaler.
        step: Global step number.
        epoch: Completed epoch number.
        metrics: Latest scalar metrics.
        auxiliary_state: Optional auxiliary module state payloads.
        ema_model_state: Optional averaged model state used for sampling.
    """
    write_checkpoint_payload(
        checkpoint_path,
        build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            epoch=epoch,
            metrics=metrics,
            auxiliary_state=auxiliary_state,
            ema_model_state=ema_model_state,
        ),
    )


def build_checkpoint_payload(
    model: HMCConditionedDiTPointCloud,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    auxiliary_state: dict[str, dict[str, Any]] | None = None,
    ema_model_state: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Assemble the checkpoint dictionary written by `save_checkpoint`."""
    return {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "metrics": metrics,
        "auxiliary_state": {} if auxiliary_state is None else auxiliary_state,
        "ema_model_state": ema_model_state,
    }


def write_checkpoint_payload(checkpoint_path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint payload via a temporary file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: Path,
    model: HMCConditionedDiTPointCloud,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    auxiliary_modules: dict[str, torch.nn.Module] | None = None,
    model_ema: ModelEMA | None = None,
) -> TrainingCheckpointState:
    """Load a saved training checkpoint into the active training state.

    Args:
        checkpoint_path: Saved checkpoint path.
        model: Training model to restore.
        optimizer: Optional optimizer to restore.
        scaler: Optional AMP scaler to restore.
        device: Active torch device.
        auxiliary_modules: Optional named auxiliary modules restored from the
            checkpoint payload.
        model_ema: Optional EMA tracker restored from the checkpoint payload.

    Returns:
        TrainingCheckpointState: Restored checkpoint metadata.
    """
    if not checkpoint_path.exists():
        message = f"Resume checkpoint does not exist: {checkpoint_path!s}."
        LOGGER.error(message)
        raise SmokeTrainingError(message)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    required_keys = (
        "epoch",
        "step",
        "model_state",
        "optimizer_state",
        "scaler_state",
        "metrics",
    )
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        message = (
            "Checkpoint is missing required keys: "
            + ", ".join(sorted(missing_keys))
            + "."
        )
        LOGGER.error(message)
        raise SmokeTrainingError(message)

    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint["scaler_state"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    checkpoint_auxiliary_state = checkpoint.get("auxiliary_state", {})
    if not isinstance(checkpoint_auxiliary_state, dict):
        message = "Checkpoint auxiliary_state must be stored as a mapping."
        LOGGER.error(message)
        raise SmokeTrainingError(message)
    if auxiliary_modules is not None:
        for module_name, module in auxiliary_modules.items():
            if module_name not in checkpoint_auxiliary_state:
                message = (
                    "Checkpoint is missing required auxiliary state for "
                    f"{module_name!r}."
                )
                LOGGER.error(message)
                raise SmokeTrainingError(message)
            module.load_state_dict(checkpoint_auxiliary_state[module_name])
    if model_ema is not None:
        ema_model_state = checkpoint.get("ema_model_state")
        if not isinstance(ema_model_state, dict):
            message = "Checkpoint is missing ema_model_state required for EMA resume."
            LOGGER.error(message)
            raise SmokeTrainingError(message)
        model_ema.load_state_dict(ema_model_state)

    metrics = checkpoint["metrics"]
    if not isinstance(metrics, dict):
        message = "Checkpoint metrics must be stored as a mapping."
        LOGGER.error(message)
        raise SmokeTrainingError(message)
    restored_metrics = {str(key): float(value) for key, value in metrics.items()}
    return TrainingCheckpointState(
        epoch=int(checkpoint["epoch"]),
        step=int(checkpoint["step"]),
        metrics=restored_metrics,
        auxiliary_state={
            str(key): value
            for key, value in checkpoint_auxiliary_state.items()
            if isinstance(value, dict)
        },
    )


def run_smoke_training(config: ExperimentConfig) -> dict[str, float]:
    """Run the smoke-training loop.

    Args:
        config: Full experiment configuration.

    Returns:
        dict[str, float]: Final training metrics.
    """
    set_global_seed(config.train.seed)
    device = detect_device(config.train.prefer_cuda)
    dataloader = build_dataloader(config)
    if len(dataloader) == 0:
        message = (
            "Smoke dataloader is empty. Check batch_size, drop_last, and dataset size."
        )
        LOGGER.error(message)
        raise SmokeTrainingError(message)
    extractor = HMCFeatureExtractor(config.hmc)
    diffusion = GaussianDiffusion(config.diffusion)
    model = build_model(config, device)
    model_ema = (
        ModelEMA(model, config.train.ema_decay) if config.train.use_ema else None
    )
    multifractal_predictor = (
        build_multifractal_predictor(config, device) if config.train.use_l_mf else None
    )
    optimizer = AdamW(
        chain(
            model.parameters(),
            ()
            if multifractal_predictor is None
            else multifractal_predictor.parameters(),
        ),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = (
        torch.amp.GradScaler(device="cuda", enabled=True)
        if config.train.use_amp and device.type == "cuda"
        else None
    )
    output_dir = config.train.results_dir / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting smoke training in %s", output_dir)
    LOGGER.info("Dataset categories: %s", ", ".join(config.data.categories))
    LOGGER.info("Dataset size: %d", len(dataloader.dataset))
    if config.train.use_l_mf and config.train.use_amp and device.type == "cuda":
        LOGGER.info(
            "L_mf is enabled; autocast is disabled inside train steps "
            "for gradient stability."
        )

    global_step = 0
    last_metrics = {
        "loss": float("nan"),
        "loss_diff": float("nan"),
        "loss_mf": float("nan"),
        "grad_norm": float("nan"),
        "hmc_drop_fraction": float("nan"),
        "snr_weight": float("nan"),
    }
    for epoch in range(config.train.epochs):
        model.train()
        for batch in dataloader:
            global_step += 1
            optimizer_step = global_step % config.train.grad_accum_steps == 0
            last_metrics = train_one_step(
                batch=batch,
                diffusion=diffusion,
                model=model,
                extractor=extractor,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=config.train.use_amp,
                grad_clip=config.train.grad_clip,
                grad_accum_steps=config.train.grad_accum_steps,
                optimizer_step=optimizer_step,
                multifractal_predictor=multifractal_predictor,
                l_mf_weight=config.train.l_mf_weight,
                hmc_point_source=config.data.hmc_point_source,
                hmc_dropout_prob=config.train.hmc_dropout_prob,
                min_snr_gamma=config.train.min_snr_gamma,
            )
            if optimizer_step and model_ema is not None:
                model_ema.update(model)
            if global_step % config.train.log_every == 0:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.6f "
                    "loss_diff=%.6f loss_mf=%.6f grad_norm=%.6f",
                    epoch,
                    global_step,
                    last_metrics["loss"],
                    last_metrics["loss_diff"],
                    last_metrics["loss_mf"],
                    last_metrics["grad_norm"],
                )
            if (
                config.train.max_steps is not None
                and global_step >= config.train.max_steps
            ):
                break
        if config.train.max_steps is not None and global_step >= config.train.max_steps:
            break

    checkpoint_path = output_dir / config.train.checkpoint_name
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=global_step,
        epoch=epoch,
        metrics=last_metrics,
        auxiliary_state=None
        if multifractal_predictor is None
        else {"multifractal_predictor": multifractal_predictor.state_dict()},
        ema_model_state=None if model_ema is None else model_ema.state_dict(),
    )
    LOGGER.info("Saved smoke checkpoint to %s", checkpoint_path)
    return {
        **last_metrics,
        "steps": float(global_step),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke runner."""
    parser = argparse.ArgumentParser(description="Run HMC-DiT3D smoke training.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path for the smoke experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for smoke training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_experiment_config(args.config)
    metrics = run_smoke_training(config)
    LOGGER.info(
        "Smoke training finished: loss=%.6f grad_norm=%.6f steps=%d",
        metrics["loss"],
        metrics["grad_norm"],
        int(metrics["steps"]),
    )


if __name__ == "__main__":
    main()
