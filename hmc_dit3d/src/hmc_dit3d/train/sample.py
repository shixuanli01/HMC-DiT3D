"""Reference-conditioned sampling entrypoints for HMC-DiT3D."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from hmc_dit3d.data.hmc_condition_bank import (
    HMCConditionBank,
    _serialize_hmc_config,
    load_hmc_condition_bank,
    sample_hmc_condition_bank,
)
from hmc_dit3d.data.shapenet_pc15k import ShapeNetPC15KDataset
from hmc_dit3d.hmc.condition_vae import (
    HMCConditionVAE,
    load_condition_vae_checkpoint,
)
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor
from hmc_dit3d.hmc.latent_gmm import (
    assign_diagonal_gmm,
    component_center_indices,
    fit_diagonal_gmm,
    kcenter_select,
    sample_diagonal_gmm,
)
from hmc_dit3d.train.config import ExperimentConfig, load_experiment_config
from hmc_dit3d.train.diffusion import GaussianDiffusion
from hmc_dit3d.train.smoke import (
    build_model,
    prepare_hmc_batch,
    select_hmc_condition_points,
)
from hmc_dit3d.utils.runtime import detect_device, set_global_seed

LOGGER = logging.getLogger(__name__)


class SamplingError(RuntimeError):
    """Raised when reference-conditioned sampling fails."""


def _parse_denoise_trajectory_steps(text: str) -> tuple[int, ...]:
    """Parse ``1000,600,...,0`` style lists for trajectory checkpoints."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        message = "--denoise-trajectory-steps must list at least one integer."
        LOGGER.error(message)
        raise SamplingError(message)
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        message = f"Invalid --denoise-trajectory-steps value: {text!r}."
        LOGGER.error(message)
        raise SamplingError(message) from exc


def _validate_denoise_trajectory_steps(
    steps: tuple[int, ...],
    num_timesteps: int,
) -> None:
    nt = int(num_timesteps)
    bad = sorted(t for t in steps if t != nt and not (0 <= t < nt))
    if bad:
        message = (
            f"denoise trajectory steps must be {nt} (initial noise) or "
            f"in [0, {nt - 1}], invalid: {bad!r}."
        )
        LOGGER.error(message)
        raise SamplingError(message)


def _synset_id_from_shape_npy_path(source_path: str | Path) -> str:
    """Return the synset folder from a ShapeNet point-cloud path."""
    resolved = Path(source_path).expanduser().resolve()
    return str(resolved.parent.parent.name)


def _collated_str_field(batch: dict[str, Any], key: str, count: int) -> list[str]:
    """Turn a default-collated batch field into ``count`` strings."""
    raw = batch[key]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw[:count]]
    if isinstance(raw, str):
        return [raw] * count
    message = f"Unexpected batch[{key!r}] type for sampling metadata: {type(raw)!r}."
    LOGGER.error(message)
    raise SamplingError(message)


def _load_reference_points_from_paths(
    source_paths: list[str],
    sample_size: int,
) -> Tensor:
    """Load deterministic reference point clouds from stored source paths.

    Args:
        source_paths: Absolute `.npy` paths stored in the condition source.
        sample_size: Number of points retained per reference cloud.

    Returns:
        Tensor: Reference point clouds with shape `(B, sample_size, 3)`.
    """
    reference_points: list[Tensor] = []
    for source_path in source_paths:
        points = np.load(source_path)
        if points.ndim != 2 or points.shape[1] != 3:
            message = (
                "Reference point cloud must have shape (N, 3). "
                f"Got {points.shape!r} from {source_path!r}."
            )
            LOGGER.error(message)
            raise SamplingError(message)
        if points.shape[0] < sample_size:
            message = (
                f"Reference point cloud {source_path!r} has only {points.shape[0]} "
                f"points, but sample_size={sample_size} was requested."
            )
            LOGGER.error(message)
            raise SamplingError(message)
        if not np.isfinite(points).all():
            message = f"Reference point cloud contains NaN or Inf: {source_path!r}."
            LOGGER.error(message)
            raise SamplingError(message)
        reference_points.append(
            torch.from_numpy(np.ascontiguousarray(points[:sample_size])).to(
                torch.float32
            )
        )
    return torch.stack(reference_points, dim=0)


def _merge_denoise_trajectories(
    chunks: list[dict[int, Tensor]],
) -> dict[int, Tensor]:
    """Concatenate per-batch trajectory dicts along batch dimension."""
    if not chunks:
        return {}
    keys = chunks[0].keys()
    out: dict[int, Tensor] = {}
    for key in keys:
        out[int(key)] = torch.cat([chunk[int(key)] for chunk in chunks], dim=0)
    return out


def _generate_batch_samples(
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    labels: Tensor,
    descriptors: Tensor,
    sequences: list[Tensor],
    *,
    point_count: int,
    device: torch.device,
    clip_denoised: bool,
    hmc_guidance_scale: float = 1.0,
    denoise_trajectory_stride: int | None = None,
    denoise_trajectory_steps: tuple[int, ...] | None = None,
) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
    """Run reverse diffusion for one batch of already-prepared HMC conditions.

    Args:
        diffusion: Diffusion helper.
        model: Trained point-cloud denoiser.
        labels: Category labels with shape `(B,)`.
        descriptors: HMC descriptors with shape `(B, D_hmc)`.
        sequences: One Hilbert sequence tensor per scale.
        point_count: Number of output points per sample.
        device: Target torch device.
        clip_denoised: Whether predicted clean points are clipped during sampling.
        hmc_guidance_scale: Classifier-free guidance scale for HMC conditions.
        denoise_trajectory_stride: If set (and ``denoise_trajectory_steps`` is None),
            collect ``x`` every ``stride`` DDPM indices plus initial noise.
        denoise_trajectory_steps: If set, collect only these labels (see
            ``GaussianDiffusion.p_sample_loop_collect_timesteps``). Mutually exclusive
            with ``denoise_trajectory_stride``.

    Returns:
        Generated point clouds with shape `(B, point_count, 3)` on CPU, or that tensor
        plus a trajectory dict mapping timestep label -> tensor with the same batch dim.
    """
    if denoise_trajectory_stride is not None and denoise_trajectory_steps is not None:
        message = (
            "Pass at most one of denoise_trajectory_stride and "
            "denoise_trajectory_steps."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if hmc_guidance_scale < 0.0:
        message = "hmc_guidance_scale must be greater than or equal to zero."
        LOGGER.error(message)
        raise SamplingError(message)
    batch_size = labels.shape[0]

    def _denoise_fn(
        x_t: Tensor,
        step_ids: Tensor,
        *,
        bound_labels: Tensor = labels,
        bound_descriptors: Tensor = descriptors,
        bound_sequences: list[Tensor] = sequences,
    ) -> Tensor:
        """Bind one batch of HMC conditions into the denoiser closure."""
        conditional = model(
            x_t,
            step_ids,
            bound_labels,
            bound_descriptors,
            bound_sequences,
        )
        if hmc_guidance_scale == 1.0:
            return conditional
        unconditional = model(
            x_t,
            step_ids,
            bound_labels,
            torch.zeros_like(bound_descriptors),
            [torch.zeros_like(sequence) for sequence in bound_sequences],
        )
        return unconditional + hmc_guidance_scale * (conditional - unconditional)

    if denoise_trajectory_stride is None and denoise_trajectory_steps is None:
        sampled = diffusion.p_sample_loop(
            denoise_fn=_denoise_fn,
            shape=(batch_size, model.out_channels, point_count),
            device=device,
            clip_denoised=clip_denoised,
        )
        return sampled.transpose(1, 2).cpu()

    if denoise_trajectory_steps is not None:
        final, trajectory = diffusion.p_sample_loop_collect_timesteps(
            denoise_fn=_denoise_fn,
            shape=(batch_size, model.out_channels, point_count),
            device=device,
            clip_denoised=clip_denoised,
            collect_at=denoise_trajectory_steps,
        )
    else:
        final, trajectory = diffusion.p_sample_loop_collect_timesteps(
            denoise_fn=_denoise_fn,
            shape=(batch_size, model.out_channels, point_count),
            device=device,
            clip_denoised=clip_denoised,
            collect_stride=denoise_trajectory_stride,
        )
    return final.transpose(1, 2).cpu(), trajectory


def load_sampling_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, GaussianDiffusion, dict[str, Any]]:
    """Load a trained checkpoint for sampling.

    Args:
        config: Full experiment configuration.
        checkpoint_path: Checkpoint path produced by training.
        device: Target torch device.

    Returns:
        tuple[torch.nn.Module, GaussianDiffusion, dict[str, Any]]:
            Loaded model, diffusion helper, and raw checkpoint payload.
    """
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.exists():
        message = f"Checkpoint file does not exist: {resolved_path!s}."
        LOGGER.error(message)
        raise SamplingError(message)
    try:
        checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved_path, map_location=device)
    required_keys = {"model_state", "step", "epoch"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        message = f"Checkpoint is missing required keys: {sorted(missing_keys)!r}."
        LOGGER.error(message)
        raise SamplingError(message)

    model = build_model(config, device)
    model_state = checkpoint["model_state"]
    if config.train.use_ema:
        ema_model_state = checkpoint.get("ema_model_state")
        if not isinstance(ema_model_state, dict):
            message = (
                "Config requests EMA sampling but checkpoint has no ema_model_state."
            )
            LOGGER.error(message)
            raise SamplingError(message)
        model_state = ema_model_state
    model.load_state_dict(model_state, strict=True)
    model.eval()
    diffusion = GaussianDiffusion(config.diffusion)
    return model, diffusion, checkpoint


def build_sampling_dataloader(
    config: ExperimentConfig,
    *,
    split: str,
) -> DataLoader[dict[str, Any]]:
    """Build the deterministic reference dataloader used for HMC conditions.

    Args:
        config: Full experiment configuration.
        split: Dataset split used to fetch reference conditions.

    Returns:
        DataLoader[dict[str, Any]]: Deterministic dataloader for sampling.
    """
    dataset = ShapeNetPC15KDataset(
        root_dir=config.data.root_dir,
        categories=config.data.categories,
        split=split,
        sample_size=config.data.sample_size,
        random_subsample=False,
        return_full_points=config.data.hmc_point_source == "full",
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )


def generate_bank_conditioned_samples(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    *,
    limit: int | None = None,
    bank_categories: tuple[str, ...] | None = None,
    clip_denoised: bool = False,
    hmc_guidance_scale: float = 1.0,
    device: torch.device | None = None,
    denoise_trajectory_stride: int | None = None,
    denoise_trajectory_steps: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Generate samples conditioned on an offline HMC condition bank.

    Args:
        config: Full experiment configuration.
        checkpoint_path: Trained checkpoint path.
        bank_path: Saved HMC condition-bank path.
        limit: Optional maximum number of samples to generate.
        bank_categories: Optional category subset used when drawing from the bank.
        clip_denoised: Whether reverse diffusion clips predicted clean points.
        hmc_guidance_scale: Classifier-free guidance scale for HMC conditions.
        device: Optional explicit torch device.
        denoise_trajectory_stride: If set (and ``denoise_trajectory_steps`` is None),
            collect trajectory on a stride grid.
        denoise_trajectory_steps: Optional explicit DDPM labels to store.

    Returns:
        dict[str, Any]: Generated samples, matched references, and metadata.
    """
    if limit is not None and limit <= 0:
        message = f"limit must be positive when provided, got {limit}."
        LOGGER.error(message)
        raise SamplingError(message)

    set_global_seed(config.train.seed)
    resolved_device = (
        device if device is not None else detect_device(config.train.prefer_cuda)
    )
    model, diffusion, checkpoint = load_sampling_checkpoint(
        config,
        checkpoint_path,
        resolved_device,
    )
    bank = load_hmc_condition_bank(bank_path)
    if bank.hmc_config != config.hmc:
        message = "HMC condition bank config does not match the experiment HMC config."
        LOGGER.error(message)
        raise SamplingError(message)

    num_samples = config.data.batch_size if limit is None else limit
    bank_generator = torch.Generator().manual_seed(config.train.seed)
    sampled_conditions = sample_hmc_condition_bank(
        bank,
        num_samples=num_samples,
        categories=bank_categories,
        replacement=True,
        generator=bank_generator,
        device=resolved_device,
    )
    use_traj = (
        denoise_trajectory_stride is not None or denoise_trajectory_steps is not None
    )
    with torch.no_grad():
        if not use_traj:
            sampled_bnc = _generate_batch_samples(
                diffusion,
                model,
                sampled_conditions["labels"],
                sampled_conditions["descriptors"],
                sampled_conditions["sequences"],
                point_count=config.data.sample_size,
                device=resolved_device,
                clip_denoised=clip_denoised,
                hmc_guidance_scale=hmc_guidance_scale,
            )
            denoise_trajectory = None
        else:
            sampled_bnc, denoise_trajectory = _generate_batch_samples(
                diffusion,
                model,
                sampled_conditions["labels"],
                sampled_conditions["descriptors"],
                sampled_conditions["sequences"],
                point_count=config.data.sample_size,
                device=resolved_device,
                clip_denoised=clip_denoised,
                hmc_guidance_scale=hmc_guidance_scale,
                denoise_trajectory_stride=denoise_trajectory_stride,
                denoise_trajectory_steps=denoise_trajectory_steps,
            )
    references = _load_reference_points_from_paths(
        sampled_conditions["source_paths"],
        config.data.sample_size,
    )
    bank_synset_ids = [
        _synset_id_from_shape_npy_path(p) for p in sampled_conditions["source_paths"]
    ]
    payload: dict[str, Any] = {
        "samples": sampled_bnc,
        "references": references,
        "labels": sampled_conditions["labels"].cpu(),
        "categories": sampled_conditions["categories"],
        "model_ids": sampled_conditions["model_ids"],
        "synset_ids": bank_synset_ids,
        "source_paths": sampled_conditions["source_paths"],
        "bank_indices": sampled_conditions["indices"].cpu(),
        "bank_path": str(Path(bank_path).expanduser().resolve()),
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "hmc_guidance_scale": float(hmc_guidance_scale),
        "condition_source": "bank",
        "config_path": None,
    }
    if denoise_trajectory is not None:
        payload["denoise_trajectory"] = denoise_trajectory
    return payload


@torch.no_grad()
def _sample_train_bank_gmm_conditions(
    vae: HMCConditionVAE,
    bank: HMCConditionBank,
    num_samples: int,
    *,
    device: torch.device,
    num_components: int,
    max_iterations: int,
    candidate_multiplier: int,
    candidate_source: str,
    stratified_batch_size: int | None,
    encode_batch_size: int,
    temperature: float,
    sequence_threshold: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a train-bank latent GMM and decode a diverse selected subset."""
    if bank.split != "train":
        message = (
            "The VAE GMM prior must be fitted from a train-split condition bank; "
            f"got split={bank.split!r}."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if not 1 <= num_components <= min(len(bank), num_samples):
        message = (
            "vae_gmm_components must be in [1, min(bank_size, num_samples)], "
            f"got {num_components}."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if max_iterations <= 0 or candidate_multiplier <= 0 or encode_batch_size <= 0:
        message = (
            "GMM iterations, candidate multiplier, and encode batch size must "
            "all be positive."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if candidate_source not in {"sampled", "train"}:
        message = (
            "GMM candidate source must be 'sampled' or 'train', got "
            f"{candidate_source!r}."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if stratified_batch_size is not None:
        if candidate_source != "train":
            message = "Stratified GMM batches require candidate_source='train'."
            LOGGER.error(message)
            raise SamplingError(message)
        if stratified_batch_size != num_components:
            message = (
                "Stratified GMM batch size must equal the number of components; "
                f"got batch_size={stratified_batch_size}, components={num_components}."
            )
            LOGGER.error(message)
            raise SamplingError(message)
        if num_samples % stratified_batch_size != 0:
            message = "num_samples must be divisible by the stratified GMM batch size."
            LOGGER.error(message)
            raise SamplingError(message)
    if vae.descriptor_dim != bank.descriptors.shape[1] or vae.sequence_lengths != tuple(
        sequence.shape[1] for sequence in bank.sequences
    ):
        message = "Condition VAE dimensions do not match the GMM training bank."
        LOGGER.error(message)
        raise SamplingError(message)

    posterior_chunks: list[Tensor] = []
    feature_chunks: list[Tensor] = []
    for start in range(0, len(bank), encode_batch_size):
        end = min(start + encode_batch_size, len(bank))
        descriptors = bank.descriptors[start:end].to(device)
        sequences = [sequence[start:end].to(device) for sequence in bank.sequences]
        features = vae.encode_features(descriptors, sequences)
        posterior_mean = vae.fc_mu(features)
        posterior_chunks.append(posterior_mean)
        feature_chunks.append(features)
    posterior_means = torch.cat(posterior_chunks, dim=0)
    encoder_features = torch.cat(feature_chunks, dim=0)
    mixture = fit_diagonal_gmm(
        posterior_means,
        num_components,
        max_iterations=max_iterations,
        seed=seed,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    if candidate_source == "sampled":
        candidate_count = max(num_samples, num_samples * candidate_multiplier)
        candidate_latents, candidate_components = sample_diagonal_gmm(
            mixture,
            candidate_count,
            balanced=True,
            temperature=temperature,
            generator=generator,
        )
        selection_features = candidate_latents
        selection_center = mixture.data_mean
        selection_scale = mixture.data_std
    else:
        candidate_latents = posterior_means
        if stratified_batch_size is None:
            candidate_components = assign_diagonal_gmm(mixture, candidate_latents)
        else:
            standardized_candidates = (
                candidate_latents - mixture.data_mean
            ) / mixture.data_std
            standardized_means = (mixture.means - mixture.data_mean) / mixture.data_std
            candidate_components = torch.cdist(
                standardized_candidates,
                standardized_means,
            ).argmin(dim=1)
        candidate_count = int(candidate_latents.shape[0])
        # The pre-latent representation retains condition details that the KL
        # bottleneck intentionally removes, so it is a better diversity space
        # while selected latents remain valid train-posterior prototypes.
        selection_features = encoder_features
        selection_center = None
        selection_scale = None
    if stratified_batch_size is None:
        anchors = component_center_indices(
            candidate_latents,
            candidate_components,
            mixture,
        )
        selected_indices = kcenter_select(
            selection_features,
            num_samples,
            center=selection_center,
            scale=selection_scale,
            initial_indices=anchors,
        )
        selected_indices = selected_indices[
            torch.randperm(num_samples, generator=generator, device=device)
        ]
        selection_strategy = "global_kcenter"
    else:
        num_batches = num_samples // stratified_batch_size
        component_indices = [
            (candidate_components == component).nonzero(as_tuple=False).flatten()
            for component in range(num_components)
        ]
        if all(indices.numel() >= num_batches for indices in component_indices):
            per_component: list[Tensor] = []
            for indices in component_indices:
                local_indices = kcenter_select(
                    selection_features[indices],
                    num_batches,
                )
                per_component.append(indices[local_indices])
            batch_rows = [
                torch.stack([indices[batch] for indices in per_component])
                for batch in range(num_batches)
            ]
            selection_strategy = "component_stratified_kcenter"
        else:
            counts = [int(indices.numel()) for indices in component_indices]
            LOGGER.warning(
                "GMM components are too imbalanced for strict stratification "
                "(counts=%s); using independent batchwise K-center",
                counts,
            )
            remaining = torch.arange(candidate_count, device=device)
            batch_rows = []
            for _ in range(num_batches):
                local_indices = kcenter_select(
                    selection_features[remaining],
                    stratified_batch_size,
                )
                row = remaining[local_indices]
                batch_rows.append(row)
                keep = torch.ones(
                    remaining.numel(),
                    dtype=torch.bool,
                    device=device,
                )
                keep[local_indices] = False
                remaining = remaining[keep]
            selection_strategy = "batchwise_kcenter_fallback"
        shuffled_rows: list[Tensor] = []
        for row in batch_rows:
            shuffled_rows.append(
                row[
                    torch.randperm(
                        stratified_batch_size,
                        generator=generator,
                        device=device,
                    )
                ]
            )
        selected_indices = torch.cat(shuffled_rows)
    selected_latents = candidate_latents[selected_indices]
    selected_components = candidate_components[selected_indices]
    conditions = vae.decode_conditions(
        selected_latents,
        sequence_threshold=sequence_threshold,
    )
    component_counts = selected_components.bincount(minlength=num_components)
    metadata: dict[str, Any] = {
        "vae_prior": "train_gmm_kcenter",
        "vae_gmm_components": int(num_components),
        "vae_gmm_iterations": int(mixture.iterations),
        "vae_gmm_log_likelihood": float(mixture.log_likelihood),
        "vae_gmm_candidate_count": int(candidate_count),
        "vae_gmm_candidate_multiplier": int(candidate_multiplier),
        "vae_gmm_candidate_source": candidate_source,
        "vae_gmm_selection_space": (
            "latent" if candidate_source == "sampled" else "encoder_features"
        ),
        "vae_gmm_selection_strategy": selection_strategy,
        "vae_gmm_stratified_batch_size": stratified_batch_size,
        "vae_gmm_encode_batch_size": int(encode_batch_size),
        "vae_gmm_balanced_sampling": True,
        "vae_gmm_selected_component_counts": component_counts.cpu(),
        "vae_gmm_selected_components": selected_components.cpu(),
        "vae_gmm_weights": mixture.weights.cpu(),
    }
    LOGGER.info(
        "Fitted %d-component train-bank latent GMM in %d iterations "
        "(mean log likelihood %.6f); selected %d conditions from %d candidates",
        num_components,
        mixture.iterations,
        mixture.log_likelihood,
        num_samples,
        candidate_count,
    )
    return conditions, metadata


def generate_vae_conditioned_samples(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    vae_path: str | Path,
    *,
    split: str = "test",
    limit: int | None = None,
    vae_prior: str = "normal",
    vae_prior_bank_path: str | Path | None = None,
    vae_temperature: float = 1.0,
    vae_sequence_threshold: float = 0.0,
    vae_gmm_components: int = 32,
    vae_gmm_iterations: int = 100,
    vae_gmm_candidate_multiplier: int = 8,
    vae_gmm_candidate_source: str = "sampled",
    vae_gmm_stratified_batch_size: int | None = None,
    vae_gmm_encode_batch_size: int = 128,
    generation_batch_size: int | None = None,
    clip_denoised: bool = False,
    hmc_guidance_scale: float = 1.0,
    device: torch.device | None = None,
    denoise_trajectory_stride: int | None = None,
    denoise_trajectory_steps: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Generate samples conditioned on VAE-sampled synthetic HMC conditions.

    Conditions are drawn from a learned prior over raw HMC features
    (descriptor + Hilbert measure sequences), so no ground-truth point cloud
    is consulted to build the conditioning signal. Reference clouds from the
    requested split are stored alongside samples purely for metric evaluation.

    Args:
        config: Full experiment configuration.
        checkpoint_path: Trained diffusion checkpoint path.
        vae_path: Trained condition-VAE checkpoint path.
        split: Dataset split providing evaluation reference clouds.
        limit: Optional number of samples to generate (defaults to batch size).
        vae_prior: Either ``normal`` or train-bank ``gmm`` plus K-center.
        vae_prior_bank_path: Train condition bank used to fit the GMM prior.
        vae_temperature: Latent prior std multiplier.
        vae_sequence_threshold: Relative sparsification threshold applied to
            decoded measure sequences (0 disables it).
        vae_gmm_components: Number of diagonal mixture components.
        vae_gmm_iterations: Maximum number of EM iterations.
        vae_gmm_candidate_multiplier: Candidate pool size relative to output.
        vae_gmm_candidate_source: Draw candidates from the GMM or use train
            posterior means as on-manifold prototypes.
        vae_gmm_stratified_batch_size: If set, build every output batch with
            one train prototype from each GMM component.
        vae_gmm_encode_batch_size: Batch size when encoding the train bank.
        generation_batch_size: Optional GPU batch size for reverse diffusion.
            This only chunks generation; the returned payload still contains
            ``limit`` samples in its original order.
        clip_denoised: Whether reverse diffusion clips predicted clean points.
        hmc_guidance_scale: Classifier-free guidance scale for HMC conditions.
        device: Optional explicit torch device.
        denoise_trajectory_stride: If set (and ``denoise_trajectory_steps`` is
            None), collect trajectory on a stride grid.
        denoise_trajectory_steps: Optional explicit DDPM labels to store.

    Returns:
        dict[str, Any]: Generated samples, evaluation references, and metadata.
    """
    if limit is not None and limit <= 0:
        message = f"limit must be positive when provided, got {limit}."
        LOGGER.error(message)
        raise SamplingError(message)
    if generation_batch_size is not None and generation_batch_size <= 0:
        message = (
            "generation_batch_size must be positive when provided, got "
            f"{generation_batch_size}."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if vae_prior not in {"normal", "gmm"}:
        message = f"vae_prior must be 'normal' or 'gmm', got {vae_prior!r}."
        LOGGER.error(message)
        raise SamplingError(message)
    if vae_prior == "gmm" and vae_prior_bank_path is None:
        message = "vae_prior_bank_path is required when vae_prior='gmm'."
        LOGGER.error(message)
        raise SamplingError(message)

    set_global_seed(config.train.seed)
    resolved_device = (
        device if device is not None else detect_device(config.train.prefer_cuda)
    )
    model, diffusion, checkpoint = load_sampling_checkpoint(
        config,
        checkpoint_path,
        resolved_device,
    )
    vae, vae_payload = load_condition_vae_checkpoint(vae_path, device=resolved_device)
    if vae_payload.get("hmc_config") != _serialize_hmc_config(config.hmc):
        message = "Condition-VAE HMC config does not match the experiment HMC config."
        LOGGER.error(message)
        raise SamplingError(message)

    num_samples = config.data.batch_size if limit is None else limit
    prior_metadata: dict[str, Any]
    if vae_prior == "normal":
        condition_generator = torch.Generator(device=resolved_device).manual_seed(
            config.train.seed
        )
        conditions = vae.sample(
            num_samples,
            device=resolved_device,
            temperature=vae_temperature,
            sequence_threshold=vae_sequence_threshold,
            generator=condition_generator,
        )
        prior_metadata = {"vae_prior": "normal"}
    else:
        prior_bank = load_hmc_condition_bank(vae_prior_bank_path)
        if prior_bank.hmc_config != config.hmc:
            message = "GMM prior bank HMC config does not match the experiment."
            LOGGER.error(message)
            raise SamplingError(message)
        conditions, prior_metadata = _sample_train_bank_gmm_conditions(
            vae,
            prior_bank,
            num_samples,
            device=resolved_device,
            num_components=vae_gmm_components,
            max_iterations=vae_gmm_iterations,
            candidate_multiplier=vae_gmm_candidate_multiplier,
            candidate_source=vae_gmm_candidate_source,
            stratified_batch_size=vae_gmm_stratified_batch_size,
            encode_batch_size=vae_gmm_encode_batch_size,
            temperature=vae_temperature,
            sequence_threshold=vae_sequence_threshold,
            seed=config.train.seed,
        )
        prior_metadata["vae_prior_bank_path"] = str(
            Path(vae_prior_bank_path).expanduser().resolve()
        )

    reference_dataset = ShapeNetPC15KDataset(
        root_dir=config.data.root_dir,
        categories=config.data.categories,
        split=split,
        sample_size=config.data.sample_size,
        random_subsample=False,
    )
    reference_generator = torch.Generator().manual_seed(config.train.seed)
    if num_samples <= len(reference_dataset):
        reference_indices = torch.randperm(
            len(reference_dataset), generator=reference_generator
        )[:num_samples]
    else:
        reference_indices = torch.randint(
            0, len(reference_dataset), (num_samples,), generator=reference_generator
        )
    references: list[Tensor] = []
    labels: list[int] = []
    categories: list[str] = []
    reference_model_ids: list[str] = []
    reference_synset_ids: list[str] = []
    for index in reference_indices.tolist():
        entry = reference_dataset[index]
        references.append(torch.as_tensor(entry["points"], dtype=torch.float32))
        labels.append(int(entry["label"]))
        categories.append(str(entry["category"]))
        reference_model_ids.append(str(entry["model_id"]))
        reference_synset_ids.append(_synset_id_from_shape_npy_path(entry["path"]))

    label_tensor = torch.tensor(labels, dtype=torch.int64, device=resolved_device)
    use_traj = (
        denoise_trajectory_stride is not None or denoise_trajectory_steps is not None
    )
    chunk_size = (
        num_samples
        if generation_batch_size is None
        else min(generation_batch_size, num_samples)
    )
    sample_chunks: list[Tensor] = []
    trajectory_chunks: list[dict[int, Tensor]] = []
    with torch.no_grad():
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            generated = _generate_batch_samples(
                diffusion,
                model,
                label_tensor[start:end],
                conditions["descriptors"][start:end],
                [sequence[start:end] for sequence in conditions["sequences"]],
                point_count=config.data.sample_size,
                device=resolved_device,
                clip_denoised=clip_denoised,
                hmc_guidance_scale=hmc_guidance_scale,
                denoise_trajectory_stride=denoise_trajectory_stride,
                denoise_trajectory_steps=denoise_trajectory_steps,
            )
            if use_traj:
                sample_chunk, trajectory_chunk = generated
                trajectory_chunks.append(trajectory_chunk)
            else:
                sample_chunk = generated
            sample_chunks.append(sample_chunk)
            LOGGER.info(
                "Generated VAE-prior samples %d:%d of %d",
                start,
                end,
                num_samples,
            )
    sampled_bnc = torch.cat(sample_chunks, dim=0)
    denoise_trajectory = (
        _merge_denoise_trajectories(trajectory_chunks) if use_traj else None
    )

    payload: dict[str, Any] = {
        "samples": sampled_bnc,
        "references": torch.stack(references, dim=0),
        "labels": label_tensor.cpu(),
        "categories": categories,
        "reference_model_ids": reference_model_ids,
        "reference_synset_ids": reference_synset_ids,
        "reference_split": split,
        "vae_path": str(Path(vae_path).expanduser().resolve()),
        "vae_temperature": float(vae_temperature),
        "vae_sequence_threshold": float(vae_sequence_threshold),
        "generation_batch_size": int(chunk_size),
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "hmc_guidance_scale": float(hmc_guidance_scale),
        "condition_source": "vae",
        "config_path": None,
    }
    payload.update(prior_metadata)
    if denoise_trajectory is not None:
        payload["denoise_trajectory"] = denoise_trajectory
    return payload


def generate_reference_conditioned_samples(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    split: str,
    limit: int | None = None,
    clip_denoised: bool = False,
    hmc_guidance_scale: float = 1.0,
    device: torch.device | None = None,
    denoise_trajectory_stride: int | None = None,
    denoise_trajectory_steps: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Generate samples conditioned on reference HMC descriptors.

    The current project has not yet implemented a learned HMC prior. To keep the
    experimental contract explicit, sampling here uses real reference point clouds
    to derive HMC conditions and generates one sample per reference shape.

    Args:
        config: Full experiment configuration.
        checkpoint_path: Trained checkpoint path.
        split: Dataset split used to fetch reference shapes.
        limit: Optional maximum number of samples to generate.
        clip_denoised: Whether reverse diffusion clips predicted clean points.
        hmc_guidance_scale: Classifier-free guidance scale for HMC conditions.
        device: Optional explicit torch device.
        denoise_trajectory_stride: If set (and ``denoise_trajectory_steps`` is None),
            collect trajectory on a stride grid.
        denoise_trajectory_steps: Optional explicit DDPM labels to store.

    Returns:
        dict[str, Any]: Generated samples, references, and metadata.
    """
    if limit is not None and limit <= 0:
        message = f"limit must be positive when provided, got {limit}."
        LOGGER.error(message)
        raise SamplingError(message)

    set_global_seed(config.train.seed)
    resolved_device = (
        device if device is not None else detect_device(config.train.prefer_cuda)
    )
    model, diffusion, checkpoint = load_sampling_checkpoint(
        config,
        checkpoint_path,
        resolved_device,
    )
    dataloader = build_sampling_dataloader(config, split=split)
    extractor = HMCFeatureExtractor(config.hmc)

    samples: list[Tensor] = []
    references: list[Tensor] = []
    labels: list[Tensor] = []
    categories: list[str] = []
    model_ids: list[str] = []
    synset_ids: list[str] = []
    traj_chunks: list[dict[int, Tensor]] = []
    use_traj = (
        denoise_trajectory_stride is not None or denoise_trajectory_steps is not None
    )

    # Reuse real reference-cloud HMC conditions to strictly validate condition-chain
    # generation, without presenting this path as an HMC-prior free generator.
    with torch.no_grad():
        for batch in dataloader:
            points = torch.as_tensor(batch["points"], dtype=torch.float32)
            batch_labels = torch.as_tensor(
                batch["label"],
                device=resolved_device,
                dtype=torch.int64,
            )
            hmc_points = select_hmc_condition_points(
                batch,
                config.data.hmc_point_source,
            )
            descriptors, sequences = prepare_hmc_batch(
                hmc_points,
                extractor,
                resolved_device,
            )
            if not use_traj:
                sampled_bnc = _generate_batch_samples(
                    diffusion,
                    model,
                    batch_labels,
                    descriptors,
                    sequences,
                    point_count=config.data.sample_size,
                    device=resolved_device,
                    clip_denoised=clip_denoised,
                    hmc_guidance_scale=hmc_guidance_scale,
                )
                traj_batch = None
            else:
                sampled_bnc, traj_batch = _generate_batch_samples(
                    diffusion,
                    model,
                    batch_labels,
                    descriptors,
                    sequences,
                    point_count=config.data.sample_size,
                    device=resolved_device,
                    clip_denoised=clip_denoised,
                    hmc_guidance_scale=hmc_guidance_scale,
                    denoise_trajectory_stride=denoise_trajectory_stride,
                    denoise_trajectory_steps=denoise_trajectory_steps,
                )
            points = points.cpu()

            if limit is not None:
                remaining = limit - len(samples)
                if remaining <= 0:
                    break
                sampled_bnc = sampled_bnc[:remaining]
                points = points[:remaining]
                batch_labels = batch_labels[:remaining]
                batch_categories = list(batch["category"][:remaining])
                batch_model_ids = _collated_str_field(batch, "model_id", remaining)
                batch_synset_ids = _collated_str_field(batch, "synset_id", remaining)
                if traj_batch is not None:
                    traj_batch = {k: v[:remaining] for k, v in traj_batch.items()}
            else:
                n_batch = int(points.shape[0])
                batch_categories = list(batch["category"])
                batch_model_ids = _collated_str_field(batch, "model_id", n_batch)
                batch_synset_ids = _collated_str_field(batch, "synset_id", n_batch)

            samples.extend(sampled_bnc.unbind(dim=0))
            references.extend(points.unbind(dim=0))
            labels.extend(batch_labels.cpu().unbind(dim=0))
            categories.extend(batch_categories)
            model_ids.extend(batch_model_ids)
            synset_ids.extend(batch_synset_ids)

            if traj_batch is not None:
                traj_chunks.append(traj_batch)

            if limit is not None and len(samples) >= limit:
                break

    if len(samples) == 0:
        message = "No samples were generated. Check dataset split and limit settings."
        LOGGER.error(message)
        raise SamplingError(message)

    payload_ref: dict[str, Any] = {
        "samples": torch.stack(samples, dim=0),
        "references": torch.stack(references, dim=0),
        "labels": torch.stack(labels, dim=0),
        "categories": categories,
        "model_ids": model_ids,
        "synset_ids": synset_ids,
        "split": split,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "hmc_guidance_scale": float(hmc_guidance_scale),
        "condition_source": "reference",
        "config_path": None,
    }
    if use_traj:
        payload_ref["denoise_trajectory"] = _merge_denoise_trajectories(traj_chunks)
    return payload_ref


def save_denoise_trajectory_shards(
    payload: dict[str, Any],
    trajectory_dir: str | Path,
) -> Path:
    """Write one ``.pt`` file per collected timestep into ``trajectory_dir``.

    Each shard contains ``timestep``, ``samples_bnc`` with shape ``(B, N, 3)``,
    and ``model_ids`` when present in the payload.
    """
    traj = payload.get("denoise_trajectory")
    if not traj:
        message = "Payload has no denoise_trajectory; nothing to export."
        LOGGER.error(message)
        raise SamplingError(message)
    resolved = Path(trajectory_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    model_ids = payload.get("model_ids")
    for t_key, tensor in sorted(traj.items(), key=lambda kv: int(kv[0])):
        ti = int(t_key)
        shard = {
            "timestep": ti,
            "samples_bnc": tensor,
        }
        if isinstance(model_ids, list):
            shard["model_ids"] = model_ids
        torch.save(shard, resolved / f"timestep_{ti:05d}.pt")
    LOGGER.info(
        "Wrote %d denoise trajectory shards to %s",
        len(traj),
        resolved,
    )
    return resolved


def save_sample_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Persist generated samples and metadata to disk.

    Writes:
        - ``*.pt`` full payload with samples, references, and metadata.
        - ``*_manifest.json`` when IDs match the samples batch dimension.

    Args:
        payload: Sampling payload returned by `generate_reference_conditioned_samples`.
        output_path: Target `.pt` file path.

    Returns:
        Path: Resolved output path.
    """
    resolved_path = Path(output_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved_path)
    samples = payload.get("samples")
    model_ids = payload.get("model_ids")
    if hasattr(samples, "shape") and isinstance(model_ids, list):
        batch = int(samples.shape[0])
        if batch > 0 and len(model_ids) == batch:
            manifest: dict[str, Any] = {
                "model_ids": model_ids,
                "n_samples": batch,
                "note": (
                    "Row k matches samples[k], references[k], and categories[k] "
                    "when present."
                ),
            }
            synset_ids = payload.get("synset_ids")
            if isinstance(synset_ids, list) and len(synset_ids) == batch:
                manifest["synset_ids"] = synset_ids
            cats = payload.get("categories")
            if isinstance(cats, list) and len(cats) == batch:
                manifest["categories"] = cats
            manifest_path = resolved_path.with_name(
                f"{resolved_path.stem}_manifest.json"
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("Saved sample id manifest to %s", manifest_path)
    return resolved_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for HMC-conditioned sampling."""
    parser = argparse.ArgumentParser(
        description="Generate HMC-conditioned samples from a trained checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path for the trained experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path produced by training.",
    )
    parser.add_argument(
        "--condition-source",
        type=str,
        choices=("reference", "bank", "vae"),
        default="reference",
        help=(
            "Choose whether HMC conditions come from a dataset split, a saved "
            "bank, or a learned VAE prior."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=(
            "Dataset split used when --condition-source=reference, or the "
            "evaluation-reference split when --condition-source=vae."
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=None,
        help="Saved HMC condition-bank path used when --condition-source=bank.",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=None,
        help="Condition-VAE checkpoint used when --condition-source=vae.",
    )
    parser.add_argument(
        "--vae-prior",
        choices=("normal", "gmm"),
        default="normal",
        help=(
            "Latent prior for VAE conditions. The gmm option fits only the "
            "train bank supplied through --vae-prior-bank."
        ),
    )
    parser.add_argument(
        "--vae-prior-bank",
        type=Path,
        default=None,
        help="Train-split HMC condition bank used by --vae-prior=gmm.",
    )
    parser.add_argument(
        "--vae-temperature",
        type=float,
        default=1.0,
        help="Latent prior std multiplier used when --condition-source=vae.",
    )
    parser.add_argument(
        "--vae-sequence-threshold",
        type=float,
        default=0.0,
        help=(
            "Relative sparsification threshold for decoded measure sequences "
            "(entries below threshold/L are zeroed, then renormalized)."
        ),
    )
    parser.add_argument("--vae-gmm-components", type=int, default=32)
    parser.add_argument("--vae-gmm-iterations", type=int, default=100)
    parser.add_argument("--vae-gmm-candidate-multiplier", type=int, default=8)
    parser.add_argument(
        "--vae-gmm-candidate-source",
        choices=("sampled", "train"),
        default="sampled",
        help=(
            "Use synthetic GMM samples or on-manifold train posterior means "
            "as the K-center candidate pool."
        ),
    )
    parser.add_argument(
        "--vae-gmm-stratified-batch-size",
        type=int,
        default=None,
        help=(
            "Build each output group with one prototype per component. The "
            "value must equal --vae-gmm-components and divide --limit."
        ),
    )
    parser.add_argument("--vae-gmm-encode-batch-size", type=int, default=128)
    parser.add_argument(
        "--bank-categories",
        nargs="+",
        default=None,
        help="Optional category subset used when drawing from the HMC bank.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to generate.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Optional GPU batch size for reverse diffusion when using a VAE "
            "condition source. The saved payload still contains --limit samples."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .pt file for generated samples and metadata.",
    )
    parser.add_argument(
        "--clip-denoised",
        action="store_true",
        help="Clip predicted clean points into [-1, 1] during reverse diffusion.",
    )
    parser.add_argument(
        "--hmc-guidance-scale",
        type=float,
        default=1.0,
        help="HMC classifier-free guidance scale (1 disables guidance).",
    )
    parser.add_argument(
        "--denoise-trajectory-every",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, record intermediate point clouds every N DDPM steps "
            "(stores denoise_trajectory in the output payload). "
            "Ignored when --denoise-trajectory-steps is set."
        ),
    )
    parser.add_argument(
        "--denoise-trajectory-steps",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated DDPM labels to save, e.g. 1000,600,...,0. "
            "Use the configured num_timesteps (e.g. 1000) for initial noise x_T. "
            "Overrides --denoise-trajectory-every when both are provided."
        ),
    )
    parser.add_argument(
        "--denoise-trajectory-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for per-timestep .pt shards "
            "(requires --denoise-trajectory-every or --denoise-trajectory-steps)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for HMC-conditioned sampling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    config = load_experiment_config(args.config)
    traj_steps: tuple[int, ...] | None = None
    if args.denoise_trajectory_steps is not None:
        traj_steps = _parse_denoise_trajectory_steps(args.denoise_trajectory_steps)
        _validate_denoise_trajectory_steps(traj_steps, config.diffusion.num_timesteps)

    traj_stride = args.denoise_trajectory_every
    if traj_steps is not None and traj_stride is not None:
        LOGGER.warning(
            "Both --denoise-trajectory-steps and --denoise-trajectory-every set; "
            "using steps only."
        )
        traj_stride = None

    if (
        args.denoise_trajectory_dir is not None
        and traj_steps is None
        and args.denoise_trajectory_every is None
    ):
        message = (
            "--denoise-trajectory-dir requires --denoise-trajectory-steps "
            "or --denoise-trajectory-every."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if args.denoise_trajectory_every is not None and args.denoise_trajectory_every <= 0:
        message = (
            "--denoise-trajectory-every must be a positive integer when provided, "
            f"got {args.denoise_trajectory_every}."
        )
        LOGGER.error(message)
        raise SamplingError(message)
    if args.condition_source == "reference":
        payload = generate_reference_conditioned_samples(
            config,
            args.checkpoint,
            split=args.split,
            limit=args.limit,
            clip_denoised=args.clip_denoised,
            hmc_guidance_scale=args.hmc_guidance_scale,
            denoise_trajectory_stride=traj_stride,
            denoise_trajectory_steps=traj_steps,
        )
    elif args.condition_source == "vae":
        if args.vae is None:
            message = "--vae is required when --condition-source=vae."
            LOGGER.error(message)
            raise SamplingError(message)
        payload = generate_vae_conditioned_samples(
            config,
            args.checkpoint,
            args.vae,
            split=args.split,
            limit=args.limit,
            vae_prior=args.vae_prior,
            vae_prior_bank_path=args.vae_prior_bank,
            vae_temperature=args.vae_temperature,
            vae_sequence_threshold=args.vae_sequence_threshold,
            vae_gmm_components=args.vae_gmm_components,
            vae_gmm_iterations=args.vae_gmm_iterations,
            vae_gmm_candidate_multiplier=args.vae_gmm_candidate_multiplier,
            vae_gmm_candidate_source=args.vae_gmm_candidate_source,
            vae_gmm_stratified_batch_size=args.vae_gmm_stratified_batch_size,
            vae_gmm_encode_batch_size=args.vae_gmm_encode_batch_size,
            generation_batch_size=args.generation_batch_size,
            clip_denoised=args.clip_denoised,
            hmc_guidance_scale=args.hmc_guidance_scale,
            denoise_trajectory_stride=traj_stride,
            denoise_trajectory_steps=traj_steps,
        )
    else:
        if args.bank is None:
            message = "--bank is required when --condition-source=bank."
            LOGGER.error(message)
            raise SamplingError(message)
        payload = generate_bank_conditioned_samples(
            config,
            args.checkpoint,
            args.bank,
            limit=args.limit,
            bank_categories=None
            if args.bank_categories is None
            else tuple(args.bank_categories),
            clip_denoised=args.clip_denoised,
            hmc_guidance_scale=args.hmc_guidance_scale,
            denoise_trajectory_stride=traj_stride,
            denoise_trajectory_steps=traj_steps,
        )
    payload["config_path"] = str(args.config.expanduser().resolve())
    if traj_steps is not None:
        payload["denoise_trajectory_steps"] = list(traj_steps)
    output_path = save_sample_payload(payload, args.output)
    if args.denoise_trajectory_dir is not None:
        save_denoise_trajectory_shards(payload, args.denoise_trajectory_dir)
    LOGGER.info(
        "Saved %d %s-conditioned samples to %s",
        payload["samples"].shape[0],
        payload["condition_source"],
        output_path,
    )


if __name__ == "__main__":
    main()
