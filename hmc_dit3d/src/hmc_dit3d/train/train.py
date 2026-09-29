"""Formal training entrypoint for HMC-DiT3D experiments."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from hmc_dit3d.data.hmc_condition_bank import _serialize_hmc_config
from hmc_dit3d.hmc.condition_vae import (
    HMCConditionVAE,
    load_condition_vae_checkpoint,
)
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor
from hmc_dit3d.train.config import ExperimentConfig, load_experiment_config
from hmc_dit3d.train.diffusion import GaussianDiffusion
from hmc_dit3d.train.ema import ModelEMA
from hmc_dit3d.train.smoke import (
    SmokeTrainingError,
    apply_condition_vae_reconstruction,
    build_dataloader,
    build_eval_dataloader,
    build_model,
    build_multifractal_predictor,
    load_training_checkpoint,
    prepare_hmc_batch,
    save_checkpoint,
    select_hmc_condition_points,
    train_one_step,
)
from hmc_dit3d.utils.runtime import detect_device, set_global_seed

LOGGER = logging.getLogger(__name__)


class TrainingRunError(RuntimeError):
    """Raised when the formal training pipeline fails."""


def append_metrics(metrics_path: Path, payload: dict[str, float | int]) -> None:
    """Append one metrics record to a JSONL file.

    Args:
        metrics_path: Output JSONL path.
        payload: Flat scalar metrics payload.
    """
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def append_evolution(
    evolution_path: Path,
    payload: dict[str, float | int | str | None],
) -> None:
    """Append one epoch-level evolution record to a JSONL file.

    Args:
        evolution_path: Output JSONL path.
        payload: Flat epoch-level payload.
    """
    evolution_path.parent.mkdir(parents=True, exist_ok=True)
    with evolution_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _save_epoch_checkpoint(
    output_dir: Path,
    epoch_index: int,
    model: torch.nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
    step: int,
    metrics: dict[str, float],
    auxiliary_state: dict[str, dict[str, object]] | None = None,
    ema_model_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Save one epoch-level checkpoint aligned with baseline naming habits.

    Args:
        output_dir: Experiment output directory.
        epoch_index: Zero-based epoch index.
        model: Training model.
        optimizer: Optimizer state.
        scaler: Optional AMP scaler.
        step: Global optimization step.
        metrics: Latest scalar metrics.
        auxiliary_state: Optional auxiliary module state payloads.

    Returns:
        Path: Saved checkpoint path.
    """
    checkpoint_path = output_dir / f"epoch_{epoch_index + 1}.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        epoch=epoch_index,
        metrics=metrics,
        auxiliary_state=auxiliary_state,
        ema_model_state=ema_model_state,
    )
    return checkpoint_path


def _save_named_checkpoint(
    checkpoint_path: Path,
    epoch_index: int,
    model: torch.nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler | None,
    step: int,
    metrics: dict[str, float],
    auxiliary_state: dict[str, dict[str, object]] | None = None,
    ema_model_state: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Save one named checkpoint such as `latest`, `best_train`, or `best_val`.

    Args:
        checkpoint_path: Output checkpoint path.
        epoch_index: Zero-based epoch index.
        model: Training model.
        optimizer: Optimizer state.
        scaler: Optional AMP scaler.
        step: Global optimization step.
        metrics: Scalar metrics stored into the checkpoint.
        auxiliary_state: Optional auxiliary module state payloads.

    Returns:
        Path: Saved checkpoint path.
    """
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        epoch=epoch_index,
        metrics=metrics,
        auxiliary_state=auxiliary_state,
        ema_model_state=ema_model_state,
    )
    return checkpoint_path


def _average_epoch_metrics(
    metric_total: dict[str, float],
    batch_count: int,
) -> dict[str, float]:
    """Average accumulated per-step metrics into one epoch-level summary."""
    if batch_count <= 0:
        message = "Epoch metric averaging requires at least one batch."
        LOGGER.error(message)
        raise TrainingRunError(message)
    return {key: value / float(batch_count) for key, value in metric_total.items()}


def _load_checkpoint_loss(checkpoint_path: Path, metric_name: str) -> float:
    """Load one scalar metric from an existing checkpoint if available."""
    if not checkpoint_path.exists():
        return float("inf")
    payload = torch.load(checkpoint_path, map_location="cpu")
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        return float("inf")
    value = metrics.get(metric_name)
    if value is None:
        return float("inf")
    return float(value)


def evaluate_validation_loss(
    config: ExperimentConfig,
    *,
    dataloader: torch.utils.data.DataLoader[dict[str, object]],
    diffusion: GaussianDiffusion,
    model: torch.nn.Module,
    extractor: HMCFeatureExtractor,
    multifractal_predictor: torch.nn.Module | None,
    condition_vae: HMCConditionVAE | None = None,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one epoch-level validation loss summary.

    Args:
        config: Full experiment configuration.
        dataloader: Validation dataloader.
        diffusion: Diffusion schedule helper.
        model: Main denoising model.
        extractor: HMC extractor.
        multifractal_predictor: Optional auxiliary predictor for `L_mf`.
        condition_vae: Optional frozen VAE used for inference-aligned validation.
        device: Active torch device.

    Returns:
        dict[str, float]: Validation metrics averaged over the validation set.
    """
    model.eval()
    if multifractal_predictor is not None:
        multifractal_predictor.eval()

    metric_total = {
        "val_loss": 0.0,
        "val_loss_diff": 0.0,
        "val_loss_mf": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if (
                config.train.validation_max_batches is not None
                and batch_index >= config.train.validation_max_batches
            ):
                break

            points = torch.as_tensor(batch["points"], dtype=torch.float32)
            labels = torch.as_tensor(batch["label"], device=device, dtype=torch.int64)
            hmc_points = select_hmc_condition_points(
                batch,
                config.data.hmc_point_source,
            )
            descriptors, sequences = prepare_hmc_batch(
                hmc_points,
                extractor,
                device,
            )
            if condition_vae is not None:
                descriptors, sequences, _ = apply_condition_vae_reconstruction(
                    descriptors,
                    sequences,
                    condition_vae,
                    1.0,
                    sequence_threshold=(config.train.condition_vae_sequence_threshold),
                    posterior_temperature=0.0,
                    deterministic=True,
                )
            inputs = points.transpose(1, 2).to(device)
            timesteps = torch.randint(
                0,
                diffusion.num_timesteps,
                (inputs.shape[0],),
                device=device,
            )
            noise = torch.randn_like(inputs)
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
            if config.train.min_snr_gamma is None:
                diff_loss = per_sample_diff_loss.mean()
            else:
                snr_weights = diffusion.min_snr_weights(
                    timesteps,
                    config.train.min_snr_gamma,
                ).to(per_sample_diff_loss.dtype)
                diff_loss = (per_sample_diff_loss * snr_weights).mean()
            total_loss = diff_loss
            loss_mf = torch.zeros((), device=device, dtype=torch.float32)
            if multifractal_predictor is not None and config.train.use_l_mf:
                predicted_xstart = diffusion.predict_xstart_from_eps(
                    x_t.float(),
                    timesteps,
                    predicted_noise.float(),
                )
                predicted_descriptor = multifractal_predictor(predicted_xstart)
                loss_mf = F.mse_loss(predicted_descriptor, descriptors.float())
                total_loss = total_loss + config.train.l_mf_weight * loss_mf

            metric_total["val_loss"] += float(total_loss.detach().item())
            metric_total["val_loss_diff"] += float(diff_loss.detach().item())
            metric_total["val_loss_mf"] += float(loss_mf.detach().item())
            batch_count += 1

    if batch_count <= 0:
        message = (
            "Validation dataloader produced zero batches. Check split, "
            "batch_size, and validation_max_batches."
        )
        LOGGER.error(message)
        raise TrainingRunError(message)

    return _average_epoch_metrics(metric_total, batch_count)


def run_training(config: ExperimentConfig) -> dict[str, float]:
    """Run the formal training loop.

    Args:
        config: Full experiment configuration.

    Returns:
        dict[str, float]: Final scalar metrics.
    """
    set_global_seed(config.train.seed)
    device = detect_device(config.train.prefer_cuda)
    dataloader = build_dataloader(config)
    if len(dataloader) == 0:
        message = (
            "Training dataloader is empty. Check batch_size, drop_last, "
            "and dataset size."
        )
        LOGGER.error(message)
        raise TrainingRunError(message)
    val_dataloader = None
    if config.train.validation_split is not None:
        val_dataloader = build_eval_dataloader(
            config,
            split=config.train.validation_split,
        )
        if len(val_dataloader) == 0:
            message = (
                "Validation dataloader is empty. Check validation split, "
                "batch_size, and dataset size."
            )
            LOGGER.error(message)
            raise TrainingRunError(message)

    extractor = HMCFeatureExtractor(config.hmc)
    condition_vae = None
    if config.train.condition_vae_path is not None:
        condition_vae, condition_vae_payload = load_condition_vae_checkpoint(
            config.train.condition_vae_path,
            device=device,
        )
        if condition_vae_payload.get("hmc_config") != _serialize_hmc_config(config.hmc):
            message = "Training condition VAE HMC config does not match experiment."
            LOGGER.error(message)
            raise TrainingRunError(message)
        condition_vae.requires_grad_(False)
        condition_vae.eval()
        LOGGER.info(
            "Using frozen condition VAE %s with reconstruction probability %.3f",
            config.train.condition_vae_path,
            config.train.condition_vae_reconstruction_prob,
        )
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
    metrics_path = output_dir / config.train.metrics_name
    evolution_path = output_dir / config.train.evolution_name
    latest_checkpoint_path = output_dir / config.train.latest_checkpoint_name
    best_train_checkpoint_path = output_dir / config.train.best_train_checkpoint_name
    best_val_checkpoint_path = output_dir / config.train.best_val_checkpoint_name

    start_epoch = 0
    global_step = 0
    last_metrics = {
        "loss": float("nan"),
        "loss_diff": float("nan"),
        "loss_mf": float("nan"),
        "grad_norm": float("nan"),
        "hmc_drop_fraction": float("nan"),
        "vae_reconstruction_fraction": float("nan"),
        "snr_weight": float("nan"),
    }
    best_train_loss = _load_checkpoint_loss(best_train_checkpoint_path, "train_loss")
    best_val_loss = _load_checkpoint_loss(best_val_checkpoint_path, "val_loss")

    resume_checkpoint = config.train.resume_checkpoint
    if (
        resume_checkpoint is None
        and config.train.auto_resume
        and latest_checkpoint_path.exists()
    ):
        resume_checkpoint = latest_checkpoint_path
    if resume_checkpoint is not None:
        resume_state = load_training_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            auxiliary_modules=None
            if multifractal_predictor is None
            else {"multifractal_predictor": multifractal_predictor},
            model_ema=model_ema,
        )
        # Resume at epoch granularity to avoid promising exact mid-epoch recovery
        # when sampler state is unavailable.
        start_epoch = resume_state.epoch + 1
        global_step = resume_state.step
        last_metrics = resume_state.metrics
        LOGGER.info(
            "Resumed training from %s at epoch=%d step=%d",
            resume_checkpoint,
            start_epoch,
            global_step,
        )

    if start_epoch >= config.train.epochs:
        message = (
            "Resume checkpoint already reached or exceeded train.epochs; "
            "increase train.epochs or remove train.resume_checkpoint."
        )
        LOGGER.error(message)
        raise TrainingRunError(message)
    if config.train.max_steps is not None and global_step >= config.train.max_steps:
        message = (
            "Resume checkpoint already reached or exceeded train.max_steps; "
            "increase train.max_steps or remove train.resume_checkpoint."
        )
        LOGGER.error(message)
        raise TrainingRunError(message)

    LOGGER.info("Starting formal training in %s", output_dir)
    LOGGER.info("Dataset categories: %s", ", ".join(config.data.categories))
    LOGGER.info("Dataset size: %d", len(dataloader.dataset))
    if val_dataloader is not None:
        LOGGER.info(
            "Validation split: %s (size=%d)",
            config.train.validation_split,
            len(val_dataloader.dataset),
        )
    if config.train.use_l_mf and config.train.use_amp and device.type == "cuda":
        LOGGER.info(
            "L_mf is enabled; autocast is disabled inside train steps "
            "for gradient stability."
        )

    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        completed_epoch = epoch
        epoch_metric_total = {
            "loss": 0.0,
            "loss_diff": 0.0,
            "loss_mf": 0.0,
            "grad_norm": 0.0,
            "hmc_drop_fraction": 0.0,
            "vae_reconstruction_fraction": 0.0,
            "snr_weight": 0.0,
        }
        epoch_batch_count = 0
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
                condition_vae=condition_vae,
                condition_vae_reconstruction_prob=(
                    config.train.condition_vae_reconstruction_prob
                ),
                condition_vae_sequence_threshold=(
                    config.train.condition_vae_sequence_threshold
                ),
                condition_vae_posterior_temperature=(
                    config.train.condition_vae_posterior_temperature
                ),
                min_snr_gamma=config.train.min_snr_gamma,
            )
            if optimizer_step and model_ema is not None:
                model_ema.update(model)
            for metric_name in epoch_metric_total:
                epoch_metric_total[metric_name] += last_metrics[metric_name]
            epoch_batch_count += 1
            if global_step % config.train.log_every == 0:
                payload: dict[str, float | int] = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": last_metrics["loss"],
                    "loss_diff": last_metrics["loss_diff"],
                    "loss_mf": last_metrics["loss_mf"],
                    "grad_norm": last_metrics["grad_norm"],
                    "hmc_drop_fraction": last_metrics["hmc_drop_fraction"],
                    "vae_reconstruction_fraction": last_metrics[
                        "vae_reconstruction_fraction"
                    ],
                    "snr_weight": last_metrics["snr_weight"],
                }
                append_metrics(metrics_path, payload)
                LOGGER.info(
                    "epoch=%d step=%d loss=%.6f "
                    "loss_diff=%.6f loss_mf=%.6f grad_norm=%.6f",
                    epoch + 1,
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

        train_epoch_metrics = _average_epoch_metrics(
            epoch_metric_total,
            epoch_batch_count,
        )
        save_every = config.train.resolved_save_every
        if save_every is not None and (epoch + 1) % save_every == 0:
            checkpoint_path = _save_epoch_checkpoint(
                output_dir=output_dir,
                epoch_index=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                metrics={
                    **train_epoch_metrics,
                    "train_loss": train_epoch_metrics["loss"],
                },
                auxiliary_state=None
                if multifractal_predictor is None
                else {"multifractal_predictor": multifractal_predictor.state_dict()},
                ema_model_state=None if model_ema is None else model_ema.state_dict(),
            )
            LOGGER.info("Saved epoch checkpoint to %s", checkpoint_path)
        else:
            checkpoint_path = None

        latest_path = None
        if (epoch + 1) % config.train.latest_every == 0:
            latest_path = _save_named_checkpoint(
                checkpoint_path=latest_checkpoint_path,
                epoch_index=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                metrics={
                    **train_epoch_metrics,
                    "train_loss": train_epoch_metrics["loss"],
                },
                auxiliary_state=None
                if multifractal_predictor is None
                else {"multifractal_predictor": multifractal_predictor.state_dict()},
                ema_model_state=None if model_ema is None else model_ema.state_dict(),
            )

        best_train_path = None
        if (
            config.train.save_best_train
            and train_epoch_metrics["loss"] <= best_train_loss
        ):
            best_train_loss = train_epoch_metrics["loss"]
            best_train_path = _save_named_checkpoint(
                checkpoint_path=best_train_checkpoint_path,
                epoch_index=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=global_step,
                metrics={
                    **train_epoch_metrics,
                    "train_loss": train_epoch_metrics["loss"],
                },
                auxiliary_state=None
                if multifractal_predictor is None
                else {"multifractal_predictor": multifractal_predictor.state_dict()},
                ema_model_state=None if model_ema is None else model_ema.state_dict(),
            )
            LOGGER.info("Saved best-train checkpoint to %s", best_train_path)

        val_epoch_metrics: dict[str, float] | None = None
        best_val_path = None
        if (
            val_dataloader is not None
            and (epoch + 1) % config.train.validation_every == 0
        ):
            val_epoch_metrics = evaluate_validation_loss(
                config,
                dataloader=val_dataloader,
                diffusion=diffusion,
                model=model,
                extractor=extractor,
                multifractal_predictor=multifractal_predictor,
                condition_vae=condition_vae,
                device=device,
            )
            LOGGER.info(
                "epoch=%d val_loss=%.6f val_loss_diff=%.6f val_loss_mf=%.6f",
                epoch + 1,
                val_epoch_metrics["val_loss"],
                val_epoch_metrics["val_loss_diff"],
                val_epoch_metrics["val_loss_mf"],
            )
            if val_epoch_metrics["val_loss"] <= best_val_loss:
                best_val_loss = val_epoch_metrics["val_loss"]
                best_val_path = _save_named_checkpoint(
                    checkpoint_path=best_val_checkpoint_path,
                    epoch_index=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=global_step,
                    metrics={
                        **train_epoch_metrics,
                        **val_epoch_metrics,
                        "train_loss": train_epoch_metrics["loss"],
                    },
                    auxiliary_state=None
                    if multifractal_predictor is None
                    else {
                        "multifractal_predictor": multifractal_predictor.state_dict()
                    },
                    ema_model_state=None
                    if model_ema is None
                    else model_ema.state_dict(),
                )
                LOGGER.info("Saved best-val checkpoint to %s", best_val_path)

        # Keep a dedicated epoch-level evolution log for overfitting checks and
        # trajectory review.
        append_evolution(
            evolution_path,
            {
                "epoch": epoch + 1,
                "step": global_step,
                "loss": train_epoch_metrics["loss"],
                "loss_diff": train_epoch_metrics["loss_diff"],
                "loss_mf": train_epoch_metrics["loss_mf"],
                "grad_norm": train_epoch_metrics["grad_norm"],
                "hmc_drop_fraction": train_epoch_metrics["hmc_drop_fraction"],
                "vae_reconstruction_fraction": train_epoch_metrics[
                    "vae_reconstruction_fraction"
                ],
                "snr_weight": train_epoch_metrics["snr_weight"],
                "checkpoint": None if checkpoint_path is None else checkpoint_path.name,
                "latest_checkpoint": None if latest_path is None else latest_path.name,
                "best_train_checkpoint": None
                if best_train_path is None
                else best_train_path.name,
                "best_val_checkpoint": None
                if best_val_path is None
                else best_val_path.name,
                "train_loss": train_epoch_metrics["loss"],
                "val_loss": None
                if val_epoch_metrics is None
                else val_epoch_metrics["val_loss"],
                "val_loss_diff": None
                if val_epoch_metrics is None
                else val_epoch_metrics["val_loss_diff"],
                "val_loss_mf": None
                if val_epoch_metrics is None
                else val_epoch_metrics["val_loss_mf"],
                "record_type": "epoch",
            },
        )
        last_metrics = train_epoch_metrics

        if config.train.max_steps is not None and global_step >= config.train.max_steps:
            break

    final_checkpoint_path = output_dir / config.train.checkpoint_name
    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=global_step,
        epoch=completed_epoch,
        metrics={
            **last_metrics,
            "train_loss": last_metrics["loss"],
        },
        auxiliary_state=None
        if multifractal_predictor is None
        else {"multifractal_predictor": multifractal_predictor.state_dict()},
        ema_model_state=None if model_ema is None else model_ema.state_dict(),
    )
    LOGGER.info("Saved final checkpoint to %s", final_checkpoint_path)
    append_evolution(
        evolution_path,
        {
            "epoch": completed_epoch + 1,
            "step": global_step,
            "loss": last_metrics["loss"],
            "loss_diff": last_metrics["loss_diff"],
            "loss_mf": last_metrics["loss_mf"],
            "grad_norm": last_metrics["grad_norm"],
            "hmc_drop_fraction": last_metrics["hmc_drop_fraction"],
            "vae_reconstruction_fraction": last_metrics["vae_reconstruction_fraction"],
            "snr_weight": last_metrics["snr_weight"],
            "checkpoint": final_checkpoint_path.name,
            "record_type": "final",
        },
    )
    return {
        **last_metrics,
        "epoch": float(completed_epoch + 1),
        "steps": float(global_step),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the formal trainer."""
    parser = argparse.ArgumentParser(description="Run HMC-DiT3D formal training.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config path for the formal experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for formal training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    try:
        config = load_experiment_config(args.config)
        metrics = run_training(config)
    except SmokeTrainingError as error:
        raise TrainingRunError(str(error)) from error
    LOGGER.info(
        "Formal training finished: epoch=%d loss=%.6f grad_norm=%.6f steps=%d",
        int(metrics["epoch"]),
        metrics["loss"],
        metrics["grad_norm"],
        int(metrics["steps"]),
    )


if __name__ == "__main__":
    main()
