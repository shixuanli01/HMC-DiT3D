"""Configuration loading for HMC-DiT3D training experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hmc_dit3d.hmc.config import HMCConfig, HMCEncoderConfig
from hmc_dit3d.models.hmc_dit import HMCConditionedDiTConfig
from hmc_dit3d.train.diffusion import DiffusionConfig

LOGGER = logging.getLogger(__name__)


class TrainingConfigError(RuntimeError):
    """Raised when a training configuration is invalid."""


@dataclass(slots=True, frozen=True)
class DataConfig:
    """Dataset and dataloader settings.

    Args:
        root_dir: ShapeNetCore.v2.PC15k root directory.
        categories: Requested ShapeNet categories.
        split: Dataset split.
        sample_size: Number of points sampled from each shape.
        batch_size: Dataloader batch size.
        num_workers: Dataloader worker count.
        pin_memory: Whether to pin dataloader memory.
        random_subsample: Whether to randomly subsample points each access.
        drop_last: Whether the dataloader should drop the final short batch.
        hmc_point_source: Point set used to extract HMC conditions. ``sampled``
            reuses the denoising target; ``full`` uses all 15k source points.
    """

    root_dir: Path
    categories: tuple[str, ...]
    split: str = "train"
    sample_size: int = 2048
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    random_subsample: bool = True
    drop_last: bool = True
    hmc_point_source: str = "sampled"

    def __post_init__(self) -> None:
        """Validate dataset configuration."""
        object.__setattr__(
            self,
            "categories",
            tuple(str(category) for category in self.categories),
        )
        if len(self.categories) == 0:
            message = "data.categories must contain at least one category."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.sample_size <= 0:
            message = "data.sample_size must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.batch_size <= 0:
            message = "data.batch_size must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.num_workers < 0:
            message = "data.num_workers must be greater than or equal to zero."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.hmc_point_source not in {"sampled", "full"}:
            message = "data.hmc_point_source must be one of {'sampled', 'full'}."
            LOGGER.error(message)
            raise TrainingConfigError(message)


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Model hyperparameters for the smoke training path.

    Args:
        model_variant: Backbone fusion variant. `cross_attention` maps to variant 3.1,
            while `bottleneck` maps to variant 3.2.
        voxel_size: Cubic voxel resolution.
        patch_size: Cubic patch size.
        in_channels: Input point/voxel channels.
        out_channels: Output point/voxel channels. Defaults to `in_channels`.
        input_dim: Patch token dimension before the DiT backbone projection.
        model_dim: Hidden token dimension.
        depth: Number of DiT-style blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio in MLP blocks.
        class_dropout_prob: CFG label dropout probability.
        num_bottleneck_latents: Number of learned bottleneck latents for variant 3.2.
        resampler_depth: Number of down/up resampler layers for variant 3.2.
        resampler_mlp_ratio: Optional MLP expansion used inside the resampler.
    """

    model_variant: str = "cross_attention"
    voxel_size: int = 8
    patch_size: int = 2
    in_channels: int = 3
    out_channels: int | None = None
    input_dim: int = 16
    model_dim: int = 32
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_bottleneck_latents: int = 32
    resampler_depth: int = 2
    resampler_mlp_ratio: float | None = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        valid_variants = {"cross_attention", "bottleneck"}
        if self.model_variant not in valid_variants:
            message = (
                "model.model_variant must be one of "
                f"{sorted(valid_variants)!r}, got {self.model_variant!r}."
            )
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.voxel_size <= 0:
            message = "model.voxel_size must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.patch_size <= 0:
            message = "model.patch_size must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.voxel_size % self.patch_size != 0:
            message = "model.voxel_size must be divisible by model.patch_size."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.in_channels <= 0:
            message = "model.in_channels must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.out_channels is not None and self.out_channels <= 0:
            message = "model.out_channels must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.input_dim <= 0:
            message = "model.input_dim must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.model_dim <= 0:
            message = "model.model_dim must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.depth <= 0:
            message = "model.depth must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.num_heads <= 0:
            message = "model.num_heads must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.model_dim % self.num_heads != 0:
            message = "model.model_dim must be divisible by model.num_heads."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.mlp_ratio <= 0.0:
            message = "model.mlp_ratio must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if not 0.0 <= self.class_dropout_prob < 1.0:
            message = "model.class_dropout_prob must be in [0, 1)."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.num_bottleneck_latents <= 0:
            message = "model.num_bottleneck_latents must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.resampler_depth <= 0:
            message = "model.resampler_depth must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.resampler_mlp_ratio is not None and self.resampler_mlp_ratio <= 0.0:
            message = "model.resampler_mlp_ratio must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)

    @property
    def resolved_out_channels(self) -> int:
        """Return the effective output channel count."""
        return self.in_channels if self.out_channels is None else self.out_channels

    def build_backbone_config(self, num_classes: int) -> HMCConditionedDiTConfig:
        """Build a validated backbone config from model-level settings.

        Args:
            num_classes: Number of experiment categories.

        Returns:
            HMCConditionedDiTConfig: Backbone configuration.
        """
        patch_grid_size = self.voxel_size // self.patch_size
        return HMCConditionedDiTConfig(
            input_dim=self.input_dim,
            model_dim=self.model_dim,
            condition_dim=self.model_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            num_classes=num_classes,
            class_dropout_prob=self.class_dropout_prob,
            max_tokens=patch_grid_size**3,
            out_dim=(self.patch_size**3) * self.resolved_out_channels,
        )


@dataclass(slots=True, frozen=True)
class TrainConfig:
    """Training loop settings for smoke and formal runners.

    Args:
        seed: Global random seed.
        results_dir: Base directory for logs and checkpoints.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Number of dataset passes.
        max_steps: Optional step limit across all epochs.
        log_every: Logging interval in steps.
        checkpoint_name:
            Final checkpoint filename saved under
            `results_dir/experiment_name`.
        latest_checkpoint_name: Rolling latest checkpoint filename.
        latest_every: Epoch interval for refreshing the rolling checkpoint.
        best_train_checkpoint_name: Best-train checkpoint filename.
        save_best_train: Whether to save checkpoints selected by training loss.
        best_val_checkpoint_name: Best-validation checkpoint filename.
        resume_checkpoint: Optional checkpoint path used to resume training.
        auto_resume: Whether to resume from ``latest_checkpoint_name`` when it
            already exists in the experiment output directory.
        checkpoint_every: Optional epoch interval for intermediate checkpoints.
        save_every: Preferred epoch interval for intermediate checkpoints.
        metrics_name: JSONL metrics filename saved under `results_dir/experiment_name`.
        evolution_name: Epoch-level evolution JSONL filename saved under
            `results_dir/experiment_name`.
        validation_split: Optional validation split used to track `best_val`.
        validation_every: Validation interval measured in epochs.
        validation_max_batches: Optional cap for validation batches per epoch.
        use_l_mf: Whether to enable the auxiliary multifractal descriptor loss.
        l_mf_weight: Weight applied to the auxiliary descriptor loss.
        l_mf_hidden_dim: Hidden width of the auxiliary descriptor predictor.
        grad_clip: Optional gradient clipping threshold.
        grad_accum_steps: Number of micro-batches accumulated before one
            optimizer update. Use `1` to disable gradient accumulation.
        use_amp: Whether to enable CUDA autocast.
        prefer_cuda: Whether to prefer CUDA devices when available.
        min_snr_gamma: Optional Min-SNR gamma for epsilon-prediction weighting.
        hmc_dropout_prob: Per-sample probability of replacing HMC with zeros.
        condition_vae_path: Optional frozen condition-VAE checkpoint used to
            reconstruct HMC conditions during DiT training.
        condition_vae_reconstruction_prob: Per-sample probability of replacing
            an exact HMC condition with its VAE reconstruction.
        condition_vae_sequence_threshold: Relative sparsification threshold for
            VAE-reconstructed measure sequences.
        condition_vae_posterior_temperature: Posterior standard-deviation
            multiplier used for stochastic VAE reconstructions.
        use_ema: Whether to maintain an exponential moving average of the model.
        ema_decay: EMA decay applied after each optimizer update.
    """

    seed: int = 3407
    results_dir: Path = Path("results")
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    epochs: int = 1
    max_steps: int | None = 2
    log_every: int = 1
    checkpoint_name: str = "smoke_checkpoint.pt"
    latest_checkpoint_name: str = "latest.pt"
    latest_every: int = 1
    best_train_checkpoint_name: str = "best_train.pt"
    save_best_train: bool = True
    best_val_checkpoint_name: str = "best_val.pt"
    resume_checkpoint: Path | None = None
    auto_resume: bool = False
    checkpoint_every: int | None = None
    save_every: int | None = None
    metrics_name: str = "train_metrics.jsonl"
    evolution_name: str = "evolution.jsonl"
    validation_split: str | None = None
    validation_every: int = 1
    validation_max_batches: int | None = None
    use_l_mf: bool = False
    l_mf_weight: float = 0.0
    l_mf_hidden_dim: int = 128
    grad_clip: float | None = None
    grad_accum_steps: int = 1
    use_amp: bool = True
    prefer_cuda: bool = True
    min_snr_gamma: float | None = None
    hmc_dropout_prob: float = 0.0
    condition_vae_path: Path | None = None
    condition_vae_reconstruction_prob: float = 0.0
    condition_vae_sequence_threshold: float = 0.0
    condition_vae_posterior_temperature: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.9999

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.learning_rate <= 0.0:
            message = "train.learning_rate must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.weight_decay < 0.0:
            message = "train.weight_decay must be greater than or equal to zero."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.epochs <= 0:
            message = "train.epochs must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.max_steps is not None and self.max_steps <= 0:
            message = "train.max_steps must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.log_every <= 0:
            message = "train.log_every must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.checkpoint_name.strip()) == 0:
            message = "train.checkpoint_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.latest_checkpoint_name.strip()) == 0:
            message = "train.latest_checkpoint_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.latest_every <= 0:
            message = "train.latest_every must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.best_train_checkpoint_name.strip()) == 0:
            message = "train.best_train_checkpoint_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.best_val_checkpoint_name.strip()) == 0:
            message = "train.best_val_checkpoint_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            message = "train.checkpoint_every must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.save_every is not None and self.save_every <= 0:
            message = "train.save_every must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if (
            self.checkpoint_every is not None
            and self.save_every is not None
            and self.checkpoint_every != self.save_every
        ):
            message = (
                "train.checkpoint_every and train.save_every must match "
                "when both are provided."
            )
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.metrics_name.strip()) == 0:
            message = "train.metrics_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if len(self.evolution_name.strip()) == 0:
            message = "train.evolution_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.l_mf_weight < 0.0:
            message = "train.l_mf_weight must be greater than or equal to zero."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.use_l_mf and self.l_mf_weight <= 0.0:
            message = "train.use_l_mf=True requires train.l_mf_weight > 0."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.l_mf_hidden_dim <= 0:
            message = "train.l_mf_hidden_dim must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            message = "train.grad_clip must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.grad_accum_steps <= 0:
            message = "train.grad_accum_steps must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.validation_split is not None and self.validation_split not in {
            "train",
            "val",
            "test",
        }:
            message = (
                "train.validation_split must be one of {'train', 'val', 'test'} "
                "when provided."
            )
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.validation_every <= 0:
            message = "train.validation_every must be positive."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.validation_max_batches is not None and self.validation_max_batches <= 0:
            message = "train.validation_max_batches must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.min_snr_gamma is not None and self.min_snr_gamma <= 0.0:
            message = "train.min_snr_gamma must be positive when provided."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if not 0.0 <= self.hmc_dropout_prob < 1.0:
            message = "train.hmc_dropout_prob must be in [0, 1)."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if not 0.0 <= self.condition_vae_reconstruction_prob <= 1.0:
            message = "train.condition_vae_reconstruction_prob must be in [0, 1]."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if (
            self.condition_vae_reconstruction_prob > 0.0
            and self.condition_vae_path is None
        ):
            message = (
                "train.condition_vae_reconstruction_prob > 0 requires "
                "train.condition_vae_path."
            )
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.condition_vae_sequence_threshold < 0.0:
            message = "train.condition_vae_sequence_threshold must be non-negative."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if self.condition_vae_posterior_temperature < 0.0:
            message = "train.condition_vae_posterior_temperature must be non-negative."
            LOGGER.error(message)
            raise TrainingConfigError(message)
        if not 0.0 < self.ema_decay < 1.0:
            message = "train.ema_decay must be in (0, 1)."
            LOGGER.error(message)
            raise TrainingConfigError(message)

    @property
    def resolved_save_every(self) -> int | None:
        """Return the effective periodic checkpoint interval."""
        return self.save_every if self.save_every is not None else self.checkpoint_every


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    """Top-level smoke experiment configuration."""

    experiment_name: str
    hmc: HMCConfig
    encoder: HMCEncoderConfig
    diffusion: DiffusionConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        """Validate the top-level experiment configuration."""
        if len(self.experiment_name.strip()) == 0:
            message = "experiment_name must not be empty."
            LOGGER.error(message)
            raise TrainingConfigError(message)


def _resolve_path(raw_path: str | Path, config_dir: Path) -> Path:
    """Resolve a config path relative to the config file directory."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_dir / path).resolve()


def _require_section(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Extract one required mapping section from raw YAML config."""
    section = raw_config.get(section_name)
    if not isinstance(section, dict):
        message = f"Config section {section_name!r} must be a mapping."
        LOGGER.error(message)
        raise TrainingConfigError(message)
    return section


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML.

    Args:
        config_path: YAML configuration path.

    Returns:
        ExperimentConfig: Validated experiment configuration.
    """
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.exists():
        message = f"Config file does not exist: {resolved_path!s}."
        LOGGER.error(message)
        raise TrainingConfigError(message)
    with resolved_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    if not isinstance(raw_config, dict):
        message = "Top-level config must be a mapping."
        LOGGER.error(message)
        raise TrainingConfigError(message)

    config_dir = resolved_path.parent
    data_section = dict(_require_section(raw_config, "data"))
    model_section = dict(_require_section(raw_config, "model"))
    train_section = dict(_require_section(raw_config, "train"))
    hmc_section = dict(_require_section(raw_config, "hmc"))
    encoder_section = dict(_require_section(raw_config, "encoder"))
    diffusion_section = dict(_require_section(raw_config, "diffusion"))

    data_section["root_dir"] = _resolve_path(data_section["root_dir"], config_dir)
    train_section["results_dir"] = _resolve_path(
        train_section["results_dir"], config_dir
    )
    if train_section.get("resume_checkpoint") is not None:
        train_section["resume_checkpoint"] = _resolve_path(
            train_section["resume_checkpoint"], config_dir
        )
    if train_section.get("condition_vae_path") is not None:
        train_section["condition_vae_path"] = _resolve_path(
            train_section["condition_vae_path"], config_dir
        )

    experiment_name = raw_config.get("experiment_name")
    if not isinstance(experiment_name, str):
        message = "experiment_name must be a string."
        LOGGER.error(message)
        raise TrainingConfigError(message)

    return ExperimentConfig(
        experiment_name=experiment_name,
        hmc=HMCConfig(**hmc_section),
        encoder=HMCEncoderConfig(**encoder_section),
        diffusion=DiffusionConfig(**diffusion_section),
        data=DataConfig(**data_section),
        model=ModelConfig(**model_section),
        train=TrainConfig(**train_section),
    )
