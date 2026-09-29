"""Training utilities for HMC-DiT3D."""

from __future__ import annotations

from hmc_dit3d.train.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    TrainingConfigError,
    load_experiment_config,
)
from hmc_dit3d.train.diffusion import (
    DiffusionConfig,
    DiffusionConfigError,
    GaussianDiffusion,
    get_beta_schedule,
)

__all__ = [
    "DataConfig",
    "DiffusionConfig",
    "DiffusionConfigError",
    "EvaluationError",
    "ExperimentConfig",
    "GaussianDiffusion",
    "ModelConfig",
    "SamplingError",
    "TrainingCheckpointState",
    "TrainConfig",
    "TrainingConfigError",
    "TrainingRunError",
    "evaluate_saved_samples",
    "generate_bank_conditioned_samples",
    "generate_reference_conditioned_samples",
    "get_beta_schedule",
    "load_training_checkpoint",
    "load_sampling_checkpoint",
    "load_experiment_config",
    "run_training",
    "save_sample_payload",
]


def __getattr__(name: str) -> object:
    """Lazily expose sampling and evaluation helpers without eager submodule import."""
    if name in {"EvaluationError", "evaluate_saved_samples"}:
        from hmc_dit3d.train.evaluate import EvaluationError, evaluate_saved_samples

        return {
            "EvaluationError": EvaluationError,
            "evaluate_saved_samples": evaluate_saved_samples,
        }[name]
    if name in {
        "SamplingError",
        "generate_bank_conditioned_samples",
        "generate_reference_conditioned_samples",
        "load_sampling_checkpoint",
        "save_sample_payload",
    }:
        from hmc_dit3d.train.sample import (
            SamplingError,
            generate_bank_conditioned_samples,
            generate_reference_conditioned_samples,
            load_sampling_checkpoint,
            save_sample_payload,
        )

        return {
            "SamplingError": SamplingError,
            "generate_bank_conditioned_samples": generate_bank_conditioned_samples,
            "generate_reference_conditioned_samples": (
                generate_reference_conditioned_samples
            ),
            "load_sampling_checkpoint": load_sampling_checkpoint,
            "save_sample_payload": save_sample_payload,
        }[name]
    if name in {
        "TrainingCheckpointState",
        "TrainingRunError",
        "load_training_checkpoint",
        "run_training",
    }:
        from hmc_dit3d.train.smoke import (
            TrainingCheckpointState,
            load_training_checkpoint,
        )
        from hmc_dit3d.train.train import TrainingRunError, run_training

        return {
            "TrainingCheckpointState": TrainingCheckpointState,
            "TrainingRunError": TrainingRunError,
            "load_training_checkpoint": load_training_checkpoint,
            "run_training": run_training,
        }[name]
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
