"""HMC-DiT3D package."""

from __future__ import annotations

from hmc_dit3d.data.hmc_condition_bank import (
    HMCConditionBank,
    HMCConditionBankError,
    build_hmc_condition_bank,
    load_hmc_condition_bank,
    sample_hmc_condition_bank,
    save_hmc_condition_bank,
)
from hmc_dit3d.data.shapenet_pc15k import (
    CATEGORY_TO_SYNSETID,
    SYNSETID_TO_CATEGORY,
    ShapeNetDataError,
    ShapeNetPC15KDataset,
)
from hmc_dit3d.hmc.config import (
    HMCComputationError,
    HMCConfig,
    HMCConfigurationError,
    HMCEncoderConfig,
    HMCError,
    NormalizationMode,
)
from hmc_dit3d.hmc.encoder import HMCConditionEncoder, HMCConditionOutput
from hmc_dit3d.hmc.extractor import HMCFeatureExtractor, HMCResult, HMCScaleSummary
from hmc_dit3d.metrics.pointcloud import (
    PointCloudMetricError,
    chamfer_distance,
    compute_minimal_pointcloud_metrics,
    jensen_shannon_divergence,
    jsd_between_point_cloud_sets,
    pairwise_chamfer_distance_matrix,
)
from hmc_dit3d.models.hmc_dit import (
    HMCConditionedBottleneckDiT,
    HMCConditionedBottleneckDiTBackbone,
    HMCConditionedBottleneckDiTPointCloud,
    HMCConditionedBottleneckDiTVoxel,
    HMCConditionedDiT,
    HMCConditionedDiTBackbone,
    HMCConditionedDiTBlock,
    HMCConditionedDiTConfig,
    HMCConditionedDiTPointCloud,
    HMCConditionedDiTVoxel,
    HMCModelConfigurationError,
    HMCModelInputError,
    PointCloudVoxelizer,
    trilinear_devoxelize,
)
from hmc_dit3d.models.multifractal_predictor import (
    MultifractalDescriptorPredictor,
    MultifractalPredictorError,
)
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
    "CATEGORY_TO_SYNSETID",
    "DataConfig",
    "DiffusionConfig",
    "DiffusionConfigError",
    "ExperimentConfig",
    "GaussianDiffusion",
    "HMCComputationError",
    "HMCConditionBank",
    "HMCConditionBankError",
    "HMCConditionEncoder",
    "HMCConditionOutput",
    "HMCConditionedDiT",
    "HMCConditionedBottleneckDiT",
    "HMCConditionedBottleneckDiTBackbone",
    "HMCConditionedBottleneckDiTPointCloud",
    "HMCConditionedBottleneckDiTVoxel",
    "HMCConditionedDiTBackbone",
    "HMCConditionedDiTBlock",
    "HMCConditionedDiTConfig",
    "HMCConditionedDiTPointCloud",
    "HMCConditionedDiTVoxel",
    "HMCConfig",
    "HMCConfigurationError",
    "HMCEncoderConfig",
    "HMCError",
    "HMCFeatureExtractor",
    "HMCModelConfigurationError",
    "HMCModelInputError",
    "ModelConfig",
    "MultifractalDescriptorPredictor",
    "MultifractalPredictorError",
    "PointCloudVoxelizer",
    "HMCResult",
    "HMCScaleSummary",
    "NormalizationMode",
    "PointCloudMetricError",
    "SYNSETID_TO_CATEGORY",
    "SamplingError",
    "ShapeNetDataError",
    "ShapeNetPC15KDataset",
    "TrainingCheckpointState",
    "TrainConfig",
    "TrainingConfigError",
    "TrainingRunError",
    "build_hmc_condition_bank",
    "EvaluationError",
    "chamfer_distance",
    "compute_minimal_pointcloud_metrics",
    "evaluate_saved_samples",
    "generate_bank_conditioned_samples",
    "generate_reference_conditioned_samples",
    "get_beta_schedule",
    "jensen_shannon_divergence",
    "jsd_between_point_cloud_sets",
    "load_sampling_checkpoint",
    "load_training_checkpoint",
    "load_experiment_config",
    "load_hmc_condition_bank",
    "pairwise_chamfer_distance_matrix",
    "run_training",
    "sample_hmc_condition_bank",
    "save_hmc_condition_bank",
    "save_sample_payload",
    "trilinear_devoxelize",
]


def __getattr__(name: str) -> object:
    """Lazily expose runtime-heavy helpers to avoid CLI import side effects."""
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
        from hmc_dit3d.train import (
            TrainingCheckpointState,
            TrainingRunError,
            load_training_checkpoint,
            run_training,
        )

        return {
            "TrainingCheckpointState": TrainingCheckpointState,
            "TrainingRunError": TrainingRunError,
            "load_training_checkpoint": load_training_checkpoint,
            "run_training": run_training,
        }[name]
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
