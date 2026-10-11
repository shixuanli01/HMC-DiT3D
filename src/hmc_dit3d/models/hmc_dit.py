"""HMC-conditioned DiT-style blocks and a minimal backbone."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hmc_dit3d.hmc.config import HMCConfig, HMCEncoderConfig
from hmc_dit3d.hmc.encoder import HMCConditionEncoder, HMCConditionOutput

LOGGER = logging.getLogger(__name__)


class HMCModelError(RuntimeError):
    """Base exception for HMC-conditioned model failures."""


class HMCModelConfigurationError(HMCModelError):
    """Raised when model configuration is invalid."""


class HMCModelInputError(HMCModelError):
    """Raised when model inputs do not match the expected contract."""


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN-style affine modulation.

    Args:
        x: Token tensor with shape `(B, N, D)`.
        shift: Per-sample shift tensor with shape `(B, D)`.
        scale: Per-sample scale tensor with shape `(B, D)`.

    Returns:
        Tensor: Modulated token tensor with shape `(B, N, D)`.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_1d_sincos_pos_embed(embed_dim: int, positions: Tensor) -> Tensor:
    """Build 1D sinusoidal embeddings for a vector of positions.

    Args:
        embed_dim: Target embedding dimension.
        positions: Position tensor with shape `(N,)`.

    Returns:
        Tensor: Sinusoidal embeddings with shape `(N, embed_dim)`.
    """
    if embed_dim <= 0:
        return torch.zeros(positions.shape[0], 0, dtype=torch.float32)

    positions = positions.to(dtype=torch.float32).reshape(-1, 1)
    half_dim = embed_dim // 2
    if half_dim == 0:
        return torch.zeros(positions.shape[0], embed_dim, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    )
    angles = positions * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding.shape[1] < embed_dim:
        embedding = torch.cat(
            [
                embedding,
                torch.zeros(
                    positions.shape[0],
                    embed_dim - embedding.shape[1],
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
    return embedding


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: int) -> Tensor:
    """Build fixed 3D sine-cosine position embeddings.

    Args:
        embed_dim: Target token dimension.
        grid_size: Number of patches per axis.

    Returns:
        Tensor: Position embeddings with shape `(grid_size**3, embed_dim)`.
    """
    if embed_dim <= 0:
        message = "embed_dim must be positive for 3D position embeddings."
        LOGGER.error(message)
        raise HMCModelConfigurationError(message)
    if grid_size <= 0:
        message = "grid_size must be positive for 3D position embeddings."
        LOGGER.error(message)
        raise HMCModelConfigurationError(message)

    dim_x = embed_dim // 3
    dim_y = embed_dim // 3
    dim_z = embed_dim - dim_x - dim_y

    grid_positions = torch.arange(grid_size, dtype=torch.float32)
    grid_x, grid_y, grid_z = torch.meshgrid(
        grid_positions,
        grid_positions,
        grid_positions,
        indexing="ij",
    )
    emb_x = _get_1d_sincos_pos_embed(dim_x, grid_x.reshape(-1))
    emb_y = _get_1d_sincos_pos_embed(dim_y, grid_y.reshape(-1))
    emb_z = _get_1d_sincos_pos_embed(dim_z, grid_z.reshape(-1))
    return torch.cat([emb_x, emb_y, emb_z], dim=1)


class PatchEmbedVoxel(nn.Module):
    """Voxel-grid to patch-token embedding.

    This mirrors the patchification logic used by DiT-3D, but stays within the
    new project and only handles dense voxel tensors.
    """

    def __init__(
        self,
        voxel_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        """Initialize the 3D patch embedding module.

        Args:
            voxel_size: Side length of the cubic voxel grid.
            patch_size: Side length of each cubic patch.
            in_channels: Number of voxel feature channels.
            embed_dim: Patch token dimension.
        """
        super().__init__()
        if voxel_size <= 0:
            message = "voxel_size must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if patch_size <= 0:
            message = "patch_size must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if voxel_size % patch_size != 0:
            message = "voxel_size must be divisible by patch_size."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if in_channels <= 0:
            message = "in_channels must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if embed_dim <= 0:
            message = "embed_dim must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_grid_size = voxel_size // patch_size
        self.num_patches = self.patch_grid_size**3
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize the patch embedding like a linear projection."""
        weight = self.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a cubic voxel grid into patch tokens.

        Args:
            x: Voxel tensor with shape `(B, C, X, Y, Z)`.

        Returns:
            Tensor: Patch tokens with shape `(B, num_patches, embed_dim)`.
        """
        if x.ndim != 5:
            message = f"Voxel input must have rank 5, got shape {tuple(x.shape)!r}."
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if x.shape[1] != self.in_channels:
            message = (
                f"Voxel input channel count must be {self.in_channels}, "
                f"got {x.shape[1]}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        spatial_shape = x.shape[2:]
        expected_shape = (self.voxel_size, self.voxel_size, self.voxel_size)
        if spatial_shape != expected_shape:
            message = (
                "Voxel input must match the configured cubic resolution. "
                f"Expected {expected_shape}, got {spatial_shape!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        return self.proj(x.float()).flatten(2).transpose(1, 2)


class PointCloudVoxelizer(nn.Module):
    """Pure PyTorch average voxelization for point clouds."""

    def __init__(
        self, resolution: int, normalize: bool = True, eps: float = 0.0
    ) -> None:
        """Initialize the point-cloud voxelizer.

        Args:
            resolution: Side length of the cubic voxel grid.
            normalize: Whether to normalize coordinates into `[0, 1]`.
            eps: Numerical stabilizer for coordinate normalization.
        """
        super().__init__()
        if resolution <= 0:
            message = "resolution must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if eps < 0.0:
            message = "eps must be greater than or equal to zero."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.resolution = resolution
        self.normalize = normalize
        self.eps = eps

    def forward(self, features: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Voxelize point features by average pooling into a dense grid.

        Args:
            features: Point features with shape `(B, C, N)`.
            coords: Point coordinates with shape `(B, 3, N)`.

        Returns:
            tuple[Tensor, Tensor]:
                Dense voxel tensor `(B, C, R, R, R)` and normalized coordinates
                in voxel space `(B, 3, N)`.
        """
        if features.ndim != 3:
            message = (
                "Point features must have shape (B, C, N), "
                f"got {tuple(features.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if coords.ndim != 3 or coords.shape[1] != 3:
            message = (
                "Point coordinates must have shape (B, 3, N), "
                f"got {tuple(coords.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if features.shape[0] != coords.shape[0] or features.shape[2] != coords.shape[2]:
            message = (
                "Point features and coordinates must share batch size and point count. "
                f"Got features {tuple(features.shape)!r} and "
                f"coords {tuple(coords.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)

        device = features.device
        dtype = features.dtype
        coords = torch.as_tensor(coords, device=device, dtype=dtype).detach()
        norm_coords = coords - coords.mean(dim=2, keepdim=True)
        if self.normalize:
            denom = (
                norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values
            )
            denom = torch.clamp(
                denom * 2.0 + self.eps,
                min=torch.finfo(dtype).eps,
            )
            norm_coords = norm_coords / denom + 0.5
        else:
            norm_coords = (norm_coords + 1.0) / 2.0
        # Use continuous mapping with (R - 1) so [0, 1] maps exactly to
        # [0, R - 1], avoiding early saturation near 1.0.
        max_coordinate = float(self.resolution - 1)
        norm_coords = torch.clamp(
            norm_coords * max_coordinate,
            min=0.0,
            max=max_coordinate,
        )
        voxel_coords = torch.round(norm_coords).to(torch.int64)
        batch_size, channels, point_count = features.shape
        resolution = self.resolution
        flat_indices = (
            voxel_coords[:, 0] * resolution * resolution
            + voxel_coords[:, 1] * resolution
            + voxel_coords[:, 2]
        )
        output = torch.zeros(
            batch_size,
            channels,
            resolution**3,
            device=device,
            dtype=dtype,
        )
        expanded_indices = flat_indices.unsqueeze(1).expand(-1, channels, -1)
        output.scatter_add_(2, expanded_indices, features)
        counts = torch.zeros(
            batch_size,
            1,
            resolution**3,
            device=device,
            dtype=dtype,
        )
        counts.scatter_add_(
            2,
            flat_indices.unsqueeze(1),
            torch.ones(batch_size, 1, point_count, device=device, dtype=dtype),
        )
        output = output / counts.clamp_min(1.0)
        return output.view(
            batch_size,
            channels,
            resolution,
            resolution,
            resolution,
        ), norm_coords


def trilinear_devoxelize(
    voxel_features: Tensor,
    coords: Tensor,
    resolution: int,
) -> Tensor:
    """Sample dense voxel features back onto point coordinates.

    Args:
        voxel_features: Dense voxel tensor with shape `(B, C, R, R, R)`.
        coords: Point coordinates in voxel space with shape `(B, 3, N)`.
        resolution: Side length of the cubic voxel grid.

    Returns:
        Tensor: Point features with shape `(B, C, N)`.
    """
    if voxel_features.ndim != 5:
        message = (
            "voxel_features must have shape (B, C, R, R, R), "
            f"got {tuple(voxel_features.shape)!r}."
        )
        LOGGER.error(message)
        raise HMCModelInputError(message)
    if coords.ndim != 3 or coords.shape[1] != 3:
        message = f"coords must have shape (B, 3, N), got {tuple(coords.shape)!r}."
        LOGGER.error(message)
        raise HMCModelInputError(message)
    if voxel_features.shape[0] != coords.shape[0]:
        message = (
            "voxel_features and coords must share batch size. "
            f"Got {voxel_features.shape[0]} and {coords.shape[0]}."
        )
        LOGGER.error(message)
        raise HMCModelInputError(message)
    spatial_shape = voxel_features.shape[2:]
    expected_shape = (resolution, resolution, resolution)
    if spatial_shape != expected_shape:
        message = (
            "voxel_features spatial shape must match resolution. "
            f"Expected {expected_shape}, got {spatial_shape!r}."
        )
        LOGGER.error(message)
        raise HMCModelInputError(message)

    device = voxel_features.device
    dtype = voxel_features.dtype
    coords = torch.as_tensor(coords, device=device, dtype=dtype)
    coords = torch.clamp(coords, min=0.0, max=float(resolution - 1))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    z0 = torch.floor(z).to(torch.int64)
    x1 = torch.clamp(x0 + 1, max=resolution - 1)
    y1 = torch.clamp(y0 + 1, max=resolution - 1)
    z1 = torch.clamp(z0 + 1, max=resolution - 1)
    wx1 = x - x0.to(dtype)
    wy1 = y - y0.to(dtype)
    wz1 = z - z0.to(dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    flattened = voxel_features.reshape(
        voxel_features.shape[0],
        voxel_features.shape[1],
        -1,
    )

    def _gather(ix: Tensor, iy: Tensor, iz: Tensor) -> Tensor:
        indices = ix * resolution * resolution + iy * resolution + iz
        expanded = indices.unsqueeze(1).expand(-1, flattened.shape[1], -1)
        return torch.gather(flattened, 2, expanded)

    c000 = _gather(x0, y0, z0)
    c001 = _gather(x0, y0, z1)
    c010 = _gather(x0, y1, z0)
    c011 = _gather(x0, y1, z1)
    c100 = _gather(x1, y0, z0)
    c101 = _gather(x1, y0, z1)
    c110 = _gather(x1, y1, z0)
    c111 = _gather(x1, y1, z1)

    w000 = (wx0 * wy0 * wz0).unsqueeze(1)
    w001 = (wx0 * wy0 * wz1).unsqueeze(1)
    w010 = (wx0 * wy1 * wz0).unsqueeze(1)
    w011 = (wx0 * wy1 * wz1).unsqueeze(1)
    w100 = (wx1 * wy0 * wz0).unsqueeze(1)
    w101 = (wx1 * wy0 * wz1).unsqueeze(1)
    w110 = (wx1 * wy1 * wz0).unsqueeze(1)
    w111 = (wx1 * wy1 * wz1).unsqueeze(1)
    return (
        w000 * c000
        + w001 * c001
        + w010 * c010
        + w011 * c011
        + w100 * c100
        + w101 * c101
        + w110 * c110
        + w111 * c111
    )


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into model-dimension vectors."""

    def __init__(self, model_dim: int, frequency_embedding_size: int = 256) -> None:
        """Initialize the timestep embedder.

        Args:
            model_dim: Output embedding dimension.
            frequency_embedding_size: Sinusoidal frequency embedding dimension.
        """
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    @staticmethod
    def timestep_embedding(
        timesteps: Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timestep tensor with shape `(B,)`.
            dim: Embedding dimension.
            max_period: Maximum sinusoidal period.

        Returns:
            Tensor: Sinusoidal embeddings with shape `(B, dim)`.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(
                start=0, end=half, dtype=torch.float32, device=timesteps.device
            )
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed diffusion timesteps.

        Args:
            timesteps: Timestep tensor with shape `(B,)`.

        Returns:
            Tensor: Embedded timesteps with shape `(B, D)`.
        """
        return self.mlp(
            self.timestep_embedding(timesteps, self.frequency_embedding_size)
        )


class LabelEmbedder(nn.Module):
    """Embed class labels with optional classifier-free dropout."""

    def __init__(self, num_classes: int, model_dim: int, dropout_prob: float) -> None:
        """Initialize the label embedder.

        Args:
            num_classes: Number of semantic classes.
            model_dim: Embedding dimension.
            dropout_prob: Label dropout probability for CFG-style training.
        """
        super().__init__()
        use_cfg_embedding = dropout_prob > 0.0
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding),
            model_dim,
        )

    def token_drop(
        self,
        labels: Tensor,
        force_drop_ids: Tensor | None = None,
    ) -> Tensor:
        """Drop labels to enable classifier-free guidance.

        Args:
            labels: Label tensor with shape `(B,)`.
            force_drop_ids: Optional explicit dropout mask.

        Returns:
            Tensor: Dropped or original labels with shape `(B,)`.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: Tensor,
        train: bool,
        force_drop_ids: Tensor | None = None,
    ) -> Tensor:
        """Embed labels.

        Args:
            labels: Label tensor with shape `(B,)`.
            train: Whether the caller is in training mode.
            force_drop_ids: Optional explicit dropout mask.

        Returns:
            Tensor: Label embeddings with shape `(B, D)`.
        """
        use_dropout = self.dropout_prob > 0.0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class MultiheadSelfAttention(nn.Module):
    """Standard multi-head self-attention over token sequences."""

    def __init__(self, model_dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        """Initialize the self-attention module.

        Args:
            model_dim: Token dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to enable QKV bias.
        """
        super().__init__()
        if model_dim % num_heads != 0:
            message = "model_dim must be divisible by num_heads for self-attention."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention to tokens.

        Args:
            x: Input tokens with shape `(B, N, D)`.

        Returns:
            Tensor: Updated tokens with shape `(B, N, D)`.
        """
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            self.model_dim,
        )
        return self.proj(attended)


class MultiheadCrossAttention(nn.Module):
    """Cross-attention from geometry tokens to HMC condition tokens."""

    def __init__(self, model_dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        """Initialize the cross-attention module.

        Args:
            model_dim: Token dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to enable projection bias.
        """
        super().__init__()
        if model_dim % num_heads != 0:
            message = "model_dim must be divisible by num_heads for cross-attention."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim, bias=qkv_bias)
        self.key_value = nn.Linear(model_dim, model_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(model_dim, model_dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Apply cross-attention.

        Args:
            x: Query tokens with shape `(B, N, D)`.
            context: Condition tokens with shape `(B, M, D)`.

        Returns:
            Tensor: Cross-attended tokens with shape `(B, N, D)`.
        """
        batch_size, token_count, _ = x.shape
        _, context_count, _ = context.shape
        query = self.query(x).view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key_value = self.key_value(context).view(
            batch_size,
            context_count,
            2,
            self.num_heads,
            self.head_dim,
        )
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value.unbind(0)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            self.model_dim,
        )
        return self.proj(attended)


class BottleneckResampler(nn.Module):
    """Perceiver-style resampler used by the 3.2 bottleneck fusion path."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
    ) -> None:
        """Initialize the bottleneck resampler.

        Args:
            model_dim: Token dimension.
            num_heads: Number of attention heads.
            depth: Number of resampler blocks.
            mlp_ratio: Hidden expansion ratio in the feed-forward branch.
        """
        super().__init__()
        if depth <= 0:
            message = "depth must be positive for bottleneck resampling."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_query": nn.LayerNorm(
                            model_dim,
                            elementwise_affine=False,
                            eps=1e-6,
                        ),
                        "norm_context": nn.LayerNorm(
                            model_dim,
                            elementwise_affine=False,
                            eps=1e-6,
                        ),
                        "cross_attn": MultiheadCrossAttention(model_dim, num_heads),
                        "mlp": FeedForward(model_dim, mlp_ratio),
                    }
                )
                for _ in range(depth)
            ]
        )

    def forward(self, query_tokens: Tensor, context_tokens: Tensor) -> Tensor:
        """Resample context information into a query token set.

        Args:
            query_tokens: Query tokens with shape `(B, N, D)`.
            context_tokens: Context tokens with shape `(B, M, D)`.

        Returns:
            Tensor: Updated query tokens with shape `(B, N, D)`.
        """
        x = query_tokens
        for layer in self.layers:
            x = x + layer["cross_attn"](
                layer["norm_query"](x),
                layer["norm_context"](context_tokens),
            )
            x = x + layer["mlp"](x)
        return x


class FeedForward(nn.Module):
    """Transformer MLP branch."""

    def __init__(self, model_dim: int, mlp_ratio: float) -> None:
        """Initialize the MLP branch.

        Args:
            model_dim: Token dimension.
            mlp_ratio: Hidden expansion ratio.
        """
        super().__init__()
        hidden_dim = int(model_dim * mlp_ratio)
        self.layers = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP branch.

        Args:
            x: Input tensor with shape `(B, N, D)`.

        Returns:
            Tensor: Output tensor with shape `(B, N, D)`.
        """
        return self.layers(x)


class HMCConditionedDiTBlock(nn.Module):
    """DiT-style block with HMC cross-attention and AdaLN modulation."""

    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize the HMC-conditioned DiT block.

        Args:
            model_dim: Token dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden expansion ratio of the MLP branch.
        """
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.cross_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiheadSelfAttention(model_dim, num_heads)
        self.cross_attn = MultiheadCrossAttention(model_dim, num_heads)
        self.mlp = FeedForward(model_dim, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 9),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize DiT-style modulation layers."""
        nn.init.constant_(self.adaln_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0.0)

    def forward(self, x: Tensor, condition: Tensor, hmc_tokens: Tensor) -> Tensor:
        """Apply one HMC-conditioned DiT block.

        Args:
            x: Geometry tokens with shape `(B, N, D)`.
            condition: Global condition vector with shape `(B, D)`.
            hmc_tokens: HMC condition tokens with shape `(B, M, D)`.

        Returns:
            Tensor: Updated geometry tokens with shape `(B, N, D)`.
        """
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_modulation(condition).chunk(9, dim=1)

        x = x + gate_self.unsqueeze(1) * self.self_attn(
            modulate(self.self_norm(x), shift_self, scale_self)
        )
        # Variant 3.1: queries are geometry tokens; keys/values are HMC tokens.
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(
            modulate(self.cross_norm(x), shift_cross, scale_cross),
            hmc_tokens,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        )
        return x


class AdaLNSelfAttentionBlock(nn.Module):
    """DiT-style self-attention block used inside the 3.2 bottleneck latent path."""

    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        """Initialize the bottleneck processor block.

        Args:
            model_dim: Token dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden expansion ratio in the MLP branch.
        """
        super().__init__()
        self.self_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiheadSelfAttention(model_dim, num_heads)
        self.mlp = FeedForward(model_dim, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, model_dim * 6),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize DiT-style modulation layers."""
        nn.init.constant_(self.adaln_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0.0)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply self-attention and MLP updates with AdaLN modulation.

        Args:
            x: Bottleneck tokens with shape `(B, N, D)`.
            condition: Global condition vector with shape `(B, D)`.

        Returns:
            Tensor: Updated bottleneck tokens with shape `(B, N, D)`.
        """
        (
            shift_self,
            scale_self,
            gate_self,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_modulation(condition).chunk(6, dim=1)
        x = x + gate_self.unsqueeze(1) * self.self_attn(
            modulate(self.self_norm(x), shift_self, scale_self)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        )
        return x


@dataclass(slots=True, frozen=True)
class HMCConditionedDiTConfig:
    """Configuration for the minimal HMC-conditioned DiT backbone.

    Args:
        input_dim: Input token dimension before projection.
        model_dim: Internal token dimension.
        condition_dim: Dimension of `(g, C)` produced by the HMC encoder.
        depth: Number of DiT-style blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden expansion ratio in MLP layers.
        num_classes: Number of semantic labels.
        class_dropout_prob: Label dropout probability.
        max_tokens: Maximum supported input token length.
        out_dim: Output token dimension. Defaults to `input_dim`.
    """

    input_dim: int = 256
    model_dim: int = 256
    condition_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_classes: int = 1
    class_dropout_prob: float = 0.1
    max_tokens: int = 1024
    out_dim: int | None = None

    def __post_init__(self) -> None:
        """Validate the backbone configuration."""
        if self.input_dim <= 0:
            message = "input_dim must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.model_dim <= 0:
            message = "model_dim must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.condition_dim <= 0:
            message = "condition_dim must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.depth <= 0:
            message = "depth must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.num_heads <= 0 or self.model_dim % self.num_heads != 0:
            message = "num_heads must be positive and divide model_dim."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.mlp_ratio <= 0.0:
            message = "mlp_ratio must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.num_classes <= 0:
            message = "num_classes must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if not 0.0 <= self.class_dropout_prob < 1.0:
            message = "class_dropout_prob must be in [0, 1)."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.max_tokens <= 0:
            message = "max_tokens must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if self.out_dim is None:
            object.__setattr__(self, "out_dim", self.input_dim)
        elif self.out_dim <= 0:
            message = "out_dim must be positive when provided."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)


class HMCConditionedDiTBackbone(nn.Module):
    """Minimal DiT-style token backbone with HMC conditioning."""

    def __init__(self, config: HMCConditionedDiTConfig) -> None:
        """Initialize the minimal HMC-conditioned DiT backbone.

        Args:
            config: Backbone configuration.
        """
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_tokens, config.model_dim)
        )
        self.timestep_embedder = TimestepEmbedder(config.model_dim)
        self.label_embedder = LabelEmbedder(
            config.num_classes,
            config.model_dim,
            config.class_dropout_prob,
        )
        self.hmc_global_projection = nn.Linear(config.condition_dim, config.model_dim)
        self.hmc_token_projection = nn.Linear(config.condition_dim, config.model_dim)
        self.blocks = nn.ModuleList(
            [
                HMCConditionedDiTBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.model_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim * 2),
        )
        self.output_projection = nn.Linear(config.model_dim, config.out_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers and DiT-style modulation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0.0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_adaln[-1].weight, 0.0)
        nn.init.constant_(self.final_adaln[-1].bias, 0.0)
        nn.init.constant_(self.output_projection.weight, 0.0)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(
        self,
        x_tokens: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        hmc_conditions: HMCConditionOutput,
    ) -> Tensor:
        """Apply the HMC-conditioned DiT backbone.

        Args:
            x_tokens: Input tokens with shape `(B, N, input_dim)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            hmc_conditions: HMC encoder outputs `(g, C)`.

        Returns:
            Tensor: Output tokens with shape `(B, N, out_dim)`.
        """
        device = self.input_projection.weight.device
        dtype = self.input_projection.weight.dtype
        x_tokens = torch.as_tensor(x_tokens, device=device, dtype=dtype)
        timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.int64)
        labels = torch.as_tensor(labels, device=device, dtype=torch.int64)
        global_embedding = torch.as_tensor(
            hmc_conditions.global_embedding,
            device=device,
            dtype=dtype,
        )
        condition_tokens = torch.as_tensor(
            hmc_conditions.condition_tokens,
            device=device,
            dtype=dtype,
        )

        if x_tokens.ndim != 3 or x_tokens.shape[-1] != self.config.input_dim:
            message = (
                "x_tokens must have shape (B, N, input_dim). "
                f"Received {tuple(x_tokens.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        batch_size, token_count, _ = x_tokens.shape
        if token_count > self.config.max_tokens:
            message = (
                f"token_count={token_count} exceeds "
                f"max_tokens={self.config.max_tokens}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if timesteps.shape != (batch_size,):
            message = (
                f"timesteps must have shape ({batch_size},), "
                f"got {tuple(timesteps.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if labels.shape != (batch_size,):
            message = (
                f"labels must have shape ({batch_size},), got {tuple(labels.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        invalid_labels = (labels < 0) | (labels >= self.config.num_classes)
        if invalid_labels.any():
            invalid_label_values = labels[invalid_labels].detach().cpu().tolist()
            message = (
                "labels must be in [0, num_classes - 1] before CFG dropout. "
                f"Got invalid labels {invalid_label_values!r} for "
                f"num_classes={self.config.num_classes}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if global_embedding.shape != (batch_size, self.config.condition_dim):
            message = (
                "hmc_conditions.global_embedding must have shape "
                f"({batch_size}, {self.config.condition_dim}), "
                f"got {tuple(global_embedding.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size:
            message = (
                "hmc_conditions.condition_tokens must have shape "
                f"({batch_size}, M, {self.config.condition_dim}), "
                f"got {tuple(condition_tokens.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if condition_tokens.shape[-1] != self.config.condition_dim:
            message = (
                "hmc_conditions.condition_tokens last dim must equal condition_dim. "
                f"Expected {self.config.condition_dim}, "
                f"got {condition_tokens.shape[-1]}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)

        x = self.input_projection(x_tokens)
        x = x + self.position_embedding[:, :token_count]

        timestep_condition = self.timestep_embedder(timesteps)
        label_condition = self.label_embedder(labels, self.training)
        hmc_global = self.hmc_global_projection(global_embedding)
        hmc_tokens = self.hmc_token_projection(condition_tokens)
        condition = timestep_condition + label_condition + hmc_global

        for block in self.blocks:
            x = block(x, condition, hmc_tokens)

        shift, scale = self.final_adaln(condition).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.output_projection(x)


class HMCConditionedBottleneckDiTBackbone(nn.Module):
    """Variant 3.2 bottleneck-fusion backbone with Perceiver-style resampling."""

    def __init__(
        self,
        config: HMCConditionedDiTConfig,
        num_bottleneck_latents: int = 32,
        resampler_depth: int = 2,
        resampler_mlp_ratio: float | None = None,
    ) -> None:
        """Initialize the 3.2 bottleneck-fusion backbone.

        Args:
            config: Backbone configuration.
            num_bottleneck_latents: Number of learned bottleneck latents.
            resampler_depth: Number of cross-attention resampler blocks.
            resampler_mlp_ratio: Optional MLP ratio for the resampler path.
        """
        super().__init__()
        if num_bottleneck_latents <= 0:
            message = "num_bottleneck_latents must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if resampler_depth <= 0:
            message = "resampler_depth must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if resampler_mlp_ratio is not None and resampler_mlp_ratio <= 0.0:
            message = "resampler_mlp_ratio must be positive when provided."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.config = config
        self.num_bottleneck_latents = num_bottleneck_latents
        effective_resampler_ratio = (
            config.mlp_ratio if resampler_mlp_ratio is None else resampler_mlp_ratio
        )
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_tokens, config.model_dim)
        )
        self.timestep_embedder = TimestepEmbedder(config.model_dim)
        self.label_embedder = LabelEmbedder(
            config.num_classes,
            config.model_dim,
            config.class_dropout_prob,
        )
        self.hmc_global_projection = nn.Linear(config.condition_dim, config.model_dim)
        self.hmc_token_projection = nn.Linear(config.condition_dim, config.model_dim)
        self.bottleneck_latents = nn.Parameter(
            torch.randn(num_bottleneck_latents, config.model_dim)
        )
        self.down_resampler = BottleneckResampler(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            depth=resampler_depth,
            mlp_ratio=effective_resampler_ratio,
        )
        self.bottleneck_blocks = nn.ModuleList(
            [
                AdaLNSelfAttentionBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                )
                for _ in range(config.depth)
            ]
        )
        self.up_resampler = BottleneckResampler(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            depth=resampler_depth,
            mlp_ratio=effective_resampler_ratio,
        )
        self.final_norm = nn.LayerNorm(
            config.model_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim * 2),
        )
        self.output_projection = nn.Linear(config.model_dim, config.out_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize linear layers and DiT-style modulation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.bottleneck_latents, std=0.02)
        for block in self.bottleneck_blocks:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0.0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_adaln[-1].weight, 0.0)
        nn.init.constant_(self.final_adaln[-1].bias, 0.0)
        nn.init.constant_(self.output_projection.weight, 0.0)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(
        self,
        x_tokens: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        hmc_conditions: HMCConditionOutput,
    ) -> Tensor:
        """Apply the variant 3.2 bottleneck-fusion backbone.

        Args:
            x_tokens: Input tokens with shape `(B, N, input_dim)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            hmc_conditions: HMC encoder outputs `(g, C)`.

        Returns:
            Tensor: Output tokens with shape `(B, N, out_dim)`.
        """
        device = self.input_projection.weight.device
        dtype = self.input_projection.weight.dtype
        x_tokens = torch.as_tensor(x_tokens, device=device, dtype=dtype)
        timesteps = torch.as_tensor(timesteps, device=device, dtype=torch.int64)
        labels = torch.as_tensor(labels, device=device, dtype=torch.int64)
        global_embedding = torch.as_tensor(
            hmc_conditions.global_embedding,
            device=device,
            dtype=dtype,
        )
        condition_tokens = torch.as_tensor(
            hmc_conditions.condition_tokens,
            device=device,
            dtype=dtype,
        )

        if x_tokens.ndim != 3 or x_tokens.shape[-1] != self.config.input_dim:
            message = (
                "x_tokens must have shape (B, N, input_dim). "
                f"Received {tuple(x_tokens.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        batch_size, token_count, _ = x_tokens.shape
        if token_count > self.config.max_tokens:
            message = (
                f"token_count={token_count} exceeds "
                f"max_tokens={self.config.max_tokens}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if timesteps.shape != (batch_size,):
            message = (
                f"timesteps must have shape ({batch_size},), "
                f"got {tuple(timesteps.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if labels.shape != (batch_size,):
            message = (
                f"labels must have shape ({batch_size},), got {tuple(labels.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        invalid_labels = (labels < 0) | (labels >= self.config.num_classes)
        if invalid_labels.any():
            invalid_label_values = labels[invalid_labels].detach().cpu().tolist()
            message = (
                "labels must be in [0, num_classes - 1] before CFG dropout. "
                f"Got invalid labels {invalid_label_values!r} for "
                f"num_classes={self.config.num_classes}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if global_embedding.shape != (batch_size, self.config.condition_dim):
            message = (
                "hmc_conditions.global_embedding must have shape "
                f"({batch_size}, {self.config.condition_dim}), "
                f"got {tuple(global_embedding.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size:
            message = (
                "hmc_conditions.condition_tokens must have shape "
                f"({batch_size}, M, {self.config.condition_dim}), "
                f"got {tuple(condition_tokens.shape)!r}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if condition_tokens.shape[-1] != self.config.condition_dim:
            message = (
                "hmc_conditions.condition_tokens last dim must equal condition_dim. "
                f"Expected {self.config.condition_dim}, "
                f"got {condition_tokens.shape[-1]}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)

        x = self.input_projection(x_tokens)
        x = x + self.position_embedding[:, :token_count]

        timestep_condition = self.timestep_embedder(timesteps)
        label_condition = self.label_embedder(labels, self.training)
        hmc_global = self.hmc_global_projection(global_embedding)
        hmc_tokens = self.hmc_token_projection(condition_tokens)
        condition = timestep_condition + label_condition + hmc_global

        bottleneck = self.bottleneck_latents.unsqueeze(0).expand(batch_size, -1, -1)
        # Variant 3.2: first compress geometry and HMC tokens into bottlenecks.
        bottleneck = self.down_resampler(
            bottleneck,
            torch.cat([x, hmc_tokens], dim=1),
        )
        for block in self.bottleneck_blocks:
            bottleneck = block(bottleneck, condition)
        # Then inject bottleneck information back into geometry tokens.
        x = self.up_resampler(x, bottleneck)

        shift, scale = self.final_adaln(condition).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.output_projection(x)


class HMCConditionedDiT(nn.Module):
    """End-to-end HMC-conditioned DiT wrapper.

    This module composes the HMC condition encoder and the DiT-style backbone
    so callers can pass raw HMC descriptor/sequence tensors directly.
    """

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
    ) -> None:
        """Initialize the end-to-end HMC-conditioned DiT module.

        Args:
            hmc_config: Numerical HMC configuration.
            encoder_config: HMC condition encoder hyperparameters.
            backbone_config: DiT backbone hyperparameters.
        """
        super().__init__()
        if encoder_config.model_dim != backbone_config.condition_dim:
            message = (
                "encoder_config.model_dim must equal backbone_config.condition_dim. "
                f"Got {encoder_config.model_dim} and {backbone_config.condition_dim}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.hmc_encoder = HMCConditionEncoder(hmc_config, encoder_config)
        self.backbone = HMCConditionedDiTBackbone(backbone_config)

    def forward(
        self,
        x_tokens: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Encode HMC conditions and apply the DiT-style backbone.

        Args:
            x_tokens: Input geometry tokens with shape `(B, N, input_dim)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            descriptor: HMC global descriptor with shape `(B, D_hmc)`.
            sequences: One Hilbert sequence tensor per configured scale.

        Returns:
            Tensor: Output tokens with shape `(B, N, out_dim)`.
        """
        hmc_conditions = self.hmc_encoder(descriptor, sequences)
        return self.backbone(x_tokens, timesteps, labels, hmc_conditions)


class HMCConditionedBottleneckDiT(nn.Module):
    """End-to-end variant 3.2 bottleneck-fusion HMC-DiT wrapper."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
        num_bottleneck_latents: int = 32,
        resampler_depth: int = 2,
        resampler_mlp_ratio: float | None = None,
    ) -> None:
        """Initialize the end-to-end 3.2 HMC-DiT module.

        Args:
            hmc_config: Numerical HMC configuration.
            encoder_config: HMC condition encoder hyperparameters.
            backbone_config: DiT backbone hyperparameters.
            num_bottleneck_latents: Number of learned bottleneck latents.
            resampler_depth: Number of bottleneck resampler blocks.
            resampler_mlp_ratio: Optional MLP ratio in the resampler path.
        """
        super().__init__()
        if encoder_config.model_dim != backbone_config.condition_dim:
            message = (
                "encoder_config.model_dim must equal backbone_config.condition_dim. "
                f"Got {encoder_config.model_dim} and {backbone_config.condition_dim}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        self.hmc_encoder = HMCConditionEncoder(hmc_config, encoder_config)
        self.backbone = HMCConditionedBottleneckDiTBackbone(
            config=backbone_config,
            num_bottleneck_latents=num_bottleneck_latents,
            resampler_depth=resampler_depth,
            resampler_mlp_ratio=resampler_mlp_ratio,
        )

    def forward(
        self,
        x_tokens: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Encode HMC conditions and apply the 3.2 bottleneck backbone.

        Args:
            x_tokens: Input geometry tokens with shape `(B, N, input_dim)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            descriptor: HMC global descriptor with shape `(B, D_hmc)`.
            sequences: One Hilbert sequence tensor per configured scale.

        Returns:
            Tensor: Output tokens with shape `(B, N, out_dim)`.
        """
        hmc_conditions = self.hmc_encoder(descriptor, sequences)
        return self.backbone(x_tokens, timesteps, labels, hmc_conditions)


class HMCConditionedDiTVoxel(nn.Module):
    """Voxel-grid wrapper around the HMC-conditioned DiT token backbone."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
        voxel_size: int,
        patch_size: int,
        in_channels: int = 3,
        out_channels: int | None = None,
    ) -> None:
        """Initialize the voxel-grid HMC-conditioned DiT wrapper.

        Args:
            hmc_config: Numerical HMC configuration.
            encoder_config: HMC condition encoder hyperparameters.
            backbone_config: DiT backbone hyperparameters.
            voxel_size: Side length of the cubic voxel grid.
            patch_size: Side length of each cubic patch.
            in_channels: Number of voxel feature channels.
            out_channels: Number of output voxel channels. Defaults to
                `in_channels`.
        """
        super().__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.out_channels <= 0:
            message = "out_channels must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.patch_embed = PatchEmbedVoxel(
            voxel_size=voxel_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=backbone_config.input_dim,
        )
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_grid_size = self.patch_embed.patch_grid_size
        self.num_patches = self.patch_embed.num_patches
        expected_out_dim = (patch_size**3) * self.out_channels
        if backbone_config.max_tokens < self.num_patches:
            message = (
                f"backbone_config.max_tokens={backbone_config.max_tokens} is too small "
                f"for num_patches={self.num_patches}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if backbone_config.out_dim != expected_out_dim:
            message = (
                "backbone_config.out_dim must match patch_size**3 * out_channels. "
                f"Expected {expected_out_dim}, got {backbone_config.out_dim}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.token_model = HMCConditionedDiT(
            hmc_config=hmc_config,
            encoder_config=encoder_config,
            backbone_config=backbone_config,
        )
        self._initialize_fixed_position_embedding()

    def _initialize_fixed_position_embedding(self) -> None:
        """Fill the backbone token positions with fixed 3D sin-cos embeddings."""
        position_embedding = get_3d_sincos_pos_embed(
            embed_dim=self.token_model.backbone.config.model_dim,
            grid_size=self.patch_grid_size,
        )
        target = self.token_model.backbone.position_embedding
        if target.shape[1] < self.num_patches:
            message = "Backbone position embedding is shorter than the patch grid."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        target.data.zero_()
        target.data[:, : self.num_patches] = position_embedding.unsqueeze(0)
        target.requires_grad_(False)

    def _unpatchify(self, patch_tokens: Tensor) -> Tensor:
        """Convert patch tokens back to a dense voxel grid.

        Args:
            patch_tokens: Patch predictions with shape
                `(B, num_patches, patch_size**3 * out_channels)`.

        Returns:
            Tensor: Dense voxel tensor with shape
                `(B, out_channels, voxel_size, voxel_size, voxel_size)`.
        """
        batch_size, token_count, channel_dim = patch_tokens.shape
        expected_channel_dim = (self.patch_size**3) * self.out_channels
        if token_count != self.num_patches:
            message = (
                f"Patch token count must be {self.num_patches}, got {token_count}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if channel_dim != expected_channel_dim:
            message = (
                "Patch token channel dimension does not match the voxel layout. "
                f"Expected {expected_channel_dim}, got {channel_dim}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)

        grid = self.patch_grid_size
        patch = self.patch_size
        channels = self.out_channels
        voxels = patch_tokens.reshape(
            batch_size,
            grid,
            grid,
            grid,
            patch,
            patch,
            patch,
            channels,
        )
        voxels = torch.einsum("nxyzpqrc->ncxpyqzr", voxels)
        return voxels.reshape(
            batch_size,
            channels,
            grid * patch,
            grid * patch,
            grid * patch,
        )

    def forward(
        self,
        voxels: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Apply the HMC-conditioned DiT to dense voxel grids.

        Args:
            voxels: Dense voxel tensor with shape `(B, C, X, Y, Z)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            descriptor: HMC global descriptor with shape `(B, D_hmc)`.
            sequences: One Hilbert sequence tensor per configured scale.

        Returns:
            Tensor: Output voxel tensor with shape
                `(B, out_channels, voxel_size, voxel_size, voxel_size)`.
        """
        device = self.patch_embed.proj.weight.device
        dtype = self.patch_embed.proj.weight.dtype
        voxels = torch.as_tensor(voxels, device=device, dtype=dtype)
        patch_tokens = self.patch_embed(voxels)
        patch_outputs = self.token_model(
            patch_tokens,
            timesteps,
            labels,
            descriptor,
            sequences,
        )
        return self._unpatchify(patch_outputs)


class HMCConditionedBottleneckDiTVoxel(nn.Module):
    """Voxel-grid wrapper around the variant 3.2 bottleneck-fusion backbone."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
        voxel_size: int,
        patch_size: int,
        in_channels: int = 3,
        out_channels: int | None = None,
        num_bottleneck_latents: int = 32,
        resampler_depth: int = 2,
        resampler_mlp_ratio: float | None = None,
    ) -> None:
        """Initialize the voxel-grid 3.2 bottleneck-fusion wrapper."""
        super().__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.out_channels <= 0:
            message = "out_channels must be positive."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.patch_embed = PatchEmbedVoxel(
            voxel_size=voxel_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=backbone_config.input_dim,
        )
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_grid_size = self.patch_embed.patch_grid_size
        self.num_patches = self.patch_embed.num_patches
        expected_out_dim = (patch_size**3) * self.out_channels
        if backbone_config.max_tokens < self.num_patches:
            message = (
                f"backbone_config.max_tokens={backbone_config.max_tokens} is too small "
                f"for num_patches={self.num_patches}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        if backbone_config.out_dim != expected_out_dim:
            message = (
                "backbone_config.out_dim must match patch_size**3 * out_channels. "
                f"Expected {expected_out_dim}, got {backbone_config.out_dim}."
            )
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)

        self.token_model = HMCConditionedBottleneckDiT(
            hmc_config=hmc_config,
            encoder_config=encoder_config,
            backbone_config=backbone_config,
            num_bottleneck_latents=num_bottleneck_latents,
            resampler_depth=resampler_depth,
            resampler_mlp_ratio=resampler_mlp_ratio,
        )
        self._initialize_fixed_position_embedding()

    def _initialize_fixed_position_embedding(self) -> None:
        """Fill the backbone token positions with fixed 3D sin-cos embeddings."""
        position_embedding = get_3d_sincos_pos_embed(
            embed_dim=self.token_model.backbone.config.model_dim,
            grid_size=self.patch_grid_size,
        )
        target = self.token_model.backbone.position_embedding
        if target.shape[1] < self.num_patches:
            message = "Backbone position embedding is shorter than the patch grid."
            LOGGER.error(message)
            raise HMCModelConfigurationError(message)
        target.data.zero_()
        target.data[:, : self.num_patches] = position_embedding.unsqueeze(0)
        target.requires_grad_(False)

    def _unpatchify(self, patch_tokens: Tensor) -> Tensor:
        """Convert patch tokens back to a dense voxel grid."""
        batch_size, token_count, channel_dim = patch_tokens.shape
        expected_channel_dim = (self.patch_size**3) * self.out_channels
        if token_count != self.num_patches:
            message = (
                f"Patch token count must be {self.num_patches}, got {token_count}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)
        if channel_dim != expected_channel_dim:
            message = (
                "Patch token channel dimension does not match the voxel layout. "
                f"Expected {expected_channel_dim}, got {channel_dim}."
            )
            LOGGER.error(message)
            raise HMCModelInputError(message)

        grid = self.patch_grid_size
        patch = self.patch_size
        channels = self.out_channels
        voxels = patch_tokens.reshape(
            batch_size,
            grid,
            grid,
            grid,
            patch,
            patch,
            patch,
            channels,
        )
        voxels = torch.einsum("nxyzpqrc->ncxpyqzr", voxels)
        return voxels.reshape(
            batch_size,
            channels,
            grid * patch,
            grid * patch,
            grid * patch,
        )

    def forward(
        self,
        voxels: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Apply the 3.2 HMC-conditioned DiT to dense voxel grids."""
        device = self.patch_embed.proj.weight.device
        dtype = self.patch_embed.proj.weight.dtype
        voxels = torch.as_tensor(voxels, device=device, dtype=dtype)
        patch_tokens = self.patch_embed(voxels)
        patch_outputs = self.token_model(
            patch_tokens,
            timesteps,
            labels,
            descriptor,
            sequences,
        )
        return self._unpatchify(patch_outputs)


class HMCConditionedDiTPointCloud(nn.Module):
    """Point-cloud wrapper around the HMC-conditioned DiT voxel backbone."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
        voxel_size: int,
        patch_size: int,
        in_channels: int = 3,
        out_channels: int | None = None,
        normalize: bool = True,
        eps: float = 0.0,
    ) -> None:
        """Initialize the point-cloud HMC-conditioned DiT wrapper.

        Args:
            hmc_config: Numerical HMC configuration.
            encoder_config: HMC condition encoder hyperparameters.
            backbone_config: DiT backbone hyperparameters.
            voxel_size: Side length of the cubic voxel grid.
            patch_size: Side length of each cubic patch.
            in_channels: Number of input point channels.
            out_channels: Number of output point channels.
            normalize: Whether voxelization should normalize coordinates.
            eps: Numerical stabilizer used during voxelization normalization.
        """
        super().__init__()
        self.voxelizer = PointCloudVoxelizer(
            resolution=voxel_size,
            normalize=normalize,
            eps=eps,
        )
        self.voxel_model = HMCConditionedDiTVoxel(
            hmc_config=hmc_config,
            encoder_config=encoder_config,
            backbone_config=backbone_config,
            voxel_size=voxel_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.out_channels = self.voxel_model.out_channels
        self.resolution = voxel_size

    def forward(
        self,
        points: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Apply the HMC-conditioned DiT to point clouds.

        Args:
            points: Point tensor with shape `(B, C, N)`.
            timesteps: Diffusion timesteps with shape `(B,)`.
            labels: Class labels with shape `(B,)`.
            descriptor: HMC global descriptor with shape `(B, D_hmc)`.
            sequences: One Hilbert sequence tensor per configured scale.

        Returns:
            Tensor: Point predictions with shape `(B, out_channels, N)`.
        """
        device = self.voxel_model.patch_embed.proj.weight.device
        dtype = self.voxel_model.patch_embed.proj.weight.dtype
        points = torch.as_tensor(points, device=device, dtype=dtype)
        voxels, norm_coords = self.voxelizer(points, points)
        voxel_outputs = self.voxel_model(
            voxels,
            timesteps,
            labels,
            descriptor,
            sequences,
        )
        return trilinear_devoxelize(voxel_outputs, norm_coords, self.resolution)


class HMCConditionedBottleneckDiTPointCloud(nn.Module):
    """Point-cloud wrapper around the variant 3.2 bottleneck-fusion voxel backbone."""

    def __init__(
        self,
        hmc_config: HMCConfig,
        encoder_config: HMCEncoderConfig,
        backbone_config: HMCConditionedDiTConfig,
        voxel_size: int,
        patch_size: int,
        in_channels: int = 3,
        out_channels: int | None = None,
        normalize: bool = True,
        eps: float = 0.0,
        num_bottleneck_latents: int = 32,
        resampler_depth: int = 2,
        resampler_mlp_ratio: float | None = None,
    ) -> None:
        """Initialize the point-cloud 3.2 bottleneck-fusion wrapper."""
        super().__init__()
        self.voxelizer = PointCloudVoxelizer(
            resolution=voxel_size,
            normalize=normalize,
            eps=eps,
        )
        self.voxel_model = HMCConditionedBottleneckDiTVoxel(
            hmc_config=hmc_config,
            encoder_config=encoder_config,
            backbone_config=backbone_config,
            voxel_size=voxel_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_bottleneck_latents=num_bottleneck_latents,
            resampler_depth=resampler_depth,
            resampler_mlp_ratio=resampler_mlp_ratio,
        )
        self.out_channels = self.voxel_model.out_channels
        self.resolution = voxel_size

    def forward(
        self,
        points: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        descriptor: Tensor,
        sequences: Sequence[Tensor],
    ) -> Tensor:
        """Apply the 3.2 HMC-conditioned DiT to point clouds."""
        device = self.voxel_model.patch_embed.proj.weight.device
        dtype = self.voxel_model.patch_embed.proj.weight.dtype
        points = torch.as_tensor(points, device=device, dtype=dtype)
        voxels, norm_coords = self.voxelizer(points, points)
        voxel_outputs = self.voxel_model(
            voxels,
            timesteps,
            labels,
            descriptor,
            sequences,
        )
        return trilinear_devoxelize(voxel_outputs, norm_coords, self.resolution)
