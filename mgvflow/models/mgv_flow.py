"""Measurement-Gradient-Conditioned Video Flow (MGV-Flow).

The velocity network predicts the complete flow-matching velocity of the future latent
from the noisy latent ``z_tau``, the encoded history of the five preceding frames, the
normalized lifted residual image and the flow time with a quality descriptor.  It reads
the physical direction as an input rather than adding it to its output.  History-FM is
the same network trained with the physical direction and quality descriptor set to zero.

With the default configuration the network has 107,515,898 parameters.  Trained weights
are not distributed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


MGV_FLOW_ARCHITECTURE = "mgv-flow-v1"


def _group_count(channels: int) -> int:
    return next(group for group in range(min(16, channels), 0, -1) if channels % group == 0)


def _sinusoidal(time: torch.Tensor, width: int) -> torch.Tensor:
    if time.ndim != 1:
        raise ValueError("flow_time must be [B]")
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=time.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = time.float()[:, None] * frequencies[None]
    return F.pad(torch.cat((phase.sin(), phase.cos()), dim=1), (0, width - 2 * half))


class _TimeQualityEmbedding(nn.Module):
    def __init__(self, width: int, quality_features: int) -> None:
        super().__init__()
        self.width = width
        self.time = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))
        self.quality = nn.Sequential(nn.LayerNorm(quality_features), nn.Linear(quality_features, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, flow_time: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        dtype = next(self.time.parameters()).dtype
        time_input = _sinusoidal(flow_time, self.width).to(dtype=dtype)
        quality_input = quality.to(dtype=next(self.quality.parameters()).dtype)
        return self.time(time_input) + self.quality(quality_input)


class _AdaNorm(nn.Module):
    def __init__(self, width: int, condition_width: int | None = None) -> None:
        super().__init__()
        condition_width = width if condition_width is None else condition_width
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(condition_width, 2 * width))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        shape = (condition.shape[0],) + (1,) * (value.ndim - 2) + (scale.shape[-1],)
        return self.norm(value) * (1.0 + scale.reshape(shape)) + shift.reshape(shape)


class _ResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_width: int, *, stride: int = 1) -> None:
        super().__init__()
        self.first = nn.Conv3d(in_channels, out_channels, 3, stride=(1, stride, stride), padding=1)
        self.second = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_width, 2 * out_channels))
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv3d(in_channels, out_channels, 1, stride=(1, stride, stride))

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.first(value)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        hidden = self.norm2(self.second(F.silu(hidden)))
        return F.silu(hidden + self.skip(value))


class _SpatialCrossAttention(nn.Module):
    """Cross-attend each future-latent spatial site to pooled five-frame history."""

    def __init__(self, width: int, heads: int, condition_width: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must divide heads")
        self.query_norm = _AdaNorm(width, condition_width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = _AdaNorm(width, condition_width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width))
        self.condition_projection = nn.Sequential(nn.SiLU(), nn.Linear(condition_width, width))

    def forward(self, value: torch.Tensor, history: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # value [B,D,T,H,W]; history [B,D,Th,H,W].  History is pooled in time
        # but keeps every spatial location as a context token.
        batch, width, frames, height, image_width = value.shape
        if history.ndim != 5 or history.shape[0] != batch or history.shape[1] != width:
            raise ValueError("cross-attention history must align in batch and channel width")
        query = value.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * image_width, width)
        # The history tokens use their own grid, which can be one element larger than the
        # strided future path at odd latent resolutions (30x54 versus 30x53 at 480p).
        history_height, history_width = history.shape[-2:]
        context = history.mean(dim=2).permute(0, 2, 3, 1).reshape(
            batch, history_height * history_width, width
        )
        conditioned = self.query_norm(query, condition)
        attended = self.attention(conditioned, self.context_norm(context), self.context_norm(context), need_weights=False)[0]
        query = query + attended + self.condition_projection(condition)[:, None]
        query = query + self.ff(self.ff_norm(query, condition))
        return query.reshape(batch, frames, height, image_width, width).permute(0, 4, 1, 2, 3)


@dataclass(frozen=True)
class MGVFlowConfig:
    latent_channels: int = 16
    # Five *video* frames map to two LeanVAE latent-time slices under its
    # causal 4n+1 convention.  Names intentionally say latent_frames so this
    # cannot be confused with the five raw seed/future video frames.
    history_latent_frames: int = 2
    future_latent_frames: int = 2
    base_width: int = 288
    attention_heads: int = 8
    residual_blocks_per_scale: int = 3
    quality_features: int = 5
    activation_checkpointing: bool = True
    architecture_version: str = MGV_FLOW_ARCHITECTURE

    def validate(self) -> None:
        if self.latent_channels != 16:
            raise ValueError("MGV-Flow is pinned to 16-channel LeanVAE latents")
        if self.history_latent_frames < 1 or self.future_latent_frames < 1:
            raise ValueError("history and future frame counts must be positive")
        if self.base_width < 32 or self.base_width % self.attention_heads:
            raise ValueError("base_width must be >=32 and divisible by attention_heads")
        if self.residual_blocks_per_scale < 1 or self.quality_features < 1:
            raise ValueError("block count and quality_features must be positive")
        if self.architecture_version != MGV_FLOW_ARCHITECTURE:
            raise ValueError("unsupported MGV-Flow architecture version")


class MGVFlow(nn.Module):
    """Predict the latent flow velocity conditioned on history and measurement gradients."""

    def __init__(self, config: MGVFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or MGVFlowConfig()
        self.config.validate()
        c, width = self.config, self.config.base_width
        # z_tau, spatially-aligned normalized detector descent, and a learned
        # history summary enter separately before fusion.
        self.noisy_projection = nn.Conv3d(c.latent_channels, width, 1)
        self.physics_projection = nn.Conv3d(c.latent_channels, width, 1)
        self.history_projection = nn.Conv3d(c.latent_channels, width, 1)
        self.embedding = _TimeQualityEmbedding(width, c.quality_features)
        self.in_blocks = nn.ModuleList(_ResidualBlock3d(width, width, width) for _ in range(c.residual_blocks_per_scale))
        self.down = _ResidualBlock3d(width, 2 * width, width, stride=2)
        self.middle = nn.ModuleList(_ResidualBlock3d(2 * width, 2 * width, width) for _ in range(c.residual_blocks_per_scale))
        self.cross = _SpatialCrossAttention(2 * width, c.attention_heads, width)
        self.history_cross_projection = nn.Conv3d(width, 2 * width, 1)
        self.up = nn.ConvTranspose3d(2 * width, width, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.out_blocks = nn.ModuleList(
            [_ResidualBlock3d(2 * width, width, width)]
            + [_ResidualBlock3d(width, width, width) for _ in range(c.residual_blocks_per_scale - 1)]
        )
        self.output = nn.Sequential(nn.GroupNorm(_group_count(width), width), nn.SiLU(), nn.Conv3d(width, c.latent_channels, 3, padding=1))
        # Standard zero-initialised FM output makes the first predictions
        # stable, without forcing a direct addition of the physics field.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _block(self, block: nn.Module, value: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, value, embedding, use_reentrant=False)
        return block(value, embedding)

    def forward(
        self,
        z_tau: torch.Tensor,
        history_latent: torch.Tensor,
        physics_direction: torch.Tensor,
        flow_time: torch.Tensor,
        physics_quality: torch.Tensor,
    ) -> torch.Tensor:
        c = self.config
        if z_tau.ndim != 5 or z_tau.shape[1] != c.latent_channels:
            raise ValueError("z_tau must be [B,16,T,H,W]")
        if z_tau.shape[2] != c.future_latent_frames:
            raise ValueError("z_tau temporal length does not match future_latent_frames")
        if history_latent.ndim != 5 or history_latent.shape[1] != c.latent_channels:
            raise ValueError("history_latent must be [B,16,Th,H,W]")
        if history_latent.shape[2] != c.history_latent_frames or history_latent.shape[0] != z_tau.shape[0] or history_latent.shape[-2:] != z_tau.shape[-2:]:
            raise ValueError("history latent must align in batch/spatial shape and configured history length")
        if physics_direction.shape != z_tau.shape:
            raise ValueError("physics_direction must exactly align with z_tau")
        batch = z_tau.shape[0]
        if flow_time.ndim == 0:
            flow_time = flow_time.expand(batch)
        if flow_time.shape != (batch,):
            raise ValueError("flow_time must be scalar or [B]")
        if physics_quality.shape != (batch, c.quality_features):
            raise ValueError("physics_quality must be [B,quality_features]")

        model_dtype = self.noisy_projection.weight.dtype
        embedding = self.embedding(flow_time.to(z_tau.device), physics_quality.to(z_tau.device))
        value = self.noisy_projection(z_tau.to(dtype=model_dtype)) + self.physics_projection(physics_direction.to(dtype=model_dtype))
        history = self.history_projection(history_latent.to(dtype=model_dtype))
        # The history grid has its own future-time-independent representation;
        # an average prior is fused before the cross-attention refinement.
        value = value + history.mean(dim=2, keepdim=True)
        for block in self.in_blocks:
            value = self._block(block, value, embedding)
        skip = value
        value = self._block(self.down, value, embedding)
        history_half = self.history_cross_projection(
            F.avg_pool3d(history, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        for block in self.middle:
            value = self._block(block, value, embedding)
        value = self.cross(value, history_half, embedding)
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=(value.shape[2], skip.shape[3], skip.shape[4]), mode="trilinear", align_corners=False)
        value = torch.cat((value, skip), dim=1)
        for block in self.out_blocks:
            value = self._block(block, value, embedding)
        return self.output(value)


def flow_velocity_target(noise: torch.Tensor, clean_latent: torch.Tensor) -> torch.Tensor:
    """The standard straight-line Flow Matching target ``z_1 - z_0``."""

    if noise.shape != clean_latent.shape:
        raise ValueError("noise and clean_latent must share shape")
    return clean_latent - noise


def flow_interpolate(noise: torch.Tensor, clean_latent: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
    """Construct ``z_tau=(1-tau)z_0+tau z_1`` without target leakage to inputs."""

    if noise.shape != clean_latent.shape or noise.ndim != 5:
        raise ValueError("noise and clean_latent must be matching 5D tensors")
    batch = noise.shape[0]
    if flow_time.ndim == 0:
        flow_time = flow_time.expand(batch)
    if flow_time.shape != (batch,) or (flow_time < 0).any() or (flow_time > 1).any():
        raise ValueError("flow_time must lie in [0,1] with shape [B]")
    tau = flow_time.to(device=noise.device, dtype=noise.dtype)[:, None, None, None, None]
    return (1.0 - tau) * noise + tau * clean_latent
