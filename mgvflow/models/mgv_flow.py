"""Measurement-Gradient-Conditioned Video Flow (MGV-Flow).

The velocity network predicts the complete flow-matching velocity of the future latent
from four inputs: the noisy latent ``z_tau``, the encoded history of the five preceding
frames, the normalized lifted residual image, and the flow time with a quality
descriptor.  History-FM is the same network without the measurement inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MGVFlowConfig:
    latent_channels: int = 16       # LeanVAE video latent
    history_latent_frames: int = 2  # five history frames under the causal 4n+1 convention
    future_latent_frames: int = 2   # five target frames
    base_width: int = 288
    attention_heads: int = 8
    residual_blocks_per_scale: int = 3
    quality_features: int = 5
    read_measurements: bool = True  # False gives the History-FM baseline


class MGVFlow(nn.Module):
    """Velocity ``v_theta(z_tau, h, d_phy, tau, q)``.

    Shapes: ``z_tau`` and ``physics_direction`` are ``[B, 16, Tz, Hz, Wz]``,
    ``history_latent`` is ``[B, 16, Th, Hz, Wz]``, ``flow_time`` is ``[B]`` and
    ``physics_quality`` is ``[B, 5]``.  The output has the shape of ``z_tau``.
    """

    def __init__(self, config: MGVFlowConfig | None = None) -> None:
        super().__init__()
        self.config = config or MGVFlowConfig()
        raise NotImplementedError("the network definition is released together with the trained weights")

    def forward(
        self,
        z_tau: torch.Tensor,
        history_latent: torch.Tensor,
        physics_direction: torch.Tensor,
        flow_time: torch.Tensor,
        physics_quality: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> "MGVFlow":
        raise NotImplementedError("pretrained weights will be released upon publication")


def flow_interpolate(noise: torch.Tensor, clean: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
    """Straight-line interpolation ``z_tau = (1 - tau) z_0 + tau z_1``."""
    tau = flow_time.view(-1, *([1] * (clean.ndim - 1)))
    return (1.0 - tau) * noise + tau * clean


def flow_velocity_target(noise: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Flow-matching target ``z_1 - z_0``."""
    return clean - noise
