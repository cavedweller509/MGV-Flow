"""Lifting the residual image into the video latent space.

For a clip decoded as ``x_hat = D(z)``, the chain rule lifts the residual image to
``g_z = J_D(z)^T g_x`` with one vector-Jacobian product through the frozen decoder.
MGV-Flow does not add ``-g_z`` to its velocity.  It reads a normalized version of it,
together with a small quality descriptor, as an input condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .operator import SinglePixelOperator


class VideoDecoder(Protocol):
    """Frozen video decoder mapping a latent ``[B, C, Tz, Hz, Wz]`` to gray frames ``[B, T, H, W]``."""

    def __call__(self, latent: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class PhysicsCondition:
    """Detached observation fed to the velocity network at one flow time."""

    direction: torch.Tensor  # normalized lifted residual image, same shape as the latent
    quality: torch.Tensor    # [B, 5] quality descriptor


def lifted_residual_image(
    decoder: VideoDecoder,
    latent: torch.Tensor,
    operator: SinglePixelOperator,
    bucket_values: torch.Tensor,
) -> torch.Tensor:
    """Lifted residual image ``g_z = J_D(z)^T g_x`` evaluated by a single decoder VJP."""
    probe = latent.detach().requires_grad_(True)
    decoded = decoder(probe)
    g_x = operator.residual_image(decoded.detach(), bucket_values)
    (g_z,) = torch.autograd.grad(decoded, probe, grad_outputs=g_x)
    return g_z.detach()


def build_physics_condition(
    decoder: VideoDecoder,
    latent: torch.Tensor,
    flow_time: torch.Tensor,
    operator: SinglePixelOperator,
    bucket_values: torch.Tensor,
) -> PhysicsCondition:
    """Normalized, time-gated direction and quality descriptor read by MGV-Flow.

    The normalization, time gate and descriptor follow Methods of the paper.  The
    reference implementation will be released with the trained weights.
    """
    raise NotImplementedError("released together with the trained MGV-Flow weights")
