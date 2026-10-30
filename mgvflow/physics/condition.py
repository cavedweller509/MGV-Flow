"""Lifting the residual image into the video latent space.

For a clip decoded as ``x_hat = D(z)``, the chain rule lifts the residual image to
``g_z = J_D(z)^T g_x`` with one vector-Jacobian product through the frozen decoder.
MGV-Flow does not add ``-g_z`` to its velocity.  It reads a normalized version of it,
together with a small quality descriptor, as an input condition.
"""

from __future__ import annotations

import math
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


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    clipped = value.clamp(0.0, 1.0)
    return clipped.square() * (3.0 - 2.0 * clipped)


def time_gate(flow_time: torch.Tensor, start: float = 0.15) -> torch.Tensor:
    """``smoothstep(clip((tau - start) / (1 - start)))``, zero for ``tau <= start``."""
    return _smoothstep((flow_time.float() - start) / (1.0 - start))


def build_physics_condition(
    decoder: VideoDecoder,
    latent: torch.Tensor,
    flow_time: torch.Tensor,
    operator: SinglePixelOperator,
    bucket_values: torch.Tensor,
    *,
    physics_start_time: float = 0.15,
    epsilon: float = 1e-6,
    direction_epsilon: float = 1e-12,
) -> PhysicsCondition:
    """Normalized, time-gated direction and quality descriptor read by MGV-Flow.

    ``d_phy = -g_z / (RMS(g_z)^2 + 1e-24)^(1/2) * gate(tau)`` and
    ``q = [log(M/1000), log L_c, log RMS(r~), log RMS(g_z), gate(tau)]`` with every
    logarithm floored at ``1e-6``, where ``L_c`` is the centered measurement loss.
    The returned tensors are detached, so no gradient passes through their construction.
    """
    batch = latent.shape[0]
    if flow_time.ndim == 0:
        flow_time = flow_time.expand(batch)
    if flow_time.shape != (batch,):
        raise ValueError("flow_time must be scalar or [B]")
    probe = latent.detach().float().requires_grad_(True)
    decoded = decoder(probe)
    residual = operator.measure(decoded.detach()) - bucket_values.to(decoded)
    centered = residual - residual.mean(dim=-1, keepdim=True)
    energy = 0.5 * centered.square().mean(dim=(1, 2))
    residual_rms = centered.square().mean(dim=(1, 2)).sqrt()
    g_x = operator.residual_image(decoded.detach(), bucket_values)
    (g_z,) = torch.autograd.grad(decoded, probe, grad_outputs=g_x)
    descent = -g_z.detach()
    raw_rms = descent.square().flatten(1).mean(dim=1).sqrt()
    normalizer = (raw_rms.square() + direction_epsilon**2).sqrt()
    shape = (batch,) + (1,) * (descent.ndim - 1)
    gate = time_gate(flow_time.to(latent.device), physics_start_time)
    direction = descent / normalizer.view(shape) * gate.view(shape)
    count = float(operator.measurement_count)
    quality = torch.stack(
        (
            torch.full_like(gate, math.log(count / 1_000.0)),
            energy.clamp_min(epsilon).log(),
            residual_rms.clamp_min(epsilon).log(),
            raw_rms.clamp_min(epsilon).log(),
            gate,
        ),
        dim=1,
    )
    return PhysicsCondition(direction=direction.detach(), quality=quality.detach())
