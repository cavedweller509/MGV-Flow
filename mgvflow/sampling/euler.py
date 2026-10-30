"""K-step Euler integration of a measurement-conditioned flow."""

from __future__ import annotations

from typing import Callable

import torch

from ..physics.condition import PhysicsCondition

Velocity = Callable[[torch.Tensor, torch.Tensor, PhysicsCondition], torch.Tensor]
ConditionFactory = Callable[[torch.Tensor, torch.Tensor], PhysicsCondition]


@torch.no_grad()
def sample_euler(
    velocity: Velocity,
    condition: ConditionFactory,
    noise: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """Integrate ``dz/dtau = v(z, tau, c(z, tau))`` from ``tau = 0`` to ``1`` in ``steps`` Euler steps.

    The condition is recomputed from the current latent at every step, so the residual
    image always refers to the current estimate.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    latent = noise
    batch = noise.shape[0]
    for k in range(steps):
        tau = torch.full((batch,), k / steps, device=noise.device)
        with torch.enable_grad():
            observation = condition(latent, tau)
        latent = latent + velocity(latent, tau, observation) / steps
    return latent
