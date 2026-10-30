"""Directional fidelity of a coding configuration and image metrics."""

from __future__ import annotations

import torch


def directional_fidelity(lifted_residual: torch.Tensor, latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    """``C = cos angle(-g_z, z* - z)`` per sample.

    ``lifted_residual`` is ``g_z`` at ``latent``, and ``target_latent`` is the encoding of the
    ground-truth clip.  Because it needs the ground truth, ``C`` ranks coding configurations
    offline on calibration clips.
    """
    direction = (-lifted_residual).flatten(1)
    correction = (target_latent - latent).flatten(1)
    return torch.nn.functional.cosine_similarity(direction, correction, dim=1)


def psnr(estimate: torch.Tensor, reference: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB per sample."""
    mse = (estimate - reference).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(peak**2 / mse.clamp_min(1e-12))
