"""Single-pixel forward operator and the residual image.

A binary coding pattern ``P_m`` modulates frame ``x_t`` and the detector integrates the
result into one bucket value ``y_{t,m} = <P_m, x_t> / (H W)``.  Correlating the
frame-centered measurement residual with the same patterns gives the residual image
``g_x`` of the paper (Eq. 1), which is the exact gradient of the centered squared
measurement loss with respect to the estimated frames.
"""

from __future__ import annotations

import torch


class SinglePixelOperator:
    """Linear single-pixel operator for a clip ``[B, T, H, W]`` and per-frame patterns.

    ``patterns`` has shape ``[T, M, H, W]`` (one pattern set per frame) or ``[M, H, W]``
    (the same set shared by every frame).  Values are expected in ``{0, 1}``.
    """

    def __init__(self, patterns: torch.Tensor) -> None:
        if patterns.ndim not in (3, 4):
            raise ValueError("patterns must be [M,H,W] or [T,M,H,W]")
        self.patterns = patterns.float()

    @property
    def measurement_count(self) -> int:
        return int(self.patterns.shape[-3])

    def _per_frame(self, frames: int) -> torch.Tensor:
        if self.patterns.ndim == 3:
            return self.patterns.unsqueeze(0).expand(frames, -1, -1, -1)
        if self.patterns.shape[0] != frames:
            raise ValueError("pattern sets and frames differ in number")
        return self.patterns

    def measure(self, video: torch.Tensor) -> torch.Tensor:
        """Bucket values ``[B, T, M]`` of a clip ``[B, T, H, W]``."""
        if video.ndim != 4:
            raise ValueError("video must be [B,T,H,W]")
        batch, frames, height, width = video.shape
        patterns = self._per_frame(frames).to(video)
        return torch.einsum("bthw,tmhw->btm", video, patterns) / (height * width)

    def residual_image(self, estimate: torch.Tensor, bucket_values: torch.Tensor) -> torch.Tensor:
        """Residual image ``g_x`` of an estimate ``[B, T, H, W]`` given acquired values ``[B, T, M]``.

        ``g_x[t] = sum_m r~_{t,m} P_m / (T M H W)`` with the residual ``r = A x_hat - y`` centered
        over ``m`` within each frame.
        """
        batch, frames, height, width = estimate.shape
        residual = self.measure(estimate) - bucket_values.to(estimate)
        residual = residual - residual.mean(dim=-1, keepdim=True)
        patterns = self._per_frame(frames).to(estimate)
        scale = frames * self.measurement_count * height * width
        return torch.einsum("btm,tmhw->bthw", residual, patterns) / scale

    def correlation_image(self, bucket_values: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Scene correlation image of traditional correlation imaging (centered bucket values)."""
        frames = bucket_values.shape[1]
        centered = bucket_values - bucket_values.mean(dim=-1, keepdim=True)
        patterns = self._per_frame(frames).to(bucket_values)
        patterns = patterns - patterns.mean(dim=1, keepdim=True)
        return torch.einsum("btm,tmhw->bthw", centered, patterns) / self.measurement_count
