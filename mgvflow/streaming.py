"""Sliding-window single-pixel photography.

The detector delivers bucket values as a continuous stream.  Every five frames the
camera takes the most recent ``M`` bucket values per frame, reconstructs the next
five-frame block with the five preceding reconstructed frames as history, and appends
it to the output.  The latency of one block is the only delay between acquisition and
display, so the sustained frame rate is ``5 / latency``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import torch

BLOCK = 5

# reconstruct(history [B, 5, H, W], bucket values [B, 5, M]) -> next block [B, 5, H, W]
BlockReconstructor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class SlidingWindowCamera:
    reconstruct: BlockReconstructor
    history: torch.Tensor                       # [B, 5, H, W], bootstrap frames
    measurements_per_frame: int = 1_000
    _buffer: deque = field(default_factory=deque, init=False)

    def push(self, bucket_values: torch.Tensor) -> torch.Tensor | None:
        """Add the bucket values ``[B, M']`` of one frame and return a block when five frames are ready."""
        self._buffer.append(bucket_values[..., -self.measurements_per_frame:])
        if len(self._buffer) < BLOCK:
            return None
        window = torch.stack([self._buffer.popleft() for _ in range(BLOCK)], dim=1)
        block = self.reconstruct(self.history, window)
        self.history = block
        return block

    def run(self, stream: Iterable[torch.Tensor]) -> Iterator[torch.Tensor]:
        for frame_values in stream:
            block = self.push(frame_values)
            if block is not None:
                yield block


def output_frame_rate(block_latency_seconds: float) -> float:
    """Sustained output frame rate of the sliding window for a given block latency."""
    return BLOCK / block_latency_seconds
