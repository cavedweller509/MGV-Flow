"""MGV-Flow: learning to read residual images for single-pixel photography."""

from .fidelity import directional_fidelity, psnr
from .physics import PhysicsCondition, SinglePixelOperator, build_physics_condition, lifted_residual_image
from .sampling import sample_euler
from .streaming import SlidingWindowCamera, output_frame_rate

__version__ = "0.1.0"

__all__ = [
    "PhysicsCondition",
    "SinglePixelOperator",
    "SlidingWindowCamera",
    "build_physics_condition",
    "directional_fidelity",
    "lifted_residual_image",
    "output_frame_rate",
    "psnr",
    "sample_euler",
]
