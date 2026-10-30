import math

import torch

from mgvflow import SinglePixelOperator, build_physics_condition, sample_euler
from mgvflow.models import MGVFlow, MGVFlowConfig


def test_default_network_matches_the_paper_parameter_count():
    assert sum(p.numel() for p in MGVFlow().parameters()) == 107_515_898


def test_velocity_has_the_latent_shape_at_480p():
    model = MGVFlow(MGVFlowConfig(base_width=32, attention_heads=4, residual_blocks_per_scale=1)).eval()
    z = torch.randn(1, 16, 2, 60, 107)
    with torch.no_grad():
        velocity = model(z, torch.randn_like(z), torch.zeros_like(z), torch.tensor([0.5]), torch.zeros(1, 5))
    assert velocity.shape == z.shape


def test_physics_condition_is_unit_rms_gated_direction_and_quality():
    torch.manual_seed(0)
    operator = SinglePixelOperator((torch.rand(5, 40, 6, 8) > 0.5).float())
    weight = torch.randn(16 * 2 * 3 * 4, 5 * 6 * 8)
    decoder = lambda z: (z.flatten(1) @ weight).view(-1, 5, 6, 8)
    latent = torch.randn(2, 16, 2, 3, 4)
    y = operator.measure(torch.rand(2, 5, 6, 8))
    tau = torch.tensor([0.1, 1.0])
    condition = build_physics_condition(decoder, latent, tau, operator, y)
    rms = condition.direction.square().flatten(1).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.tensor([0.0, 1.0]), atol=1e-5)
    assert condition.quality.shape == (2, 5)
    assert torch.allclose(condition.quality[:, 0], torch.full((2,), math.log(40 / 1000)))
    assert torch.allclose(condition.quality[:, 4], torch.tensor([0.0, 1.0]))
    residual = operator.measure(decoder(latent)) - y
    centered = residual - residual.mean(dim=-1, keepdim=True)
    assert torch.allclose(condition.quality[:, 2], centered.square().mean(dim=(1, 2)).sqrt().log(), atol=1e-5)


def test_euler_sampler_runs_the_network_with_the_condition():
    torch.manual_seed(0)
    model = MGVFlow(MGVFlowConfig(base_width=32, attention_heads=4, residual_blocks_per_scale=1)).eval()
    operator = SinglePixelOperator((torch.rand(5, 20, 8, 8) > 0.5).float())
    weight = torch.randn(16 * 2 * 8 * 8, 5 * 8 * 8) * 0.01
    decoder = lambda z: (z.flatten(1) @ weight).view(-1, 5, 8, 8)
    history = torch.randn(1, 16, 2, 8, 8)
    y = operator.measure(torch.rand(1, 5, 8, 8))
    velocity = lambda z, tau, c: model(z, history, c.direction, tau, c.quality)
    condition = lambda z, tau: build_physics_condition(decoder, z, tau, operator, y)
    out = sample_euler(velocity, condition, torch.randn(1, 16, 2, 8, 8), steps=5)
    assert out.shape == (1, 16, 2, 8, 8) and torch.isfinite(out).all()
