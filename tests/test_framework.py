import torch

from mgvflow import SinglePixelOperator, SlidingWindowCamera, directional_fidelity, lifted_residual_image, output_frame_rate


def _operator(frames=5, m=64, h=12, w=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return SinglePixelOperator((torch.rand(frames, m, h, w, generator=generator) > 0.5).float())


def test_residual_image_is_gradient_of_centered_loss():
    operator = _operator()
    truth = torch.rand(2, 5, 12, 20)
    estimate = torch.rand(2, 5, 12, 20, requires_grad=True)
    y = operator.measure(truth)
    residual = operator.measure(estimate) - y
    centered = residual - residual.mean(dim=-1, keepdim=True)
    loss = 0.5 * centered.square().mean(dim=(1, 2)).sum()
    (gradient,) = torch.autograd.grad(loss, estimate)
    assert torch.allclose(operator.residual_image(estimate.detach(), y), gradient, atol=1e-6)


def test_lifted_residual_image_matches_chain_rule():
    operator = _operator()
    weight = torch.randn(4, 5 * 12 * 20)
    decoder = lambda z: (z.flatten(1) @ weight).view(-1, 5, 12, 20)
    z = torch.randn(2, 4)
    y = operator.measure(torch.rand(2, 5, 12, 20))
    g_x = operator.residual_image(decoder(z), y)
    expected = g_x.flatten(1) @ weight.T
    assert torch.allclose(lifted_residual_image(decoder, z, operator, y), expected, atol=1e-5)


def test_directional_fidelity_is_one_along_true_correction():
    z, target = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(directional_fidelity(z - target, z, target), torch.ones(3), atol=1e-6)


def test_sliding_window_emits_one_block_per_five_frames():
    calls = []

    def reconstruct(history, window):
        calls.append(window.shape)
        return history + 1

    camera = SlidingWindowCamera(reconstruct, history=torch.zeros(1, 5, 4, 4), measurements_per_frame=10)
    blocks = list(camera.run(torch.rand(1, 25) for _ in range(12)))
    assert len(blocks) == 2 and calls[0] == (1, 5, 10)
    assert torch.equal(blocks[-1], torch.full((1, 5, 4, 4), 2.0))
    assert abs(output_frame_rate(1.552) - 3.22) < 0.01
