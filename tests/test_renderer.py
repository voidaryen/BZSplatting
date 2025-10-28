import torch

from bzsplatting import BezierPath, BezierSplatRenderer, RendererConfig


def _simple_path(require_grad: bool = False) -> BezierPath:
    control_points = torch.tensor(
        [
            [[0.2, 0.2], [0.4, 0.0], [0.6, 0.4], [0.8, 0.2]],
        ],
        dtype=torch.float32,
        requires_grad=require_grad,
    )
    width = torch.tensor([0.05], dtype=torch.float32)
    color = torch.tensor([[0.9, 0.2, 0.2, 1.0]], dtype=torch.float32)
    return BezierPath.from_control_points(control_points=control_points, width=width, color=color)


def test_renderer_output_shape_and_range() -> None:
    renderer = BezierSplatRenderer(
        RendererConfig(canvas_width=64, canvas_height=64, curve_samples=32)
    )
    path = _simple_path()
    image = renderer.render(path)
    assert image.shape == (64, 64, 3)
    assert torch.all(image >= 0.0)
    assert torch.all(image <= 1.0 + 1e-4)


def test_renderer_supports_backpropagation() -> None:
    renderer = BezierSplatRenderer(
        RendererConfig(canvas_width=32, canvas_height=32, curve_samples=16)
    )
    path = _simple_path(require_grad=True)
    image = renderer.render(path)
    loss = image.mean()
    loss.backward()
    grads = path.control_points.grad
    assert grads is not None
    assert torch.isfinite(grads).all()
