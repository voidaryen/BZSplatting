"""Torch implementation of Bézier splatting. / 基于 Torch 的贝塞尔溅射实现。

The goal of splatting is to transform a sparse vector representation (Bézier curves with
widths and colours) into a dense raster image while keeping the process differentiable. /
溅射的目标是在保持可微性的同时，将稀疏的矢量表示（带宽度与颜色的贝塞尔曲线）转换为稠密的光栅图像。
This implementation mirrors the intuition from the paper: sample curves in parameter space,
convert them into Gaussian "splats" in screen space, and accumulate them on a pixel grid. /
本实现遵循论文思路：在参数空间对曲线采样，将样本转换为屏幕空间的高斯“斑点”，再在像素网格上累加。
The code stays compact and well-commented so it can be studied easily rather than used as production renderer. /
代码保持紧凑并配有详细注释，旨在提供学习资源，而非直接投入生产环境。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .bezier import CubicBezier

Tensor = torch.Tensor


@dataclass
class BezierPath:
    """Container describing a coloured Bézier curve. / 描述带颜色贝塞尔曲线的容器。"""

    curve: CubicBezier  # Underlying cubic geometry / 底层三次贝塞尔曲线
    width: Tensor  # (...,) shape / 张量形状 (...,)
    color: Tensor  # (..., 4) RGBA in [0, 1] / RGBA 分量范围 [0, 1]

    def __post_init__(self) -> None:
        """Ensure tensors match the Bézier batch layout. / 确认张量与贝塞尔批次布局一致。"""

        batch_shape = self.curve.control_points.shape[:-2]
        expected_width_shape = batch_shape if batch_shape else (1,)
        expected_color_shape = (*batch_shape, 4) if batch_shape else (1, 4)

        if self.width.shape != expected_width_shape:
            raise ValueError(
                "BezierPath.width must match the Bézier batch shape. / BezierPath.width 的形状必须与贝塞尔批次形状一致。"
            )
        if self.color.shape != expected_color_shape:
            raise ValueError(
                "BezierPath.color must be (..., 4) to align with the batch. / BezierPath.color 必须为 (..., 4) 以匹配批次形状。"
            )

    @classmethod
    def from_control_points(cls, control_points: Tensor, width: Tensor, color: Tensor) -> "BezierPath":
        """Build a path from raw control points. / 使用原始控制点构建路径。"""

        return cls(curve=CubicBezier(control_points), width=width, color=color)

    def as_bezier(self) -> CubicBezier:
        """Expose the cubic curve for downstream modules. / 为下游模块提供三次曲线对象。"""

        return self.curve

    @property
    def control_points(self) -> Tensor:
        """Convenience accessor mirroring the old API. / 兼容旧版 API 的控制点访问器。"""

        return self.curve.control_points


def _prepare_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """Create ``(H, W)`` mesh grids measured in pixel coordinates. / 生成以像素坐标计量的 ``(H, W)`` 网格。"""

    y = torch.linspace(0.0, height - 1, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, width - 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


@dataclass
class RendererConfig:
    """Configuration options shared across render calls. / 跨渲染调用共享的配置选项。"""

    canvas_width: int = 256
    canvas_height: int = 256
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    curve_samples: int = 64
    blur_radius: float = 1.0  # Gaussian sigma in curve-width units / 高斯标准差以曲线宽度计
    def validate(self) -> None:
        if self.canvas_height <= 0 or self.canvas_width <= 0:
            raise ValueError("Canvas dimensions must be positive integers")
        if self.curve_samples < 2:
            raise ValueError("curve_samples must be at least 2 to approximate arc length")
        if self.blur_radius <= 0:
            raise ValueError("blur_radius must be positive")


class BezierSplatRenderer:
    """Differentiable renderer implementing the Bézier splatting idea. / 实现贝塞尔溅射思想的可微渲染器。"""

    def __init__(self, config: RendererConfig | None = None, *, device: Optional[torch.device] = None):
        self.config = config or RendererConfig()
        self.config.validate()
        self.device = device or torch.device("cpu")

    def render(self, path: BezierPath) -> Tensor:
        """Render a single ``BezierPath`` to an ``(H, W, 3)`` tensor. / 将单个 ``BezierPath`` 渲染为 ``(H, W, 3)`` 张量。

        The renderer assumes control points use **normalised** device coordinates: ``(0, 0)`` is the top-left corner,
        ``(1, 1)`` is the bottom-right. / 渲染器假设控制点使用**归一化**设备坐标：``(0, 0)`` 对应画布左上角，``(1, 1)`` 对应右下角。
        Width values use the same units so a vector graphic can be optimised without fixing an output resolution.
        / 宽度同样采用该单位，使矢量图在无需预设输出分辨率的情况下即可优化。
        """

        cfg = self.config

        # Convert inputs to tensors on the correct device. / 将输入张量移动到目标设备。
        curve = path.as_bezier()
        control_points = curve.control_points.to(self.device)
        width = path.width.to(self.device)
        color = path.color.to(self.device)
        background = torch.tensor(cfg.background, device=self.device, dtype=color.dtype)

        if control_points.dim() == 2:
            control_points = control_points.unsqueeze(0)
            width = width.unsqueeze(0)
            color = color.unsqueeze(0)

        curve_on_device = CubicBezier(control_points)
        positions, lengths = curve_on_device.sample(cfg.curve_samples)

        # Convert to pixel coordinates. / 将坐标转换为像素空间。
        # Width scales so one unit equals the smaller canvas side, keeping strokes consistent. /
        # 宽度按画布较短边缩放，使不同纵横比的画布上笔划视觉一致。
        scale = torch.tensor([cfg.canvas_width - 1, cfg.canvas_height - 1], device=self.device, dtype=positions.dtype)
        positions_px = positions * scale
        width_px = width * min(cfg.canvas_width, cfg.canvas_height)

        # Each sample becomes a Gaussian splat with sigma proportional to curve width. /
        # 每个采样点转换为高斯斑点，标准差与曲线宽度成正比。
        sigma = torch.clamp(width_px, min=1e-6) * cfg.blur_radius
        sigma = sigma.unsqueeze(-1)  # (..., 1) shape / 张量形状 (..., 1)

        grid_x, grid_y = _prepare_grid(cfg.canvas_height, cfg.canvas_width, self.device, positions_px.dtype)

        # Flatten batch dimensions for easier broadcasting while keeping the sample axis. /
        # 扁平化批次维度以便广播，同时保留采样维。
        flat_pos = positions_px.reshape(-1, cfg.curve_samples, 2)
        flat_len = lengths.reshape(-1, cfg.curve_samples)
        flat_sigma = sigma.reshape(-1, 1)
        flat_width = width_px.reshape(-1, 1)
        flat_color = color.reshape(-1, color.shape[-1])

        # Compute Gaussian weights for each sample across all curves. /
        # 对所有曲线与采样点矢量化计算高斯权重。
        dx = flat_pos[..., 0].unsqueeze(-1).unsqueeze(-1) - grid_x  # (N, S, H, W) shape / 张量形状 (N, S, H, W)
        dy = flat_pos[..., 1].unsqueeze(-1).unsqueeze(-1) - grid_y
        inv_sigma2 = 1.0 / torch.clamp(flat_sigma ** 2, min=1e-6)
        gaussians = torch.exp(-0.5 * (dx**2 + dy**2) * inv_sigma2.unsqueeze(-1).unsqueeze(-1))
        normalisation = (inv_sigma2 / (2.0 * torch.pi)).unsqueeze(-1).unsqueeze(-1)
        gaussians = gaussians * normalisation

        # Weight by approximate local arc length and stroke width; epsilon avoids degeneracy. /
        # 依据近似弧长和笔划宽度加权，微小 epsilon 避免退化数值问题。
        sample_weight = flat_len / (cfg.curve_samples - 1 + 1e-6)
        sample_weight = sample_weight * torch.clamp(flat_width, min=1e-6)
        gaussians = gaussians * sample_weight.unsqueeze(-1).unsqueeze(-1)

        # Alpha controls coverage; we use premultiplied alpha for stability as in the paper. /
        # Alpha 控制覆盖度；与论文一致，我们使用预乘 Alpha 以提高稳定性。
        alpha = flat_color[..., 3].clamp(0.0, 1.0)
        rgb = flat_color[..., :3].clamp(0.0, 1.0)

        alpha_map = torch.einsum("nshw,n->hw", gaussians, alpha)
        color_map = torch.einsum("nshw,nc->hwc", gaussians, rgb * alpha.unsqueeze(-1))

        # Composite splats over the background to keep gradients healthy and avoid normalisation. /
        # 将溅射结果覆盖在背景上，以保持良好梯度并避免额外的 Alpha 归一化。
        alpha_map = alpha_map.clamp(0.0, 1.0)
        color_map = color_map.clamp(0.0, 1.0)
        image = color_map + background * (1.0 - alpha_map).unsqueeze(-1)
        return image


__all__ = ["BezierPath", "BezierSplatRenderer", "RendererConfig"]
