"""Utilities for working with Bézier curves. / 用于处理贝塞尔曲线的工具。

The paper *Bézier Splatting for Fast and Differentiable Vector Graphics Rendering*
represents 2D vector geometry through cubic Bézier curves. / 论文 *Bézier Splatting for Fast and Differentiable Vector Graphics Rendering* 使用三次贝塞尔曲线来描述二维矢量几何。
This module implements a minimal subset of the mathematics required for the reproducer:
evaluating curve positions, computing tangents and normals, and sampling points along the curve. /
本模块实现了复现所需的最小数学集合：计算曲线上各点的位置、切向量和法向量，并在曲线上进行采样。
All operations are implemented with PyTorch tensors so that they participate in automatic
\:math:`\nabla` differentiation, allowing end-to-end optimisation of vector graphics parameters. /
所有操作均以 PyTorch 张量实现，以便参与自动 \:math:`\nabla` 求导，从而实现矢量图形参数的端到端优化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

Tensor = torch.Tensor


@dataclass
class CubicBezier:
    """Light-weight container for batches of cubic Bézier control points. / 用于批量存储三次贝塞尔控制点的轻量级容器。

    The control points are stored in the order ``(p0, p1, p2, p3)`` where each entry is a 2D vector.
    / 控制点按照 ``(p0, p1, p2, p3)`` 的顺序存储，每个元素都是二维向量。
    The tensor layout follows the PyTorch convention ``(..., 4, 2)`` allowing arbitrary batch dimensions.
    / 张量布局遵循 PyTorch 的 ``(..., 4, 2)`` 约定，可灵活支持任意批次维度。
    Using a dataclass keeps the implementation close to the notation in the paper and displays nicely in notebooks.
    / 采用 dataclass 既贴合论文记号，又能在交互式环境中获得友好的显示效果。
    """

    control_points: Tensor

    def __post_init__(self) -> None:
        if self.control_points.shape[-2:] != (4, 2):
            raise ValueError(
                "CubicBezier.control_points must have shape (..., 4, 2). "
                f"Received {tuple(self.control_points.shape)}"
            )

    @property
    def device(self) -> torch.device:
        return self.control_points.device

    @property
    def dtype(self) -> torch.dtype:
        return self.control_points.dtype

    def evaluate(self, t: Tensor) -> Tensor:
        """Evaluate positions along the curve for parameter ``t``. / 计算参数 ``t`` 对应的曲线上位置。

        Parameters
        ----------
        t:
            A tensor of shape ``(...,)`` with values in ``[0, 1]`` supporting broadcasting.
            / 形状为 ``(...,)`` 且数值范围在 ``[0, 1]`` 内的张量，支持与控制点批次维广播。
        """

        t = t.unsqueeze(-1)  # (..., 1) shape / 张量形状 (..., 1)
        u = 1.0 - t
        cp = self.control_points
        return (
            (u ** 3) * cp[..., 0, :]  # type: ignore[index]
            + 3.0 * (u ** 2) * t * cp[..., 1, :]
            + 3.0 * u * (t ** 2) * cp[..., 2, :]
            + (t ** 3) * cp[..., 3, :]
        )

    def tangent(self, t: Tensor) -> Tensor:
        """Compute the first derivative (tangent) with respect to ``t``. / 计算关于 ``t`` 的一阶导数（切向量）。"""

        t = t.unsqueeze(-1)
        u = 1.0 - t
        cp = self.control_points
        return (
            3.0 * (u ** 2) * (cp[..., 1, :] - cp[..., 0, :])
            + 6.0 * u * t * (cp[..., 2, :] - cp[..., 1, :])
            + 3.0 * (t ** 2) * (cp[..., 3, :] - cp[..., 2, :])
        )

    def speed(self, t: Tensor) -> Tensor:
        """Return the magnitude of the tangent vector for parameter ``t``. / 返回参数 ``t`` 对应切向量的模长。"""

        return torch.linalg.norm(self.tangent(t), dim=-1)

    def sample(self, num_samples: int, *, include_endpoints: bool = True) -> Tuple[Tensor, Tensor]:
        """Sample positions and approximate arc-length weights. / 采样曲线位置及相应的近似弧长权重。

        The function returns curve positions together with an approximate arc-length weight for each sample.
        / 函数返回曲线位置以及每个采样点的近似弧长权重。
        These weights keep the contribution of a curve stable regardless of sampling density.
        / 这些权重可确保无论采样多密集，曲线对最终图像的贡献都保持稳定。
        """

        if num_samples < 2:
            raise ValueError("num_samples must be at least 2 to estimate arc length")

        if include_endpoints:
            t_values = torch.linspace(0.0, 1.0, num_samples, device=self.device, dtype=self.dtype)
        else:
            # Offset by half a step so samples align with segment centres. / 将采样点偏移半步，使其位于线段中心。
            step = 1.0 / num_samples
            t_values = torch.linspace(
                step / 2.0, 1.0 - step / 2.0, num_samples, device=self.device, dtype=self.dtype
            )

        positions = self.evaluate(t_values)
        # Estimate local arc length using forward differences. / 使用前向差分估计局部弧长。
        # We pad the first segment so counts match the positions tensor. / 通过填充首段保持张量长度一致。
        deltas = positions[..., 1:, :] - positions[..., :-1, :]
        lengths = torch.linalg.norm(deltas, dim=-1)
        lengths = torch.nn.functional.pad(lengths, (1, 0), mode="replicate")
        return positions, lengths


__all__ = ["CubicBezier"]
