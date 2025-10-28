"""Differentiable Bézier splatting reference implementation. / 可微贝塞尔溅射参考实现。

This package provides a pedagogical reproduction of the renderer from
"Bézier Splatting for Fast and Differentiable Vector Graphics Rendering". /
本包复现了论文“Bézier Splatting for Fast and Differentiable Vector Graphics Rendering”中的渲染器，强调教学友好性。
The code stays small and well documented to ease reading and modification for experiments or coursework. /
代码保持精炼且注释充足，便于在科研实验或课程项目中阅读与修改。
"""

from .bezier import CubicBezier
from .splatting import BezierPath, BezierSplatRenderer, RendererConfig

__all__ = [
    "CubicBezier",
    "BezierPath",
    "BezierSplatRenderer",
    "RendererConfig",
]
