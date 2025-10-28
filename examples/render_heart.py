"""Render a simple heart shape using Bézier splatting. / 使用贝塞尔溅射渲染一个简单的心形。

Run the script with ``python examples/render_heart.py``; it saves a PNG in ``examples`` and prints instructions. /
使用 ``python examples/render_heart.py`` 运行脚本，会在 ``examples`` 目录保存 PNG 并打印提示。
The example highlights how easy it is to employ :class:`bzsplatting.splatting.BezierSplatRenderer` for differentiable vector graphics. /
该示例展示了使用 :class:`bzsplatting.splatting.BezierSplatRenderer` 进行可微矢量图形渲染的便捷性。
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from bzsplatting import BezierPath, BezierSplatRenderer, RendererConfig

OUTPUT_PATH = Path(__file__).with_suffix(".png")


def make_heart_path() -> BezierPath:
    # Two cubic Bézier curves forming the left and right halves of a heart. / 两条三次贝塞尔曲线构成心形的左右两半。
    # Control points are specified in normalised coordinates. / 控制点使用归一化坐标表示。
    control_points = torch.tensor(
        [
            [[0.5, 0.75], [0.1, 0.9], [0.0, 0.4], [0.5, 0.2]],
            [[0.5, 0.75], [0.9, 0.9], [1.0, 0.4], [0.5, 0.2]],
        ],
        dtype=torch.float32,
    )
    width = torch.tensor([0.05, 0.05], dtype=torch.float32)
    color = torch.tensor(
        [
            [0.9, 0.1, 0.2, 1.0],
            [0.9, 0.1, 0.2, 1.0],
        ],
        dtype=torch.float32,
    )
    return BezierPath(control_points=control_points, width=width, color=color)


def main() -> None:
    renderer = BezierSplatRenderer(
        RendererConfig(canvas_width=512, canvas_height=512, curve_samples=96)
    )
    image = renderer.render(make_heart_path())
    image_np = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(image_np).save(OUTPUT_PATH)
    print(f"Saved heart example to {OUTPUT_PATH}")  # 提示保存路径 / Notify where the output was saved


if __name__ == "__main__":
    main()
