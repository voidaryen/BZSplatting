# BZSplatting / 基于贝塞尔溅射的矢量图形渲染

Reproduction of the paper **Bézier Splatting for Fast and Differentiable Vector Graphics Rendering** (2025). / 复现论文 **Bézier Splatting for Fast and Differentiable Vector Graphics Rendering**（2025）。
The goal of this repository is to provide a compact and well-documented PyTorch implementation that students and researchers can read in an afternoon. / 本仓库旨在提供一个紧凑且注释详尽的 PyTorch 实现，方便学习者和研究者在短时间内理解。
The core logic lives in `src/bzsplatting` and focuses on the differentiable rasterisation stage of the paper. / 核心逻辑位于 `src/bzsplatting` 目录，聚焦于论文中的可微分光栅化阶段。

## Features / 功能特点

- Cubic Bézier utilities implemented directly in PyTorch. / 直接使用 PyTorch 实现的三次贝塞尔工具函数。
- Differentiable Gaussian splatting renderer that maps curves to raster images. / 将曲线映射为光栅图像的可微高斯溅射渲染器。
- Example script (`examples/render_heart.py`) that renders a heart-shaped logo. / 示例脚本（`examples/render_heart.py`）用于渲染心形图案。
- Automated tests that check differentiability and numerical stability. / 自动化测试用于检查可微性与数值稳定性。

## Installation / 安装

Create and activate a dedicated Conda environment named `bzsplatting`, then install the package in editable mode. /
请先创建并激活名为 `bzsplatting` 的 Conda 虚拟环境，再以可编辑模式安装本项目。

```bash
conda create -n bzsplatting python=3.10
conda activate bzsplatting
pip install -e .
```

The renderer depends only on PyTorch and Pillow. / 渲染器仅依赖 PyTorch 与 Pillow。
Installing in editable mode makes it convenient to experiment with the source code. / 以可编辑模式安装便于直接实验与修改源码。

## Usage / 使用示例

```python
import torch
from bzsplatting import BezierPath, BezierSplatRenderer, RendererConfig

control_points = torch.tensor(
    [
        [[0.2, 0.2], [0.4, 0.0], [0.6, 0.4], [0.8, 0.2]],
    ],
    dtype=torch.float32,
)
width = torch.tensor([0.05], dtype=torch.float32)
color = torch.tensor([[0.9, 0.2, 0.2, 1.0]], dtype=torch.float32)

renderer = BezierSplatRenderer(RendererConfig(canvas_width=256, canvas_height=256))
image = renderer.render(BezierPath.from_control_points(control_points, width, color))
```

The snippet above shows how to render a single cubic Bézier curve. / 上述代码片段演示了如何渲染单条三次贝塞尔曲线。

To save an image run the example script: / 如需保存图像，请运行示例脚本：

```bash
python examples/render_heart.py
```

The script produces `examples/render_heart.png`. / 该脚本会生成 `examples/render_heart.png` 图像。

## Tests / 测试

```bash
pytest
```

The tests exercise the differentiable renderer and ensure gradients flow through the control points. / 测试用例用于验证可微渲染器，并确保梯度能正确传递至控制点。
