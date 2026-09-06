---
title: "CUDA 与设备 — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# CUDA 与设备 — demo.py 代码详解

<a href="/notebook/code/programming/cuda/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/programming/cuda/code
python demo.py
```

探测 `cuda` / `mps` / `cpu`，把一张量 `.to(device)` 再搬回 CPU。有 NVIDIA GPU 时对 512 矩阵乘计一次时（`synchronize` 后再停表）。无 GPU 只画 CPU 柱。图 `prog-cuda-device.png`。

## 源码位置

`docs/programming/cuda/code/demo.py`

不要写死 `.cuda()`：用 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`。
