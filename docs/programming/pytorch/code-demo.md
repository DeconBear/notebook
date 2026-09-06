---
title: "PyTorch 张量与自动求导 — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# PyTorch 张量与自动求导 — demo.py 代码详解

<a href="/notebook/code/programming/pytorch/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/programming/pytorch/code
python demo.py
```

三件事：`@` 与 `cat` 的形状；\(L=(x^2+3x)^2\) 在 \(x=2\) 的梯度应接近 140；手写 \(w\) 做 80 步 SGD，图 `prog-pytorch-sgd.png`。

## 源码位置

`docs/programming/pytorch/code/demo.py`

## 为什么先不写 nn.Module

优化器吃的是「带 `requires_grad` 的张量列表」。`nn.Linear` 只是把这些张量收进 `parameters()`。骨架仍是 `zero_grad → 前向 → backward → step`。
