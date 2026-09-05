---
title: "扩散模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# 扩散模型 — demo.py 代码详解

<a href="/notebook/code/world-models/video/diffusion/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/diffusion/code
python demo.py
```

一维双峰混合上的微型 DDPM，不是 U-Net。

## 代码逐段详解

```python
self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
xt = sqrt_ab * x0 + sqrt_om * noise
```

闭式 $q(x_t|x_0)$，不必逐步加噪。`t` 是整数下标，用它去取 $\sqrt{\bar\alpha_t}$。`unsqueeze(-1)` 让形状从 `(N,)` 变成 `(N,1)` 才能乘 `(N,1)` 的 $x$。

损失：`((pred - noise)**2).mean()` — 预测的是 $\varepsilon$ 不是 $x_0$。

反向：`reversed(range(T))` 从 $T-1$ 走到 0。$t>0$ 时加 $\sqrt{\beta}\,\epsilon$，最后一步不再加噪（否则终态又糊）。

`@torch.no_grad()` 装饰采样循环：不建图。

## 源码位置

`docs/world-models/video/diffusion/code/demo.py`
