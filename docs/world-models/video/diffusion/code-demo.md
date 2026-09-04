---
title: "扩散模型 — demo.py"
---

# 扩散模型 — demo.py

<a href="/notebook/code/world-models/video/diffusion/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/diffusion/code
python demo.py
```

在一维双峰混合上训练微型 DDPM：`q_sample` 闭式加噪，MLP 预测 $\epsilon$，再逐步反向采样。前向直方图应越来越像高斯，反向终态应重新出现两个峰。

<<< @/world-models/video/diffusion/code/demo.py
