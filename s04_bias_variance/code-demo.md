---
title: "s04 偏差-方差权衡 — demo.py"
---

# s04 偏差-方差权衡 — demo.py

<a href="../code/s04_bias_variance/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

通过多项式回归拟合正弦波数据（y = sin(2*pi*x) + noise），全面展示欠拟合/拟合良好/过拟合的直观对比、Bias-Variance U形曲线、L1和L2正则化的实现与效果、K-Fold交叉验证、以及回归系数随正则化强度变化的路径图。

## Run It

```bash
cd s04_bias_variance/code
python demo.py
```

## Key Output

- 数据集：80个样本，y = sin(2*pi*x) + N(0, 0.3^2)，70%训练/30%验证
- 多项式拟合：1次严重欠拟合（训练MSE > 0.3），3-7次拟合良好，12次以上严重过拟合
- Bias-Variance曲线：最优复杂度约 4-6 次多项式，验证误差先降后升形成经典U形
- 正则化效果：对15次多项式（严重过拟合），L2(λ=0.01)和L1(λ=0.01)都能大幅降低过拟合

<<< @/snippets/s04_bias_variance/demo.py
