---
title: "s02 线性回归 — demo.py"
---

# s02 线性回归 — demo.py

<a href="../code/s02_linear_regression/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现线性回归，对比梯度下降法和正规方程两种求解方式，并与 sklearn 的 LinearRegression 结果互相验证。演示了损失函数的几何形状、梯度下降的优化轨迹、以及不同学习率对收敛速度的影响。

## Run It

```bash
cd s02_linear_regression/code
python demo.py
```

## Key Output

- 数据集：100 个样本，真实函数 y = 2x + 5 + N(0, 3^2)
- 梯度下降法：经约 300 轮收敛，最终 MSE < 10，R^2 ≈ 0.85
- 正规方程解：w_ne, b_ne 接近真实值 (2, 5)，MSE 与梯度下降法一致
- sklearn 解：与正规方程解几乎完全一致（验证了实现的正确性）

<<< @/snippets/s02_linear_regression/demo.py
