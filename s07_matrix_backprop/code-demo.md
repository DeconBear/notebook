---
title: "s07 多层网络的矩阵反向传播 — demo.py"
---

# s07 多层网络的矩阵反向传播 — demo.py

<a href="../code/s07_matrix_backprop/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

用纯NumPy实现完整的多层感知机（正向传播 + 矩阵反向传播 + 参数更新），包括梯度检查（有限差分验证）、梯度范数监控、训练前后决策边界对比和权重热力图可视化。在自定义的双月形二分类数据集上训练，展示了矩阵形式反向传播中Delta递推公式和权重梯度外积的核心机制。

## Run It

```bash
cd s07_matrix_backprop/code
python demo.py
```

## Key Output

- 数据集：300个双月形样本，2分类，带噪声(noise=0.15)
- 网络结构：2输入 -> 16隐藏(ReLU) -> 8隐藏(ReLU) -> 1输出(Sigmoid)，共 203 个参数
- 梯度检查：有限差分验证通过，各参数矩阵相对误差 < 1e-5
- 训练：lr=0.5, 2000 epochs，最终训练准确率约 95-98%

<<< @/snippets/s07_matrix_backprop/demo.py
