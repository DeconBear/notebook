---
title: "s01 AI概述 — demo.py"
---

# s01 AI概述 — demo.py

<a href="../code/s01_ai_overview/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零使用纯NumPy实现感知机（Perceptron）二分类器。生成线性可分的二维合成数据，通过感知机学习算法迭代更新权重和偏置，最后展示决策边界和训练收敛过程。这是深度学习最基本的构建模块演示。

## Run It

```bash
cd s01_ai_overview/code
python demo.py
```

## Key Output

- 数据集：200个样本（每类100个），2个特征，线性可分
- 收敛情况：感知机通常在 5-15 轮内收敛（数据线性可分时保证收敛）
- 训练集准确率：100%（线性可分数据）

<<< @/snippets/s01_ai_overview/demo.py
