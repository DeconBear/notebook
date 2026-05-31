---
title: "ml14 核方法与高斯过程 — exercise.py"
---

# ml14 核方法与高斯过程 — exercise.py 练习指南

<a href="../code/ml14_kernel_gp/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现 RBF 核函数、GP 回归的预测均值和方差、以及 GP 先验采样，从代码层面深入理解核方法和 GP 的数学原理。

## 预备知识

- RBF 核：$k(x, x') = \sigma^2 \exp(-\|x-x'\|^2 / 2\ell^2)$
- Cholesky 分解：$\mathbf{K} = \mathbf{L}\mathbf{L}^T$，用于数值稳定解线性方程组
- GP 预测均值：$\bar{f}^* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{y}$
- GP 预测方差：$\text{Var}(f^*) = k(x^*, x^*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{k}_*$
- GP 先验采样：$\mathbf{f} = \mathbf{L}\mathbf{z}$，$\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$

## 任务清单

### 任务1：RBF 核函数 `rbf_kernel_manual(X1, X2, lengthscale, variance)`

- **平方欧氏距离**：$\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2x^T y$
- **向量化实现**：利用广播展开三个矩阵（$(n_1,1)$、$(1,n_2)$、$(n_1,n_2)$）
- **验证**：$k(x, x) = \sigma^2$（距离 0，核值 = 方差）；$k(0, 1) = \sigma^2 \exp(-0.5)$

### 任务2：GP 预测均值 `gp_predict_mean(X_train, y_train, X_test, kernel_fn, noise_var)`

- **关键步骤**：
  1. 计算 $\mathbf{K} = k(\mathbf{X}, \mathbf{X})$（$(N,N)$ 矩阵）
  2. Cholesky 分解 $\mathbf{K} + \sigma_n^2\mathbf{I} = \mathbf{L}\mathbf{L}^T$
  3. 解 $\boldsymbol{\alpha} = \mathbf{L}^{-T} \mathbf{L}^{-1} \mathbf{y}$（两次三角求解）
  4. 计算 $\mathbf{k}_* = k(\mathbf{X}_*, \mathbf{X})$（$(M,N)$ 矩阵）
  5. 预测 $\bar{\mathbf{f}}^* = \mathbf{k}_* \boldsymbol{\alpha}$
- **为什么用 Cholesky 而非直接求逆？**
  - 更稳定：Cholesky 是专门针对对称正定矩阵设计的高效分解
  - 更快：$O(N^3/6)$ vs $O(N^3)$

### 任务3：GP 预测方差 `gp_predict_variance(X_train, X_test, kernel_fn, L, noise_var)`

- **方差公式**：对每个测试点 $i$，
  - $\mathbf{v} = \mathbf{L}^{-1} \mathbf{k}_*$（解三角方程）
  - $\text{Var}_i = k(x_i^*, x_i^*) - \|\mathbf{v}\|^2$
- **不确定性行为**：训练数据附近 → $\|\mathbf{v}\|^2$ 大 → 方差小；远离数据 → $\|\mathbf{v}\|^2$ 小 → 方差大

### 任务4：GP 先验采样 `sample_gp_prior(X, kernel_fn, n_samples)`

- **采样方法**：$\mathbf{f} = \mathbf{L}\mathbf{z}$，$\mathbf{K} = \mathbf{L}\mathbf{L}^T$
- **直观理解**：$\mathbf{L}$ 本质上是对核矩阵进行"平方根"分解，然后通过 $\mathbf{L}\mathbf{z}$ 将独立高斯噪声 $\mathbf{z}$ 转换为具有 $\mathbf{K}$ 协方差结构的函数值

## 完整代码

<<< @/snippets/ml14_kernel_gp/exercise.py
