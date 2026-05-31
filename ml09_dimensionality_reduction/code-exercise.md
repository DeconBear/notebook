---
title: "ml09 降维与特征工程 — exercise.py"
---

# ml09 降维与特征工程 — exercise.py 练习指南

<a href="../code/ml09_dimensionality_reduction/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现 PCA 的协方差矩阵方法、LDA 的散布矩阵计算、等宽分箱和交互特征生成，从代码层面深入理解降维和特征工程的数学原理。

## 预备知识

- 协方差矩阵 $\mathbf{S} = \frac{1}{n-1}\tilde{\mathbf{X}}^T \tilde{\mathbf{X}}$ 的特征向量 = 主成分方向
- SVD 与特征值分解的等价关系：$\tilde{\mathbf{X}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$，$\mathbf{V}$ 的列是 $\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$ 的特征向量
- 类内散布矩阵 $\mathbf{S}_W$ 和类间散布矩阵 $\mathbf{S}_B$ 的定义
- 分箱将连续特征离散化，赋予线性模型非线性能力
- 交互特征 $x_i \times x_j$ 捕获特征间的联合效应

## 任务清单

### 任务1：实现 PCA 的协方差矩阵 + 特征值分解 `pca_eigen(X, n_components)`

- **与 SVD 方法的区别**：直接构建协方差矩阵 $\mathbf{S}_{d \times d}$，然后做特征分解
- **适用场景**：当 $d$ 不太大时，协方差矩阵方法更直观；当 $d$ 很大时（如图像像素），SVD 更高效
- **实现步骤**：
  1. 数据中心化：`X_c = X - X.mean(axis=0)`
  2. 协方差矩阵：`S = (X_c.T @ X_c) / (n - 1)`
  3. 特征分解：`np.linalg.eigh(S)`（返回升序排列的特征值）
  4. 取最大的 k 个：`np.argsort(eigenvalues)[::-1][:k]`
  5. 投影：`X_c @ components.T`
- **需要调用的函数**：`np.linalg.eigh()`, `np.argsort()`, `np.mean()`
- **注意**：`eigh` 返回按**升序**排列，所以最大的特征向量在最后几列

### 任务2：计算 LDA 的散布矩阵 `compute_scatter_matrices(X, y)`

- **公式**：
  - $\mathbf{S}_W = \sum_{c} \sum_{i:y_i=c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^T$
  - $\mathbf{S}_B = \sum_{c} N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$
- **实现提示**：
  - 遍历每个类别 `classes = np.unique(y)`
  - 用布尔索引获取类内样本：`X_c = X[y == c]`
  - $\mathbf{S}_W$ 的高效计算：`(X_c - mu_c).T @ (X_c - mu_c)`
  - $\mathbf{S}_B$ 需要外积：`np.outer(diff, diff)` 或 `diff.reshape(-1,1) @ diff.reshape(1,-1)`
- **需要调用的函数**：`np.unique()`, `np.mean()`, `np.outer()` 或 `@`
- **验证**：$\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B$ 成立（总散布 = 类内 + 类间）

### 任务3：等宽分箱 `equal_width_binning(x, n_bins)`

- **算法步骤**：
  1. 箱边界：`bins = np.linspace(x_min, x_max, n_bins + 1)`
  2. 分配：`bin_labels = np.digitize(x, bins[:-1]) - 1`
  3. One-Hot：`one_hot[np.arange(len(x)), bin_labels] = 1`
- **需要调用的函数**：`np.linspace()`, `np.digitize()` 或 `np.searchsorted()`
- **验证**：每个箱子中的样本数应大致相等（对于均匀分布的数据）

### 任务4：交互特征生成 `generate_interaction_features(X, max_degree)`

- **算法步骤**：
  1. 从偏置列（全 1）开始
  2. 对 degree = 1, 2, ..., max_degree，生成所有 degree 个特征的乘积组合
  3. 使用 `itertools.combinations_with_replacement` 生成组合索引
  4. 对每个组合 `combo`，计算 `np.prod(X[:, combo], axis=1)`
- **需要调用的函数**：`itertools.combinations_with_replacement()`, `np.prod()`, `np.column_stack()`
- **注意**：特征数量随 degree 和 d 快速增长——对于 d=10, degree=2，有 $1 + 10 + 55 = 66$ 个特征

## 完整代码

<<< @/snippets/ml09_dimensionality_reduction/exercise.py
