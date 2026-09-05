---
title: "ml14 核方法与高斯过程 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml14 核方法与高斯过程 — demo.py 代码详解

<a href="/notebook/code/ml/advanced/kernel-gp/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/advanced/kernel-gp/code
python demo.py
```

## 代码逐段详解

### 第1步：RBF 核函数的向量化实现

```python
def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)  # (n1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)  # (1, n2)
    sq_dist = sq_norm1 + sq_norm2 - 2 * X1 @ X2.T    # (n1, n2)
    return variance * np.exp(-0.5 * sq_dist / (lengthscale**2))
```

这个向量化实现利用了平方欧氏距离的展开式：

$$
\|x - x'\|^2 = \|x\|^2 + \|x'\|^2 - 2 x^T x'
$$

通过 NumPy 广播，三个矩阵（$(n_1, 1)$、$(1, n_2)$、$(n_1, n_2)$）自动对齐计算出完整的 $(n_1, n_2)$ 距离矩阵。这比双重循环快 100-1000 倍。

### 第2步：核岭回归

```python
class KernelRidgeRegression:
    def fit(self, X, y):
        K = self.kernel(X, X, **self.kernel_params)      # (N, N)
        K_reg = K + self.alpha * np.eye(len(X))          # K + λI
        L = np.linalg.cholesky(K_reg)                     # Cholesky 分解
        self.alpha_coef_ = np.linalg.solve(L.T, np.linalg.solve(L, y))
```

核岭回归的解是 $\boldsymbol{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$。代码使用 Cholesky 分解 $\mathbf{K} + \lambda\mathbf{I} = \mathbf{L}\mathbf{L}^T$ 然后解两次三角方程组来替代直接求逆。Cholesky 分解只需 $O(N^3/6)$ 而非 $O(N^3)$，且数值稳定性更好。

注意：核岭回归只给出点估计，不提供任何关于预测不确定性的信息。

### 第3步：高斯过程回归

```python
class GaussianProcessRegressor:
    def predict(self, X_test):
        # 均值: f* = k_*^T (K + σ_n²I)^{-1} y
        mean = K_test @ self.K_inv_y_

        # 方差: var = k(x*,x*) - k_*^T (K + σ_n²I)^{-1} k_*
        for i in range(n_test):
            v = np.linalg.solve(self.L_, K_test[i, :])
            var[i] = k_self - v @ v
```

GP 的核心优势：同时给出预测均值和预测方差。均值公式与核岭回归完全一致，但 GP 还额外提供了：
- **噪声方差** $\sigma_n^2$：显式建模观测噪声（核岭回归的 $\lambda$ 在这里有了明确的概率含义）
- **预测方差**：$k(x^*, x^*) - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$

方差公式有两部分：
- $k(x^*, x^*)$：GP 先验对 $x^*$ 处的不确定性（总是正的）
- 减去 $\mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$：训练数据带来的**不确定性缩减**

**不确定性行为**：
- 在数据点处：$x^*$ 靠近 $x_i$，$\mathbf{k}_*$ 与 $\mathbf{K}$ 中的对应列高度相关 → 方差缩减大 → 置信带窄
- 远离数据点：$k(x^*, x_i)$ 都很小 → 方差缩减小 → 置信带宽 → 反映了模型的"不自知"

### 第4步：GP 先验采样

```python
L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
f_sample = L @ np.random.randn(len(X))
```

从 GP 先验中采样函数的方法：$\mathbf{f} = \mathbf{L}\mathbf{z}$，其中 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$，$\mathbf{K} = \mathbf{L}\mathbf{L}^T$。这利用了正态分布的线性变换性质：
- 如果 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$，则 $\mathbf{L}\mathbf{z} \sim \mathcal{N}(0, \mathbf{L}\mathbf{L}^T = \mathbf{K})$

采样的函数展示了核的"个性"：RBF 产生极其平滑的函数，Matern 产生稍粗糙的函数，周期核产生有周期结构的函数。

### 第5步：核岭回归 vs GP 对比

两者的预测均值完全一致（数学上等价），但 GP 额外提供了：
- **蓝色的置信带**：展示模型在不同位置的不确定性
- **置信带的行为**：数据点附近窄（高置信），远离数据点时宽（模型承认"我不知道"）
- **真实函数的包裹**：$\sin(x)$ 真值应在置信带内（约 95% 概率）

这是 GP 相对于"黑箱预测"模型的根本优势——它不仅告诉你预测值，还告诉你这个预测的可靠程度。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| RBF 核 | $\exp(-\|x-x'\|^2 / 2\ell^2)$ | `rbf_kernel()` | 向量化平方距离计算 |
| Matern 3/2 | $(1+\sqrt{3}r)\exp(-\sqrt{3}r)$ | `matern32_kernel()` | 比 RBF 更"粗糙" |
| Cholesky 求解 | $LL^T = K$, 解两次三角方程 | `np.linalg.cholesky()` | 数值优于直接求逆 |
| 核岭回归预测 | $\mathbf{k}_*^T (\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{y}$ | `KernelRidgeRegression` | 仅点估计 |
| GP 预测均值 | 同上 | `GaussianProcessRegressor` | 同核岭回归 |
| GP 预测方差 | $k_* - \mathbf{k}_*^T\mathbf{K}^{-1}\mathbf{k}_*$ | `predict()` var 循环 | 置信带来源 |
| GP 先验采样 | $\mathbf{f} = \mathbf{L}\mathbf{z}, \mathbf{z}\sim\mathcal{N}(0,\mathbf{I})$ | `sample_gp_prior()` | 可视化核的"个性" |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/kernel-gp/code/demo.py`
