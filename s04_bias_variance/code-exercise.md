---
title: "s04 偏差-方差权衡 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s04 偏差-方差权衡 — exercise.py 练习指南

<a href="../code/s04_bias_variance/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全正则化梯度计算、K-Fold 交叉验证和 Bias-Variance 分解三个模块，从代码层面理解机器学习泛化理论的核心实践。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- L1/L2 正则化的损失函数形式：$J_{\text{Ridge}} = \text{MSE} + \lambda \|\theta\|_2^2$，$J_{\text{Lasso}} = \text{MSE} + \lambda \|\theta\|_1$
- L2 正则化梯度：$\partial(\lambda \sum \theta_j^2) / \partial \theta_j = 2\lambda \theta_j$
- L1 正则化梯度：$\partial(\lambda \sum \|\theta_j\|) / \partial \theta_j = \lambda \cdot \text{sign}(\theta_j)$
- 偏置项不参与正则化：正则化的目标是约束模型复杂度（特征系数），偏置只影响截距
- K-Fold 交叉验证的原理：$K$ 份数据轮流做验证集，其余做训练集
- Bias-Variance 分解公式：
  $$\mathbb{E}[(y - \hat{f})^2] = \text{Bias}[\hat{f}]^2 + \text{Var}[\hat{f}] + \sigma^2$$

## 任务清单

### 任务1：实现带正则化的梯度计算 `compute_regularized_gradient(X, y, w, lambda_, reg_type)`

- **用到的公式**：
  - MSE 梯度：$\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n} X^T (Xw - y)$
  - L2 正则化梯度：$\frac{\partial (\lambda \sum w_j^2)}{\partial w_j} = 2\lambda w_j$
  - L1 正则化梯度：$\frac{\partial (\lambda \sum |w_j|)}{\partial w_j} = \lambda \cdot \text{sign}(w_j)$
  - 总梯度 = MSE 梯度 + 正则化梯度
- **实现步骤**：
  1. 计算 MSE 梯度 `dw_mse = (2/n) * X.T @ (X @ w - y)`
  2. 创建全零的正则化梯度数组 `dw_reg = np.zeros(len(w))`
  3. 根据 `reg_type` 填充 `dw_reg[1:]`（跳过索引 0 即偏置项）：
     - `'l2'`：`dw_reg[1:] = 2 * lambda_ * w[1:]`
     - `'l1'`：`dw_reg[1:] = lambda_ * np.sign(w[1:])`
  4. 返回 `dw_mse + dw_reg`
- **需要调用的函数**：`@` 运算符（矩阵乘法）、`np.sign()`、`np.zeros()`
- **关键细节**：`w[1:]` 跳过偏置项（索引 0），`dw_reg[1:]` 也只填充非偏置位置

### 任务2：实现 K-Fold 交叉验证 `kfold_cross_validation(X, y, k, degree)`

- **算法流程**：
  1. 计算每折大小 `fold_size = n // k`
  2. 对 $i = 0, 1, \dots, k-1$：
     - 确定验证集起止索引：`val_start = i * fold_size`, `val_end = (i+1) * fold_size`（最后一折到末尾）
     - 验证集索引：`val_idx = np.arange(val_start, val_end)`
     - 训练集索引：使用 `np.setdiff1d(np.arange(n), val_idx)` 排除验证集索引
     - 划分数据：`X_train = X[train_idx]`, `y_train = y[train_idx]`
     - 生成多项式特征并训练（使用正规方程或 `np.linalg.pinv`）
     - 验证集预测并计算 MSE
  3. 返回平均 MSE 和所有折的 MSE 列表
- **需要调用的函数**：`np.arange()`、`np.setdiff1d()`、`polynomial_features()`、`np.linalg.pinv()`、`np.mean()`
- **期望输出**：`val_mses` 列表应有恰好 $k$ 个元素

### 任务3（Bonus）：实现 Bias-Variance 的经验估计 `compute_bias_variance(X_test, y_true, n_trials, degree, noise_std)`

- **算法**：
  1. 生成 $M$ 个不同的训练集（同一组 $X$ 但每次加上新的随机噪声）
  2. 对每个训练集训练一个模型，记录对测试集的预测
  3. 计算所有模型预测的均值 $\mathbb{E}[\hat{f}(x)]$、偏差 $(\mathbb{E}[\hat{f}] - f)^2$、方差 $\text{Var}(\hat{f})$
- **实现步骤**：
  1. 循环 `n_trials` 次，每次：生成含噪声的 $y$，用正规方程训练模型，记录对 $X_{\text{test}}$ 的预测
  2. 计算 `mean_preds = np.mean(predictions, axis=0)`（所有模型在各测试点的平均预测）
  3. `bias_sq = np.mean((mean_preds - y_true) ** 2)`（平均预测与真实值的平方差）
  4. `variance = np.mean(np.var(predictions, axis=0))`（各测试点上预测的方差再平均）
- **直觉理解**：
  - Bias 高 = 模型太简单，平均预测偏离真实函数
  - Variance 高 = 模型太复杂，对不同的噪声采样得到差异很大的模型

## 验证标准

运行 `python exercise.py`：

1. `test_l2_gradient()`：偏置梯度 `dw[1]` 不应受正则化影响（因为偏置被排除在正则化之外）
2. `test_kfold()`：应生成恰好 5 个验证误差值
3. `test_bias_variance()`（Bonus）：Bias² 和 Variance 均为正值，预测形状为 `(n_trials, n_test)`

## 完整代码

<<< @/snippets/s04_bias_variance/exercise.py
