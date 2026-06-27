---
title: "ml04 支持向量机 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml04 支持向量机 (SVM) — exercise.py 练习指南

<a href="../code/ml04_svm/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全 Hinge Loss 子梯度、SGD 更新步和 RBF 核矩阵三个模块，从代码层面掌握 SVM 的核心优化。

## 预备知识

- Hinge Loss: $L = \max(0, 1 - y(\mathbf{w}^T \mathbf{x} + b))$，子梯度 $\partial L/\partial \mathbf{w} = -y\mathbf{x}$ 若 $yf < 1$
- L2 正则化梯度: $\partial(\lambda\|\mathbf{w}\|^2)/\partial \mathbf{w} = 2\lambda \mathbf{w}$
- $\lambda$ 与 $C$ 的关系: $\lambda = 1/(2C)$
- RBF 核: $K(\mathbf{x},\mathbf{y}) = e^{-\gamma \|\mathbf{x}-\mathbf{y}\|^2}$

## 任务清单

### 任务1：实现 Hinge Loss 及其子梯度

- **步骤**：
  1. 计算 margin = $y_i \cdot (\mathbf{w}^T \mathbf{x}_i + b)$
  2. Hinge Loss = `max(0, 1 - margin)`
  3. L2 正则化损失 = $\lambda \cdot \|\mathbf{w}\|^2$
  4. L2 梯度: `dw = 2 * lambda * w`
  5. 若 margin < 1: `dw -= y_i * x_i`, `db = -y_i`
  6. 否则: `db = 0`

### 任务2：实现 SGD 单步更新

- **步骤**：
  1. 打乱数据索引（`np.random.permutation`）
  2. 遍历每个样本，调用 `hinge_loss_and_gradient` 获取梯度和损失
  3. 更新 `w -= lr * dw`, `b -= lr * db`
  4. 返回平均损失

### 任务3（Bonus）：RBF 核矩阵

- **步骤**：
  1. 计算 `sq_X = sum(X^2, axis=1, keepdims=True)` 得到 `(m, 1)`
  2. 计算 `sq_Y = sum(Y^2, axis=1)` 得到 `(n,)`
  3. `sq_dists = sq_X + sq_Y - 2*X @ Y.T`
  4. 返回 `exp(-gamma * max(sq_dists, 0))`

## 验证标准

1. `test_hinge_loss()`：已知参数下的 Hinge Loss 和梯度应精确匹配理论值
2. `test_sgd_step()`：执行一轮 SGD 后损失应下降
3. `test_rbf_kernel()`：相同向量的 RBF 核值为 1，远距离向量核值接近 0

## 完整代码

<<< @/snippets/ml04_svm/exercise.py
