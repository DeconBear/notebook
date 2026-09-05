---
title: "s03 逻辑回归 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s03 逻辑回归 — exercise.py 练习指南

<a href="/notebook/code/ml/foundations/logistic-regression/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现逻辑回归的四个核心数学函数，深刻理解 Sigmoid、交叉熵损失、梯度计算以及 Softmax 的数学原理和代码表达。这些函数是几乎所有现代神经网络分类器的基础组件。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- Sigmoid 函数的定义与性质：$\sigma(z) = \frac{1}{1+e^{-z}}$，值域 $(0,1)$，导数 $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- 交叉熵损失的设计动机：为什么分类问题用交叉熵而不是 MSE
- Sigmoid + 交叉熵的"黄金梯度"：$\partial L / \partial z = \hat{y} - y$（Sigmoid 导数项在链式法则中被约掉）
- Softmax 的数值稳定技巧：每行减去该行的最大值再计算指数

## 任务清单

### 任务1：实现 Sigmoid 函数 `sigmoid(z)`

- **用到的公式**：
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **提示**：
  - 使用 `np.exp()` 计算指数函数
  - 使用 `np.clip(z, -500, 500)` 限制 $z$ 的范围，防止 $e^z$ 溢出（当 $z$ 过大时 $e^z$ 会变成无穷大）
  - 返回 `1.0 / (1.0 + np.exp(-z_clipped))`
- **验证**：$\sigma(0) = 0.5$，$\sigma(-10) \approx 0$，$\sigma(10) \approx 1$

### 任务2：实现交叉熵损失 `cross_entropy_loss(y_pred, y_true)`

- **用到的公式**：
  $$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$
- **实现步骤**：
  1. 用 `np.clip(y_pred, eps, 1-eps)` 裁剪预测概率（`eps = 1e-15`），防止 $\log(0) = -\infty$
  2. 套用公式计算损失
- **需要调用的函数**：`np.clip()`、`np.log()`、`np.sum()`、`np.mean()` 或其等价写法
- **验证**：完美预测时损失接近于 0（例如 `y_pred=[0.99, 0.01]`, `y_true=[1, 0]`），完全错误时损失很大

### 任务3：实现交叉熵梯度 `compute_gradients(X, y_true, y_pred)`

- **用到的公式**：
  $$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)$$
  $$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{y}_i - y_i)$$
- **实现步骤**：
  1. 计算预测误差 `errors = y_pred - y_true`（这就是 $\partial L/\partial z$！）
  2. $dw = (1/n) \cdot X^T @ \text{errors}$——矩阵乘法
  3. $db = (1/n) \cdot \sum \text{errors}$——求和
- **需要调用的函数**：`@` 运算符或 `np.dot()`、`np.sum()`
- **关键理解**：`errors` 就是 $\partial L / \partial z$，这是 Sigmoid + 交叉熵"黄金组合"的精髓——梯度等于预测误差，极其简洁

### 任务4（Bonus）：实现 Softmax 函数 `softmax(z)`

- **用到的公式**：
  $$\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$
- **数值稳定技巧**：
  1. `z_stable = z - np.max(z, axis=1, keepdims=True)`（每行减去该行的最大值）
  2. `exp_z = np.exp(z_stable)`（计算稳定后的指数）
  3. `return exp_z / np.sum(exp_z, axis=1, keepdims=True)`（归一化）
- **需要调用的函数**：`np.max()`、`np.exp()`、`np.sum()`
- **验证**：每行之和应等于 1，每个值在 $(0, 1)$ 之间，得分最高的类别概率也最高

## 验证标准

运行 `python exercise.py`，如果你的实现正确：

1. `test_sigmoid()`：基本值正确，$\sigma(0) = 0.5$
2. `test_cross_entropy()`：错误预测的损失 > 正确预测的损失
3. 只有完成 Softmax 部分后，`test_softmax()` 才会通过：每行和为 1，概率值在 $[0,1]$ 内，最大概率对应最大得分


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/foundations/logistic-regression/code/exercise.py`
