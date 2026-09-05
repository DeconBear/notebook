---
title: "s02 线性回归 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s02 线性回归 — exercise.py 练习指南

<a href="/notebook/code/ml/foundations/linear-regression/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全线性回归的三个核心组件（MSE 损失计算、梯度计算、参数更新），以及一个 Bonus 任务（Mini-batch 梯度下降），深入理解线性回归从数学公式到代码实现的完整映射关系。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 线性模型 $\hat{y} = wx + b$ 的含义
- MSE 损失函数 $J(w,b) = \frac{1}{n} \sum (\hat{y}_i - y_i)^2$ 及其为什么可导、凸、对大误差敏感
- 梯度下降更新规则 $w \leftarrow w - \eta \cdot \frac{\partial J}{\partial w}$
- Mini-batch GD 与 Full-batch GD 的区别：前者每轮更新多次（每个 batch 一次），后者每轮只更新一次

## 任务清单

### 任务1：实现 MSE 损失计算 `_compute_loss(X, y)`

- **用到的公式**：
  $$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
- **实现步骤**：
  1. 用 `self.predict(X)` 得到预测值 $\hat{y}$
  2. 计算每个样本的平方误差 $(\hat{y} - y)^2$
  3. 对所有样本取平均
- **需要调用的函数**：`self.predict()`、`np.mean()`
- **期望输出**：一个标量 float，表示当前参数下的 MSE 损失值

### 任务2：实现梯度计算 `_compute_gradients(X, y)`

- **用到的公式**：
  $$\frac{\partial J}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i$$
  $$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$
- **实现步骤**：
  1. 用 `self.predict(X)` 得到预测值
  2. 计算误差向量 `errors = y_pred - y`
  3. $dw = (2/n) \cdot \sum (\text{errors} \cdot x_i)$
  4. $db = (2/n) \cdot \sum \text{errors}$
- **需要调用的函数**：`self.predict()`、`np.sum()`
- **关键细节**：$dw$ 公式中的 `errors * X` 是逐元素乘法——每个样本的误差乘以该样本的特征值 $x_i$

### 任务3：实现梯度下降参数更新

在 `fit()` 方法的循环中，找到 TODO 3 标记的位置。

- **用到的公式**：
  $$w \leftarrow w - \eta \cdot \frac{\partial J}{\partial w}$$
  $$b \leftarrow b - \eta \cdot \frac{\partial J}{\partial b}$$
- **实现**：两行代码，分别更新 `self.w` 和 `self.b`
- **注意**：是 `-=`（减等于），不是 `+=`！因为我们沿梯度的反方向走（下降而非上升）
- **变量名对应**：`self.learning_rate` 是 $\eta$，`dw` 是 $\partial J/\partial w$，`db` 是 $\partial J/\partial b$

### 任务4（Bonus）：实现 Mini-batch 梯度下降

完成 `MiniBatchLinearRegression` 类的 `fit()` 方法。

- **算法流程**：
  1. 打乱数据顺序（`np.random.permutation(n)`）
  2. 按 `batch_size`（默认 32）切分成多个小批次
  3. 对每个批次：用批次内数据计算梯度，更新参数
  4. 所有批次处理完毕 = 1 个 epoch
- **需要调用的函数**：`np.random.permutation()`
- **遍历批次的技巧**：`range(0, n, self.batch_size)` 按 batch 大小步进，每次取 `X_shuffled[start:start+batch_size]`
- **梯度公式与 full-batch 相同**，只是用 `X_batch, y_batch` 代替 `X, y`

## 验证标准

运行 `python exercise.py`，如果你的实现正确：

1. Full-batch GD 输出参数接近真实值 $w=2.0, b=5.0$，误差满足 $|w - 2.0| < 0.1$ 且 $|b - 5.0| < 0.5$
2. 打印 `✓ 参数接近真实值，你的实现基本正确！`
3. Mini-batch GD 的参数也接近真实值（Bonus）
4. 可视化图中，Full-batch GD 和 Mini-batch GD 的拟合直线都穿过数据点云的中心


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/foundations/linear-regression/code/exercise.py`
