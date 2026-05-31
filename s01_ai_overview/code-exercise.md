---
title: "s01 AI概述 — exercise.py"
---

# s01 AI概述 — exercise.py 练习指南

<a href="../code/s01_ai_overview/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全感知机的三个核心组件，深入理解感知机的数学原理和代码实现。完成这个练习后，你将能够：

1. 用一行 NumPy 代码实现阶跃激活函数（理解向量化操作）
2. 实现感知机的预测方法（理解 $w \cdot x + b \to \text{sign}$ 的完整流程）
3. 掌握感知机权重更新规则（理解"朝正确方向微调参数"的直觉）

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中对这些概念的详细解释）：

- 感知机的数学模型：$\hat{y} = \text{sign}(w \cdot x + b)$
- 阶跃函数的定义：$\text{sign}(z) = +1 \ (z \ge 0), -1 \ (z < 0)$
- 感知机学习规则：$w \leftarrow w + \eta \cdot y_i \cdot x_i$（仅当误分类时）
- 向量化运算：`np.dot()` 和 `np.where()` 的用法

## 任务清单

### 任务1：实现阶跃激活函数 `_activation(z)`

在 `PerceptronExercise` 类的 `_activation` 方法中，你需要实现阶跃函数。

- **提示**：使用 `np.where(condition, value_if_true, value_if_false)` 实现向量化判断
- **期望行为**：
  - $z \geq 0$ 时输出 $+1$
  - $z < 0$ 时输出 $-1$
- **测试方法**：输入 `np.array([-1, 0, 1])`，期望输出 `[-1, 1, 1]`

### 任务2：实现预测方法 `predict(X)`

在 `PerceptronExercise` 类的 `predict` 方法中，你需要完成两个步骤：

1. 计算线性组合 $z = w \cdot X + b$（对所有样本批量计算）
2. 将 $z$ 通过激活函数得到类别标签

- **提示**：
  - 使用 `np.dot(X, self.w) + self.b` 进行批量计算（$X$ 是 $(n, d)$ 矩阵，$w$ 是 $(d,)$ 向量，结果自动广播）
  - 调用 `self._activation()` 将得分转换为类别
- **需要调用的函数**：
  - `np.dot()`：矩阵乘法
  - `self._activation()`：你在任务1中实现的阶跃函数
- **期望输出**：返回形状为 $(n\_samples,)$ 的数组，每个元素为 $+1$ 或 $-1$

### 任务3：实现感知机权重更新规则

在 `fit` 方法中，找到 `TODO 3` 标记的位置。当条件 `y_i * z <= 0` 成立（即样本被误分类）时，实现权重和偏置的更新。

- **用到的公式**：
  - $w \leftarrow w + \eta \cdot y_i \cdot x_i$
  - $b \leftarrow b + \eta \cdot y_i$
- **直觉解释**：
  - 如果 $y_i = +1$ 但预测为 $-1$：把 $w$ 往 $x_i$ 的正方向推
  - 如果 $y_i = -1$ 但预测为 $+1$：把 $w$ 往 $x_i$ 的反方向推
- **需要调用的变量**：
  - `self.learning_rate`：学习率 $\eta$
  - `x_i`：当前样本的特征向量
  - `y_i`：当前样本的真实标签
  - `self.w`：权重向量（需要就地修改）
  - `self.b`：偏置标量（需要就地修改）
- **期望输出**：训练完成后，对线性可分数据应达到 100% 准确率，损失曲线（误分类数）单调下降至 0

## 验证标准

运行 `python exercise.py`，如果你的实现正确：

1. 训练过程中的误分类数逐渐减少到 0
2. 最终打印 `训练集准确率: 100.00%`
3. 显示 `✓ 完美！你的感知机实现正确，所有样本分类正确。`
4. 可视化图显示清晰的决策边界将红蓝两类数据完全分开

## 完整代码

<<< @/snippets/s01_ai_overview/exercise.py
