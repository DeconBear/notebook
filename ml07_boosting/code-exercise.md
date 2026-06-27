---
title: "ml07 集成学习：Boosting 与 Stacking — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml07 集成学习：Boosting 与 Stacking — exercise.py 练习指南

<a href="../code/ml07_boosting/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全 AdaBoost 的 alpha 计算、样本权重更新和 GBDT 单轮迭代三个模块，从代码层面掌握 Boosting 的核心机制。

## 预备知识

- AdaBoost 学习器权重：$\alpha = \frac{1}{2} \ln\left(\frac{1-\epsilon}{\epsilon}\right)$
- 权重更新：$w_i \leftarrow w_i \cdot e^{\alpha \cdot \mathbb{I}(\text{error})}$，然后归一化
- GBDT (MSE)：$r_i = y_i - F(\mathbf{x}_i)$，训练树拟合 $r$，$F \leftarrow F + \eta \cdot h(\mathbf{x})$

## 任务清单

### 任务1：实现 `compute_alpha(error_rate)`

- **步骤**：
  1. `error_rate = max(error_rate, 1e-10)`（防除零）
  2. `alpha = 0.5 * np.log((1.0 - error_rate) / error_rate)`
  3. 返回 alpha
- **关键细节**：当 $\epsilon = 0.5$ 时 $\alpha = 0$（学习器无贡献）；当 $\epsilon \to 0$ 时 $\alpha \to \infty$

### 任务2：实现 `update_weights(weights, alpha, incorrect_mask)`

- **步骤**：
  1. 将 bool 转为 float：`incorrect_float = incorrect_mask.astype(float)`
  2. 乘 `exp(alpha * incorrect_float)`
  3. 归一化：`weights / np.sum(weights)`

### 任务3（Bonus）：理解 GBDT 单轮迭代

- 伪代码流程：
```python
residuals = y - y_current_pred
tree = DecisionTreeRegressor(max_depth=max_depth)
tree.fit(X, residuals)
new_pred = y_current_pred + learning_rate * tree.predict(X)
```

## 验证标准

1. `test_compute_alpha()`：$\alpha(0.1) \approx 1.099$，$\alpha(0.5) = 0$
2. `test_update_weights()`：错误样本权重之和增大，权重归一化
3. `test_adaboost_simple()`：完整的 5 轮 AdaBoost 流程能正常运行

## 完整代码

<<< @/snippets/ml07_boosting/exercise.py
