---
title: "ml05 决策树 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml05 决策树 (CART) — exercise.py 练习指南

<a href="/notebook/code/ml/classic/decision-tree/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全基尼指数、信息增益计算和最佳分裂点搜索三个模块，从代码层面掌握决策树核心机制。

## 预备知识

- 基尼指数：$\text{Gini} = 1 - \sum p_j^2 = \sum p_j (1-p_j)$
- 熵：$H = -\sum p_j \log_2 p_j$
- 信息增益（基尼减少量）：$\Delta \text{Gini} = \text{Gini}_{\text{parent}} - \frac{n_L}{n}\text{Gini}_L - \frac{n_R}{n}\text{Gini}_R$
- 连续特征分裂点：排序后扫描相邻取值的 M=n/2 处

## 任务清单

### 任务1：实现基尼指数 `compute_gini(y)`

- **步骤**：
  1. `np.unique(y, return_counts=True)` 获取类别计数
  2. `probs = counts / len(y)`
  3. `return 1 - np.sum(probs ** 2)`

### 任务2：实现熵 `compute_entropy(y)`

- **步骤**：
  1. 获取各类别占比
  2. `ent = -np.sum(probs * np.log2(probs + 1e-10))`
  3. 返回 ent

### 任务3：实现信息增益 `information_gain(y_parent, y_left, y_right)`

- **步骤**：
  1. 计算 `gini_parent`, `gini_left`, `gini_right`
  2. 计算 `w_impurity = (n_L/n)*Gini_L + (n_R/n)*Gini_R`
  3. 返回 `Gini_parent - w_impurity`

### 任务4（Bonus）：搜索最佳分裂点 `best_split_single_feature(X_col, y)`

- **步骤**：
  1. `argsort` 排序特征值
  2. 扫描 `n-1` 个候选阈值：`threshold = (X[i] + X[i+1]) / 2`
  3. 跳过重复值
  4. 记录最大增益的阈值

## 验证标准

1. `test_gini()`：纯净集 Gini=0，各半集 Gini=0.5
2. `test_information_gain()`：完美分裂 IG=0.5
3. `test_best_split()`：能找到正确的分裂阈值


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/classic/decision-tree/code/exercise.py`
