---
title: "ml06 集成学习：Bagging 与随机森林 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml06 集成学习：Bagging 与随机森林 — exercise.py 练习指南

<a href="/notebook/code/ml/advanced/random-forest/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全 Bootstrap 采样、多数投票和 OOB 误差计算三个模块，从代码层面掌握 Bagging/随机森林的核心机制。

## 预备知识

- Bootstrap：有放回采样 $n$ 个样本，每个样本不被选中的概率 $\approx 36.8\%$
- 多数投票：`scipy.stats.mode` 或 `np.bincount + np.argmax`
- OOB：对每个样本，只用未使用该样本训练的树来预测

## 任务清单

### 任务1：实现 Bootstrap 采样 `bootstrap_sample(X, y)`

- **步骤**：
  1. `boot_idx = np.random.choice(n, size=n, replace=True)`（注意是 `replace=True`，不是 `False`！）
  2. 提取采样后的数据和 OOB 索引
  3. `oob_idx = np.setdiff1d(np.arange(n), np.unique(boot_idx))`

### 任务2：实现多数投票 `majority_vote_predict(trees, X)`

- **步骤**：
  1. 收集所有树的预测：`all_preds = np.array([tree.predict(X) for tree in trees])`
  2. 用 `scipy.stats.mode(all_preds, axis=0)` 找每列的众数

### 任务3（Bonus）：计算 OOB 误差 `compute_oob_score(trees, bootstrap_indices, X, y)`

- **步骤**：
  1. 对每个样本 $i$，找到所有 `i not in bootstrap_indices[t]` 的树
  2. 用这些树预测样本 $i$
  3. 计算有 OOB 覆盖的样本上的准确率

## 验证标准

1. `test_bootstrap()`：OOB 样本数应接近 36.8%
2. `test_majority_vote()`：多数投票结果应为众数
3. `test_oob_score()`：OOB 准确率应在 [0, 1] 范围内


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/random-forest/code/exercise.py`
