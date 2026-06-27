---
title: "ml06 集成学习：Bagging 与随机森林 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml06 集成学习：Bagging 与随机森林 — demo.py 代码详解

<a href="../code/ml06_random_forest/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml06_random_forest/code
python demo.py
```

## 代码逐段详解

### 第1步：从 Bootstrap 到 Bagging

```python
boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
X_boot, y_boot = X[boot_idx], y[boot_idx]
```

`np.random.choice` 的核心参数是 `replace=True`（有放回采样）。这意味着：

- 每个样本每次被选中的概率都是 $1/n$
- 一个样本可能被选中多次（在 `X_boot` 中出现多次）
- 一个样本可能一次都不被选中（成为 OOB 样本）

数学上，某样本不被选中的概率：
$$
P = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368
$$

这就是为什么 OOB 样本约占总体的 36.8%。

### 第2步：随机特征子空间

```python
if self.max_features is None:
    features_to_use = list(range(n_features))
else:
    m = min(self.max_features, n_features)
    features_to_use = np.random.choice(n_features, size=m, replace=False)
```

这是随机森林区别于普通 Bagging 的核心机制。在决策树的每个分裂节点（不是每棵树！），只随机选择 $m$ 个特征来评估：

- $m = \sqrt{d}$：Breiman 推荐的分类默认值
- $m = d/3$：回归的推荐值
- $m = d$（使用全部特征）：退化为普通 Bagging

这种随机化迫使各树使用不同的特征进行分裂，从而**降低树之间的相关性** $\rho$。根据集成方差公式 $\text{Var} \to \rho\sigma^2$（当 $B \to \infty$），$\rho$ 越小，集成方差越低。

### 第3步：多数投票

```python
def predict(self, X):
    all_preds = np.array([tree.predict(X) for tree in self.trees])
    predictions = scipy_stats.mode(all_preds, axis=0, keepdims=False)[0]
    return predictions.ravel()
```

`scipy.stats.mode` 对每个测试样本（axis=0）找出在 `n_estimators` 棵树中出现次数最多的类别。如果不使用 scipy，也可以用：

```python
# 等价实现（手动投票）
predictions = [np.bincount(all_preds[:, i]).argmax() for i in range(n_samples)]
```

### 第4步：OOB 误差 — 免费的验证集

```python
def oob_score(self, X, y):
    for i in range(n_samples):
        tree_indices = [t for t in range(self.n_estimators)
                        if i not in self.bootstrap_indices[t]]
        votes = [self.trees[t].predict(X[i:i+1])[0] for t in tree_indices]
        oob_predictions[i] = scipy_stats.mode(votes)[0]
```

OOB 的核心逻辑：
1. 对于每个样本 $i$，检查它**不在**哪些树的 Bootstrap 抽样中
2. 只用这些树（它们"没见过"样本 $i$）来预测样本 $i$
3. 所有树对该样本的平均约 36.8% 未使用该样本

OOB 误差通常能很好地近似测试集误差——它和交叉验证一样是"用未见过模型的数据来评估"，但不需要额外的计算开销（因为这些树本来就"免费地"未见过某些数据）。

### 第5步：三路对比 — 单树 vs Bagging vs 随机森林

`plot_single_vs_rf_boundary()` 函数并排对比了三种模型在同一数据上的决策边界：

1. **单棵决策树**（`max_depth=None`）：边界非常复杂，布满锯齿。训练精度高，测试精度低——过拟合。
2. **Bagging**（`max_features='all'`）：边界明显平滑，但仍有一些不规则的锯齿。方差有所降低但树之间仍有较高相关性。
3. **随机森林**（`max_features='sqrt'`）：边界最平滑，区域最规整。特征子空间随机化进一步降低了树间相关性。

## 关键概念速查表

| 概念 | 公式/描述 | 代码位置 | 关键说明 |
|------|----------|---------|---------|
| Bootstrap | `np.random.choice(n, n, replace=True)` | `fit()` | 有放回采样 |
| OOB 概率 | $(1-1/n)^n \approx 36.8\%$ | `oob_score()` 中 | 自然验证集 |
| 随机特征子空间 | `np.random.choice(d, m, replace=False)` | `_best_split()` | 降低树间相关性 |
| 多数投票 | `scipy.stats.mode` | `predict()` | 分类的聚合方式 |
| 集成方差 | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ | 理论 | $\rho$ 越低越好 |
| MDI 重要性 | 加权不纯度减少 | 理论 | 特征对纯度的贡献 |

## 完整代码

<<< @/snippets/ml06_random_forest/demo.py
