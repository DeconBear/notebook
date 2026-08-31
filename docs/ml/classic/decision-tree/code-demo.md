---
title: "ml05 决策树 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml05 决策树 (CART) — demo.py 代码详解

<a href="/notebook/code/ml/classic/decision-tree/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/classic/decision-tree/code
python demo.py
```

## 代码逐段详解

### 第1步：不纯度度量 — 基尼与熵

```python
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))
```

这两个函数度量了数据集的"不纯度"——标签越混杂，值越大。

**基尼指数**的公式：
$$
\text{Gini} = 1 - \sum_{j} p_j^2 = \sum_{j} p_j (1 - p_j)
$$

概率解释：随机从数据集中抽取两个样本，它们属于不同类别的概率。

**熵**的公式：
$$
H = -\sum_{j} p_j \log_2 p_j
$$

概率解释：描述数据集中类别标签所需的最小比特数。

两者在 $p = 0.5$ 时都取得最大值。基尼指数不需要计算对数，因此比熵的计算效率稍高——这是 sklearn 默认使用基尼指数的原因。

`1e-10` 的微小正数防止 `log_2(0) = -inf`。当某个类别在数据集中不存在时（$p_j = 0$），$0 \cdot \log_2(1e-10) \approx 0$，不影响结果。

### 第2步：CART 树节点定义

```python
class Node:
    __slots__ = ('feature_idx', 'threshold', 'left', 'right', 'value', 'is_leaf')
```

`__slots__` 是一个 Python 性能优化——预声明实例属性，阻止动态 `__dict__` 创建，节省内存。对一棵可能有数百节点的决策树来说，这个优化是有意义的。

### 第3步：递归构建树

```python
def _build_tree(self, X, y, depth):
    # 停止条件检查
    if len(np.unique(y)) == 1:     # 所有样本同一类
    if depth >= max_depth:         # 达到最大深度
    if n_samples < min_samples_split:  # 样本太少
```

这是决策树的核心递归逻辑。每次调用 `_build_tree` 处理一个子集：

1. 检查是否应该停止分裂（四个停止条件）
2. 如果不应停止，搜索最佳分裂特征和阈值
3. 创建内部节点，将数据按阈值分为左右两部分
4. 递归构建左右子树

停止条件的顺序很重要——先检查"是否已纯"再检查"是否达到深度限制"，因为一个已纯的节点不应该再浪费时间去搜索分裂。

### 第4步：最佳分裂搜索

```python
def _best_split(self, X, y):
    for feat_idx in range(n_features):
        sorted_indices = np.argsort(feature_values)
        for i in range(n_samples - 1):
            if sorted_X[i] == sorted_X[i + 1]:  # 跳过相同值
                continue
            threshold = (sorted_X[i] + sorted_X[i + 1]) / 2.0
            y_left = sorted_y[:i+1]
            y_right = sorted_y[i+1:]
            gain = parent_impurity - weighted_child_impurity
```

这个双重循环是决策树训练的**计算瓶颈**——复杂度为 $O(d \cdot n^2)$。

优化技巧：
- 按排序后的顺序扫描，每次只将"一个样本"从右边移到左边，可以增量更新统计量（但这里用了直接计算，代码更清晰）
- 跳跃相同值的检查 `if sorted_X[i] == sorted_X[i+1]: continue` 避免了无效分裂（同值无法区分）

### 第5步：从根到叶的预测

```python
def _traverse(self, x, node):
    if node.is_leaf:
        return node.value
    if x[node.feature_idx] <= node.threshold:
        return self._traverse(x, node.left)
    else:
        return self._traverse(x, node.right)
```

预测的复杂度为 $O(\text{depth})$——极其高效。每层只需比较一个特征的值，沿树走到叶节点。这就是为什么即使是深度为 20 的决策树，对单个样本的预测也是瞬间完成的。

### 第6步：max_depth 的可视化对比

从 `max_depth=1` 到 `max_depth=None`（无限），决策边界从简单的垂直/水平线逐步变为复杂的阶梯状区域。无限制的树会围绕每个训练样本形成极小区域——这是过拟合的典型表现。

**max_depth 的角色**：它是最简单有效的预剪枝参数。增加深度 → 更低偏差 + 更高方差（Bias-Variance 权衡在决策树中体现得非常直观）。

## 关键概念速查表

| 概念 | 公式 | 代码位置 | 关键说明 |
|------|------|---------|---------|
| 基尼指数 | $1 - \sum p_j^2$ | `gini()` | 随机抽取两个样本类别不同的概率 |
| 熵 | $-\sum p_j\log_2 p_j$ | `entropy()` | 描述标签所需的最少比特数 |
| 信息增益 | $H_{\text{parent}} - \sum\frac{n_k}{n}H_k$ | `_best_split()` | 分裂前后的不纯度减少量 |
| 连续值阈值 | $(v_i + v_{i+1})/2$ | `threshold` | 排序后相邻取值的 M=n/2 |
| CART | 二叉树 + 基尼/熵 | `CARTDecisionTree` | sklearn 默认实现 |
| 预剪枝 | max_depth, min_samples_* | `_build_tree()` 的条件 | 提前停止生长 |
| 叶节点值 | `np.bincount(y).argmax()` | `_majority_vote()` | 多数类标签 |

## 完整代码

<<< @/ml/classic/decision-tree/code/demo.py
