---
title: "ml07 集成学习：Boosting 与 Stacking — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml07 集成学习：Boosting 与 Stacking — demo.py 代码详解

<a href="../code/ml07_boosting/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml07_boosting/code
python demo.py
```

## 代码逐段详解

### 第1步：DecisionStump — 最小的弱学习器

```python
class DecisionStump:
    def fit(self, X, y, sample_weight=None):
        # 对每个特征搜索最优阈值
        # 区分两种极性：左=0/右=1 和 左=1/右=0
```

决策树桩（Decision Stump）是深度为 1 的决策树——它只做一次分裂。作为 AdaBoost 的弱学习器，它只需比随机猜测稍好即可（错误率 < 50%）。

树桩搜索最优分裂的复杂度为 $O(d \cdot n \log n)$，因为它对每个特征排序后扫描所有可能的阈值——这与完整决策树的第一步完全相同。

### 第2步：AdaBoost 迭代训练

```python
def fit(self, X, y):
    w = np.ones(n_samples) / n_samples
    for m in range(self.n_estimators):
        stump.fit(X, y, sample_weight=w)
        err = np.sum(w * incorrect) / np.sum(w)
        alpha = 0.5 * np.log((1.0 - err) / err)
        w = w * np.exp(alpha * incorrect)
        w = w / np.sum(w)
```

这段代码是 AdaBoost 的核心。让我们逐步理解：

1. **加权错误率**：不是简单的 `incorrect/n`，而是 `sum(w * incorrect) / sum(w)`——考虑了每个样本的不同权重（困难样本权重大）

2. **alpha 的计算**：
   $$\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$$
   
   当 $\epsilon_m = 0.5$（随机猜测）：$\alpha_m = 0$，该学习器不被考虑
   
   当 $\epsilon_m \to 0$（完美）：$\alpha_m \to \infty$，该学习器权重极大

3. **权重更新**：
   $$w_i \leftarrow w_i \cdot e^{\alpha_m \cdot \mathbb{I}(h_m(\mathbf{x}_i) \neq y_i)}$$
   
   正确样本：权重不变（乘 $e^0 = 1$）
   
   错误样本：权重乘 $e^{\alpha_m} > 1$（放大，让下轮更关注这些样本）

4. **归一化**：`w = w / np.sum(w)`——确保权重和为 1

### 第3步：AdaBoost 预测 — 累积置信度

```python
def predict(self, X):
    for alpha, stump in zip(self.alphas, self.stumps):
        pred_svm = 2 * stump.predict(X) - 1  # 0/1 → -1/+1
        scores += alpha * pred_svm
    return (scores >= 0).astype(np.int64)
```

最终分类结果 = 所有学习器加权投票的符号：

$$
F(\mathbf{x}) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m \cdot h_m(\mathbf{x})\right)
$$

其中 $h_m(\mathbf{x}) \in \{-1, +1\}$。每个学习器的话语权由它自己的准确率（通过 $\alpha_m$）决定——准确的权重大，不准的权重小。

### 第4步：SimpleGBDT — 拟合残差的直觉

```python
class SimpleGBDT:
    def fit(self, X, y):
        F = np.full(len(y), np.mean(y))  # 初始: 全部预测均值
        for m in range(self.n_estimators):
            residuals = y - F  # MSE 的负梯度 = 残差
            tree.fit(X, residuals)  # 训练树拟合残差
            F += self.learning_rate * tree.predict(X)  # 更新
```

GBDT 使用平方损失时的直观理解：

1. **初始化**：$F_0(x) = \text{mean}(y)$——最朴素的预测
2. **每轮迭代**：
   - 计算残差 $r_i = y_i - F_{m-1}(\mathbf{x}_i)$
   - 训练一棵树来预测这些残差：$h_m(\mathbf{x}) \approx r$
   - 更新：$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$

3. **学习率 $\eta$**（shrinkage）：每棵树贡献的一部分（通常 $0.01 \sim 0.1$）。小的学习率需要更多的树，但泛化效果更好。

当 $L = \frac{1}{2}(y-F)^2$ 时，负梯度为 $-\partial L / \partial F = y - F$——恰好是残差。所以 GBDT 回归可以直观理解为"反复训练树来拟合残差"。

### 第5步：对比实验 — 树桩 vs AdaBoost vs 随机森林

```python
models = {
    'Decision Stump': DecisionStump(),
    'AdaBoost (50 stumps)': AdaBoost(n_estimators=50),
    'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50),
}
```

这个对比展示了三种模型的决策边界差异：

- **决策树桩**：一条直线分割——最简单的分类器
- **AdaBoost**：50 个树桩的加权组合——尽管每个树桩只是一条直线，但它们的加权累加产生了复杂而平滑的边界
- **随机森林**：50 棵树投票——边界也平滑但形成机制不同（平均 vs 加权累加）

AdaBoost 的边界对噪声敏感——因为错误样本的权重会不断增大，一个异常值可能获得极大的权重。这是 AdaBoost 的一个已知缺点。

## 关键概念速查表

| 概念 | 公式 | 代码位置 | 关键说明 |
|------|------|---------|---------|
| 决策树桩 | depth=1 的决策树 | `DecisionStump` | AdaBoost 的弱学习器 |
| 加权错误率 | $\sum w_i \mathbb{I}(err) / \sum w_i$ | `AdaBoost.fit()` | 考虑了样本重要性 |
| Alpha | $\frac{1}{2}\ln\frac{1-\epsilon}{\epsilon}$ | `compute_alpha` | 学习器的"话语权" |
| 权重更新 | $w_i \leftarrow w_i e^{\alpha \mathbb{I}(err)}$ | `update_weights` | 错误样本权重放大 |
| 指数损失 | $\exp(-yF(\mathbf{x}))$ | 理论 | AdaBoost 的理论损失函数 |
| 伪残差 | $-\partial L/\partial F$ | `SimpleGBDT.fit()` | 在函数空间中的梯度 |
| MSE 残差 | $y - F$ | `residuals = y - F` | 平方损失的负梯度 |
| Shrinkage | $\eta$ (learning_rate) | `self.learning_rate` | 缩小每棵树的贡献 |

## 完整代码

<<< @/snippets/ml07_boosting/demo.py
