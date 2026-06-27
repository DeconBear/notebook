---
title: "ml03 朴素贝叶斯与贝叶斯网络 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml03 朴素贝叶斯与贝叶斯网络 — demo.py 代码详解

<a href="../code/ml03_naive_bayes/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml03_naive_bayes/code
python demo.py
```

## 代码逐段详解

### 第1步：GaussianNB — 从对角协方差说起

```python
class GaussianNB:
    def fit(self, X, y):
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx] = np.mean(X_c, axis=0)
            self.vars_[idx] = np.var(X_c, axis=0) + self.var_smoothing
```

与 ml02 中完整高斯贝叶斯的区别：这里只需要估计**各特征独立的均值和方差**，而**不需要协方差矩阵**。

参数数量对比：
- 完整高斯贝叶斯（`cov_type='full'`）：$C \times d$ 个均值 + $C \times d(d+1)/2$ 个协方差元素
- 朴素贝叶斯（GaussianNB）：$C \times d$ 个均值 + $C \times d$ 个方差 = 仅 $O(C \cdot d)$

对于 $d=100$ 的特征空间，前者需要约 $5,000+$ 个协方差参数，后者只需要 $100$ 个方差参数。

对数似然计算利用条件独立假设：

```python
log_lik = -0.5 * (np.log(2 * np.pi * var) + ((X - mu) ** 2) / var)
log_lik_sum = np.sum(log_lik, axis=1)  # sum over features
```

在独立假设下，联合对数似然 = 各特征对数似然之和：
$$
\ln P(\mathbf{x} | \omega_j) = \sum_{i=1}^{d} \ln P(x_i | \omega_j)
$$

而完整高斯贝叶斯的对数似然需要计算马氏距离 $\propto (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$。

### 第2步：MultinomialNB — 文本分类的利器

```python
class MultinomialNB:
    def fit(self, X, y):
        count_c = np.sum(X_c, axis=0)  # 每个特征的总计数
        total_count = np.sum(count_c)
        prob_c = (count_c + self.alpha) / (total_count + self.alpha * n_features)
        self.feature_log_prob_[idx] = np.log(prob_c)
```

MultinomialNB 的核心估计公式为：

$$
P(\text{word}_i | \text{class}_j) = \frac{\text{count}_j(i) + \alpha}{\sum_k \text{count}_j(k) + \alpha \cdot |V|}
$$

其中：
- $\text{count}_j(i)$：类别 $j$ 中词 $i$ 出现的总次数
- $|V|$：词汇表大小
- $\alpha$：拉普拉斯平滑参数

代码中将特征概率取对数存储（`np.log(prob_c)`），因为预测时需要的是对数概率的加权和：

```python
def predict(self, X):
    log_joint = X @ self.feature_log_prob_.T + np.log(self.priors_)
    return self.classes_[np.argmax(log_joint, axis=1)]
```

这里 $$X$$ 与 $$\text{feature\_log\_prob\_}^{T}$$ 的矩阵乘法（`X @ feature_log_prob_.T`）一次性完成了所有样本的加权和计算：$\sum_i x_i \cdot \ln P(x_i | \text{class}_j)$。

### 第3步：拉普拉斯平滑的可视化

三种平滑强度的效果对比：

```python
alphas = [0.001, 1.0, 5.0]
```

- **$\alpha = 0.001$（接近 MLE）**：训练中未出现的特征概率接近 0，过拟合
- **$\alpha = 1.0$（拉普拉斯平滑）**：每个特征至少有 $\frac{1}{\text{total} + |V|}$ 的概率，不过度自信
- **$\alpha = 5.0$（强平滑）**：先验信息过强，所有人的概率都接近均匀分布，欠拟合

### 第4步：垃圾邮件检测

使用模拟的垃圾邮件/正常邮件数据集（20 个词，500 封邮件），展示 MultinomialNB 的实际效果。自定义实现的结果应与 sklearn MultinomialNB 完全一致（因为两者使用了完全相同的公式）。

### 第5步：模型对比 — 特征类型决定最佳选择

最后一个实验对比了 GaussianNB 和 MultinomialNB 在两种数据集上的表现：
- 连续特征数据：GaussianNB 占优（因为 MultinomialNB 不适合取负值或小数的特征）
- 计数特征数据：MultinomialNB 占优（因为数据生成过程与多项分布假设吻合）

**关键教训**：朴素贝叶斯变体的选择应由**特征的类型**决定，而非算法本身的"好坏"——每个变体各有其适用场景。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 条件独立 | $P(\mathbf{x}\|\omega) = \prod_i P(x_i\|\omega)$ | `_log_prob()` | 参数 $O(C \cdot d)$ |
| GaussianNB 似然 | $\mathcal{N}(x_i\|\mu_{ji}, \sigma_{ji}^2)$ | `GaussianNB._log_prob()` | 对角协方差 |
| MultinomialNB 估计 | $(N_{ji} + \alpha) / (N_j + \alpha d)$ | `MultinomialNB.fit()` | 拉普拉斯平滑 |
| 平滑参数 | $\alpha$ | `self.alpha` | $\alpha=0$=MLE, $\alpha=1$=Laplace |
| 零概率问题 | $P(x_i\|\omega) = 0 \to P(\mathbf{x}\|\omega) = 0$ | 避免用 `alpha` | 平滑是必须的 |
| 加权和预测 | $X \cdot \ln P_{\text{c}}^T + \ln P_{\text{prior}}$ | `MultinomialNB.predict()` | 矩阵乘法一次完成 |

## 完整代码

<<< @/snippets/ml03_naive_bayes/demo.py
