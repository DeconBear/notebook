---
title: "ml02 贝叶斯决策理论 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml02 贝叶斯决策理论 — demo.py 代码详解

<a href="../code/ml02_bayesian_decision/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml02_bayesian_decision/code
python demo.py
```

## 代码逐段详解

### 第1步：GaussianBayesClassifier 类 — 三种协方差假设

```python
class GaussianBayesClassifier:
    def __init__(self, cov_type='full'):
        self.cov_type = cov_type  # 'isotropic', 'shared', 'full'
```

这个分类器假设各类别的数据服从多元高斯分布：

$$
P(\mathbf{x} | \omega_i) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_i, \Sigma_i)
$$

三种 `cov_type` 对应不同的协方差假设：

- **`isotropic`**：$\Sigma_i = \sigma^2 \mathbf{I}$（各类别共享，各向同性）。这是最简单的假设，只需要估计 1 个参数（$\sigma^2$）+ 各类别的均值。决策边界是各类别均值连线的垂直平分线——等价于最近均值分类器。

- **`shared`**：$\Sigma_i = \Sigma$（各类别共享同一个协方差矩阵）。需要估计 $d(d+1)/2$ 个协方差参数。决策边界是线性超平面——等价于 LDA。

- **`full`**：每个类别有自己独立的 $\Sigma_i$。需要估计 $C \cdot d(d+1)/2$ 个协方差参数。决策边界是二次曲面——等价于 QDA。

参数越少（isotropic）→ 偏差越高，方差越低。参数越多（full）→ 偏差越低，方差越高。这正是经典的 Bias-Variance 权衡。

### 第2步：fit() — 参数估计

```python
def fit(self, X, y):
    # priors: P(omega_j) = n_j / n  (MLE)
    self.priors_[idx] = len(X_c) / len(X)
    # means: mu_j = mean of class j  (MLE)
    self.means_[idx] = np.mean(X_c, axis=0)
```

这里使用了**最大似然估计**（Maximum Likelihood Estimation, MLE）：

- 先验概率的 MLE 就是各类别的频率：$\hat{P}(\omega_j) = n_j / n$
- 均值的 MLE 就是样本均值：$\hat{\boldsymbol{\mu}}_j = \frac{1}{n_j} \sum_{x_i \in \omega_j} \mathbf{x}_i$
- 协方差矩阵的 MLE 取决于 `cov_type` 的选择

对于 `full` 协方差，代码中加了微小的正则化：

```python
cov_c = np.cov(X_c.T, bias=False)
cov_c += 1e-6 * np.eye(n_features)  # 防止奇异
```

这个 $10^{-6} \mathbf{I}$ 的"抖动"确保协方差矩阵可逆，即使样本数 < 特征数。

### 第3步：_log_likelihood() — 对数似然计算

```python
def _log_likelihood(self, X):
    for idx in range(n_classes):
        diff = X - mu
        if self.cov_type == 'isotropic':
            mahal = np.sum(diff ** 2, axis=1) / sigma2
        else:
            mahal = np.sum((diff @ cov_inv) * diff, axis=1)
        log_gauss = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
```

对数高斯密度的公式：

$$
\ln \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \Sigma) = -\frac{1}{2} \left[ d \ln(2\pi) + \ln |\Sigma| + (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right]
$$

其中第三项 $(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ 称为**马氏距离的平方**。它度量了 $\mathbf{x}$ 到 $\boldsymbol{\mu}$ 的"协方差校正"距离。

**向量化技巧**：`np.sum((diff @ cov_inv) * diff, axis=1)` 批量计算了 $n$ 个样本的马氏距离平方。解释如下：
- `diff @ cov_inv` 得到 $(n, d)$ 矩阵，每一行是 $\text{diff}_i \Sigma^{-1}$
- 再逐元素乘以 `diff`，得到 $\text{diff}_i \Sigma^{-1} \cdot \text{diff}_i$
- `sum(axis=1)` 得到每行的标量结果

### 第4步：predict_proba() — 后验概率与 Log-Sum-Exp

```python
def predict_proba(self, X):
    log_joint = log_lik + log_prior
    log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
    log_posteriors = log_joint - log_evidence
    return np.exp(log_posteriors)
```

后验概率的计算需要先用贝叶斯公式归一化：

$$
P(\omega_j | \mathbf{x}) = \frac{P(\mathbf{x} | \omega_j) P(\omega_j)}{\sum_k P(\mathbf{x} | \omega_k) P(\omega_k)}
$$

如果直接计算，分子和分母都涉及 `exp(log_joint)`，当 `log_joint` 非常负时，`exp` 会下溢到 0。解决方法是**在对数空间中做归一化**：

1. `log_joint = log_lik + log_prior`：在对数空间中计算分子
2. `log_evidence = logsumexp(log_joint)`：$\ln \sum e^{\text{log\_joint}}$（`scipy.special.logsumexp` 内部做了数值稳定性处理——先减最大值再 exp）
3. `log_posteriors = log_joint - log_evidence`：$\ln P(\omega_j | \mathbf{x}) = \ln P(\mathbf{x}, \omega_j) - \ln P(\mathbf{x})$（对数中的除法 = 减法）

最后才 `np.exp()` 得到实际的后验概率值。

### 第5步：最小风险决策

```python
def minimum_risk_predict(posteriors, loss_matrix):
    for i in range(n_classes):
        for j in range(n_classes):
            risks[:, i] += loss_matrix[i, j] * posteriors[:, j]
    return np.argmin(risks, axis=1)
```

条件风险的计算公式为：

$$
R(\alpha_i | \mathbf{x}) = \sum_{j} \lambda(\alpha_i | \omega_j) \cdot P(\omega_j | \mathbf{x})
$$

`loss_matrix[i, j]` 表示当真实类别为 $j$ 时预测类别为 $i$ 的代价。

演示中使用的不对称损失矩阵：
```python
loss_asymmetric = np.array([[0, 5],   # 将 class 1 误判为 class 0: 代价 5
                             [1, 0]])  # 将 class 0 误判为 class 1: 代价 1
```

当 FN（漏诊）代价远大于 FP（误报）代价时，分类器会更倾向于将样本判为"有病的"那个类别。

### 第6步：ROC 曲线绘制

```python
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
```

`sklearn.metrics.roc_curve` 通过变化决策阈值，返回不同 (FPR, TPR) 点。原理：
1. 按模型输出的后验概率从高到低排序
2. 逐个将阈值设为每个样本的得分，计算此时的 FPR 和 TPR
3. 连接所有点得到 ROC 曲线

AUC 使用梯形积分计算：`sklearn.metrics.auc(fpr, tpr)`。AUC 的概率解释：随机抽取一个正样本和一个负样本，正样本的得分高于负样本的概率。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 贝叶斯定理 | $P(\omega\|\mathbf{x}) \propto P(\mathbf{x}\|\omega)P(\omega)$ | `predict_proba()` | 先验 + 似然 → 后验 |
| 高斯密度 | $\mathcal{N}(\mathbf{x}\|\mu,\Sigma)$ | `_log_likelihood()` | 假设各类别为正态分布 |
| 马氏距离 | $(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)$ | `_log_likelihood()` 中 `mahal` | 协方差校正的距离 |
| 各向同性协方差 | $\Sigma = \sigma^2\mathbf{I}$ | `cov_type='isotropic'` | 1 参数，决策面为直线 |
| 共享协方差 | $\Sigma$ 各类别相同 | `cov_type='shared'` | 类似 LDA，线性决策面 |
| 独立协方差 | $\Sigma_i$ 各类别不同 | `cov_type='full'` | 类似 QDA，二次决策面 |
| Log-Sum-Exp | $\ln\sum e^{x_i} = m + \ln\sum e^{x_i-m}$ | `predict_proba()` | 数值稳定归一化 |
| 条件风险 | $R(\alpha_i\|\mathbf{x}) = \sum_j \lambda_{ij}P(\omega_j\|\mathbf{x})$ | `minimum_risk_predict()` | 不对称损失 |
| ROC 曲线 | FPR vs TPR（变阈值） | `plot_roc_auc_eer()` | 分类器排序能力 |
| AUC | $\int_0^1 \text{TPR}(\text{FPR}) d\text{FPR}$ | `auc(fpr, tpr)` | 正样本得分 > 负样本得分的概率 |
| EER | FPR = FNR 时的错误率 | 代码中 `eer_idx` | 安全与便利平衡点 |

## 完整代码

<<< @/snippets/ml02_bayesian_decision/demo.py
