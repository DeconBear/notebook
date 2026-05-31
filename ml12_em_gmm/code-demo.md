---
title: "ml12 EM算法与高斯混合模型 — demo.py"
---

# ml12 EM算法与高斯混合模型 — demo.py 代码详解

<a href="../code/ml12_em_gmm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd ml12_em_gmm/code
python demo.py
```

## 代码逐段详解

### 第1步：GMM 的 E 步——责任计算

```python
def _e_step(self, X):
    weighted_log_prob = np.zeros((n, K))
    for k in range(K):
        log_pi_k = np.log(self.weights_[k] + 1e-300)
        log_pdf_k = self._log_multivariate_normal(X, self.means_[k], self.covariances_[k])
        weighted_log_prob[:, k] = log_pi_k + log_pdf_k

    # log-sum-exp: 数值稳定的 softmax 归一化
    log_likelihood_per_sample = self._logsumexp(weighted_log_prob, axis=1)
    log_resp = weighted_log_prob - log_likelihood_per_sample[:, np.newaxis]
    resp = np.exp(log_resp)
```

E 步的核心是计算每个样本 $i$ 对每个成分 $k$ 的**责任（responsibility）**：

$$
\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x_i | \mu_j, \Sigma_j)}
$$

分子是样本 $i$ 在成分 $k$ 下的（加权）概率密度，分母是归一化常数。在 log 空间中，分子/分母的除法变成了减法，数值上更稳定。

**log-sum-exp trick**：计算 $\log \sum_k e^{a_k}$ 时，直接做 exp 可能导致上溢。解决方案是先减去最大值：

$$
\log\sum_k e^{a_k} = \max a + \log\sum_k e^{a_k - \max a}
$$

减去最大值后，指数的参数都 $\le 0$，结果 $\le 1$，不存在上溢风险。

### 第2步：GMM 的 M 步——参数更新

```python
def _m_step(self, X, resp):
    Nk = resp.sum(axis=0) + 1e-10                 # 每个成分的有效样本数
    self.weights_ = Nk / n                         # π_k = N_k / N
    self.means_ = (resp.T @ X) / Nk[:, np.newaxis]  # μ_k = 加权均值
    for k in range(K):
        diff = X - self.means_[k]
        weighted_diff = diff * resp[:, k, np.newaxis]
        cov_k = (weighted_diff.T @ diff) / Nk[k]   # Σ_k = 加权协方差
        cov_k += np.eye(d) * self.reg_covar         # 正则化防奇异
```

M 步的三个更新公式有着优雅的统计解释：

- **混合系数** $\pi_k = N_k / N$：每个成分的"有效样本比例"
- **均值** $\mu_k = \sum_i \gamma_{ik} x_i / N_k$：用责任作为权重的加权均值
- **协方差** $\Sigma_k = \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T / N_k$：用责任作为权重的加权协方差

对比 K-Means：如果把责任替换为 0/1 硬分配，M 步公式就退化为简单的样本均值/协方差——这正是 K-Means 的更新规则。

### 第3步：EM 收敛性

```python
change = log_likelihood - prev_log_likelihood
if iteration > 0 and abs(change) < self.tol:
    break
```

EM 的一个精妙性质：**对数似然在每次迭代后单调不降**。这是数学保证的，不需要学习率调参。收敛曲线（`plot_em_convergence`）清晰展示了这一单调递增行为——前几步快速提升，随后进入缓慢的精细化调整期。

### 第4步：GMM vs K-Means 对比

```python
def plot_gmm_vs_kmeans(X):
```

在**各向异性**（anisotropic）的非球形数据上：
- K-Means 用圆形/硬边界划分——无法捕捉数据的拉伸和旋转
- GMM (full covariance) 用椭圆形等概率轮廓——精确捕捉每个成分的协方差结构
- GMM 的软分配在簇边界处产生颜色混合（过渡色），反映了"这个点属于多个簇的不确定性"

### 第5步：AIC / BIC 模型选择

```python
aic = -2 * log_l + 2 * n_params
bic = -2 * log_l + n_params * np.log(n)
```

两个准则都在"拟合度（-2 log L）"和"复杂度（参数惩罚项）"之间做权衡：
- AIC 的参数惩罚系数固定为 2
- BIC 的惩罚系数为 $\log N$（样本越多，越倾向于选简单模型）

BIC 源自贝叶斯框架（近似边际似然），在大样本下具有一致性——它会选择真实的模型复杂度。AIC 则倾向于最小化预测误差（通过 KL 散度），可能选偏大的模型。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| E 步（责任） | $\gamma_{ik} \propto \pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)$ | `_e_step()` | 软分配，后验概率 |
| M 步（更新） | 加权均值/协方差/混合系数 | `_m_step()` | 最大似然 + 软分配 |
| log-sum-exp | $\max a + \log\sum e^{a - \max a}$ | `_logsumexp()` | 数值稳定归一化 |
| 正则化 | $\Sigma_k \leftarrow \Sigma_k + \epsilon I$ | `reg_covar` | 防止协方差奇异 |
| AIC | $-2\log L + 2p$ | `plot_aic_bic()` | 预测导向 |
| BIC | $-2\log L + p\log N$ | `plot_aic_bic()` | 模型一致性选择 |
| K-Means 极限 | GMM when $\Sigma \to 0$ | 概念联系 | EM 退化为 Lloyd |

## 完整代码

<<< @/snippets/ml12_em_gmm/demo.py
