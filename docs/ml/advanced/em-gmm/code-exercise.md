---
title: "ml12 EM算法与高斯混合模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml12 EM算法与高斯混合模型 — exercise.py 练习指南

<a href="/notebook/code/ml/advanced/em-gmm/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现 GMM 的 E 步和 M 步、计算 AIC/BIC 以及将软分配转为硬分配，从代码层面深入理解 EM 算法的数学原理和 GMM 的参数学习过程。

## 预备知识

- E 步责任公式：$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}$
- M 步更新公式：$\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$ 等
- log-sum-exp trick：数值稳定的 softmax 归一化
- AIC = $-2\log L + 2p$，BIC = $-2\log L + p\log n$
- K-Means 是 GMM 当 $\Sigma \to 0$ 时的极限（硬分配）

## 任务清单

### 任务1：GMM E 步 `e_step_responsibilities(X, weights, means, covariances)`

- **计算流程**：
  1. 对每个成分 $k$，计算 $\log \pi_k + \log \mathcal{N}(x_i | \mu_k, \Sigma_k)$
  2. 用 log-sum-exp trick 计算每行的对数归一化常数
  3. 减去归一化常数得到 $\log \gamma_{ik}$，再 exp 回概率空间
- **log-sum-exp 实现**：`max_val = weighted_log_prob.max(axis=1, keepdims=True)`，然后 `max_val + np.log(np.sum(np.exp(weighted_log_prob - max_val), axis=1))`
- **scipy 工具**：`multivariate_normal.logpdf(X, mean=mu, cov=Sigma)` 直接给出对数概率密度

### 任务2：GMM M 步 `m_step_update(X, resp)`

- **关键矩阵操作**：
  - 有效样本数：`Nk = resp.sum(axis=0)`（每列求和）
  - 加权均值：`(resp.T @ X) / Nk[:, np.newaxis]`——$(K, n) \times (n, d) = (K, d)$
  - 加权协方差：`(weighted_diff.T @ diff) / Nk[k]`——$(d, n) \times (n, d) = (d, d)$
- **加权 diff**：`diff * resp[:, k, np.newaxis]` 实现逐元素的权重乘法（利用 NumPy 广播）

### 任务3：AIC / BIC 计算 `compute_aic_bic(log_likelihood, n_params, n_samples)`

- 直接套公式：`aic = -2 * log_likelihood + 2 * n_params`
- BIC 的惩罚项用 $\log n$ 而非 2：当 $n > e^2 \approx 7.4$ 时 BIC 惩罚比 AIC 更重

### 任务4：软分配转硬分配 `hard_assignment_from_responsibilities(resp)`

- 最简单但重要的任务：`np.argmax(resp, axis=1)`
- 这体现了 K-Means 与 GMM 的核心差异——GMM 保留概率信息，K-Means 只做 0/1 分配


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/em-gmm/code/exercise.py`
