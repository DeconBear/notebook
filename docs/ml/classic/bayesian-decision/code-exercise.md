---
title: "ml02 贝叶斯决策理论 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml02 贝叶斯决策理论 — exercise.py 练习指南

<a href="/notebook/code/ml/classic/bayesian-decision/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全高斯对数似然、贝叶斯后验概率和最小风险决策三个模块，从代码层面掌握贝叶斯决策理论的核心计算。

## 预备知识

在开始练习前，确保你已经理解了以下概念：

- 多元高斯密度的对数形式：$\ln \mathcal{N}(\mathbf{x}|\mu,\Sigma) = -\frac{1}{2}[d\ln(2\pi) + \ln|\Sigma| + (\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)]$
- Log-Sum-Exp 技巧：$\ln\sum e^{x_i} = m + \ln\sum e^{x_i-m}$（`m = max(x_i)`），防止指数溢出
- 后验概率归一化：$P(\omega_j|\mathbf{x}) = \frac{\exp(\ln P(\mathbf{x},\omega_j))}{\sum_k \exp(\ln P(\mathbf{x},\omega_k))}$
- 条件风险：$R(\alpha_i|\mathbf{x}) = \sum_j \lambda(\alpha_i|\omega_j) P(\omega_j|\mathbf{x})$

## 任务清单

### 任务1：实现对数似然 `compute_log_likelihood(X, mu, cov)`

- **实现步骤**：
  1. 用 `np.linalg.pinv(cov)` 计算协方差矩阵的伪逆
  2. 用 `np.linalg.slogdet(cov)` 计算协方差矩阵行列式的对数（注意返回值是 `(sign, logdet)`，取 `logdet`）
  3. 计算差值 `diff = X - mu`
  4. 计算马氏距离平方：`mahal = np.sum((diff @ cov_inv) * diff, axis=1)`（向量化批量运算）
  5. 返回 `-0.5 * (d * np.log(2*np.pi) + log_det + mahal)`
- **关键细节**：`np.linalg.pinv` vs `np.linalg.inv`——当协方差接近奇异时，伪逆给出稳定解

### 任务2：实现后验概率 `compute_posteriors(log_likelihoods, priors)`

- **实现步骤**：
  1. `log_prior = np.log(priors)`
  2. `log_joint = log_likelihoods + log_prior`（NumPy broadcasting：`(n,C) + (C,) → (n,C)`）
  3. `log_evidence = logsumexp(log_joint, axis=1, keepdims=True)`
  4. `log_posteriors = log_joint - log_evidence`
  5. 返回 `np.exp(log_posteriors)`
- **直觉**：在对数空间中，乘法变加法（$\ln(ab) = \ln a + \ln b$），除法变减法（$\ln(a/b) = \ln a - \ln b$），一切都变成了简单的加减运算

### 任务3：实现最小风险决策 `minimum_risk_decision(posteriors, loss_matrix)`

- **实现步骤**：
  1. 初始化 `risks = np.zeros((n_samples, n_classes))`
  2. 双重循环：对每个行动 $i$ 和每个真实类别 $j$，累加 $\lambda_{ij} \times P(\omega_j|\mathbf{x})$
  3. 返回 `np.argmin(risks, axis=1)`
- **关键测试**：不对称损失矩阵应能改变决策结果——当将 "真 $\omega_1$ 判为 $\omega_0$" 的代价很高时，分类器倾向于判为 $\omega_1$

## 验证标准

运行 `python exercise.py`：

1. `test_log_likelihood()`：2D 标准高斯在原点的对数似然应为 $-\ln(2\pi) \approx -1.838$
2. `test_posteriors()`：每行后验概率之和应为 1；似然差距大 + 先验大 → 后验更大
3. `test_risk_decision()`：对称损失下的决策应等价于最大后验；不对称损失应改变决策


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/classic/bayesian-decision/code/exercise.py`
