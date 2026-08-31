# -*- coding: utf-8 -*-
"""
===============================================================================
ml02_bayesian_decision/code/exercise.py — 贝叶斯决策理论练习
===============================================================================
本练习文件中，你需要完成以下任务：

练习目标：
  1. 实现高斯类条件密度的对数似然函数 log N(x | mu, Sigma)
  2. 实现贝叶斯后验概率的计算（使用 log-sum-exp 技巧）
  3. 实现最小风险决策（考虑不对称损失矩阵）

提示：
  - 高斯密度: log N(x|mu, Sigma) = -0.5 * [d*log(2pi) + log|Sigma| + (x-mu)^T Sigma^{-1} (x-mu)]
  - Log-Sum-Exp: log(sum exp(x_i)) = m + log(sum exp(x_i - m)) where m = max(x_i)
  - 条件风险: R(alpha_i|x) = sum_j loss[i,j] * P(w_j|x)

运行方式：
  python exercise.py
===============================================================================
"""

import numpy as np
from scipy.special import logsumexp


# ============================================================================
# 任务 1: 实现对数似然函数 (约 12 行)
# ============================================================================

def compute_log_likelihood(X, mu, cov):
    """
    计算多元高斯分布的对数似然 log N(X | mu, Sigma)。

    公式:
      log N(X|mu, Sigma) =
        -0.5 * (d * log(2*pi) + log|Sigma| + (X-mu)^T Sigma^{-1} (X-mu))

    参数:
        X: (n, d), n 个样本，d 维特征
        mu: (d,), 均值向量
        cov: (d, d), 协方差矩阵
    返回:
        log_lik: (n,), 每个样本的对数似然
    """
    d = X.shape[1]
    # TODO: 完成以下步骤
    # 1. 计算 cov 的逆（用 np.linalg.pinv 保证鲁棒性）
    # 2. 计算 cov 的行列式的对数（用 np.linalg.slogdet, 取 sign 和 logdet 的第2个返回值 logdet）
    # 3. 计算 X 与 mu 的差值 diff = X - mu (shape: (n, d))
    # 4. 计算马氏距离: mahal[i] = diff[i] @ cov_inv @ diff[i].T
    #    技巧: (diff @ cov_inv) 得到 (n, d), 然后逐元素乘以 diff, 再 sum(axis=1)
    # 5. 带入高斯公式: -0.5 * (d * log(2*pi) + log_det + mahal)
    # --- BEGIN YOUR CODE ---
    cov_inv = np.linalg.pinv(cov)
    sign, log_det = np.linalg.slogdet(cov)
    diff = X - mu  # (n, d)
    # mahal = sum_j (diff @ cov_inv)_j * diff_j  for each sample
    mahal = np.sum((diff @ cov_inv) * diff, axis=1)
    log_lik = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return log_lik


# ============================================================================
# 任务 2: 实现后验概率计算 (约 8 行)
# ============================================================================

def compute_posteriors(log_likelihoods, priors):
    """
    从对数似然和先验概率计算后验概率。

    公式:
      P(w_j | x) = exp(log_lik_j + log_prior_j - logsumexp_j(log_lik_j + log_prior_j))

    参数:
        log_likelihoods: (n, C), log P(x_i | w_j)
        priors: (C,), P(w_j)
    返回:
        posteriors: (n, C), P(w_j | x_i), 每行和为 1
    """
    # TODO:
    # 1. 计算 log_prior = log(priors) (shape: (C,))
    # 2. 计算 log_joint = log_likelihoods + log_prior (broadcasting: (n,C) + (C,) → (n,C))
    # 3. 计算 log_evidence = logsumexp(log_joint, axis=1, keepdims=True)
    # 4. 计算 log_posteriors = log_joint - log_evidence
    # 5. 返回 exp(log_posteriors)
    # --- BEGIN YOUR CODE ---
    log_prior = np.log(priors)  # (C,)
    log_joint = log_likelihoods + log_prior  # (n, C)
    log_evidence = logsumexp(log_joint, axis=1, keepdims=True)  # (n, 1)
    log_posteriors = log_joint - log_evidence
    posteriors = np.exp(log_posteriors)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return posteriors


# ============================================================================
# 任务 3: 实现最小风险决策 (约 10 行)
# ============================================================================

def minimum_risk_decision(posteriors, loss_matrix):
    """
    最小风险决策: 选择使条件风险最小的行动。

    R(alpha_i | x) = sum_j loss_matrix[i, j] * P(w_j | x)
    alpha* = argmin_i R(alpha_i | x)

    参数:
        posteriors: (n, C), P(w_j | x_i)
        loss_matrix: (C, C), loss[i,j] = cost of predicting class i when truth is j
    返回:
        predictions: (n,), 最优行动索引
    """
    n_samples, n_classes = posteriors.shape
    # TODO:
    # 1. 初始化 risks = np.zeros((n_samples, n_classes))
    # 2. 对每个行动 (action) i 和每个真实类别 j:
    #    risks[:, i] += loss_matrix[i, j] * posteriors[:, j]
    # 3. 返回 np.argmin(risks, axis=1)
    # --- BEGIN YOUR CODE ---
    risks = np.zeros((n_samples, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            risks[:, i] += loss_matrix[i, j] * posteriors[:, j]
    predictions = np.argmin(risks, axis=1)
    # --- END YOUR CODE ---
    # ^^^ TODO: 请删除上面的实现代码，自己重写 ^^^
    return predictions


# ============================================================================
# 验证代码
# ============================================================================

def test_log_likelihood():
    """验证对数似然计算。"""
    print("[测试 1] 对数似然函数...")
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mu = np.array([0.0, 0.0])
    cov = np.eye(2)  # 标准高斯

    log_lik = compute_log_likelihood(X, mu, cov)

    # 手动验证: log N(0|0,1) = log(1/sqrt(2*pi)) ≈ -0.919 (per dimension)
    # 2D standard normal at 0: -log(2*pi) ≈ -1.838
    expected_0 = -np.log(2 * np.pi)  # = -1.8379...
    assert np.abs(log_lik[0] - expected_0) < 0.01, \
        f"log_lik[0] error: got {log_lik[0]:.4f}, expected {expected_0:.4f}"
    print(f"  [PASS] 对数似然计算正确! log N([0,0]|0,I) = {log_lik[0]:.4f}")


def test_posteriors():
    """验证后验概率计算。"""
    print("[测试 2] 后验概率...")

    # 两类别，一维特征
    log_lik = np.array([
        [-2.0, -1.0],  # sample 0: class 1 more likely
        [-3.0, -0.5],  # sample 1: class 1 much more likely
    ])
    priors = np.array([0.3, 0.7])

    posteriors = compute_posteriors(log_lik, priors)

    # 每行和应为 1
    assert np.allclose(posteriors.sum(axis=1), 1.0), "后验概率每行和不为1"
    # sample 1 的后验 P(class 1|x) 应该更大（因为似然差距大 + 先验也大）
    assert posteriors[1, 1] > posteriors[1, 0], "分类错误"
    print(f"  [PASS] 后验概率计算正确! P(class 1|x) for sample 1 = {posteriors[1,1]:.4f}")


def test_risk_decision():
    """验证最小风险决策。"""
    print("[测试 3] 最小风险决策...")

    # 2 类，2 样本
    posteriors = np.array([
        [0.6, 0.4],  # 样本 0: 更可能是 class 0
        [0.3, 0.7],  # 样本 1: 更可能是 class 1
    ])

    # 对称损失 (0-1 loss): 应等价于最大后验
    loss_sym = np.array([[0, 1], [1, 0]])
    pred_sym = minimum_risk_decision(posteriors, loss_sym)
    assert pred_sym[0] == 0, f"对称损失的样本0预测错误: {pred_sym[0]}"
    assert pred_sym[1] == 1, f"对称损失的样本1预测错误: {pred_sym[1]}"

    # 不对称损失: 预测 class 0 当真实为 class 1 时代价很大 (FN cost = 10)
    loss_asym = np.array([[0, 10], [1, 0]])
    pred_asym = minimum_risk_decision(posteriors, loss_asym)

    # 样本 0: P(0|x)=0.6, P(1|x)=0.4
    # risk(action=0) = 0*0.6 + 10*0.4 = 4
    # risk(action=1) = 1*0.6 + 0*0.4 = 0.6
    # → 选 action=1 (虽然 class 0 的概率更高!)
    assert pred_asym[0] == 1, \
        f"不对称损失: 应选 action=1 (风险更低) 但选了 {pred_asym[0]}"

    print(f"  [PASS] 最小风险决策正确! 不对称损失下改变了决策")


if __name__ == "__main__":
    print("=" * 60)
    print("ml02_bayesian_decision exercise.py — 贝叶斯决策练习")
    print("=" * 60)

    try:
        test_log_likelihood()
        test_posteriors()
        test_risk_decision()
        print("\n" + "=" * 60)
        print("所有测试通过!")  # All tests passed
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
