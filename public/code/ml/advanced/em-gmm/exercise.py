# -*- coding: utf-8 -*-
"""
ml12 EM算法与高斯混合模型 — 练习代码
=====================================
请完成以下 TODO 任务，巩固对 EM 算法和 GMM 参数更新的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np
from scipy.stats import multivariate_normal


# ============================================================================
# TODO 1: 实现 GMM 的 E 步（责任计算）
# ============================================================================
def e_step_responsibilities(X, weights, means, covariances):
    """
    计算 E 步的责任（responsibilities）γ_{ik}。

    公式：
        γ_{ik} = π_k · N(x_i | μ_k, Σ_k) / Σ_j π_j · N(x_i | μ_j, Σ_j)

    参数:
        X: 数据 (n, d)
        weights: 混合系数 π (K,)
        means: 均值 μ (K, d)
        covariances: 协方差 Σ (K, d, d) 或 (K, d) for diag

    返回:
        resp: 责任矩阵 (n, K)，每行求和为 1
        log_likelihood: 当前参数下的对数似然
    """
    n, d = X.shape
    K = len(weights)

    # TODO: 步骤 1 —— 计算加权对数概率矩阵
    weighted_log_prob = np.zeros((n, K))
    for k in range(K):
        # log π_k + log N(x_i | μ_k, Σ_k)
        log_pi = None  # ← TODO: np.log(weights[k] + 1e-300)
        log_pdf = None  # ← TODO: multivariate_normal.logpdf(X, mean=means[k], cov=covariances[k])
        weighted_log_prob[:, k] = None  # ← TODO: log_pi + log_pdf

    # TODO: 步骤 2 —— log-sum-exp 计算每行的归一化常数
    # log_likelihood_i = log Σ_k exp(weighted_log_prob[i, k])
    # 提示：用 log-sum-exp trick 防数值溢出
    max_val = None  # ← TODO: weighted_log_prob.max(axis=1, keepdims=True)
    log_likelihood_i = None  # ← TODO: max_val + np.log(np.sum(np.exp(weighted_log_prob - max_val), axis=1))
    log_likelihood = None  # ← TODO: log_likelihood_i.sum()

    # TODO: 步骤 3 —— 归一化得到责任矩阵
    log_resp = None  # ← TODO: weighted_log_prob - log_likelihood_i[:, np.newaxis]
    resp = None  # ← TODO: np.exp(log_resp)
    # 确保每行和为 1
    resp = resp / resp.sum(axis=1, keepdims=True)

    return resp, log_likelihood


def test_e_step():
    """测试 E 步"""
    print("=" * 60)
    print("TODO 1 测试: GMM E 步")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.randn(100, 2)
    weights = np.array([0.3, 0.7])
    means = np.array([[0, 0], [3, 3]])
    covariances = np.array([np.eye(2), np.eye(2) * 2])

    resp, log_l = e_step_responsibilities(X, weights, means, covariances)

    if resp is None:
        print("  TODO 1 未完成，请补全 e_step_responsibilities 函数")
    else:
        print(f"  责任矩阵形状: {resp.shape} (期望: (100, 2))")
        print(f"  每行和: {resp[0].sum():.4f} (期望: 1.0)")
        print(f"  对数似然: {log_l:.2f}")
        assert abs(resp[0].sum() - 1.0) < 1e-6, "每行责任和应为 1"


# ============================================================================
# TODO 2: 实现 GMM 的 M 步（参数更新）
# ============================================================================
def m_step_update(X, resp):
    """
    执行 M 步，更新 GMM 参数。

    公式：
        N_k = Σ_i γ_{ik}                        — 有效样本数
        π_k = N_k / N                             — 混合系数
        μ_k = Σ_i γ_{ik} x_i / N_k                — 加权均值
        Σ_k = Σ_i γ_{ik} (x_i-μ_k)(x_i-μ_k)^T / N_k — 加权协方差

    参数:
        X: 数据 (n, d)
        resp: 责任矩阵 (n, K)

    返回:
        weights: (K,)
        means: (K, d)
        covariances: (K, d, d)
    """
    n, d = X.shape
    K = resp.shape[1]

    # TODO: 步骤 1 —— 计算每个成分的有效样本数
    Nk = None  # ← TODO: resp.sum(axis=0) + 1e-10  (加小值防除零)

    # TODO: 步骤 2 —— 更新混合系数 π_k = N_k / N
    weights = None  # ← TODO: Nk / n

    # TODO: 步骤 3 —— 更新均值 μ_k = (resp[:, k]^T @ X) / N_k
    # 提示: (K, n) @ (n, d) = (K, d)，然后除以 Nk (广播)
    means = None  # ← TODO: (resp.T @ X) / Nk[:, np.newaxis]

    # TODO: 步骤 4 —— 更新协方差 Σ_k
    covariances = np.zeros((K, d, d))
    for k in range(K):
        diff = None  # ← TODO: X - means[k]  # (n, d)
        # weighted sum: Σ_i γ_{ik} (x_i-μ_k)(x_i-μ_k)^T
        weighted_diff = None  # ← TODO: diff * resp[:, k, np.newaxis]  # (n, d)
        cov_k = None  # ← TODO: (weighted_diff.T @ diff) / Nk[k]  # (d, d)
        covariances[k] = cov_k

    return weights, means, covariances


def test_m_step():
    """测试 M 步"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: GMM M 步")
    print("=" * 60)

    np.random.seed(42)
    X = np.array([[0., 0.], [0.1, 0.1], [3., 3.], [3.1, 3.1]])
    resp = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.2, 0.8],
    ])

    weights, means, covs = m_step_update(X, resp)

    if weights is None:
        print("  TODO 2 未完成，请补全 m_step_update 函数")
    else:
        print(f"  混合系数: {weights} (期望: [0.5, 0.5] 左右)")
        print(f"  均值:\n{means}")
        print(f"  协方差形状: {covs.shape} (期望: (2, 2, 2))")
        assert abs(weights.sum() - 1.0) < 1e-6, "混合系数和应为 1"


# ============================================================================
# TODO 3: 计算 AIC 和 BIC
# ============================================================================
def compute_aic_bic(log_likelihood, n_params, n_samples):
    """
    计算 AIC 和 BIC。

    AIC = -2 * log L + 2 * p
    BIC = -2 * log L + p * log n

    参数:
        log_likelihood: 模型的对数似然值
        n_params: 模型参数个数 p
        n_samples: 样本数 n

    返回:
        (aic, bic) 元组
    """
    # TODO
    aic = None  # ← TODO: -2 * log_likelihood + 2 * n_params
    bic = None  # ← TODO: -2 * log_likelihood + n_params * np.log(n_samples)
    return aic, bic


def test_aic_bic():
    """测试 AIC/BIC"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: AIC / BIC")
    print("=" * 60)

    # 模拟：K=1 模型对数似然 -500，参数 5 个，100 样本
    aic, bic = compute_aic_bic(-500, 5, 100)

    if aic is None:
        print("  TODO 3 未完成，请补全 compute_aic_bic 函数")
    else:
        print(f"  AIC = {aic:.2f} (期望: 1010.00)")
        print(f"  BIC = {bic:.2f} (期望: 1000 + 5*ln 100 ≈ 1023.03)")
        # BIC 的惩罚 > AIC 的惩罚 (当 n > e^2 ≈ 7.4)
        assert bic > aic, "当 n > e^2 时 BIC 应 > AIC"


# ============================================================================
# TODO 4: 实现 K-Means 风格硬分配作为 GMM 极限
# ============================================================================
def hard_assignment_from_responsibilities(resp):
    """
    从软责任中提取硬分配（取概率最大的成分）。

    这种硬分配对应 K-Means 的行为——每个样本只属于一个簇。

    参数:
        resp: 责任矩阵 (n, K)，每行是一个概率分布

    返回:
        labels: (n,) 每个样本的簇标签
    """
    # TODO
    labels = None  # ← TODO: np.argmax(resp, axis=1)
    return labels


def test_hard_assignment():
    """测试硬分配"""
    print("\n" + "=" * 60)
    print("TODO 4 测试: 硬分配")
    print("=" * 60)

    resp = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.1, 0.7],
    ])
    labels = hard_assignment_from_responsibilities(resp)

    if labels is None:
        print("  TODO 4 未完成，请补全 hard_assignment_from_responsibilities 函数")
    else:
        print(f"  责任矩阵:\n{resp}")
        print(f"  硬分配: {labels} (期望: [0, 1, 2])")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml12 EM算法与高斯混合模型 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_e_step()
    test_m_step()
    test_aic_bic()
    test_hard_assignment()

    print("\n全部测试完成！")
