# -*- coding: utf-8 -*-
"""
ml10 蒙特卡洛方法 — 练习代码
=============================
请完成以下 TODO 任务，巩固对 MC 积分、重要性采样和 Metropolis-Hastings 的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np
from scipy.stats import norm


# ============================================================================
# TODO 1: 实现简单的蒙特卡洛积分
# ============================================================================
def mc_integral(f, a, b, N=10000):
    """
    用蒙特卡洛方法估计定积分 ∫_a^b f(x) dx。

    方法：从均匀分布 U(a,b) 采样 N 个点，用样本均值近似：
        ∫_a^b f(x) dx ≈ (b - a) * (1/N) * sum f(x_i)

    参数:
        f: 被积函数 callable
        a, b: 积分区间
        N: 样本数

    返回:
        estimate: 积分估计值
        std_error: 标准误差（估计的标准差）
    """
    # TODO: 步骤 1 —— 从 U(a, b) 采样 N 个点
    x = None  # ← TODO: np.random.uniform(a, b, N)

    # TODO: 步骤 2 —— 计算样本均值乘以区间长度
    estimate = None  # ← TODO: (b - a) * f(x).mean()

    # TODO: 步骤 3 —— 计算标准误差 σ/√N
    # 提示: std_error = (b - a) * np.std(f(x)) / np.sqrt(N)
    std_error = None  # ← TODO

    return estimate, std_error


def test_mc_integral():
    """测试 MC 积分"""
    print("=" * 60)
    print("TODO 1 测试: 蒙特卡洛积分")
    print("=" * 60)

    # 测试 ∫_0^1 x^2 dx = 1/3 ≈ 0.3333
    f = lambda x: x**2
    est, se = mc_integral(f, 0, 1, N=50000)

    if est is None:
        print("  TODO 1 未完成，请补全 mc_integral 函数")
    else:
        true_val = 1.0 / 3.0
        print(f"  ∫_0^1 x^2 dx: 估计值={est:.5f}, 真实值={true_val:.5f}")
        print(f"  标准误差: {se:.5f}")
        print(f"  是否在 3σ 内: {abs(est - true_val) < 3 * se} (期望: True)")
        assert abs(est - true_val) < 3 * se + 0.01, "估计值应在 3σ 置信区间内"


# ============================================================================
# TODO 2: 实现重要性采样的权重计算
# ============================================================================
def importance_sampling_weights(samples, p_logpdf, q_logpdf):
    """
    计算重要性采样权重 w_i = p(x_i) / q(x_i)。

    使用 log-space 计算防数值溢出：
        log w_i = log p(x_i) - log q(x_i)
        w_i = exp(log w_i - max(log w))  # 减最大值防溢出（数值稳定）
        w_i = w_i / sum(w_i)             # 归一化

    参数:
        samples: 从 q 中采样的样本，shape (N,)
        p_logpdf: 目标分布的对数 PDF 函数
        q_logpdf: 提议分布的对数 PDF 函数

    返回:
        weights: 归一化后的权重，shape (N,)
    """
    # TODO: 步骤 1 —— 计算对数权重
    log_p_vals = None  # ← TODO: p_logpdf(samples)
    log_q_vals = None  # ← TODO: q_logpdf(samples)
    log_weights = None  # ← TODO: log_p_vals - log_q_vals

    # TODO: 步骤 2 —— 数值稳定化（减去最大值再 exp）
    log_weights_stable = None  # ← TODO: log_weights - log_weights.max()

    # TODO: 步骤 3 —— exp + 归一化
    weights = None  # ← TODO: np.exp(log_weights_stable)
    weights_normalized = None  # ← TODO: weights / weights.sum()

    return weights_normalized


def test_is_weights():
    """测试重要性采样权重"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: 重要性采样权重")
    print("=" * 60)

    # 目标: N(0, 1), 提议: N(2, 1)
    np.random.seed(42)
    samples = np.random.randn(1000) + 2  # ~ N(2, 1)
    p_logpdf = lambda x: norm.logpdf(x, 0, 1)
    q_logpdf = lambda x: norm.logpdf(x, 2, 1)

    weights = importance_sampling_weights(samples, p_logpdf, q_logpdf)

    if weights is None:
        print("  TODO 2 未完成，请补全 importance_sampling_weights 函数")
    else:
        print(f"  权重和: {weights.sum():.4f} (期望: 1.0)")
        print(f"  权重均值: {weights.mean():.4f} (期望: 0.001)")
        print(f"  权重标准差: {weights.std():.4f}")
        assert abs(weights.sum() - 1.0) < 1e-6, "权重和应为 1"
        # 检查 ESS: (sum w)^2 / sum w^2
        ESS = 1.0 / np.sum(weights ** 2)
        print(f"  有效样本量 (ESS): {ESS:.1f} / {len(samples)}")


# ============================================================================
# TODO 3: 实现 Metropolis-Hastings 的单步更新
# ============================================================================
def mh_step(x_current, target_logpdf, proposal_std, rng=None):
    """
    执行 Metropolis-Hastings 算法的一步更新。

    步骤:
        1. 提议: x' = x_current + ε, ε ~ N(0, proposal_std^2)
        2. 计算 log_alpha = log p(x') - log p(x_current)
        3. 接受/拒绝: if log(rand()) < log_alpha: x_next = x', else x_next = x_current

    参数:
        x_current: 当前状态（标量或数组）
        target_logpdf: 目标分布的对数密度函数
        proposal_std: 提议分布的标准差
        rng: 随机数生成器（可选）

    返回:
        x_next: 下一步状态
        accepted: 是否被接受 (bool)
    """
    if rng is None:
        rng = np.random

    # TODO: 步骤 1 —— 生成提议
    x_proposal = None  # ← TODO: x_current + rng.randn(*x_current.shape) * proposal_std

    # TODO: 步骤 2 —— 计算对数接受率
    log_p_current = None   # ← TODO: target_logpdf(x_current)
    log_p_proposal = None  # ← TODO: target_logpdf(x_proposal)
    log_alpha = None       # ← TODO: log_p_proposal - log_p_current

    # TODO: 步骤 3 —— 接受/拒绝判断
    # 提示: if np.log(rng.rand()) < log_alpha: 接受，否则拒绝
    accepted = None  # ← TODO
    if accepted:
        x_next = None  # ← TODO: x_proposal
    else:
        x_next = None  # ← TODO: x_current.copy()

    return x_next, accepted


def test_mh_step():
    """测试 MH 单步更新"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: MH 单步更新")
    print("=" * 60)

    # 目标分布: N(0, 1) — 对数 PDF = -0.5 * x^2 + const
    target_logpdf = lambda x: -0.5 * x**2
    rng = np.random.RandomState(42)

    # 从 x=3 开始，连续执行 1000 步
    x = 3.0
    samples = np.zeros(1000)
    n_accepted = 0
    for i in range(1000):
        x, acc = mh_step(x, target_logpdf, proposal_std=1.5, rng=rng)
        samples[i] = x
        if acc:
            n_accepted += 1

    if n_accepted == 0 and samples[0] == 3.0:
        # 全部拒绝且值未变，说明 TODO 未完成
        print("  TODO 3 可能未完成，请检查 mh_step 实现")
    else:
        print(f"  接受率: {n_accepted/1000:.3f} (期望: ~0.3-0.5)")
        print(f"  后 500 步均值: {samples[500:].mean():.3f} (期望: ~0.0)")
        print(f"  后 500 步标准差: {samples[500:].std():.3f} (期望: ~1.0)")
        # 简单验证：后 500 步的均值和标准差应接近 (0, 1)
        assert abs(samples[500:].mean()) < 0.3, "均值应接近 0"
        assert 0.5 < samples[500:].std() < 1.5, "标准差应接近 1"


# ============================================================================
# TODO 4: 比较 MC 积分与确定性积分
# ============================================================================
def compare_mc_vs_analytic():
    """
    对 f(x) = sin(x)，比较 MC 积分与解析解的精度随 N 的变化。

    积分: ∫_0^π sin(x) dx = 2（解析解）

    对不同的 N 值（100, 1000, 10000），用 MC 估计积分，
    记录估计值和绝对误差，验证误差 ~ O(1/√N)。
    """
    print("\n" + "=" * 60)
    print("TODO 4 测试: MC vs 解析解")
    print("=" * 60)

    f = np.sin
    true_val = 2.0  # ∫_0^π sin(x) dx = 2

    print(f"  解析解: {true_val}")
    print(f"  {'N':<10} {'估计值':<12} {'绝对误差':<12} {'1/√N':<10}")

    for N in [100, 1000, 10000, 100000]:
        # TODO: 多次运行求平均误差
        errors = []
        for _ in range(10):  # 10 次重复
            x = None  # ← TODO: np.random.uniform(0, np.pi, N)
            est = None  # ← TODO: np.pi * f(x).mean()
            errors.append(abs(est - true_val))
        mean_error = np.mean(errors)

        # TODO: 理论收敛率对比: 理论误差 ∝ 1/√N
        theoretical = None  # ← TODO: 1.0 / np.sqrt(N)
        print(f"  {N:<10} {np.pi * f(np.random.uniform(0, np.pi, N)).mean():<12.5f} "
              f"{mean_error:<12.5f} {theoretical:<10.5f}")

    print(f"\n  理论: 误差 ∝ 1/√N, N 增大 100 倍则误差减小约 10 倍")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml10 蒙特卡洛方法 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_mc_integral()
    test_is_weights()
    test_mh_step()
    compare_mc_vs_analytic()

    print("\n全部测试完成！")
