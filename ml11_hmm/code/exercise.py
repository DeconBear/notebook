# -*- coding: utf-8 -*-
"""
ml11 隐马尔可夫模型 (HMM) — 练习代码
=====================================
请完成以下 TODO 任务，巩固对前向算法、Viterbi 算法和 HMM 的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np


# ============================================================================
# TODO 1: 实现前向算法的递推步骤
# ============================================================================
def forward_algorithm(pi, A, B, observations):
    """
    前向算法（不带缩放），计算 P(X | λ)。

    递推公式:
        α_1(i) = π_i · B[i, x_1]
        α_t(j) = [Σ_i α_{t-1}(i) · A[i, j]] · B[j, x_t]

    参数:
        pi: 初始分布 (N,)
        A: 转移矩阵 (N, N)
        B: 发射矩阵 (N, M)
        observations: 观测序列（索引），长度 T

    返回:
        alphas: α 矩阵 (T, N)
        prob: P(X | λ) — 注意可能因下溢为 0，但逻辑应正确
    """
    N = len(pi)
    T = len(observations)
    alphas = np.zeros((T, N))

    # TODO: 初始化 t=1
    o1 = observations[0]
    for i in range(N):
        alphas[0, i] = None  # ← TODO: pi[i] * B[i, o1]

    # TODO: 递推 t = 2, ..., T
    for t in range(1, T):
        ot = observations[t]
        for j in range(N):
            # α_t(j) = [Σ_i α_{t-1}(i) · A[i, j]] · B[j, o_t]
            sum_term = None  # ← TODO: np.dot(alphas[t-1, :], A[:, j])
            alphas[t, j] = None  # ← TODO: sum_term * B[j, ot]

    # TODO: 终止
    prob = None  # ← TODO: alphas[T-1, :].sum()

    return alphas, prob


def test_forward():
    """测试前向算法"""
    print("=" * 60)
    print("TODO 1 测试: 前向算法")
    print("=" * 60)

    # 简单的 2 状态 HMM
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.5], [0.2, 0.8]])  # 状态0产生0/1均衡，状态1偏1

    obs = [0, 1, 0]  # 观测序列

    alphas, prob = forward_algorithm(pi, A, B, obs)

    if alphas is None:
        print("  TODO 1 未完成，请补全 forward_algorithm 函数")
    else:
        print(f"  α 矩阵形状: {alphas.shape} (期望: (3, 2))")
        print(f"  α:\n{alphas}")
        print(f"  P(X | λ) = {prob:.6f}")
        # 检查：概率应在 (0, 1] 之间
        assert 0 < prob <= 1.0 + 1e-10, "概率应在 (0, 1] 内"


# ============================================================================
# TODO 2: 实现 Viterbi 算法的回溯步骤
# ============================================================================
def viterbi_decoding(pi, A, B, observations):
    """
    Viterbi 算法：找到最可能的状态序列（使用 log 空间）。

    递推:
        δ_1(i) = log(π_i) + log(B[i, x_1])
        δ_t(j) = max_i [δ_{t-1}(i) + log(A[i, j])] + log(B[j, x_t])
        ψ_t(j) = argmax_i [δ_{t-1}(i) + log(A[i, j])]

    参数:
        pi, A, B, observations: 同 forward_algorithm

    返回:
        best_path: 最优状态序列（索引），长度 T
        log_prob: 最优路径的对数概率
    """
    N = len(pi)
    T = len(observations)
    delta = np.zeros((T, N))      # log δ
    psi = np.zeros((T, N), dtype=int)  # 回溯指针

    eps = 1e-300  # 防止 log(0)

    # 初始化 t=1
    o1 = observations[0]
    for i in range(N):
        delta[0, i] = np.log(pi[i] + eps) + np.log(B[i, o1] + eps)

    # 递推
    for t in range(1, T):
        ot = observations[t]
        for j in range(N):
            # TODO: 计算 candidates = δ_{t-1}(i) + log(A[i, j])
            candidates = None  # ← TODO: delta[t-1, :] + np.log(A[:, j] + eps)
            # TODO: 找最大值及其索引
            best_val = None    # ← TODO: np.max(candidates)
            best_idx = None    # ← TODO: np.argmax(candidates)
            # TODO: 更新 delta 和 psi
            delta[t, j] = None  # ← TODO: best_val + np.log(B[j, ot] + eps)
            psi[t, j] = None    # ← TODO: best_idx

    # TODO: 终止 — 找到最优终点
    best_final_state = None  # ← TODO: np.argmax(delta[T-1, :])
    log_prob = None           # ← TODO: delta[T-1, best_final_state]

    # TODO: 回溯
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_final_state
    for t in range(T-2, -1, -1):
        best_path[t] = None  # ← TODO: psi[t+1, best_path[t+1]]

    return best_path, log_prob


def test_viterbi():
    """测试 Viterbi 算法"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: Viterbi 解码")
    print("=" * 60)

    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.5], [0.2, 0.8]])

    obs = [0, 1, 0]
    best_path, log_prob = viterbi_decoding(pi, A, B, obs)

    if best_path is None:
        print("  TODO 2 未完成，请补全 viterbi_decoding 函数")
    else:
        print(f"  最优路径: {best_path}")
        print(f"  对数概率: {log_prob:.4f}")
        # 检查：路径长度应等于观测长度
        assert len(best_path) == len(obs), "路径长度应等于观测长度"


# ============================================================================
# TODO 3: 计算平稳分布
# ============================================================================
def stationary_distribution(A, tol=1e-10, max_iter=10000):
    """
    通过反复迭代计算马尔可夫链的平稳分布。

    方法: π^{(n+1)} = π^{(n)} A，直到收敛。
    或者直接解: π = πA, Σπ = 1 的线性方程组。

    参数:
        A: 转移矩阵 (N, N)
        tol: 收敛容忍度
        max_iter: 最大迭代次数

    返回:
        pi: 平稳分布 (N,)
    """
    N = A.shape[0]
    pi = np.ones(N) / N  # 初始均匀分布

    # TODO: 迭代 π ← π A 直到收敛
    for _ in range(max_iter):
        pi_new = None  # ← TODO: pi @ A  (或 pi.dot(A))
        # TODO: 检查是否收敛 ||pi_new - pi|| < tol
        # 如果收敛，break
        if None:  # ← TODO
            break
        pi = pi_new

    return pi


def test_stationary():
    """测试平稳分布"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: 平稳分布")
    print("=" * 60)

    A = np.array([[0.5, 0.3, 0.2],
                   [0.2, 0.6, 0.2],
                   [0.1, 0.3, 0.6]])
    pi = stationary_distribution(A)

    if pi is None or np.allclose(pi, np.ones(3)/3):
        print("  TODO 3 可能未完成，请补全 stationary_distribution 函数")
    else:
        print(f"  平稳分布: {pi}")
        print(f"  验证 πA ≈ π: {pi @ A}")
        assert np.allclose(pi @ A, pi, atol=1e-4), "πA 应约等于 π"


# ============================================================================
# TODO 4: 用 HMM 生成观测序列
# ============================================================================
def sample_hmm(pi, A, B, T, random_state=None):
    """
    从 HMM 中采样一条观测序列和对应的状态序列。

    步骤:
        1. 从 π 采样初始状态 z_1
        2. 从 B[z_1, :] 采样观测 x_1
        3. 对 t = 2..T: 从 A[z_{t-1}, :] 采样 z_t，从 B[z_t, :] 采样 x_t

    参数:
        pi: 初始分布 (N,)
        A: 转移矩阵 (N, N)
        B: 发射矩阵 (N, M)
        T: 序列长度
        random_state: 随机种子

    返回:
        states: 状态序列 (T,)
        observations: 观测序列 (T,)
    """
    rng = np.random.RandomState(random_state)
    N = len(pi)
    states = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)

    # TODO: 步骤 1 — 从 π 采样初始状态
    z_1 = None  # ← TODO: rng.choice(N, p=pi)
    states[0] = z_1

    # TODO: 步骤 2 — 从 B[z_1, :] 采样初始观测
    obs[0] = None  # ← TODO: rng.choice(B.shape[1], p=B[z_1])

    # TODO: 步骤 3 — 递推生成序列
    for t in range(1, T):
        z_prev = states[t-1]
        states[t] = None  # ← TODO: rng.choice(N, p=A[z_prev])
        obs[t] = None     # ← TODO: rng.choice(B.shape[1], p=B[states[t]])

    return states, obs


def test_sample():
    """测试 HMM 采样"""
    print("\n" + "=" * 60)
    print("TODO 4 测试: HMM 序列生成")
    print("=" * 60)

    pi = np.array([0.5, 0.5])
    A = np.array([[0.8, 0.2], [0.1, 0.9]])
    B = np.array([[0.7, 0.3], [0.4, 0.6]])

    states, obs = sample_hmm(pi, A, B, T=20, random_state=42)

    if states is None:
        print("  TODO 4 未完成，请补全 sample_hmm 函数")
    else:
        print(f"  状态序列: {states}")
        print(f"  观测序列: {obs}")
        assert len(states) == 20, "序列长度应为 20"
        # 检查状态值在合法范围
        assert all(0 <= s < len(pi) for s in states), "状态值超出范围"


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml11 隐马尔可夫模型 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_forward()
    test_viterbi()
    test_stationary()
    test_sample()

    print("\n全部测试完成！")
