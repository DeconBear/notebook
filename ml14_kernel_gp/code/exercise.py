# -*- coding: utf-8 -*-
"""
ml14 核方法与高斯过程 — 练习代码
===============================
请完成以下 TODO 任务，巩固对核方法和 GP 回归的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np


# ============================================================================
# TODO 1: 实现 RBF 核函数
# ============================================================================
def rbf_kernel_manual(X1, X2, lengthscale=1.0, variance=1.0):
    """
    手动实现 RBF 核函数。

    公式:
        k(x, x') = σ² · exp(-||x - x'||² / (2ℓ²))

    参数:
        X1: (n1, d)
        X2: (n2, d)
        lengthscale: ℓ
        variance: σ²

    返回:
        K: 核矩阵 (n1, n2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # TODO: 计算平方欧氏距离矩阵
    # 方法 1: 使用 ||x-y||² = ||x||² + ||y||² - 2x^Ty
    sq_norm1 = None  # ← TODO: np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm2 = None  # ← TODO: np.sum(X2**2, axis=1).reshape(1, -1)
    sq_dist = None   # ← TODO: sq_norm1 + sq_norm2 - 2 * X1 @ X2.T
    sq_dist = np.maximum(sq_dist, 0)  # 数值稳定性

    # TODO: 应用 RBF 公式
    K = None  # ← TODO: variance * np.exp(-0.5 * sq_dist / (lengthscale**2))
    return K


def test_rbf_kernel():
    """测试 RBF 核"""
    print("=" * 60)
    print("TODO 1 测试: RBF 核函数")
    print("=" * 60)

    X1 = np.array([[0.], [1.], [2.]])
    X2 = np.array([[0.], [1.]])
    K = rbf_kernel_manual(X1, X2, lengthscale=1.0, variance=1.0)

    if K is None:
        print("  TODO 1 未完成，请补全 rbf_kernel_manual 函数")
    else:
        print(f"  核矩阵形状: {K.shape} (期望: (3, 2))")
        print(f"  K:\n{K}")
        # 验证：对角线应为 1.0（方差=1, lengthscale=1, 距离=0）
        print(f"  K[0,0] = {K[0,0]:.4f} (期望: 1.0)")
        print(f"  K[1,0] = {K[1,0]:.4f} (期望: exp(-0.5) ≈ 0.6065)")
        assert abs(K[0, 0] - 1.0) < 1e-6, "距离 0 时核值应为 1"
        assert abs(K[1, 0] - np.exp(-0.5)) < 1e-4, "K[1,0] 应为 exp(-0.5)"


# ============================================================================
# TODO 2: 实现 GP 回归的后验均值预测
# ============================================================================
def gp_predict_mean(X_train, y_train, X_test, kernel_fn, noise_var=1e-4):
    """
    计算 GP 回归的后验预测均值（不含方差）。

    公式:
        f* = k_*^T (K + σ_n²I)^{-1} y

    参数:
        X_train: (N, d)
        y_train: (N,)
        X_test: (M, d)
        kernel_fn: 核函数
        noise_var: 观测噪声方差 σ_n²

    返回:
        mean: (M,) 后验预测均值
    """
    N = len(X_train)

    # TODO: 步骤 1 — 计算训练核矩阵 K = k(X_train, X_train)
    K = None  # ← TODO: kernel_fn(X_train, X_train)  # (N, N)

    # TODO: 步骤 2 — 计算 K + σ_n²I 和它的逆乘以 y
    K_noisy = None  # ← TODO: K + noise_var * np.eye(N)
    # 使用 Cholesky 分解求解，更稳定
    L = None  # ← TODO: np.linalg.cholesky(K_noisy)
    alpha = None  # ← TODO: np.linalg.solve(L.T, np.linalg.solve(L, y_train))  # (N,)

    # TODO: 步骤 3 — 计算测试核矩阵和预测均值
    K_test = None  # ← TODO: kernel_fn(X_test, X_train)  # (M, N)
    mean = None  # ← TODO: K_test @ alpha  # (M,)

    return mean


def test_gp_predict():
    """测试 GP 预测"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: GP 预测均值")
    print("=" * 60)

    # 简单数据：sin(x) + 噪声
    def rbf(X1, X2, lengthscale=1.0, variance=1.0):
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dist = np.maximum(sq1 + sq2 - 2 * X1 @ X2.T, 0)
        return variance * np.exp(-0.5 * sq_dist / lengthscale**2)

    X_train = np.array([[-3.], [-1.], [1.], [3.]])
    y_train = np.sin(X_train.ravel()) + 0.1 * np.random.randn(4)
    X_test = np.linspace(-4, 4, 50).reshape(-1, 1)

    kernel_fn = lambda X1, X2: rbf(X1, X2)
    mean = gp_predict_mean(X_train, y_train, X_test, kernel_fn, noise_var=0.01)

    if mean is None:
        print("  TODO 2 未完成，请补全 gp_predict_mean 函数")
    else:
        print(f"  预测均值形状: {mean.shape} (期望: (50,))")
        print(f"  预测范围: [{mean.min():.3f}, {mean.max():.3f}]")
        # 均值应该在训练数据 y 的范围附近
        assert mean.shape == (50,), "输出形状应为 (M,)"


# ============================================================================
# TODO 3: 实现 GP 回归的后验预测方差
# ============================================================================
def gp_predict_variance(X_train, X_test, kernel_fn, L, noise_var=1e-4):
    """
    计算 GP 回归的后验预测方差。

    公式:
        var(f*) = k(x*, x*) - k_*^T (K + σ_n²I)^{-1} k_*

    参数:
        X_train: (N, d)
        X_test: (M, d)
        kernel_fn: 核函数
        L: Cholesky 分解因子 L (N, N)，满足 K + σ_n²I = LL^T
        noise_var: 噪声方差

    返回:
        var: (M,) 后验预测方差
    """
    M = len(X_test)

    # TODO: 计算测试核矩阵
    K_test = None  # ← TODO: kernel_fn(X_test, X_train)  # (M, N)

    var = np.zeros(M)
    for i in range(M):
        # k_* 向量
        k_star = K_test[i, :]  # (N,)

        # TODO: 求解 v = L^{-1} k_star
        v = None  # ← TODO: np.linalg.solve(L, k_star)

        # TODO: 计算 var = k(x*, x*) - v^T v
        k_self = None  # ← TODO: kernel_fn(X_test[i:i+1], X_test[i:i+1]).item()
        var[i] = None  # ← TODO: k_self - v @ v
        var[i] = max(var[i], 1e-15)  # 防止负方差

    return var


def test_gp_variance():
    """测试 GP 预测方差"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: GP 预测方差")
    print("=" * 60)

    def rbf(X1, X2, lengthscale=1.0, variance=1.0):
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dist = np.maximum(sq1 + sq2 - 2 * X1 @ X2.T, 0)
        return variance * np.exp(-0.5 * sq_dist / lengthscale**2)

    X_train = np.array([[-3.], [-1.], [1.], [3.]])
    K = rbf(X_train, X_train) + 0.01 * np.eye(4)
    L = np.linalg.cholesky(K)
    X_test = np.linspace(-4, 4, 50).reshape(-1, 1)

    kernel_fn = lambda X1, X2: rbf(X1, X2)
    var = gp_predict_variance(X_train, X_test, kernel_fn, L)

    if var is None:
        print("  TODO 3 未完成，请补全 gp_predict_variance 函数")
    else:
        print(f"  方差形状: {var.shape} (期望: (50,))")
        print(f"  方差范围: [{var.min():.4f}, {var.max():.4f}]")
        print(f"  所有方差 > 0: {np.all(var > 0)} (期望: True)")
        # 在远离训练数据的地方方差应更大
        # x=0 处方差应较大（两个极端之间的中间点没有数据）
        mid_idx = len(X_test) // 2
        edge_var = var[0]
        mid_var = var[mid_idx]
        print(f"  边缘方差: {edge_var:.4f}, 中间方差: {mid_var:.4f}")
        print(f"  (中间方差应 ≥ 边缘方差，因为中间缺乏数据)")


# ============================================================================
# TODO 4: 从 GP 先验中采样函数
# ============================================================================
def sample_gp_prior(X, kernel_fn, n_samples=3, random_state=42):
    """
    从 GP 先验中采样随机函数。

    方法: f = Lz, 其中 LL^T = K (Cholesky), z ~ N(0, I)

    参数:
        X: (N, d) 输入点
        kernel_fn: 核函数
        n_samples: 采样数量
        random_state: 随机种子

    返回:
        samples: (n_samples, N) 采样的函数值
    """
    rng = np.random.RandomState(random_state)
    N = len(X)

    # TODO: 步骤 1 — 计算核矩阵 K
    K = None  # ← TODO: kernel_fn(X, X)  # (N, N)

    # TODO: 步骤 2 — Cholesky 分解 K = LL^T
    L = None  # ← TODO: np.linalg.cholesky(K + 1e-8 * np.eye(N))

    # TODO: 步骤 3 — 采样 f = Lz
    samples = np.zeros((n_samples, N))
    for i in range(n_samples):
        z = rng.randn(N)  # (N,)
        samples[i] = None  # ← TODO: L @ z

    return samples


def test_gp_sample():
    """测试 GP 采样"""
    print("\n" + "=" * 60)
    print("TODO 4 测试: GP 先验采样")
    print("=" * 60)

    def rbf(X1, X2, lengthscale=1.0, variance=1.0):
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        sq_dist = np.maximum(sq1 + sq2 - 2 * X1 @ X2.T, 0)
        return variance * np.exp(-0.5 * sq_dist / lengthscale**2)

    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    kernel_fn = lambda X1, X2: rbf(X1, X2)
    samples = sample_gp_prior(X, kernel_fn, n_samples=3)

    if samples is None:
        print("  TODO 4 未完成，请补全 sample_gp_prior 函数")
    else:
        print(f"  采样形状: {samples.shape} (期望: (3, 100))")
        print(f"  第1条样本均值: {samples[0].mean():.4f} (期望: ~0)")
        print(f"  第1条样本标准差: {samples[0].std():.4f} (期望: ~1)")
        # GP 先验均值为 0，方差为 k(x,x)=variance=1


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml14 核方法与高斯过程 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_rbf_kernel()
    test_gp_predict()
    test_gp_variance()
    test_gp_sample()

    print("\n全部测试完成！")
