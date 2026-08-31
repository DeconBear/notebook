# -*- coding: utf-8 -*-
"""
ml14 核方法与高斯过程 — 演示代码
===============================
功能：从零实现核岭回归和高斯过程回归（RBF 核），
      展示 GP 先验/后验可视化、不同核函数的对比。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml14_kernel_gp/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：核函数库
# ============================================================================

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    RBF (Radial Basis Function / Squared Exponential) 核。

    k(x, x') = σ² · exp(-||x - x'||² / (2ℓ²))

    参数:
        X1: (n1, d) 或 (n1,)
        X2: (n2, d) 或 (n2,)
        lengthscale: ℓ，控制平滑度
        variance: σ²，信号方差

    返回:
        K: 核矩阵 (n1, n2)，K[i,j] = k(X1[i], X2[j])
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    # 计算平方欧氏距离矩阵: dist²[i,j] = ||X1[i] - X2[j]||²
    # 展开: ||x-y||² = ||x||² + ||y||² - 2<x,y>
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)  # (n1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)  # (1, n2)
    sq_dist = sq_norm1 + sq_norm2 - 2 * X1 @ X2.T   # (n1, n2)
    sq_dist = np.maximum(sq_dist, 0)  # 数值稳定性

    return variance * np.exp(-0.5 * sq_dist / (lengthscale**2))


def matern32_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Matern ν=3/2 核。

    k(r) = σ² · (1 + √3 r/ℓ) · exp(-√3 r/ℓ)
    其中 r = ||x - x'||
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
    sq_dist = sq_norm1 + sq_norm2 - 2 * X1 @ X2.T
    dist = np.sqrt(np.maximum(sq_dist, 0))
    sqrt3 = np.sqrt(3)
    scaled = sqrt3 * dist / lengthscale
    return variance * (1 + scaled) * np.exp(-scaled)


def periodic_kernel(X1, X2, period=1.0, lengthscale=1.0, variance=1.0):
    """
    周期核。

    k(x, x') = σ² · exp(-2 sin²(π|x-x'|/p) / ℓ²)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n1, n2, d)
    sin_sq = np.sin(np.pi * diff / period) ** 2
    sin_sq_sum = sin_sq.sum(axis=2)  # (n1, n2)
    return variance * np.exp(-2 * sin_sq_sum / (lengthscale**2))


# ============================================================================
# 第二部分：核岭回归（Kernel Ridge Regression）
# ============================================================================

class KernelRidgeRegression:
    """
    核岭回归：f(x*) = k_*^T (K + λI)^{-1} y

    参数:
        kernel: 核函数 callable(X1, X2, **kernel_params)
        alpha: 正则化参数 λ
        kernel_params: 核函数的额外参数（如 lengthscale）
    """

    def __init__(self, kernel=rbf_kernel, alpha=1e-4, **kernel_params):
        self.kernel = kernel
        self.alpha = alpha
        self.kernel_params = kernel_params
        self.X_train_ = None
        self.alpha_coef_ = None  # α = (K + λI)^{-1} y

    def fit(self, X, y):
        """
        训练：计算 α = (K + λI)^{-1} y
        """
        self.X_train_ = X.copy()
        K = self.kernel(X, X, **self.kernel_params)  # (N, N)
        K_reg = K + self.alpha * np.eye(len(X))       # K + λI

        # Cholesky 分解求解 (K + λI) α = y
        # Cholesky 比直接求逆更稳定，复杂度和求逆相同
        L = np.linalg.cholesky(K_reg)                  # K_reg = LL^T
        self.alpha_coef_ = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return self

    def predict(self, X):
        """
        预测：f(x*) = k_*^T α
        """
        K_test = self.kernel(X, self.X_train_, **self.kernel_params)  # (n_test, N)
        return K_test @ self.alpha_coef_


# ============================================================================
# 第三部分：高斯过程回归（GP Regression）
# ============================================================================

class GaussianProcessRegressor:
    """
    高斯过程回归。

    模型：y = f(x) + ε, f ~ GP(0, k), ε ~ N(0, σ_n^2)

    预测分布：
        f* | X, y, x* ~ N(k_*^T (K + σ_n²I)^{-1} y,
                          k(x*,x*) - k_*^T (K + σ_n²I)^{-1} k_*)

    参数:
        kernel: 核函数
        noise_var: 观测噪声方差 σ_n²
        kernel_params: 核参数
    """

    def __init__(self, kernel=rbf_kernel, noise_var=1e-4, **kernel_params):
        self.kernel = kernel
        self.noise_var = noise_var
        self.kernel_params = kernel_params
        self.X_train_ = None
        self.y_train_ = None
        self.K_inv_y_ = None  # (K + σ_n²I)^{-1} y
        self.L_ = None        # Cholesky 因子

    def fit(self, X, y):
        """训练：计算 Cholesky 分解和预测所需向量"""
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        N = len(X)

        K = self.kernel(X, X, **self.kernel_params)  # (N, N)
        K_noisy = K + self.noise_var * np.eye(N)     # K + σ_n²I

        # Cholesky: K_noisy = LL^T
        self.L_ = np.linalg.cholesky(K_noisy)

        # 求解 L L^T α = y → α = (K + σ_n²I)^{-1} y
        self.K_inv_y_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, y))
        return self

    def predict(self, X_test):
        """
        预测均值和方差。

        均值: f* = k_*^T (K + σ_n²I)^{-1} y
        方差: var = k(x*,x*) - k_*^T (K + σ_n²I)^{-1} k_*
        """
        K_test = self.kernel(X_test, self.X_train_, **self.kernel_params)  # (n_test, N)
        mean = K_test @ self.K_inv_y_

        # 方差：对每个测试点计算
        # var[i] = k(x_i*, x_i*) - k_i*^T L^{-T} L^{-1} k_i*
        n_test = len(X_test)
        var = np.zeros(n_test)
        for i in range(n_test):
            k_star = K_test[i, :]  # (N,)
            # 求解 v = L^{-1} k_star
            v = np.linalg.solve(self.L_, k_star)  # (N,)
            # var = k(x*,x*) - v^T v
            k_self = self.kernel(
                X_test[i:i+1], X_test[i:i+1], **self.kernel_params
            ).item()
            var[i] = k_self - v @ v
            var[i] = max(var[i], 1e-15)  # 防止数值误差导致负方差

        return mean, var


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_gp_prior(lengthscale=1.0, variance=1.0, n_samples=5):
    """从 GP 先验中采样并可视化"""
    X = np.linspace(-5, 5, 200).reshape(-1, 1)
    K = rbf_kernel(X, X, lengthscale=lengthscale, variance=variance)

    # Cholesky 分解用于采样: f = L z, z ~ N(0, I), LL^T = K
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))

    fig, ax = plt.subplots(figsize=(10, 5))

    # 先验均值（0）和 ±2σ 置信带
    std = np.sqrt(np.diag(K))
    ax.fill_between(X.ravel(), -2 * std, 2 * std, alpha=0.2, color='gray',
                    label='±2σ prior')

    # 采样函数
    np.random.seed(42)
    for i in range(n_samples):
        f_sample = L @ np.random.randn(len(X))
        ax.plot(X.ravel(), f_sample, linewidth=1.2, alpha=0.8,
                label=f'Sample {i+1}' if i == 0 else '')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'GP Prior: RBF Kernel (ℓ={lengthscale}, σ²={variance})')
    ax.legend(loc='upper right')
    ax.set_ylim(-4, 4)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml14-04-gp-prior-samples.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_kernel_ridge_vs_gp():
    """对比核岭回归和 GP 回归，展示 GP 的不确定性优势"""
    np.random.seed(42)
    X_train = np.sort(np.random.uniform(-5, 5, 15)).reshape(-1, 1)
    y_train = np.sin(X_train.ravel()) + 0.2 * np.random.randn(15)
    X_test = np.linspace(-6, 6, 300).reshape(-1, 1)

    # 核岭回归
    krr = KernelRidgeRegression(kernel=rbf_kernel, alpha=0.01, lengthscale=1.0, variance=1.0)
    krr.fit(X_train, y_train)
    y_krr = krr.predict(X_test)

    # GP 回归
    gp = GaussianProcessRegressor(kernel=rbf_kernel, noise_var=0.04,
                                   lengthscale=1.0, variance=1.0)
    gp.fit(X_train, y_train)
    y_gp_mean, y_gp_var = gp.predict(X_test)
    y_gp_std = np.sqrt(y_gp_var)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左图：核岭回归（只有点估计）
    ax = axes[0]
    ax.plot(X_test, y_krr, 'b-', linewidth=2, label='KRR Prediction')
    ax.scatter(X_train, y_train, c='red', s=40, zorder=5, edgecolors='black',
               linewidth=0.5, label='Training Data')
    ax.plot(X_test, np.sin(X_test.ravel()), 'gray', linestyle='--', linewidth=1.5,
            alpha=0.5, label='True f(x)=sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Kernel Ridge Regression (Point Estimate Only)')
    ax.legend(fontsize=9)
    ax.set_ylim(-2, 2)

    # 右图：GP 回归（均值 + 置信带）
    ax2 = axes[1]
    ax2.fill_between(X_test.ravel(), y_gp_mean - 2*y_gp_std, y_gp_mean + 2*y_gp_std,
                     alpha=0.3, color='blue', label='±2σ confidence')
    ax2.plot(X_test, y_gp_mean, 'b-', linewidth=2, label='GP Posterior Mean')
    ax2.scatter(X_train, y_train, c='red', s=40, zorder=5, edgecolors='black',
                linewidth=0.5, label='Training Data')
    ax2.plot(X_test, np.sin(X_test.ravel()), 'gray', linestyle='--', linewidth=1.5,
             alpha=0.5, label='True f(x)=sin(x)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Gaussian Process Regression\n(Prediction + Uncertainty)')
    ax2.legend(fontsize=9)
    ax2.set_ylim(-2, 2)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml14-05-krr-vs-gp.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_kernel_comparison():
    """可视化不同核函数的形状和 GP 先验采样"""
    kernels = [
        ('RBF', rbf_kernel, {'lengthscale': 1.0, 'variance': 1.0}),
        ('Matern ν=3/2', matern32_kernel, {'lengthscale': 1.0, 'variance': 1.0}),
        ('Periodic', periodic_kernel, {'period': 2.0, 'lengthscale': 0.5, 'variance': 1.0}),
    ]

    fig, axes = plt.subplots(len(kernels), 2, figsize=(14, 10))

    # 用于计算核形状的距离范围
    r = np.linspace(0, 5, 200).reshape(-1, 1)
    X_vis = np.linspace(-5, 5, 200).reshape(-1, 1)

    for row, (name, kernel_fn, params) in enumerate(kernels):
        # 左列：核函数形状 k(r) vs r
        ax_k = axes[row, 0]
        k_vals = kernel_fn(r, np.zeros((1, 1)), **params).ravel()
        ax_k.plot(r.ravel(), k_vals, 'b-', linewidth=2)
        ax_k.set_xlabel('Distance r = |x - x\'|')
        ax_k.set_ylabel('k(r)')
        ax_k.set_title(f'{name} Kernel Shape')
        ax_k.grid(True, alpha=0.3)
        ax_k.set_ylim(-0.1, 1.2)

        # 右列：从 GP 先验中采样 3 个随机函数
        ax_f = axes[row, 1]
        K = kernel_fn(X_vis, X_vis, **params)
        L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_vis)))
        np.random.seed(42 + row)
        for i in range(3):
            f = L @ np.random.randn(len(X_vis))
            ax_f.plot(X_vis.ravel(), f, linewidth=1, alpha=0.8)
        ax_f.set_xlabel('x')
        ax_f.set_ylabel('f(x)')
        ax_f.set_title(f'GP Prior Samples ({name})')

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml14-06-kernel-comparison-samples.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml14 核方法与高斯过程 — 演示代码')
    print('=' * 60)

    # 1. GP 先验可视化
    print('\n[1/3] GP 先验采样（RBF 核）...')
    plot_gp_prior(lengthscale=1.0, variance=1.0)

    # 2. 核岭回归 vs GP 回归
    print('[2/3] 核岭回归 vs GP 回归对比...')
    plot_kernel_ridge_vs_gp()

    # 3. 不同核函数对比
    print('[3/3] 核函数对比...')
    plot_kernel_comparison()

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
