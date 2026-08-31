# -*- coding: utf-8 -*-
"""
ml12 EM算法与高斯混合模型 — 演示代码
=====================================
功能：从零实现 GMM 的 EM 算法，对比 K-Means vs GMM 在非球形簇上的效果，
      展示 AIC/BIC 模型选择，可视化软分配（responsibilities）。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml12_em_gmm/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：GMM 从零实现（EM 算法）
# ============================================================================

class GMM:
    """
    高斯混合模型（Gaussian Mixture Model）—— 从零实现 EM 算法。

    参数:
        n_components: 高斯成分数 K
        covariance_type: 'full', 'diag', 'spherical' 之一
        max_iter: EM 最大迭代次数
        tol: 对数似然变化容忍度（收敛判断）
        reg_covar: 协方差正则化（防止奇异矩阵）
        random_state: 随机种子
    """

    def __init__(self, n_components=3, covariance_type='full', max_iter=100,
                 tol=1e-3, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        # 待学习的参数
        self.weights_ = None       # 混合系数 π_k (K,)
        self.means_ = None         # 均值 μ_k (K, d)
        self.covariances_ = None   # 协方差 Σ_k
        self.converged_ = False
        self.n_iter_ = 0

    def _initialize(self, X):
        """用 K-Means++ 思想初始化参数"""
        rng = np.random.RandomState(self.random_state)
        n, d = X.shape

        # 选择 K 个数据点作为初始均值
        indices = rng.choice(n, self.n_components, replace=False)
        self.means_ = X[indices].copy()  # (K, d)

        # 协方差初始化为数据协方差
        data_cov = np.cov(X.T)  # (d, d)
        if self.covariance_type == 'full':
            self.covariances_ = np.array([data_cov.copy() for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            diag = np.diag(data_cov)
            self.covariances_ = np.array([np.diag(diag) for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            # 球面：方差为所有维度的平均方差
            avg_var = np.trace(data_cov) / d
            self.covariances_ = np.array([np.eye(d) * avg_var for _ in range(self.n_components)])

        # 混合系数初始化为均匀分布
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _log_multivariate_normal(self, X, mean, cov):
        """计算多元高斯对数 PDF"""
        # 使用 scipy 的 multivariate_normal
        return multivariate_normal.logpdf(X, mean=mean, cov=cov)

    def _e_step(self, X):
        """
        E 步：计算责任（responsibilities）γ_{ik}。

        γ_{ik} = P(z_i = k | x_i) ∝ π_k · N(x_i | μ_k, Σ_k)

        返回:
            log_resp: 对数责任矩阵 (n, K)
            resp: 责任矩阵 (n, K)，每行和为 1
            log_likelihood: 对数似然值
        """
        n = X.shape[0]
        K = self.n_components

        # 计算每个样本在每个成分下的对数概率密度
        weighted_log_prob = np.zeros((n, K))
        for k in range(K):
            # log(π_k) + log N(x_i | μ_k, Σ_k)
            log_pi_k = np.log(self.weights_[k] + 1e-300)
            log_pdf_k = self._log_multivariate_normal(X, self.means_[k],
                                                       self.covariances_[k])
            weighted_log_prob[:, k] = log_pi_k + log_pdf_k

        # log-sum-exp trick 计算对数似然和归一化
        # log P(x_i) = log Σ_k exp(log(π_k) + log N(x_i|...))
        log_likelihood_per_sample = self._logsumexp(weighted_log_prob, axis=1)  # (n,)
        log_likelihood = log_likelihood_per_sample.sum()

        # 对数责任: log γ_{ik} = log(π_k) + log N(x_i|μ_k,Σ_k) - log P(x_i)
        log_resp = weighted_log_prob - log_likelihood_per_sample[:, np.newaxis]
        # 转回概率空间
        resp = np.exp(log_resp)
        # 确保每行和为 1（数值稳定性）
        resp = resp / resp.sum(axis=1, keepdims=True)

        return log_resp, resp, log_likelihood

    def _logsumexp(self, x, axis=1):
        """数值稳定的 log-sum-exp 计算"""
        x_max = np.max(x, axis=axis, keepdims=True)
        return np.squeeze(x_max + np.log(np.sum(np.exp(x - x_max), axis=axis)))

    def _m_step(self, X, resp):
        """
        M 步：最大化 Q 函数，更新参数。

        π_k = (1/N) Σ_i γ_{ik}                     — 有效样本数
        μ_k = Σ_i γ_{ik} x_i / Σ_i γ_{ik}          — 加权均值
        Σ_k = Σ_i γ_{ik} (x_i-μ_k)(x_i-μ_k)^T / Σ_i γ_{ik}  — 加权协方差
        """
        n, d = X.shape
        K = self.n_components

        Nk = resp.sum(axis=0) + 1e-10  # 每个成分的有效样本数 (K,)
        # 防止除零

        # 更新权重（混合系数）
        self.weights_ = Nk / n  # (K,)

        # 更新均值
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]  # (K, d)

        # 更新协方差
        for k in range(K):
            diff = X - self.means_[k]  # (n, d)
            # Σ_k = Σ_i γ_{ik} (x_i-μ_k)(x_i-μ_k)^T / Nk
            weighted_diff = diff * resp[:, k, np.newaxis]  # (n, d)
            cov_k = (weighted_diff.T @ diff) / Nk[k]  # (d, d)
            cov_k += np.eye(d) * self.reg_covar  # 正则化防奇异

            if self.covariance_type == 'full':
                self.covariances_[k] = cov_k
            elif self.covariance_type == 'diag':
                self.covariances_[k] = np.diag(np.diag(cov_k))
            elif self.covariance_type == 'spherical':
                avg_var = np.trace(cov_k) / d
                self.covariances_[k] = np.eye(d) * avg_var

    def fit(self, X):
        """执行 EM 算法"""
        self._initialize(X)
        prev_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E 步
            log_resp, resp, log_likelihood = self._e_step(X)

            # 检查收敛
            change = log_likelihood - prev_log_likelihood
            if iteration > 0 and abs(change) < self.tol:
                self.converged_ = True
                break

            prev_log_likelihood = log_likelihood

            # M 步
            self._m_step(X, resp)

        self.n_iter_ = iteration + 1
        self.responsibilities_ = resp  # 保存最终的责任值
        return self

    def predict_proba(self, X):
        """预测每个样本属于每个成分的概率（软分配）"""
        _, resp, _ = self._e_step(X)
        return resp

    def predict(self, X):
        """硬分配：每个样本分配到概率最高的成分"""
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        """返回对数似然值（用于 AIC/BIC 计算）"""
        _, _, log_likelihood = self._e_step(X)
        return log_likelihood


# ============================================================================
# 第二部分：可视化
# ============================================================================

def plot_gmm_vs_kmeans(X, n_components=3):
    """对比 K-Means 和 GMM 在非球形数据上的聚类效果"""
    from sklearn.cluster import KMeans

    # 生成各向异性（anisotropic）数据
    np.random.seed(42)
    # 旋转矩阵
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # 生成三个拉伸+旋转的高斯团
    covs = [
        R @ np.diag([0.3, 2.0]) @ R.T,
        R @ np.diag([1.5, 0.3]) @ R.T,
        np.diag([0.5, 0.5])
    ]
    X = np.vstack([
        np.random.multivariate_normal([-2, 0], covs[0], 100),
        np.random.multivariate_normal([2, 0], covs[1], 100),
        np.random.multivariate_normal([0, 2], covs[2], 100),
    ])

    # K-Means
    km = KMeans(n_clusters=n_components, random_state=42).fit(X)

    # GMM (full covariance)
    gmm = GMM(n_components=n_components, covariance_type='full', random_state=42).fit(X)
    gmm_labels = gmm.predict(X)
    resp = gmm.responsibilities_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：K-Means
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='tab10', s=20, alpha=0.8,
               edgecolors='k', linewidth=0.3)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               marker='X', c='red', s=150, edgecolors='black', linewidth=1.5)
    ax.set_title('K-Means\n(Hard Assignment)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # 右图：GMM（显示软分配）
    ax2 = axes[1]
    # 混合颜色：每个点颜色 = 各成分颜色的加权混合（按 responsibility 加权）
    rgba = resp @ plt.cm.tab10(np.arange(n_components) / n_components)[:, :3]
    ax2.scatter(X[:, 0], X[:, 1], c=rgba, s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
    # 画等高线
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                          np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for k in range(n_components):
        zz = multivariate_normal.pdf(grid, mean=gmm.means_[k], cov=gmm.covariances_[k])
        zz = zz.reshape(100, 100)
        ax2.contour(xx, yy, zz, levels=4, colors=[plt.cm.tab10(k / n_components)],
                     alpha=0.5, linewidths=1)
    ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', c='red', s=150,
                edgecolors='black', linewidth=1.5)
    ax2.set_title('GMM (Full Covariance)\n(Soft Assignment + Elliptical Contours)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml12-04-gmm-vs-kmeans.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_aic_bic(X, K_range=range(1, 8)):
    """使用 AIC 和 BIC 选择 GMM 的最佳 K"""
    aic_scores = []
    bic_scores = []
    n = X.shape[0]
    d = X.shape[1]

    for k in K_range:
        gmm = GMM(n_components=k, covariance_type='full', max_iter=100,
                   random_state=42).fit(X)
        log_l = gmm.score(X)

        # 计算参数个数（full covariance）
        # weights: K-1, means: K*d, cov: K*d*(d+1)/2
        n_params = (k - 1) + k * d + k * d * (d + 1) / 2

        aic = -2 * log_l + 2 * n_params
        bic = -2 * log_l + n_params * np.log(n)

        aic_scores.append(aic)
        bic_scores.append(bic)
        print(f'  K={k}: logL={log_l:.1f}, AIC={aic:.1f}, BIC={bic:.1f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores, name, color in [
        (axes[0], aic_scores, 'AIC', 'steelblue'),
        (axes[1], bic_scores, 'BIC', 'darkorange')
    ]:
        ax.plot(list(K_range), scores, f'{color[0]}-', marker='o',
                markersize=8, linewidth=2, color=color)
        best_k = K_range[np.argmin(scores)]
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate(f'Best K={best_k}', xy=(best_k, min(scores)),
                    xytext=(best_k + 0.5, min(scores) * 1.01),
                    fontsize=11, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))
        ax.set_xlabel('Number of Components K')
        ax.set_ylabel(f'{name} Score')
        ax.set_title(f'{name} vs K (lower is better)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml12-05-aic-bic.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_em_convergence(X, n_components=3):
    """展示 EM 算法的收敛过程——对数似然单调递增"""
    gmm = GMM(n_components=n_components, covariance_type='full', random_state=42)

    # 手动运行 EM，记录每步的对数似然
    gmm._initialize(X)
    log_likelihoods = []

    for iteration in range(30):
        _, _, log_l = gmm._e_step(X)
        log_likelihoods.append(log_l)
        gmm._m_step(X, gmm.responsibilities_ if hasattr(gmm, 'responsibilities_')
                     else np.ones((X.shape[0], n_components)) / n_components)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(log_likelihoods)), log_likelihoods, 'b-o', markersize=5, linewidth=2)
    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('EM Algorithm: Monotonically Increasing Log-Likelihood')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml12-06-em-convergence.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')
    print(f'  最终对数似然: {log_likelihoods[-1]:.2f} (从 {log_likelihoods[0]:.2f} 开始)')


# ============================================================================
# 第三部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml12 EM算法与高斯混合模型 — 演示代码')
    print('=' * 60)

    # 1. 生成数据
    print('\n[1/4] 生成非球形（anisotropic）合成数据...')
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)

    # 2. GMM vs K-Means
    print('[2/4] 对比 K-Means vs GMM...')
    plot_gmm_vs_kmeans(X, n_components=3)

    # 3. AIC / BIC 模型选择
    print('[3/4] AIC / BIC 模型选择...')
    plot_aic_bic(X, K_range=range(1, 8))

    # 4. EM 收敛曲线
    print('[4/4] EM 收敛曲线...')
    plot_em_convergence(X, n_components=4)

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
