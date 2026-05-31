# -*- coding: utf-8 -*-
"""
ml08 聚类 — 演示代码
====================
功能：从零实现 K-Means 和 DBSCAN，在合成数据集（blobs, moons, circles）上对比三种聚类算法，
      并进行轮廓系数分析确定最佳 K 值。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 ml08_clustering/ 目录下执行 python code/demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from scipy.spatial.distance import cdist

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：K-Means 从零实现
# ============================================================================

class KMeans:
    """
    K-Means 聚类算法（Lloyd 算法 + K-Means++ 初始化）。

    数学目标：最小化簇内平方误差
        J = sum_{k=1}^{K} sum_{x_i in C_k} ||x_i - mu_k||^2

    参数:
        n_clusters: 簇的数量 K
        init: 初始化方式，"random" 或 "kmeans++"
        max_iter: 最大迭代次数
        tol: 收敛容忍度（质心移动距离小于 tol 时提前停止）
        random_state: 随机种子
    """

    def __init__(self, n_clusters=3, init='kmeans++', max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None  # 质心，shape (K, d)
        self.labels_ = None     # 每个样本的簇标签
        self.inertia_ = None    # 最终的目标函数值 J
        self.n_iter_ = 0        # 实际迭代次数

    def _init_centroids(self, X):
        """K-Means++ 或随机初始化质心"""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if self.init == 'random':
            # 随机选择 K 个数据点作为初始质心
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()

        # K-Means++ 初始化
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        # 步骤 1：均匀随机选第一个质心
        centroids[0] = X[rng.randint(n_samples)]

        for k in range(1, self.n_clusters):
            # 计算每个点到最近已选质心的距离平方 D(x)^2
            dists = cdist(X, centroids[:k]) ** 2  # (n_samples, k)
            min_dists = dists.min(axis=1)           # 每行的最小值
            # 步骤 2-3：以 D(x)^2 为权重进行概率采样
            probs = min_dists / min_dists.sum()     # 归一化为概率
            centroids[k] = X[rng.choice(n_samples, p=probs)]
        return centroids

    def fit(self, X):
        """执行 K-Means 聚类"""
        self.centroids_ = self._init_centroids(X)

        for iteration in range(self.max_iter):
            # 分配步：计算每个点到所有质心的距离，分配到最近的质心
            distances = cdist(X, self.centroids_)           # (n, K)
            labels = np.argmin(distances, axis=1)            # (n,)

            # 更新步：计算每个簇的新质心 = 簇内点的均值
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)  # mu_k = mean(x_i in C_k)
                else:
                    # 空簇处理：随机选一个点作为新质心
                    new_centroids[k] = X[np.random.randint(X.shape[0])]

            # 检查收敛：质心移动距离 < tol？
            centroid_shift = np.linalg.norm(new_centroids - self.centroids_, axis=1).max()
            self.centroids_ = new_centroids

            if centroid_shift < self.tol:
                break

        self.labels_ = labels
        self.n_iter_ = iteration + 1
        # 计算 inertia（簇内平方误差）
        self.inertia_ = sum(
            ((X[self.labels_ == k] - self.centroids_[k]) ** 2).sum()
            for k in range(self.n_clusters)
        )
        return self


# ============================================================================
# 第二部分：DBSCAN 从零实现
# ============================================================================

class DBSCAN:
    """
    DBSCAN 密度聚类算法。

    核心思想：簇是密度相连的点的最大集合。

    参数:
        eps: 邻域半径 ε
        min_samples: 最小邻居数 minPts
        metric: 距离度量，默认欧氏距离
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None       # 标签：-1 表示噪声点
        self.core_sample_indices_ = None  # 核心点的索引

    def _region_query(self, X, point_idx):
        """查找点 point_idx 的 ε-邻域内所有点的索引"""
        # 计算所有点到 point_idx 的距离
        dists = np.linalg.norm(X - X[point_idx], axis=1)  # (n,)
        return np.where(dists <= self.eps)[0]

    def fit(self, X):
        """执行 DBSCAN 聚类"""
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # 初始全为噪声
        cluster_id = 0  # 当前簇编号
        visited = np.zeros(n_samples, dtype=bool)  # 是否已访问
        core_mask = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            # 查询 ε-邻域
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                # 暂时标记为噪声（后续可能被其他核心点"收编"为边界点）
                continue

            # 核心点：扩展新簇
            core_mask[i] = True
            cluster_id += 1
            self.labels_[i] = cluster_id - 1

            # 种子集：用于区域扩展（类似 BFS）
            seeds = list(neighbors)
            seeds.remove(i)

            # 扩张簇：遍历种子集中的每个点
            j = 0
            while j < len(seeds):
                current = seeds[j]
                j += 1

                if visited[current]:
                    # 已被访问过：如果是噪声，重新标记为边界点
                    if self.labels_[current] == -1:
                        self.labels_[current] = cluster_id - 1
                    continue

                visited[current] = True
                self.labels_[current] = cluster_id - 1

                # 查询当前点的邻域
                current_neighbors = self._region_query(X, current)
                if len(current_neighbors) >= self.min_samples:
                    # 当前点也是核心点，将其邻域新点加入种子集
                    core_mask[current] = True
                    for nb in current_neighbors:
                        if nb not in seeds:
                            seeds.append(nb)

        self.core_sample_indices_ = np.where(core_mask)[0]
        return self


# ============================================================================
# 第三部分：可视化
# ============================================================================

def plot_clustering_comparison(X_list, titles, figsize=(18, 5)):
    """
    对比 K-Means、DBSCAN 在三种合成数据集上的聚类结果。

    参数:
        X_list: [(X1, y1), (X2, y2), (X3, y3)] 数据集列表
        titles: 每个数据集的名称列表
    """
    algorithms = {
        'K-Means (K=2)': lambda X: KMeans(n_clusters=2, random_state=42).fit(X).labels_,
        'DBSCAN (eps=0.3)': lambda X: DBSCAN(eps=0.3, min_samples=5).fit(X).labels_,
    }

    n_datasets = len(X_list)
    n_algos = len(algorithms)

    fig, axes = plt.subplots(n_algos, n_datasets, figsize=figsize)

    for col, (X, y_true, title) in enumerate(X_list):
        for row, (algo_name, algo_fn) in enumerate(algorithms.items()):
            ax = axes[row, col]
            labels = algo_fn(X)
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                                 s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
            ax.set_title(f'{algo_name}\n{title}', fontsize=11)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml08-05-clustering-comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_silhouette_analysis(X, K_range=range(2, 7)):
    """
    轮廓系数分析：画出不同 K 值下的轮廓图，帮助选择最佳 K。

    参数:
        X: 输入数据，shape (n, d)
        K_range: 要测试的 K 值范围
    """
    n_plots = len(K_range)
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8))

    for idx, k in enumerate(K_range):
        # 运行 K-Means
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = km.labels_
        silhouette_avg = silhouette_score(X, labels)

        # 子图 1：轮廓图
        ax1 = axes[0, idx]
        sample_sil_values = silhouette_samples(X, labels)
        y_lower = 10
        for i in range(k):
            ith_cluster_sil_values = sample_sil_values[labels == i]
            ith_cluster_sil_values.sort()
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = plt.cm.viridis(i / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=1.5)
        ax1.set_title(f'K={k}\nAvg={silhouette_avg:.3f}')
        ax1.set_xlabel('Silhouette Coefficient')
        ax1.set_ylabel('Cluster')
        ax1.set_xlim([-0.1, 1.0])

        # 子图 2：聚类结果散点图
        ax2 = axes[1, idx]
        colors = plt.cm.viridis(labels / k)
        ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                    s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
        ax2.scatter(km.centroids_[:, 0], km.centroids_[:, 1],
                    marker='X', c='red', s=100, edgecolors='black', linewidth=1)
        ax2.set_title(f'K={k} Clusters')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml08-06-silhouette-analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_dbscan_parameter_effect(X, param_list, figsize=(18, 10)):
    """展示不同 (eps, min_samples) 参数对 DBSCAN 结果的影响"""
    n_params = len(param_list)
    cols = 3
    rows = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (eps, min_samples) in enumerate(param_list):
        ax = axes[idx]
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        colors = plt.cm.tab10(labels % 10) if n_clusters > 0 else np.full((len(labels), 4), [0.5, 0.5, 0.5, 1.0])
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10',
                   s=20, alpha=0.8, edgecolors='k', linewidth=0.3)
        ax.set_title(f'eps={eps}, minPts={min_samples}\n'
                     f'Clusters: {n_clusters}, Noise: {n_noise}', fontsize=10)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余的子图
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml08-07-dbscan-parameters.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


def plot_inertia_vs_k(X, K_range=range(1, 11)):
    """肘部法则：画出 inertia（WCSS）随 K 变化曲线"""
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_range), inertias, 'bo-', markersize=8, linewidth=2)
    ax.set_xlabel('Number of Clusters K')
    ax.set_ylabel('Inertia (WCSS)')
    ax.set_title('Elbow Method for Optimal K')
    # 标出"肘部"位置
    ax.axvline(x=4, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate('Possible Elbow (K=4)', xy=(4, inertias[3]),
                xytext=(5.5, inertias[3] * 1.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'ml08-08-elbow-method.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[保存] {path}')


# ============================================================================
# 第四部分：主程序
# ============================================================================

def main():
    print('=' * 60)
    print('ml08 聚类 — 演示代码')
    print('=' * 60)

    # 1. 生成三个合成数据集
    print('\n[1/6] 生成合成数据集...')
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, random_state=42,
                                   cluster_std=0.8)
    X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5,
                                         random_state=42)

    datasets = [
        (X_blobs, y_blobs, 'Blobs\n(spherical)'),
        (X_moons, y_moons, 'Moons\n(non-convex)'),
        (X_circles, y_circles, 'Circles\n(nested)'),
    ]

    # 2. 对比 K-Means vs DBSCAN 在不同数据集上的效果
    print('[2/6] 对比 K-Means vs DBSCAN...')
    plot_clustering_comparison(datasets, [d[2] for d in datasets])

    # 3. 轮廓系数分析（在 blobs 数据上演示）
    print('[3/6] 轮廓系数分析...')
    plot_silhouette_analysis(X_blobs, K_range=range(2, 7))

    # 4. DBSCAN 参数影响
    print('[4/6] DBSCAN 参数选择分析...')
    param_list = [
        (0.15, 3), (0.15, 5), (0.15, 10),
        (0.30, 3), (0.30, 5), (0.30, 10),
    ]
    plot_dbscan_parameter_effect(X_circles, param_list)

    # 5. 肘部法则
    print('[5/6] 肘部法则...')
    plot_inertia_vs_k(X_blobs)

    # 6. 打印核心指标
    print('\n[6/6] 聚类评估指标汇总：')
    print('-' * 50)
    X = X_blobs  # 以 blobs 数据为例
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        labels = km.labels_
        sil = silhouette_score(X, labels)
        print(f'  K={k}: 轮廓系数={sil:.4f}, Inertia={km.inertia_:.2f}, '
              f'迭代次数={km.n_iter_}')

        if k == 4:
            print(f'  >>> 数据真实有 4 个簇，轮廓系数分析应在 K=4 附近表现最好')

    print('\n完成！所有图表已保存到 images/ 目录。')


if __name__ == '__main__':
    main()
