# -*- coding: utf-8 -*-
"""
ml08 聚类 — 练习代码
====================
请完成以下 TODO 任务，巩固对 K-Means、DBSCAN 和聚类评估的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 index.md 和 code-demo.md，再尝试独立补全代码。
"""

import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist


# ============================================================================
# TODO 1: 实现 K-Means 的单次迭代
# ============================================================================
def kmeans_single_iteration(X, centroids):
    """
    执行 K-Means 的一轮迭代（分配 + 更新）。

    数学公式：
        分配步: label[i] = argmin_k ||X[i] - centroids[k]||^2
        更新步: centroids[k] = mean_{i: label[i]=k} X[i]

    参数:
        X: 数据矩阵，shape (n_samples, d)
        centroids: 当前质心，shape (K, d)

    返回:
        labels: 分配结果，shape (n_samples,)
        new_centroids: 更新后的质心，shape (K, d)
        inertia: 簇内平方误差 J = sum sum ||x - mu||^2
    """
    K = centroids.shape[0]

    # TODO: 步骤 1 —— 计算每个点到所有质心的欧氏距离平方
    # 提示：使用 cdist(X, centroids, metric='euclidean') 或手动计算
    distances = None  # ← TODO: shape (n_samples, K)

    # TODO: 步骤 2 —— 分配步：每个点归属到距离最近的质心
    labels = None  # ← TODO: np.argmin(distances, axis=1)

    # TODO: 步骤 3 —— 更新步：计算每个簇的新质心 = 簇内点的均值
    new_centroids = np.zeros_like(centroids)
    for k in range(K):
        mask = None  # ← TODO: (labels == k)
        if mask.sum() > 0:
            new_centroids[k] = None  # ← TODO: X[mask].mean(axis=0)
        else:
            new_centroids[k] = centroids[k]  # 空簇保持原质心

    # TODO: 步骤 4 —— 计算 inertia（簇内平方误差）
    inertia = None  # ← TODO: 对每个簇 k，计算 sum(||X[labels==k] - centroids[k]||^2) 然后求和

    return labels, new_centroids, inertia


# TODO 1 测试
def test_kmeans_iteration():
    """测试 K-Means 单次迭代"""
    print("=" * 60)
    print("TODO 1 测试: K-Means 单次迭代")
    print("=" * 60)
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    # 随机初始化 3 个质心
    centroids = X[np.random.choice(len(X), 3, replace=False)]
    labels, new_centroids, inertia = kmeans_single_iteration(X, centroids)

    if labels is None:
        print("  TODO 1 未完成，请补全 kmeans_single_iteration 函数")
    else:
        print(f"  标签数: {len(labels)} (期望: 100)")
        print(f"  标签类型: {np.unique(labels)} (期望: [0, 1, 2])")
        print(f"  新质心形状: {new_centroids.shape} (期望: (3, 2))")
        print(f"  Inertia: {inertia:.2f}")
        # 验证惯性非负
        assert inertia >= 0, "Inertia 必须非负"


# ============================================================================
# TODO 2: 实现轮廓系数计算
# ============================================================================
def silhouette_score_manual(X, labels):
    """
    手动计算平均轮廓系数。

    对每个样本 i:
        a(i) = 样本 i 到同簇其他样本的平均距离
        b(i) = 样本 i 到最近的其他簇所有样本的平均距离
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    参数:
        X: 数据矩阵，shape (n_samples, d)
        labels: 聚类标签，shape (n_samples,)

    返回:
        float: 平均轮廓系数
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0  # 边缘情况

    # 预计算所有点对距离矩阵（上三角部分）
    dist_matrix = cdist(X, X)  # (n, n)

    silhouette_vals = np.zeros(n_samples)

    for i in range(n_samples):
        label_i = labels[i]

        # TODO: 计算 a(i) —— 样本 i 到同簇其他样本的平均距离
        # 提示：用 (labels == label_i) 筛选同簇样本，排除 i 自身
        same_cluster_mask = None  # ← TODO
        a_i = None  # ← TODO: dist_matrix[i, same_cluster_mask].mean()

        # TODO: 计算 b(i) —— 样本 i 到最近的其他簇的平均距离
        # 提示：对每个不等于 label_i 的簇 label_k，计算 i 到该簇所有样本的平均距离，取最小值
        b_i = float('inf')
        for label_k in unique_labels:
            if label_k == label_i:
                continue
            other_cluster_mask = None  # ← TODO: (labels == label_k)
            avg_dist = None  # ← TODO: dist_matrix[i, other_cluster_mask].mean()
            if avg_dist < b_i:
                b_i = avg_dist

        # TODO: 计算 s(i)
        silhouette_vals[i] = None  # ← TODO: (b_i - a_i) / max(a_i, b_i)

    return silhouette_vals.mean()


# TODO 2 测试
def test_silhouette():
    """测试轮廓系数计算"""
    print("\n" + "=" * 60)
    print("TODO 2 测试: 轮廓系数计算")
    print("=" * 60)
    X, y = make_blobs(n_samples=100, centers=3, random_state=42)
    score = silhouette_score_manual(X, y)

    if score is None:
        print("  TODO 2 未完成，请补全 silhouette_score_manual 函数")
    else:
        print(f"  手动轮廓系数: {score:.4f} (应接近 0.6-0.8)")
        assert -1.0 <= score <= 1.0, "轮廓系数必须在 [-1, 1] 范围内"


# ============================================================================
# TODO 3: 实现 DBSCAN 的区域查询
# ============================================================================
def region_query(X, point_idx, eps):
    """
    查找点 point_idx 的 ε-邻域（欧氏距离 ≤ eps 的所有点）。

    参数:
        X: 数据矩阵，shape (n_samples, d)
        point_idx: 查询点的索引
        eps: 邻域半径

    返回:
        neighbors: 邻域内所有点的索引数组
    """
    # TODO: 计算所有点到 X[point_idx] 的欧氏距离
    # 提示: np.linalg.norm(X - X[point_idx], axis=1) 得到距离向量
    dists = None  # ← TODO: shape (n,)

    # TODO: 返回距离 ≤ eps 的点的索引
    neighbors = None  # ← TODO: np.where(dists <= eps)[0]
    return neighbors


# TODO 3 测试
def test_region_query():
    """测试区域查询"""
    print("\n" + "=" * 60)
    print("TODO 3 测试: DBSCAN 区域查询")
    print("=" * 60)
    X = np.array([[0., 0.], [0.3, 0.3], [0.5, 0.5], [2., 2.]])
    neighbors = region_query(X, 0, eps=0.5)

    if neighbors is None:
        print("  TODO 3 未完成，请补全 region_query 函数")
    else:
        print(f"  点0的eps邻域: {neighbors} (期望: [0, 1])")
        print(f"  邻域大小: {len(neighbors)} (期望: 2)")


# ============================================================================
# TODO 4: 选择最佳 K（综合肘部法则和轮廓系数）
# ============================================================================
def find_optimal_k(X, K_range=range(2, 9)):
    """
    综合肘部法则和轮廓系数，找到最佳 K 值。

    策略：
        1. 对每个 K 运行 K-Means，记录 inertia 和平均轮廓系数
        2. 找轮廓系数最高的 K
        3. 用肘部法则辅助验证
    提示：你可以 import 上面实现的 kmeans_single_iteration
         或者用 sklearn.cluster.KMeans

    返回:
        best_k_silhouette: 轮廓系数最高的 K
        results: dict {K: {'inertia': ..., 'silhouette': ...}}
    """
    # 这里可以使用 sklearn（简化迭代循环）
    from sklearn.cluster import KMeans as SKMeans
    from sklearn.metrics import silhouette_score as sk_sil

    results = {}
    print("\n" + "=" * 60)
    print("TODO 4 测试: 选择最佳 K")
    print("=" * 60)

    # 生成测试数据
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    for k in K_range:
        # TODO: 运行 K-Means 并记录结果
        km = None  # ← TODO: SKMeans(n_clusters=k, random_state=42).fit(X)
        inertia = None  # ← TODO: km.inertia_
        labels = None   # ← TODO: km.labels_
        sil = None      # ← TODO: sk_sil(X, labels)

        results[k] = {'inertia': inertia, 'silhouette': sil}
        print(f'  K={k}: inertia={inertia:.2f}, silhouette={sil:.4f}')

    # TODO: 找到轮廓系数最高的 K
    best_k = None  # ← 基于 silhouette 选择
    print(f'\n  最佳 K (轮廓系数): {best_k}')

    return best_k, results


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print("ml08 聚类 — 练习代码")
    print("请完成所有 TODO 标记的代码")
    print()

    test_kmeans_iteration()
    test_silhouette()
    test_region_query()
    find_optimal_k(X=np.random.randn(100, 2))  # 将在函数内生成测试数据

    print("\n全部测试完成！")
