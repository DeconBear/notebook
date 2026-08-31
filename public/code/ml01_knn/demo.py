# -*- coding: utf-8 -*-
"""
===============================================================================
ml01_knn/code/demo.py — k-近邻算法与距离度量
===============================================================================
本演示通过实现 k-NN 分类器和回归器，全面展示：
  1. k-NN 算法的核心原理（多数投票 vs 距离加权）
  2. 四种距离度量的作用和差异
  3. k 值对决策边界的影响（k=1, 5, 15）
  4. 维数灾难的直观展示
  5. k-d 树的简单模拟（非完整实现，展示思想）
  6. 与 sklearn 的 KNeighborsClassifier 对比验证

通过本演示，你将理解：
  - k-NN 的惰性学习本质（无显式训练过程）
  - 不同的距离度量如何影响"近邻"的定义
  - k 值的偏差-方差权衡
  - 高维空间中距离概念的失效

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
matplotlib.rcParams['axes.unicode_minus'] = False

# 图片保存目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：距离度量函数
# ============================================================================

def euclidean_distance(X_test, X_train):
    """
    计算欧氏距离矩阵。

    使用展开公式: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    避免了显式广播大矩阵，内存效率更高。

    参数:
        X_test: (m, d), m 个测试样本
        X_train: (n, d), n 个训练样本
    返回:
        distances: (m, n), 每对样本之间的欧氏距离
    """
    # 平方和项: (m, 1) 和 (n,)
    sq_test = np.sum(X_test ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_train = np.sum(X_train ** 2, axis=1)                # (n,)
    # 交叉项: (m, d) @ (d, n) = (m, n)
    cross = 2.0 * X_test @ X_train.T
    # 展开公式: a^2 + b^2 - 2ab
    sq_dists = sq_test + sq_train - cross
    # 数值稳定性: 把极小负值 clamp 到 0（浮点误差）
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


def manhattan_distance(X_test, X_train):
    """
    计算曼哈顿距离矩阵 (L1 距离)。

    L1 距离 = sum(|x_i - y_i|)
    比欧氏距离对异常值更鲁棒——不平方放大差异。
    """
    # (m, 1, d) - (n, d) = (m, n, d)
    abs_diff = np.abs(X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :])
    return np.sum(abs_diff, axis=2)  # (m, n)


def cosine_distance(X_test, X_train):
    """
    计算余弦距离矩阵。

    余弦距离 = 1 - 余弦相似度
    度量向量的方向差异，忽略长度。
    广泛用于文本分析、推荐系统。
    """
    # 计算 L2 范数
    norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)   # (m, 1)
    norm_train = np.linalg.norm(X_train, axis=1)                # (n,)
    # 余弦相似度
    dot = X_test @ X_train.T  # (m, n)
    denom = norm_test @ norm_train[np.newaxis, :] + 1e-10
    cos_sim = dot / denom
    # 限制在 [-1, 1] 范围内（浮点误差）
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim


def mahalanobis_distance(X_test, X_train):
    """
    计算马氏距离矩阵。

    马氏距离 = sqrt((x - y)^T @ Sigma^{-1} @ (x - y))
    考虑特征之间的相关性，等价于先做白化变换，再算欧氏距离。

    步骤:
        1. 计算协方差矩阵 Sigma
        2. 用伪逆近似 Sigma^{-1}
        3. 按定义计算马氏距离
    """
    n_train = X_train.shape[0]
    # 无偏协方差估计
    Sigma = np.cov(X_train.T)
    # 用伪逆代替逆（接近奇异时仍能给出解）
    Sigma_inv = np.linalg.pinv(Sigma)

    m, n = X_test.shape[0], X_train.shape[0]
    distances = np.zeros((m, n))
    for i in range(m):
        diff = X_test[i] - X_train  # (n, d)
        # batch: diff @ Sigma_inv @ diff.T -> 取对角线
        distances[i] = np.sqrt(np.sum((diff @ Sigma_inv) * diff, axis=1))
    return distances


# 距离度量映射表
METRIC_FUNCTIONS = {
    'euclidean': euclidean_distance,
    'manhattan': manhattan_distance,
    'cosine': cosine_distance,
    'mahalanobis': mahalanobis_distance,
}


# ============================================================================
# 第二部分：自定义 k-NN 分类器
# ============================================================================

class KNNClassifier:
    """
    从零实现的 k-NN 分类器。

    支持多种距离度量和投票策略:
      - weights='uniform': 每个邻居的票权重相同
      - weights='distance': 使用反距离权重

    参数:
        k: int, 近邻数量
        metric: str, 距离度量 ('euclidean', 'manhattan', 'cosine', 'mahalanobis')
        weights: str, 'uniform' 或 'distance'
    """

    def __init__(self, k=3, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        if metric not in METRIC_FUNCTIONS:
            raise ValueError(f"不支持的度量: {metric}，请选择 {list(METRIC_FUNCTIONS.keys())}")

    def fit(self, X, y):
        """
        k-NN 的训练就是记住数据——惰性学习（lazy learning）。
        没有权重更新、没有梯度下降、没有迭代。
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X_test):
        """
        预测流程:
          1. 计算所有测试样本到训练样本的距离
          2. 找最近的 k 个邻居
          3. 投票决定类别
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        # Step 1: 计算距离矩阵 (m_test, n_train)
        dist_func = METRIC_FUNCTIONS[self.metric]
        distances = dist_func(X_test, self.X_train)

        # Step 2: 找到最近的 k 个邻居的索引
        # argsort 返回从小到大的索引排列
        top_k_indices = np.argsort(distances, axis=1)[:, :self.k]  # (m_test, k)

        # Step 3: 收集这些邻居的标签
        top_k_labels = self.y_train[top_k_indices]  # (m_test, k)
        top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)

        # Step 4: 投票
        if self.weights == 'uniform':
            predictions = self._vote_uniform(top_k_labels)
        elif self.weights == 'distance':
            predictions = self._vote_distance_weighted(top_k_labels, top_k_distances)
        else:
            raise ValueError(f"不支持的权重策略: {self.weights}")

        return predictions

    def _vote_uniform(self, top_k_labels):
        """
        多数投票: 每个邻居一票，票数最多的类别获胜。

        bincount 统计每个类别出现的次数，argmax 选出最多的类别。
        """
        m_test = top_k_labels.shape[0]
        predictions = np.zeros(m_test, dtype=self.y_train.dtype)

        for i in range(m_test):
            counts = np.bincount(top_k_labels[i].astype(int))
            predictions[i] = np.argmax(counts)

        return predictions

    def _vote_distance_weighted(self, top_k_labels, top_k_distances):
        """
        距离加权投票: 每个邻居的票权重 = 1 / (distance + epsilon)。

        步骤:
          1. 计算权重: w = 1 / (d + 1e-6)
          2. 对每个测试样本，累加各邻居对各类别贡献的权重
          3. 选权重最大的类别
        """
        m_test = top_k_labels.shape[0]
        predictions = np.zeros(m_test, dtype=self.y_train.dtype)

        # 权重: 距离越小权重越大
        weights = 1.0 / (top_k_distances + 1e-6)  # (m_test, k)

        # 获取类别数
        n_classes = int(np.max(self.y_train)) + 1

        for i in range(m_test):
            # 对每个类别累计权重
            class_weights = np.zeros(n_classes)
            for j in range(self.k):
                label = int(top_k_labels[i, j])
                class_weights[label] += weights[i, j]
            predictions[i] = np.argmax(class_weights)

        return predictions


# ============================================================================
# 第三部分：可视化函数
# ============================================================================

def plot_decision_boundary(ax, model, X, y, title, step=0.05):
    """
    绘制二维特征空间的决策边界。

    方法: 在特征空间打网格 -> 预测每个格点的类别 -> 用不同的颜色填充
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    # 绘制训练样本
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=50)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')


def plot_distance_comparison():
    """
    展示四种距离度量在二维空间中的"等距线"(isocontour)。
    对于每种度量，展示到中心点的距离等高线。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 生成协方差相关数据来展示马氏距离的特点
    np.random.seed(42)
    n = 300
    # 有相关性的数据
    mean = [0, 0]
    cov = [[2.0, 1.5], [1.5, 1.0]]  # 正相关的协方差矩阵
    data = np.random.multivariate_normal(mean, cov, n)

    center = np.array([0.0, 0.0])
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    Xg, Yg = np.meshgrid(x, y)
    grid = np.c_[Xg.ravel(), Yg.ravel()]

    metrics = ['euclidean', 'manhattan', 'cosine', 'mahalanobis']
    titles = ['Euclidean (L2) Distance Level Sets',
              'Manhattan (L1) Distance Level Sets',
              'Cosine Distance Level Sets',
              'Mahalanobis Distance Level Sets']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        dist_func = METRIC_FUNCTIONS[metric]
        distances = dist_func(grid, center.reshape(1, -1)).reshape(Xg.shape)

        # 绘制等距线
        levels = np.linspace(distances.min(), distances.max(), 15)
        ax.contour(Xg, Yg, distances, levels=levels, cmap='viridis', alpha=0.7)
        # 标注中心点
        ax.scatter([center[0]], [center[1]], c='red', s=120, marker='*',
                   edgecolors='darkred', zorder=5, label='Reference Point')
        # 绘制数据散点
        ax.scatter(data[:, 0], data[:, 1], alpha=0.15, s=10, c='gray')
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')

    plt.tight_layout()
    fp = _save_path('distance_metrics_levelsets.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] distance comparison saved to {fp}")


def plot_k_effect():
    """
    对比不同 k 值下的决策边界: k=1, k=5, k=15。
    展示 k 值如何影响模型的复杂度。
    """
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    # 标准化
    X = StandardScaler().fit_transform(X)

    k_values = [1, 5, 15]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for k, ax in zip(k_values, axes):
        knn = KNNClassifier(k=k, metric='euclidean', weights='uniform')
        knn.fit(X, y)

        title = f'k-NN Classifier (k={k})\n'
        if k == 1:
            title += 'Complex boundary, Low Bias, High Variance'
        elif k == 5:
            title += 'Balanced boundary, Moderate complexity'
        else:
            title += 'Smooth boundary, High Bias, Low Variance'

        plot_decision_boundary(ax, knn, X, y, title)

    plt.tight_layout()
    fp = _save_path('k_effect_boundary.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] k-effect boundary saved to {fp}")


def plot_weight_effect():
    """
    对比均匀投票 vs 距离加权投票的决策差异。
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=2,
        flip_y=0.1, random_state=42
    )
    X = StandardScaler().fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 均匀投票
    knn_uniform = KNNClassifier(k=7, metric='euclidean', weights='uniform')
    knn_uniform.fit(X, y)
    plot_decision_boundary(axes[0], knn_uniform, X, y,
                          'Uniform Weights (k=7)\nEqual vote for each neighbor')

    # 距离加权
    knn_dist = KNNClassifier(k=7, metric='euclidean', weights='distance')
    knn_dist.fit(X, y)
    plot_decision_boundary(axes[1], knn_dist, X, y,
                          'Distance-Weighted (k=7)\nCloser neighbors have more influence')

    plt.tight_layout()
    fp = _save_path('weight_effect_boundary.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] weight effect saved to {fp}")


def plot_curse_of_dimensionality():
    """
    展示维数灾难: 随着维度增加，最近邻和最远邻的距离比值趋近 1。
    """
    np.random.seed(42)
    dims = range(1, 101, 5)  # 1 维到 100 维
    ratios = []
    n_points = 1000

    for d in dims:
        # 在 d 维单位超立方体中均匀采样
        points = np.random.uniform(0, 1, (n_points, d))
        # 计算任意两个点之间的距离（随机采样对来近似）
        idx1 = np.random.choice(n_points, size=min(500, n_points), replace=False)
        idx2 = np.random.choice(n_points, size=min(500, n_points), replace=False)

        diffs = points[idx1] - points[idx2]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))

        if dists.max() > 1e-10:
            ratios.append(dists.min() / (dists.max() + 1e-10))
        else:
            ratios.append(1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(dims), ratios, 'b-o', linewidth=2, markersize=5)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='All distances equal')
    ax.set_xlabel('Dimensionality (d)', fontsize=12)
    ax.set_ylabel('min(dist) / max(dist)', fontsize=12)
    ax.set_title('Curse of Dimensionality\nmin/max distance ratio approaches 1',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('curse_of_dimensionality.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] curse of dimensionality saved to {fp}")


def plot_metric_comparison_accuracy():
    """
    对比不同距离度量在分类任务上的准确率。
    """
    np.random.seed(42)

    # benchmark 数据集
    datasets = {
        'Moons': make_moons(n_samples=300, noise=0.2, random_state=42),
        'Classification': make_classification(
            n_samples=300, n_features=2, n_redundant=0, n_clusters_per_class=1,
            random_state=42
        ),
    }

    metrics = ['euclidean', 'manhattan', 'cosine']
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, (X, y)) in zip(axes, datasets.items()):
        X = StandardScaler().fit_transform(X)

        for metric in metrics:
            accuracies = []
            for k in k_values:
                knn = KNNClassifier(k=k, metric=metric, weights='uniform')
                knn.fit(X, y)
                y_pred = knn.predict(X)
                accuracies.append(accuracy_score(y, y_pred))
            ax.plot(k_values, accuracies, 'o-', linewidth=2, markersize=6,
                   label=metric.capitalize())

        ax.set_xlabel('k (number of neighbors)', fontsize=12)
        ax.set_ylabel('Training Accuracy', fontsize=12)
        ax.set_title(f'Dataset: {name}\nMetric & k value impact on accuracy',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('metric_accuracy_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] metric accuracy comparison saved to {fp}")


def plot_sklearn_comparison():
    """
    对比自定义 k-NN 与 sklearn KNeighborsClassifier。
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_clusters_per_class=1,
        random_state=42
    )
    X = StandardScaler().fit_transform(X)

    # 训练集/测试集划分
    n_train = 200
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    custom_acc = []
    sklearn_acc = []

    for k in k_values:
        # 自定义 k-NN
        knn_custom = KNNClassifier(k=k, metric='euclidean', weights='uniform')
        knn_custom.fit(X_train, y_train)
        custom_acc.append(accuracy_score(y_test, knn_custom.predict(X_test)))

        # sklearn k-NN
        knn_sk = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
        knn_sk.fit(X_train, y_train)
        sklearn_acc.append(accuracy_score(y_test, knn_sk.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, custom_acc, 'bo-', linewidth=2, markersize=8,
            label='Custom KNN (NumPy)')
    ax.plot(k_values, sklearn_acc, 'rs--', linewidth=2, markersize=8,
            markerfacecolor='none', label='Sklearn KNeighborsClassifier')
    ax.set_xlabel('k (number of neighbors)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Custom KNN vs Sklearn Comparison\n(Should give identical results)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('sklearn_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] sklearn comparison saved to {fp}")


# ============================================================================
# 第四部分：主函数
# ============================================================================

def main():
    """主执行流程: 逐步展示所有可视化结果。"""
    print("=" * 70)
    print("ml01_knn/demo.py — k-近邻算法与距离度量")
    print("=" * 70)

    # 1. 距离度量对比
    print("\n[1/6] 绘制四种距离度量的等距线...")
    plot_distance_comparison()

    # 2. k 值对决策边界的影响
    print("\n[2/6] 绘制不同 k 值下的决策边界...")
    plot_k_effect()

    # 3. 均匀投票 vs 距离加权
    print("\n[3/6] 对比均匀投票与距离加权投票...")
    plot_weight_effect()

    # 4. 维数灾难
    print("\n[4/6] 展示维数灾难...")
    plot_curse_of_dimensionality()

    # 5. 不同度量对准确率的影响
    print("\n[5/6] 对比不同距离度量的分类准确率...")
    plot_metric_comparison_accuracy()

    # 6. 与 sklearn 对比
    print("\n[6/6] 对比自定义实现与 sklearn...")
    plot_sklearn_comparison()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
