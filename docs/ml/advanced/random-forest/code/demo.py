# -*- coding: utf-8 -*-
"""
===============================================================================
ml06_random_forest/code/demo.py — Bagging 与随机森林
===============================================================================
本演示从零实现 Bagging 和随机森林，基于 ml05 的 CART 决策树：
  1. Bootstrap 采样的实现和效果演示
  2. Bagging 分类器（多棵树 + 投票）
  3. 随机森林（Bagging + 随机特征子空间）
  4. OOB 误差估计
  5. 特征重要性（MDI）的计算与可视化
  6. 单决策树 vs Bagging vs 随机森林的全面对比
  7. 与 sklearn RandomForestClassifier 对比验证

通过本演示，你将理解：
  - Bootstrap 采样如何产生训练集的多样性
  - 随机特征子空间如何降低树间相关性
  - 为什么随机森林的泛化性能优于单棵决策树
  - OOB 误差如何作为免费的泛化能力估计

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats as scipy_stats
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.metrics import accuracy_score
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：从 ml05 复用的 CART 决策树
# ============================================================================

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


class DecisionTree:
    """CART 决策树（简化版，用于集成）。"""

    class Node:
        __slots__ = ('feature_idx', 'threshold', 'left', 'right', 'value', 'is_leaf')
        def __init__(self, value=None):
            self.feature_idx = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = value
            self.is_leaf = True

    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features  # None = use all features
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        if len(np.unique(y)) == 1:
            return self.Node(value=y[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return self.Node(value=np.bincount(y).argmax())
        if n_samples < 2:
            return self.Node(value=np.bincount(y).argmax())

        best_feat, best_thresh, best_gain = self._best_split(X, y)
        if best_feat is None or best_gain <= 0:
            return self.Node(value=np.bincount(y).argmax())

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self.Node(value=np.bincount(y).argmax())

        node = self.Node()
        node.is_leaf = False
        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_impurity = gini(y)
        best_gain = -np.inf
        best_feat = None
        best_thresh = None

        # 随机特征子空间
        if self.max_features is None:
            features_to_use = list(range(n_features))
        else:
            m = min(self.max_features, n_features)
            features_to_use = np.random.choice(n_features, size=m, replace=False)

        for feat_idx in features_to_use:
            feature_values = X[:, feat_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_X = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            for i in range(n_samples - 1):
                if sorted_X[i] == sorted_X[i + 1]:
                    continue
                threshold = (sorted_X[i] + sorted_X[i + 1]) / 2.0
                y_left = sorted_y[:i + 1]
                y_right = sorted_y[i + 1:]
                n_left, n_right = len(y_left), len(y_right)
                child_impurity = (n_left/n_samples)*gini(y_left) + \
                                (n_right/n_samples)*gini(y_right)
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh, best_gain

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)


# ============================================================================
# 第二部分：随机森林（Bagging + 随机特征子空间）
# ============================================================================

class RandomForest:
    """
    随机森林分类器。

    核心机制:
      1. Bootstrap 采样: 每棵树用不同的 Bootstrap 子集训练
      2. 随机特征子空间: 每个分裂节点只考虑 max_features 个随机特征
      3. 多数投票: 所有树的预测结果取众数

    参数:
        n_estimators: int, 树的数量
        max_depth: int or None, 每棵树的最大深度
        max_features: int or str, 每次分裂考虑的特征数
                      ('sqrt' = sqrt(n_features), 'all' = n_features)
    """

    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.bootstrap_indices = []  # 记录每棵树的 Bootstrap 采样索引

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape

        # 确定每次分裂的特征数
        if self.max_features == 'sqrt':
            m = int(np.sqrt(n_features))
        elif self.max_features == 'all':
            m = n_features
        else:
            m = min(self.max_features, n_features)

        self.trees = []
        self.bootstrap_indices = []

        for i in range(self.n_estimators):
            # Bootstrap 采样 (有放回)
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            # 训练单棵树
            tree = DecisionTree(max_depth=self.max_depth, max_features=m)
            tree.fit(X_boot, y_boot)

            self.trees.append(tree)
            self.bootstrap_indices.append(boot_idx)

        return self

    def predict(self, X):
        """多数投票。"""
        X = np.asarray(X, dtype=np.float64)
        # 收集所有树的预测 (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # 对每个样本，取出现次数最多的类别
        predictions = scipy_stats.mode(all_preds, axis=0, keepdims=False)[0]
        return predictions.ravel()

    def oob_score(self, X, y):
        """
        计算 OOB (Out-of-Bag) 准确率。

        对于每个样本 i，找到所有未使用样本 i 的树，用这些树的多数投票预测。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples = len(y)

        oob_predictions = np.zeros(n_samples, dtype=np.int64)
        oob_covered = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            # 找到所有未使用样本 i 的树
            tree_indices = []
            for t_idx in range(self.n_estimators):
                if i not in self.bootstrap_indices[t_idx]:
                    tree_indices.append(t_idx)

            if len(tree_indices) == 0:
                continue

            oob_covered[i] = True
            # 只用这些树做预测
            votes = [self.trees[t].predict(X[i:i+1])[0] for t in tree_indices]
            oob_predictions[i] = scipy_stats.mode(votes, keepdims=False)[0]

        # 只对有 OOB 覆盖的样本计算准确率
        if np.sum(oob_covered) == 0:
            return 0.0
        return accuracy_score(y[oob_covered], oob_predictions[oob_covered])


# ============================================================================
# 第三部分：可视化函数
# ============================================================================

def plot_bootstrap_demo():
    """展示 Bootstrap 采样的效果。"""
    np.random.seed(42)
    n_samples = 100

    # 生成一些示例数据
    original = np.arange(n_samples)

    # 三次 Bootstrap 采样
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ['Bootstrap Sample 1', 'Bootstrap Sample 2', 'Bootstrap Sample 3']

    for ax, title in zip(axes, titles):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        # 统计每个样本被选中的次数
        counts = np.bincount(boot_idx, minlength=n_samples)
        # 统计 OOB 样本
        oob_mask = counts == 0

        colors = np.where(counts > 0, 'steelblue', 'lightgray')
        ax.bar(original, counts, color=colors, alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        n_oob = np.sum(oob_mask)
        ax.set_title(f'{title}\nSelected: {n_samples - n_oob}, OOB: {n_oob} ({n_oob/n_samples*100:.1f}%)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Selection Count', fontsize=9)

    plt.tight_layout()
    fp = _save_path('bootstrap_demo.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] bootstrap demo saved to {fp}")


def plot_single_vs_rf_boundary():
    """对比单决策树 vs Bagging vs 随机森林的决策边界。"""
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 单棵树
    single_tree = DecisionTree(max_depth=None)
    single_tree.fit(X_train, y_train)

    # Bagging（无随机特征子空间）
    bagging = RandomForest(n_estimators=50, max_depth=None, max_features='all')
    bagging.fit(X_train, y_train)

    # 随机森林（有随机特征子空间）
    rf = RandomForest(n_estimators=50, max_depth=None, max_features='sqrt')
    rf.fit(X_train, y_train)

    models = [
        (single_tree, 'Single Decision Tree\n(High Variance)'),
        (bagging, 'Bagging (50 trees, all features)\n(Bootstrap only)'),
        (rf, 'Random Forest (50 trees, sqrt features)\n(Bootstrap + Feature Sampling)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (model, title) in zip(axes, models):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k',
                  cmap=plt.cm.RdYlBu, s=30)

        test_acc = accuracy_score(y_test, model.predict(X_test))
        ax.set_title(f'{title}\nTest Acc = {test_acc:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('single_vs_rf_boundary.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] single vs RF boundary saved to {fp}")


def plot_oob_error_curve():
    """绘制 OOB 误差随树数量变化的曲线。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10,
                                n_informative=5, n_redundant=2,
                                random_state=42)

    n_estimators_list = [5, 10, 20, 30, 50, 75, 100, 150]

    oob_scores = []
    train_scores = []
    test_scores = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for n_est in n_estimators_list:
        rf = RandomForest(n_estimators=n_est, max_depth=None, max_features='sqrt')
        rf.fit(X_train, y_train)
        oob_scores.append(rf.oob_score(X_train, y_train))
        train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, rf.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_estimators_list, oob_scores, 'go-', linewidth=2, markersize=8,
            label='OOB Score (free validation)')
    ax.plot(n_estimators_list, train_scores, 'bs-', linewidth=2, markersize=8,
            label='Train Score')
    ax.plot(n_estimators_list, test_scores, 'r^--', linewidth=2, markersize=8,
            markerfacecolor='none', label='Test Score')

    ax.set_xlabel('Number of Trees (n_estimators)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('OOB Error vs Tree Count in Random Forest\n(OOB approx. tracks test accuracy)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('oob_error_curve.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] OOB error curve saved to {fp}")


def plot_sklearn_comparison():
    """与 sklearn RandomForestClassifier 对比。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=5,
                                n_informative=3, n_redundant=1,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_estimators_list = [5, 10, 20, 30, 50]
    custom_scores = []
    sklearn_scores = []

    for n_est in n_estimators_list:
        # 自定义
        rf_custom = RandomForest(n_estimators=n_est, max_depth=None, max_features='sqrt')
        rf_custom.fit(X_train, y_train)
        custom_scores.append(accuracy_score(y_test, rf_custom.predict(X_test)))

        # sklearn
        rf_sk = SkRandomForest(n_estimators=n_est, max_depth=None,
                               max_features='sqrt', random_state=42)
        rf_sk.fit(X_train, y_train)
        sklearn_scores.append(accuracy_score(y_test, rf_sk.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_estimators_list, custom_scores, 'bo-', linewidth=2, markersize=8,
            label='Custom Random Forest (NumPy)')
    ax.plot(n_estimators_list, sklearn_scores, 'rs--', linewidth=2, markersize=8,
            markerfacecolor='none', label='Sklearn RandomForest')
    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Custom RF vs Sklearn RandomForest', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('sklearn_rf_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] sklearn comparison saved to {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml06_random_forest/demo.py — Bagging 与随机森林")
    print("=" * 70)

    print("\n[1/4] Bootstrap 采样演示...")
    plot_bootstrap_demo()

    print("\n[2/4] 单树 vs Bagging vs 随机森林决策边界...")
    plot_single_vs_rf_boundary()

    print("\n[3/4] OOB 误差曲线...")
    plot_oob_error_curve()

    print("\n[4/4] 与 sklearn 对比...")
    plot_sklearn_comparison()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
