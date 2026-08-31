# -*- coding: utf-8 -*-
"""
===============================================================================
ml05_decision_tree/code/demo.py — 决策树 (CART)
===============================================================================
本演示从零实现 CART 决策树分类器，全面展示：
  1. 信息熵和基尼指数的计算
  2. 最佳分裂点搜索（连续特征）
  3. 完整决策树的递归构建
  4. 决策边界可视化（对比不同 max_depth）
  5. 预剪枝的效果展示
  6. 与 sklearn DecisionTreeClassifier 的对比

通过本演示，你将理解：
  - 决策树如何通过贪心策略选择最优分裂
  - 基尼指数和信息增益的实际计算过程
  - 为什么无限制的决策树容易过拟合
  - 预剪枝参数如何控制树的复杂度

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.metrics import accuracy_score
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：不纯度度量函数
# ============================================================================

def gini(y):
    """计算基尼指数 Gini = 1 - sum(p_j^2)。"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def entropy(y):
    """计算信息熵 H = -sum(p_j * log2(p_j))。"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))


# ============================================================================
# 第二部分：CART 决策树分类器
# ============================================================================

class CARTDecisionTree:
    """
    CART 决策树分类器（二叉树）。

    支持基尼指数和信息增益两种分裂准则。
    每次分裂选择一个特征和一个阈值，将数据分为左右两个子集。

    参数:
        max_depth: int or None, 最大深度（预剪枝）
        min_samples_split: int, 节点至少多少样本才继续分裂（预剪枝）
        min_samples_leaf: int, 叶节点至少多少样本（预剪枝）
        criterion: str, 'gini' 或 'entropy'
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None

    class Node:
        """决策树节点。"""
        __slots__ = ('feature_idx', 'threshold', 'left', 'right', 'value', 'is_leaf')

        def __init__(self, value=None):
            self.feature_idx = None   # 分裂特征索引
            self.threshold = None     # 分裂阈值
            self.left = None          # 左子树 (X[:, feat] <= threshold)
            self.right = None         # 右子树 (X[:, feat] > threshold)
            self.value = value        # 叶节点: 预测类别
            self.is_leaf = True       # 是否为叶节点

    def fit(self, X, y):
        """构建决策树。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """递归构建子树。"""
        n_samples = len(y)

        # 停止条件
        # 1. 节点中所有样本属于同一类
        if len(np.unique(y)) == 1:
            return self.Node(value=y[0])

        # 2. 达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            return self.Node(value=self._majority_vote(y))

        # 3. 样本数不足
        if n_samples < self.min_samples_split:
            return self.Node(value=self._majority_vote(y))

        # 搜索最佳分裂
        best_feat, best_thresh, best_gain = self._best_split(X, y)

        # 4. 无法找到有效分裂（增益为零）
        if best_feat is None or best_gain <= 1e-10:
            return self.Node(value=self._majority_vote(y))

        # 执行分裂
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # 5. 左右子集样本数不足
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self.Node(value=self._majority_vote(y))

        # 创建内部节点
        node = self.Node()
        node.is_leaf = False
        node.feature_idx = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _best_split(self, X, y):
        """
        搜索最佳分裂 (feature_idx, threshold)。

        对每个特征:
          1. 排序该特征的所有取值
          2. 在相邻取值的中点设置候选阈值
          3. 计算该阈值下的不纯度减少
          4. 保留增益最大的 (feature, threshold)

        返回:
            best_feat: int or None
            best_thresh: float or None
            best_gain: float
        """
        n_samples, n_features = X.shape
        parent_impurity = gini(y) if self.criterion == 'gini' else entropy(y)
        best_gain = -np.inf
        best_feat = None
        best_thresh = None

        for feat_idx in range(n_features):
            # 该特征的所有取值，排序
            feature_values = X[:, feat_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_X = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]

            # 在相邻取值的中点处设为候选阈值
            for i in range(n_samples - 1):
                # 跳过相同值（无法分裂）
                if sorted_X[i] == sorted_X[i + 1]:
                    continue

                threshold = (sorted_X[i] + sorted_X[i + 1]) / 2.0

                # 分裂
                y_left = sorted_y[:i + 1]
                y_right = sorted_y[i + 1:]
                n_left = len(y_left)
                n_right = len(y_right)

                # 计算加权不纯度
                if self.criterion == 'gini':
                    child_impurity = (n_left / n_samples) * gini(y_left) + \
                                    (n_right / n_samples) * gini(y_right)
                else:
                    child_impurity = (n_left / n_samples) * entropy(y_left) + \
                                    (n_right / n_samples) * entropy(y_right)

                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold

        return best_feat, best_thresh, best_gain

    def _majority_vote(self, y):
        """返回出现次数最多的类别。"""
        return np.bincount(y).argmax()

    def predict(self, X):
        """对每个样本，从根节点开始沿树走到叶节点，返回叶节点的值。"""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        """从 node 开始向下遍历，返回叶节点的预测值。"""
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)


# ============================================================================
# 第三部分：树结构与可视化工具
# ============================================================================

def print_tree(node, depth=0, feature_names=None):
    """递归打印决策树结构。"""
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}└── Leaf: class {node.value}")
        return

    feat_name = f"X[{node.feature_idx}]" if feature_names is None else \
                feature_names[node.feature_idx]
    print(f"{indent}├── [{feat_name} <= {node.threshold:.3f}] (internal)")
    print_tree(node.left, depth + 1, feature_names)
    print_tree(node.right, depth + 1, feature_names)


# ============================================================================
# 第四部分：可视化函数
# ============================================================================

def plot_impurity_curves():
    """绘制熵和基尼指数随二分类概率变化的曲线。"""
    p = np.linspace(0.001, 0.999, 500)

    # Entropy: -p*log2(p) - (1-p)*log2(1-p)
    ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # Gini: 2 * p * (1-p)
    gini_val = 2 * p * (1 - p)
    # Classification error: min(p, 1-p) → 1 - max(p, 1-p)
    cls_err = 1 - np.maximum(p, 1 - p)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(p, ent, 'b-', linewidth=2.5, label='Entropy: -p*log2(p) - (1-p)*log2(1-p)')
    ax.plot(p, gini_val, 'r--', linewidth=2.5, label='Gini: 2*p*(1-p)')
    ax.plot(p, cls_err, 'g-.', linewidth=2.5, label='Classification Error: 1-max(p,1-p)')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # 重新缩放让 Gini 和 Entropy 在同一尺度
    ax2 = ax.twinx()
    ax2.plot(p, ent / 2.0, 'b-', linewidth=2, alpha=0.3)

    ax.set_xlabel('p (proportion of class 1)', fontsize=12)
    ax.set_ylabel('Impurity', fontsize=12)
    ax.set_title('Impurity Measures Comparison\n(All peak at p=0.5 → maximally impure)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    fp = _save_path('impurity_curves.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] impurity curves saved to {fp}")


def plot_tree_decision_boundary():
    """展示不同 max_depth 下的决策边界。"""
    np.random.seed(42)
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    max_depths = [1, 2, 3, 5, 10, None]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, md in zip(axes.flat, max_depths):
        tree = CARTDecisionTree(max_depth=md, criterion='gini')
        tree.fit(X, y)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = tree.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                  cmap=plt.cm.RdYlBu, s=25)

        label = f'Unlimited (depth={get_tree_depth(tree.root)})' if md is None else f'Limit={md}'
        acc = accuracy_score(y, tree.predict(X))
        ax.set_title(f'max_depth = {label}\nTrain Acc = {acc:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('tree_depth_boundaries.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] tree depth boundaries saved to {fp}")


def get_tree_depth(node):
    """计算树的实际深度。"""
    if node.is_leaf:
        return 0
    return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))


def plot_sklearn_comparison():
    """与 sklearn DecisionTreeClassifier 对比。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2,
                                n_redundant=0, n_clusters_per_class=1,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    depths = [1, 2, 3, 5, 10, 15]
    custom_scores = []
    sklearn_scores = []

    for md in depths:
        # 自定义
        custom_tree = CARTDecisionTree(max_depth=md, criterion='gini')
        custom_tree.fit(X_train, y_train)
        custom_scores.append(accuracy_score(y_test, custom_tree.predict(X_test)))

        # sklearn
        sk_tree = SkDecisionTree(max_depth=md, criterion='gini', random_state=42)
        sk_tree.fit(X_train, y_train)
        sklearn_scores.append(accuracy_score(y_test, sk_tree.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, custom_scores, 'bo-', linewidth=2, markersize=8,
            label='Custom CART (NumPy)')
    ax.plot(depths, sklearn_scores, 'rs--', linewidth=2, markersize=8,
            markerfacecolor='none', label='Sklearn DecisionTree')
    ax.set_xlabel('max_depth', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Custom CART vs Sklearn DecisionTree', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('sklearn_tree_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] sklearn comparison saved to {fp}")


def plot_tree_structure_text():
    """打印小数据集上的树结构。"""
    np.random.seed(42)
    # 小数据集，方便打印树结构
    X, y = make_classification(n_samples=50, n_features=2,
                                n_redundant=0, n_clusters_per_class=1,
                                random_state=42)

    tree = CARTDecisionTree(max_depth=3, criterion='gini')
    tree.fit(X, y)

    print("\n[Tree Structure (max_depth=3)]")
    print_tree(tree.root)
    print(f"  Total nodes: {count_nodes(tree.root)}, Leaves: {count_leaves(tree.root)}")


def count_nodes(node):
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)


def count_leaves(node):
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml05_decision_tree/demo.py — 决策树 (CART)")
    print("=" * 70)

    print("\n[1/5] 绘制不纯度曲线对比...")
    plot_impurity_curves()

    print("\n[2/5] 不同 max_depth 的决策边界...")
    plot_tree_decision_boundary()

    print("\n[3/5] 与 sklearn 对比验证...")
    plot_sklearn_comparison()

    print("\n[4/5] 打印树结构...")
    plot_tree_structure_text()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
