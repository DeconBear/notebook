# -*- coding: utf-8 -*-
"""
===============================================================================
ml07_boosting/code/demo.py — Boosting 与 Stacking
===============================================================================
本演示从零实现 AdaBoost 和简单 GBDT 回归器，全面展示：
  1. AdaBoost：指数损失 + 样本权重自适应更新
  2. 简单 GBDT 回归器：拟合残差的梯度提升
  3. AdaBoost vs 单决策树 vs 随机森林的性能对比
  4. AdaBoost 的每轮错误率和学习器权重变化
  5. GBDT 的迭代改进过程可视化

通过本演示，你将理解：
  - Boosting 如何通过串行修正错误来降低偏差
  - AdaBoost 的样本权重更新机制
  - GBDT "拟合残差"的直观含义
  - Boosting 与 Bagging 的关键区别（降低偏差 vs 降低方差）

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：决策树桩（弱学习器）
# ============================================================================

class DecisionStump:
    """
    决策树桩——深度为 1 的决策树，只做一个分裂。

    这是 AdaBoost 最常用的弱学习器。
    """

    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1  # 1 表示左=负类，-1 表示左=正类
        self.left_value = None
        self.right_value = None

    def fit(self, X, y, sample_weight=None):
        """找到最优的单特征单阈值分裂。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)

        n_samples, n_features = X.shape
        best_error = np.inf

        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            sorted_indices = np.argsort(feature_values)
            sorted_X = feature_values[sorted_indices]
            sorted_y = y[sorted_indices]
            sorted_w = sample_weight[sorted_indices]

            # 累积加权误差的左右计数
            for i in range(n_samples - 1):
                if sorted_X[i] == sorted_X[i + 1]:
                    continue
                threshold = (sorted_X[i] + sorted_X[i + 1]) / 2.0

                # 左子集和右子集
                y_left = sorted_y[:i + 1]
                w_left = sorted_w[:i + 1]
                y_right = sorted_y[i + 1:]
                w_right = sorted_w[i + 1:]

                # 假设 left=class 0, right=class 1
                error_left = np.sum(w_left[y_left != 0]) + np.sum(w_right[y_right != 1])
                # 假设 left=class 1, right=class 0
                error_right = np.sum(w_left[y_left != 1]) + np.sum(w_right[y_right != 0])

                if error_left < best_error:
                    best_error = error_left
                    self.feature_idx = feat_idx
                    self.threshold = threshold
                    self.polarity = 1

                if error_right < best_error:
                    best_error = error_right
                    self.feature_idx = feat_idx
                    self.threshold = threshold
                    self.polarity = -1

        return self

    def predict(self, X):
        """预测标签 0/1。"""
        X = np.asarray(X, dtype=np.float64)
        left_mask = X[:, self.feature_idx] <= self.threshold

        predictions = np.zeros(len(X), dtype=np.int64)
        if self.polarity == 1:
            predictions[left_mask] = 0
            predictions[~left_mask] = 1
        else:
            predictions[left_mask] = 1
            predictions[~left_mask] = 0

        return predictions


# ============================================================================
# 第二部分：AdaBoost 分类器
# ============================================================================

class AdaBoost:
    """
    AdaBoost 分类器。

    算法:
      1. 初始化样本权重 w_i = 1/n
      2. 每一轮:
         a. 用加权样本训练弱学习器
         b. 计算加权错误率 epsilon
         c. 计算学习器权重 alpha = 0.5 * log((1-epsilon)/epsilon)
         d. 更新样本权重: 错误样本 w_i *= exp(alpha)
         e. 归一化权重
      3. 最终预测: sign(sum alpha_m * h_m(x))

    参数:
        n_estimators: int, 弱学习器数量
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []       # 每个学习器的权重 alpha_m
        self.stumps = []       # 决策树桩
        self.errors_per_round = []

    def fit(self, X, y):
        """训练 AdaBoost。标签 0/1 内部转为 -1/+1。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        y_svm = np.where(y <= 0, -1, 1)  # 转为 -1/+1

        n_samples = len(y)
        w = np.ones(n_samples) / n_samples  # 初始化等权重

        self.alphas = []
        self.stumps = []
        self.errors_per_round = []

        for m in range(self.n_estimators):
            # 1. 训练弱学习器（深度 1 的决策树桩）
            stump = DecisionStump()
            stump.fit(X, y, sample_weight=w)

            # 2. 计算加权错误率
            y_pred = stump.predict(X)
            incorrect = (y_pred != y).astype(float)
            err = np.sum(w * incorrect) / np.sum(w)
            err = max(err, 1e-10)  # 防止除零

            self.errors_per_round.append(err)

            # 3. 计算学习器权重
            alpha = 0.5 * np.log((1.0 - err) / err)
            self.alphas.append(alpha)
            self.stumps.append(stump)

            # 如果错误率为 0（完美分类），提前停止
            if err < 1e-10:
                break

            # 4. 更新样本权重（错误样本权重增大）
            w = w * np.exp(alpha * incorrect)
            w = w / np.sum(w)  # 归一化

        self.alphas = np.array(self.alphas)
        return self

    def predict(self, X):
        """预测标签 0/1。"""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # 累加 alpha_m * (2*h_m(x) - 1)（将 0/1 转为 -1/+1）
        scores = np.zeros(n_samples)
        for alpha, stump in zip(self.alphas, self.stumps):
            # 将 0/1 预测转为 -1/+1
            pred_svm = 2 * stump.predict(X) - 1
            scores += alpha * pred_svm

        return (scores >= 0).astype(np.int64)

    def staged_score(self, X, y):
        """返回每一轮迭代后的累积预测（依序）。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        staged_preds = []

        for alpha, stump in zip(self.alphas, self.stumps):
            pred_svm = 2 * stump.predict(X) - 1
            scores += alpha * pred_svm
            staged_preds.append((scores >= 0).astype(np.int64))

        return np.array(staged_preds)


# ============================================================================
# 第三部分：简单 GBDT 回归器
# ============================================================================

class SimpleGBDT:
    """
    简单梯度提升回归树（使用平方损失）。

    当损失函数为 MSE = 0.5*(y - F)^2 时:
      负梯度 = -dL/dF = y - F = residual
    因此每轮只需拟合残差即可。

    参数:
        n_estimators: int, 树的数量
        learning_rate: float, 学习率（收缩因子）
        max_depth: int, 每棵树的最大深度
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.init_value = None
        self.trees = []

    def fit(self, X, y):
        """训练 GBDT 回归器。"""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # 初始化: 常数预测（均值）
        self.init_value = np.mean(y)
        F = np.full(len(y), self.init_value)

        self.trees = []

        for m in range(self.n_estimators):
            # 1. 计算残差 (MSE 的负梯度)
            residuals = y - F

            # 2. 训练回归树拟合残差
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=m)
            tree.fit(X, residuals)

            # 3. 更新集成预测
            F += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

        return self

    def predict(self, X):
        """预测。"""
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.init_value)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        """返回各轮迭代后的预测序列。"""
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.init_value)
        staged = []
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
            staged.append(F.copy())
        return np.array(staged)


# ============================================================================
# 第四部分：可视化函数
# ============================================================================

def plot_adaboost_training():
    """展示 AdaBoost 训练过程中的错误率和 alpha 变化。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10,
                                n_informative=5, n_redundant=2,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ada = AdaBoost(n_estimators=100)
    ada.fit(X_train, y_train)

    # 计算每轮的测试准确率
    staged_preds = ada.staged_score(X_test, y_test)
    test_acc = [accuracy_score(y_test, staged_preds[i]) for i in range(len(staged_preds))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1: 错误率和 alpha
    ax = axes[0]
    rounds = range(len(ada.errors_per_round))
    ax.plot(rounds, ada.errors_per_round, 'r-', linewidth=2, label='Weighted Training Error')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('Weighted Error', fontsize=12, color='red')
    ax.tick_params(axis='y', labelcolor='red')

    ax2 = ax.twinx()
    ax2.plot(rounds, ada.alphas, 'b-', linewidth=1.5, label='Alpha (learner weight)')
    ax2.set_ylabel('Alpha', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax.set_title('AdaBoost Training: Error & Alpha per Round', fontsize=12, fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # 子图 2: 测试准确率
    ax = axes[1]
    ax.plot(rounds[:len(test_acc)], test_acc, 'g-', linewidth=2)
    ax.set_xlabel('Boosting Round', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'AdaBoost Test Accuracy Over Rounds\n(Final: {test_acc[-1]:.4f})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('adaboost_training.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] AdaBoost training saved to {fp}")


def plot_stump_vs_adaboost_vs_rf():
    """对比单决策树桩 vs AdaBoost vs 随机森林。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=400, n_features=2,
                                n_redundant=0, n_clusters_per_class=2,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Decision Stump': DecisionStump(),
        'AdaBoost (50 stumps)': AdaBoost(n_estimators=50),
        'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
    }

    for name in models:
        models[name].fit(X_train, y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (name, model) in zip(axes, models.items()):
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
        ax.set_title(f'{name}\nTest Acc = {test_acc:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('stump_vs_adaboost_vs_rf.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] stump vs AdaBoost vs RF saved to {fp}")


def plot_gbdt_iterations():
    """展示 GBDT 回归的迭代改进过程。"""
    np.random.seed(42)
    X, y = make_friedman1(n_samples=200, n_features=5, noise=0.2, random_state=42)
    # 只使用第一个特征以便可视化
    X_1d = X[:, 0:1]
    y = y + np.random.randn(len(y)) * 2.0

    X_train, X_test, y_train, y_test = train_test_split(X_1d, y, test_size=0.3, random_state=42)

    gbdt = SimpleGBDT(n_estimators=200, learning_rate=0.1, max_depth=2)
    gbdt.fit(X_train, y_train)

    # 获取阶段性预测
    rounds_to_show = [0, 4, 9, 49, 199]  # 第 1, 5, 10, 50, 200 轮
    staged_preds = gbdt.staged_predict(X_test)

    x_line = np.linspace(X_1d.min(), X_1d.max(), 500).reshape(-1, 1)
    staged_preds_line = gbdt.staged_predict(x_line)

    fig, axes = plt.subplots(1, len(rounds_to_show), figsize=(22, 4.5))

    for ax, r in zip(axes, rounds_to_show):
        if r < len(staged_preds_line):
            ax.plot(x_line, staged_preds_line[r], 'r-', linewidth=2, label='GBDT Prediction')
        else:
            ax.plot(x_line, gbdt.predict(x_line), 'r-', linewidth=2, label='GBDT Prediction')
        ax.scatter(X_test, y_test, c='steelblue', s=20, alpha=0.6, label='Test Data')

        if r < len(staged_preds):
            mse = mean_squared_error(y_test, staged_preds[r])
        else:
            mse = mean_squared_error(y_test, gbdt.predict(X_test))
        ax.set_title(f'Round {r+1}\nMSE = {mse:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('x', fontsize=9)
        ax.set_ylabel('y', fontsize=9)
        if r == rounds_to_show[0]:
            ax.legend(fontsize=7)

    plt.tight_layout()
    fp = _save_path('gbdt_iterations.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] GBDT iterations saved to {fp}")


def plot_boosting_sklearn_comparison():
    """与 sklearn AdaBoost 对比。"""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=5,
                                n_informative=3, n_redundant=1,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_estimators_list = [5, 10, 20, 30, 50, 75, 100]
    custom_scores = []
    sklearn_scores = []

    for n_est in n_estimators_list:
        ada_custom = AdaBoost(n_estimators=n_est)
        ada_custom.fit(X_train, y_train)
        custom_scores.append(accuracy_score(y_test, ada_custom.predict(X_test)))

        ada_sk = SkAdaBoost(n_estimators=n_est, algorithm='SAMME.R', random_state=42)
        ada_sk.fit(X_train, y_train)
        sklearn_scores.append(accuracy_score(y_test, ada_sk.predict(X_test)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_estimators_list, custom_scores, 'bo-', linewidth=2, markersize=8,
            label='Custom AdaBoost (NumPy)')
    ax.plot(n_estimators_list, sklearn_scores, 'rs--', linewidth=2, markersize=8,
            markerfacecolor='none', label='Sklearn AdaBoost')
    ax.set_xlabel('Number of Estimators', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Custom AdaBoost vs Sklearn AdaBoost', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('adaboost_sklearn_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] AdaBoost sklearn comparison saved to {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml07_boosting/demo.py — Boosting 与 Stacking")
    print("=" * 70)

    print("\n[1/4] AdaBoost 训练过程可视化...")
    plot_adaboost_training()

    print("\n[2/4] 决策树桩 vs AdaBoost vs 随机森林...")
    plot_stump_vs_adaboost_vs_rf()

    print("\n[3/4] GBDT 回归迭代过程...")
    plot_gbdt_iterations()

    print("\n[4/4] 与 sklearn AdaBoost 对比...")
    plot_boosting_sklearn_comparison()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
