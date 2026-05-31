# -*- coding: utf-8 -*-
"""
===============================================================================
ml04_svm/code/demo.py — 支持向量机 (SVM)
===============================================================================
本演示从零实现线性 SVM（Hinge Loss + L2 正则化，SGD 优化），全面展示：
  1. 最大间隔分类器的几何直觉
  2. 线性 SVM 的 SGD 实现（Hinge Loss + L2 正则化）
  3. 软间隔与惩罚参数 C 的效果
  4. 核技巧：线性核、多项式核、RBF 核对比
  5. 支持向量的可视化
  6. 与 sklearn SVC 的对比验证

通过本演示，你将理解：
  - SVM 如何通过最大化间隔来降低泛化误差
  - Hinge Loss 的梯度形式和 SGD 优化过程
  - C 参数如何控制间隔和错误的权衡
  - 核函数如何将非线性问题变为线性可分

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：自定义线性 SVM（Hinge Loss + SGD）
# ============================================================================

class LinearSVM:
    """
    线性 SVM，使用 Hinge Loss + L2 正则化，通过 SGD 优化。

    损失函数: J(w,b) = (1/n) * sum(max(0, 1 - y_i*(w^T x_i + b))) + lambda * ||w||^2

    Hinge Loss 的子梯度:
      - 若 y_i*(w^T x_i + b) < 1: 梯度含 -y_i*x_i (违反间隔)
      - 若 y_i*(w^T x_i + b) >= 1: 梯度不含此项 (在间隔外，不产生损失)

    参数:
        C: float, 惩罚参数（越大越像硬间隔，等价于 lambda = 1/(2C)）
        learning_rate: float, 学习率
        n_epochs: int, 训练轮数
    """

    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=200):
        self.C = C
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        """SGD 训练线性 SVM。"""
        X = np.asarray(X, dtype=np.float64)
        # 将标签 0/1 转为 -1/+1
        y_svm = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0.0

        # lambda = 1 / (2 * C)，使得 C 越大正则化越弱
        lambda_ = 1.0 / (2.0 * self.C) if self.C > 1e-10 else 0.0

        for epoch in range(self.n_epochs):
            # 随机打乱数据（SGD 的标准做法）
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_svm[indices]

            for i in range(n_samples):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                # 计算间隔
                margin = y_i * (np.dot(self.w, x_i) + self.b)

                # L2 正则化梯度
                dw = 2 * lambda_ * self.w
                db = 0.0

                # Hinge Loss 子梯度（仅当违反间隔时）
                if margin < 1:
                    dw -= y_i * x_i
                    db -= y_i

                # 梯度更新
                self.w -= self.lr * dw
                self.b -= self.lr * db

        return self

    def decision_function(self, X):
        """返回 f(x) = w^T x + b。"""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w + self.b

    def predict(self, X):
        """预测类别 0/1。"""
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def get_support_vector_mask(self, X, y):
        """
        识别支持向量: margin <= 1 + epsilon 且 margin >= 0.99
        即位于决策边界的样本。
        """
        y_svm = np.where(y <= 0, -1, 1)
        margins = y_svm * self.decision_function(X)
        sv_mask = (margins >= 0.99) & (margins <= 1.01)
        return sv_mask


# ============================================================================
# 第二部分：RBF 核 SVM（预计算核矩阵的简单版本）
# ============================================================================

def rbf_kernel(X, Y, gamma):
    """
    RBF (高斯) 核函数。

    K(x, y) = exp(-gamma * ||x - y||^2)

    参数:
        X: (m, d)
        Y: (n, d)
        gamma: float
    返回:
        K: (m, n) 核矩阵
    """
    sq_X = np.sum(X ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_Y = np.sum(Y ** 2, axis=1)                 # (n,)
    sq_dists = np.maximum(sq_X + sq_Y - 2 * X @ Y.T, 0.0)
    return np.exp(-gamma * sq_dists)


# ============================================================================
# 第三部分：可视化函数
# ============================================================================

def plot_maximum_margin_geometry():
    """展示最大间隔分类器的几何概念。"""
    np.random.seed(42)

    # 生成完全线性可分的数据
    X0 = np.random.randn(20, 2) * 1.0 + np.array([-2, -2])
    X1 = np.random.randn(20, 2) * 1.0 + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(20), np.ones(20)])

    # 训练线性 SVM
    svm = SVC(kernel='linear', C=1e5)  # 很大 C → 近似硬间隔
    svm.fit(X, y)

    w = svm.coef_[0]
    b = svm.intercept_[0]
    support_vectors = svm.support_vectors_

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制数据点
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='steelblue', s=60,
              edgecolors='k', label='Class 0 (-1)')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='coral', s=60,
              edgecolors='k', label='Class 1 (+1)')

    # 标注支持向量
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1],
              s=200, facecolors='none', edgecolors='green',
              linewidths=2.5, label='Support Vectors')

    # 绘制决策边界和间隔边界
    x_limits = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
    y_limits = np.array([X[:, 1].min() - 1, X[:, 1].max() + 1])

    # 决策边界: w^T x + b = 0
    xx = np.linspace(x_limits[0], x_limits[1], 100)
    yy = -(w[0] * xx + b) / w[1]

    ax.plot(xx, yy, 'k-', linewidth=2, label='Decision Boundary (w^Tx+b=0)')

    # 间隔边界: w^T x + b = ±1
    margin = 1.0 / np.linalg.norm(w)
    yy_plus = -(w[0] * xx + b - 1) / w[1]
    yy_minus = -(w[0] * xx + b + 1) / w[1]

    ax.plot(xx, yy_plus, 'k--', linewidth=1.5, alpha=0.6, label='Margin (w^Tx+b=±1)')
    ax.plot(xx, yy_minus, 'k--', linewidth=1.5, alpha=0.6)

    # 填充间隔区域
    ax.fill_between(xx, yy_minus, yy_plus, alpha=0.1, color='green')

    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'Maximum Margin Classifier\nMargin width = {2*margin:.3f}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    plt.tight_layout()
    fp = _save_path('maximum_margin_geometry.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] maximum margin geometry saved to {fp}")


def plot_soft_margin_C_effect():
    """展示不同 C 值对决策边界的影响。"""
    np.random.seed(42)
    # 生成近乎线性可分但有些重叠的数据
    X, y = make_classification(n_samples=200, n_features=2,
                                n_redundant=0, n_clusters_per_class=1,
                                flip_y=0.15, random_state=42)
    X = StandardScaler().fit_transform(X)

    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for C, ax in zip(C_values, axes):
        svm = LinearSVM(C=C, learning_rate=0.01, n_epochs=300)
        svm.fit(X, y)

        # 决策边界
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                            np.linspace(y_min, y_max, 150))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                  cmap=plt.cm.RdYlBu, s=30)

        # 标注支持向量
        sv_mask = svm.get_support_vector_mask(X, y)
        if np.any(sv_mask):
            ax.scatter(X[sv_mask, 0], X[sv_mask, 1],
                      facecolors='none', edgecolors='lime',
                      linewidths=1.5, s=100)

        train_acc = accuracy_score(y, svm.predict(X))
        ax.set_title(f'C = {C}\nAcc = {train_acc:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('soft_margin_C_effect.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] soft margin C effect saved to {fp}")


def plot_kernel_comparison():
    """对比线性、多项式、RBF 核在非线性可分数据上的效果。"""
    np.random.seed(42)

    # 生成三个不同难度的数据集
    datasets = {
        'Moons': make_moons(n_samples=200, noise=0.2, random_state=42),
        'Circles': make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42),
    }
    kernels = ['linear', 'poly', 'rbf']
    kernel_names = {'linear': 'Linear', 'poly': 'Polynomial', 'rbf': 'RBF'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row, (name, (X, y)) in enumerate(datasets.items()):
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        for col, kernel in enumerate(kernels):
            ax = axes[row, col]

            if kernel == 'linear':
                svm = LinearSVM(C=1.0, learning_rate=0.02, n_epochs=300)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                score_func = svm.decision_function
            else:
                sk_svm = SVC(kernel=kernel, C=1.0, gamma='scale')
                sk_svm.fit(X_train, y_train)
                y_pred = sk_svm.predict(X_test)
                score_func = lambda X: sk_svm.decision_function(X)

            # 绘制决策边界
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                np.linspace(y_min, y_max, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = (score_func(grid) >= 0) if kernel == 'linear' else \
                sk_svm.predict(grid)
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                      cmap=plt.cm.RdYlBu, s=25, alpha=0.6)

            acc = accuracy_score(y_test, y_pred)
            ax.set_title(f'{name} — {kernel_names[kernel]}\nTest Acc = {acc:.3f}',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Feature 1', fontsize=9)
            ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('kernel_comparison.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] kernel comparison saved to {fp}")


def plot_rbf_gamma_effect():
    """展示 RBF 核的 gamma 参数对决策边界的影响。"""
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)

    gammas = [0.1, 1.0, 10.0, 50.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for gamma, ax in zip(gammas, axes):
        svm = SVC(kernel='rbf', C=1.0, gamma=gamma)
        svm.fit(X, y)

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k',
                  cmap=plt.cm.RdYlBu, s=25)

        n_sv = len(svm.support_vectors_)
        description = 'Underfit' if gamma < 0.2 else \
                     'Good fit' if gamma < 5 else \
                     'Overfit'
        ax.set_title(f'RBF Kernel (gamma={gamma})\n{description}, {n_sv} Support Vectors',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)

    plt.tight_layout()
    fp = _save_path('rbf_gamma_effect.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] RBF gamma effect saved to {fp}")


def plot_hinge_loss_function():
    """绘制 Hinge Loss 与其他损失函数的对比。"""
    z = np.linspace(-2, 3, 500)

    # Hinge Loss: max(0, 1 - z)
    hinge = np.maximum(0, 1 - z)
    # 0-1 Loss: 指示 z < 0
    zero_one = (z < 0).astype(float)
    # Logistic Loss: log(1 + exp(-z))
    logistic = np.log(1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(z, hinge, 'b-', linewidth=2.5, label='Hinge Loss: max(0, 1-z)')
    ax.plot(z, zero_one, 'r--', linewidth=2, label='0-1 Loss')
    ax.plot(z, logistic, 'g-.', linewidth=2, label='Logistic Loss: log(1+e^{-z})')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 标注关键区域
    ax.axvspan(-2, 0, alpha=0.08, color='red', label='Misclassified region')
    ax.axvspan(0, 1, alpha=0.08, color='yellow', label='Margin region')
    ax.axvspan(1, 3, alpha=0.08, color='green', label='Correct & confident')

    ax.set_xlabel('z = y * f(x)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Hinge Loss vs Other Classification Losses', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 3.5)

    plt.tight_layout()
    fp = _save_path('hinge_loss.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] hinge loss saved to {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("ml04_svm/demo.py — 支持向量机 (SVM)")
    print("=" * 70)

    print("\n[1/5] 展示最大间隔分类器几何...")
    plot_maximum_margin_geometry()

    print("\n[2/5] 对比不同 C 值的效果（软间隔）...")
    plot_soft_margin_C_effect()

    print("\n[3/5] 核函数对比...")
    plot_kernel_comparison()

    print("\n[4/5] RBF 核 gamma 参数效果...")
    plot_rbf_gamma_effect()

    print("\n[5/5] Hinge Loss 与其他损失函数对比...")
    plot_hinge_loss_function()

    print("\n" + "=" * 70)
    print("全部可视化完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
