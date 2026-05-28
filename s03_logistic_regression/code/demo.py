# -*- coding: utf-8 -*-
"""
===============================================================================
s03_logistic_regression/code/demo.py — 逻辑回归从零实现
===============================================================================
本演示从零实现逻辑回归，涵盖二分类和多分类（Softmax），使用 Iris 数据集。
内容包括 Sigmoid 函数、交叉熵损失、梯度下降、决策边界可视化等。

通过本演示，你将理解：
  1. Sigmoid 函数如何将实数映射为 (0,1) 的概率
  2. 交叉熵损失为何是分类问题的标准选择
  3. 逻辑回归的梯度为何如此简洁：∂L/∂z = ŷ - y
  4. 决策边界的几何含义
  5. Softmax 如何将二分类推广到多分类
  6. 概率热力图——展示模型的「置信度」分布

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_iris  # 加载经典的 Iris 数据集
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.linear_model import LogisticRegression as SklearnLR  # sklearn 实现（对比用）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：核心函数
# ============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid 激活函数。

    σ(z) = 1 / (1 + e^(-z))
    将任意实数 z 映射到 (0, 1) 区间，解释为概率。

    参数:
        z: np.ndarray, 输入值（可以是标量、向量或矩阵）

    返回:
        np.ndarray, Sigmoid 输出，形状与 z 相同
    """
    # np.clip 防止数值溢出：当 z 很大时 e^(-z) ≈ 0，z 很小时 e^(-z) ≈ inf
    z_clipped = np.clip(z, -500, 500)  # 将 z 限制在 [-500, 500] 内
    return 1.0 / (1.0 + np.exp(-z_clipped))  # 计算 Sigmoid 函数


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax 函数：将 K 个原始得分转化为概率分布。

    softmax(z_k) = e^{z_k} / Σ_j e^{z_j}

    技巧：先减去最大值 (z - max(z)) 以提高数值稳定性，
    因为 e^{z_i - max(z)} 的最大值为 1，不会溢出。

    参数:
        z: np.ndarray, 形状 (n_samples, n_classes)，原始得分矩阵

    返回:
        np.ndarray, 形状 (n_samples, n_classes)，概率分布矩阵
    """
    # 数值稳定技巧：减去每行的最大值
    z_stable = z - np.max(z, axis=1, keepdims=True)  # 保持形状 (n, 1) 以便广播
    exp_z = np.exp(z_stable)  # 计算指数
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # 归一化为概率


# ============================================================================
# 第二部分：二分类逻辑回归
# ============================================================================

class LogisticRegression:
    """
    二分类逻辑回归（使用梯度下降法）。

    模型: P(y=1|x) = σ(w^T x + b)，其中 σ 是 Sigmoid 函数
    损失: 二元交叉熵 J = -(1/n) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
    梯度: ∂J/∂w = (1/n) X^T (ŷ - y), ∂J/∂b = (1/n) Σ (ŷ_i - y_i)

    注意梯度的简洁性：∂L/∂z = ŷ - y。这正是 Sigmoid + 交叉熵「黄金组合」
    的数学之美——Sigmoid 的导数项在链式法则中被约掉。

    属性:
        w: np.ndarray, 权重向量，形状 (n_features,)
        b: float, 偏置
        loss_history: list, 训练过程中的损失记录
    """

    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 1000,
                 tolerance: float = 1e-6):
        """
        初始化逻辑回归模型。

        参数:
            learning_rate: float, 学习率 η
            max_epochs: int, 最大训练轮数
            tolerance: float, 收敛容忍度
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.w = None  # 权重向量
        self.b = None  # 偏置
        self.loss_history = []  # 损失记录

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        计算样本属于正类的概率 P(y=1|x)。

        ŷ = σ(X @ w + b)

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)

        返回:
            np.ndarray, 形状 (n_samples,)，每个样本属于正类的概率
        """
        z = X @ self.w + self.b  # 线性组合，形状 (n_samples,)
        return sigmoid(z)  # 通过 Sigmoid 得到概率

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        对样本进行类别预测。

        ŷ >= threshold 预测为正类 (1)，否则为负类 (0)。

        参数:
            X: np.ndarray, 形状 (n_samples, n_features)
            threshold: float, 分类阈值，默认 0.5

        返回:
            np.ndarray, 形状 (n_samples,)，预测的类别标签 (0 或 1)
        """
        proba = self._predict_proba(X)  # 计算概率
        return (proba >= threshold).astype(int)  # 转换为 0/1 标签

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算二元交叉熵损失。

        J = -(1/n) Σ [y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i)]

        为了防止 log(0)，对 ŷ 加一个极小值 eps。

        参数:
            X: np.ndarray, 特征矩阵
            y: np.ndarray, 真实标签 (0 或 1)

        返回:
            float, 交叉熵损失
        """
        y_pred = self._predict_proba(X)  # 预测概率
        n = len(y)  # 样本数
        eps = 1e-15  # 小常数，防止 log(0)
        # 限制概率在 [eps, 1-eps] 之间以保证 log 的数值稳定
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # 交叉熵公式
        loss = -(1.0 / n) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        return loss

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        计算损失对 w 和 b 的梯度。

        ∂J/∂w = (1/n) X^T (ŷ - y)
        ∂J/∂b = (1/n) Σ (ŷ_i - y_i)

        这是 Sigmoid + 交叉熵组合的优美结果——梯度等于「预测误差」的加权和。

        参数:
            X: np.ndarray, 特征矩阵 (n_samples, n_features)
            y: np.ndarray, 真实标签 (n_samples,)

        返回:
            dw: np.ndarray, ∂J/∂w, 形状 (n_features,)
            db: float, ∂J/∂b
        """
        y_pred = self._predict_proba(X)  # 预测概率
        n = len(y)  # 样本数
        errors = y_pred - y  # 预测误差 (n_samples,)

        # ∂J/∂w = (1/n) X^T @ errors
        dw = (1.0 / n) * (X.T @ errors)  # 形状 (n_features,)
        # ∂J/∂b = (1/n) Σ errors
        db = (1.0 / n) * np.sum(errors)  # 标量

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        使用梯度下降法训练逻辑回归模型。

        算法步骤:
            1. 随机初始化 w 和 b
            2. 重复:
                a. 前向计算: ŷ = σ(Xw + b)
                b. 计算损失: J = cross_entropy(ŷ, y)
                c. 反向传播: 计算 ∂J/∂w 和 ∂J/∂b
                d. 参数更新: w ← w - η·∂J/∂w, b ← b - η·∂J/∂b
            3. 直到收敛或达到最大轮数

        参数:
            X: np.ndarray, 特征矩阵 (n_samples, n_features)
            y: np.ndarray, 真实标签 (n_samples,)，取值为 0 或 1
            verbose: bool, 是否打印训练日志
        """
        n_samples, n_features = X.shape

        # 初始化参数：使用小随机数，偏置初始化为 0
        self.w = np.random.randn(n_features) * 0.01  # 权重：小随机数
        self.b = 0.0  # 偏置：初始化为 0
        self.loss_history = []

        for epoch in range(self.max_epochs):
            # 前向计算 + 损失
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            # 计算梯度
            dw, db = self._compute_gradients(X, y)

            # 梯度下降更新参数
            self.w -= self.learning_rate * dw  # w ← w - η·∂J/∂w
            self.b -= self.learning_rate * db  # b ← b - η·∂J/∂b

            # 检查收敛
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-2] - loss) < self.tolerance:
                    if verbose:
                        print(f"第 {epoch + 1} 轮收敛！损失变化 < {self.tolerance}")
                    break

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1:4d}: loss={loss:.6f}")

        if verbose:
            print(f"\n训练完成，共 {epoch + 1} 轮，最终损失: {self.loss_history[-1]:.6f}")


# ============================================================================
# 第三部分：多分类逻辑回归（Softmax 回归）
# ============================================================================

class SoftmaxRegression:
    """
    多分类逻辑回归（Softmax 回归）。

    模型: P(y=k|x) = softmax(X @ W + b)_k
    损失: 多分类交叉熵 J = -(1/n) Σ Σ y_{ik} log(ŷ_{ik})
    梯度: ∂J/∂W = (1/n) X^T (ŷ - y_onehot)

    其中 y_onehot 是真实标签的 one-hot 编码。

    属性:
        W: np.ndarray, 权重矩阵，形状 (n_features, n_classes)
        b: np.ndarray, 偏置向量，形状 (n_classes,)
        loss_history: list, 损失记录
    """

    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 2000,
                 tolerance: float = 1e-6):
        """初始化多分类逻辑回归。"""
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.W = None  # 权重矩阵 (n_features, n_classes)
        self.b = None  # 偏置向量 (n_classes,)
        self.loss_history = []
        self.n_classes = None

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        计算每个样本属于每个类别的概率。

        ŷ = softmax(X @ W + b)

        参数:
            X: np.ndarray, (n_samples, n_features)

        返回:
            np.ndarray, (n_samples, n_classes)，每行是一个概率分布
        """
        z = X @ self.W + self.b  # 线性组合，形状 (n_samples, n_classes)
        return softmax(z)  # Softmax 归一化

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别（取概率最大的类别）。

        参数:
            X: np.ndarray, (n_samples, n_features)

        返回:
            np.ndarray, (n_samples,)，类别标签
        """
        proba = self._predict_proba(X)  # 计算概率
        return np.argmax(proba, axis=1)  # 取每行最大概率的索引

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算多分类交叉熵损失。

        J = -(1/n) Σ_i Σ_k y_{ik} log(ŷ_{ik})

        参数:
            X: np.ndarray, 特征矩阵
            y: np.ndarray, 类别标签 (0, 1, ..., K-1)

        返回:
            float, 交叉熵损失
        """
        proba = self._predict_proba(X)  # 预测概率矩阵
        n = len(y)  # 样本数
        eps = 1e-15  # 防止 log(0)
        proba = np.clip(proba, eps, 1 - eps)  # 数值稳定

        # 交叉熵：只取真实类别位置的对数概率
        # proba[np.arange(n), y] 取出每个样本正确类别的预测概率
        loss = -(1.0 / n) * np.sum(np.log(proba[np.arange(n), y]))
        return loss

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        计算多分类交叉熵对 W 和 b 的梯度。

        ∂J/∂W = (1/n) X^T (ŷ - y_onehot)
        ∂J/∂b = (1/n) Σ (ŷ_i - y_onehot_i)

        参数:
            X: np.ndarray, (n_samples, n_features)
            y: np.ndarray, (n_samples,)

        返回:
            dW: np.ndarray, (n_features, n_classes)
            db: np.ndarray, (n_classes,)
        """
        n = len(y)
        proba = self._predict_proba(X)  # (n, K)

        # 构建 one-hot 编码的真实标签
        y_onehot = np.zeros((n, self.n_classes))  # (n, K) 全零矩阵
        y_onehot[np.arange(n), y] = 1  # 在真实类别位置置 1

        errors = proba - y_onehot  # 预测误差矩阵 (n, K)

        dW = (1.0 / n) * (X.T @ errors)  # (d, K)
        db = (1.0 / n) * np.sum(errors, axis=0)  # (K,)

        return dW, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        训练 Softmax 回归模型。

        参数:
            X: np.ndarray, (n_samples, n_features)
            y: np.ndarray, (n_samples,)，类别标签为 0, 1, ..., K-1
            verbose: bool, 是否打印日志
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))  # 类别数

        # 初始化参数
        self.W = np.random.randn(n_features, self.n_classes) * 0.01  # (d, K)
        self.b = np.zeros(self.n_classes)  # (K,)
        self.loss_history = []

        for epoch in range(self.max_epochs):
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            dW, db = self._compute_gradients(X, y)

            self.W -= self.learning_rate * dW  # 矩阵更新
            self.b -= self.learning_rate * db  # 向量更新

            if len(self.loss_history) > 1:
                if abs(self.loss_history[-2] - loss) < self.tolerance:
                    if verbose:
                        print(f"第 {epoch + 1} 轮收敛！")
                    break

            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1:4d}: loss={loss:.6f}")

        if verbose:
            print(f"\nSoftmax 训练完成，共 {epoch + 1} 轮，最终损失: {self.loss_history[-1]:.6f}")


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_sigmoid_curve():
    """绘制 Sigmoid 函数曲线，帮助理解其形态。"""
    z = np.linspace(-8, 8, 500)  # z 从 -8 到 8
    s = sigmoid(z)  # 对应的 σ(z) 值

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, s, 'b-', linewidth=2.5, label=r'$\sigma(z) = 1/(1+e^{-z})$')

    # 关键点标记
    ax.scatter([0], [0.5], c='red', s=100, zorder=5)  # z=0 时 σ=0.5
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)  # 0.5 水平线
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)  # z=0 竖直线
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)  # 渐近线 y=0
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.8)  # 渐近线 y=1

    # 区域标注
    ax.annotate('正类区域\n(z > 0 → σ > 0.5)', xy=(3, 0.95), fontsize=11,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.annotate('负类区域\n(z < 0 → σ < 0.5)', xy=(-3, 0.05), fontsize=11,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax.annotate('决策边界\n(z = 0, σ = 0.5)', xy=(0, 0.5), fontsize=10,
                xytext=(1.5, 0.35), arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', color='red')

    ax.set_xlabel('z = w·x + b', fontsize=13)
    ax.set_ylabel('σ(z)', fontsize=13)
    ax.set_title('Sigmoid 函数：将实数映射到 [0, 1]', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sigmoid_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sigmoid 曲线已保存为 sigmoid_curve.png")


def plot_decision_boundary(model, X, y, title='逻辑回归决策边界'):
    """
    绘制二维特征空间中的决策边界和概率热力图。

    对于每个 (x1, x2) 坐标点，计算模型输出的概率 P(y=1|x)，
    然后用颜色表示概率大小，形成热力图。

    参数:
        model: 训练好的 LogisticRegression 模型
        X: np.ndarray, 形状 (n, 2) 的特征矩阵（仅用前两个特征）
        y: np.ndarray, 真实标签
        title: str, 图表标题
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # 计算网格上每个点的预测概率
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # 展开为 (N, 2)
    Z = model._predict_proba(grid_points)  # 概率值
    Z = Z.reshape(xx.shape)  # 恢复为网格形状

    fig, ax = plt.subplots(figsize=(9, 7))

    # 绘制概率热力图（蓝色=低概率，红色=高概率）
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)

    # 绘制决策边界线（σ = 0.5 的等高线）
    ax.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2.5,
               linestyles='-')

    # 绘制数据点
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o',
               edgecolors='k', s=60, label='负类 (y=0)', alpha=0.8)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='^',
               edgecolors='k', s=60, label='正类 (y=1)', alpha=0.8)

    # 添加颜色条
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('P(y=1|x)', fontsize=11)

    ax.set_xlabel('特征 x1', fontsize=13)
    ax.set_ylabel('特征 x2', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('logistic_regression_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("决策边界图已保存为 logistic_regression_boundary.png")


def plot_loss_curve(loss_history, title='训练损失曲线'):
    """绘制训练过程中的损失变化。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch（训练轮数）', fontsize=12)
    ax.set_ylabel('交叉熵损失', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("损失曲线已保存为 loss_curve.png")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """
    主函数：演示逻辑回归的二分类和多分类完整流程。
    """
    print("=" * 60)
    print("逻辑回归从零实现 — s03_logistic_regression")
    print("=" * 60)

    # ---- 1. Sigmoid 函数可视化 ----
    print("\n[步骤 1] 可视化 Sigmoid 函数...")
    plot_sigmoid_curve()

    # ---- 2. 加载 Iris 数据集 ----
    print("\n[步骤 2] 加载 Iris 数据集...")
    iris = load_iris()
    X_full = iris.data  # 全部 4 个特征
    y_full = iris.target  # 全部 3 个类别
    print(f"数据集: {X_full.shape[0]} 样本, {X_full.shape[1]} 特征, "
          f"{len(np.unique(y_full))} 个类别")
    print(f"类别名称: {iris.target_names}")
    print(f"特征名称: {iris.feature_names}")

    # ---- 3. 二分类 ----
    print("\n" + "=" * 40)
    print("[步骤 3] 二分类逻辑回归（类别 0 vs 类别 1）")
    print("=" * 40)

    # 取前两个类别和前两个特征（便于可视化）
    mask_binary = (y_full == 0) | (y_full == 1)  # 只取类别 0 和 1
    X_binary = X_full[mask_binary][:, :2]  # 取前两个特征用于二维可视化
    y_binary = y_full[mask_binary]
    print(f"二分类数据: {X_binary.shape[0]} 样本, 类别 0: {np.sum(y_binary == 0)}, "
          f"类别 1: {np.sum(y_binary == 1)}")

    # 划分训练/测试集（80/20）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42
    )

    # 训练逻辑回归模型
    model_binary = LogisticRegression(learning_rate=0.5, max_epochs=2000)
    model_binary.fit(X_tr, y_tr)

    # 评估
    y_pred = model_binary.predict(X_te)
    accuracy = np.mean(y_pred == y_te)
    print(f"\n测试集准确率: {accuracy:.2%}")

    # 混淆矩阵
    tp = np.sum((y_pred == 1) & (y_te == 1))  # 真正例
    tn = np.sum((y_pred == 0) & (y_te == 0))  # 真负例
    fp = np.sum((y_pred == 1) & (y_te == 0))  # 假正例
    fn = np.sum((y_pred == 0) & (y_te == 1))  # 假负例
    print(f"混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # 可视化
    plot_decision_boundary(model_binary, X_tr, y_tr,
                           title='逻辑回归二分类 — 决策边界与概率热力图')
    plot_loss_curve(model_binary.loss_history, title='二分类逻辑回归训练损失')

    # ---- 4. 多分类（Softmax 回归） ----
    print("\n" + "=" * 40)
    print("[步骤 4] 多分类 Softmax 回归（全部 3 个类别）")
    print("=" * 40)

    # 取前两个特征用于训练和可视化
    X_multi = X_full[:, :2]
    y_multi = y_full

    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    # 训练 Softmax 回归
    model_softmax = SoftmaxRegression(learning_rate=0.5, max_epochs=5000)
    model_softmax.fit(X_tr_m, y_tr_m)

    # 评估
    y_pred_m = model_softmax.predict(X_te_m)
    accuracy_m = np.mean(y_pred_m == y_te_m)
    print(f"\n测试集准确率: {accuracy_m:.2%}")

    # 可视化多分类决策边界
    x_min, x_max = X_tr_m[:, 0].min() - 0.5, X_tr_m[:, 0].max() + 0.5
    y_min, y_max = X_tr_m[:, 1].min() - 0.5, X_tr_m[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_m = model_softmax.predict(grid_points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z_m, levels=np.arange(-0.5, 3.5, 1),
                colors=['#E8F5E9', '#FFF3E0', '#E3F2FD'], alpha=0.6)
    scatter = ax.scatter(X_tr_m[:, 0], X_tr_m[:, 1], c=y_tr_m,
                         cmap='viridis', edgecolors='k', s=60, alpha=0.8)
    legend = ax.legend(*scatter.legend_elements(), title='类别',
                        fontsize=10, title_fontsize=11)
    ax.set_xlabel('特征 x1 (花萼长度)', fontsize=13)
    ax.set_ylabel('特征 x2 (花萼宽度)', fontsize=13)
    ax.set_title('Softmax 多分类 — 决策区域', fontsize=14)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('softmax_multiclass_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("多分类决策区域图已保存为 softmax_multiclass_boundary.png")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
