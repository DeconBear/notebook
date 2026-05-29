# -*- coding: utf-8 -*-
"""
===============================================================================
s04_bias_variance/code/demo.py — 过拟合、正则化与 Bias-Variance 权衡
===============================================================================
本演示通过多项式回归拟合正弦波数据的实验，全面展示：
  1. 欠拟合、拟合良好、过拟合的直观对比
  2. 训练误差 vs 验证误差的 U 形曲线（Bias-Variance 权衡）
  3. L1（Lasso）和 L2（Ridge）正则化的实现与效果
  4. K-Fold 交叉验证的实现
  5. 回归系数随正则化强度的变化

通过本演示，你将理解：
  - 为什么模型复杂度需要与数据量匹配
  - 正则化如何压缩模型参数、防止过拟合
  - L1 和 L2 正则化的不同效果（稀疏 vs 平滑压缩）
  - 交叉验证如何帮助选择最优超参数

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import PolynomialFeatures  # 用于生成多项式特征
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # sklearn 标准实现
from sklearn.model_selection import KFold  # K-Fold 交叉验证
from sklearn.pipeline import make_pipeline  # 构建处理管道
matplotlib.rcParams['axes.unicode_minus'] = False

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

def _save_path(filename):
    """返回本章节 images/ 目录下的图片保存路径。"""
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：数据生成
# ============================================================================

def generate_sine_data(n_samples: int = 80, noise_std: float = 0.3,
                       random_seed: int = 42):
    """
    生成模拟正弦波数据的回归数据集。

    真实函数: f(x) = sin(2πx) + noise
    这是一个非线性函数，用不同次数的多项式去拟合可以看到
    欠拟合和过拟合的现象。

    参数:
        n_samples: int, 样本数量
        noise_std: float, 高斯噪声的标准差
        random_seed: int, 随机种子

    返回:
        X: np.ndarray, (n_samples,)，输入特征 x，范围 [0, 1]
        y: np.ndarray, (n_samples,)，目标值 y = sin(2πx) + noise
    """
    np.random.seed(random_seed)
    X = np.random.uniform(0, 1, n_samples)  # 在 [0, 1] 均匀采样
    X = np.sort(X)  # 排序，便于画曲线（非必须，但让曲线更美观）
    y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * noise_std  # sin(2πx) + 噪声
    return X, y


# ============================================================================
# 第二部分：多项式回归工具
# ============================================================================

def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    手动生成多项式特征矩阵。

    将一维特征 x 扩展为 [1, x, x², x³, ..., x^{degree}]。
    注意：包含常数项（全 1 列），用于拟合偏置。

    例如，如果 degree=3:
        [x] → [1, x, x², x³]

    参数:
        X: np.ndarray, (n,) 或 (n, 1)，原始特征
        degree: int, 多项式最高次数

    返回:
        np.ndarray, (n, degree+1)，多项式特征矩阵
    """
    X = X.reshape(-1, 1)  # 确保 X 是列向量 (n, 1)
    # 使用 np.hstack 堆叠每一列: x^0, x^1, x^2, ..., x^{degree}
    return np.hstack([X ** d for d in range(degree + 1)])


def fit_polynomial(X: np.ndarray, y: np.ndarray, degree: int):
    """
    使用正规方程拟合指定次数的多项式。

    步骤:
        1. 生成多项式特征矩阵 Φ = [1, x, x², ..., x^{degree}]
        2. 使用正规方程求解: θ = (Φ^T Φ)^(-1) Φ^T y

    参数:
        X: np.ndarray, (n,)，输入特征
        y: np.ndarray, (n,)，目标值
        degree: int, 多项式次数

    返回:
        theta: np.ndarray, (degree+1,)，多项式系数 [θ₀, θ₁, ..., θ_{degree}]
    """
    Phi = polynomial_features(X, degree)  # 多项式特征矩阵 (n, d+1)
    # 正规方程: θ = (Φ^T Φ)^(+) Φ^T y，使用伪逆避免奇异矩阵
    theta = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    return theta


def predict_polynomial(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    使用多项式系数进行预测。

    ŷ = θ₀ + θ₁·x + θ₂·x² + ... + θ_d·x^d

    参数:
        X: np.ndarray, (n,)，输入特征
        theta: np.ndarray, (degree+1,)，多项式系数

    返回:
        np.ndarray, (n,)，预测值
    """
    degree = len(theta) - 1  # 从系数数量推断多项式次数
    Phi = polynomial_features(X, degree)  # 构建多项式特征矩阵
    return Phi @ theta  # 矩阵乘法得到预测值


# ============================================================================
# 第三部分：自定义正则化线性回归
# ============================================================================

class RegularizedLinearRegression:
    """
    带 L1 和 L2 正则化的线性回归（使用梯度下降）。

    在标准 MSE 损失基础上增加了正则化项:
      L2 (Ridge): J = MSE + λ * Σ wⱼ²
      L1 (Lasso): J = MSE + λ * Σ |wⱼ|

    其中 λ (lambda_) 是正则化强度。

    注意：偏置项通常不参与正则化，因为正则化的目标是约束
    模型复杂度（特征系数），而不是偏置。

    属性:
        lambda_: float, 正则化强度
        reg_type: str, 正则化类型 ('l2', 'l1', 'none')
        learning_rate: float, 学习率
        max_epochs: int, 最大训练轮数
        w: np.ndarray, 权重向量（包含偏置）
        loss_history: list, 损失记录
    """

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 5000,
                 lambda_: float = 0.0, reg_type: str = 'l2'):
        """
        初始化正则化线性回归模型。

        参数:
            learning_rate: float, 学习率 η
            max_epochs: int, 最大训练轮数
            lambda_: float, 正则化强度 λ（0 表示无正则化）
            reg_type: str, 正则化类型：'l2'（Ridge）、'l1'（Lasso）、'none'（无正则化）
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.lambda_ = lambda_
        self.reg_type = reg_type
        self.w = None  # 权重向量（包含偏置 b = w[0]）
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        """
        使用梯度下降训练带正则化的线性回归。

        对于多项式回归，X 已经是多项式特征矩阵 Φ（含常数项列）。

        梯度计算（包含正则化项）:
          - MSE 部分: dw_mse = (2/n) * Φ^T @ (Φw - y)
          - L2 梯度:  dw_l2 = 2 * lambda_ * w（偏置项不加: dw_l2[0] = 0）
          - L1 梯度:  dw_l1 = lambda_ * sign(w)（偏置项不加: dw_l1[0] = 0）

        参数:
            X: np.ndarray, (n, d)，特征矩阵（包含常数项）
            y: np.ndarray, (n,)，目标值
            verbose: bool, 是否打印训练日志
        """
        n_samples, n_features = X.shape

        # 初始化权重为小随机数
        self.w = np.random.randn(n_features) * 0.01
        self.loss_history = []

        for epoch in range(self.max_epochs):
            # 计算预测值和误差
            y_pred = X @ self.w  # 前向计算
            errors = y_pred - y  # 预测误差

            # MSE 损失部分
            mse_loss = np.mean(errors ** 2)

            # 正则化损失
            # 偏置项（索引 0）不参与正则化
            weights_no_bias = self.w[1:]  # 排除偏置项
            if self.reg_type == 'l2':
                reg_loss = self.lambda_ * np.sum(weights_no_bias ** 2)
            elif self.reg_type == 'l1':
                reg_loss = self.lambda_ * np.sum(np.abs(weights_no_bias))
            else:
                reg_loss = 0.0

            total_loss = mse_loss + reg_loss
            self.loss_history.append(total_loss)

            # MSE 梯度
            dw_mse = (2.0 / n_samples) * (X.T @ errors)

            # 正则化梯度（偏置项不参与正则化）
            dw_reg = np.zeros(n_features)
            if self.reg_type == 'l2':
                # ∂(λ·Σwⱼ²)/∂wⱼ = 2λ·wⱼ
                dw_reg[1:] = 2.0 * self.lambda_ * weights_no_bias
            elif self.reg_type == 'l1':
                # ∂(λ·Σ|wⱼ|)/∂wⱼ = λ·sign(wⱼ)
                dw_reg[1:] = self.lambda_ * np.sign(weights_no_bias)

            # 总梯度 = MSE 梯度 + 正则化梯度
            dw = dw_mse + dw_reg

            # 梯度下降更新
            self.w -= self.learning_rate * dw

            # 收敛检查
            if len(self.loss_history) > 1 and epoch > 100:
                if abs(self.loss_history[-2] - total_loss) < 1e-8:
                    if verbose:
                        print(f"  第 {epoch+1} 轮收敛")
                    break

        if verbose:
            # 统计有多少权重接近 0（用于 L1 的稀疏性分析）
            n_zeros = np.sum(np.abs(self.w[1:]) < 1e-4)
            print(f"  训练完成 (lambda={self.lambda_}, {self.reg_type}): "
                  f"loss={self.loss_history[-1]:.4f}, "
                  f"非零权重数={len(self.w) - 1 - n_zeros}/{len(self.w) - 1}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的参数进行预测。"""
        return X @ self.w


# ============================================================================
# 第四部分：K-Fold 交叉验证
# ============================================================================

def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int = 5,
                           degree: int = 3, lambda_: float = 0.0,
                           reg_type: str = 'none'):
    """
    执行 K-Fold 交叉验证。

    将数据分成 K 折，每次用 K-1 折训练，1 折验证，
    计算平均验证 MSE。

    参数:
        X: np.ndarray, (n,)，原始特征
        y: np.ndarray, (n,)，目标值
        k: int, 折数
        degree: int, 多项式次数
        lambda_: float, 正则化强度
        reg_type: str, 正则化类型

    返回:
        avg_val_mse: float, K 次验证 MSE 的平均值
        val_mses: list, 每次验证的 MSE 列表
    """
    n = len(X)
    fold_size = n // k  # 每折的大小
    val_mses = []  # 记录每次验证的 MSE

    for i in range(k):
        # 确定验证集的起止索引
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n

        # 划分训练集和验证集
        val_idx = np.arange(val_start, val_end)  # 验证集索引
        train_idx = np.setdiff1d(np.arange(n), val_idx)  # 训练集索引（所有其它索引）

        X_train, y_train = X[train_idx], y[train_idx]  # 训练数据
        X_val, y_val = X[val_idx], y[val_idx]  # 验证数据

        # 生成多项式特征
        Phi_train = polynomial_features(X_train, degree)
        Phi_val = polynomial_features(X_val, degree)

        # 训练模型
        if reg_type == 'none' or lambda_ == 0.0:
            # 无正则化：使用正规方程（伪逆，避免奇异矩阵）
            theta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train
            y_pred = Phi_val @ theta
        else:
            # 带正则化：使用梯度下降
            model = RegularizedLinearRegression(
                learning_rate=0.1, max_epochs=5000,
                lambda_=lambda_, reg_type=reg_type
            )
            model.fit(Phi_train, y_train)
            y_pred = model.predict(Phi_val)

        # 计算验证 MSE
        val_mse = np.mean((y_pred - y_val) ** 2)
        val_mses.append(val_mse)

    return np.mean(val_mses), val_mses


# ============================================================================
# 第五部分：可视化
# ============================================================================

def plot_polynomial_fits(X_train, y_train, X_true, y_true, degrees, max_degree=15):
    """
    可视化不同次数多项式的拟合效果。

    绘制一个 3 行 5 列的网格，展示从 1 次到 max_degree 次多项式
    的拟合结果。同时标出训练/验证 MSE。

    参数:
        X_train: np.ndarray, 训练数据特征
        y_train: np.ndarray, 训练数据目标值
        X_true: np.ndarray, 测试数据特征（密集采样，用于画平滑曲线）
        y_true: np.ndarray, 测试数据目标值（无噪声的真实值）
        degrees: list, 要展示的多项式次数列表
        max_degree: int, 最大多项式次数
    """
    n_degrees = len(degrees)
    n_rows = (n_degrees + 4) // 5  # 向上取整的行数
    n_cols = min(n_degrees, 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()  # 将 axes 展平为一维数组

    for idx, deg in enumerate(degrees):
        ax = axes[idx]

        # 拟合多项式
        theta = fit_polynomial(X_train, y_train, deg)

        # 预测密集点（平滑曲线）
        Phi_true = polynomial_features(X_true, deg)
        y_pred_true = Phi_true @ theta

        # 计算训练 MSE
        Phi_train = polynomial_features(X_train, deg)
        y_pred_train = Phi_train @ theta
        train_mse = np.mean((y_pred_train - y_train) ** 2)

        # 计算测试 MSE（用无噪声的真实函数值）
        test_mse = np.mean((y_pred_true - y_true) ** 2)

        # 绘制
        ax.scatter(X_train, y_train, c='steelblue', s=15, alpha=0.6,
                   edgecolors='white', linewidth=0.3, label='Training Data')
        ax.plot(X_true, y_pred_true, 'r-', linewidth=1.5, label=f'deg={deg}')

        # 判断拟合质量并设置标题颜色
        if train_mse > 0.15:  # 欠拟合
            quality = 'Underfitting'
            color = 'red'
        elif deg > 12 and test_mse > 3 * train_mse:  # 过拟合
            quality = 'Overfitting'
            color = 'orange'
        else:
            quality = 'Good Fit'
            color = 'green'

        ax.set_title(f'deg={deg}: Train={train_mse:.3f}, Test={test_mse:.3f} ({quality})',
                     fontsize=8, color=color)
        ax.grid(True, alpha=0.2)

    # 隐藏多余的子图
    for idx in range(len(degrees), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Polynomial Fit Comparison at Different Degrees', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(_save_path('polynomial_fits_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"多项式拟合对比图已保存为 {_save_path('polynomial_fits_comparison.png')}")


def plot_bias_variance_curve(X_train, y_train, X_val, y_val, max_degree=15):
    """
    绘制训练误差 vs 验证误差随模型复杂度的变化（Bias-Variance U 形曲线）。

    这展示了经典的 Bias-Variance 权衡：
    - 训练误差随模型复杂度增加而单调递减
    - 验证误差先降后升，呈 U 形

    参数:
        X_train: np.ndarray, 训练数据特征
        y_train: np.ndarray, 训练数据目标值
        X_val: np.ndarray, 验证数据特征
        y_val: np.ndarray, 验证数据目标值
        max_degree: int, 最大多项式次数
    """
    degrees = range(1, max_degree + 1)
    train_errors = []  # 记录每个度数的训练误差
    val_errors = []  # 记录每个度数的验证误差

    for deg in degrees:
        # 生成多项式特征
        Phi_train = polynomial_features(X_train, deg)
        Phi_val = polynomial_features(X_val, deg)

        # 拟合模型（伪逆，避免奇异矩阵）
        theta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train

        # 计算训练和验证误差
        y_pred_train = Phi_train @ theta
        y_pred_val = Phi_val @ theta
        train_errors.append(np.mean((y_pred_train - y_train) ** 2))
        val_errors.append(np.mean((y_pred_val - y_val) ** 2))

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(degrees, train_errors, 'b-o', markersize=6, linewidth=1.5,
            label='Training Error')
    ax.plot(degrees, val_errors, 'r-s', markersize=6, linewidth=1.5,
            label='Validation Error')

    # 标注最佳模型复杂度（验证误差最小的点）
    best_deg = degrees[np.argmin(val_errors)]
    best_val_error = min(val_errors)
    ax.axvline(x=best_deg, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.annotate(f'Best Complexity: deg={best_deg}\nValidation Error={best_val_error:.3f}',
                xy=(best_deg, best_val_error),
                xytext=(best_deg + 2, best_val_error + 0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 标注区域
    ax.axvspan(1, best_deg - 1, alpha=0.1, color='orange')
    ax.text(2, max(train_errors) * 0.95, 'Underfitting Region', fontsize=11,
            color='orange', ha='center')
    ax.axvspan(best_deg + 1, max_degree, alpha=0.1, color='red')
    ax.text(max_degree - 1, max(val_errors) * 0.95, 'Overfitting Region', fontsize=11,
            color='red', ha='center')

    ax.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=13)
    ax.set_ylabel('MSE Error', fontsize=13)
    ax.set_title('Bias-Variance Trade-off: Training Error vs Validation Error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # 使用对数刻度使小值差异更明显
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(_save_path('bias_variance_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Bias-Variance 曲线已保存为 {_save_path('bias_variance_curve.png')}")


def plot_regularization_effect(X_train, y_train, X_val, y_val, degree=15):
    """
    比较不同正则化方法的效果。

    对 15 次多项式（严重过拟合），比较：
      - 无正则化（严重过拟合）
      - L2 正则化（Ridge）
      - L1 正则化（Lasso）

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        degree: int, 多项式次数（高次以展示过拟合）
    """
    # 生成多项式特征
    Phi_train = polynomial_features(X_train, degree)
    Phi_val = polynomial_features(X_val, degree)

    # 密集采样点用于画平滑曲线
    X_dense = np.linspace(0, 1, 500)
    Phi_dense = polynomial_features(X_dense, degree)

    # 配置四种情况
    configs = [
        ('No Regularization', 'none', 0.0, 'gray'),
        ('L2 (Ridge)', 'l2', 0.01, 'blue'),
        ('L1 (Lasso)', 'l1', 0.01, 'green'),
        ('L2 (Ridge, λ=0.1)', 'l2', 0.1, 'red'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (label, reg_type, lam, color) in enumerate(configs):
        ax = axes[idx]

        if reg_type == 'none':
            # 无正则化：用正规方程直接解（伪逆，避免奇异矩阵）
            theta = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train
            y_pred_train = Phi_train @ theta
            y_pred_val = Phi_val @ theta
            train_mse = np.mean((y_pred_train - y_train) ** 2)
            val_mse = np.mean((y_pred_val - y_val) ** 2)
        else:
            # 有正则化：用梯度下降
            model = RegularizedLinearRegression(
                learning_rate=0.1, max_epochs=10000,
                lambda_=lam, reg_type=reg_type
            )
            model.fit(Phi_train, y_train)
            theta = model.w
            y_pred_train = model.predict(Phi_train)
            y_pred_val = model.predict(Phi_val)
            train_mse = np.mean((y_pred_train - y_train) ** 2)
            val_mse = np.mean((y_pred_val - y_val) ** 2)

        # 预测密集曲线
        y_pred_dense = Phi_dense @ theta

        # 绘制
        ax.scatter(X_train, y_train, c='steelblue', s=20, alpha=0.5,
                   edgecolors='white', linewidth=0.3)
        ax.plot(X_dense, y_pred_dense, color=color, linewidth=2, label=f'{label}')
        ax.plot(X_dense, np.sin(2 * np.pi * X_dense), 'k--', linewidth=1,
                alpha=0.5, label='True Function sin(2*pi*x)')

        ax.set_title(f'{label}: Train MSE={train_mse:.3f}, Val MSE={val_mse:.3f}',
                     fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'Effect of Regularization on Degree-{degree} Polynomial Fit', fontsize=14)
    plt.tight_layout()
    plt.savefig(_save_path('regularization_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"正则化对比图已保存为 {_save_path('regularization_comparison.png')}")


def plot_coefficient_paths(X_train, y_train, degree=15):
    """
    展示回归系数如何随正则化强度 λ 的变化而变化。

    横轴是 λ（对数刻度），纵轴是各系数的值。
    可以观察到 L1 正则化如何将系数推向精确的零（稀疏性）。

    参数:
        X_train, y_train: 训练数据
        degree: int, 多项式次数
    """
    Phi_train = polynomial_features(X_train, degree)
    lambdas = np.logspace(-4, 2, 50)  # 从 10^-4 到 10^2 对数均匀采样

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, reg_type in enumerate(['l2', 'l1']):
        ax = axes[ax_idx]
        coef_paths = []  # 记录每个 λ 下的系数向量

        for lam in lambdas:
            model = RegularizedLinearRegression(
                learning_rate=0.1, max_epochs=10000,
                lambda_=lam, reg_type=reg_type
            )
            model.fit(Phi_train, y_train)
            coef_paths.append(model.w[1:])  # 排除偏置项

        coef_paths = np.array(coef_paths)  # (50, degree)

        # 绘制每条系数路径
        for d in range(degree):
            ax.plot(lambdas, coef_paths[:, d], linewidth=1.5,
                    alpha=0.7, label=f'w{d+1}' if d < 5 else '')

        ax.set_xscale('log')  # λ 轴使用对数刻度
        ax.set_xlabel('Regularization Strength lambda', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title(
            f'{"L2 (Ridge)" if reg_type == "l2" else "L1 (Lasso)"} - '
            f'Coefficient vs lambda', fontsize=13
        )
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)

        if reg_type == 'l2':
            ax.legend(fontsize=8, loc='upper right', ncol=2)

    plt.suptitle('Regularization Path: How Coefficients Shrink as lambda Increases', fontsize=14)
    plt.tight_layout()
    plt.savefig(_save_path('coefficient_paths.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"系数路径图已保存为 {_save_path('coefficient_paths.png')}")


def plot_cv_results(X_train, y_train, max_degree=15):
    """
    使用 K-Fold 交叉验证选择最优多项式次数。

    对每个次数，计算 K-Fold CV 的平均验证误差，然后选择最佳次数。

    参数:
        X_train, y_train: 训练数据
        max_degree: int, 最大多项式次数
    """
    degrees = range(1, max_degree + 1)
    cv_means = []  # 每个度数的平均 CV 误差
    cv_stds = []  # 每个度数的 CV 误差标准差

    print("\nK-Fold 交叉验证进行中...")
    for deg in degrees:
        avg_mse, val_mses = kfold_cross_validation(
            X_train, y_train, k=5, degree=deg
        )
        cv_means.append(avg_mse)
        cv_stds.append(np.std(val_mses))
        if deg % 3 == 0:  # 每 3 个打印一次进度
            print(f"  次数 {deg:2d}: CV 平均误差 = {avg_mse:.4f} ± {cv_stds[-1]:.4f}")

    # 找到最优多项式次数
    best_deg = degrees[np.argmin(cv_means)]
    print(f"\n✓ 最优多项式次数: {best_deg} (CV 误差 = {min(cv_means):.4f})")

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制均值和标准差阴影
    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)
    ax.plot(degrees, cv_means, 'b-o', markersize=6, linewidth=1.5,
            label='5-Fold CV Mean Error')
    ax.fill_between(degrees, cv_means - cv_stds, cv_means + cv_stds,
                     alpha=0.2, color='blue', label='±1 Std Dev')

    # 标注最优次数
    ax.axvline(x=best_deg, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.scatter([best_deg], [min(cv_means)], c='red', s=150, zorder=5, marker='*',
               label=f'Best: deg={best_deg}')

    ax.set_xlabel('Polynomial Degree', fontsize=13)
    ax.set_ylabel('Cross-Validation MSE', fontsize=13)
    ax.set_title('K-Fold Cross-Validation for Model Complexity Selection', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_save_path('cross_validation_selection.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"交叉验证选择图已保存为 {_save_path('cross_validation_selection.png')}")


# ============================================================================
# 第六部分：主程序
# ============================================================================

def main():
    """
    主函数：串联过拟合/正则化/Bias-Variance 的完整教学演示。
    """
    print("=" * 60)
    print("过拟合、正则化与 Bias-Variance 权衡 — s04_bias_variance")
    print("=" * 60)

    # 1. 生成数据
    print("\n[步骤 1] 生成正弦波数据 y = sin(2πx) + noise...")
    X, y = generate_sine_data(n_samples=80, noise_std=0.3, random_seed=42)

    # 划分训练集和验证集
    n_train = int(0.7 * len(X))  # 70% 训练
    X_train, y_train = X[:n_train], y[:n_train]  # 前 70% 训练
    X_val, y_val = X[n_train:], y[n_train:]  # 后 30% 验证

    # 密集采样点（用于画平滑曲线和无噪声的真实函数值）
    X_dense = np.linspace(0, 1, 500)
    y_dense = np.sin(2 * np.pi * X_dense)
    print(f"训练集: {len(X_train)} 样本，验证集: {len(X_val)} 样本")

    # 2. 可视化不同次数的多项式拟合
    print("\n[步骤 2] 可视化不同次数多项式拟合...")
    degrees_to_show = list(range(1, 16))
    plot_polynomial_fits(X_train, y_train, X_dense, y_dense, degrees_to_show)

    # 3. 绘制训练误差 vs 验证误差的 U 形曲线
    print("\n[步骤 3] 绘制 Bias-Variance 权衡曲线...")
    plot_bias_variance_curve(X_train, y_train, X_val, y_val, max_degree=15)

    # 4. 展示正则化的效果
    print("\n[步骤 4] 展示正则化对过拟合的抑制效果...")
    plot_regularization_effect(X_train, y_train, X_val, y_val, degree=15)

    # 5. 展示正则化系数路径
    print("\n[步骤 5] 展示正则化系数路径...")
    plot_coefficient_paths(X_train, y_train, degree=15)

    # 6. K-Fold 交叉验证
    print("\n[步骤 6] 使用 K-Fold 交叉验证选择最优模型...")
    plot_cv_results(X_train, y_train, max_degree=15)

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
