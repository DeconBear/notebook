# -*- coding: utf-8 -*-
"""
===============================================================================
s02_linear_regression/code/demo.py — 线性回归从零实现
===============================================================================
本演示从零实现线性回归，涵盖梯度下降法和正规方程两种求解方式，
并可视化数据、拟合直线、损失曲线等内容。

通过本演示，你将理解：
  1. 线性模型 ŷ = wx + b 的数学形式和参数含义
  2. MSE 损失函数及其梯度的推导和计算
  3. 梯度下降法：如何沿着梯度方向一步步逼近最优解
  4. 正规方程：线性回归的封闭形式解析解
  5. 学习率对收敛速度的影响
  6. 与 sklearn 标准实现的对比验证

作者：learn-ai 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # 用于绘制 3D 损失曲面
from sklearn.linear_model import LinearRegression as SklearnLR  # sklearn 的线性回归
matplotlib.rcParams['axes.unicode_minus'] = False

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================================
# 第一部分：生成合成数据
# ============================================================================

def generate_regression_data(n_samples: int = 100, noise_std: float = 3.0,
                             true_w: float = 2.0, true_b: float = 5.0,
                             random_seed: int = 42):
    """
    生成线性回归的合成数据。

    按照 y = true_w * x + true_b + noise 生成数据，
    其中 noise 是服从 N(0, noise_std²) 的高斯噪声。

    参数:
        n_samples: int, 样本数量
        noise_std: float, 噪声的标准差（越大数据越散）
        true_w: float, 真实的斜率
        true_b: float, 真实的截距
        random_seed: int, 随机种子

    返回:
        X: np.ndarray, 形状 (n_samples,) 的特征向量
        y: np.ndarray, 形状 (n_samples,) 的目标值向量
    """
    np.random.seed(random_seed)  # 固定随机种子，保证可复现

    # 在 [0, 10] 范围内均匀采样特征 x
    X = np.random.uniform(low=0.0, high=10.0, size=n_samples)

    # 生成噪声：从 N(0, noise_std²) 中采样
    noise = np.random.randn(n_samples) * noise_std

    # 生成目标值：y = 2x + 5 + noise
    y = true_w * X + true_b + noise

    return X, y


# ============================================================================
# 第二部分：线性回归模型（梯度下降法）
# ============================================================================

class LinearRegressionGD:
    """
    使用梯度下降法求解的线性回归模型。

    模型假设: ŷ = w * x + b
    损失函数: J(w,b) = (1/n) * Σ (ŷ_i - y_i)²  (MSE)
    优化方法: 批量梯度下降

    梯度推导:
        ∂J/∂w = (2/n) * Σ (ŷ_i - y_i) * x_i
        ∂J/∂b = (2/n) * Σ (ŷ_i - y_i)

    属性:
        w: float, 权重（斜率）
        b: float, 偏置（截距）
        loss_history: list, 每个 epoch 的损失值（用于绘制损失曲线）
        params_history: list, 每个 epoch 的 (w, b) 值（用于绘制优化轨迹）
    """

    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000,
                 tolerance: float = 1e-6):
        """
        初始化线性回归模型。

        参数:
            learning_rate: float, 学习率 η，控制参数更新的步长
            max_epochs: int, 最大训练轮数
            tolerance: float, 收敛容忍度——当损失变化小于此值时提前停止
        """
        self.learning_rate = learning_rate  # 学习率 η
        self.max_epochs = max_epochs  # 最大迭代次数
        self.tolerance = tolerance  # 收敛判定阈值
        self.w = None  # 权重（斜率），将在 fit 中初始化
        self.b = None  # 偏置（截距），将在 fit 中初始化
        self.loss_history = []  # 记录每轮的损失值
        self.params_history = []  # 记录每轮的参数值

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入 X 进行预测。

        预测公式: ŷ = w * X + b

        参数:
            X: np.ndarray, 形状 (n_samples,) 的特征

        返回:
            np.ndarray, 形状 (n_samples,) 的预测值
        """
        return self.w * X + self.b  # 向量化运算，直接对所有样本预测

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算 MSE 损失。

        J(w,b) = (1/n) * Σ (ŷ_i - y_i)²

        参数:
            X: np.ndarray, 特征
            y: np.ndarray, 真实目标值

        返回:
            float, 当前参数下的 MSE 损失值
        """
        y_pred = self.predict(X)  # 计算预测值
        n = len(y)  # 样本数
        return np.mean((y_pred - y) ** 2)  # MSE = 平均平方误差

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        计算 MSE 损失对 w 和 b 的梯度。

        ∂J/∂w = (2/n) * Σ (ŷ_i - y_i) * x_i
        ∂J/∂b = (2/n) * Σ (ŷ_i - y_i)

        参数:
            X: np.ndarray, 特征
            y: np.ndarray, 真实目标值

        返回:
            dw: float, 损失对 w 的偏导数
            db: float, 损失对 b 的偏导数
        """
        y_pred = self.predict(X)  # 预测值
        n = len(y)  # 样本数
        errors = y_pred - y  # 预测误差向量

        # ∂J/∂w = (2/n) * Σ (ŷ_i - y_i) * x_i
        dw = (2.0 / n) * np.sum(errors * X)

        # ∂J/∂b = (2/n) * Σ (ŷ_i - y_i)
        db = (2.0 / n) * np.sum(errors)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        使用梯度下降法训练线性回归模型。

        算法流程:
            1. 随机初始化 w 和 b
            2. 重复以下步骤直到收敛或达到最大轮数:
                a. 计算当前预测 ŷ = w*x + b
                b. 计算损失 J = MSE(ŷ, y)
                c. 计算梯度 ∂J/∂w 和 ∂J/∂b
                d. 更新参数: w = w - η * ∂J/∂w, b = b - η * ∂J/∂b

        参数:
            X: np.ndarray, 特征
            y: np.ndarray, 真实目标值
            verbose: bool, 是否打印训练日志
        """
        # 用正态分布小随机数初始化 w 和 b，使初始参数接近 0 但不全为 0
        self.w = np.random.randn() * 0.1  # 权重初始化为小随机数
        self.b = np.random.randn() * 0.1  # 偏置初始化为小随机数
        self.loss_history = []  # 清空损失历史
        self.params_history = []  # 清空参数历史

        for epoch in range(self.max_epochs):
            # 步骤 1: 计算当前损失
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)  # 记录损失
            self.params_history.append((self.w, self.b))  # 记录当前参数

            # 步骤 2: 计算梯度
            dw, db = self._compute_gradients(X, y)

            # 步骤 3: 使用梯度下降更新参数
            self.w -= self.learning_rate * dw  # w ← w - η * ∂J/∂w
            self.b -= self.learning_rate * db  # b ← b - η * ∂J/∂b

            # 步骤 4: 检查收敛条件
            if len(self.loss_history) > 1:
                loss_change = abs(self.loss_history[-2] - loss)  # 当前损失与上一轮的差值
                if loss_change < self.tolerance:  # 损失变化很小，认为已收敛
                    if verbose:
                        print(f"第 {epoch + 1} 轮收敛！损失变化 {loss_change:.8f} < {self.tolerance}")
                    break

            # 每 50 轮打印一次训练进度
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1:4d}: loss={loss:.6f}, w={self.w:.4f}, b={self.b:.4f}")

        if verbose:
            print(f"\n训练完成，共 {epoch + 1} 轮")
            print(f"学到的参数: w = {self.w:.4f}, b = {self.b:.4f}")
            print(f"最终损失: {self.loss_history[-1]:.6f}")


# ============================================================================
# 第三部分：正规方程求解
# ============================================================================

def normal_equation_solution(X: np.ndarray, y: np.ndarray):
    """
    使用正规方程直接求解线性回归的解析解。

    正规方程: θ* = (X^T X)^(-1) X^T y

    其中 X 是包含偏置列的特征矩阵（X 被添加了一列全 1），
    θ 是包含 [w, b] 的参数向量。

    解析解的推导过程:
        J(θ) = (1/n) * ||Xθ - y||²
        令 ∂J/∂θ = 0
        => X^T X θ = X^T y
        => θ = (X^T X)^(-1) X^T y

    参数:
        X: np.ndarray, 形状 (n_samples,) 的特征向量
        y: np.ndarray, 形状 (n_samples,) 的目标值向量

    返回:
        w: float, 权重（斜率）
        b: float, 偏置（截距）
    """
    n = len(X)
    # 构建增广特征矩阵 X_aug = [x, 1]，形状 (n, 2)
    # 第一列是特征 x，第二列是全 1（用于计算偏置）
    X_aug = np.column_stack([X, np.ones(n)])

    # 使用正规方程求解: θ = (X^T X)^(-1) X^T y
    # @ 是 Python 3.5+ 的矩阵乘法运算符，等价于 np.matmul()
    theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y

    w = theta[0]  # 权重（斜率）
    b = theta[1]  # 偏置（截距）
    return w, b


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_results(X, y, model_gd, w_ne, b_ne, w_sk, b_sk):
    """
    全面可视化线性回归的结果。

    生成一个包含 2 行 2 列的复合图，展示:
      (1) 数据散点与拟合直线
      (2) 训练损失曲线
      (3) 3D 损失曲面上的梯度下降轨迹
      (4) 不同方法的参数对比

    参数:
        X: np.ndarray, 特征
        y: np.ndarray, 目标值
        model_gd: LinearRegressionGD, 梯度下降训练好的模型
        w_ne, b_ne: float, 正规方程解出的参数
        w_sk, b_sk: float, sklearn 解出的参数
    """
    fig = plt.figure(figsize=(16, 12))

    # ---- 子图 1: 数据散点和拟合直线 ----
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(X, y, c='steelblue', alpha=0.7, s=40, label='Training Data',
                edgecolors='white', linewidth=0.5)

    # 生成一条平滑的 x 序列用于画拟合直线
    X_line = np.linspace(X.min(), X.max(), 200)
    # 梯度下降法拟合的直线（红色）
    ax1.plot(X_line, model_gd.predict(X_line), 'r-', linewidth=2,
             label=f'Gradient Descent: y = {model_gd.w:.2f}x + {model_gd.b:.2f}')
    # 正规方程拟合的直线（绿色虚线）
    ax1.plot(X_line, w_ne * X_line + b_ne, 'g--', linewidth=2,
             label=f'Normal Equation: y = {w_ne:.2f}x + {b_ne:.2f}')
    # sklearn 拟合的直线（蓝色点划线）
    ax1.plot(X_line, w_sk * X_line + b_sk, 'b-.', linewidth=1.5, alpha=0.6,
             label=f'sklearn: y = {w_sk:.2f}x + {b_sk:.2f}')

    ax1.set_xlabel('Feature x', fontsize=12)
    ax1.set_ylabel('Target y', fontsize=12)
    ax1.set_title('Linear Regression: Data and Fitted Lines', fontsize=14)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ---- 子图 2: 训练损失曲线 ----
    ax2 = fig.add_subplot(2, 2, 2)
    epochs = range(1, len(model_gd.loss_history) + 1)
    ax2.plot(epochs, model_gd.loss_history, 'b-', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_title('Loss During Gradient Descent Training', fontsize=14)
    ax2.grid(True, alpha=0.3)
    # 用对数刻度显示 y 轴（因为损失在早期下降很快，后期趋于平稳）
    ax2.set_yscale('log')
    ax2.annotate(f'Initial Loss: {model_gd.loss_history[0]:.2f}',
                 xy=(1, model_gd.loss_history[0]),
                 fontsize=9, color='red')
    ax2.annotate(f'Final Loss: {model_gd.loss_history[-1]:.3f}',
                 xy=(len(epochs), model_gd.loss_history[-1]),
                 fontsize=9, color='green')

    # ---- 子图 3: 损失函数等高线图与优化轨迹 ----
    ax3 = fig.add_subplot(2, 2, 3)

    # 在 (w, b) 平面上计算损失函数的网格值
    w_range = np.linspace(model_gd.w - 1.5, model_gd.w + 1.5, 100)
    b_range = np.linspace(model_gd.b - 3.0, model_gd.b + 3.0, 100)
    W_grid, B_grid = np.meshgrid(w_range, b_range)  # 创建网格

    # 对每个 (w, b) 网格点计算损失
    Z_grid = np.zeros_like(W_grid)
    for i in range(len(b_range)):
        for j in range(len(w_range)):
            w_val = W_grid[i, j]
            b_val = B_grid[i, j]
            y_pred = w_val * X + b_val  # 用当前参数预测
            Z_grid[i, j] = np.mean((y_pred - y) ** 2)  # 计算 MSE

    # 绘制损失函数等高线
    contour = ax3.contour(W_grid, B_grid, Z_grid, levels=20, cmap='viridis', alpha=0.7)
    ax3.clabel(contour, inline=True, fontsize=7)  # 在等高线上标注数值

    # 绘制梯度下降的优化轨迹
    params_arr = np.array(model_gd.params_history)
    ax3.plot(params_arr[:, 0], params_arr[:, 1], 'r.-', markersize=2, linewidth=1,
             label='GD Trajectory')

    # 标记起点和终点
    ax3.scatter(params_arr[0, 0], params_arr[0, 1], c='blue', s=100, marker='o',
                zorder=5, label=f'Start (w={params_arr[0,0]:.2f}, b={params_arr[0,1]:.2f})')
    ax3.scatter(params_arr[-1, 0], params_arr[-1, 1], c='red', s=100, marker='*',
                zorder=5, label=f'End (w={params_arr[-1,0]:.2f}, b={params_arr[-1,1]:.2f})')

    ax3.set_xlabel('Weight w', fontsize=12)
    ax3.set_ylabel('Bias b', fontsize=12)
    ax3.set_title('Loss Contour and Gradient Descent Trajectory', fontsize=14)
    ax3.legend(fontsize=8, loc='upper right')

    # ---- 子图 4: 方法对比条形图 ----
    ax4 = fig.add_subplot(2, 2, 4)
    methods = ['Gradient Descent', 'Normal Equation', 'sklearn']
    w_values = [model_gd.w, w_ne, w_sk]
    b_values = [model_gd.b, b_ne, b_sk]
    x_pos = np.arange(len(methods))
    width = 0.35

    bars1 = ax4.bar(x_pos - width/2, w_values, width, label='Weight w', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x_pos + width/2, b_values, width, label='Bias b', color='coral', alpha=0.8)

    # 在每个柱状图上方标注数值
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods, fontsize=11)
    ax4.set_ylabel('Parameter Value', fontsize=12)
    ax4.set_title('Parameter Comparison Across Three Methods', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='True w=2.0')
    ax4.axhline(y=5.0, color='gray', linestyle=':', alpha=0.5, label='True b=5.0')

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES_DIR, 'linear_regression_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n图片已保存为 {os.path.join(_IMAGES_DIR, 'linear_regression_results.png')}")


def compare_learning_rates(X, y):
    """
    比较不同学习率对梯度下降收敛的影响。

    分别用 3 种不同的学习率训练模型，并在同一张图上对比它们的损失曲线。

    参数:
        X: np.ndarray, 特征
        y: np.ndarray, 目标值
    """
    rates = [0.001, 0.01, 0.05]  # 三种学习率
    colors = ['blue', 'green', 'red']  # 对应的颜色

    fig, ax = plt.subplots(figsize=(10, 5))

    for lr, color in zip(rates, colors):
        model = LinearRegressionGD(learning_rate=lr, max_epochs=200)
        model.fit(X, y, verbose=False)
        epochs = range(1, len(model.loss_history) + 1)
        ax.plot(epochs, model.loss_history, color=color, linewidth=1.5,
                label=f'lr={lr} (loss={model.loss_history[-1]:.2e})')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Effect of Learning Rate on Convergence Speed', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES_DIR, 'learning_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图片已保存为 {os.path.join(_IMAGES_DIR, 'learning_rate_comparison.png')}")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """
    主函数：串联整个线性回归的教学演示流程。
    """
    print("=" * 60)
    print("线性回归从零实现 — s02_linear_regression")
    print("=" * 60)

    # 1. 生成合成数据（真实参数: w=2, b=5）
    print("\n[步骤 1] 生成合成数据 (y = 2x + 5 + noise)...")
    X, y = generate_regression_data(n_samples=100, true_w=2.0, true_b=5.0,
                                     noise_std=3.0, random_seed=42)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"X 范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y 范围: [{y.min():.2f}, {y.max():.2f}]")

    # 2. 使用梯度下降法训练
    print("\n[步骤 2] 使用梯度下降法训练线性回归模型...")
    model_gd = LinearRegressionGD(learning_rate=0.01, max_epochs=500)
    model_gd.fit(X, y, verbose=True)

    # 3. 使用正规方程求解
    print("\n[步骤 3] 使用正规方程求解...")
    w_ne, b_ne = normal_equation_solution(X, y)
    print(f"正规方程解: w = {w_ne:.4f}, b = {b_ne:.4f}")

    # 4. 使用 sklearn 求解（对比验证）
    print("\n[步骤 4] 使用 sklearn 求解（作为基准）...")
    X_sk = X.reshape(-1, 1)  # sklearn 要求特征为二维: (n_samples, n_features)
    model_sk = SklearnLR()
    model_sk.fit(X_sk, y)
    w_sk = model_sk.coef_[0]  # sklearn 的权重在 coef_ 属性中
    b_sk = model_sk.intercept_  # sklearn 的偏置在 intercept_ 属性中
    print(f"sklearn 解: w = {w_sk:.4f}, b = {b_sk:.4f}")

    # 5. 评估模型
    print("\n[步骤 5] 评估模型...")
    y_pred = model_gd.predict(X)  # 用梯度下降模型预测
    mse = np.mean((y_pred - y) ** 2)  # 计算 MSE
    # R² 分数: 1 - SS_res / SS_tot，越接近 1 表示模型解释力越强
    ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
    r2 = 1 - ss_res / ss_tot
    print(f"MSE = {mse:.4f}")
    print(f"R²  = {r2:.4f}")

    # 6. 综合可视化
    print("\n[步骤 6] 综合可视化...")
    plot_results(X, y, model_gd, w_ne, b_ne, w_sk, b_sk)

    # 7. 不同学习率对比
    print("\n[步骤 7] 比较不同学习率的效果...")
    compare_learning_rates(X, y)

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
