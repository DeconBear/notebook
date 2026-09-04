# -*- coding: utf-8 -*-
"""
s08 优化器：从 SGD 到 Adam — 演示代码
======================================
功能：在二维损失地形上可视化对比 SGD、Momentum、RMSProp、Adam 的优化轨迹。
      包括损失曲线、超参数游乐场（可调学习率和 β 值）。

运行方式：在 s08_optimizers_sgd_to_adam/ 目录下执行 python code/demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyBboxPatch
from typing import Tuple, List, Dict, Callable
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第一部分：定义损失函数（狭长峡谷形 2D 二次型）
# ============================================================================

class LossLandscape:
    """
    二维二次型损失函数：L(θ₁, θ₂) = 0.5 * (a·θ₁² + b·θ₂²)

    当 a >> b 时，形成狭长峡谷地形：
      - θ₁ 方向（大系数 a）：陡峭方向，梯度大
      - θ₂ 方向（小系数 b）：平缓方向，梯度小

    参数:
        a: θ₁ 方向的曲率（默认 20.0，陡峭）
        b: θ₂ 方向的曲率（默认 1.0，平缓）
    """

    def __init__(self, a: float = 20.0, b: float = 1.0):
        self.a = a  # 陡峭方向曲率
        self.b = b  # 平缓方向曲率

    def __call__(self, theta: np.ndarray) -> float:
        """
        计算损失值。

        参数:
            theta: 参数向量，shape (2,)

        返回:
            损失值（标量）
        """
        theta1, theta2 = theta[0], theta[1]
        return 0.5 * (self.a * theta1 ** 2 + self.b * theta2 ** 2)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        计算梯度 ∇L(θ)。

        参数:
            theta: 参数向量，shape (2,)

        返回:
            梯度向量，shape (2,): [a·θ₁, b·θ₂]
        """
        theta1, theta2 = theta[0], theta[1]
        return np.array([self.a * theta1, self.b * theta2])

    @property
    def optimum(self) -> np.ndarray:
        """返回全局最优点 θ* = (0, 0)，最小损失值为 0"""
        return np.array([0.0, 0.0])


# ============================================================================
# 第二部分：优化器实现
# ============================================================================

class SGDOptimizer:
    """
    朴素 SGD 优化器。

    更新公式: θ_{t+1} = θ_t - α · g_t

    参数:
        lr: 学习率 α
    """

    def __init__(self, lr: float = 0.02):
        self.lr = lr
        self.name = "SGD"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 SGD 更新。

        参数:
            theta: 当前参数
            grad: 当前梯度

        返回:
            更新后的参数
        """
        return theta - self.lr * grad  # θ := θ - α·∇L


class MomentumOptimizer:
    """
    Momentum 优化器（带动量的 SGD）。

    公式:
        m_t = β · m_{t-1} + (1-β) · g_t
        θ_{t+1} = θ_t - α · m_t

    参数:
        lr: 学习率 α
        beta: 动量衰减系数 β（默认 0.9）
    """

    def __init__(self, lr: float = 0.02, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.m = None  # 速度向量，初始化为 None，第一次 step 时初始化
        self.name = "Momentum"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 Momentum 更新。

        参数:
            theta: 当前参数
            grad: 当前梯度

        返回:
            更新后的参数
        """
        # 首次调用时，初始化动量向量为零
        if self.m is None:
            self.m = np.zeros_like(theta)  # m_0 = 0

        # m_t = β · m_{t-1} + (1-β) · g_t
        self.m = self.beta * self.m + (1 - self.beta) * grad
        # θ_{t+1} = θ_t - α · m_t
        return theta - self.lr * self.m


class RMSPropOptimizer:
    """
    RMSProp 优化器。

    公式:
        v_t = β · v_{t-1} + (1-β) · g_t²
        θ_{t+1} = θ_t - α · g_t / (√v_t + ε)

    参数:
        lr: 学习率 α
        beta: 衰减系数 β（默认 0.9）——注意 RMSProp 通常也用 0.9
        eps: 数值稳定常数 ε
    """

    def __init__(self, lr: float = 0.02, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = None  # 梯度平方的滑动平均，第一次 step 时初始化
        self.name = "RMSProp"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 RMSProp 更新。

        参数:
            theta: 当前参数
            grad: 当前梯度

        返回:
            更新后的参数
        """
        if self.v is None:
            self.v = np.zeros_like(theta)  # v_0 = 0

        # v_t = β · v_{t-1} + (1-β) · g_t²
        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)
        # θ_{t+1} = θ_t - α · g_t / (√v_t + ε)
        return theta - self.lr * grad / (np.sqrt(self.v) + self.eps)


class AdamOptimizer:
    """
    Adam 优化器（带偏差修正）。

    公式:
        m_t = β₁ · m_{t-1} + (1-β₁) · g_t          (一阶矩/方向)
        v_t = β₂ · v_{t-1} + (1-β₂) · g_t²         (二阶矩/尺度)
        m̂_t = m_t / (1 - β₁^t)                      (一阶矩偏差修正)
        v̂_t = v_t / (1 - β₂^t)                      (二阶矩偏差修正)
        θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)       (参数更新)

    参数:
        lr: 学习率 α（默认 0.1）
        beta1: 一阶矩衰减率（默认 0.9）
        beta2: 二阶矩衰减率（默认 0.999）
        eps: 数值稳定常数（默认 1e-8）
    """

    def __init__(self, lr: float = 0.1, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None    # 一阶矩向量
        self.v = None    # 二阶矩向量
        self.t = 0       # 迭代步数计数器
        self.name = "Adam"

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一步 Adam 更新（含偏差修正）。

        参数:
            theta: 当前参数
            grad: 当前梯度

        返回:
            更新后的参数
        """
        if self.m is None:
            self.m = np.zeros_like(theta)  # m_0 = 0
            self.v = np.zeros_like(theta)  # v_0 = 0

        self.t += 1  # 迭代步数 +1

        # ---- step 1: 更新一阶矩（动量） ----
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # ---- step 2: 更新二阶矩（梯度平方的均值） ----
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        # ---- step 3: 偏差修正 ----
        m_hat = self.m / (1 - self.beta1 ** self.t)  # 修正一阶矩
        v_hat = self.v / (1 - self.beta2 ** self.t)  # 修正二阶矩

        # ---- step 4: 参数更新 ----
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================================
# 第三部分：训练与轨迹记录
# ============================================================================

def run_optimizer(
    optimizer,
    landscape: LossLandscape,
    theta_init: np.ndarray,
    n_steps: int = 100,
    add_noise: bool = False,
    noise_std: float = 0.0
) -> Tuple[np.ndarray, List[float]]:
    """
    在给定的损失地形上运行一个优化器，记录完整轨迹。

    参数:
        optimizer: 优化器对象（SGD / Momentum / RMSProp / Adam）
        landscape: 损失函数对象
        theta_init: 初始参数位置
        n_steps: 迭代步数
        add_noise: 是否在梯度中添加噪声（模拟 mini-batch 噪声）
        noise_std: 噪声标准差

    返回:
        trajectory: 每一步的参数位置，shape (n_steps+1, 2)
        losses: 每一步的损失值列表
    """
    theta = theta_init.copy()  # 当前参数
    trajectory = [theta.copy()]  # 记录初始位置
    losses = [landscape(theta)]  # 记录初始损失

    for _ in range(n_steps):
        grad = landscape.gradient(theta)  # 计算当前梯度

        # 添加噪声（模拟 mini-batch SGD 的梯度噪声）
        if add_noise:
            grad = grad + np.random.randn(*grad.shape) * noise_std

        theta = optimizer.step(theta, grad)  # 优化器更新一步
        trajectory.append(theta.copy())  # 记录位置
        losses.append(landscape(theta))  # 记录损失

    return np.array(trajectory), losses


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_contour_comparison(
    landscape: LossLandscape,
    all_trajectories: Dict[str, np.ndarray],
    filename: str = "optimizer_trajectories.png"
):
    """
    在同一张等高线图上绘制所有优化器的轨迹，对比收敛行为。

    参数:
        landscape: 损失函数
        all_trajectories: {优化器名称: 轨迹数组} 的字典
        filename: 保存的文件名
    """
    # 创建等高线网格
    x_range = np.linspace(-4, 4, 200)
    y_range = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 0.5 * (landscape.a * X ** 2 + landscape.b * Y ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 绘制损失等高线（对数刻度以更好地显示峡谷结构）
    levels = np.logspace(-2, 2, 15)  # 对数间隔的等高线
    contour = ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.6, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # 绘制填充等高线背景
    ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.15)

    # 标记最优点
    optimum = landscape.optimum
    ax.plot(optimum[0], optimum[1], 'r*', markersize=15, label='θ* (Optimum)', zorder=10)

    # 绘制每条优化器轨迹
    colors = {'SGD': '#E74C3C', 'Momentum': '#2ECC71',
              'RMSProp': '#3498DB', 'Adam': '#9B59B6'}
    markers = {'SGD': 'o', 'Momentum': 's', 'RMSProp': '^', 'Adam': 'D'}

    for name, traj in all_trajectories.items():
        color = colors.get(name, 'gray')
        marker = markers.get(name, 'o')
        # 绘制轨迹线
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2,
                alpha=0.8, label=f'{name}')
        # 标注起点
        ax.plot(traj[0, 0], traj[0, 1], marker=marker, color=color,
                markersize=10, markeredgecolor='white', markeredgewidth=1.5)
        # 标注终点
        ax.plot(traj[-1, 0], traj[-1, 1], marker=marker, color=color,
                markersize=12, markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlabel('θ₁ (Flat Direction)', fontsize=13)
    ax.set_ylabel('θ₂ (Steep Direction)', fontsize=13)
    ax.set_title(f'Optimizer Trajectory Comparison\nL(theta) = 0.5*({landscape.a}*theta1^2 + {landscape.b}*theta2^2)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # 添加文字注释
    ax.text(-3.5, 3.5, f'Condition Number κ = {landscape.a/landscape.b:.0f}',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 优化器轨迹对比图已保存至 {out}")


def plot_loss_curves(
    all_losses: Dict[str, List[float]],
    filename: str = "loss_curves.png"
):
    """
    绘制各优化器的损失-迭代步数曲线对比。

    参数:
        all_losses: {优化器名称: 损失列表} 的字典
        filename: 保存的文件名
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'SGD': '#E74C3C', 'Momentum': '#2ECC71',
              'RMSProp': '#3498DB', 'Adam': '#9B59B6'}

    for name, losses in all_losses.items():
        color = colors.get(name, 'gray')
        ax.plot(losses, '-', color=color, linewidth=2, alpha=0.8, label=name)

    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Loss L(θ)', fontsize=13)
    ax.set_title('Loss Curve Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # 对数刻度
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 标注最终损失值
    y_max = ax.get_ylim()[1]
    for i, (name, losses) in enumerate(all_losses.items()):
        final_loss = losses[-1]
        ax.annotate(f'{name}: {final_loss:.2e}',
                    xy=(len(losses) - 1, final_loss),
                    xytext=(len(losses) - 1 - 30, y_max * (0.5 ** i * 0.8)),
                    fontsize=9, color=colors.get(name, 'gray'),
                    arrowprops=dict(arrowstyle='->', color=colors.get(name, 'gray')))

    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 损失曲线对比图已保存至 {out}")


def plot_hyperparameter_playground(
    landscape: LossLandscape,
    theta_init: np.ndarray,
    filename: str = "hyperparameter_playground.png"
):
    """
    超参数游乐场：展示不同学习率下各优化器的表现。

    对比不同学习率 (0.01, 0.05, 0.1, 0.5) 对四种优化器的影响。

    参数:
        landscape: 损失函数
        theta_init: 初始参数
        filename: 保存的文件名
    """
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    colors = {'SGD': '#E74C3C', 'Momentum': '#2ECC71',
              'RMSProp': '#3498DB', 'Adam': '#9B59B6'}

    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]

        # 创建等高线背景
        x_range = np.linspace(-4, 4, 150)
        y_range = np.linspace(-4, 4, 150)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 0.5 * (landscape.a * X ** 2 + landscape.b * Y ** 2)
        levels = np.logspace(-2, 2, 12)
        ax.contour(X, Y, Z, levels=levels, cmap='Blues', alpha=0.4, linewidths=0.5)
        ax.contourf(X, Y, Z, levels=levels, cmap='Blues', alpha=0.08)

        # 标记最优点
        ax.plot(0, 0, 'r*', markersize=12, zorder=10)

        # 对每个优化器使用当前学习率
        n_steps = 80
        optimizers = [
            SGDOptimizer(lr=lr),
            MomentumOptimizer(lr=lr),
            RMSPropOptimizer(lr=lr),
            AdamOptimizer(lr=lr),
        ]

        for opt in optimizers:
            name = opt.name
            # 重置优化器状态并运行
            opt.m = None
            opt.v = None
            if hasattr(opt, 't'):
                opt.t = 0

            traj, _ = run_optimizer(opt, landscape, theta_init, n_steps)
            color = colors.get(name, 'gray')
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.8,
                    alpha=0.8, label=name)
            ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6)
            ax.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=1)

        ax.set_title(f'Learning Rate α = {lr}', fontsize=13, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Hyperparameter Playground: Effect of Learning Rate on Optimizers',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(_IMAGES, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 超参数游乐场已保存至 {out}")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   s08 优化器：从 SGD 到 Adam — 损失地形上的轨迹对比             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ---- 1. 创建损失地形 ----
    # 条件数 κ = a/b = 20，形成狭长峡谷
    landscape = LossLandscape(a=20.0, b=1.0)
    theta_init = np.array([3.0, 2.5])  # 初始位置在右上角
    n_steps = 150  # 迭代步数

    print(f"\n[损失地形] L(θ) = 0.5·({landscape.a}·θ₁² + {landscape.b}·θ₂²)")
    print(f"  条件数 κ = {landscape.a/landscape.b:.0f} (狭长峡谷)")
    print(f"  初始位置 θ₀ = {theta_init}")
    print(f"  迭代步数: {n_steps}")

    # ---- 2. 创建优化器 ----
    optimizers = [
        SGDOptimizer(lr=0.02),
        MomentumOptimizer(lr=0.02, beta=0.9),
        RMSPropOptimizer(lr=0.05, beta=0.9),
        AdamOptimizer(lr=0.1, beta1=0.9, beta2=0.999),
    ]

    # ---- 3. 运行优化并记录轨迹 ----
    print("\n[训练] 运行各优化器...")
    all_trajectories = {}
    all_losses = {}

    for opt in optimizers:
        traj, losses = run_optimizer(opt, landscape, theta_init, n_steps)
        all_trajectories[opt.name] = traj
        all_losses[opt.name] = losses

        final_loss = losses[-1]
        final_dist = np.linalg.norm(traj[-1] - landscape.optimum)
        print(f"  {opt.name:<10}: 最终损失={final_loss:.2e}, "
              f"距最优解={final_dist:.4f}")

    # ---- 4. 打印对比总结表 ----
    print("\n" + "=" * 70)
    print("【优化器对比总结】")
    print("=" * 70)
    print(f"{'优化器':<12} {'最终损失':<16} {'距最优解':<14} {'记忆量'}")
    print("-" * 70)
    for opt in optimizers:
        final_loss = all_losses[opt.name][-1]
        final_dist = np.linalg.norm(all_trajectories[opt.name][-1] - landscape.optimum)
        memory = "0" if isinstance(opt, SGDOptimizer) else \
                 "m_t" if isinstance(opt, MomentumOptimizer) else \
                 "v_t" if isinstance(opt, RMSPropOptimizer) else \
                 "m_t, v_t"
        print(f"{opt.name:<12} {final_loss:<16.6e} {final_dist:<14.6f} {memory}")

    # 底部收敛步数（以损失 < 0.001 为标准）
    print(f"\n{'达到 L<0.001 所需步数:':<20}")
    for opt in optimizers:
        losses_arr = np.array(all_losses[opt.name])
        steps = np.argmax(losses_arr < 0.001) if np.any(losses_arr < 0.001) else "未达到"
        print(f"  {opt.name:<10}: {steps}")
    print("=" * 70)

    # ---- 5. 可视化 ----
    print("\n[可视化] 生成图表...")

    # 5a. 轨迹对比图
    plot_contour_comparison(landscape, all_trajectories)

    # 5b. 损失曲线
    plot_loss_curves(all_losses)

    # 5c. 超参数游乐场
    plot_hyperparameter_playground(landscape, theta_init)

    # ---- 6. 带噪声的 SGD 演示 ----
    print("\n[额外演示] Mini-batch 噪声对 SGD 的影响...")
    noisy_sgd = SGDOptimizer(lr=0.02)
    noisy_adam = AdamOptimizer(lr=0.1)

    traj_sgd_noisy, loss_sgd_noisy = run_optimizer(
        noisy_sgd, landscape, theta_init, n_steps=100,
        add_noise=True, noise_std=1.0
    )
    traj_adam_noisy, loss_adam_noisy = run_optimizer(
        noisy_adam, landscape, theta_init, n_steps=100,
        add_noise=True, noise_std=1.0
    )

    print(f"  SGD+噪声:   最终损失={loss_sgd_noisy[-1]:.4f}")
    print(f"  Adam+噪声:   最终损失={loss_adam_noisy[-1]:.4f}")
    print(f"  Adam 对梯度噪声的鲁棒性远优于 SGD！")

    # 噪声对比图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(loss_sgd_noisy, 'r-', linewidth=1.5, alpha=0.7, label='SGD + Noise')
    ax.plot(loss_adam_noisy, 'b-', linewidth=1.5, alpha=0.7, label='Adam + Noise')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Robustness Comparison Under Gradient Noise (sigma=1.0)', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(_IMAGES, 'noise_robustness.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 噪声鲁棒性对比图已保存至 {out}")

    # ---- 7. 总结 ----
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print("  ✓ 在狭长峡谷形损失地形上对比了 4 种优化器")
    print("  ✓ SGD 沿陡峭方向震荡、平缓方向进展慢 → 锯齿路径")
    print("  ✓ Momentum 通过惯性平滑方向 → 路径更直")
    print("  ✓ RMSProp 自适应步长 → 陡峭方向步长变小")
    print("  ✓ Adam 结合两者 → 方向平滑 + 步长自适应 == 最快收敛")
    print("  ✓ Adam 对梯度噪声（mini-batch 噪声）的鲁棒性远优于 SGD")
    print("=" * 70)


if __name__ == "__main__":
    main()
