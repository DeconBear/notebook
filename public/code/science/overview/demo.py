# -*- coding: utf-8 -*-
"""
===============================================================================
as01_ai4s_overview/code/demo.py — AI for Science 全景：PDE 残差直觉演示
===============================================================================
本演示用一个最简单的一维 Poisson 方程，直观展示"什么是 PDE 残差"，
这是理解后续 PINN（as02）、神经算子/FNO（as03）等 AI4S 方法的地基概念。

我们考虑边值问题：
    -u''(x) = f(x),  x in (0, 1)
    u(0) = u(1) = 0

取解析解 u_true(x) = sin(pi x)，则 f(x) = pi^2 sin(pi x)。
这是"制造解"（method of manufactured solutions）的常见技巧：
先假设一个解，反推出对应的右端项 f，这样我们就有了精确的真值可以对照。

本演示展示：
  1. 真解 u_true 满足 PDE（残差处处接近 0，仅有浮点/离散化误差）
  2. 一个"看起来还凑合但不满足物理约束"的错误解 u_wrong 的残差远大于 0
  3. 传统数值方法（有限差分）与"数据驱动学习一个函数"的直觉对比
  4. AI4S 方法全景示意图（算子学习地图）

通过本演示，你将理解：
  - PDE 残差 r(x) = -u''(x) - f(x) 是衡量"一个函数有多不满足物理方程"的尺子
  - 这正是 PINN（as02）用来构造损失函数的核心工具
  - 数据驱动 vs 物理驱动方法的核心差异：是否使用 PDE 残差作为监督信号

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：一维 Poisson 方程与"制造解"
# ============================================================================

def u_true(x):
    """解析真解: u(x) = sin(pi x)，自动满足 u(0)=u(1)=0 的边界条件。"""
    return np.sin(np.pi * x)


def f_source(x):
    """右端源项: 由 -u_true''(x) 反推而来。
    u_true''(x) = -pi^2 sin(pi x)
    => f(x) = -u_true''(x) = pi^2 sin(pi x)
    """
    return (np.pi ** 2) * np.sin(np.pi * x)


def second_derivative_fd(u_vals, dx):
    """
    用二阶中心差分近似二阶导数:
        u''(x_i) ≈ (u_{i-1} - 2*u_i + u_{i+1}) / dx^2

    边界点用同样的公式对称延拓（仅用于残差可视化，不影响内部结论）。
    返回与输入等长的数组，边界处直接置为 0（我们只关心内部残差）。
    """
    n = len(u_vals)
    d2u = np.zeros(n)
    d2u[1:-1] = (u_vals[:-2] - 2 * u_vals[1:-1] + u_vals[2:]) / (dx ** 2)
    return d2u


def pde_residual(u_vals, x_grid):
    """
    计算 PDE 残差: r(x) = -u''(x) - f(x)

    这是本章、也是整个 AI4S 系列最重要的一个量：
    它衡量"候选函数 u(x) 违反物理方程的程度"。
    - 如果 u 恰好是真解，r(x) 应处处（在数值误差范围内）为 0
    - 如果 u 是随便猜的函数，r(x) 会明显偏离 0

    PINN（as02）的核心想法就是：用自动微分算出精确的 u''，
    然后把 mean(r(x)^2) 当作损失函数的一部分去训练神经网络。
    这里我们先用有限差分近似 u''，让你在没有自动微分的情况下也能看到残差的样子。
    """
    dx = x_grid[1] - x_grid[0]
    d2u = second_derivative_fd(u_vals, dx)
    r = -d2u - f_source(x_grid)
    return r


# ============================================================================
# 第二部分：构造一个"错误但看起来合理"的解
# ============================================================================

def u_wrong_linear_bump(x):
    """
    一个满足边界条件 u(0)=u(1)=0、形状也大致像"中间鼓起来"的函数，
    但它并不满足 PDE —— 一个二次函数 4x(1-x) 的曲率是常数，
    而真解 sin(pi x) 的曲率是随位置变化的，两者在方程层面完全不同。

    这模拟了"单纯用边界条件/形状拍脑袋猜解"会发生的错误——
    在数据驱动方法里，如果只拟合边界数据而不显式利用 PDE 约束，
    也可能得到形状相似但物理上错误的解。
    """
    return 4.0 * x * (1.0 - x)


def u_wrong_wrong_amplitude(x):
    """另一个错误解: 振幅和频率都不对的正弦函数 (0.6 * sin(2*pi*x))。"""
    return 0.6 * np.sin(2 * np.pi * x)


# ============================================================================
# 第三部分：可视化 —— PDE 残差对比
# ============================================================================

def plot_pde_residual_comparison():
    """
    核心演示图: 上排画出真解与两个错误解，下排画出对应的 PDE 残差。
    直观展示"形状看起来还行"不等于"满足物理方程"。
    """
    n = 201
    x = np.linspace(0, 1, n)

    u1 = u_true(x)
    u2 = u_wrong_linear_bump(x)
    u3 = u_wrong_wrong_amplitude(x)

    r1 = pde_residual(u1, x)
    r2 = pde_residual(u2, x)
    r3 = pde_residual(u3, x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    solutions = [(u1, 'True solution: sin(pi*x)', 'tab:green'),
                 (u2, 'Wrong solution: 4x(1-x)', 'tab:red'),
                 (u3, 'Wrong solution: 0.6*sin(2*pi*x)', 'tab:orange')]
    residuals = [(r1, 'Residual r(x)=-u\'\'-f\n(true sol.: discretization error only)', 'tab:green'),
                 (r2, 'Residual r(x)=-u\'\'-f\n(wrong curvature -> large residual)', 'tab:red'),
                 (r3, 'Residual r(x)=-u\'\'-f\n(wrong frequency -> large residual)', 'tab:orange')]

    for ax, (u_vals, title, color) in zip(axes[0], solutions):
        ax.plot(x, u_vals, color=color, linewidth=2.5, label='candidate u(x)')
        ax.plot(x, u_true(x), 'k--', linewidth=1.2, alpha=0.5, label='true u(x)')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend(fontsize=8, loc='lower center')
        ax.grid(True, alpha=0.3)

    for ax, (r_vals, title, color) in zip(axes[1], residuals):
        ax.plot(x[2:-2], r_vals[2:-2], color=color, linewidth=2)
        ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('residual r(x)')
        ax.grid(True, alpha=0.3)

    fig.suptitle('PDE residual intuition: "looks similar" != "satisfies the equation"  (-u\'\' = f, u(0)=u(1)=0)',
                  fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    fp = _save_path('ai4s_poisson_residual.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] PDE 残差对比图已保存: {fp}")

    # 打印残差的均方根，量化对比
    print(f"  真解  RMS(residual) = {np.sqrt(np.mean(r1[2:-2] ** 2)):.6f}  (应接近 0，仅数值误差)")
    print(f"  错解1 RMS(residual) = {np.sqrt(np.mean(r2[2:-2] ** 2)):.6f}")
    print(f"  错解2 RMS(residual) = {np.sqrt(np.mean(r3[2:-2] ** 2)):.6f}")


def plot_discretization_cost():
    """
    传统数值方法的痛点示意: 有限差分/有限元的自由度随维度指数增长(维度灾难)，
    而神经网络代理模型的参数量与输入维度增长得更缓和。
    这是一个概念性的示意图（非严格测量），用来建立直觉。
    """
    dims = np.array([1, 2, 3, 4, 5])
    grid_points_per_dim = 50
    # 传统网格法: 自由度 = (每维网格点数)^维度 —— 指数增长
    classical_dof = grid_points_per_dim ** dims.astype(float)
    # 神经网络代理: 参数量随维度近似线性/温和增长（示意性设定）
    nn_params = 5000 * dims + 2000

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(dims, classical_dof, 'o-', color='tab:red', linewidth=2,
                markersize=8, label='Classical grid DoF (~ 50^d)')
    ax.semilogy(dims, nn_params, 's-', color='tab:blue', linewidth=2,
                markersize=8, label='NN surrogate params (illustrative)')
    ax.set_xlabel('Problem dimension d', fontsize=12)
    ax.set_ylabel('Degrees of freedom / params (log scale)', fontsize=12)
    ax.set_title('Curse of dimensionality: why scientific computing needs new tools\n(classical grid methods vs. neural surrogate models)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(dims)

    plt.tight_layout()
    fp = _save_path('ai4s_curse_of_dimensionality.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] 维度灾难示意图已保存: {fp}")


# ============================================================================
# 第四部分：AI4S 全景地图（概念示意图）
# ============================================================================

def _draw_box(ax, xy, w, h, text, facecolor, fontsize=10, textcolor='black'):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.9)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', color=textcolor, wrap=True)


def _draw_arrow(ax, start, end, color='black', style='-|>'):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                             linewidth=1.5, color=color)
    ax.add_patch(arrow)


def plot_ai4s_landscape_map():
    """
    AI4S / 算子学习全景示意图（概念图，非严格框架图）：
      左侧: 物理驱动方法 (数值 PDE 求解)
      右侧: 数据驱动方法 (纯数据拟合)
      中间: 物理+数据混合方法 (PINN, FNO, PINO, GNN 等)
      顶部: DeepMind 代表性 AI4S 成果 (AlphaFold, GraphCast, AlphaChip)
    """
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')

    ax.text(6.5, 8.6, 'AI for Science Landscape: from Physics-Driven to Data-Driven',
            ha='center', fontsize=15, fontweight='bold')

    # 底部光谱轴
    _draw_arrow(ax, (0.5, 6.9), (12.5, 6.9), color='gray', style='-')
    ax.text(0.5, 7.05, 'Physics-Driven', fontsize=10, color='darkred', fontweight='bold')
    ax.text(12.5, 7.05, 'Data-Driven', fontsize=10, color='darkblue',
            fontweight='bold', ha='right')

    # 左端：传统数值方法
    _draw_box(ax, (0.5, 5.6), 2.6, 1.1, 'FDM / FEM\nClassical numerical PDE\nsolvers (explicit mesh)',
              facecolor='#f4b6b6', fontsize=9)
    # 中间偏左：PINN
    _draw_box(ax, (3.6, 5.6), 2.6, 1.1, 'PINN\nNN approximates ONE\nsolution + PDE residual loss',
              facecolor='#f9d9a0', fontsize=9)
    # 中间偏右：神经算子
    _draw_box(ax, (6.7, 5.6), 2.6, 1.1, 'FNO / PINO\nLearn a "function to\nfunction" operator',
              facecolor='#a8d8b9', fontsize=9)
    # 中右：GNN
    _draw_box(ax, (9.8, 5.6), 2.6, 1.1, 'GNN for Science\nIrregular mesh / molecule\n/ weather graph data',
              facecolor='#a3c9e8', fontsize=9)

    for x0 in [1.8, 4.9, 8.0, 11.1]:
        _draw_arrow(ax, (x0, 6.7), (x0, 6.95), color='gray')

    # 说明性子标注
    ax.text(1.8, 5.35, 'Solve pointwise;\nnew params need re-solve', ha='center', fontsize=8, style='italic')
    ax.text(4.9, 5.35, '1 training = 1 solution;\nnew f/BC needs re-train', ha='center', fontsize=8, style='italic')
    ax.text(8.0, 5.35, 'Zero-shot generalizes to\nnew inputs / resolutions', ha='center', fontsize=8, style='italic')
    ax.text(11.1, 5.35, 'Natural fit for graph data\n(molecules/protein/climate)', ha='center', fontsize=8, style='italic')

    # 顶部：DeepMind 代表性成果
    ax.text(6.5, 4.55, 'Representative Google DeepMind AI4S results', ha='center', fontsize=12,
            fontweight='bold')

    _draw_box(ax, (0.8, 2.8), 3.4, 1.35,
              'AlphaFold 2/3\nProtein structure prediction\nGeometric deep learning + Transformer\nOperator learning: sequence -> 3D structure',
              facecolor='#c9b8f0', fontsize=8.5)
    _draw_box(ax, (4.8, 2.8), 3.4, 1.35,
              'GraphCast\nMedium-range weather forecast\nGNN on a mesh graph\nOperator learning: state(t) -> state(t+dt)',
              facecolor='#8fd3c7', fontsize=8.5)
    _draw_box(ax, (8.8, 2.8), 3.4, 1.35,
              'AlphaChip\nChip floorplanning\nRL-based automatic layout\nEDA placement as sequential decision-making',
              facecolor='#f4c2c2', fontsize=8.5)

    for x0 in [2.5, 6.5, 10.5]:
        _draw_arrow(ax, (x0, 4.2), (x0, 2.8 + 1.35), color='gray')

    # 底部：共同主题
    _draw_box(ax, (2.0, 0.5), 9.0, 1.4,
              'Common theme: reformulate a science problem as\n'
              '"learning a mapping G: input (structure/condition/init) -> output (result/prediction/decision)"\n'
              'then approximate G with neural networks + large-scale data / physics constraints',
              facecolor='#fef6d8', fontsize=9.5)
    for y0 in [1.9]:
        pass
    for x0, y0 in [(2.5, 2.8), (6.5, 2.8), (10.5, 2.8)]:
        _draw_arrow(ax, (x0, y0), (x0 - (x0 - 6.5) * 0.3, 1.9), color='lightgray')

    plt.tight_layout()
    fp = _save_path('as01-01-ai4s-map.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] AI4S 全景地图已保存: {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("as01_ai4s_overview/demo.py — AI for Science 全景：PDE 残差直觉")
    print("=" * 70)

    print("\n[1/3] 计算并绘制真解 vs 错解的 PDE 残差对比...")
    plot_pde_residual_comparison()

    print("\n[2/3] 绘制维度灾难示意图...")
    plot_discretization_cost()

    print("\n[3/3] 绘制 AI4S 全景地图（概念示意图）...")
    plot_ai4s_landscape_map()

    print("\n" + "=" * 70)
    print("全部完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
