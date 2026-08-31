# -*- coding: utf-8 -*-
"""
===============================================================================
as02_pinn/code/demo.py — 物理信息神经网络 (PINN) 求解一维 Poisson 方程
===============================================================================
本演示用 PyTorch 从零训练一个最小的 PINN，求解 as01 中介绍的一维 Poisson
边值问题：

    -u''(x) = f(x),  x in (0, 1)
    u(0) = u(1) = 0
    真解: u_true(x) = sin(pi x)，源项: f(x) = pi^2 sin(pi x)

核心思想: 不给神经网络任何 (x, u) 的标注数据对，而是让它自己"发现"解——
只依靠两类物理约束的损失:
  1. PDE 残差损失: 用自动微分算出 u_theta 的二阶导数，代入方程算残差
  2. 边界条件损失: 让 u_theta 在 x=0, x=1 处的值等于给定边界值 (此处为0)

通过本演示，你将理解：
  - collocation points (配点法): 在定义域内采样一批点，用于评估 PDE 残差
  - autograd 如何被用来计算任意阶的解析导数（无网格、无离散化误差）
  - 复合损失函数 L = L_pde + lambda * L_bc 的构造与权重选择
  - 训练收敛曲线与预测解 vs 真解的对比

作者：notebook 项目
日期：2025
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['axes.unicode_minus'] = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cpu')  # CPU-first: 本问题规模极小，CPU 训练几秒即可完成


def _save_path(filename):
    return os.path.join(_IMAGES_DIR, filename)


# ============================================================================
# 第一部分：问题定义
# ============================================================================

def u_true(x):
    """解析真解: u(x) = sin(pi x)（numpy 版本，用于评估/画图）"""
    return np.sin(np.pi * x)


def f_source(x):
    """源项: f(x) = pi^2 sin(pi x)（torch 版本，用于损失计算，x 为 tensor）"""
    return (np.pi ** 2) * torch.sin(np.pi * x)


# ============================================================================
# 第二部分：PINN 网络结构
# ============================================================================

class PINN(nn.Module):
    """
    一个极简的全连接网络，把坐标 x 映射到解的值 u_hat(x)。

    结构: 1 -> 20 -> 20 -> 20 -> 1，激活函数用 tanh（比 ReLU 更平滑，
    二阶导数不会退化为分段常数/0，这对需要算二阶导数的 PINN 至关重要）。
    """

    def __init__(self, hidden_dim=20, n_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

        # Xavier 初始化，配合 tanh 激活效果较好
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def compute_pde_residual(model, x):
    """
    用自动微分计算 PDE 残差 r(x) = -u_theta''(x) - f(x)。

    关键技巧: torch.autograd.grad 的 create_graph=True 让求导操作本身
    也被记录进计算图，这样我们才能对"导数"再求一次导数（即二阶导数），
    并且这个二阶导数的梯度还能继续反向传播去更新网络参数。

    参数:
        model: PINN 网络
        x: (N, 1) tensor，且必须 requires_grad=True，否则无法求导
    返回:
        residual: (N, 1) tensor，PDE 残差
    """
    u = model(x)
    # 一阶导数 du/dx
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    # 二阶导数 d^2u/dx^2 —— 对一阶导数再求一次导
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]
    residual = -d2u_dx2 - f_source(x)
    return residual


# ============================================================================
# 第三部分：训练流程
# ============================================================================

def train_pinn(n_collocation=50, n_epochs=3000, lr=1e-3, lambda_bc=10.0):
    """
    训练 PINN。

    损失函数: L = L_pde + lambda_bc * L_bc
      L_pde = mean(residual(x_f)^2)，x_f 是内部的 collocation points
      L_bc  = mean(u_hat(x_b)^2)，x_b = {0, 1}，因为真实边界值都是 0

    lambda_bc 权重通常需要比 1 大——因为边界条件只有 2 个点，
    而 PDE 残差有 n_collocation 个点，如果权重相等，边界条件容易"被忽略"。
    """
    model = PINN(hidden_dim=20, n_hidden_layers=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # collocation points: 在 (0, 1) 内均匀采样，每个 epoch 都用同一批点
    # （小规模问题下，固定 collocation points 足够；大规模问题常在每轮重新采样）
    x_f = torch.linspace(0, 1, n_collocation, device=DEVICE).reshape(-1, 1)
    x_f.requires_grad_(True)

    # 边界点: x=0 和 x=1，真实边界值都是 0
    x_b = torch.tensor([[0.0], [1.0]], device=DEVICE)
    u_b_true = torch.tensor([[0.0], [0.0]], device=DEVICE)

    history = {'total': [], 'pde': [], 'bc': []}

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # PDE 残差损失
        residual = compute_pde_residual(model, x_f)
        loss_pde = torch.mean(residual ** 2)

        # 边界条件损失
        u_b_pred = model(x_b)
        loss_bc = torch.mean((u_b_pred - u_b_true) ** 2)

        loss = loss_pde + lambda_bc * loss_bc
        loss.backward()
        optimizer.step()

        history['total'].append(loss.item())
        history['pde'].append(loss_pde.item())
        history['bc'].append(loss_bc.item())

        if (epoch + 1) % 500 == 0:
            print(f"  epoch {epoch + 1:5d}/{n_epochs} | "
                  f"loss_total={loss.item():.3e} | "
                  f"loss_pde={loss_pde.item():.3e} | "
                  f"loss_bc={loss_bc.item():.3e}")

    return model, history


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_solution_comparison(model):
    """对比 PINN 预测解与解析真解，并展示逐点误差。"""
    x_eval = np.linspace(0, 1, 300)
    x_tensor = torch.tensor(x_eval, dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_tensor).numpy().flatten()
    u_exact = u_true(x_eval)
    abs_error = np.abs(u_pred - u_exact)
    l2_rel_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(x_eval, u_exact, 'k-', linewidth=2.5, label='Exact solution sin(pi*x)')
    axes[0].plot(x_eval, u_pred, 'r--', linewidth=2.5, label='PINN prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title(f'PINN vs Exact Solution\nRelative L2 error = {l2_rel_error:.4%}',
                       fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_eval, abs_error, color='tab:purple', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('|prediction - exact|')
    axes[1].set_title('Pointwise Absolute Error', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fp = _save_path('pinn_solution.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] 解对比图已保存: {fp}")
    print(f"  最终相对 L2 误差: {l2_rel_error:.6%}")
    return l2_rel_error


def plot_loss_curves(history):
    """绘制训练过程中三种损失的收敛曲线（对数坐标）。"""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    epochs = np.arange(1, len(history['total']) + 1)
    ax.semilogy(epochs, history['total'], label='Total loss', linewidth=2, color='tab:blue')
    ax.semilogy(epochs, history['pde'], label='PDE residual loss', linewidth=1.8,
                color='tab:orange', alpha=0.85)
    ax.semilogy(epochs, history['bc'], label='Boundary condition loss', linewidth=1.8,
                color='tab:green', alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('PINN Training Loss Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fp = _save_path('pinn_loss.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] 损失曲线图已保存: {fp}")


# ============================================================================
# 第五部分：PINN 架构概念示意图
# ============================================================================

def _draw_box(ax, xy, w, h, text, facecolor, fontsize=9.5):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                          linewidth=1.5, edgecolor='black', facecolor=facecolor, alpha=0.92)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center',
             fontsize=fontsize, fontweight='bold')


def _draw_arrow(ax, start, end, color='black', style='-|>', connectionstyle=None):
    kwargs = dict(arrowstyle=style, mutation_scale=16, linewidth=1.6, color=color)
    if connectionstyle:
        kwargs['connectionstyle'] = connectionstyle
    arrow = FancyArrowPatch(start, end, **kwargs)
    ax.add_patch(arrow)


def plot_pinn_architecture():
    """
    PINN 架构概念示意图：
      输入 x --> MLP(tanh) --> 输出 u_hat(x)
      分支1: 对 u_hat 自动微分两次 --> u', u'' --> 组装 PDE 残差 --> L_pde
      分支2: 边界点 x_b 送入同一网络 --> u_hat(x_b) --> 与真实边界值比较 --> L_bc
      L_total = L_pde + lambda * L_bc --> 反向传播更新网络参数
    """
    fig, ax = plt.subplots(figsize=(12.5, 8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6.25, 7.6, 'PINN Architecture: Solving -u\'\'(x) = f(x) with a Neural Network',
            ha='center', fontsize=14, fontweight='bold')

    # 输入
    _draw_box(ax, (0.3, 5.6), 1.8, 1.0, 'Input\nx (collocation\n+ boundary pts)', '#dbe9f7')
    # 网络
    _draw_box(ax, (2.6, 5.6), 2.6, 1.0, 'MLP\n(1 -> 20 -> 20 -> 20 -> 1)\ntanh activations', '#ffe8b3')
    # 输出
    _draw_box(ax, (5.7, 5.6), 1.8, 1.0, 'Output\nu_hat(x)', '#dbe9f7')

    _draw_arrow(ax, (2.1, 6.1), (2.6, 6.1))
    _draw_arrow(ax, (5.2, 6.1), (5.7, 6.1))

    # 分支1: 自动微分 -> PDE 残差
    _draw_arrow(ax, (6.6, 5.6), (6.6, 4.6))
    _draw_box(ax, (4.6, 3.5), 4.0, 1.0,
              'Autograd twice: du/dx, d2u/dx2\n(torch.autograd.grad, create_graph=True)', '#d9f0d3')
    _draw_arrow(ax, (6.6, 3.5), (6.6, 2.5))
    _draw_box(ax, (4.6, 1.5), 4.0, 1.0,
              'PDE residual r(x) = -u\'\'(x) - f(x)\nL_pde = mean(r(x)^2)', '#f7c7c2')

    # 分支2: 边界条件
    _draw_arrow(ax, (8.5, 6.1), (9.6, 6.1), connectionstyle='arc3,rad=0.0')
    _draw_box(ax, (9.6, 5.6), 2.6, 1.0, 'Boundary points x=0,1\nu_hat(x_b) vs u_true(x_b)=0', '#e6d9f5')
    _draw_arrow(ax, (10.9, 5.6), (10.9, 2.9))
    _draw_box(ax, (9.5, 1.9), 2.8, 1.0, 'L_bc = mean((u_hat(x_b)-0)^2)', '#f7c7c2')

    # 汇总
    _draw_arrow(ax, (6.6, 1.5), (7.5, 0.9))
    _draw_arrow(ax, (10.9, 1.9), (8.5, 0.9))
    _draw_box(ax, (4.3, 0.2), 4.0, 0.9,
              'L_total = L_pde + lambda * L_bc\n--> backprop --> update weights', '#fef1b0')

    plt.tight_layout()
    fp = _save_path('as02-01-pinn-architecture.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Done] PINN 架构示意图已保存: {fp}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("as02_pinn/demo.py — 用 PINN 求解一维 Poisson 方程")
    print("=" * 70)

    print("\n[1/3] 训练 PINN (3000 轮, Adam, lr=1e-3)...")
    model, history = train_pinn(n_collocation=50, n_epochs=3000, lr=1e-3, lambda_bc=10.0)

    print("\n[2/3] 绘制预测解 vs 真解对比图...")
    plot_solution_comparison(model)

    print("\n绘制训练损失曲线...")
    plot_loss_curves(history)

    print("\n[3/3] 绘制 PINN 架构概念示意图...")
    plot_pinn_architecture()

    print("\n" + "=" * 70)
    print("全部完成! 图像保存在:", _IMAGES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
