# -*- coding: utf-8 -*-
"""
===============================================================================
as08_ai4s_frontier/code/demo.py — AI4S 综合与前沿：可微代理模型驱动的逆向设计
===============================================================================
本章是"进阶一：AI for Science"系列(as01-as08)的总结篇。demo 分三部分：

  1. AI4S 发展时间线：把 as01-as07 涉及的关键方法/论文放在时间轴上
  2. 多方法综合对比雷达图：PINN / FNO-PINO / GNN / AlphaFold式方法 / AlphaChip式方法
     在数据需求、推理速度、泛化能力、物理一致性、可解释性五个维度上的直觉对比
  3. 【前沿主题实操】可微代理模型驱动的逆向设计（differentiable inverse design）：
     这是当前 AI4S 最活跃的前沿方向之一——一旦你有了一个可微的代理模型
     （如 as03/as04 训练出的算子网络），就可以直接用梯度下降对"设计参数"
     反向优化，而不需要黑盒搜索。本节复现一个简化的 as04 变系数扩散问题，
     对比两种"逆向设计"策略：
       a) 黑盒网格搜索：枚举参数 a，找到使输出最接近目标的一个
       b) 可微代理模型 + 梯度下降：把训练好的代理模型当作"可微分的物理模拟器"，
          直接对参数 a 做梯度下降，找到让输出匹配目标的 a
     梯度方法通常能用远少于网格搜索的函数评估次数收敛到更精确的解——
     这正是"可微科学计算"范式相比传统黑盒优化/实验设计的核心优势。

运行方式：cd docs/science/frontier/code && python demo.py
依赖：numpy, torch, matplotlib
===============================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def _save_path(name):
    return os.path.join(_IMAGES_DIR, name)


# ============================================================================
# 第一部分：AI4S 发展时间线
# ============================================================================

TIMELINE = [
    (2018, 'AlphaFold 1', 'CASP13夺冠，距离图+梯度下降'),
    (2019, 'PINN (Raissi et al.)', 'PDE残差作为损失函数(as02)'),
    (2020, 'AlphaFold 2', 'CASP14中位GDT>90，Evoformer+IPA(as06)'),
    (2020, 'AlphaChip前身', '图+强化学习做芯片布局(as07)'),
    (2021, 'FNO (Li et al.)', '傅立叶神经算子，分辨率无关(as03)'),
    (2021, 'PINO', '物理约束+算子学习结合(as04)'),
    (2022, 'GraphCast', 'GNN驱动的中期天气预报'),
    (2024, 'AlphaFold 3', '扩散模型扩展到复合物结构'),
    (2024, 'AI4S基础模型探索', '大规模预训练科学基础模型、LLM+工具做科研助理'),
]


def plot_timeline(save_name='as08-01-ai4s-timeline.png'):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    years = [t[0] for t in TIMELINE]
    ax.axhline(0, color='gray', linewidth=1.5, zorder=1)
    ax.set_xlim(2017.3, 2025.2)
    ax.set_ylim(-1.6, 1.9)
    ax.axis('off')
    ax.set_title('AI for Science 发展时间线（本系列 as01-as08 涉及的代表性节点）',
                 fontsize=13.5, fontweight='bold')

    for idx, (year, name, note) in enumerate(TIMELINE):
        up = idx % 2 == 0
        y_text = 0.55 if up else -0.55
        y_line_top = 0.15 if up else -0.15
        color = '#2E86AB' if up else '#C0392B'
        ax.plot([year, year], [0, y_line_top], color=color, linewidth=1.5, zorder=2)
        circ = Circle((year, 0), 0.05, facecolor=color, edgecolor='black', zorder=3)
        ax.add_patch(circ)
        va = 'bottom' if up else 'top'
        ax.text(year, y_text, f'{year}\n{name}', ha='center', va=va, fontsize=9.5,
                fontweight='bold', color=color)
        y_note = y_text + (0.35 if up else -0.35)
        ax.text(year, y_note, note, ha='center', va=va, fontsize=7.8, color='#444444', wrap=True)

    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] AI4S 时间线已保存至 images/{save_name}')


# ============================================================================
# 第二部分：多方法综合对比雷达图
# ============================================================================

RADAR_DIMS = ['数据需求(低=好)', '推理速度', '泛化到新参数', '物理一致性', '可解释性']
RADAR_SCORES = {
    'PINN (as02)':          [5, 2, 1, 5, 4],
    'FNO/PINO (as03-04)':   [3, 5, 4, 3, 3],
    'GNN (as05)':           [2, 4, 3, 2, 3],
    'AlphaFold式 (as06)':   [1, 4, 3, 2, 4],
    'AlphaChip式 (as07)':   [2, 4, 3, 2, 2],
}
RADAR_COLORS = ['#2E86AB', '#27AE60', '#F39C12', '#9B59B6', '#C0392B']


def plot_method_radar(save_name='as08-02-method-radar.png'):
    dims = RADAR_DIMS
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for (name, scores), color in zip(RADAR_SCORES.items(), RADAR_COLORS):
        values = scores + scores[:1]
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.06, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5)
    ax.set_title('本系列五类方法的多维度直觉对比（主观定性打分，非严格评测）',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] 方法对比雷达图已保存至 images/{save_name}')


# ============================================================================
# 第三部分：可微代理模型驱动的逆向设计
# ============================================================================

N_GRID = 41
X_GRID = np.linspace(0.0, 1.0, N_GRID)
DX = X_GRID[1] - X_GRID[0]
X_TORCH = torch.tensor(X_GRID, dtype=torch.float32)


def diffusivity_field(a, x):
    return 1.0 + a * np.sin(np.pi * x)


def solve_reference(a):
    """有限体积法数值求解（与 as04 相同的问题，此处独立复现，避免跨章节依赖）。"""
    k = diffusivity_field(a, X_GRID)
    k_half = 0.5 * (k[:-1] + k[1:])
    n_inner = N_GRID - 2
    A = np.zeros((n_inner, n_inner))
    b = np.zeros(n_inner)
    f = np.sin(np.pi * X_GRID)
    for i in range(1, N_GRID - 1):
        idx = i - 1
        b[idx] = f[i]
        A[idx, idx] = (k_half[i - 1] + k_half[i]) / DX ** 2
        if idx - 1 >= 0:
            A[idx, idx - 1] = -k_half[i - 1] / DX ** 2
        if idx + 1 <= n_inner - 1:
            A[idx, idx + 1] = -k_half[i] / DX ** 2
    u_inner = np.linalg.solve(A, b)
    u = np.zeros(N_GRID)
    u[1:-1] = u_inner
    return u


class TinySurrogate(nn.Module):
    """
    一个极简代理模型：输入标量参数 a，直接回归出整条解曲线 u(x)（长度 N_GRID）。
    这不是严格的算子网络（如 as03/as04 的 FNO），而是最简单的"参数->曲线"回归器，
    足以支撑本节演示"对代理模型做梯度下降"这个核心思想。
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, N_GRID),
        )

    def forward(self, a):
        return self.net(a)


def train_surrogate(n_epochs=2000, lr=2e-3):
    a_train = np.linspace(0.5, 3.5, 25)
    u_train = np.stack([solve_reference(a) for a in a_train]).astype(np.float32)
    a_train_t = torch.tensor(a_train, dtype=torch.float32).view(-1, 1)
    u_train_t = torch.tensor(u_train, dtype=torch.float32)

    model = TinySurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(a_train_t)
        loss = torch.mean((pred - u_train_t) ** 2)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history


def blackbox_grid_search(model, u_target_t, a_min=0.5, a_max=3.5, n_grid=200):
    """黑盒网格搜索：枚举 a，用代理模型前向评估，找到误差最小的一个（不使用梯度）。"""
    a_candidates = np.linspace(a_min, a_max, n_grid)
    best_a, best_err, n_evals = None, np.inf, 0
    errs = []
    with torch.no_grad():
        for a in a_candidates:
            pred = model(torch.tensor([[a]], dtype=torch.float32))
            err = torch.mean((pred[0] - u_target_t) ** 2).item()
            errs.append(err)
            n_evals += 1
            if err < best_err:
                best_err, best_a = err, a
    return best_a, best_err, n_evals, a_candidates, errs


def gradient_inverse_design(model, u_target_t, a_init=2.0, n_steps=60, lr=0.15):
    """可微逆向设计：直接对参数 a 做梯度下降，让代理模型的输出逼近目标曲线。"""
    a = torch.tensor([[a_init]], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([a], lr=lr)
    err_history = []
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(a)
        loss = torch.mean((pred[0] - u_target_t) ** 2)
        loss.backward()
        optimizer.step()
        err_history.append(loss.item())
    return a.item(), err_history[-1], n_steps, err_history


# ============================================================================
# 可视化：逆向设计对比
# ============================================================================

def plot_inverse_design_comparison(a_true, a_grid, a_grad, u_target, u_grid_pred, u_grad_pred,
                                    grid_evals, grad_evals, grid_errs, grad_err_history,
                                    save_name='as08-03-inverse-design.png'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(X_GRID, u_target, 'k-', linewidth=2.5, label=f'目标曲线 (真实a={a_true:.2f})')
    ax.plot(X_GRID, u_grid_pred, '--', linewidth=1.8, color='#8E44AD',
            label=f'网格搜索找到 a={a_grid:.3f}')
    ax.plot(X_GRID, u_grad_pred, '-.', linewidth=1.8, color='#27AE60',
            label=f'梯度逆向设计找到 a={a_grad:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title('两种逆向设计方法找到的参数 a 对应的曲线', fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    a_candidates = np.linspace(0.5, 3.5, len(grid_errs))
    ax.plot(a_candidates, grid_errs, color='#8E44AD', linewidth=1.5)
    ax.axvline(a_true, color='black', linestyle=':', label=f'真实 a={a_true:.2f}')
    ax.axvline(a_grid, color='#8E44AD', linestyle='--', alpha=0.7, label=f'网格搜索最优 a={a_grid:.3f}')
    ax.set_xlabel('参数 a')
    ax.set_ylabel('与目标曲线的MSE')
    ax.set_title(f'黑盒网格搜索: {grid_evals}次前向评估', fontsize=11.5)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(grad_err_history, color='#27AE60', linewidth=1.8)
    ax.set_yscale('log')
    ax.set_xlabel('梯度下降迭代步数')
    ax.set_ylabel('与目标曲线的MSE (log尺度)')
    ax.set_title(f'可微梯度逆向设计: {grad_evals}次迭代收敛', fontsize=11.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(_save_path(save_name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[可视化] 逆向设计对比已保存至 images/{save_name}')


# ============================================================================
# 主程序
# ============================================================================

def main():
    print('=' * 70)
    print('as08 AI4S 综合与前沿')
    print('=' * 70)

    print('\n[1/4] 绘制 AI4S 发展时间线...')
    plot_timeline()

    print('\n[2/4] 绘制多方法综合对比雷达图...')
    plot_method_radar()

    print('\n[3/4] 训练一个极简代理模型 (参数a -> 解曲线u(x))...')
    model, loss_hist = train_surrogate(n_epochs=2000)
    print(f'  代理模型训练完成，最终训练损失={loss_hist[-1]:.6f}')

    print('\n[4/4] 前沿实操: 可微逆向设计 vs 黑盒网格搜索...')
    a_true = 2.35  # "真实"目标参数（训练范围[0.5,3.5]内的插值点，训练时从未见过）
    u_target = solve_reference(a_true)
    u_target_t = torch.tensor(u_target, dtype=torch.float32)

    a_grid, err_grid, n_evals_grid, a_candidates, grid_errs = blackbox_grid_search(model, u_target_t)
    a_grad, err_grad, n_evals_grad, grad_err_history = gradient_inverse_design(model, u_target_t)

    with torch.no_grad():
        u_grid_pred = model(torch.tensor([[a_grid]], dtype=torch.float32))[0].numpy()
        u_grad_pred = model(torch.tensor([[a_grad]], dtype=torch.float32))[0].numpy()

    print(f'  真实参数: a={a_true:.3f}')
    print(f'  黑盒网格搜索: 找到 a={a_grid:.4f} (误差 |Δa|={abs(a_grid-a_true):.4f}), '
          f'用了 {n_evals_grid} 次前向评估, 最终MSE={err_grid:.6f}')
    print(f'  梯度逆向设计: 找到 a={a_grad:.4f} (误差 |Δa|={abs(a_grad-a_true):.4f}), '
          f'用了 {n_evals_grad} 次迭代(每次都有梯度信息), 最终MSE={err_grad:.6f}')

    plot_inverse_design_comparison(a_true, a_grid, a_grad, u_target, u_grid_pred, u_grad_pred,
                                    n_evals_grid, n_evals_grad, grid_errs, grad_err_history)

    print('\n' + '=' * 70)
    print('【总结】')
    print('=' * 70)
    print('  可微代理模型的核心优势：一旦训练好，"参数 -> 输出"这个映射自带梯度信息，')
    print('  逆向设计（给定目标输出，反推设计参数）就从"黑盒搜索"变成了"梯度下降"，')
    print('  在高维设计参数空间中，这个优势会随维度增加而愈发明显——网格搜索的评估')
    print('  次数随维度指数增长(回忆as01的维度灾难)，而梯度下降的每步成本基本不随')
    print('  维度显著增加。这正是当前 AI4S 前沿（可微仿真、代理模型驱动的科学发现与')
    print('  工程优化闭环）最核心的方法论基础之一。')
    print(f'\n  所有图片已保存至 {_IMAGES_DIR}')
    print('=' * 70)
    print('\n  运行完成！本系列 as01-as08 到此结束，感谢阅读！\n')


if __name__ == '__main__':
    main()
