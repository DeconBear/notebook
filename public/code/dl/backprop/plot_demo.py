# -*- coding: utf-8 -*-
"""
=== s06 反向传播 — 配图脚本（与 autograd 主线分开） ===
画两张图：MSE 与对权重的梯度；演示 1 表达式 (a×b+c)×d 的前向/反向。
数学与 Value 引擎在 demo.py；本文件只负责 matplotlib。

运行: python plot_demo.py
"""
import os

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)


def plot_mse_and_weight_gradient():
    """左：残差平方；右：L(w) 上的梯度方向 vs 参数更新方向。"""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # ---- 左图：ℓ = ½ e²，e = ŷ − y ----
    ax = axes[0]
    e = np.linspace(-2.4, 2.4, 400)
    ell = 0.5 * e ** 2
    ax.plot(e, ell, color='#2E86AB', linewidth=2.6, label=r'$\ell=\frac{1}{2}(\hat{y}-y)^2$')
    ax.axhline(0, color='#888888', linewidth=0.8)
    ax.axvline(0, color='#888888', linewidth=0.8)

    e0 = 1.2
    ell0 = 0.5 * e0 ** 2
    ax.scatter([e0], [ell0], s=70, zorder=5, color='#C73E1D')
    ax.plot([e0, e0], [0, ell0], color='#C73E1D', linestyle='--', linewidth=1.2)
    ax.annotate(
        r'残差 $e=\hat{y}-y>0$' + '\n预测偏高，损失 > 0',
        xy=(e0, ell0), xytext=(0.35, 2.15),
        fontsize=10, color='#C73E1D',
        arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=1.2),
    )
    ax.annotate(
        '残差 = 0\n预测完全命中',
        xy=(0.0, 0.0), xytext=(-2.25, 0.85),
        fontsize=10, color='#2A9D8F',
        arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=1.2),
    )
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.15, 3.0)
    ax.set_xlabel(r'残差 $e=\hat{y}-y$', fontsize=11)
    ax.set_ylabel(r'单样本损失 $\ell$', fontsize=11)
    ax.set_title('MSE 从哪来：把误差平方成一个非负数', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.28)

    # ---- 右图：ŷ = w·x，x=2, y=1，L(w)=½(2w−1)² ----
    ax = axes[1]
    w = np.linspace(-0.4, 1.5, 400)
    L = 0.5 * (2 * w - 1) ** 2
    ax.plot(w, L, color='#2E86AB', linewidth=2.6, label=r'$L(w)=\frac{1}{2}(2w-1)^2$')

    w_now = 1.0
    L_now = 0.5 * (2 * w_now - 1) ** 2
    dLdw = 4 * w_now - 2  # (2w-1)*2
    # 切线
    w_tan = np.linspace(w_now - 0.35, w_now + 0.35, 20)
    ax.plot(w_tan, L_now + dLdw * (w_tan - w_now), color='#C73E1D',
            linestyle='--', linewidth=1.6, label=fr'切线斜率 $\partial L/\partial w={dLdw:.0f}$')
    ax.scatter([w_now], [L_now], s=70, zorder=5, color='#C73E1D')
    ax.scatter([0.5], [0.0], s=70, zorder=5, color='#2A9D8F')

    ax.annotate(
        '', xy=(w_now + 0.28, L_now + dLdw * 0.28), xytext=(w_now, L_now),
        arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=2.0),
    )
    ax.text(w_now + 0.18, L_now + 0.55, '梯度方向\n（上坡，L 增大）',
            fontsize=10, color='#C73E1D', ha='left')

    ax.annotate(
        '', xy=(w_now - 0.32, L_now), xytext=(w_now, L_now),
        arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=2.0),
    )
    ax.text(0.52, 0.72, r'更新 $w\leftarrow w-\alpha\partial L/\partial w$' + '\n（下坡，L 减小）',
            fontsize=10, color='#2A9D8F', ha='left')

    ax.annotate('最优 $w^*=0.5$\n此时 $\\hat{y}=y$',
                xy=(0.5, 0.0), xytext=(-0.32, 0.85),
                fontsize=10, color='#2A9D8F',
                arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=1.2))
    ax.text(0.92, -0.12, r'当前 $w=1$', fontsize=10, color='#C73E1D')

    ax.set_xlim(-0.45, 1.55)
    ax.set_ylim(-0.2, 2.35)
    ax.set_xlabel(r'权重 $w$（$\hat{y}=wx$，$x=2$，$y=1$）', fontsize=11)
    ax.set_ylabel(r'损失 $L(w)$', fontsize=11)
    ax.set_title('为什么对 $w$ 求梯度：它告诉这一步该往哪改', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.28)

    fig.tight_layout()
    out = os.path.join(_IMAGES, '06-05-mse-loss-and-weight-gradient.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('[可视化] 已保存 ' + out)


def _draw_expr_graph(ax, mode):
    """mode='forward' 标 data；mode='backward' 标 grad。"""
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(-0.35, 11.4)
    ax.set_ylim(-0.35, 3.35)
    ax.axis('off')
    ax.set_aspect('equal')

    # x 坐标：叶子 / 中间 / 叶子 / 中间 / 叶子 / 输出
    pos = {
        'a': (0.7, 2.55), 'b': (0.7, 0.55),
        'e': (3.5, 1.55),
        'c': (3.5, 0.35),
        'f': (6.4, 1.55),
        'd': (6.4, 0.35),
        'L': (9.5, 1.55),
    }
    if mode == 'forward':
        label = {
            'a': 'a = 2\n叶子', 'b': 'b = 3\n叶子',
            'e': 'e = 6\na × b',
            'c': 'c = 4\n叶子',
            'f': 'f = 10\ne + c',
            'd': 'd = 5\n叶子',
            'L': 'L = 50\nf × d',
        }
        face = {
            'a': '#D6EAF8', 'b': '#D6EAF8', 'c': '#D6EAF8', 'd': '#D6EAF8',
            'e': '#D5F5E3', 'f': '#D5F5E3', 'L': '#FADBD8',
        }
        title = '前向：从左往右算数，每扇门 return 一个新节点'
        edge_color = '#2E86AB'
    else:
        label = {
            'a': 'a.grad = 15', 'b': 'b.grad = 10',
            'e': 'e.grad = 5',
            'c': 'c.grad = 5',
            'f': 'f.grad = 5',
            'd': 'd.grad = 10',
            'L': 'L.grad = 1\n起点',
        }
        face = {
            'a': '#FADBD8', 'b': '#FADBD8', 'c': '#FADBD8', 'd': '#FADBD8',
            'e': '#FDEBD0', 'f': '#FDEBD0', 'L': '#F5B7B1',
        }
        title = '反向：从 L 往左还账，一次只走一扇门'
        edge_color = '#C0392B'

    edges = [('a', 'e'), ('b', 'e'), ('e', 'f'), ('c', 'f'), ('f', 'L'), ('d', 'L')]
    if mode == 'backward':
        edges = [(dst, src) for src, dst in edges]
    for src, dst in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        dx, dy = x1 - x0, y1 - y0
        dist = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / dist, dy / dist
        pad = 0.92
        ax.annotate(
            '', xy=(x1 - ux * pad, y1 - uy * pad),
            xytext=(x0 + ux * pad, y0 + uy * pad),
            arrowprops=dict(arrowstyle='->', color=edge_color, lw=1.8),
        )

    for name, (x, y) in pos.items():
        box = FancyBboxPatch(
            (x - 0.85, y - 0.42), 1.7, 0.84,
            boxstyle='round,pad=0.08,rounding_size=0.18',
            facecolor=face[name], edgecolor='#2C3E50', linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x, y, label[name], ha='center', va='center', fontsize=10)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)


def plot_basic_expression_backprop():
    """演示 1 同一表达式：(a×b + c)×d 的前向数值与反向梯度。"""
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.6))
    _draw_expr_graph(axes[0], 'forward')
    _draw_expr_graph(axes[1], 'backward')
    fig.suptitle(
        r'$L=(a\times b+c)\times d$，取 $a=2,b=3,c=4,d=5$',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(_IMAGES, '06-06-basic-expression-backprop.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('[可视化] 已保存 ' + out)


def plot_fanout_backprop():
    """演示 3：x 走两条路再汇合，梯度必须累加。"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(-0.4, 10.6)
    ax.set_ylim(-0.4, 4.2)
    ax.axis('off')

    pos = {
        'x': (1.1, 2.05),
        'u': (4.3, 3.25),
        'v': (4.3, 0.85),
        'L': (8.2, 2.05),
    }
    label = {
        'x': 'x = 2\n被用了两次',
        'u': 'u = 2x = 4',
        'v': 'v = x+3 = 5',
        'L': 'L = u×v = 20',
    }
    face = {
        'x': '#D6EAF8', 'u': '#D5F5E3', 'v': '#D5F5E3', 'L': '#FADBD8',
    }
    for name, (px, py) in pos.items():
        box = FancyBboxPatch(
            (px - 1.05, py - 0.48), 2.1, 0.96,
            boxstyle='round,pad=0.08,rounding_size=0.18',
            facecolor=face[name], edgecolor='#2C3E50', linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(px, py, label[name], ha='center', va='center', fontsize=11)

    def arrow(src, dst, color, rad=0.0):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.add_patch(FancyArrowPatch(
            (x0 + 1.05, y0), (x1 - 1.05, y1),
            arrowstyle='-|>', mutation_scale=14,
            color=color, lw=1.8, connectionstyle=f'arc3,rad={rad}',
        ))

    arrow('x', 'u', '#2E86AB', rad=-0.12)
    arrow('x', 'v', '#2E86AB', rad=0.12)
    arrow('u', 'L', '#2E86AB', rad=-0.08)
    arrow('v', 'L', '#2E86AB', rad=0.08)

    ax.text(2.7, 3.45, r'路径1  $\times 2$', color='#1A5276', fontsize=10)
    ax.text(2.7, 0.35, r'路径2  $+3$', color='#1A5276', fontsize=10)
    ax.text(6.15, 3.35, r'$\partial L/\partial u=v=5$', color='#C0392B', fontsize=10)
    ax.text(6.15, 0.45, r'$\partial L/\partial v=u=4$', color='#C0392B', fontsize=10)
    ax.text(1.1, 3.55, r'$x$.grad $=10+4=14$', color='#C0392B', fontsize=11, ha='center')
    ax.set_title(
        r'Fan-out：$L=(2x)(x+3)$，两条路径的梯度加到同一个 $x$（所以代码用 +=）',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    out = os.path.join(_IMAGES, '06-07-fanout-gradient-sum.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('[可视化] 已保存 ' + out)


if __name__ == '__main__':
    plot_mse_and_weight_gradient()
    plot_basic_expression_backprop()
    plot_fanout_backprop()
