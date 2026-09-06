# -*- coding: utf-8 -*-
"""
=== Python 基础 ===
1) 容器与切片
2) 类是图纸，实例是机器（对照 nn.GRUCell）
运行: python demo.py
"""
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


class GRULike:
    """假细胞：只有一个可改的 scale，用来说明「实例 ≠ 隐状态」。"""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.scale = 0.9  # 相当于「权重」，造对象时定下来

    def __call__(self, x, h_prev):
        # 返回新的 h，不改 self.scale
        n = min(len(h_prev), len(x), self.hidden_size)
        return [self.scale * h_prev[i] + x[i] for i in range(n)]


def demo_containers():
    xs = [10, 20, 30, 40]
    print('切片 xs[1:3] =', xs[1:3], '  # 含 1 不含 3')
    print('推导偶数平方', [x * x for x in xs if x % 20 == 0])
    cfg = {'lr': 1e-3, 'seed': 42}
    print('dict 取值', cfg['lr'])


def demo_class_vs_instance():
    cell = GRULike(hidden_size=2)          # 造一台机器（类似 nn.GRUCell(...)）
    h = [0.0, 0.0]
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    print('self.gru 这类对象 id =', id(cell), '  scale=', cell.scale)
    for t, x in enumerate(xs, start=1):
        h = cell(x, h)                     # 同一台机器调用三次，h 在变
        print(f'  t={t}  h={ [round(v, 4) for v in h] }')
    print('三次循环后 cell 还是同一个对象，scale 仍是', cell.scale)


def draw_class_figure():
    cream, navy = '#FBF7F0', '#1F3A5F'
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.2)
    ax.axis('off')
    fig.patch.set_facecolor(cream)
    ax.set_facecolor(cream)
    ax.text(5.25, 4.85, '类是图纸，实例是机器，调用才得到 h_t',
            ha='center', fontsize=14, fontweight='bold', color=navy)

    def box(x, y, w, h, fc, title, sub):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1',
                                    facecolor=fc, edgecolor=navy, linewidth=1.3))
        ax.text(x + w / 2, y + h * 0.62, title, ha='center', fontsize=11,
                fontweight='bold', color=navy)
        ax.text(x + w / 2, y + h * 0.28, sub, ha='center', fontsize=9, color='#5A6A7A')

    box(0.4, 2.6, 3.0, 1.6, '#D6EAF8', 'class GRUCell', '图纸，只有一份')
    box(3.8, 2.6, 3.0, 1.6, '#D5F5E3', 'gru = GRUCell(...)', '实例：带着权重')
    box(7.2, 2.6, 2.9, 1.6, '#FCF3CF', 'h = gru(x, h)', '返回值才是 h_t')
    ax.annotate('', xy=(3.75, 3.4), xytext=(3.45, 3.4),
                arrowprops=dict(arrowstyle='-|>', color=navy, lw=1.5))
    ax.annotate('', xy=(7.15, 3.4), xytext=(6.85, 3.4),
                arrowprops=dict(arrowstyle='-|>', color=navy, lw=1.5))
    ax.text(5.25, 1.7, 'RSSM:  self.gru = nn.GRUCell(...)   ← 这一行只造机器',
            ha='center', fontsize=10.5, color=navy)
    ax.text(5.25, 1.15, '循环里:  h = self.gru(cat([s, a]), h)   ← 这里才更新确定性状态',
            ha='center', fontsize=10.5, color=navy)
    ax.text(5.25, 0.45, '五个时间步 = 同一实例被调用五次，不是五个类',
            ha='center', fontsize=10, color='#5A6A7A')
    out = os.path.join(_IMAGES_DIR, 'prog-python-class.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=cream)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    demo_containers()
    print()
    demo_class_vs_instance()
    draw_class_figure()
