# -*- coding: utf-8 -*-
"""
=== torch.nn 模块怎么用 ===
1) Linear / Sequential 形状
2) GRUCell 一步 ≠ 整段 GRU
3) CrossEntropy 吃 logit
运行: python demo.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
torch.manual_seed(42)


class TinyMLP(nn.Module):
    def __init__(self, d_in=4, d_hid=8, n_class=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Linear(d_hid, n_class),
        )

    def forward(self, x):
        return self.net(x)


def demo_linear_seq():
    m = TinyMLP()
    x = torch.randn(2, 4)
    logit = m(x)
    print('MLP 输入', tuple(x.shape), '→ logit', tuple(logit.shape))
    print('子模块:')
    for n, p in m.named_parameters():
        print(' ', n, tuple(p.shape))


def demo_grucell():
    stoch, act, deter = 4, 2, 8
    gru = nn.GRUCell(stoch + act, deter)     # 实例，不是 h
    h = torch.zeros(3, deter)
    s = torch.zeros(3, stoch)
    a = torch.randn(3, act)
    h1 = gru(torch.cat([s, a], dim=-1), h)
    print('GRUCell 对象 id', id(gru), '  h1.shape', tuple(h1.shape))
    print('同一 gru 再走一步...')
    h2 = gru(torch.cat([s, a], dim=-1), h1)
    print(' h2.shape', tuple(h2.shape), '  仍是同一个 gru id', id(gru))


def demo_loss():
    logit = torch.tensor([[2.0, 0.1, -1.0], [0.0, 3.0, 0.2]])
    target = torch.tensor([0, 1])
    ce = nn.CrossEntropyLoss()
    print('CrossEntropy(logit, class_id) =', float(ce(logit, target)))
    # 错误示范：先 softmax 再 CE 会把概率再当 logit，损失尺度乱掉
    wrong = float(ce(F.softmax(logit, dim=-1), target))
    print('若先 Softmax 再 CE（不要这样做）=', wrong)


def draw_nn_map():
    cream, navy = '#FBF7F0', '#1F3A5F'
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis('off')
    fig.patch.set_facecolor(cream)
    ax.set_facecolor(cream)
    ax.text(5.6, 5.05, 'nn 里常见的三种「造出来再用」',
            ha='center', fontsize=14, fontweight='bold', color=navy)

    def box(x, y, w, h, fc, t, s):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1',
                                    facecolor=fc, edgecolor=navy, linewidth=1.2))
        ax.text(x + w / 2, y + h * 0.62, t, ha='center', fontsize=11, fontweight='bold', color=navy)
        ax.text(x + w / 2, y + h * 0.28, s, ha='center', fontsize=8.5, color='#5A6A7A')

    box(0.35, 2.55, 3.3, 1.7, '#D6EAF8', 'nn.Linear(3,4)', '一层 Wx+b，不是 MLP')
    box(3.95, 2.55, 3.3, 1.7, '#D5F5E3', 'nn.Sequential(...)', '直线把层串起来')
    box(7.55, 2.55, 3.3, 1.7, '#F5EEF8', 'nn.GRUCell(in, hid)', '一步；RSSM 用这个')
    ax.text(5.6, 1.7, '左边都是类；括号里才是实例。h_t / y 是 forward 的返回值。',
            ha='center', fontsize=10.5, color=navy)
    ax.text(5.6, 1.1, '整段序列用 nn.GRU；中间要插先验/后验就自己 for + Cell。',
            ha='center', fontsize=10, color='#5A6A7A')
    ax.text(5.6, 0.45, '损失：CrossEntropy 吃 logit；MSE 吃回归值。',
            ha='center', fontsize=10, color='#5A6A7A')
    out = os.path.join(_IMAGES_DIR, 'prog-nn-three.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=cream)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    demo_linear_seq()
    print()
    demo_grucell()
    print()
    demo_loss()
    draw_nn_map()
