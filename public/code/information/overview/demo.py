# -*- coding: utf-8 -*-
"""
=== 信息论导论：四种分布的熵（bit）===
确定性 / 偏伯努利 / 公平硬币 / 四面均匀。
运行: python demo.py
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def entropy_bits(p, eps=1e-15):
    """H = -sum p log2 p，丢掉数值零。"""
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    return float(-np.sum(p * np.log2(p)))


def nats_to_bits(h_nats):
    return h_nats / np.log(2.0)


def main():
    print('=== 信息论导论 ===')
    print(f'1 nat = {nats_to_bits(1.0):.4f} bit')
    cases = [
        ('确定性 [1,0]', np.array([1.0, 0.0])),
        ('偏硬币 0.1/0.9', np.array([0.1, 0.9])),
        ('公平硬币', np.array([0.5, 0.5])),
        ('四面均匀', np.ones(4) / 4.0),
    ]
    names, hs = [], []
    for name, p in cases:
        h = entropy_bits(p)
        names.append(name)
        hs.append(h)
        print(f'{name:16s}  H={h:.4f} bit')

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    colors = ['#95A5A6', '#E67E22', '#1ABC9C', '#2980B9']
    ax.bar(names, hs, color=colors)
    ax.axhline(1.0, color='k', ls='--', alpha=0.35, label='1 bit')
    ax.axhline(2.0, color='k', ls=':', alpha=0.35, label='2 bit')
    ax.set_ylabel('熵 H (bit)')
    ax.set_title('越确定越低，均匀且符号越多越高')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'overview_entropy.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
