# -*- coding: utf-8 -*-
"""
=== 香农信息论：二元熵、BSC 容量、互信息 ===
二元对称信道交叉概率 p，C = 1 - h2(p)。
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


def h2(p, eps=1e-12):
    """二元熵（bit）：-p log2 p - (1-p) log2(1-p)。"""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def bsc_capacity(p):
    return 1.0 - h2(p)


def mutual_info_bsc(p_flip, p_x=0.5):
    """BSC：I(X;Y)=H(Y)-H(Y|X)=H(Y)-h2(p_flip)。"""
    # Y 的边缘：P(Y=1)= p_x(1-p) + (1-p_x)p
    py = p_x * (1 - p_flip) + (1 - p_x) * p_flip
    return h2(py) - h2(p_flip)


def main():
    print('=== 香农：二元熵与 BSC ===')
    ps = np.linspace(0.0, 1.0, 201)
    print(f'公平比特 H2(0.5)={h2(0.5):.4f} bit')
    print(f'BSC p=0.11 容量 C={bsc_capacity(0.11):.4f} bit/use')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ps, h2(ps), color='#2E86AB', lw=2)
    axes[0].axvline(0.5, color='k', ls='--', alpha=0.3)
    axes[0].set_xlabel('Bernoulli p')
    axes[0].set_ylabel(r'$h_2(p)$ (bit)')
    axes[0].set_title('二元熵：越确定越低')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ps, bsc_capacity(ps), color='#C1666B', lw=2, label='C=1-h2(p)')
    px = np.linspace(0.01, 0.99, 40)
    i_vals = [mutual_info_bsc(0.11, p) for p in px]
    axes[1].plot(px, i_vals, color='#27AE60', lw=2, label='I(X;Y), p_flip=0.11')
    axes[1].axhline(bsc_capacity(0.11), color='#27AE60', ls=':', alpha=0.7)
    axes[1].set_xlabel('p 或 P(X=1)')
    axes[1].set_title('容量是互信息的最大值')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'bsc_capacity.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
