# -*- coding: utf-8 -*-
"""
=== 根轨迹：1 + K / (s(s+1)(s+3)) = 0 ===
特征多项式 s³ + 4s² + 3s + K。扫描 K，画闭环极点。
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
np.random.seed(42)


def closed_loop_roots(k):
    """s³ + 4s² + 3s + K = 0 的三个根。"""
    return np.roots([1.0, 4.0, 3.0, k])


def main():
    print('=== 根轨迹 ===')
    ks = np.linspace(0.0, 40.0, 81)
    reals, imags, colors = [], [], []
    last_stable_k = 0.0
    for k in ks:
        rts = closed_loop_roots(k)
        if np.all(np.real(rts) < 0):
            last_stable_k = k
        for r in rts:
            reals.append(np.real(r))
            imags.append(np.imag(r))
            colors.append(k)
    for k in (0.0, 4.0, 10.0, 25.0):
        print(f'K={k:5.1f}  根={np.round(closed_loop_roots(k), 3)}')
    print(f'扫描网格上仍全左半平面的最大 K ≈ {last_stable_k:.1f}')

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sc = ax.scatter(reals, imags, c=colors, s=18, cmap='viridis', linewidths=0)
    fig.colorbar(sc, ax=ax, label='增益 K')
    ax.axhline(0.0, color='k', lw=0.6)
    ax.axvline(0.0, color='k', lw=0.6)
    ax.plot([0, -1, -3], [0, 0, 0], 'rx', ms=10, mew=2, label='开环极点 K=0')
    ax.set_xlabel('实部 σ')
    ax.set_ylabel('虚部 jω')
    ax.set_title(r'根轨迹  $1+K/(s(s+1)(s+3))=0$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'root_locus.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
