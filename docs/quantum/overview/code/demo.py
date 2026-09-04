# -*- coding: utf-8 -*-
"""
=== 量子信息全景 ===
对比「相干叠加」与「经典混合」：两者测量 |0>/|1| 都可以是 50/50，
但只有叠加态能被 Hadamard 拉回 |0>。
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

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def shots_from_diag(diag, n=2000):
    p = np.real(diag)
    p = p / p.sum()
    return np.random.choice(2, size=n, p=p)


def demo():
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_sup = np.outer(plus, plus.conj())
    rho_mix = 0.5 * np.eye(2)
    # 直接测量 Z
    z_sup = shots_from_diag(np.diag(rho_sup))
    z_mix = shots_from_diag(np.diag(rho_mix))
    # 先 H 再测 Z：叠加应变回 |0>，混合仍约 50/50
    rho_sup_h = H @ rho_sup @ H.conj().T
    rho_mix_h = H @ rho_mix @ H.conj().T
    h_sup = shots_from_diag(np.diag(rho_sup_h))
    h_mix = shots_from_diag(np.diag(rho_mix_h))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, (a, b), title in zip(
        axes,
        [(z_sup, z_mix), (h_sup, h_mix)],
        ['直接测 Z（都像抛硬币）', '先 H 再测 Z（叠加能「收回」）'],
    ):
        ax.bar([-0.2, 0.8], [np.mean(a == 0), np.mean(a == 1)], width=0.35, label='叠加 |+>')
        ax.bar([0.2, 1.2], [np.mean(b == 0), np.mean(b == 1)], width=0.35, label='混合 ½I')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('频率')
    fig.suptitle('叠加不是「不知道」，混合才是「真随机」')
    fig.tight_layout()
    path = os.path.join(_IMAGES_DIR, 'superposition_vs_mixture.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'已保存 {path}')


if __name__ == '__main__':
    demo()
    print('完成：叠加 vs 混合。')
