# -*- coding: utf-8 -*-
"""
=== 熵与条件熵：2×2 联合表 ===
核对链规则 H(X,Y)=H(X)+H(Y|X)，并画出 I(X;Y)。
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
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > eps]
    return float(-np.sum(p * np.log2(p)))


def from_joint(Pxy):
    """Pxy[x, y]，返回 HX, HY, HXY, HXgY, I。"""
    Pxy = np.asarray(Pxy, dtype=float)
    Pxy = Pxy / Pxy.sum()
    px = Pxy.sum(axis=1)
    py = Pxy.sum(axis=0)
    hx, hy, hxy = entropy_bits(px), entropy_bits(py), entropy_bits(Pxy)
    hx_given_y = hxy - hy
    hy_given_x = hxy - hx
    ixy = hx - hx_given_y
    return dict(HX=hx, HY=hy, HXY=hxy, HXgY=hx_given_y, HYgX=hy_given_x, I=ixy)


def main():
    print('=== 熵与条件熵 ===')
    Pxy = np.array([[0.10, 0.30],
                    [0.40, 0.20]])
    stats = from_joint(Pxy)
    for k, v in stats.items():
        print(f'{k:6s} = {v:.4f} bit')
    err = abs(stats['HXY'] - (stats['HX'] + stats['HYgX']))
    print(f'链规则误差 {err:.2e}')

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    im = axes[0].imshow(Pxy, cmap='YlGnBu', vmin=0, vmax=0.5)
    axes[0].set_xticks([0, 1], ['y=0', 'y=1'])
    axes[0].set_yticks([0, 1], ['x=0', 'x=1'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{Pxy[i, j]:.2f}', ha='center', va='center')
    axes[0].set_title('联合 p(x,y)')
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    labels = [r'$H(X)$', r'$H(Y)$', r'$H(X,Y)$', r'$H(X|Y)$', r'$I(X;Y)$']
    vals = [stats['HX'], stats['HY'], stats['HXY'], stats['HXgY'], stats['I']]
    axes[1].bar(labels, vals, color=['#1ABC9C', '#3498DB', '#8E44AD', '#E67E22', '#C0392B'])
    axes[1].set_ylabel('bit')
    axes[1].set_title('分解')
    axes[1].grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'entropy_joint.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
