# -*- coding: utf-8 -*-
"""
=== 线性代数直觉 ===
1) 旋转矩阵把向量转起来
2) 椭圆散点上的 SVD / PCA 主方向
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


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def demo_rotation():
    v = np.array([1.5, 0.4])
    angles = [0, np.pi / 6, np.pi / 3, np.pi / 2]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(0, color='#aaa', lw=0.8)
    ax.axvline(0, color='#aaa', lw=0.8)
    for i, th in enumerate(angles):
        w = rotation_matrix(th) @ v
        ax.arrow(0, 0, w[0], w[1], head_width=0.08, length_includes_head=True,
                 color=plt.cm.viridis(i / (len(angles) - 1)), label=f'{th * 180 / np.pi:.0f}°')
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 2)
    ax.legend(title='旋转角')
    ax.set_title('矩阵作用：旋转把向量转到新方向')
    out = os.path.join(_IMAGES_DIR, 'la_rotation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def demo_pca():
    # 椭圆状数据：先在轴对齐高斯上采样，再旋转拉伸
    n = 200
    z = np.random.randn(n, 2) * np.array([2.0, 0.5])
    R = rotation_matrix(np.deg2rad(35))
    X = z @ R.T
    X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = Vt  # 主方向（行）

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(X[:, 0], X[:, 1], s=12, alpha=0.5, c='#5B8FF9')
    mu = np.zeros(2)
    for i, color in enumerate(['#E8684A', '#5AD8A6']):
        d = pcs[i] * S[i] / np.sqrt(n)
        axes[0].arrow(mu[0], mu[1], d[0], d[1], head_width=0.15,
                      color=color, length_includes_head=True, lw=2,
                      label=f'PC{i+1} (σ={S[i]/np.sqrt(n):.2f})')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title('SVD：主方向对齐数据拉伸轴')

    # 投影到第一主成分
    scores = X @ pcs[0]
    proj = scores[:, None] * pcs[0][None, :]
    axes[1].scatter(X[:, 0], X[:, 1], s=10, alpha=0.25, c='#aaa', label='原始')
    axes[1].scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.7, c='#E8684A', label='投到 PC1')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].set_title('PCA 降维：用一维近似二维')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'la_pca.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print('奇异值:', np.round(S, 3))


def main():
    print('=== 线性代数直觉 ===')
    demo_rotation()
    demo_pca()


if __name__ == '__main__':
    main()
