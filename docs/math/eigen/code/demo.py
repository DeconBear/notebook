# -*- coding: utf-8 -*-
"""
=== 特征值与二次型 ===
1) 对称正定 2×2：x^T A x = 1 的椭圆 + 特征向量
2) 幂迭代逼近最大特征值，对照 np.linalg.eig
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


def power_iteration(A, n_iter=20, v0=None):
    """返回 (lambda, v, 每步 Rayleigh 商)。v 是单位特征向量。"""
    n = A.shape[0]
    v = np.ones(n) if v0 is None else np.array(v0, dtype=float)
    v = v / np.linalg.norm(v)
    hist = []
    for _ in range(n_iter):
        Av = A @ v
        lam = float(v @ Av)
        hist.append(lam)
        v = Av / np.linalg.norm(Av)
    return hist[-1], v, hist


def demo_ellipse():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    w, Q = np.linalg.eigh(A)  # 升序
    # x^T A x = 1：特征坐标里 λ1 y1^2 + λ2 y2^2 = 1，半轴 1/sqrt(λ)
    t = np.linspace(0, 2 * np.pi, 200)
    y = np.stack([np.cos(t) / np.sqrt(w[0]), np.sin(t) / np.sqrt(w[1])], axis=0)
    pts = (Q @ y).T

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pts[:, 0], pts[:, 1], color='#2E86AB', lw=2, label=r'$x^\top A x=1$')
    origin = np.zeros(2)
    colors = ['#C1666B', '#27AE60']
    for i in range(2):
        d = Q[:, i] / np.sqrt(w[i])
        ax.arrow(origin[0], origin[1], d[0], d[1],
                 head_width=0.06, color=colors[i], length_includes_head=True, lw=2,
                 label=rf'$\lambda_{i+1}={w[i]:.2f}$')
    ax.set_aspect('equal')
    ax.axhline(0, color='#ccc', lw=0.6)
    ax.axvline(0, color='#ccc', lw=0.6)
    ax.legend()
    ax.set_title('二次型椭圆：轴 = 特征向量')
    out = os.path.join(_IMAGES_DIR, 'eigen_ellipse.png')
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print('特征值', w, '特征矩阵列', Q)


def demo_power():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    lam, v, hist = power_iteration(A, n_iter=15, v0=[1.0, 0.0])
    w, Q = np.linalg.eigh(A)
    print('幂迭代 λ=', lam, ' v=', v)
    print('numpy 最大 λ=', w[-1], ' v=', Q[:, -1])

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(hist, 'o-', color='#2E86AB', label='Rayleigh 商')
    ax.axhline(w[-1], color='#C1666B', ls='--', label=rf'真值 {w[-1]:.4f}')
    ax.set_xlabel('迭代步')
    ax.set_ylabel(r'$\lambda$')
    ax.set_title('幂迭代：最大特征值被反复拉伸出来')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'eigen_power.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def main():
    print('=== 特征值与二次型 ===')
    demo_ellipse()
    demo_power()


if __name__ == '__main__':
    main()
