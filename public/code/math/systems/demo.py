# -*- coding: utf-8 -*-
"""
=== 线性方程组与秩 ===
1) 两条直线：唯一解 / 无解 / 无穷多解
2) 3×3 高斯消元对照 NumPy
3) 秩：行线性相关时掉一档
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


def gauss_solve(A, b, eps=1e-10):
    """部分主元高斯消元。A,b 会被拷贝，不改调用方。奇异返回 None。"""
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    n = A.shape[0]
    for col in range(n):
        piv = col + int(np.argmax(np.abs(A[col:, col])))
        if abs(A[piv, col]) < eps:
            return None
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
            b[[col, piv]] = b[[piv, col]]
        pv = A[col, col]
        A[col] /= pv
        b[col] /= pv
        for i in range(n):
            if i == col:
                continue
            f = A[i, col]
            A[i] -= f * A[col]
            b[i] -= f * b[col]
    return b


def numerical_rank(A, tol=1e-8):
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > tol))


def demo_geometry():
    xs = np.linspace(-1, 4, 50)
    cases = [
        ('唯一解 rank=2', (1, 1, 2), (1, -1, 0), '#2E86AB'),
        ('无解（平行）', (1, 1, 2), (1, 1, 3), '#C1666B'),
        ('无穷多解（重合）', (1, 1, 2), (2, 2, 4), '#27AE60'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, e1, e2, color) in zip(axes, cases):
        a1, b1, c1 = e1
        a2, b2, c2 = e2
        ax.plot(xs, (c1 - a1 * xs) / b1, color=color, lw=2, label='方程1')
        ax.plot(xs, (c2 - a2 * xs) / b2, color='#333', lw=2, ls='--', label='方程2')
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1.5, 3.5)
        ax.set_aspect('equal')
        ax.axhline(0, color='#ccc', lw=0.6)
        ax.axvline(0, color='#ccc', lw=0.6)
        ax.set_title(title)
        ax.legend(fontsize=8)
        if title.startswith('唯一'):
            ax.scatter([1], [1], c='#E8684A', s=50, zorder=5)
    fig.suptitle('Ax=b：两条直线交点、平行、重合', y=1.02)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'systems_lines.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('保存', out)


def demo_gauss():
    A = np.array([[2.0, 1.0, -1.0],
                  [-3.0, -1.0, 2.0],
                  [-2.0, 1.0, 2.0]])
    b = np.array([8.0, -11.0, -3.0])
    x_hand = gauss_solve(A, b)
    x_np = np.linalg.solve(A, b)
    print('手写消元 x =', x_hand)
    print('numpy.solve x =', x_np)
    print('残差 ||Ax-b|| =', np.linalg.norm(A @ x_hand - b))

    B = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0],
                  [1.0, 1.0, 1.0]])
    print('rank(A)=', numerical_rank(A), ' rank(B)=', numerical_rank(B))


def main():
    print('=== 线性方程组与秩 ===')
    demo_geometry()
    demo_gauss()


if __name__ == '__main__':
    main()
