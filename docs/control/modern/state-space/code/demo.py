# -*- coding: utf-8 -*-
"""
=== 状态空间：双积分器能控性与极点配置 ===
ẍ = u，希望闭环极点 -2, -3，K = [6, 5]。
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

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
K = np.array([[6.0, 5.0]])  # (s+2)(s+3)=s²+5s+6
DT = 0.02
T_END = 4.0
X0 = np.array([1.0, 0.8])  # 初速不为 0，开环才会漂


def controllability_matrix(A, B):
    return np.hstack([B, A @ B])


def simulate(use_feedback):
    n = int(T_END / DT)
    x = X0.copy()
    xs = [x.copy()]
    for _ in range(n):
        u = float((-K @ x).item()) if use_feedback else 0.0
        xdot = A @ x + B.ravel() * u
        x = x + DT * xdot
        xs.append(x.copy())
    t = np.arange(len(xs)) * DT
    return t, np.array(xs)


def main():
    print('=== 状态空间：极点配置 ===')
    Ctrb = controllability_matrix(A, B)
    print('能控性矩阵 [B AB] =\n', Ctrb)
    print('秩', np.linalg.matrix_rank(Ctrb))
    Acl = A - B @ K
    print('闭环特征值', np.round(np.linalg.eigvals(Acl), 6))

    t, xs0 = simulate(False)
    _, xs1 = simulate(True)
    print('无控制末端', xs0[-1])
    print('闭环末端', xs1[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, xs0[:, 0], color='#95A5A6', lw=2, label='无控制')
    axes[0].plot(t, xs1[:, 0], color='#2980B9', lw=2, label=r'$u=-Kx$')
    axes[0].set_xlabel('时间')
    axes[0].set_ylabel('位置')
    axes[0].set_title('位置')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, xs0[:, 1], color='#95A5A6', lw=2, label='无控制')
    axes[1].plot(t, xs1[:, 1], color='#2980B9', lw=2, label=r'$u=-Kx$')
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('速度')
    axes[1].set_title('速度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'pole_place.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
