# -*- coding: utf-8 -*-
"""
=== 机器人动力学：平面 2R 拉格朗日 ===
质量集中在杆末端。无主动力矩时自由落下，对比有阻尼。
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

L1 = L2 = 0.8
M1 = M2 = 1.0
G = 9.81
DT = 0.002


def mass_matrix(th2):
    """M(q) 2x2。θ1 不进惯性（平面旋转对称）。"""
    c2 = np.cos(th2)
    m11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
    m12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
    m22 = M2 * L2 ** 2
    return np.array([[m11, m12], [m12, m22]])


def h_vector(th1, th2, w1, w2):
    """科氏/离心 + 重力：H 使得 M q̈ + H = τ。"""
    s2 = np.sin(th2)
    cor = -M2 * L1 * L2 * s2
    c1 = np.cos(th1)
    c12 = np.cos(th1 + th2)
    g1 = (M1 + M2) * G * L1 * c1 + M2 * G * L2 * c12
    g2 = M2 * G * L2 * c12
    h1 = cor * (2 * w1 * w2 + w2 ** 2) + g1
    h2 = cor * (-w1 ** 2) + g2
    return np.array([h1, h2])


def step(q, w, tau, damp=0.0):
    M = mass_matrix(q[1])
    h = h_vector(q[0], q[1], w[0], w[1])
    acc = np.linalg.solve(M, tau - h - damp * w)
    w = w + DT * acc
    q = q + DT * w
    return q, w


def simulate(damp=0.0, steps=2500):
    q = np.array([0.3, 0.9])
    w = np.zeros(2)
    qs = [q.copy()]
    for _ in range(steps):
        q, w = step(q, w, tau=np.zeros(2), damp=damp)
        qs.append(q.copy())
    return np.array(qs)


def fk(th1, th2):
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    return x, y


def main():
    print('=== 2R 拉格朗日自由落体 ===')
    qs0 = simulate(damp=0.0)
    qs1 = simulate(damp=1.6)
    t = np.arange(len(qs0)) * DT
    print('无阻尼末角速度', np.diff(qs0[-20:], axis=0).mean(0) / DT)
    print('有阻尼末角速度', np.diff(qs1[-20:], axis=0).mean(0) / DT)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(t, qs0[:, 0], label=r'$\theta_1$ 无阻尼')
    axes[0].plot(t, qs0[:, 1], label=r'$\theta_2$ 无阻尼')
    axes[0].plot(t, qs1[:, 0], ls='--', label=r'$\theta_1$ 阻尼')
    axes[0].plot(t, qs1[:, 1], ls='--', label=r'$\theta_2$ 阻尼')
    axes[0].set_xlabel('时间 s')
    axes[0].set_ylabel('关节角 rad')
    axes[0].set_title('能量耗散 vs 永不停的摆')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    xs, ys = fk(qs1[::40, 0], qs1[::40, 1])
    axes[1].plot(xs, ys, 'o-', ms=3, color='#2980B9')
    axes[1].set_aspect('equal')
    axes[1].set_title('有阻尼：末端轨迹')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'lagrange_2r.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
