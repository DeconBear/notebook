# -*- coding: utf-8 -*-
"""
=== 旋量：平面刚体沿螺旋轴运动 ===
平面 se(2)：ξ=(ω, vx, vy)。指数映射给出刚体沿螺旋的位姿。
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


def se2_exp(omega, vx, vy, t=1.0):
    """T(t)=exp(t ξ̂)。ω=0 时退化为纯平移。"""
    if abs(omega) < 1e-10:
        return np.array([
            [1.0, 0.0, vx * t],
            [0.0, 1.0, vy * t],
            [0.0, 0.0, 1.0],
        ])
    w = omega * t
    c, s = np.cos(w), np.sin(w)
    trans = (1.0 / omega) * np.array([[s, -(1 - c)], [1 - c, s]]) @ np.array([vx, vy])
    T = np.eye(3)
    T[0, 0], T[0, 1], T[0, 2] = c, -s, trans[0]
    T[1, 0], T[1, 1], T[1, 2] = s, c, trans[1]
    return T


def apply(T, pts):
    h = np.c_[pts, np.ones(len(pts))]
    return (T @ h.T).T[:, :2]


def main():
    print('=== 平面旋量指数映射 ===')
    omega = 1.2
    q = np.array([0.8, 0.2])
    # 纯转动：线速度 v = ω × r 在平面上为 ω(-qy, qx) 的相反约定见练习
    v = omega * np.array([q[1], -q[0]])
    print('螺旋 ω, v =', omega, v)

    square = np.array([
        [0.15, 0.1], [0.35, 0.1], [0.35, 0.28], [0.15, 0.28], [0.15, 0.1],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*q, c='k', marker='x', s=80, label='瞬心')
    for i, t in enumerate(np.linspace(0, 1.6, 7)):
        T = se2_exp(omega, v[0], v[1], t)
        p = apply(T, square)
        ax.plot(p[:, 0], p[:, 1], color=plt.cm.viridis(i / 6), lw=2)
    ax.set_aspect('equal')
    ax.set_title('刚体沿固定螺旋轴「拧」过去')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'screw_se2.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
