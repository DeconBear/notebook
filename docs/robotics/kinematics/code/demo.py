# -*- coding: utf-8 -*-
"""
=== 机器人运动学：平面 2R 手臂 ===
正运动学、几何逆解（肘上/肘下）、雅可比与奇异。
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

L1, L2 = 1.0, 0.7


def fk(th1, th2):
    x = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    y = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    return np.array([x, y])


def joints(th1, th2):
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1 * np.cos(th1), L1 * np.sin(th1)])
    p2 = fk(th1, th2)
    return p0, p1, p2


def ik(x, y, elbow='up'):
    r2 = x * x + y * y
    c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow == 'down':
        s2 = -s2
    th2 = np.arctan2(s2, c2)
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    th1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return th1, th2


def jacobian(th1, th2):
    s1, c1 = np.sin(th1), np.cos(th1)
    s12, c12 = np.sin(th1 + th2), np.cos(th1 + th2)
    return np.array([
        [-L1 * s1 - L2 * s12, -L2 * s12],
        [L1 * c1 + L2 * c12, L2 * c12],
    ])


def main():
    print('=== 2R 正/逆运动学 ===')
    target = np.array([1.1, 0.6])
    sols = {}
    for name in ('up', 'down'):
        th = ik(*target, elbow=name)
        p = fk(*th)
        J = jacobian(*th)
        sols[name] = th
        print(f'肘{name}: θ={np.rad2deg(th)} deg  FK={p}  detJ={np.linalg.det(J):.3f}')

    # 工作空间采样
    th1s = np.linspace(-np.pi, np.pi, 80)
    th2s = np.linspace(-2.4, 2.4, 60)
    pts = []
    for a in th1s[::4]:
        for b in th2s[::3]:
            pts.append(fk(a, b))
    pts = np.array(pts)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, c='#D6EAF8', label='可达采样')
    colors = {'up': '#27AE60', 'down': '#E67E22'}
    for name, th in sols.items():
        p0, p1, p2 = joints(*th)
        ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]],
                'o-', color=colors[name], lw=2.5, label=f'肘{name}')
    ax.scatter(*target, c='k', marker='*', s=120, zorder=5, label='目标')
    ax.set_aspect('equal')
    ax.set_title('同一末端：两种肘形')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 奇异：伸直 detJ → 0
    th2_line = np.linspace(-np.pi, np.pi, 200)
    dets = [np.linalg.det(jacobian(0.4, t)) for t in th2_line]
    axes[1].plot(np.rad2deg(th2_line), dets, color='#8E44AD')
    axes[1].axhline(0, color='k', ls='--', alpha=0.4)
    axes[1].set_xlabel(r'$\theta_2$ (deg)')
    axes[1].set_ylabel(r'$\det J$')
    axes[1].set_title('伸直/折叠时雅可比奇异')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'arm_2r.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
