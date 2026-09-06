# -*- coding: utf-8 -*-
"""
=== 机器人学导论 ===
1) 工作空间：手能到的点（圆环带）
2) 构型空间：关节角的矩形，直线路径映到手却是弯的
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
    return np.array([
        L1 * np.cos(th1) + L2 * np.cos(th1 + th2),
        L1 * np.sin(th1) + L2 * np.sin(th1 + th2),
    ])


def main():
    print('=== 构型空间 vs 工作空间 ===')
    th1s = np.linspace(-np.pi, np.pi, 70)
    th2s = np.linspace(-2.5, 2.5, 50)
    pts = np.array([fk(a, b) for a in th1s[::3] for b in th2s[::2]])
    print(f'可达采样点数 {len(pts)}  半径约 [{abs(L1-L2):.2f}, {L1+L2:.2f}]')

    t = np.linspace(0, 1, 40)
    th_a = np.array([-0.4, 1.2])
    th_b = np.array([1.1, -0.6])
    th_path = th_a[None, :] + t[:, None] * (th_b - th_a)
    ee = np.array([fk(*q) for q in th_path])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    ax.scatter(pts[:, 0], pts[:, 1], s=3, c='#D6EAF8')
    ax.plot(ee[:, 0], ee[:, 1], color='#C1666B', lw=2, label='关节直线对应的手路径')
    for q, mk in ((th_a, 'o'), (th_b, 's')):
        p0 = np.zeros(2)
        p1 = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
        p2 = fk(*q)
        ax.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], 'o-', color='#2E86AB', lw=1.5)
        ax.scatter(*p2, marker=mk, c='#E8684A', s=50, zorder=5)
    ax.set_aspect('equal')
    ax.set_title('工作空间：手在平面上能到哪')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(th_path[:, 0], th_path[:, 1], color='#C1666B', lw=2, label=r'$\theta$ 空间直线')
    ax.scatter(th_a[0], th_a[1], c='#E8684A', s=50, zorder=5)
    ax.scatter(th_b[0], th_b[1], marker='s', c='#E8684A', s=50, zorder=5)
    ax.set_xlabel(r'$\theta_1$ (rad)')
    ax.set_ylabel(r'$\theta_2$ (rad)')
    ax.set_title('构型空间：关节角是坐标')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'overview_cspace.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
