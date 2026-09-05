# -*- coding: utf-8 -*-
"""
=== 机构学：铰链四杆 ===
曲柄摇杆。输入角 θ 驱动，输出角 ψ 由环闭合方程解出，画连杆曲线。
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

# 机架 / 曲柄 / 连杆 / 摇杆
L0, L1, L2, L3 = 1.4, 0.45, 1.1, 0.9


def coupler_point(theta):
    """曲柄端 A，求 B 使 |B-A|=L2 且 |B-D|=L3，D=(L0,0)。取叉积为正的解。"""
    A = np.array([L1 * np.cos(theta), L1 * np.sin(theta)])
    D = np.array([L0, 0.0])
    d = D - A
    dist = np.linalg.norm(d)
    if dist < 1e-9 or dist > L2 + L3 or dist < abs(L2 - L3):
        return None, None
    a = (L2 ** 2 - L3 ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(max(L2 ** 2 - a * a, 0.0))
    mid = A + a * d / dist
    n = np.array([-d[1], d[0]]) / dist
    B = mid + h * n
    return A, B


def main():
    print('=== 四杆机构 ===')
    N = 3 * 1 + 4  # 演示 Gruebler：平面 N=4, J1=4 → M=3(4-1)-2*4=1
    print(f'Gruebler: M=3(N-1)-2 J1 = 3(4-1)-2*4 = {3 * (4 - 1) - 2 * 4}  (期望 1)')
    thetas = np.linspace(0, 2 * np.pi, 180)
    curve = []
    snapshot = None
    for th in thetas:
        A, B = coupler_point(th)
        if B is None:
            continue
        P = 0.5 * (A + B)  # 连杆中点当「偶联点」
        curve.append(P)
        if snapshot is None and abs(th - 0.7) < 0.05:
            snapshot = (th, A, B, P)
    curve = np.array(curve)
    print('连杆中点轨迹点数', len(curve))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve[:, 0], curve[:, 1], color='#8E44AD', lw=2, label='偶联点轨迹')
    if snapshot:
        th, A, B, P = snapshot
        D = np.array([L0, 0.0])
        O = np.array([0.0, 0.0])
        ax.plot([O[0], A[0], B[0], D[0], O[0]],
                [O[1], A[1], B[1], D[1], O[1]], 'o-', color='#2E86AB', lw=2)
        ax.scatter(*P, c='#E67E22', s=50, zorder=5)
    ax.set_aspect('equal')
    ax.set_title('曲柄转一圈，偶联点画出封闭曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'fourbar.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
