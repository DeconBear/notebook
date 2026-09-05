# -*- coding: utf-8 -*-
"""
=== DH 建模：平面 3R ===
标准 DH：每行 (a, alpha, d, theta)。平面臂 α=d=0，a=杆长，θ=关节。
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

A_LEN = [0.6, 0.5, 0.35]


def dh(a, alpha, d, theta):
    """一帧 A_i：绕 z 转 θ，沿 z 移 d，沿 x 移 a，绕 x 转 α。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_chain(thetas):
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    for a, th in zip(A_LEN, thetas):
        T = T @ dh(a, 0.0, 0.0, th)
        origins.append(T[:3, 3].copy())
    return np.array(origins), T


def main():
    print('=== DH 3R ===')
    th = np.array([0.4, -0.7, 0.5])
    pts, T = fk_chain(th)
    print('末端位置', pts[-1, :2])
    print('末端旋转 R=\n', np.round(T[:3, :3], 3))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(pts[:, 0], pts[:, 1], 'o-', color='#1a7f37', lw=3, ms=8)
    ax.scatter(pts[0, 0], pts[0, 1], c='k', s=40, zorder=5)
    ax.set_aspect('equal')
    ax.set_title('DH 链：三根平面杆')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'dh_3r.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
