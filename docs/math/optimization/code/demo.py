# -*- coding: utf-8 -*-
"""
=== 优化与梯度 ===
二维碗状损失上对比大学习率 / 小学习率的梯度下降轨迹。
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


def loss(w):
    # 最小值在 (1, -0.5)
    return (w[0] - 1.0) ** 2 + 0.25 * (w[1] + 0.5) ** 2


def grad(w):
    return np.array([2 * (w[0] - 1.0), 0.5 * (w[1] + 0.5)])


def gd(w0, eta, steps=40):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    losses = [loss(w)]
    for _ in range(steps):
        w = w - eta * grad(w)
        traj.append(w.copy())
        losses.append(loss(w))
    return np.array(traj), np.array(losses)


def main():
    print('=== 优化与梯度 ===')
    w0 = np.array([-1.5, 2.0])
    traj_big, loss_big = gd(w0, eta=0.95, steps=25)
    traj_ok, loss_ok = gd(w0, eta=0.15, steps=40)

    # 等高线
    xs = np.linspace(-2, 2.5, 200)
    ys = np.linspace(-2, 2.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = (XX - 1.0) ** 2 + 0.25 * (YY + 0.5) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cs = axes[0].contour(XX, YY, ZZ, levels=20, cmap='Blues')
    axes[0].clabel(cs, inline=True, fontsize=7)
    axes[0].plot(traj_big[:, 0], traj_big[:, 1], 'o-', color='#E8684A', ms=3, label='η=0.95（过大）')
    axes[0].plot(traj_ok[:, 0], traj_ok[:, 1], 'o-', color='#5AD8A6', ms=3, label='η=0.15（合适）')
    axes[0].scatter([1], [-0.5], c='k', s=60, marker='*', label='最优')
    axes[0].set_title('参数平面上的轨迹')
    axes[0].legend()
    axes[0].set_xlabel('w1')
    axes[0].set_ylabel('w2')

    axes[1].semilogy(loss_big, label='η=0.95')
    axes[1].semilogy(loss_ok, label='η=0.15')
    axes[1].set_xlabel('迭代')
    axes[1].set_ylabel('损失（对数轴）')
    axes[1].set_title('损失下降曲线')
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'opt_gd_traj.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print('小学习率最终损失', float(loss_ok[-1]))
    print('大学习率最终损失', float(loss_big[-1]))


if __name__ == '__main__':
    main()
