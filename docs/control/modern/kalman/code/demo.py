# -*- coding: utf-8 -*-
"""
=== 卡尔曼：一维随机游走 ===
x_{k+1}=x_k+w，z=x+v。对比「直接用测量」与卡尔曼估计。
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

N = 80
Q = 0.04
R = 0.25


def simulate():
    x = 0.0
    xhat = 0.0
    P = 1.0
    xs, zs, xhs, Ps = [x], [], [xhat], [P]
    for _ in range(N):
        w = np.random.normal(0.0, np.sqrt(Q))
        x = x + w
        z = x + np.random.normal(0.0, np.sqrt(R))
        # 预测：A=1，无控制
        P_pred = P + Q
        xhat_pred = xhat
        kg = P_pred / (P_pred + R)
        xhat = xhat_pred + kg * (z - xhat_pred)
        P = (1.0 - kg) * P_pred
        xs.append(x)
        zs.append(z)
        xhs.append(xhat)
        Ps.append(P)
    return np.array(xs), np.array(zs), np.array(xhs), np.array(Ps)


def main():
    print('=== 一维卡尔曼 ===')
    xs, zs, xhs, Ps = simulate()
    rmse_z = float(np.sqrt(np.mean((zs - xs[1:]) ** 2)))
    rmse_f = float(np.sqrt(np.mean((xhs[1:] - xs[1:]) ** 2)))
    print(f'测量 RMSE={rmse_z:.3f}  滤波 RMSE={rmse_f:.3f}')

    t = np.arange(len(xs))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, xs, color='#2C3E50', lw=2, label='真值 x')
    axes[0].scatter(t[1:], zs, s=14, color='#95A5A6', alpha=0.8, label='测量 z')
    axes[0].plot(t, xhs, color='#27AE60', lw=2, label='卡尔曼估计')
    axes[0].set_xlabel('步 k')
    axes[0].set_ylabel('位置')
    axes[0].set_title('随机游走')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, Ps, color='#8E44AD', lw=2)
    axes[1].set_xlabel('步 k')
    axes[1].set_ylabel('P')
    axes[1].set_title('方差：预测变大、更新变小')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'kf_1d.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
