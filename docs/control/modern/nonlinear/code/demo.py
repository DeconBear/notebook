# -*- coding: utf-8 -*-
"""
=== 非线性：单摆能量 V 与阻尼 ===
mℓ² θ̈ + b θ̇ + mgℓ sinθ = 0。
V = ½ m (ℓω)² + m g ℓ (1-cosθ)，V̇ = -b ω²。
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

M = 1.0
L = 1.0
G = 9.8
DT = 0.01
T_END = 8.0
TH0, W0 = 1.2, 0.0


def energy(theta, omega):
    kinetic = 0.5 * M * (L * omega) ** 2
    potential = M * G * L * (1.0 - np.cos(theta))
    return kinetic + potential


def simulate(b):
    n = int(T_END / DT)
    th, w = TH0, W0
    ths, vs = [], []
    for _ in range(n):
        # ω̇ = -(g/ℓ) sinθ - (b/(m ℓ²)) ω
        wdot = -(G / L) * np.sin(th) - (b / (M * L ** 2)) * w
        w = w + DT * wdot
        th = th + DT * w
        ths.append(th)
        vs.append(energy(th, w))
    t = np.arange(n) * DT
    return t, np.array(ths), np.array(vs)


def main():
    print('=== 单摆李雅普诺夫 ===')
    print(f'初值 V={energy(TH0, W0):.4f}')
    t, th0, v0 = simulate(b=0.0)
    _, th1, v1 = simulate(b=0.6)
    print(f'无阻尼末端 V={v0[-1]:.4f}  θ={th0[-1]:.3f}')
    print(f'有阻尼末端 V={v1[-1]:.4f}  θ={th1[-1]:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, th0, color='#95A5A6', lw=2, label='b=0 无阻尼')
    axes[0].plot(t, th1, color='#E67E22', lw=2, label='b=0.6 有阻尼')
    axes[0].set_xlabel('时间 t')
    axes[0].set_ylabel(r'$\theta$ (rad)')
    axes[0].set_title('摆角')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, v0, color='#95A5A6', lw=2, label='b=0')
    axes[1].plot(t, v1, color='#E67E22', lw=2, label='b=0.6')
    axes[1].set_xlabel('时间 t')
    axes[1].set_ylabel('V')
    axes[1].set_title('能量：无阻尼守恒，有阻尼下降')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'lyapunov_pendulum.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
