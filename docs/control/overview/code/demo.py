# -*- coding: utf-8 -*-
"""
=== 控制论导论：开环 vs 闭环，以及阻尼比 ===
左图：质量-弹簧-阻尼，开环给稳态力 vs 比例反馈。
右图：二阶标准形四个 ζ 的单位阶跃。
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

M, C, KSPRING = 1.0, 0.4, 2.0
DT = 0.01
T_END = 8.0
R = 1.0
KP = 8.0
WN = 2.0


def mass_spring(open_loop=True):
    n = int(T_END / DT)
    x, v = 0.0, 0.0
    xs, ts = [], []
    for i in range(n):
        if open_loop:
            u = R * KSPRING
        else:
            u = KP * (R - x)
        a = (u - C * v - KSPRING * x) / M
        v = v + DT * a
        x = x + DT * v
        xs.append(x)
        ts.append(i * DT)
    return np.array(ts), np.array(xs)


def second_order_step(zeta, wn=WN):
    """ÿ + 2ζωn ẏ + ωn² y = ωn² r，半隐式欧拉。"""
    n = int(T_END / DT)
    y, v = 0.0, 0.0
    ys = []
    for _ in range(n):
        a = wn ** 2 * (R - y) - 2.0 * zeta * wn * v
        v = v + DT * a
        y = y + DT * v
        ys.append(y)
    t = np.arange(n) * DT
    return t, np.array(ys)


def main():
    print('=== 控制论导论 ===')
    t, x_ol = mass_spring(open_loop=True)
    _, x_cl = mass_spring(open_loop=False)
    print(f'开环末端 x={x_ol[-1]:.4f}  闭环(P)末端 x={x_cl[-1]:.4f}  参考 r={R}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(t, x_ol, color='#95A5A6', lw=2, label='开环 u=kr')
    axes[0].plot(t, x_cl, color='#1ABC9C', lw=2, label=f'闭环 P  kp={KP}')
    axes[0].axhline(R, color='k', ls='--', alpha=0.4, label='参考 r=1')
    axes[0].set_xlabel('时间 t')
    axes[0].set_ylabel('位置 x')
    axes[0].set_title('开环 vs 比例闭环')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    zetas = [(1.4, '过阻尼 ζ=1.4'), (0.5, '欠阻尼 ζ=0.5'),
             (0.0, '无阻尼 ζ=0'), (-0.15, '负阻尼 ζ=-0.15')]
    for z, name in zetas:
        tt, y = second_order_step(z)
        axes[1].plot(tt, y, lw=2, label=name)
        print(f'{name:16s}  末端 y={y[-1]:.3f}  峰值={y.max():.3f}')
    axes[1].axhline(R, color='k', ls='--', alpha=0.4)
    axes[1].set_xlabel('时间 t')
    axes[1].set_ylabel('输出 y')
    axes[1].set_title(r'同一 ωn，只改 ζ')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 2.4)

    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'overview_loop.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
