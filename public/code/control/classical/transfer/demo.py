# -*- coding: utf-8 -*-
"""
=== 传递函数：二阶系统阶跃随 ζ 变 ===
G(s)=ωn²/(s²+2ζωn s+ωn²)，积 ÿ+2ζωnẏ+ωn²y = ωn² r。
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

WN = 3.0
DT = 0.005
T_END = 6.0
R = 1.0


def step_response(zeta, wn=WN):
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


def overshoot_pct(y):
    return float(max(0.0, np.max(y) - R) / R * 100.0)


def main():
    print('=== 传递函数：二阶阶跃 ===')
    zetas = [1.5, 1.0, 0.4, 0.15]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for z in zetas:
        t, y = step_response(z)
        ax.plot(t, y, lw=2, label=rf'$\zeta={z}$')
        print(f'ζ={z:4.2f}  超调={overshoot_pct(y):5.1f}%  末端={y[-1]:.4f}')
    ax.axhline(R, color='k', ls='--', alpha=0.4, label='r=1')
    ax.set_xlabel('时间 t')
    ax.set_ylabel('y(t)')
    ax.set_title(rf'G(s)=ωn²/(s²+2ζωn s+ωn²),  ωn={WN}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'tf_step.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
