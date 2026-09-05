# -*- coding: utf-8 -*-
"""
=== 神经元：EPSP 叠加越过阈值 ===
用指数衰减的突触后电位示意「分级电位局部运算」。
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
DT = 0.1
TAU = 8.0
V_REST = -70.0
V_TH = -54.0


def psp(t, t0, amp, tau=TAU):
    x = t - t0
    out = np.zeros_like(t)
    m = x >= 0
    out[m] = amp * np.exp(-x[m] / tau)
    return out


def main():
    t = np.arange(0, 80, DT)
    arrivals = [8.0, 14.0, 19.0, 36.0, 41.0]
    amps = [6.5, 7.0, 6.8, -4.0, 8.0]
    v = np.full_like(t, V_REST)
    for t0, a in zip(arrivals, amps):
        v = v + psp(t, t0, a)
    spiked = np.any(v >= V_TH)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t, v, color='#1a5276', lw=2, label='膜电位（示意）')
    ax.axhline(V_TH, color='#c0392b', ls='--', label='阈值')
    ax.axhline(V_REST, color='#7f8c8d', ls=':', label='静息')
    for t0, a in zip(arrivals, amps):
        ax.axvline(t0, color='#27ae60' if a > 0 else '#8e44ad', alpha=0.35)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('mV')
    ax.set_title('EPSP 叠加 vs 一次 IPSP（教学示意）')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'neuron_integrate_fire.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print('是否越过阈值（示意）:', spiked, '  峰值', float(v.max()))


if __name__ == '__main__':
    main()
