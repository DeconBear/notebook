# -*- coding: utf-8 -*-
"""
=== STDP 演示：指数窗 + 因果配对轨迹 ===
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

A_PLUS, A_MINUS = 0.01, 0.012
TAU_PLUS, TAU_MINUS = 20.0, 20.0
W_MIN, W_MAX = 0.0, 1.0


def stdp_dw(dt):
    if dt > 0:
        return A_PLUS * np.exp(-dt / TAU_PLUS)
    if dt < 0:
        return -A_MINUS * np.exp(dt / TAU_MINUS)
    return 0.0


def pairwise_update(w, dt):
    return float(np.clip(w + stdp_dw(dt), W_MIN, W_MAX))


class STDPSynapse:
    def __init__(self, w0=0.5):
        self.w = w0
        self.pre_tr = 0.0
        self.post_tr = 0.0

    def decay(self, dt):
        self.pre_tr *= np.exp(-dt / TAU_PLUS)
        self.post_tr *= np.exp(-dt / TAU_MINUS)

    def on_pre(self):
        self.w = float(np.clip(self.w - A_MINUS * self.post_tr, W_MIN, W_MAX))
        self.pre_tr += 1.0

    def on_post(self):
        self.w = float(np.clip(self.w + A_PLUS * self.pre_tr, W_MIN, W_MAX))
        self.post_tr += 1.0


def main():
    dts = np.linspace(-80, 80, 401)
    dw = np.array([stdp_dw(d) for d in dts])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(dts, dw, color='#1a5276', lw=2)
    ax.axhline(0, color='#888', lw=0.8)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_xlabel(r'$\Delta t = t_{\mathrm{post}}-t_{\mathrm{pre}}$ (ms)')
    ax.set_ylabel(r'$\Delta w$')
    ax.set_title('pairwise STDP 学习窗')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'stdp_window.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    syn = STDPSynapse(0.4)
    dt_ms = 0.1
    t_end = 400.0
    n = int(t_end / dt_ms)
    W = np.empty(n)
    pre_times = set(int(round(x / dt_ms)) for x in np.arange(20, 380, 25))
    post_times = set(int(round((x + 10.0) / dt_ms)) for x in np.arange(20, 380, 25))
    for i in range(n):
        syn.decay(dt_ms)
        if i in pre_times:
            syn.on_pre()
        if i in post_times:
            syn.on_post()
        W[i] = syn.w
    t = np.arange(n) * dt_ms
    fig2, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t, W, color='#bc4c00', lw=2)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('权重')
    ax.set_title(r'重复因果配对 $\Delta t=+10$ ms')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'stdp_pair_trace.png')
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print('保存', out2)
    print('终值 w=', syn.w, '  窗上 LTP 例', pairwise_update(0.5, 10.0))


if __name__ == '__main__':
    main()
