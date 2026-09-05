# -*- coding: utf-8 -*-
"""
=== 回路演示：STDP 方向选择性 + 向量化 E–I LIF raster ===
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
np.random.seed(0)

A_PLUS, A_MINUS = 0.02, 0.022
TAU_PLUS, TAU_MINUS = 20.0, 20.0


class STDPSynapse:
    def __init__(self, w0):
        self.w = float(np.clip(w0, 0, 1))
        self.pre_tr = 0.0
        self.post_tr = 0.0

    def decay(self, dt):
        self.pre_tr *= np.exp(-dt / TAU_PLUS)
        self.post_tr *= np.exp(-dt / TAU_MINUS)

    def on_pre(self):
        self.w = float(np.clip(self.w - A_MINUS * self.post_tr, 0, 1))
        self.pre_tr += 1.0

    def on_post(self):
        self.w = float(np.clip(self.w + A_PLUS * self.pre_tr, 0, 1))
        self.post_tr += 1.0


def train_direction(preferred='LR', n_inputs=8, n_sweeps=30, w0=0.2):
    rng = np.random.default_rng(0)
    syns = [STDPSynapse(w0 + 0.05 * rng.normal()) for _ in range(n_inputs)]
    order = list(range(n_inputs)) if preferred == 'LR' else list(range(n_inputs - 1, -1, -1))
    events = []
    t0 = 20.0
    dt_pair, isi = 10.0, 100.0
    for s in range(n_sweeps):
        base = t0 + s * (n_inputs * dt_pair + isi)
        for k, idx in enumerate(order):
            events.append((base + k * dt_pair, 'pre', idx))
        events.append((base + (n_inputs - 1) * dt_pair + 2.0, 'post', -1))
    events.sort()
    t_prev = 0.0
    for t, kind, idx in events:
        dt = t - t_prev
        for syn in syns:
            syn.decay(dt)
        if kind == 'pre':
            syns[idx].on_pre()
        else:
            for syn in syns:
                syn.on_post()
        t_prev = t
    w = np.array([s.w for s in syns])
    corr = float(np.corrcoef(np.arange(n_inputs), w)[0, 1])
    if preferred == 'RL':
        corr = -corr
    return w, corr


def probe(weights, direction):
    n = len(weights)
    order = range(n) if direction == 'LR' else range(n - 1, -1, -1)
    return float(sum(weights[idx] * (k + 1) for k, idx in enumerate(order)))


def simulate_ei(n_e=40, n_i=10, t_ms=400.0, dt=0.2, seed=0):
    """向量化电流型 E–I LIF，避免逐细胞 Python 对象。"""
    rng = np.random.default_rng(seed)
    n = n_e + n_i
    is_e = np.zeros(n, dtype=bool)
    is_e[:n_e] = True
    W = np.zeros((n, n))
    p_ee = p_ei = p_ie = p_ii = 0.12
    for pre in range(n_e):
        for post in range(n_e):
            if pre != post and rng.random() < p_ee:
                W[post, pre] = 0.8
        for post in range(n_e, n):
            if rng.random() < p_ie:
                W[post, pre] = 1.2
    for pre in range(n_e, n):
        for post in range(n_e):
            if rng.random() < p_ei:
                W[post, pre] = -0.5
        for post in range(n_e, n):
            if pre != post and rng.random() < p_ii:
                W[post, pre] = -0.6
    tau, R, Vrest, Vth, Vreset, tref = 20.0, 20.0, -70.0, -50.0, -70.0, 2.0
    n_steps = int(round(t_ms / dt))
    V = np.full(n, Vrest)
    ref = np.zeros(n)
    spikes = np.zeros((n_steps, n), dtype=bool)
    I_base = np.where(is_e, 1.15, 0.75).astype(float)
    syn = np.zeros(n)
    for t_i in range(n_steps):
        I = I_base + rng.normal(0.0, 0.15, size=n) + syn
        syn[:] = 0.0
        active = ref <= 0
        V[active] += dt * (-(V[active] - Vrest) + R * I[active]) / tau
        V[~active] = Vreset
        ref = np.maximum(ref - dt, 0.0)
        fired = active & (V >= Vth)
        spikes[t_i] = fired
        if np.any(fired):
            syn += W[:, fired].sum(axis=1)
            V[fired] = Vreset
            ref[fired] = tref
    t = np.arange(n_steps) * dt
    return t, spikes, is_e


def main():
    w, score = train_direction('LR')
    print(f'LR 训练 selectivity={score:.3f}  probe LR/RL={probe(w,"LR"):.2f}/{probe(w,"RL"):.2f}')
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(np.arange(len(w)), w, color='#1a5276')
    ax.set_xlabel('输入位置（左 → 右）')
    ax.set_ylabel('权重')
    ax.set_title('STDP 后的方向选择性剖面（偏好 LR）')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'direction_weights.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    t, spikes, is_e = simulate_ei()
    rows, cols = np.where(spikes)
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(t[rows], cols, s=2, c=np.where(is_e[cols], '#1a5276', '#c0392b'), alpha=0.8)
    ax.axhline(is_e.sum() - 0.5, color='#888', lw=0.8, ls='--')
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('神经元（下=E，上=I）')
    ax.set_title('E–I LIF 网络 raster')
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'ei_raster.png')
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print('保存', out2)
    duration = (t[-1] - 100.0) / 1000.0
    start = int(100.0 / (t[1] - t[0]))
    rate = float(np.mean(spikes[start:].sum(axis=0) / duration))
    print(f'平均发放率 ≈ {rate:.1f} Hz')


if __name__ == '__main__':
    main()
