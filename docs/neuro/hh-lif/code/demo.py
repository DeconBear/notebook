# -*- coding: utf-8 -*-
"""
=== HH 与 LIF 演示 ===
枪乌贼轴突 HH（前向欧拉）+ 带不应期的电流型 LIF；输出轨迹与 I–f。
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


def _alpha_n(V):
    x = 10.0 - (V + 65.0)
    if abs(x) < 1e-6:
        return 0.1
    return 0.01 * x / (np.exp(x / 10.0) - 1.0)


def _beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def _alpha_m(V):
    x = 25.0 - (V + 65.0)
    if abs(x) < 1e-6:
        return 1.0
    return 0.1 * x / (np.exp(x / 10.0) - 1.0)


def _beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V):
    return 1.0 / (np.exp((30.0 - (V + 65.0)) / 10.0) + 1.0)


def steady_gates(V):
    an, bn = _alpha_n(V), _beta_n(V)
    am, bm = _alpha_m(V), _beta_m(V)
    ah, bh = _alpha_h(V), _beta_h(V)
    return an / (an + bn), am / (am + bm), ah / (ah + bh)


class HodgkinHuxley:
    """经典 HH，单位：mV, ms, uA/cm^2, mS/cm^2, uF/cm^2。"""

    def __init__(self, dt=0.01):
        self.dt = dt
        self.C_m, self.g_Na, self.g_K, self.g_L = 1.0, 120.0, 36.0, 0.3
        self.E_Na, self.E_K, self.E_L = 50.0, -77.0, -54.387
        self.V_rest = -65.0
        self.reset()

    def reset(self):
        self.V = self.V_rest
        self.n, self.m, self.h = steady_gates(self.V)

    def step(self, I_ext):
        V, n, m, h, dt = self.V, self.n, self.m, self.h, self.dt
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        self.V = V + dt * (I_ext - I_Na - I_K - I_L) / self.C_m
        self.n = n + dt * (_alpha_n(V) * (1.0 - n) - _beta_n(V) * n)
        self.m = m + dt * (_alpha_m(V) * (1.0 - m) - _beta_m(V) * m)
        self.h = h + dt * (_alpha_h(V) * (1.0 - h) - _beta_h(V) * h)
        return self.V

    def simulate(self, I):
        self.reset()
        V = np.empty(len(I))
        for i, cur in enumerate(I):
            V[i] = self.step(float(cur))
        t = np.arange(len(I)) * self.dt
        return t, V

    def f_i_curve(self, currents, t_ms=300.0, warmup=80.0):
        n = int(round(t_ms / self.dt))
        rates = []
        for I0 in currents:
            t, V = self.simulate(np.full(n, I0))
            mask = t >= warmup
            above = V[mask] >= 0.0
            nspk = int(np.sum((~above[:-1]) & above[1:]))
            rates.append(nspk / ((t_ms - warmup) / 1000.0))
        return np.asarray(rates)


class LeakyIntegrateFire:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.tau, self.R = 20.0, 20.0
        self.V_rest = self.V_reset = -70.0
        self.V_th = -50.0
        self.t_ref = 2.0
        self.reset()

    def reset(self):
        self.V = self.V_rest
        self.ref_left = 0.0

    def step(self, I_ext):
        if self.ref_left > 0:
            self.ref_left -= self.dt
            self.V = self.V_reset
            return self.V, False
        self.V = self.V + self.dt * (-(self.V - self.V_rest) + self.R * I_ext) / self.tau
        if self.V >= self.V_th:
            self.V = self.V_reset
            self.ref_left = self.t_ref
            return self.V_reset, True
        return self.V, False

    def simulate(self, I):
        self.reset()
        V = np.empty(len(I))
        spk = np.zeros(len(I), dtype=bool)
        for i, cur in enumerate(I):
            V[i], spk[i] = self.step(float(cur))
        t = np.arange(len(I)) * self.dt
        return t, V, spk

    def f_i_curve(self, currents, t_ms=600.0, warmup=100.0):
        n = int(round(t_ms / self.dt))
        rates = []
        for I0 in currents:
            t, V, spk = self.simulate(np.full(n, I0))
            nspk = int(np.sum(spk[t >= warmup]))
            rates.append(nspk / ((t_ms - warmup) / 1000.0))
        return np.asarray(rates)

    @staticmethod
    def rheobase():
        return (-50.0 - (-70.0)) / 20.0


def _save_trace(t, V, title, name):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, V, color='#1a5276', lw=1.4)
    ax.set_xlabel('时间 (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, name)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def _save_fi(I, rates, title, name, xlabel):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(I, rates, 'o-', color='#bc4c00', lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('发放率 (Hz)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, name)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def main():
    hh = HodgkinHuxley()
    n = int(50.0 / hh.dt)
    I = np.zeros(n)
    I[int(10 / hh.dt):int(40 / hh.dt)] = 10.0
    t, V = hh.simulate(I)
    _save_trace(t, V, 'Hodgkin–Huxley：阶跃电流', 'hh_trace.png')
    print(f'HH 峰值 {V.max():.1f} mV')
    currents = np.linspace(0, 18, 10)
    _save_fi(currents, hh.f_i_curve(currents), 'HH I–f', 'hh_fi.png', r'$I_{\mathrm{ext}}$ ($\mu$A/cm$^2$)')

    lif = LeakyIntegrateFire()
    I_l = np.zeros(int(200 / lif.dt))
    I_l[int(20 / lif.dt):] = 1.5
    t, V, spk = lif.simulate(I_l)
    _save_trace(t, V, 'LIF：恒定电流', 'lif_trace.png')
    Ig = np.linspace(0, 3.0, 13)
    _save_fi(Ig, lif.f_i_curve(Ig), 'LIF I–f', 'lif_fi.png', r'$I$ (nA)')
    print(f'LIF 尖峰数 {int(spk.sum())}  rheobase={lif.rheobase():.3f} nA')


if __name__ == '__main__':
    main()
