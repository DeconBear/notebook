# -*- coding: utf-8 -*-
"""
=== 频域：二阶 G(jω) 的 Bode 图 ===
G(s)=ωn²/(s²+2ζωn s+ωn²)，s=jω 手算模与辐角。
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

WN = 4.0
ZETA = 0.3
KGAIN = 8.0  # 直流增益>1，幅频才会穿过 0 dB


def G_jw(omega):
    """向量化：omega 可以是数组。开环再乘 K，否则 |G(0)|=1，穿越频率退化。"""
    s = 1j * omega
    return KGAIN * (WN ** 2) / (s ** 2 + 2.0 * ZETA * WN * s + WN ** 2)


def mag_db(g):
    return 20.0 * np.log10(np.abs(g))


def phase_deg(g):
    return np.angle(g, deg=True)


def main():
    print('=== 频域 Bode ===')
    w = np.logspace(-1, 2, 400)
    g = G_jw(w)
    mag = mag_db(g)
    ph = phase_deg(g)
    # 从上方穿过 0 dB 的第一个网格点
    crossed = np.where((mag[:-1] > 0.0) & (mag[1:] <= 0.0))[0]
    idx = int(crossed[0]) if crossed.size else int(np.argmin(np.abs(mag)))
    wc = float(w[idx])
    pm = float(ph[idx] + 180.0)
    print(f'K={KGAIN}, ωn={WN}, ζ={ZETA}')
    print(f'0 dB 穿越 ωc≈{wc:.3f},  相位={ph[idx]:.1f}°,  PM≈{pm:.1f}°')

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)
    axes[0].semilogx(w, mag, color='#1ABC9C', lw=2)
    axes[0].axhline(0.0, color='k', ls='--', alpha=0.4)
    axes[0].axvline(wc, color='#E67E22', ls=':', alpha=0.8)
    axes[0].set_ylabel('幅值 (dB)')
    axes[0].set_title(rf'$KG(j\omega)$, $K={KGAIN}$, $\omega_n={WN}$, $\zeta={ZETA}$')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[1].semilogx(w, ph, color='#8E44AD', lw=2)
    axes[1].axhline(-180.0, color='k', ls='--', alpha=0.4)
    axes[1].axvline(wc, color='#E67E22', ls=':', alpha=0.8, label=rf'ωc≈{wc:.2f}')
    axes[1].set_xlabel(r'$\omega$ (rad/s)')
    axes[1].set_ylabel('相位 (度)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'bode.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
