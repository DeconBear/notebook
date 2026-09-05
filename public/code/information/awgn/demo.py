# -*- coding: utf-8 -*-
"""
=== 高斯信道：C = 1/2 log2(1+SNR) ===
左图容量曲线；右图 ±√P 加噪散点。
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

N_NOISE = 1.0  # 固定噪声方差，用 P 调 SNR


def capacity_bits(snr_lin):
    return 0.5 * np.log2(1.0 + snr_lin)


def snr_db_to_lin(db):
    return 10.0 ** (np.asarray(db, dtype=float) / 10.0)


def main():
    print('=== AWGN 容量 ===')
    for db in (0.0, 10.0, 20.0):
        c = capacity_bits(snr_db_to_lin(db))
        print(f'SNR={db:.0f} dB  C={c:.4f} bit / 实数维')

    dbs = np.linspace(-5.0, 25.0, 121)
    cs = capacity_bits(snr_db_to_lin(dbs))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(dbs, cs, color='#2980B9', lw=2)
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('C (bit)')
    axes[0].set_title(r'$C=\frac{1}{2}\log_2(1+\mathrm{SNR})$')
    axes[0].grid(True, alpha=0.3)

    for db, color in ((0.0, '#95A5A6'), (10.0, '#1ABC9C')):
        P = snr_db_to_lin(db) * N_NOISE
        amp = np.sqrt(P)
        bits = np.random.randint(0, 2, size=200)
        x = np.where(bits == 1, amp, -amp)
        y = x + np.random.normal(0.0, np.sqrt(N_NOISE), size=x.shape)
        axes[1].scatter(np.full_like(y, db, dtype=float) + 0.15 * np.random.randn(len(y)),
                        y, s=12, alpha=0.5, color=color, label=f'{db:.0f} dB')
    axes[1].set_xlabel('SNR (dB)（点稍作横向抖动）')
    axes[1].set_ylabel('Y')
    axes[1].set_title(r'二进制 $\pm\sqrt{P}$ + 高斯噪声')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'awgn_cap.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
