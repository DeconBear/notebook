# -*- coding: utf-8 -*-
"""
=== 率失真：公平比特 + 汉明失真 R(D)=1-h2(D) ===
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


def h2(p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def R_D(D):
    D = np.asarray(D, dtype=float)
    scalar = D.ndim == 0
    D = np.atleast_1d(D)
    out = np.zeros_like(D, dtype=float)
    mask = D < 0.5
    out[mask] = 1.0 - h2(D[mask])
    return float(out[0]) if scalar else out


def main():
    print('=== 率失真 R(D)=1-h2(D) ===')
    for D in (0.0, 0.11, 0.25, 0.5):
        print(f'D={D:.2f}  R={R_D(D):.4f} bit')

    Ds = np.linspace(0.0, 0.5, 201)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(Ds, R_D(Ds), color='#8E44AD', lw=2, label=r'$R(D)=1-h_2(D)$')
    marks = [0.05, 0.11, 0.25]
    ax.scatter(marks, [R_D(d) for d in marks], color='#E67E22', zorder=3)
    for D in marks:
        ax.annotate(f'D={D}', (D, R_D(D)),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel('失真 D = P(重构不等于 X)')
    ax.set_ylabel('R(D) (bit)')
    ax.set_title('公平比特、汉明失真')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'rd_binary.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
