# -*- coding: utf-8 -*-
"""
=== 信道编码：Hamming(7,4) 对 BSC ===
比较未编码 4 比特与 Hamming 译码后的误比特率。
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

N_TRIALS = 4000


def encode74(data):
    """data: (4,) 位，位置 3,5,6,7（0-index: 2,4,5,6）。"""
    c = np.zeros(7, dtype=int)
    c[2], c[4], c[5], c[6] = data
    c[0] = c[2] ^ c[4] ^ c[6]
    c[1] = c[2] ^ c[5] ^ c[6]
    c[3] = c[4] ^ c[5] ^ c[6]
    return c


def decode74(r):
    r = np.array(r, dtype=int).copy()
    s0 = r[0] ^ r[2] ^ r[4] ^ r[6]
    s1 = r[1] ^ r[2] ^ r[5] ^ r[6]
    s2 = r[3] ^ r[4] ^ r[5] ^ r[6]
    pos = s0 + 2 * s1 + 4 * s2
    if pos:
        r[pos - 1] ^= 1
    return r[[2, 4, 5, 6]]


def bsc(bits, p):
    flips = np.random.rand(*bits.shape) < p
    return bits ^ flips.astype(int)


def ber_uncoded(p):
    data = np.random.randint(0, 2, size=(N_TRIALS, 4))
    recv = bsc(data, p)
    return float(np.mean(recv != data))


def ber_hamming(p):
    data = np.random.randint(0, 2, size=(N_TRIALS, 4))
    n_err = 0
    n_bit = N_TRIALS * 4
    for i in range(N_TRIALS):
        code = encode74(data[i])
        recv = bsc(code, p)
        hat = decode74(recv)
        n_err += int(np.sum(hat != data[i]))
    return n_err / n_bit


def main():
    print('=== Hamming(7,4) vs 未编码 ===')
    ps = np.array([0.01, 0.03, 0.05, 0.08, 0.12, 0.18])
    u, h = [], []
    for p in ps:
        bu, bh = ber_uncoded(p), ber_hamming(p)
        u.append(bu)
        h.append(bh)
        print(f'p={p:.2f}  未编码 BER={bu:.4f}  Hamming BER={bh:.4f}')

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ps, u, 'o-', color='#95A5A6', lw=2, label='未编码 4 bit')
    ax.plot(ps, h, 's-', color='#1ABC9C', lw=2, label='Hamming(7,4)')
    ax.set_xlabel('BSC 翻转概率 p')
    ax.set_ylabel('数据误比特率')
    ax.set_title('短码：p 小时纠 1 错有用')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'hamming_ber.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
