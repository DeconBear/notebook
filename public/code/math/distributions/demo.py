# -*- coding: utf-8 -*-
"""
=== 常见分布 ===
1) 二项 vs 泊松（计数）
2) 指数 vs 高斯（连续密度）
运行: python demo.py
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


def binomial_pmf(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def gaussian_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))


def exponential_pdf(x, lam):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x >= 0
    out[m] = lam * np.exp(-lam * x[m])
    return out


def demo_discrete():
    n, p = 10, 0.3
    lam = n * p  # 3
    ks = np.arange(0, n + 1)
    binom = [binomial_pmf(int(k), n, p) for k in ks]
    poiss = [poisson_pmf(int(k), lam) for k in ks]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(ks - 0.15, binom, width=0.3, color='#2E86AB', label=fr'Binom(n={n},p={p})')
    ax.bar(ks + 0.15, poiss, width=0.3, color='#E8684A', label=fr'Poisson(λ={lam:g})')
    ax.set_xlabel('k')
    ax.set_ylabel('P(K=k)')
    ax.set_title('计数：n 大 p 小时二项像泊松')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'dist_discrete.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print(f'Binom P(K=3)={binomial_pmf(3, n, p):.4f}  Poisson P(K=3)={poisson_pmf(3, lam):.4f}')


def demo_continuous():
    xs = np.linspace(-3, 6, 400)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, gaussian_pdf(xs, 0.0, 1.0), color='#2E86AB', lw=2, label=r'$\mathcal{N}(0,1)$')
    ax.plot(xs, exponential_pdf(xs, 1.0), color='#C1666B', lw=2, label=r'Exp(λ=1)')
    ax.set_xlabel('x')
    ax.set_ylabel('密度')
    ax.set_title('连续：高斯对称；指数只活在 x≥0')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'dist_continuous.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print(f'N(0,1) φ(0)={gaussian_pdf(0.0, 0.0, 1.0):.4f}')


def main():
    print('=== 常见分布 ===')
    demo_discrete()
    demo_continuous()


if __name__ == '__main__':
    main()
