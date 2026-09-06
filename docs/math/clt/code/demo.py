# -*- coding: utf-8 -*-
"""
=== 大数定律与中心极限 ===
1) 抛硬币：样本均值随 n 贴向 p（LLN）
2) 指数分布本身不对称，但 bar X 在 n=1,5,30 时越来越像高斯（CLT）
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


def demo_lln():
    p = 0.6
    n = 400
    x = np.random.binomial(1, p, size=n)
    running = np.cumsum(x) / np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(running, color='#2E86AB', lw=1.5, label=r'$\bar X_n$')
    ax.axhline(p, color='#C1666B', ls='--', label=fr'真 $p={p}$')
    ax.set_xlabel('n')
    ax.set_ylabel('样本均值')
    ax.set_title('大数定律：抛得越多，均值越老实')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'clt_lln.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print(f'最后 bar X = {running[-1]:.3f}')


def demo_clt():
    lam = 1.0
    mu, sig2 = 1.0 / lam, 1.0 / (lam ** 2)
    ns = [1, 5, 30]
    n_rep = 4000
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=False)
    xs = np.linspace(0, 4, 200)

    def gauss(x, m, s):
        z = (x - m) / s
        return np.exp(-0.5 * z ** 2) / (s * np.sqrt(2 * np.pi))

    for ax, n in zip(axes, ns):
        means = np.random.exponential(1.0 / lam, size=(n_rep, n)).mean(axis=1)
        ax.hist(means, bins=35, density=True, color='#5B8FF9', edgecolor='white', alpha=0.85)
        s = np.sqrt(sig2 / n)
        ax.plot(xs, gauss(xs, mu, s), color='#C1666B', lw=2, label='CLT 高斯')
        ax.set_xlim(0, 4)
        ax.set_title(f'n={n}')
        ax.set_xlabel(r'$\bar X$')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('密度')
    fig.suptitle(r'中心极限：Exp(1) 的均值变成铃铛', y=1.03)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'clt_hist.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('保存', out)
    print(f'n=30 时样本均值的经验方差={means.var():.4f}（理论 {sig2/30:.4f}）')


def main():
    print('=== 大数定律与中心极限 ===')
    demo_lln()
    demo_clt()


if __name__ == '__main__':
    main()
