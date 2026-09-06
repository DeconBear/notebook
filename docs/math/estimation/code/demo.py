# -*- coding: utf-8 -*-
"""
=== 数理统计与估计 ===
1) 伯努利 MLE = 样本均值；n 增大时 hat p 往真值挤
2) 高斯：MLE 方差除以 n，无偏方差除以 n-1
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


def bernoulli_mle(x):
    return float(np.mean(x))


def sample_variance(x, unbiased=True):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    s = np.sum((x - mu) ** 2)
    den = (x.size - 1) if unbiased else x.size
    return float(s / den)


def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    mu = float(x.mean())
    sigma2 = float(np.mean((x - mu) ** 2))
    return mu, sigma2


def demo_bernoulli():
    p_true = 0.7
    ns = [5, 20, 80]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, n in zip(axes, ns):
        hats = np.random.binomial(n, p_true, size=2000) / n
        ax.hist(hats, bins=20, color='#5B8FF9', edgecolor='white', density=True)
        ax.axvline(p_true, color='#C1666B', ls='--', lw=2, label=rf'真 $p={p_true}$')
        ax.set_title(f'n={n}')
        ax.set_xlabel(r'$\hat p$')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('密度')
    fig.suptitle('MLE：样本越多，估计越挤向真值', y=1.03)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'est_bernoulli.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('保存', out)
    coins = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    print('10 次硬币 MLE =', bernoulli_mle(coins))


def demo_gaussian_var():
    mu_true, sig2_true = 0.0, 4.0
    n = 8
    n_rep = 4000
    mle, unb = [], []
    for _ in range(n_rep):
        x = np.random.normal(mu_true, np.sqrt(sig2_true), size=n)
        _, s_mle = gaussian_mle(x)
        mle.append(s_mle)
        unb.append(sample_variance(x, unbiased=True))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(mle, bins=40, alpha=0.6, color='#2E86AB', density=True, label='MLE /n')
    ax.hist(unb, bins=40, alpha=0.6, color='#E8684A', density=True, label='无偏 /(n-1)')
    ax.axvline(sig2_true, color='k', ls='--', label=r'真 $\sigma^2=4$')
    ax.set_xlabel(r'$\widehat{\sigma}^2$')
    ax.set_title(f'n={n}：除以 n 的方差估计整体偏小')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'est_variance.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)
    print(f'E[MLE]={np.mean(mle):.3f}  E[unbiased]={np.mean(unb):.3f}  true={sig2_true}')


def main():
    print('=== 数理统计与估计 ===')
    demo_bernoulli()
    demo_gaussian_var()


if __name__ == '__main__':
    main()
