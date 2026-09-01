# -*- coding: utf-8 -*-
"""
=== 信息论精简 ===
熵、交叉熵、KL；展示 KL 不对称。
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


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def cross_entropy(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(q)))


def kl(p, q, eps=1e-12):
    return cross_entropy(p, q, eps) - entropy(p)


def main():
    print('=== 信息论精简 ===')
    fair = np.array([0.5, 0.5])
    biased = np.array([0.9, 0.1])
    print(f'公平硬币熵 H={entropy(fair):.4f} nat')
    print(f'偏置硬币熵 H={entropy(biased):.4f} nat')

    p = np.array([0.7, 0.3])
    qs = np.linspace(0.05, 0.95, 40)
    ce_list, kl_list, kl_rev = [], [], []
    for q1 in qs:
        q = np.array([q1, 1 - q1])
        ce_list.append(cross_entropy(p, q))
        kl_list.append(kl(p, q))
        kl_rev.append(kl(q, p))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(qs, ce_list, label='H(p,q) 交叉熵')
    axes[0].plot(qs, kl_list, label='KL(p‖q)')
    axes[0].axvline(p[0], color='k', ls='--', alpha=0.5, label='真实 p1=0.7')
    axes[0].set_xlabel('q 的第一类概率')
    axes[0].set_title('固定 p，扫描 q')
    axes[0].legend()

    axes[1].plot(qs, kl_list, label='KL(p‖q)')
    axes[1].plot(qs, kl_rev, label='KL(q‖p)')
    axes[1].set_xlabel('q 的第一类概率')
    axes[1].set_title('KL 不对称：两条曲线不同')
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'info_kl_curves.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    # 条形：公平 vs 偏置熵
    fig2, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(['公平 0.5/0.5', '偏置 0.9/0.1'], [entropy(fair), entropy(biased)], color=['#5B8FF9', '#E8684A'])
    ax.set_ylabel('熵 (nat)')
    ax.set_title('越确定，熵越小')
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'info_entropy_bars.png')
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print('保存', out2)


if __name__ == '__main__':
    main()
