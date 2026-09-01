# -*- coding: utf-8 -*-
"""
=== 概率与贝叶斯 ===
1) Beta-Binomial：抛硬币后验随数据更新
2) 一维/二维高斯可视化
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


def beta_pdf(theta, a, b, eps=1e-9):
    """不依赖 scipy 的 Beta 密度（仅用于画图归一化）。"""
    theta = np.clip(theta, eps, 1 - eps)
    unnorm = theta ** (a - 1) * (1 - theta) ** (b - 1)
    # 梯形积分归一化（兼容 NumPy 1.x / 2.x）
    trap = getattr(np, 'trapezoid', None) or np.trapz
    z = trap(unnorm, theta)
    return unnorm / z


def demo_bayes_coin():
    theta = np.linspace(0, 1, 400)
    # 先验 Beta(2, 2) —— 略偏好中间
    prior_a, prior_b = 2.0, 2.0
    # 观测：10 次里 7 次正面
    heads, trials = 7, 10
    post_a = prior_a + heads
    post_b = prior_b + (trials - heads)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(theta, beta_pdf(theta, prior_a, prior_b), label='先验 Beta(2,2)', lw=2)
    ax.plot(theta, beta_pdf(theta, post_a, post_b), label=f'后验 Beta({post_a:.0f},{post_b:.0f})', lw=2)
    ax.axvline(heads / trials, color='C3', ls='--', label='样本频率 0.7')
    ax.set_xlabel('θ（正面概率）')
    ax.set_ylabel('密度')
    ax.set_title('贝叶斯：看到 7/10 正面后，信念右移且变尖')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'prob_bayes_coin.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def demo_gaussian():
    x = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for sigma, c in [(0.5, 'C0'), (1.0, 'C1'), (2.0, 'C2')]:
        y = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        axes[0].plot(x, y, label=f'σ={sigma}', color=c)
    axes[0].set_title('一维高斯：σ 控制胖瘦')
    axes[0].legend()
    axes[0].set_xlabel('x')

    # 相关二维高斯
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    pts = np.random.multivariate_normal(mean, cov, size=400)
    axes[1].scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.4, c='#5B8FF9')
    axes[1].set_aspect('equal')
    axes[1].set_title('二维高斯：协方差让等高线倾斜')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'prob_gaussian.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


def main():
    print('=== 概率与贝叶斯 ===')
    demo_bayes_coin()
    demo_gaussian()


if __name__ == '__main__':
    main()
