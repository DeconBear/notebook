# -*- coding: utf-8 -*-
"""
=== 神经编码演示：调谐曲线、群体向量、稀疏度 ===
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


def cosine_tuning(theta, pref, r_max=40.0, r0=5.0):
    return r0 + r_max * np.maximum(0.0, np.cos(theta - pref))


def main():
    prefs = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    thetas = np.linspace(0, 2 * np.pi, 180)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p in prefs[::2]:
        axes[0].plot(np.degrees(thetas), cosine_tuning(thetas, p), lw=1.4)
    axes[0].set_xlabel('刺激方向 (deg)')
    axes[0].set_ylabel('速率 (Hz)')
    axes[0].set_title('余弦调谐曲线')
    axes[0].grid(True, alpha=0.3)

    stim = np.deg2rad(40.0)
    rates = cosine_tuning(stim, prefs)
    spikes = np.random.poisson(rates * 0.1)  # 100 ms 窗口
    vec = np.sum(spikes[:, None] * np.stack([np.cos(prefs), np.sin(prefs)], axis=1), axis=0)
    est = np.arctan2(vec[1], vec[0])
    axes[1].scatter(np.cos(prefs) * rates, np.sin(prefs) * rates, c='#1a5276', s=40, label='细胞')
    axes[1].arrow(0, 0, np.cos(stim), np.sin(stim), color='#c0392b', width=0.03, label='真刺激')
    axes[1].arrow(0, 0, 0.8 * np.cos(est), 0.8 * np.sin(est), color='#27ae60', width=0.03, label='群体估计')
    axes[1].set_aspect('equal')
    axes[1].set_title(f'群体向量  真={np.degrees(stim):.0f}°  估={np.degrees(est):.0f}°')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'encoding_tuning_population.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)

    r = np.clip(np.random.gamma(2.0, 8.0, size=200), 0, None)
    spike_win = (np.random.rand(200) < r * 0.02).astype(float)
    fig2, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([0, 1], [np.mean(r < 0.05), 1.0 - np.mean(spike_win > 0)],
           color=['#5dade2', '#1a5276'], width=0.5)
    ax.set_xticks([0, 1], ['速率接近 0 的比例', '短窗内沉默细胞比例'])
    ax.set_ylim(0, 1)
    ax.set_title('稠密速率 vs 稀疏尖峰')
    ax.grid(True, axis='y', alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(_IMAGES_DIR, 'encoding_sparsity.png')
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print('保存', out2)
    print(f'群体估计误差 {abs(((np.degrees(est)-40+180)%360)-180):.1f} deg')


if __name__ == '__main__':
    main()
