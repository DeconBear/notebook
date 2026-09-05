# -*- coding: utf-8 -*-
"""
=== 计算神经科学导论：尺度梯子 ===
画出「离子 → 细胞 → 回路 → 系统 → 行为」与本教程章节的对应。
运行: python demo.py
"""
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

LEVELS = [
    ('离子 / 通道', 'Nernst、门控', 'HH 电流项'),
    ('单细胞膜', '静息、动作电位', 'HH / LIF'),
    ('突触', 'EPSP / IPSP、可塑性', 'STDP'),
    ('回路', 'E–I、感受野', '方向选择性 · raster'),
    ('结构', '谁连谁', '连接组 / SONATA-lite'),
    ('与 AI', '启发 / 对齐 / 约束', 'NeuroAI'),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('计算神经科学：尺度梯子', fontsize=14, fontweight='bold', pad=12)
    colors = ['#d4e6f1', '#fdebd0', '#d5f5e3', '#fadbd8', '#e8daef', '#d6eaf8']
    for i, ((bio, q, model), c) in enumerate(zip(LEVELS, colors)):
        y = 5.6 - i * 0.9
        ax.add_patch(FancyBboxPatch((0.4, y), 3.2, 0.72, boxstyle='round,pad=0.02',
                                    facecolor=c, edgecolor='#333', linewidth=1.2))
        ax.text(2.0, y + 0.36, bio, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.add_patch(FancyBboxPatch((3.9, y), 2.8, 0.72, boxstyle='round,pad=0.02',
                                    facecolor='white', edgecolor='#888'))
        ax.text(5.3, y + 0.36, q, ha='center', va='center', fontsize=9)
        ax.add_patch(FancyBboxPatch((7.0, y), 2.6, 0.72, boxstyle='round,pad=0.02',
                                    facecolor='#f8f9f9', edgecolor='#1a5276'))
        ax.text(8.3, y + 0.36, model, ha='center', va='center', fontsize=9, color='#1a5276')
        if i < len(LEVELS) - 1:
            ax.annotate('', xy=(2.0, y - 0.08), xytext=(2.0, y - 0.16),
                        arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
    ax.text(2.0, 6.55, '生物尺度', ha='center', fontsize=10, color='#555')
    ax.text(5.3, 6.55, '典型问题', ha='center', fontsize=10, color='#555')
    ax.text(8.3, 6.55, '本教程模型', ha='center', fontsize=10, color='#555')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'neuro_scale_ladder.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
