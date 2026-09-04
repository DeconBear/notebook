# -*- coding: utf-8 -*-
"""
===============================================================================
wm08_llm_world_model/code/demo.py — LLM 世界模型直觉 + 路径对比图
===============================================================================
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['axes.unicode_minus'] = False
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)

PATHS = ['RSSM', 'Dreamer', 'MuZero', 'JEPA', 'Genie', 'Video', 'LLM']
# rough pedagogical scores 1-5
SCORES = {
    'sample_eff': [4, 5, 4, 4, 3, 2, 3],
    'controllability': [4, 4, 5, 3, 5, 2, 3],
    'semantics': [2, 2, 2, 4, 3, 4, 5],
    'compute': [4, 3, 3, 3, 2, 1, 2],  # higher = cheaper
}


def draw_compare():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(PATHS))
    w = 0.18
    for i, (k, vals) in enumerate(SCORES.items()):
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(PATHS)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel('Pedagogical score (1-5)')
    ax.set_title('World Model Paths — qualitative comparison')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, axis='y', alpha=0.3)
    path = os.path.join(_IMAGES_DIR, 'wm08-01-path-compare.png')
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print('saved', path)


def tiny_text_world():
    # state as short tokens; bigram transition counts
    corpus = [
        'room key door treasure'.split(),
        'room monster fight flee'.split(),
        'room key door monster fight'.split(),
        'room door locked need key'.split(),
    ]
    from collections import Counter, defaultdict
    trans = defaultdict(Counter)
    for seq in corpus:
        for a, b in zip(seq, seq[1:]):
            trans[a][b] += 1
    # rollout from 'room'
    s = 'room'
    traj = [s]
    for _ in range(5):
        opts = trans[s]
        if not opts:
            break
        # sample
        items, counts = zip(*opts.items())
        p = np.array(counts, dtype=float); p /= p.sum()
        s = np.random.choice(items, p=p)
        traj.append(s)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis('off')
    ax.set_title('Tiny LLM-as-WM: bigram state rollout')
    ax.text(0.5, 0.5, ' → '.join(traj), ha='center', va='center', fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#F5EEF8', edgecolor='#8E44AD'))
    path = os.path.join(_IMAGES_DIR, 'llm_text_rollout.png')
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('saved', path, 'traj=', traj)


def main():
    draw_compare()
    tiny_text_world()
    print('wm08 done')


if __name__ == '__main__':
    main()
