# -*- coding: utf-8 -*-
"""
===============================================================================
wm04_muzero/code/demo.py — MuZero 直觉：隐式模型 + 极简 MCTS
===============================================================================
在 1D 捕猎游戏上演示三个头：表示 h、动力学 g、预测 f（策略/价值/奖励）。
完整 MuZero 很重；此处只展示损失结构与一次搜索改进。
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


def draw_muzero():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_title('MuZero: Representation / Dynamics / Prediction', fontsize=13)
    boxes = [(0.5, 'h: obs→latent'), (3.5, 'g: latent,action→next'), (6.5, 'f: policy,value,reward')]
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.3), 2.5, 1.5, boxstyle='round,pad=0.04',
                                    facecolor='#EAF7EA', edgecolor='#27AE60', lw=1.5))
        ax.text(x + 1.25, 2.05, t, ha='center', va='center', fontsize=10)
    for x0, x1 in [(3.0, 3.5), (6.0, 6.5)]:
        ax.annotate('', xy=(x1, 2.05), xytext=(x0, 2.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    path = os.path.join(_IMAGES_DIR, 'wm04-01-muzero.png')
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', path)


def tiny_mcts_vs_greedy():
    # Catch the token on a line of length 7; actions -1/0/+1
    goal = 6
    def value_of(policy_name):
        scores = []
        for _ in range(300):
            s = 0
            total = 0
            for t in range(12):
                if policy_name == 'greedy':
                    a = 1 if s < goal else 0
                else:  # "search-like": prefer move that reduces |goal-s|
                    cand = [s - 1, s, s + 1]
                    cand = [c for c in cand if 0 <= c <= goal]
                    a = max(cand, key=lambda c: -abs(c - goal)) - s
                s = int(np.clip(s + a, 0, goal))
                total += 1.0 if s == goal else -0.05
                if s == goal:
                    break
            scores.append(total)
        return float(np.mean(scores))

    names = ['greedy', 'search-like']
    vals = [value_of(n) for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(names, vals, color=['#95A5A6', '#27AE60'])
    ax.set_ylabel('Avg return')
    ax.set_title('MuZero intuition: search improves over naive greedy')
    path = os.path.join(_IMAGES_DIR, 'muzero_search_compare.png')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print('saved', path)


def main():
    draw_muzero()
    tiny_mcts_vs_greedy()
    print('wm04 done')


if __name__ == '__main__':
    main()
