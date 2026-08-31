# -*- coding: utf-8 -*-
"""wm04 exercise — TODO: compare more search heuristics."""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)
np.random.seed(42)


def episode(policy):
    goal, s, total = 6, 0, 0.0
    for _ in range(12):
        if policy == 'random':
            a = np.random.choice([-1, 0, 1])
        elif policy == 'greedy':
            a = 1 if s < goal else 0
        else:
            # TODO: 实现你自己的启发式
            a = 1
        s = int(np.clip(s + a, 0, goal))
        total += 1.0 if s == goal else -0.05
        if s == goal:
            break
    return total


def main():
    names = ['random', 'greedy', 'custom']
    vals = [np.mean([episode(n) for _ in range(100)]) for n in names]
    plt.figure(figsize=(5, 3.5))
    plt.bar(names, vals)
    plt.title('wm04 exercise')
    out = os.path.join(_IMAGES_DIR, 'exercise_result.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    main()
