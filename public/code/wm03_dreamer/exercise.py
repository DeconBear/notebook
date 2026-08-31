# -*- coding: utf-8 -*-
"""wm03 exercise — TODO: change reward location and re-plot policy bias curve."""
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


def todo_rollout_return(bias):
    # TODO: 把奖励放到中间格子，观察最优 bias 变化
    n = 8
    R = np.zeros(n)
    R[-1] = 1.0  # TODO: 改成 R[n//2] = 1.0
    s, G, disc = 0, 0.0, 1.0
    for _ in range(20):
        a = 1 if np.random.rand() < bias else -1
        s = int(np.clip(s + a, 0, n - 1))
        G += disc * R[s]
        disc *= 0.95
        if R[s] > 0:
            break
    return G


def main():
    biases = np.linspace(0.1, 0.9, 9)
    vals = [np.mean([todo_rollout_return(b) for _ in range(50)]) for b in biases]
    plt.figure(figsize=(6, 3.5))
    plt.plot(biases, vals, 'o-')
    plt.title('wm03 exercise')
    plt.grid(True, alpha=0.3)
    out = os.path.join(_IMAGES_DIR, 'exercise_result.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    main()
