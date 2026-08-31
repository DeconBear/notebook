# -*- coding: utf-8 -*-
"""wm08 exercise — TODO: adjust qualitative scores and replot."""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

PATHS = ['RSSM', 'Dreamer', 'MuZero', 'JEPA', 'Genie', 'Video', 'LLM']


def main():
    # TODO: 按你的理解改写分数
    scores = np.array([3, 4, 4, 3, 3, 2, 5], dtype=float)
    plt.figure(figsize=(8, 3.5))
    plt.bar(PATHS, scores, color='#8E44AD')
    plt.ylim(0, 5.5)
    plt.title('wm08 exercise: your ranking of semantic strength')
    out = os.path.join(_IMAGES_DIR, 'exercise_result.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    main()
