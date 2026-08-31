# -*- coding: utf-8 -*-
"""wm02 exercise — TODO: plot longer imagination horizon error."""
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


def main():
    # 复合误差随 horizon 增长的玩具模型
    H = np.arange(1, 21)
    # TODO: 换成你从 demo 测得的真实曲线
    err = 0.02 * (np.exp(0.15 * H) - 1)
    plt.figure(figsize=(6, 3.5))
    plt.plot(H, err, 'o-')
    plt.xlabel('imagination horizon')
    plt.ylabel('rollout error (toy)')
    plt.title('wm02 exercise')
    plt.grid(True, alpha=0.3)
    out = os.path.join(_IMAGES_DIR, 'exercise_result.png')
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print('saved', out)


if __name__ == '__main__':
    main()
