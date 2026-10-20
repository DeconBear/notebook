# -*- coding: utf-8 -*-
"""wm07 exercise — TODO: change blob velocity and compare prediction error."""
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
    # TODO: 修改 dy, dx 并计算 naive shift 预测的 MSE
    H = W = 32
    dy, dx = 1, 2
    yy, xx = np.ogrid[:H, :W]
    f0 = ((yy - 10) ** 2 + (xx - 8) ** 2) < 12
    f1 = ((yy - 10 - dy) ** 2 + (xx - 8 - dx) ** 2) < 12
    pred = np.roll(f0, shift=(dy, dx), axis=(0, 1))
    mse = float(np.mean((pred.astype(float) - f1.astype(float)) ** 2))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(pred.astype(float) - f1.astype(float), cmap='bwr')
    ax.set_title(f'error map mse={mse:.4f}')
    out = os.path.join(_IMAGES_DIR, 'exercise_result.png')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close()
    print('saved', out, 'mse', mse)


if __name__ == '__main__':
    main()
