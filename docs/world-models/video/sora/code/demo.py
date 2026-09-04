# -*- coding: utf-8 -*-
"""
===============================================================================
wm07_video_world_models/code/demo.py — 视频世界模型直觉：运动斑点预测
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


def draw_overview():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
    ax.set_title('Video Generative World Models (Sora / Cosmos class)', fontsize=12)
    for x, t, c in [(0.4, 'Video / frames', '#D6EAF8'), (3.4, 'DiT / AR / Diff.', '#FCF3CF'), (6.4, 'Future world', '#D5F5E3')]:
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.4, 1.4, boxstyle='round,pad=0.04', facecolor=c, edgecolor='#333', lw=1.2))
        ax.text(x + 1.2, 1.7, t, ha='center', va='center')
    for x0, x1 in [(2.8, 3.4), (5.8, 6.4)]:
        ax.annotate('', xy=(x1, 1.7), xytext=(x0, 1.7), arrowprops=dict(arrowstyle='->', lw=1.4))
    path = os.path.join(_IMAGES_DIR, 'wm07-01-video-wm.png')
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print('saved', path)


def moving_blob_predict():
    T, H, W = 12, 32, 32
    frames = np.zeros((T, H, W))
    for t in range(T):
        cy, cx = 8 + t, 6 + 2 * t
        yy, xx = np.ogrid[:H, :W]
        frames[t] = ((yy - cy) ** 2 + (xx - cx) ** 2) < 12
    # naive next-frame: copy + shift estimate
    pred = np.roll(frames[-2], shift=(1, 2), axis=(0, 1))
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    axes[0].imshow(frames[-2], cmap='gray'); axes[0].set_title('t')
    axes[1].imshow(frames[-1], cmap='gray'); axes[1].set_title('t+1 true')
    axes[2].imshow(pred, cmap='gray'); axes[2].set_title('t+1 naive pred')
    for ax in axes:
        ax.axis('off')
    path = os.path.join(_IMAGES_DIR, 'video_blob_predict.png')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print('saved', path)


def main():
    draw_overview()
    moving_blob_predict()
    print('wm07 done')


if __name__ == '__main__':
    main()
