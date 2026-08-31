# -*- coding: utf-8 -*-
"""
=== 交互/3D 直觉：显式动作 vs 从轨迹发现的潜动作 ===
5x5 格子世界。显式动作=上下左右；潜动作=对状态差分做 k-means 式离散化。
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

H = W = 5
ACTIONS = {
    0: np.array([-1, 0]),
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([0, 1]),
}


def step(pos, a):
    nxt = np.clip(pos + ACTIONS[a], 0, H - 1)
    return nxt


def collect_demos(n=200, T=12):
    trajs = []
    for _ in range(n):
        pos = np.array([np.random.randint(H), np.random.randint(W)])
        seq = [pos.copy()]
        acts = []
        for _t in range(T):
            a = np.random.randint(4)
            pos = step(pos, a)
            seq.append(pos.copy())
            acts.append(a)
        trajs.append((np.array(seq), np.array(acts)))
    return trajs


def discover_latent_actions(trajs, k=4):
    """用状态差分聚类，模拟『从视频变化里发现潜动作』。"""
    deltas = []
    for seq, _ in trajs:
        d = seq[1:] - seq[:-1]
        deltas.append(d)
    deltas = np.concatenate(deltas, axis=0).astype(float)
    # 简单 k-means
    centers = deltas[np.random.choice(len(deltas), k, replace=False)]
    for _ in range(20):
        dist = ((deltas[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = dist.argmin(axis=1)
        for j in range(k):
            if np.any(lab == j):
                centers[j] = deltas[lab == j].mean(axis=0)
    return centers, lab


def rollout_explicit(start, actions):
    pos = start.copy()
    path = [pos.copy()]
    for a in actions:
        pos = step(pos, int(a))
        path.append(pos.copy())
    return np.array(path)


def rollout_latent(start, latent_ids, centers):
    pos = start.astype(float)
    path = [pos.copy()]
    for lid in latent_ids:
        pos = np.clip(pos + centers[int(lid)], 0, H - 1)
        path.append(pos.copy())
    return np.array(path)


def main():
    print('=== 显式动作 vs 潜动作 ===')
    trajs = collect_demos()
    centers, _ = discover_latent_actions(trajs)
    print('发现的潜动作中心（格子差分）:\n', np.round(centers, 3))

    start = np.array([2, 2])
    explicit_acts = [3, 3, 1, 1, 2]
    # 把显式动作映射到最近潜中心
    latent_ids = []
    for a in explicit_acts:
        d = ACTIONS[a].astype(float)
        lid = np.argmin(((centers - d) ** 2).sum(axis=1))
        latent_ids.append(lid)

    p_exp = rollout_explicit(start, explicit_acts)
    p_lat = rollout_latent(start, latent_ids, centers)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, path, title in [
        (axes[0], p_exp, '显式动作（键位）'),
        (axes[1], p_lat, '潜动作（从差分聚类）'),
    ]:
        grid = np.zeros((H, W))
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1)
        ax.plot(path[:, 1], path[:, 0], 'o-', color='C0')
        ax.scatter(path[0, 1], path[0, 0], c='C2', s=80, label='起')
        ax.scatter(path[-1, 1], path[-1, 0], c='C3', s=80, label='终')
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(range(W))
        ax.set_yticks(range(H))
    fig.suptitle('路径二直觉：动作接口可以是显式的，也可以是学出来的')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'scene3d_action_interfaces.png')
    fig.savefig(out, dpi=140)
    print('保存', out)


if __name__ == '__main__':
    main()
