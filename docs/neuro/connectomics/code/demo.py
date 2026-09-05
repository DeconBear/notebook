# -*- coding: utf-8 -*-
"""
=== 连接组演示：SONATA-lite 玩具图与邻接矩阵 ===
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
np.random.seed(0)


def make_toy_connectome(n=20, p=0.15, seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            if rng.random() < p:
                edges.append({
                    'source_node_id': int(src),
                    'target_node_id': int(dst),
                    'syn_weight': float(rng.uniform(0.1, 1.0)),
                    'delay_ms': float(rng.choice([1.0, 2.0, 5.0])),
                })
    return {
        'format': 'toy-sonata-lite',
        'version': '0.1',
        'nodes': [{'node_id': i, 'node_type': 'LIF'} for i in range(n)],
        'edges': edges,
        'meta': {'n_neurons': n, 'p_connect': p, 'seed': seed},
    }


def adjacency(data):
    n = int(data['meta']['n_neurons'])
    W = np.zeros((n, n))
    for e in data['edges']:
        W[e['target_node_id'], e['source_node_id']] = e['syn_weight']
    return W


def main():
    data = make_toy_connectome()
    W = adjacency(data)
    print('边数', len(data['edges']),
          '密度', np.count_nonzero(W) / (W.shape[0] * (W.shape[0] - 1)))
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(W, cmap='viridis', vmin=0)
    ax.set_xlabel('突触前')
    ax.set_ylabel('突触后')
    ax.set_title('玩具连接组邻接矩阵')
    fig.colorbar(im, ax=ax, fraction=0.046, label='权重')
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'toy_adjacency.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
