# -*- coding: utf-8 -*-
"""
=== 信源编码：Huffman ===
{A:0.4, B:0.3, C:0.2, D:0.1}，比较平均码长 L 与熵 H。
运行: python demo.py
"""
import os
import heapq
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

PROBS = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}


def entropy_bits(probs):
    p = np.array(list(probs.values()), dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def huffman_codes(probs):
    """返回 symbol -> '01...'。堆元素 (p, tie, node)，叶子 node 是符号字符串。"""
    if len(probs) == 1:
        return {next(iter(probs)): '0'}
    heap = []
    for i, (s, p) in enumerate(probs.items()):
        heapq.heappush(heap, (p, i, s))
    parent = {}
    nid = len(probs)
    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        nid += 1
        node = f'#{nid}'
        parent[n1] = (node, '0')
        parent[n2] = (node, '1')
        heapq.heappush(heap, (p1 + p2, nid, node))
    codes = {}
    for s in probs:
        bits = []
        cur = s
        while cur in parent:
            par, b = parent[cur]
            bits.append(b)
            cur = par
        codes[s] = ''.join(reversed(bits)) or '0'
    return codes


def avg_len(probs, codes):
    return float(sum(probs[s] * len(codes[s]) for s in probs))


def main():
    print('=== Huffman ===')
    codes = huffman_codes(PROBS)
    h = entropy_bits(PROBS)
    L = avg_len(PROBS, codes)
    for s in sorted(PROBS, key=PROBS.get, reverse=True):
        print(f'{s}  p={PROBS[s]:.1f}  码={codes[s]}  长={len(codes[s])}')
    print(f'H={h:.4f} bit   L={L:.4f}   L-H={L-h:.4f}')
    assert L + 1e-9 >= h

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    syms = list(PROBS.keys())
    ax.bar([s + '\n' + codes[s] for s in syms],
           [PROBS[s] for s in syms], color='#1ABC9C', label='概率')
    ax.set_ylabel('p')
    ax.set_title(f'Huffman 码本  H={h:.3f}, L={L:.3f}')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_IMAGES_DIR, 'huffman_len.png')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print('保存', out)


if __name__ == '__main__':
    main()
