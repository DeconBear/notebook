# -*- coding: utf-8 -*-
"""
=== MoE 练习 ===
实现 topk_gate：对路由概率做 Top-k 并重归一化。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def topk_gate(probs, k):
    """
    probs: (N,) 路由 Softmax 概率
    返回 (gate, indices)：
      - indices: 被选中的专家下标，形状 (k,)，按原 probs 从大到小
      - gate: 同形状 (k,)，对应概率重归一化后的权重
    """
    # TODO
    raise NotImplementedError


def _check():
    p = np.array([0.05, 0.40, 0.10, 0.45])
    gate, idx = topk_gate(p, 2)
    assert set(idx.tolist()) == {1, 3}
    assert abs(gate.sum() - 1.0) < 1e-6
    # 0.40 与 0.45 归一化
    assert abs(gate[list(idx).index(3)] - 0.45 / 0.85) < 1e-6
    print('通过：Top-k 门控正确。')


if __name__ == '__main__':
    _check()
