# -*- coding: utf-8 -*-
"""
=== 连接组练习 ===
从边列表重建邻接：W[target, source] = syn_weight。
运行: python exercise.py
"""
import numpy as np


def adjacency_from_edges(n, edges):
    """edges 为 dict 列表，含 source_node_id, target_node_id, syn_weight。"""
    # TODO
    raise NotImplementedError


def _check():
    edges = [
        {'source_node_id': 0, 'target_node_id': 1, 'syn_weight': 0.5},
        {'source_node_id': 2, 'target_node_id': 1, 'syn_weight': 0.2},
    ]
    W = adjacency_from_edges(3, edges)
    assert W.shape == (3, 3)
    assert abs(W[1, 0] - 0.5) < 1e-9
    assert abs(W[1, 2] - 0.2) < 1e-9
    assert W[0, 1] == 0
    print('通过：邻接矩阵重建正确。')


if __name__ == '__main__':
    _check()
