# -*- coding: utf-8 -*-
"""
=== GNN 练习 ===
实现对称归一化邻接（GCN 那一步）和按目标节点的 softmax（GAT 注意力）。
运行: python exercise.py
"""
import torch
import torch.nn.functional as F


def gcn_normalize(A: torch.Tensor) -> torch.Tensor:
    """
    输入无自环的对称邻接 A (N,N)。
    返回 D^{-1/2} (A+I) D^{-1/2}。
    """
    # TODO
    raise NotImplementedError


def dst_softmax(scores: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """
    scores, dst 长度都是 E。对每个目标节点 i，把 {scores[e]: dst[e]=i} 做 softmax。
    返回与 scores 同形状的 alpha。
    """
    # TODO
    raise NotImplementedError


def _check():
    A = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    S = gcn_normalize(A)
    assert S.shape == (3, 3)
    # A+I 的度: 节点1度=3（自环+两个邻居）, 0和2度=2
    assert torch.allclose(S, S.T, atol=1e-5)
    assert abs(float(S.sum()) - 3.0) < 0.6  # 归一化后每行和不必为1（对称归一化）
    dst = torch.tensor([0, 0, 1, 1, 1])
    scores = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    a = dst_softmax(scores, dst, 2)
    assert torch.allclose(a[:2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(a[2:], torch.full((3,), 1.0 / 3))
    print('练习通过。GCN 归一化与 GAT 的按 dst softmax 是两套最常用的聚合。')


if __name__ == '__main__':
    _check()
