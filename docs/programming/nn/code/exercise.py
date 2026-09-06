# -*- coding: utf-8 -*-
"""
=== torch.nn 练习 ===
1) 用 Sequential 搭 Linear-ReLU-Linear
2) 对 GRUCell 走一步，检查 shape
运行: python exercise.py
"""
import torch
import torch.nn as nn


def make_mlp(d_in: int, d_hid: int, d_out: int) -> nn.Module:
    """返回 nn.Sequential：Linear → ReLU → Linear，无多余层。"""
    # TODO
    raise NotImplementedError


def gru_one_step(input_size: int, hidden_size: int, batch: int):
    """
    造一颗 GRUCell，零输入、零隐状态走一步。
    返回 (gru实例, h_t)，h_t.shape == (batch, hidden_size)。
    """
    # TODO
    raise NotImplementedError


def _check():
    m = make_mlp(5, 7, 3)
    assert isinstance(m, nn.Sequential)
    y = m(torch.zeros(2, 5))
    assert tuple(y.shape) == (2, 3)
    gru, h = gru_one_step(6, 8, 4)
    assert isinstance(gru, nn.GRUCell)
    assert tuple(h.shape) == (4, 8)
    h2 = gru(torch.zeros(4, 6), h)
    assert id(gru)  # 同一实例可再走一步
    assert tuple(h2.shape) == (4, 8)
    print('练习通过。Sequential 是 MLP；GRUCell 是一步机器，h 是返回值。')


if __name__ == '__main__':
    _check()
