# -*- coding: utf-8 -*-
"""
=== 状态空间练习 ===
实现 ctrb_2(A, B)：两状态单输入的 [B, AB]，形状 (2, 2)。
运行: python exercise.py
"""
import numpy as np


def ctrb_2(A, B):
    """能控性矩阵 [B, AB]。B 为 (2,1) 列向量。"""
    # TODO
    raise NotImplementedError


def _check():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = ctrb_2(A, B)
    assert C.shape == (2, 2)
    assert np.allclose(C, np.array([[0.0, 1.0], [1.0, 0.0]]))
    print('通过：能控性矩阵正确。')


if __name__ == '__main__':
    _check()
