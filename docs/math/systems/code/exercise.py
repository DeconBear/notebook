# -*- coding: utf-8 -*-
"""
=== 线性方程组练习 ===
实现 numerical_rank：用 SVD 奇异值数大于 tol 的个数当数值秩。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def numerical_rank(A, tol=1e-8):
    """A: (m, n) -> 数值秩（大于 tol 的奇异值个数）。"""
    # TODO
    raise NotImplementedError


def _check():
    A = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0],
                  [0.0, 1.0, 1.0]])
    r = numerical_rank(A)
    assert r == 2, r
    I = np.eye(3)
    assert numerical_rank(I) == 3
    Z = np.zeros((2, 4))
    assert numerical_rank(Z) == 0
    print('通过：数值秩正确。')


if __name__ == '__main__':
    _check()
