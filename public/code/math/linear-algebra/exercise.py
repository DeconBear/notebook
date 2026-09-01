# -*- coding: utf-8 -*-
"""
=== 线性代数练习 ===
实现 apply_linear：用矩阵 A 变换一批向量（每行一个）。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def apply_linear(A, X):
    """
    A: (m, n), X: (N, n) -> (N, m)
    对每个样本行向量 x，计算 A @ x。
    """
    # TODO
    raise NotImplementedError


def _check():
    A = np.array([[0.0, -1.0], [1.0, 0.0]])  # 逆时针 90°
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    Y = apply_linear(A, X)
    assert Y.shape == (2, 2)
    assert np.allclose(Y[0], [0.0, 1.0])
    assert np.allclose(Y[1], [-2.0, 0.0])
    print('通过：批量线性变换正确。')


if __name__ == '__main__':
    _check()
