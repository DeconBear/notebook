# -*- coding: utf-8 -*-
"""
=== 特征值练习 ===
实现 power_iteration：反复 v <- Av / ||Av||，用 Rayleigh 商估特征值。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def power_iteration(A, n_iter=20, v0=None):
    """
    A: (n, n) 对称更好。
    返回 (lam, v)，v 为单位向量。
    """
    # TODO
    raise NotImplementedError


def _check():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    lam, v = power_iteration(A, n_iter=30, v0=np.array([1.0, 0.0]))
    w, Q = np.linalg.eigh(A)
    assert abs(lam - w[-1]) < 1e-6, (lam, w[-1])
    # 方向可差一个符号
    align = abs(np.dot(v / np.linalg.norm(v), Q[:, -1]))
    assert align > 0.999, align
    print('通过：幂迭代对准最大特征对。')


if __name__ == '__main__':
    _check()
