# -*- coding: utf-8 -*-
"""
=== 现代控制练习 ===
实现 lqr_u：u = -K x（K 为行向量，x 为状态）。
运行: python exercise.py
"""
import numpy as np


def lqr_u(K, x):
    """返回标量 -K @ x。"""
    # TODO
    raise NotImplementedError


def _check():
    K = np.array([[2.0, 0.5]])
    x = np.array([1.0, -4.0])
    assert abs(lqr_u(K, x) - 0.0) < 1e-9
    print('通过：LQR 控制律正确。')


if __name__ == '__main__':
    _check()
