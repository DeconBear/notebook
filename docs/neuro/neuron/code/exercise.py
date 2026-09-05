# -*- coding: utf-8 -*-
"""
=== 神经元练习 ===
实现指数 EPSP：t>=t0 时 amp * exp(-(t-t0)/tau)，否则 0。
运行: python exercise.py
"""
import numpy as np


def exp_psp(t, t0, amp, tau):
    """返回与 t 同形状的数组。"""
    # TODO
    raise NotImplementedError


def _check():
    t = np.array([0.0, 5.0, 10.0, 15.0])
    y = exp_psp(t, 5.0, 10.0, 5.0)
    assert y[0] == 0.0
    assert abs(y[1] - 10.0) < 1e-9
    assert abs(y[2] - 10.0 * np.exp(-1.0)) < 1e-9
    print('通过：指数 EPSP 形状正确。')


if __name__ == '__main__':
    _check()
