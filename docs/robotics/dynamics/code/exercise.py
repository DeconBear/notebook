# -*- coding: utf-8 -*-
"""
=== 动力学练习 ===
实现 g_term：单摆重力项 m g L cos θ（与 demo 同一符号约定）。
运行: python exercise.py
"""
import numpy as np

M, G, L = 1.0, 9.81, 0.8


def g_term(theta):
    """返回 m g L cos(theta)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(g_term(0.0) - M * G * L) < 1e-9
    print('通过：水平臂重力力矩最大。')


if __name__ == '__main__':
    _check()
