# -*- coding: utf-8 -*-
"""
=== 非线性练习 ===
实现 pendulum_V(theta, omega, m, l, g)：动能+势能。
运行: python exercise.py
"""
import math


def pendulum_V(theta, omega, m, l, g):
    """V = 0.5 m (l ω)² + m g l (1 - cos θ)。"""
    # TODO
    raise NotImplementedError


def _check():
    v0 = pendulum_V(0.0, 0.0, m=1.0, l=1.0, g=9.8)
    assert abs(v0) < 1e-12
    v = pendulum_V(0.0, 2.0, m=1.0, l=1.0, g=9.8)
    assert abs(v - 2.0) < 1e-12  # 纯动能 0.5*(2)²
    v2 = pendulum_V(math.pi, 0.0, m=1.0, l=1.0, g=2.0)
    assert abs(v2 - 4.0) < 1e-12  # 倒立：mgl(1-(-1))=4
    print('通过：单摆能量正确。')


if __name__ == '__main__':
    _check()
