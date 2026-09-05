# -*- coding: utf-8 -*-
"""
=== 旋量练习 ===
实现 planar_v_pure_rotation：瞬心 q、角速度 ω 时，v = ω * (qy, -qx) 的约定
（与 demo 中 v = -ω * (-qy, qx) 相同）。
运行: python exercise.py
"""
import numpy as np


def planar_v(omega, qx, qy):
    """纯转动：v = ω * (qy, -qx)。"""
    # TODO
    raise NotImplementedError


def _check():
    v = planar_v(2.0, 1.0, 0.0)
    assert np.allclose(v, [0.0, -2.0])
    print('通过：绕 x 轴上一点转，速度沿 -y。')


if __name__ == '__main__':
    _check()
