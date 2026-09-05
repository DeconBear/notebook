# -*- coding: utf-8 -*-
"""
=== 运动学练习 ===
实现 fk_x：返回 2R 末端 x 坐标。
运行: python exercise.py
"""
import numpy as np

L1, L2 = 1.0, 0.7


def fk_x(th1, th2):
    """x = L1 cos θ1 + L2 cos(θ1+θ2)。"""
    # TODO
    raise NotImplementedError


def _check():
    x = fk_x(0.0, 0.0)
    assert abs(x - (L1 + L2)) < 1e-9
    print('通过：伸直时 x = L1+L2。')


if __name__ == '__main__':
    _check()
