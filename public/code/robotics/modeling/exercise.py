# -*- coding: utf-8 -*-
"""
=== 建模练习 ===
实现 dh_trans_x：平面 DH（alpha=d=0）齐次矩阵的 (0,3) 元素，即 a*cos(theta)。
运行: python exercise.py
"""
import numpy as np


def dh_trans_x(a, theta):
    """返回 a * cos(theta)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(dh_trans_x(2.0, 0.0) - 2.0) < 1e-9
    print('通过：θ=0 时平移就是杆长。')


if __name__ == '__main__':
    _check()
