# -*- coding: utf-8 -*-
"""
=== 率失真练习 ===
实现 binary_rd(D)=1-h2(D)，0<D≤0.5。
运行: python exercise.py
"""
import math


def binary_rd(D):
    """公平比特汉明失真：1 - (-D log2 D -(1-D) log2(1-D))。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(binary_rd(0.5) - 0.0) < 1e-9
    h = -0.25 * math.log2(0.25) - 0.75 * math.log2(0.75)
    assert abs(binary_rd(0.25) - (1.0 - h)) < 1e-9
    print('通过：二元率失真正确。')


if __name__ == '__main__':
    _check()
