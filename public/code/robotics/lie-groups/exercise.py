# -*- coding: utf-8 -*-
"""
=== 李群练习 ===
实现 hat_02：hat(w) 的 (0,2) 元素，应等于 w_y。
运行: python exercise.py
"""
import numpy as np


def hat_02(wx, wy, wz):
    """hat(w)[0, 2] == wy。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(hat_02(1.0, 2.0, 3.0) - 2.0) < 1e-9
    print('通过：hat 的 (0,2) 是 wy。')


if __name__ == '__main__':
    _check()
