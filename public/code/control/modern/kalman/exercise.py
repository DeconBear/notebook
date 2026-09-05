# -*- coding: utf-8 -*-
"""
=== 卡尔曼练习 ===
实现 kgain(P, R)：一维卡尔曼增益 P / (P + R)。
运行: python exercise.py
"""


def kgain(P, R):
    """K_g = P / (P + R)。P、R 为正标量。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(kgain(1.0, 1.0) - 0.5) < 1e-12
    assert abs(kgain(0.0, 2.0) - 0.0) < 1e-12
    assert abs(kgain(4.0, 0.0) - 1.0) < 1e-12
    print('通过：一维卡尔曼增益正确。')


if __name__ == '__main__':
    _check()
