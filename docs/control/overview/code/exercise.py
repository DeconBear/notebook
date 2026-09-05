# -*- coding: utf-8 -*-
"""
=== 控制论导论练习 ===
实现 tracking_error：误差 e = r - y。
运行: python exercise.py
"""


def tracking_error(r, y):
    """闭环第一式：e = r - y。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(tracking_error(1.0, 0.25) - 0.75) < 1e-12
    assert abs(tracking_error(0.0, -2.0) - 2.0) < 1e-12
    print('通过：误差 e = r - y。')


if __name__ == '__main__':
    _check()
