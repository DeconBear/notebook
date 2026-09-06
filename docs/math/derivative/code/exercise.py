# -*- coding: utf-8 -*-
"""
=== 导数练习 ===
实现 central_diff(f, x, h)：中心差分 (f(x+h)-f(x-h))/(2h)。
运行: python exercise.py
"""


def central_diff(f, x, h=1e-6):
    """对一元函数 f 在 x 处做中心差分。"""
    # TODO
    raise NotImplementedError


def _check():
    def f(t):
        return t ** 3 - 2.0 * t

    approx = central_diff(f, 2.0, 1e-6)
    assert abs(approx - 10.0) < 1e-6, approx
    approx0 = central_diff(lambda t: t * t, 0.0, 1e-5)
    assert abs(approx0) < 1e-8, approx0
    print('通过：中心差分正确。')


if __name__ == '__main__':
    _check()
