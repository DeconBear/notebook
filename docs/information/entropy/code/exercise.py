# -*- coding: utf-8 -*-
"""
=== 熵练习 ===
实现 cond_entropy(h_xy, h_y)=H(X,Y)-H(Y)。
运行: python exercise.py
"""


def cond_entropy(h_xy, h_y):
    """H(X|Y) = H(X,Y) - H(Y)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(cond_entropy(1.5, 1.0) - 0.5) < 1e-12
    assert abs(cond_entropy(1.0, 1.0) - 0.0) < 1e-12
    print('通过：条件熵定义正确。')


if __name__ == '__main__':
    _check()
