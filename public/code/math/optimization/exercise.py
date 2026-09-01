# -*- coding: utf-8 -*-
"""
=== 优化练习 ===
实现 gradient_step：一次梯度下降更新。
运行: python exercise.py
"""
import numpy as np


def gradient_step(w, grad, eta):
    """返回 w - eta * grad。"""
    # TODO
    raise NotImplementedError


def _check():
    w = np.array([1.0, 2.0])
    g = np.array([0.5, -1.0])
    w2 = gradient_step(w, g, 0.2)
    assert np.allclose(w2, [0.9, 2.2])
    print('通过：梯度步更新正确。')


if __name__ == '__main__':
    _check()
