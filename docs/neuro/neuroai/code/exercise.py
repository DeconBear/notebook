# -*- coding: utf-8 -*-
"""
=== NeuroAI 练习 ===
实现 Hebb：w <- w + lr * post * pre。
运行: python exercise.py
"""
import numpy as np


def hebbian_update(w, pre, post, lr=0.1):
    """w, pre 为一维数组，post 为标量。"""
    # TODO
    raise NotImplementedError


def _check():
    w = np.array([0.0, 1.0])
    pre = np.array([1.0, 0.0])
    out = hebbian_update(w, pre, 2.0, lr=0.5)
    assert np.allclose(out, np.array([1.0, 1.0]))
    print('通过：Hebb 更新正确。')


if __name__ == '__main__':
    _check()
