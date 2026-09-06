# -*- coding: utf-8 -*-
"""
=== 估计练习 ===
实现 sample_variance(x, unbiased=True)。
运行: python exercise.py
"""
import numpy as np


def sample_variance(x, unbiased=True):
    """
    x: 一维数组。
    unbiased=True → 除以 n-1；False → 除以 n（MLE）。
    """
    # TODO
    raise NotImplementedError


def _check():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v_u = sample_variance(x, True)
    v_m = sample_variance(x, False)
    assert abs(v_u - 2.5) < 1e-12, v_u
    assert abs(v_m - 2.0) < 1e-12, v_m
    print('通过：样本方差 / MLE 方差正确。')


if __name__ == '__main__':
    _check()
