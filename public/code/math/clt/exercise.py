# -*- coding: utf-8 -*-
"""
=== CLT 练习 ===
实现 standardized_mean：sqrt(n) * (mean(x) - mu) / sigma。
运行: python exercise.py
"""
import numpy as np


def standardized_mean(x, mu, sigma):
    """
    x: 一维样本。返回标准化样本均值，CLT 下应接近 N(0,1)。
    """
    # TODO
    raise NotImplementedError


def _check():
    x = np.array([1.0, 3.0, 5.0])  # mean=3, n=3
    z = standardized_mean(x, mu=3.0, sigma=2.0)
    # sqrt(3) * (3-3)/2 = 0
    assert abs(z) < 1e-12, z
    z2 = standardized_mean(np.array([2.0, 2.0, 2.0, 2.0]), mu=0.0, sigma=2.0)
    # sqrt(4)*2/2 = 2
    assert abs(z2 - 2.0) < 1e-12, z2
    print('通过：标准化样本均值正确。')


if __name__ == '__main__':
    _check()
