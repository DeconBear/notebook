# -*- coding: utf-8 -*-
"""
=== 常见分布练习 ===
实现 gaussian_pdf(x, mu, sigma)。
运行: python exercise.py
"""
import math


def gaussian_pdf(x, mu, sigma):
    """一维高斯密度。x 可以是标量。"""
    # TODO
    raise NotImplementedError


def _check():
    phi0 = gaussian_pdf(0.0, 0.0, 1.0)
    assert abs(phi0 - 1.0 / math.sqrt(2 * math.pi)) < 1e-9, phi0
    assert gaussian_pdf(1.0, 1.0, 2.0) > gaussian_pdf(3.0, 1.0, 2.0)
    print('通过：高斯密度正确。')


if __name__ == '__main__':
    _check()
