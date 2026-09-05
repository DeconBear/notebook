# -*- coding: utf-8 -*-
"""
=== 频域练习 ===
实现 mag_db(g)：20*log10(|g|)。g 为复数。
运行: python exercise.py
"""
import math


def mag_db(g):
    """幅值分贝：20 log10 |g|。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(mag_db(1.0) - 0.0) < 1e-9
    assert abs(mag_db(10.0) - 20.0) < 1e-9
    assert abs(mag_db(1j) - 0.0) < 1e-9
    assert abs(mag_db(0.1) - (-20.0)) < 1e-9
    print('通过：分贝换算正确。')


if __name__ == '__main__':
    _check()
