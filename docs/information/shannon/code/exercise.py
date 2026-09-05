# -*- coding: utf-8 -*-
"""
=== 信息论练习 ===
实现 h2_half：h2(0.5) 应接近 1 bit。
运行: python exercise.py
"""
import numpy as np


def binary_entropy(p):
    """-p log2 p - (1-p) log2(1-p)，p 在 (0,1)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(binary_entropy(0.5) - 1.0) < 1e-9
    print('通过：公平比特熵为 1 bit。')


if __name__ == '__main__':
    _check()
