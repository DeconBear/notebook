# -*- coding: utf-8 -*-
"""
=== 信息论练习 ===
实现 kl_divergence(p, q)。
运行: python exercise.py
"""
import numpy as np


def kl_divergence(p, q, eps=1e-12):
    """
    离散 KL(p‖q) = sum p log(p/q)。
    对 q 做 eps 裁剪；忽略 p==0 的项。
    """
    # TODO
    raise NotImplementedError


def _check():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert abs(kl_divergence(p, q)) < 1e-9
    q2 = np.array([0.9, 0.1])
    d = kl_divergence(p, q2)
    # 手工：0.5*log(0.5/0.9)+0.5*log(0.5/0.1)
    expected = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
    assert abs(d - expected) < 1e-9
    print('通过：KL 计算正确。')


if __name__ == '__main__':
    _check()
