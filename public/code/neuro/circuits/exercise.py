# -*- coding: utf-8 -*-
"""
=== 回路练习 ===
给定权重剖面，计算 LR 探针分数 sum w[i]*(i+1)（从左扫到右）。
运行: python exercise.py
"""
import numpy as np


def probe_lr(weights):
    """weights 为从左到右的一维数组。"""
    # TODO
    raise NotImplementedError


def _check():
    w = np.array([0.1, 0.2, 0.3])
    assert abs(probe_lr(w) - (0.1 * 1 + 0.2 * 2 + 0.3 * 3)) < 1e-9
    print('通过：LR 探针分数正确。')


if __name__ == '__main__':
    _check()
