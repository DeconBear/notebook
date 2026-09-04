# -*- coding: utf-8 -*-
"""
=== 量子存储练习 ===
实现 population_1：|1⟩ 经 T1 衰减后的布居。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def population_1(t, t1=1.0):
    """初始 |1⟩，振幅阻尼后 ρ_11(t)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert np.isclose(population_1(0.0), 1.0)
    assert np.isclose(population_1(np.log(2)), 0.5)
    assert population_1(10.0) < 0.01
    print('通过：T1 布居衰减正确。')


if __name__ == '__main__':
    _check()
