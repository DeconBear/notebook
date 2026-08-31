# -*- coding: utf-8 -*-
"""
=== 因果练习 ===
运行: python exercise.py
实现 observational_vs_do_gap：返回关联模型在干预分布上的 MSE
减去因果模型在干预分布上的 MSE（应明显为正）。
"""
import numpy as np

np.random.seed(0)


def observational_vs_do_gap(z_obs, a_obs, y_obs, z_do, a_do, y_do):
    """
    用观测数据分别拟合 y~z 与 y~a。
    返回 mse_z_on_do - mse_a_on_do。
    """
    # TODO
    raise NotImplementedError


def _check():
    n = 3000
    z = np.random.randn(n)
    a = np.tanh(z) + 0.05 * np.random.randn(n)
    y = a + 0.1 * np.random.randn(n)
    z2 = np.random.randn(n)
    a2 = np.random.uniform(-1, 1, n)
    y2 = a2 + 0.1 * np.random.randn(n)
    gap = observational_vs_do_gap(z, a, y, z2, a2, y2)
    assert gap > 0.05, gap
    print('通过：干预下关联模型明显更差。')


if __name__ == '__main__':
    _check()
