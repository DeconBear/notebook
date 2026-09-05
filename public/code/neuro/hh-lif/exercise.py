# -*- coding: utf-8 -*-
"""
=== HH/LIF 练习 ===
实现流变阈值 I_rh = (V_th - V_rest) / R。
运行: python exercise.py
"""


def rheobase(V_th, V_rest, R):
    """恒流下刚能到阈值的电流（忽略不应期）。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(rheobase(-50.0, -70.0, 20.0) - 1.0) < 1e-9
    print('通过：rheobase 计算正确。')


if __name__ == '__main__':
    _check()
