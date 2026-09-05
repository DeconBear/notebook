# -*- coding: utf-8 -*-
"""
=== 编码练习 ===
由尖峰指示序列估计发放率：n_spikes / (n * dt_s)。
运行: python exercise.py
"""
import numpy as np


def spike_rate_hz(spikes, dt_s):
    """spikes 为 0/1 数组，dt_s 为步长（秒）。"""
    # TODO
    raise NotImplementedError


def _check():
    spk = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
    r = spike_rate_hz(spk, 0.001)
    assert abs(r - 300.0) < 1e-6, r
    print('通过：发放率估计正确。')


if __name__ == '__main__':
    _check()
