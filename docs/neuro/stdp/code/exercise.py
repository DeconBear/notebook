# -*- coding: utf-8 -*-
"""
=== STDP 练习 ===
实现 Δt>0 时 A+ exp(-Δt/τ+)，Δt<0 时 -A- exp(Δt/τ-)，并 clip 到 [0,1]。
运行: python exercise.py
"""
import numpy as np

A_PLUS, A_MINUS, TAU = 0.01, 0.012, 20.0


def pairwise_stdp_update(w, delta_t_ms):
    """返回更新后的权重。"""
    # TODO
    raise NotImplementedError


def _check():
    w = pairwise_stdp_update(0.5, 20.0)
    expect = 0.5 + A_PLUS * np.exp(-1.0)
    assert abs(w - expect) < 1e-9, (w, expect)
    w2 = pairwise_stdp_update(0.5, -20.0)
    expect2 = 0.5 - A_MINUS * np.exp(-1.0)
    assert abs(w2 - expect2) < 1e-9
    assert pairwise_stdp_update(0.99, 0.0) == 0.99
    print('通过：STDP 配对更新正确。')


if __name__ == '__main__':
    _check()
