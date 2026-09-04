# -*- coding: utf-8 -*-
"""
=== 量子网络练习 ===
实现 fidelity(a, b)：纯态保真度 |⟨a|b⟩|^2。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def fidelity(a, b):
    """两个归一化纯态的保真度。"""
    # TODO
    raise NotImplementedError


def _check():
    a = np.array([1, 0], dtype=complex)
    b = np.array([0, 1], dtype=complex)
    h = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.isclose(fidelity(a, a), 1.0)
    assert np.isclose(fidelity(a, b), 0.0)
    assert np.isclose(fidelity(a, h), 0.5)
    print('通过：纯态保真度正确。')


if __name__ == '__main__':
    _check()
