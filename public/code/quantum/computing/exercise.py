# -*- coding: utf-8 -*-
"""
=== 量子计算练习 ===
实现 apply_gate：把门矩阵作用到态矢量上。
实现 bell_state：用 H⊗I 再 CNOT 制备 |Φ+>。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)


def apply_gate(gate, state):
    """gate @ state。gate 与 state 维度匹配。"""
    # TODO
    raise NotImplementedError


def bell_state():
    """返回 |Φ+> = (|00>+|11>)/√2。"""
    # TODO
    raise NotImplementedError


def _check():
    z = np.array([1, 0], dtype=complex)
    hz = apply_gate(H, z)
    assert np.allclose(hz, np.array([1, 1], dtype=complex) / np.sqrt(2))
    phi = bell_state()
    target = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(phi, target)
    print('通过：门作用与 Bell 态制备正确。')


if __name__ == '__main__':
    _check()
