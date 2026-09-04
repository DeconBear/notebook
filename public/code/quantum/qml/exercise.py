# -*- coding: utf-8 -*-
"""
=== 量子机器学习练习 ===
实现 angle_from_hidden：把隐藏特征线性映射到 [0, 2π] 的门角度。
这对应原仓「角度头 + Sigmoid × 2π」，不依赖 pyvqnet。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def angle_from_hidden(hidden, weight, bias):
    """
    hidden: (batch, d)
    weight: (n_qubits, d)
    bias: (n_qubits,)
    返回 angles ∈ [0, 2π]，shape (batch, n_qubits)
    步骤：linear = hidden @ W.T + b，再 sigmoid，再乘 2π。
    """
    # TODO
    raise NotImplementedError


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _check():
    hidden = np.zeros((2, 4))
    weight = np.zeros((3, 4))
    bias = np.zeros(3)
    ang = angle_from_hidden(hidden, weight, bias)
    assert ang.shape == (2, 3)
    assert np.allclose(ang, np.pi)  # sigmoid(0)=0.5 → π
    print('通过：角度头映射正确。')


if __name__ == '__main__':
    _check()
