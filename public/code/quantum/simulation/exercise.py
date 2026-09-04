# -*- coding: utf-8 -*-
"""
=== 量子模拟练习 ===
实现 first_order_trotter_error_scale：一阶 Trotter 误差大致随 1/n 下降。
这里请实现 split_hamiltonian 返回 (A, B)，使得 H = A + B。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def split_hamiltonian(j=1.0, h=0.7):
    """
    两自旋 H = J Z⊗Z + h (X⊗I + I⊗X)
    返回 (A, B) 满足 A+B = H。
    """
    # TODO
    raise NotImplementedError


def _check():
    a, b = split_hamiltonian()
    h = a + b
    target = 1.0 * np.kron(Z, Z) + 0.7 * (np.kron(X, I2) + np.kron(I2, X))
    assert np.allclose(h, target)
    print('通过：哈密顿量拆分正确。')


if __name__ == '__main__':
    _check()
