# -*- coding: utf-8 -*-
"""
=== 量子信息全景练习 ===
实现 purity：Tr(ρ²)。纯态为 1，完全混合 1/2（单比特）。
运行: python exercise.py
"""
import numpy as np

np.random.seed(0)


def purity(rho):
    """Tr(ρ @ ρ)，rho 为 (2,2) 密度矩阵。"""
    # TODO
    raise NotImplementedError


def _check():
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_p = np.outer(plus, plus.conj())
    rho_m = 0.5 * np.eye(2)
    assert np.isclose(purity(rho_p), 1.0)
    assert np.isclose(purity(rho_m), 0.5)
    print('通过：纯度计算正确。')


if __name__ == '__main__':
    _check()
