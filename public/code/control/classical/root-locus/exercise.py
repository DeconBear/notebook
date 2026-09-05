# -*- coding: utf-8 -*-
"""
=== 根轨迹练习 ===
实现 char_poly_coeffs(k)：返回 s³+4s²+3s+K 从高次到常数的系数列表。
运行: python exercise.py
"""


def char_poly_coeffs(k):
    """[1, 4, 3, k]，供 np.roots 使用。"""
    # TODO
    raise NotImplementedError


def _check():
    c = char_poly_coeffs(6.0)
    assert list(c) == [1.0, 4.0, 3.0, 6.0] or list(map(float, c)) == [1.0, 4.0, 3.0, 6.0]
    print('通过：特征多项式系数正确。')


if __name__ == '__main__':
    _check()
