# -*- coding: utf-8 -*-
"""
=== 积分练习 ===
实现 trapezoid(f, a, b, n)：把 [a,b] 切成 n 段的复合梯形法则。
运行: python exercise.py
"""


def trapezoid(f, a, b, n):
    """
    ∫_a^b f ≈ (h/2)*(f(a)+f(b)) + h * Σ_{i=1}^{n-1} f(a+ih)
    h = (b-a)/n
    """
    # TODO
    raise NotImplementedError


def _check():
    def sq(x):
        return x * x

    val = trapezoid(sq, 0.0, 1.0, 64)
    assert abs(val - 1.0 / 3.0) < 5e-4, val
    # 线性函数梯形应精确
    exact_line = trapezoid(lambda x: 2.0 * x + 1.0, 0.0, 2.0, 4)
    assert abs(exact_line - 6.0) < 1e-12, exact_line
    print('通过：梯形法则正确。')


if __name__ == '__main__':
    _check()
