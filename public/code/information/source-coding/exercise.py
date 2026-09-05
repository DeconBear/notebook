# -*- coding: utf-8 -*-
"""
=== 信源编码练习 ===
实现 avg_code_len(probs, codes)。
运行: python exercise.py
"""


def avg_code_len(probs, codes):
    """L = sum_s p(s) * len(code(s))。"""
    # TODO
    raise NotImplementedError


def _check():
    probs = {'A': 0.5, 'B': 0.5}
    codes = {'A': '0', 'B': '1'}
    assert abs(avg_code_len(probs, codes) - 1.0) < 1e-12
    codes2 = {'A': '0', 'B': '10'}
    assert abs(avg_code_len(probs, codes2) - 1.5) < 1e-12
    print('通过：平均码长正确。')


if __name__ == '__main__':
    _check()
