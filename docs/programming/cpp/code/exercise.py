# -*- coding: utf-8 -*-
"""
=== C++ 基础 练习 ===
用 Python 复现 vec2.hpp 的加法与模长平方，确认你读懂头文件。
运行: python exercise.py
"""


def add_vec(a, b):
    """a, b 是长度为 2 的 tuple/list。返回 (x, y)。"""
    # TODO
    raise NotImplementedError


def norm2(v):
    """||v||^2 = x*x + y*y。"""
    # TODO
    raise NotImplementedError


def _check():
    assert add_vec((3.0, 4.0), (1.0, 0.0)) == (4.0, 4.0)
    assert norm2((3.0, 4.0)) == 25.0
    print('练习通过。对照: g++ -std=c++17 demo.cpp -o cpp_demo')


if __name__ == '__main__':
    _check()
