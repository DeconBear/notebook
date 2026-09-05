# -*- coding: utf-8 -*-
"""
=== 机构学练习 ===
实现 gruebler_planar：M = 3(N-1) - 2*J1。
运行: python exercise.py
"""


def gruebler_planar(n_links, n_hinges):
    """平面全铰链机构自由度。"""
    # TODO
    raise NotImplementedError


def _check():
    assert gruebler_planar(4, 4) == 1
    print('通过：四杆 DOF=1。')


if __name__ == '__main__':
    _check()
