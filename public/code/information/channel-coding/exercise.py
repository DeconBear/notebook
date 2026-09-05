# -*- coding: utf-8 -*-
"""
=== 信道编码练习 ===
实现 syndrome_pos(s0,s1,s2)：三个校验合成 1-index 错误位置，全 0 表示无错。
运行: python exercise.py
"""


def syndrome_pos(s0, s1, s2):
    """pos = s0 + 2*s1 + 4*s2，范围 0..7。"""
    # TODO
    raise NotImplementedError


def _check():
    assert syndrome_pos(0, 0, 0) == 0
    assert syndrome_pos(1, 0, 0) == 1
    assert syndrome_pos(0, 1, 0) == 2
    assert syndrome_pos(1, 1, 1) == 7
    print('通过：校验子位置正确。')


if __name__ == '__main__':
    _check()
