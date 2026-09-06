# -*- coding: utf-8 -*-
"""
=== 机器人学导论练习 ===
实现 workspace_radius_bounds(l1, l2)：返回 (rmin, rmax)。
运行: python exercise.py
"""


def workspace_radius_bounds(l1, l2):
    """平面 2R 可达半径的闭区间 [ |l1-l2|, l1+l2 ]。"""
    # TODO
    raise NotImplementedError


def _check():
    rmin, rmax = workspace_radius_bounds(1.0, 0.7)
    assert abs(rmin - 0.3) < 1e-12, (rmin, rmax)
    assert abs(rmax - 1.7) < 1e-12, (rmin, rmax)
    a, b = workspace_radius_bounds(0.5, 0.5)
    assert abs(a) < 1e-12 and abs(b - 1.0) < 1e-12
    print('通过：工作空间半径边界正确。')


if __name__ == '__main__':
    _check()
