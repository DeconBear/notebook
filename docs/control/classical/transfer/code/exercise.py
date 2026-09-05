# -*- coding: utf-8 -*-
"""
=== 传递函数练习 ===
实现 damping_regime(zeta)：返回 'over' / 'critical' / 'under' / 'undamped' / 'unstable'。
运行: python exercise.py
"""


def damping_regime(zeta):
    """按 ζ 分类二阶系统。临界用 |ζ-1|<1e-9 判断。"""
    # TODO
    raise NotImplementedError


def _check():
    assert damping_regime(1.2) == 'over'
    assert damping_regime(1.0) == 'critical'
    assert damping_regime(0.3) == 'under'
    assert damping_regime(0.0) == 'undamped'
    assert damping_regime(-0.1) == 'unstable'
    print('通过：阻尼分类正确。')


if __name__ == '__main__':
    _check()
