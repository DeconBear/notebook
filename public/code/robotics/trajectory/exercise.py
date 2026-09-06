# -*- coding: utf-8 -*-
"""
=== 轨迹规划练习 ===
实现 smoothstep(s)：s∈[0,1] → s^2 (3-2s)，供三次零速度插值使用。
运行: python exercise.py
"""


def smoothstep(s):
    """标准 3s^2-2s^3。s=0→0，s=1→1，端点导数为 0。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(smoothstep(0.0)) < 1e-12
    assert abs(smoothstep(1.0) - 1.0) < 1e-12
    assert abs(smoothstep(0.5) - 0.5) < 1e-12
    # 中点斜率应大于线性 1（起步慢、中间快）
    ds = (smoothstep(0.51) - smoothstep(0.49)) / 0.02
    assert ds > 1.2, ds
    print('通过：smoothstep 正确。')


if __name__ == '__main__':
    _check()
