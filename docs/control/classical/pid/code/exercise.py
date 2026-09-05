# -*- coding: utf-8 -*-
"""
=== 经典控制练习 ===
实现 pid_output：给定误差、积分、微分与增益，返回控制量。
运行: python exercise.py
"""


def pid_output(e, integ, de, kp, ki, kd):
    """u = kp*e + ki*integ + kd*de。"""
    # TODO
    raise NotImplementedError


def _check():
    u = pid_output(e=0.5, integ=2.0, de=-1.0, kp=2.0, ki=0.5, kd=1.0)
    assert abs(u - 1.0) < 1e-9
    print('通过：PID 合成正确。')


if __name__ == '__main__':
    _check()
