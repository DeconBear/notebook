# -*- coding: utf-8 -*-
"""
=== 信息论导论练习 ===
实现 nats_to_bits：H_bit = H_nat / ln(2)。
运行: python exercise.py
"""
import math


def nats_to_bits(h_nats):
    """nat → bit。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(nats_to_bits(math.log(2.0)) - 1.0) < 1e-12
    assert abs(nats_to_bits(0.0) - 0.0) < 1e-12
    print('通过：nat 与 bit 换算正确。')


if __name__ == '__main__':
    _check()
