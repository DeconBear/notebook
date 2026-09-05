# -*- coding: utf-8 -*-
"""
=== 高斯信道练习 ===
实现 capacity_awgn(snr_lin)=0.5*log2(1+snr)。
运行: python exercise.py
"""
def capacity_awgn(snr_lin):
    """每实数维 bit：0.5 * log2(1+SNR)。"""
    # TODO
    raise NotImplementedError


def _check():
    assert abs(capacity_awgn(0.0) - 0.0) < 1e-12
    assert abs(capacity_awgn(1.0) - 0.5) < 1e-12
    assert abs(capacity_awgn(3.0) - 1.0) < 1e-12
    print('通过：AWGN 容量公式正确。')


if __name__ == '__main__':
    _check()
