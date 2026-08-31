# -*- coding: utf-8 -*-
"""
=== PETS 练习 ===
运行: python exercise.py
完成 CEM 的精英更新，观察分布是否收缩。
"""
import numpy as np

np.random.seed(42)

# TODO: 实现 cem_update(mu, std, seqs, scores, n_elite)
# 从 scores 最大的 n_elite 条序列估计新的 mu, std


def cem_update(mu, std, seqs, scores, n_elite):
    """返回 (new_mu, new_std)。"""
    # TODO
    raise NotImplementedError


def _check():
    seqs = np.random.randn(50, 8)
    scores = -np.sum(seqs ** 2, axis=1)
    mu, std = np.zeros(8), np.ones(8)
    nmu, nstd = cem_update(mu, std, seqs, scores, 10)
    assert nmu.shape == (8,)
    assert np.mean(nstd) < 1.0
    print('通过：精英更新让标准差下降。')


if __name__ == '__main__':
    _check()
