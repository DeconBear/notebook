# -*- coding: utf-8 -*-
"""
=== LeWM 练习 ===
运行: python exercise.py
实现 latent_goal_cost：开环滚动后计算与目标嵌入的 L2 距离。
"""
import numpy as np

np.random.seed(42)


def latent_goal_cost(z0, zg, actions, predict_fn):
    """
    从 z0 出发，按 actions 逐步 predict_fn(z, a)，
    返回最终嵌入与 zg 的平方 L2 距离。
    """
    # TODO
    raise NotImplementedError


def _check():
    def predict(z, a):
        return z + 0.5 * np.asarray(a)

    z0 = np.zeros(3)
    zg = np.array([1.0, 0.0, 0.0])
    actions = [np.array([0.4, 0.0, 0.0]), np.array([0.4, 0.0, 0.0])]
    c = latent_goal_cost(z0, zg, actions, predict)
    # 0.4+0.4=0.8，距离平方 = (0.2)^2 = 0.04
    assert abs(c - 0.04) < 1e-6, c
    print('通过：终端潜距离计算正确。')


if __name__ == '__main__':
    _check()
