# -*- coding: utf-8 -*-
"""
=== scene-3d 练习 ===
实现 nearest_latent：把显式位移映射到最近的潜动作中心下标。
"""
import numpy as np

np.random.seed(0)


def nearest_latent(delta, centers):
    """delta: (2,), centers: (K,2) -> int"""
    # TODO
    raise NotImplementedError


def _check():
    centers = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert nearest_latent(np.array([0.9, 0.1]), centers) == 0
    assert nearest_latent(np.array([0.0, -0.8]), centers) == 3
    print('通过')


if __name__ == '__main__':
    _check()
