# -*- coding: utf-8 -*-
"""
=== 概率练习 ===
实现 bayes_posterior_beta：Beta 先验 + 二项观测的后验参数。
运行: python exercise.py
"""


def bayes_posterior_beta(prior_a, prior_b, heads, trials):
    """
    先验 Beta(prior_a, prior_b)，观测 trials 次中 heads 次正面。
    返回 (posterior_a, posterior_b)。
    """
    # TODO
    raise NotImplementedError


def _check():
    a, b = bayes_posterior_beta(1, 1, 3, 10)
    assert (a, b) == (4, 8)
    a2, b2 = bayes_posterior_beta(2, 5, 0, 4)
    assert (a2, b2) == (2, 9)
    print('通过：Beta-Binomial 后验参数正确。')


if __name__ == '__main__':
    _check()
