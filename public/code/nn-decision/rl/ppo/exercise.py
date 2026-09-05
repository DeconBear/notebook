# -*- coding: utf-8 -*-
"""
=== PPO 练习：裁剪替代目标 + GAE ===
补全 clipped_surrogate 与 compute_gae。
运行: python exercise.py
"""
import numpy as np


def clipped_surrogate(ratio, advantage, eps=0.2):
    """
    PPO-Clip 的逐元素目标（尚未取均值）:
        min( r * A,  clip(r, 1-eps, 1+eps) * A )
    ratio, advantage: 同形状的 ndarray
    """
    # TODO: 用 np.minimum 与 np.clip 实现正文里的 L^CLIP
    return np.zeros_like(ratio, dtype=float)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    从后往前:
        delta_t = r_t + gamma * next_v * (1-done_t) - V_t
        A_t = delta_t + gamma * lam * (1-done_t) * A_{t+1}
    返回 advantages，形状与 rewards 相同。
    dones[t]=True 表示该步之后回合结束，next_v 不再自举。
    """
    # TODO: 倒序累加 TD 残差
    return np.zeros_like(rewards, dtype=float)


def _check_clip():
    ratio = np.array([0.5, 1.0, 1.5, 2.0])
    pos = np.ones_like(ratio)
    out = clipped_surrogate(ratio, pos, eps=0.2)
    # A>0 时目标被封在 1.2 * A
    expected = np.minimum(ratio * pos, np.clip(ratio, 0.8, 1.2) * pos)
    assert np.allclose(out, expected), f'正优势裁剪不对: {out}'
    neg = -np.ones_like(ratio)
    out_n = clipped_surrogate(ratio, neg, eps=0.2)
    expected_n = np.minimum(ratio * neg, np.clip(ratio, 0.8, 1.2) * neg)
    assert np.allclose(out_n, expected_n), f'负优势裁剪不对: {out_n}'
    print('通过：clipped_surrogate')


def _check_gae():
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.array([0.0, 0.0, 0.0])
    dones = np.array([False, False, True])
    # gamma=lam=1, V=0, 最后一步 done → A = [3, 2, 1]
    adv = compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=1.0)
    assert np.allclose(adv, [3.0, 2.0, 1.0]), f'λ=1 的 GAE 应为累计回报, 得到 {adv}'
    # λ=0 → 单步 TD：A = r + γ next_v (1-d) - V = r（此处 V=0 且最后 done）
    adv0 = compute_gae(rewards, values, dones, last_value=0.0, gamma=1.0, lam=0.0)
    assert np.allclose(adv0, [1.0, 1.0, 1.0]), f'λ=0 应为单步 TD, 得到 {adv0}'
    print('通过：compute_gae')


if __name__ == '__main__':
    _check_clip()
    _check_gae()
    print('PPO 练习全部通过。')
