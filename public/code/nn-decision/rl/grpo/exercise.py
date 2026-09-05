# -*- coding: utf-8 -*-
"""
=== GRPO 练习：组内相对优势 + 套上 PPO 裁剪 ===
补全 group_advantages 与 grpo_clipped_objective。
运行: python exercise.py
"""
import numpy as np


def group_advantages(rewards, eps=1e-8):
    """
    rewards: 形状 (G,) 的一组标量奖励
    返回同形状的 z-score。若标准差过小（全对或全错），返回全 0。
    """
    # TODO: (r - mean) / (std + eps)；std 过小则全 0
    return np.zeros_like(rewards, dtype=float)


def grpo_clipped_objective(ratio, advantage, eps=0.2):
    """
    与 PPO 相同的逐元素裁剪目标（advantage 在一条输出的所有 token 上共享）:
        min( r * A, clip(r, 1-eps, 1+eps) * A )
    """
    # TODO: 和 PPO 练习里的 clipped_surrogate 是同一公式
    return np.zeros_like(ratio, dtype=float)


def _check_group():
    r = np.array([1.0, 0.0, 1.0, 0.0])
    a = group_advantages(r)
    expect = (r - r.mean()) / (r.std() + 1e-8)
    assert np.allclose(a, expect), f'普通组 z-score 不对: {a}'
    flat = group_advantages(np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.allclose(flat, 0.0), f'全对时应无信号, 得到 {flat}'
    print('通过：group_advantages')


def _check_clip():
    ratio = np.array([0.5, 1.0, 1.5])
    adv = np.array([1.0, 1.0, 1.0])  # 整段输出共享同一个 A
    out = grpo_clipped_objective(ratio, adv, eps=0.2)
    expected = np.minimum(ratio * adv, np.clip(ratio, 0.8, 1.2) * adv)
    assert np.allclose(out, expected), f'GRPO 裁剪应与 PPO 相同, 得到 {out}'
    print('通过：grpo_clipped_objective')


if __name__ == '__main__':
    _check_group()
    _check_clip()
    print('GRPO 练习全部通过。')
