# -*- coding: utf-8 -*-
"""
s21 RLHF：当强化学习遇见大模型 — 练习代码
=============================================
请完成以下 TODO 任务，巩固对 PPO 和 DPO 核心机制的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md 和 demo.py，再尝试独立补全代码。
运行方式：在 s21_rlhf/ 目录下执行 python code/exercise.py
"""

import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# TODO 1: 实现 PPO 裁剪目标
# ============================================================================
def ppo_clipped_objective(
    ratio: torch.Tensor,                                          # r_t(θ) = π_θ(a|s) / π_old(a|s)
    advantage: torch.Tensor,                                      # 优势估计 Â_t
    clip_epsilon: float = 0.2,                                    # 裁剪参数 ε
) -> torch.Tensor:
    """
    TODO 1: 实现 PPO 的裁剪替代目标函数。

    PPO 的裁剪目标:
        L_CLIP = E[ min( r_t(θ) · Â_t,  clip(r_t(θ), 1-ε, 1+ε) · Â_t ) ]

    其中:
        - r_t(θ) = π_θ(a|s) / π_old(a|s) 是新旧策略的概率比
        - Â_t 是优势估计（正 = 好动作，负 = 坏动作）
        - ε 控制更新幅度（通常 ε = 0.2）

    PPO 的裁剪机制:
        当 Â > 0 (好动作): 允许增加概率，但最多增加到 r = 1+ε
                          (防止"过度自信"——一次更新不能增加太多)
        当 Â < 0 (坏动作): 允许降低概率，但最多降低到 r = 1-ε
                          (防止"过度惩罚"——一次更新不能减少太多)

    参数:
        ratio: 概率比 r_t(θ)，shape (N,) -- N 个时间步
        advantage: 优势估计 Â_t，shape (N,)
        clip_epsilon: 裁剪范围 ε

    返回:
        objective: 标量目标值（取 mean，不含负号 — 调用者需要取负用于梯度下降）
    """
    # TODO: 补全 PPO 裁剪目标

    # 步骤 1: 计算未裁剪的目标: r_t(θ) * Â_t
    surr1 = None  # ← TODO: ratio * advantage

    # 步骤 2: 计算裁剪后的目标: clip(r, 1-ε, 1+ε) * Â_t
    # 提示: 使用 torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped_ratio = None  # ← TODO: torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surr2 = None  # ← TODO: clipped_ratio * advantage

    # 步骤 3: 取 min (PPO 的保守策略)
    # PPO 的关键: 取 min(surr1, surr2) 确保保守更新
    objective = None  # ← TODO: torch.min(surr1, surr2).mean()

    return objective


# ---- 测试 TODO 1 ----
def test_ppo_clipped():
    """测试 PPO 裁剪目标。"""
    print("=" * 60)
    print("TODO 1 测试: PPO 裁剪目标")
    print("=" * 60)

    # 测试情景 1: 正优势 (好动作)
    ratio = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])              # 各种 ratio
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])          # 正优势
    result = ppo_clipped_objective(ratio, advantage, clip_epsilon=0.2)

    if result is None:
        print("  TODO 未完成，请补全 ppo_clipped_objective 函数")
        return

    print(f"  测试 1 [正优势 Â > 0]:")
    print(f"    ratio      = {ratio.tolist()}")
    print(f"    advantage  = {advantage.tolist()}")
    print(f"    objective  = {result.item():.4f}")
    # 手动验证: 未裁剪 (1.0+1.5+1.2+1.2+1.2)/5 = 1.22 (ε=0.2)
    # ratio > 1+ε 时被裁剪为 1+ε=1.2
    expected = (1.0 + 1.0 + 1.2 + 1.2 + 1.2) / 5
    print(f"    预期       ≈ {expected:.4f}")
    if abs(result.item() - expected) < 0.1:
        print(f"    ✓ 测试通过!")
    else:
        print(f"    ✗ 测试失败: 预期 ≈{expected:.4f}")

    # 测试情景 2: 负优势 (坏动作)
    advantage_neg = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])  # 负优势
    result2 = ppo_clipped_objective(ratio, advantage_neg, clip_epsilon=0.2)

    print(f"\n  测试 2 [负优势 Â < 0]:")
    print(f"    ratio      = {ratio.tolist()}")
    print(f"    advantage  = {advantage_neg.tolist()}")
    print(f"    objective  = {result2.item():.4f}")
    # 负优势下 ratio < 1-ε 时被裁剪为 1-ε=0.8
    # (max(-0.5,-0.8)*0.8 + max(-1,...)未裁剪) / 5
    # min(preserves max negativity) — let me think
    # surr1: [-0.5, -1, -1.5, -2, -3]; surr2 with clip (min 0.8): [-0.5, -0.8, -0.8, -0.8, -0.8]
    # min: [-0.5, -1, -1.5, -2, -3] — wait, min of -0.5 and -0.8 is -0.8 (more neg), wrong
    # Actually: min(surr1, surr2) where both are negative
    # When both negative, min selects the MORE negative (larger absolute value)
    # For ratio 0.5: surr1=-0.5, surr2=-0.5 (r=0.5 within [0.8,1.2]? No, 0.5<0.8, so clipped to 0.8)
    #   surr1=-0.5, surr2=-0.8, min=-0.8 → clipped activation
    # For ratio 0.5: surr1=-0.5, surr2=-0.8, min=-0.8
    # Wait no — the point is when A<0, min picks the more negative value, meaning:
    # if r < 1-ε, clipped ratio = 1-ε, surr2 = (1-ε)*A which is LESS negative (closer to 0)
    # if r > 1+ε, clipped ratio = 1+ε, surr2 = (1+ε)*A which is MORE negative
    # The min with advantage<0 prevents ratio from going below 1-ε
    print(f"    (负优势下 min 选择更负的目标, 裁剪防止 ratio 过度降低)")

    # 测试情景 3: 检查裁剪边界行为
    # ratio 正好在边界上
    ratio_edge = torch.tensor([0.79, 0.80, 1.19, 1.20, 1.21])
    # clipped to [0.8, 1.2]
    clipped = torch.clamp(ratio_edge, 0.8, 1.2)
    print(f"\n  测试 3 [裁剪边界验证 ε=0.2]:")
    print(f"    ratio   = {ratio_edge.tolist()}")
    print(f"    clipped = {clipped.tolist()}")
    print(f"    预期   = [0.8, 0.8, 1.19, 1.2, 1.2]")
    if torch.allclose(clipped, torch.tensor([0.8, 0.8, 1.19, 1.2, 1.2])):
        print(f"    ✓ 裁剪区间正确! [0.8, 1.2]")
    else:
        print(f"    ✗ 裁剪区间可能有问题")

    print()


# ============================================================================
# TODO 2: 实现 DPO 损失函数
# ============================================================================
def dpo_loss(
    log_p_w: torch.Tensor,                                        # log π_θ(y_w | x)
    log_p_l: torch.Tensor,                                        # log π_θ(y_l | x)
    ref_log_p_w: torch.Tensor,                                    # log π_ref(y_w | x)
    ref_log_p_l: torch.Tensor,                                    # log π_ref(y_l | x)
    beta: float = 0.1,                                            # DPO 温度参数
) -> torch.Tensor:
    """
    TODO 2: 实现 DPO 损失函数。

    DPO 损失:
        L_DPO = -log σ( β·(log π_θ(y_w|x) - log π_ref(y_w|x)
                          - log π_θ(y_l|x) + log π_ref(y_l|x)) )

    或者等价地:
        L_DPO = -log σ( β · (log(π_θ(y_w)/π_ref(y_w))
                           - log(π_θ(y_l)/π_ref(y_l))) )

    直观理解:
        - 如果策略更偏好 y_w（相对参考模型），而更不偏好 y_l，
          则括号内的差值变大，sigmoid 趋近于 1，损失趋近于 0
        - 反之，如果策略偏好 y_l 多于 y_w，差值变小甚至变负，
          sigmoid 趋近于 0，损失很大

    参数:
        log_p_w: 策略对偏好回复的 log 概率 (标量 Tensor)
        log_p_l: 策略对不偏好回复的 log 概率 (标量 Tensor)
        ref_log_p_w: 参考模型对偏好回复的 log 概率 (标量 Tensor)
        ref_log_p_l: 参考模型对不偏好回复的 log 概率 (标量 Tensor)
        beta: 温度参数，控制允许策略偏离参考模型的程度

    返回:
        loss: DPO 损失值 (标量)
    """
    # TODO: 补全 DPO 损失

    # 步骤 1: 计算对数比率
    # log(π_θ / π_ref) = log π_θ - log π_ref
    log_ratio_w = None  # ← TODO: log_p_w - ref_log_p_w
    log_ratio_l = None  # ← TODO: log_p_l - ref_log_p_l

    # 步骤 2: 计算差值 diff = β * (log_ratio_w - log_ratio_l)
    diff = None  # ← TODO: beta * (log_ratio_w - log_ratio_l)

    # 步骤 3: 计算 DPO 损失 = -log σ(diff)
    # 提示: 使用 F.logsigmoid(diff) 得到 log σ(diff)，再取负号
    #      这样可以避免数值不稳定（比直接 -log(sigmoid(diff)) 更稳定）
    loss = None  # ← TODO: -F.logsigmoid(diff)

    return loss


# ---- 测试 TODO 2 ----
def test_dpo_loss():
    """测试 DPO 损失函数。"""
    print("=" * 60)
    print("TODO 2 测试: DPO 损失函数")
    print("=" * 60)

    # 情景 1: 策略偏好 y_w（好情况 — 损失应该小）
    log_p_w = torch.tensor(-2.0)                                  # 策略对 y_w 的 log 概率
    log_p_l = torch.tensor(-5.0)                                  # 策略对 y_l 的 log 概率
    ref_log_p_w = torch.tensor(-3.0)                              # 参考对 y_w 的 log 概率
    ref_log_p_l = torch.tensor(-3.0)                              # 参考对 y_l 的 log 概率

    loss1 = dpo_loss(log_p_w, log_p_l, ref_log_p_w, ref_log_p_l, beta=0.1)

    if loss1 is None:
        print("  TODO 未完成，请补全 dpo_loss 函数")
        return

    print(f"  测试 1 [策略偏好 y_w (正确方向)]:")
    print(f"    log π_θ(y_w)={log_p_w.item()},  log π_θ(y_l)={log_p_l.item()}")
    print(f"    log π_ref(y_w)={ref_log_p_w.item()}, log π_ref(y_l)={ref_log_p_l.item()}")
    print(f"    DPO loss = {loss1.item():.4f}")
    # log_ratio_w = -2 - (-3) = 1
    # log_ratio_l = -5 - (-3) = -2
    # diff = 0.1 * (1 - (-2)) = 0.3
    # loss = -log σ(0.3) ≈ -log(0.574) ≈ 0.555
    expected1 = -np.log(1 / (1 + np.exp(-0.3)))
    print(f"    预期 loss ≈ {expected1:.4f}")
    if abs(loss1.item() - expected1) < 0.05:
        print(f"    ✓ 测试通过!")
    else:
        print(f"    ✗ 测试失败")

    # 情景 2: 策略偏好 y_l（坏情况 — 损失应该大）
    log_p_w_bad = torch.tensor(-5.0)                              # 策略给 y_w 低概率
    log_p_l_bad = torch.tensor(-2.0)                              # 策略给 y_l 高概率
    # 参考模型不变

    loss2 = dpo_loss(log_p_w_bad, log_p_l_bad, ref_log_p_w, ref_log_p_l, beta=0.1)

    print(f"\n  测试 2 [策略偏好 y_l (错误方向)]:")
    print(f"    log π_θ(y_w)={log_p_w_bad.item()}, log π_θ(y_l)={log_p_l_bad.item()}")
    print(f"    DPO loss = {loss2.item():.4f}")
    # log_ratio_w = -5 - (-3) = -2
    # log_ratio_l = -2 - (-3) = 1
    # diff = 0.1 * (-2 - 1) = -0.3
    # loss = -log σ(-0.3) ≈ -log(0.426) ≈ 0.854
    expected2 = -np.log(1 / (1 + np.exp(0.3)))
    print(f"    预期 loss ≈ {expected2:.4f}")
    if abs(loss2.item() - expected2) < 0.05:
        print(f"    ✓ 测试通过!")
    else:
        print(f"    ✗ 测试失败")

    # 验证 loss1 < loss2 (正确偏好应有更小损失)
    if loss1.item() < loss2.item():
        print(f"\n    ✓ loss(正确方向)={loss1.item():.4f} < "
              f"loss(错误方向)={loss2.item():.4f}, 符合预期!")
    else:
        print(f"\n    ✗ 损失比较不符合预期")

    print()


# ============================================================================
# TODO 3: 实现 GAE 优势估计
# ============================================================================
def compute_gae(
    rewards: List[float],                                         # 每步奖励 r_t
    values: List[float],                                          # 每步价值 V(s_t)
    gamma: float = 0.99,                                          # 折扣因子 γ
    gae_lambda: float = 0.95,                                     # GAE λ 参数
    done: bool = False,                                           # 是否终止 (用于最后一步)
) -> torch.Tensor:
    """
    TODO 3: 实现 GAE (Generalized Advantage Estimation) 优势估计。

    GAE 公式:
        Â_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

    其中 δ_t = r_t + γ·V(s_{t+1}) - V(s_t) (TD 误差)

    递推形式（从后往前计算）:
        Â_T = 0  (最后一步之后没有优势)
        Â_t = δ_t + γλ · Â_{t+1}    (递推关系)

    参数:
        rewards: 每步的奖励 r_0, r_1, ..., r_{T-1}
        values: 每步的状态价值 V(s_0), V(s_1), ..., V(s_{T-1})
        gamma: 折扣因子 γ
        gae_lambda: GAE λ 参数
        done: 是否为终止 episode（如果是，最后一步没有下一个状态价值）

    返回:
        advantages: GAE 优势估计，shape (T,)

    提示:
        - T = len(rewards) = len(values)
        - 最后一步的下一状态价值: 如果 done, V(s_T)=0; 否则 V(s_T)=values[-1]
        - 从 t=T-1 到 t=0 递推计算
    """
    # TODO: 补全 GAE 计算

    T = len(rewards)                                               # 步数
    advantages = torch.zeros(T)                                    # 初始化优势

    # 最后一步的下一状态价值
    if done:
        next_value = 0.0                                          # 终止后价值为 0
    else:
        next_value = values[-1] if T > 0 else 0.0                # 非终止: 用最后一个 value

    gae = 0.0                                                      # 累积优势变量

    # 从后往前递推
    for t in reversed(range(T)):
        # 步骤 1: 计算 TD 误差 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        # 对于最后一步 (t == T-1): V(s_{t+1}) = next_value
        # 对于其他步: V(s_{t+1}) = values[t+1]
        if t == T - 1:
            next_v = next_value                                   # 最后一步的下一状态
        else:
            next_v = values[t + 1]                                # 一般情况

        delta = None  # ← TODO: rewards[t] + gamma * next_v - values[t]

        # 步骤 2: 递推计算 GAE
        # gae = delta + gamma * gae_lambda * gae
        gae = None  # ← TODO: delta + gamma * gae_lambda * gae

        # 步骤 3: 存储优势
        advantages[t] = None  # ← TODO: gae

    return advantages


# ---- 测试 TODO 3 ----
def test_gae():
    """测试 GAE 优势估计。"""
    print("=" * 60)
    print("TODO 3 测试: GAE 优势估计")
    print("=" * 60)

    # 测试数据: 5 步 episode，奖励逐渐变好
    rewards = [0.0, 0.0, 0.0, 0.0, 10.0]                         # 最后一步大奖励
    values = [1.0, 1.0, 1.0, 1.0, 1.0]                           # 价值估计（简化）

    adv = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95, done=True)

    if adv is None or adv.sum() == 0:
        print("  TODO 未完成，请补全 compute_gae 函数")
        return

    print(f"  测试数据: rewards={rewards}")
    print(f"  GAE 优势: {adv.tolist()}")
    print(f"  优势之和 ≈ {adv.sum().item():.4f}")

    # 验证: 最后一步的优势应该 ≈ 9.0
    # δ_4 = 10 + 0*0 - 1 = 9.0
    # gae_4 = 9.0 (因为之后没有步骤了)
    print(f"\n  验证:")
    print(f"    优势[4] (最后一步) ≈ {adv[4].item():.4f} (预期 ≈ 9.0)")
    if abs(adv[4].item() - 9.0) < 0.5:
        print(f"    ✓ 最后一步优势正确!")
    else:
        print(f"    ✗ 最后一步优势错误: 预期 ≈9.0")

    # 验证梯度传导: 前几步的优势应该为正（因为最后有大奖励）
    # GAE 会将未来的好信号反向传播
    all_positive = all(a > 0 for a in adv)
    print(f"    所有优势 > 0: {'✓ 是' if all_positive else '✗ 否'}")
    if all_positive:
        print(f"    ✓ GAE 正确地将大奖励的价值传播到了前几步!")

    # 手动验证第一步
    # δ_4=9.0, δ_3=0+0.99*1-1=-0.01, δ_2=-0.01, δ_1=-0.01, δ_0=-0.01
    # gae_4 = 9.0
    # gae_3 = -0.01 + 0.99*0.95*9.0 ≈ 8.46
    # gae_2 = -0.01 + 0.99*0.95*8.46 ≈ 7.94
    # gae_1 = -0.01 + 0.99*0.95*7.94 ≈ 7.45
    # gae_0 = -0.01 + 0.99*0.95*7.45 ≈ 6.99
    print(f"\n  手动计算 (前几步):")
    expected_first = -0.01 + 0.99 * 0.95 * (-0.01 + 0.99 * 0.95
                      * (-0.01 + 0.99 * 0.95 * (-0.01 + 0.99 * 0.95 * 9.0)))
    print(f"    adv[0] 预期 ≈ {expected_first:.4f}, 实际 = {adv[0].item():.4f}")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s21 RLHF：当强化学习遇见大模型 — 动手练习                 ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_ppo_clipped()
    test_dpo_loss()
    test_gae()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
    print()
    print("提示: 完成 TODO 后，可以运行 demo.py 查看完整的 PPO/DPO 训练流程。")
    print("  python code/demo.py")
    print()
    print("关键公式速查:")
    print("  PPO:  L_CLIP = E[min(r·Â, clip(r, 1-ε, 1+ε)·Â)]")
    print("  DPO:  L_DPO = -E[log σ(β·(log π_w/π_ref_w - log π_l/π_ref_l))]")
    print("  GAE:  Â_t = Σ (γλ)^l · δ_{t+l},  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)")
    print()
