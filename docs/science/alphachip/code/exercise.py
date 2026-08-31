# -*- coding: utf-8 -*-
"""
===============================================================================
as07_alphachip_eda/code/exercise.py — AlphaChip 练习：线长计算 + REINFORCE
===============================================================================
本练习聚焦芯片布局强化学习的两个核心计算：
  任务1：实现总线长（wirelength）计算 —— 强化学习的奖励信号来源
  任务2：实现带掩码的放置动作采样 —— 保证每个格子只能被占用一次
  任务3（Bonus）：实现 REINFORCE 损失（带滑动平均基线），训练一个迷你放置策略

运行方式：python exercise.py
===============================================================================
"""

import numpy as np
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_MACROS = 8
GRID_SIZE = 3  # 3x3=9 个格子，够放 8 个宏单元（留1个空位）


def generate_toy_netlist(n_macros=N_MACROS, seed=42):
    """生成一个简单的合成 netlist（已实现，可直接使用）。"""
    rng = np.random.RandomState(seed)
    edges = []
    for i in range(n_macros):
        for j in range(i + 1, n_macros):
            if rng.random() < 0.35:
                edges.append((i, j, rng.uniform(1.0, 3.0)))
    return edges


EDGES = generate_toy_netlist()


# ============================================================================
# 任务 1: 实现总线长计算 (约 4 行核心代码)
# ============================================================================

def wirelength(positions, edges):
    """
    计算总线长：对每条边，用 Manhattan 距离近似连线长度，乘以权重后求和。

    参数:
        positions: (n_macros, 2) 的数组或列表，每个宏单元的 (row, col) 坐标
        edges: list of (i, j, weight)
    返回:
        total: float

    实现步骤:
      对每条边 (i, j, w):
        d = |positions[i][0] - positions[j][0]| + |positions[i][1] - positions[j][1]|
        累加 w * d 到 total
    """
    total = 0.0
    # TODO: 完成累加逻辑
    # --- BEGIN YOUR CODE ---
    total = None
    # --- END YOUR CODE ---
    return total


# ============================================================================
# 任务 2: 实现带掩码的动作采样 (约 4 行核心代码)
# ============================================================================

def sample_masked_action(logits, occupied_mask):
    """
    从 logits 中采样一个动作（网格位置），但已被占用的位置概率必须为 0。

    参数:
        logits: (n_cells,) 未归一化的分数
        occupied_mask: (n_cells,) 0/1 张量，1 表示该位置已被占用
    返回:
        cell: int，采样到的格子编号
        log_prob: 0维 tensor，该动作的对数概率（用于后续 REINFORCE 梯度计算）

    实现步骤:
      1. 用 masked_fill 把 occupied_mask 对应位置的 logits 设为 -inf：
         masked_logits = logits.masked_fill(occupied_mask.bool(), float('-inf'))
      2. dist = torch.distributions.Categorical(logits=masked_logits)
      3. cell = dist.sample()
      4. log_prob = dist.log_prob(cell)
    """
    # TODO: 完成以下 4 个步骤
    # --- BEGIN YOUR CODE ---
    cell, log_prob = None, None
    # --- END YOUR CODE ---
    return cell, log_prob


# ============================================================================
# 任务 3 (Bonus): 实现 REINFORCE 损失
# ============================================================================

class TinyPlacementPolicy(nn.Module):
    """一个极简策略网络：直接学习每个宏单元在每个格子上的偏好分数（已实现）。"""

    def __init__(self, n_macros, n_cells):
        super().__init__()
        self.logits_table = nn.Parameter(torch.randn(n_macros, n_cells) * 0.1)

    def forward(self, macro_id, occupied_mask):
        return self.logits_table[macro_id]


def compute_reinforce_loss(log_probs, reward, baseline):
    """
    TODO (Bonus): 计算 REINFORCE 损失（带基线降方差）。

    公式: L = -(reward - baseline) * sum(log_probs)

    参数:
        log_probs: list of 0维 tensor，一个episode中每一步动作的对数概率
        reward: float，该 episode 的总奖励（这里用 -wirelength）
        baseline: float，滑动平均基线
    返回:
        loss: 0维 tensor，可以直接调用 .backward()
    """
    # TODO: 完成损失计算
    # --- BEGIN YOUR CODE ---
    loss = None
    # --- END YOUR CODE ---
    return loss


def train_toy_placement(n_episodes=500, lr=0.05):
    """用任务1-3实现的组件，训练一个迷你放置策略（已实现训练循环骨架）。"""
    n_cells = GRID_SIZE * GRID_SIZE
    order = list(range(N_MACROS))
    policy = TinyPlacementPolicy(N_MACROS, n_cells)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    baseline = None
    wl_history = []
    for ep in range(n_episodes):
        occupied = torch.zeros(n_cells)
        positions = np.zeros((N_MACROS, 2), dtype=int)
        log_probs = []
        for macro_id in order:
            logits = policy(macro_id, occupied)
            cell, log_prob = sample_masked_action(logits, occupied)
            if cell is None:
                print('[提示] TODO 2 (sample_masked_action) 尚未实现，无法继续训练。')
                return None
            log_probs.append(log_prob)
            occupied[cell] = 1.0
            positions[macro_id] = [cell // GRID_SIZE, cell % GRID_SIZE]

        wl = wirelength(positions, EDGES)
        if wl is None:
            print('[提示] TODO 1 (wirelength) 尚未实现，无法继续训练。')
            return None
        reward = -wl
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward

        loss = compute_reinforce_loss(log_probs, reward, baseline)
        if loss is None:
            print('[提示] TODO 3 (compute_reinforce_loss) 尚未实现，无法继续训练。')
            return None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wl_history.append(wl)

    return wl_history


# ============================================================================
# 验证代码
# ============================================================================

def test_wirelength():
    print("[测试 1] 总线长计算...")
    positions = [[0, 0], [0, 1], [1, 0]]
    edges = [(0, 1, 1.0), (0, 2, 2.0)]  # d(0,1)=1, d(0,2)=1 -> total = 1*1 + 2*1 = 3
    total = wirelength(positions, edges)
    assert total is not None, "wirelength 返回了 None，请完成 TODO"
    assert abs(total - 3.0) < 1e-6, f"计算不正确: {total}, 期望 3.0"
    print(f"  [PASS] 总线长计算正确: {total}")


def test_sample_masked_action():
    print("[测试 2] 带掩码的动作采样...")
    torch.manual_seed(0)
    logits = torch.zeros(5)
    occupied = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])  # 只有位置2可用
    for _ in range(10):
        cell, log_prob = sample_masked_action(logits, occupied)
        assert cell is not None, "sample_masked_action 返回了 None，请完成 TODO"
        assert cell == 2, f"应只能采样到位置2（唯一未占用），实际采到 {cell}"
    print("  [PASS] 掩码采样正确，10次采样均落在唯一可用位置")


def test_train_toy_placement():
    print("[测试 3 (Bonus)] 迷你放置策略训练...")
    wl_history = train_toy_placement(n_episodes=500, lr=0.05)
    if wl_history is None:
        print("  [SKIP] 依赖的 TODO 尚未全部完成，跳过。")
        return
    early_avg = np.mean(wl_history[:20])
    late_avg = np.mean(wl_history[-20:])
    assert late_avg < early_avg, f"训练后线长应比训练初期更短: 初期={early_avg:.2f}, 后期={late_avg:.2f}"
    print(f"  [PASS] 训练初期平均线长={early_avg:.2f} -> 训练后期平均线长={late_avg:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("as07_alphachip_eda exercise.py — AlphaChip 练习")
    print("=" * 60)
    try:
        test_wirelength()
        test_sample_masked_action()
        test_train_toy_placement()
        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
