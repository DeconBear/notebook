# -*- coding: utf-8 -*-
"""
s19 强化学习入门：MDP 与 Q-Learning — 练习代码
================================================
请完成以下 TODO 任务，巩固对 Q-Learning 核心机制的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md，再尝试独立补全代码。
运行方式：在 s19_rl_qlearning/ 目录下执行 python code/exercise.py
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# ============================================================================
# 辅助类：简化版 GridWorld（复用 demo 中的逻辑）
# ============================================================================

class MiniGridWorld:
    """
    简化版网格世界——用于练习。

    一个 5×5 网格，起点 (0,0)，终点 (4,4)，无陷阱。
    动作: 0=上, 1=下, 2=左, 3=右
    """

    def __init__(self):
        self.size = 5                                          # 5×5 小网格
        self.start = (0, 0)                                    # 起点
        self.goal = (4, 4)                                     # 终点
        self.state = self.start                                # 当前状态
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上/下/左/右
        self.action_names = ["上", "下", "左", "右"]

    def reset(self):
        """重置环境到起点。"""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行动作。到达终点获得 +10 奖励，每步 -0.1。
        """
        dr, dc = self.action_deltas[action]
        new_r = self.state[0] + dr
        new_c = self.state[1] + dc
        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.state = (new_r, new_c)
        if self.state == self.goal:
            return self.state, 10.0, True                       # 到达终点
        else:
            return self.state, -0.1, False                      # 普通步数

    def get_state_index(self, state: Tuple[int, int]) -> int:
        """状态 → 行优先索引。"""
        return state[0] * self.size + state[1]


# ============================================================================
# TODO 1: 实现 Q-Learning 的 TD 更新规则
# ============================================================================
def q_learning_update(
    q_table: np.ndarray,
    state_idx: int,
    action: int,
    reward: float,
    next_state_idx: int,
    done: bool,
    alpha: float = 0.1,
    gamma: float = 0.95,
) -> np.ndarray:
    """
    TODO 1: 实现 Q-Learning 的单步 TD 更新。

    更新公式:
        Q(s,a) ← Q(s,a) + α × (r + γ × max_{a'} Q(s',a') - Q(s,a))

    其中:
        - 如果 done=True（终止状态），则 max_{a'} Q(s',a') = 0
        - 否则，max_{a'} Q(s',a') 是下一状态 Q 值中最大的那个

    参数:
        q_table: Q 值表，shape (n_states, n_actions)
        state_idx: 当前状态索引 s
        action: 执行的动作 a
        reward: 获得的奖励 r
        next_state_idx: 下一状态索引 s'
        done: 是否到达终止状态
        alpha: 学习率 α
        gamma: 折扣因子 γ

    返回:
        q_table: 更新后的 Q 表（原地修改也需返回）
    """
    # TODO: 补全以下代码

    # 步骤 1: 获取当前 Q(s, a) 值
    current_q = None  # ← TODO: q_table[state_idx, action]

    # 步骤 2: 计算 TD 目标
    # 提示: 如果 done=True，TD 目标 = reward
    #       如果 done=False，TD 目标 = reward + gamma * max(Q[next_state_idx, :])
    if done:
        td_target = None  # ← TODO: 终止状态: TD 目标只有即时奖励
    else:
        max_next_q = None  # ← TODO: np.max(q_table[next_state_idx])
        td_target = None   # ← TODO: reward + gamma * max_next_q

    # 步骤 3: 计算 TD 误差 δ = td_target - current_q
    td_error = None  # ← TODO: td_target - current_q

    # 步骤 4: 更新 Q(s, a)
    # q_table[state_idx, action] += alpha * td_error
    q_table[state_idx, action] = None  # ← TODO: 实现更新

    return q_table


# ---- 测试 TODO 1 ----
def test_q_update():
    """测试 Q-Learning 更新规则。"""
    print("=" * 60)
    print("TODO 1 测试: Q-Learning 的 TD 更新规则")
    print("=" * 60)

    # 创建一个小型 Q 表: 3 个状态，2 个动作
    n_states, n_actions = 3, 2
    q_table = np.zeros((n_states, n_actions))

    # 测试 1: 非终止状态下的更新
    # 状态 0 执行动作 1 → 到达状态 1，奖励 +5
    q_before = q_table[0, 1].copy()
    q_table = q_learning_update(
        q_table, state_idx=0, action=1, reward=5.0,
        next_state_idx=1, done=False, alpha=0.1, gamma=0.9
    )

    if q_table[0, 1] != 0.0:
        print(f"  测试 1 [非终止状态]:")
        print(f"    更新前 Q(0,1) = {q_before}")
        print(f"    更新后 Q(0,1) = {q_table[0,1]:.4f}")
        print(f"    预期: 0.0 + 0.1 × (5.0 + 0.9 × 0 - 0.0) = 0.5")
        if abs(q_table[0, 1] - 0.5) < 0.001:
            print(f"    ✓ 测试通过!")
        else:
            print(f"    ✗ 测试失败: 期望 0.5, 得到 {q_table[0,1]:.4f}")
    else:
        print("  TODO 未完成，请补全 q_learning_update 函数")

    # 测试 2: 终止状态下的更新
    q_table = np.zeros((n_states, n_actions))                  # 重置 Q 表
    q_table[1, 0] = 1.0                                        # 设置一个初始值
    q_table = q_learning_update(
        q_table, state_idx=1, action=0, reward=10.0,
        next_state_idx=2, done=True, alpha=0.5, gamma=0.9
    )

    if q_table[1, 0] != 1.0:
        print(f"  测试 2 [终止状态]:")
        print(f"    更新前 Q(1,0) = 1.0")
        print(f"    更新后 Q(1,0) = {q_table[1,0]:.4f}")
        # Q = 1.0 + 0.5 × (10.0 - 1.0) = 5.5
        print(f"    预期: 1.0 + 0.5 × (10.0 - 1.0) = 5.5")
        if abs(q_table[1, 0] - 5.5) < 0.001:
            print(f"    ✓ 测试通过!")
        else:
            print(f"    ✗ 测试失败: 期望 5.5, 得到 {q_table[1,0]:.4f}")
    else:
        print("  TODO 未完成，请补全 q_learning_update 函数 (done=True 分支)")

    print()


# ============================================================================
# TODO 2: 实现 ε-贪婪动作选择
# ============================================================================
def epsilon_greedy_action(
    q_table: np.ndarray,
    state_idx: int,
    epsilon: float,
) -> int:
    """
    TODO 2: 实现 ε-贪婪策略的动作选择。

    规则:
        - 以概率 ε: 随机选择一个动作（探索）
        - 以概率 1-ε: 选择 Q 值最大的动作（利用）
        - 当多个动作 Q 值相等时，随机选择其中一个

    参数:
        q_table: Q 值表，shape (n_states, n_actions)
        state_idx: 当前状态索引
        epsilon: 探索率 (0 ≤ ε ≤ 1)

    返回:
        action: 选择的动作索引
    """
    # TODO: 补全以下代码

    # 提示: 使用 np.random.random() 生成 [0,1) 之间的随机数
    # 如果 random < epsilon: 随机选择动作
    # 否则: 选择 argmax Q(s, :)
    # 提示: 当多个动作 Q 值相同时，用 np.argmax 只返回第一个，
    #       可以用 np.where(q_vals == q_vals.max())[0] 获取所有最大值，再随机选

    action = None  # ← TODO: 实现 ε-贪婪动作选择
    return action


# ---- 测试 TODO 2 ----
def test_epsilon_greedy():
    """测试 ε-贪婪动作选择。"""
    print("=" * 60)
    print("TODO 2 测试: ε-贪婪动作选择")
    print("=" * 60)

    # 创建 Q 表: 1 个状态，4 个动作
    q_table = np.array([[1.0, 5.0, 1.0, 5.0]])                # 动作 1 和 3 都是最大值

    # 测试 1: ε=0 → 100% 利用，应该选动作 1 或 3
    if epsilon_greedy_action(q_table, 0, 0.0) is None:
        print("  TODO 未完成，请补全 epsilon_greedy_action 函数")
    else:
        actions_chosen = [epsilon_greedy_action(q_table, 0, 0.0)
                         for _ in range(100)]
        unique = set(actions_chosen)
        if unique == {1, 3} or unique == {1} or unique == {3}:
            print(f"  测试 1 [ε=0, 纯利用]: 100 次选择结果={sorted(unique)}")
            print(f"    ✓ 测试通过! (全部为最大 Q 值动作: 1 和/或 3)")
        else:
            print(f"  测试 1 [ε=0, 纯利用]: 100 次选择结果={sorted(unique)}")
            print(f"    ✗ 测试失败: 含非最大 Q 值动作")

        # 测试 2: ε=1 → 100% 探索，应该覆盖所有 4 个动作
        actions_chosen = [epsilon_greedy_action(q_table, 0, 1.0)
                         for _ in range(500)]
        unique = set(actions_chosen)
        if unique == {0, 1, 2, 3}:
            print(f"  测试 2 [ε=1, 纯探索]: 500 次选择的动作集合={sorted(unique)}")
            print(f"    ✓ 测试通过! (覆盖了全部 4 个动作)")
        else:
            print(f"  测试 2 [ε=1, 纯探索]: 500 次选择的动作集合={sorted(unique)}")
            print(f"    ✗ 测试失败: 未覆盖全部动作")

        # 测试 3: ε=0.5 → 混合，约 50% 探索 + 50% 利用
        actions_chosen = [epsilon_greedy_action(q_table, 0, 0.5)
                         for _ in range(1000)]
        exploit_count = sum(1 for a in actions_chosen if a in [1, 3])  # 利用动作
        explore_count = sum(1 for a in actions_chosen if a in [0, 2])  # 探索动作
        print(f"  测试 3 [ε=0.5, 混合]: 利用={exploit_count}, 探索={explore_count}")
        print(f"    (预期约为 500 探索 + 500 利用)")

    print()


# ============================================================================
# TODO 3: 添加新环境特征——动态障碍/陷阱
# ============================================================================
def add_moving_obstacle(
    env: MiniGridWorld,
    cur_episode: int,
):
    """
    TODO 3: 为 MiniGridWorld 添加动态变化的障碍物。

    任务:
        创建一个"移动障碍"机制，障碍物的位置随 episode 编号变化，
        而不是固定在某个位置。Agent 踩到障碍物时获得 -5 的惩罚。

    具体要求:
    1. 设计一个函数，根据 cur_episode（当前 episode 编号）返回障碍物位置列表
    2. 障碍物位置随 episode 周期性地改变（例如每 100 个 episode 换一个位置）
    3. 在 step() 函数中添加障碍物检测逻辑
    4. 修改训练循环，每 episode 开始时调用此函数更新障碍

    参数:
        env: MiniGridWorld 环境实例
        cur_episode: 当前 episode 编号

    返回:
        obstacle_positions: 当前 episode 的障碍物位置列表 [(row,col), ...]
    """
    # TODO: 实现动态障碍机制
    # 提示 1: 使用 cur_episode // 100 作为周期索引
    # 提示 2: 定义几个障碍物位置候选集，根据周期索引选择不同的集合
    # 提示 3: 障碍物不能与起点 (0,0) 或终点 (4,4) 重合

    periodic_idx = cur_episode // 100                           # 每 100 episode 更换

    # 定义几个障碍物配置（不覆盖起点和终点）
    obstacle_configs = [
        [(1, 1), (3, 3)],                                       # 配置 0
        [(1, 3), (3, 1)],                                       # 配置 1
        [(2, 2)],                                                # 配置 2
        [(1, 2), (2, 1), (3, 2)],                               # 配置 3
    ]

    # TODO: 根据 periodic_idx 选择障碍物配置
    # idx = periodic_idx % len(obstacle_configs)
    # return obstacle_configs[idx]

    obstacles = None  # ← TODO: 返回对应周期的障碍物列表
    return obstacles


# ---- 测试 TODO 3 ----
def test_moving_obstacle():
    """测试动态障碍功能。"""
    print("=" * 60)
    print("TODO 3 测试: 动态障碍物")
    print("=" * 60)

    env = MiniGridWorld()

    # 测试不同 episode 下障碍物是否变化
    test_episodes = [0, 50, 100, 199, 300, 399]
    results = []
    for ep in test_episodes:
        obs = add_moving_obstacle(env, ep)
        results.append((ep, obs))

    if results[0][1] is None:
        print("  TODO 未完成，请补全 add_moving_obstacle 函数")
        return

    for ep, obs in results:
        if obs:
            print(f"  Episode {ep:4d} → 障碍物: {obs}")
        else:
            print(f"  Episode {ep:4d} → 障碍物: [] (无障碍)")

    # 验证周期性——episodes 0 和 100 应该有相同配置（同属周期 0）
    if results[0][1] == results[2][1]:
        print("  ✓ 周期性正确! Episode 0 和 100 配置相同")
    else:
        print("  ✗ 周期性可能有问题: Episode 0 和 100 的配置不同")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s19 强化学习入门：MDP 与 Q-Learning — 动手练习            ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_q_update()
    test_epsilon_greedy()
    test_moving_obstacle()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
    print()
    print("提示: 完成 TODO 后，可以回到 demo.py 查看完整实现作为参考。")
    print()
