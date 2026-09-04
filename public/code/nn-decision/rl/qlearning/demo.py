# -*- coding: utf-8 -*-
"""
s19 强化学习入门：MDP 与 Q-Learning — 演示代码
================================================
功能：从零实现 GridWorld 环境和 Q-Learning 算法，
      可视化 Q 值热力图、训练奖励曲线、最优策略路径。
      对比不同 ε 衰减策略和学习率的效果。

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s19_rl_qlearning/ 目录下执行 python code/demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
# 中文字体配置
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from typing import Tuple, List, Dict, Optional
import time

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第一部分：GridWorld 环境实现
# ============================================================================

class GridWorld:
    """
    网格世界环境 —— 一个经典的强化学习测试平台。

    Agent 在二维网格中移动，目标是到达终点，避免陷阱。
    每次移动有微小的步数惩罚，鼓励 Agent 学习最短路径。

    属性:
        size: 网格大小 (size × size)
        start: 起点坐标 (row, col)
        goal: 终点坐标 (row, col)
        traps: 陷阱坐标列表 [(row, col), ...]
        state: 当前状态 (row, col)
        action_space: 动作空间大小 (4: 上/下/左/右)
    """

    def __init__(
        self,
        size: int = 10,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (9, 9),
        traps: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -0.1,
        goal_reward: float = 100.0,
        trap_reward: float = -50.0,
    ):
        """
        初始化网格世界环境。

        参数:
            size: 网格尺寸，默认 10×10
            start: 起点坐标 (row, col)，默认 (0, 0)
            goal: 终点坐标 (row, col)，默认 (9, 9)
            traps: 陷阱坐标列表，默认 [(3,3), (5,5), (7,7)]
            step_reward: 每步的基础奖励（负数表示惩罚每步移动）
            goal_reward: 到达终点奖励
            trap_reward: 踩到陷阱奖励
        """
        self.size = size                                          # 网格大小
        self.start = start                                        # 起点位置
        self.goal = goal                                          # 终点位置
        self.traps = traps if traps is not None else [(3, 3), (5, 5), (7, 7)]  # 默认陷阱位置
        self.step_reward = step_reward                            # 每步惩罚
        self.goal_reward = goal_reward                            # 终点奖励
        self.trap_reward = trap_reward                            # 陷阱惩罚
        self.action_space = 4                                     # 4 个离散动作
        self.state = start                                        # 初始化当前位置
        # 动作到坐标偏移的映射: 0=上, 1=下, 2=左, 3=右
        self.action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dr, dc) 偏移
        self.action_names = ["上", "下", "左", "右"]               # 动作中文名

    def reset(self) -> Tuple[int, int]:
        """
        重置环境：将 Agent 放回起点。

        返回:
            state: 初始状态 (row, col)
        """
        self.state = self.start  # 回到起点
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一个动作，返回 (下一状态, 奖励, 是否终止)。

        动作会被执行，但如果会导致移出网格边界，则 Agent 留在原地。
        到达终点或踩到陷阱时，episode 终止。

        参数:
            action: 动作索引 (0=上, 1=下, 2=左, 3=右)

        返回:
            next_state: 转移后的状态 (row, col)
            reward: 获得的即时奖励
            done: 是否终止 episode
        """
        dr, dc = self.action_deltas[action]             # 获取该动作的行列偏移
        new_r = self.state[0] + dr                      # 计算新行坐标
        new_c = self.state[1] + dc                      # 计算新列坐标

        # ---- 边界检查：如果移出网格，留在原地 ----
        if 0 <= new_r < self.size and 0 <= new_c < self.size:
            self.state = (new_r, new_c)                  # 更新位置
        # 否则 state 保持不变（撞墙）

        # ---- 判断奖励和终止条件 ----
        if self.state == self.goal:
            reward = self.goal_reward                    # 到达终点，获得大奖励
            done = True                                  # episode 结束
        elif self.state in self.traps:
            reward = self.trap_reward                    # 踩到陷阱，获得负奖励
            done = True                                  # episode 结束
        else:
            reward = self.step_reward                    # 普通移动，获得步数惩罚
            done = False                                 # 继续探索

        return self.state, reward, done

    def get_state_index(self, state: Tuple[int, int]) -> int:
        """
        将 (row, col) 状态转换为 Q-Table 的行索引。

        参数:
            state: 状态坐标 (row, col)

        返回:
            index: 0 到 size*size-1 之间的整数索引
        """
        return state[0] * self.size + state[1]           # 行优先编码: index = row * size + col


# ============================================================================
# 第二部分：Q-Learning Agent 实现
# ============================================================================

class QLearningAgent:
    """
    Q-Learning Agent —— 用表格方法学习最优策略。

    核心数据结构是一个 2D numpy 数组 Q[s][a]，
    其中 s 是状态索引（0 到 n_states-1），a 是动作索引（0 到 n_actions-1）。

    属性:
        q_table: Q 值表，shape (n_states, n_actions)
        epsilon: 当前探索率
        epsilon_init: 初始探索率
        epsilon_min: 最小探索率
        epsilon_decay: 每次 episode 后 epsilon 的衰减因子
        alpha: 学习率
        gamma: 折扣因子
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        初始化 Q-Learning Agent。

        参数:
            n_states: 状态总数 (size × size)
            n_actions: 动作总数 (4)
            alpha: 学习率，控制每次更新的步长
            gamma: 折扣因子，控制对未来奖励的重视程度
            epsilon_init: 初始探索概率
            epsilon_min: 最小探索概率
            epsilon_decay: 探索率衰减因子（每次 episode 乘以此值）
        """
        self.n_states = n_states                               # 状态空间大小
        self.n_actions = n_actions                             # 动作空间大小
        self.alpha = alpha                                     # 学习率 α
        self.gamma = gamma                                     # 折扣因子 γ
        self.epsilon = epsilon_init                            # 当前探索率 ε
        self.epsilon_init = epsilon_init                       # 初始探索率
        self.epsilon_min = epsilon_min                         # 最小探索率
        self.epsilon_decay = epsilon_decay                     # 探索率衰减因子
        # Q-Table 初始化为全零: shape (n_states, n_actions)
        self.q_table = np.zeros((n_states, n_actions))         # Q(s, a) 查找表

    def choose_action(self, state_idx: int) -> int:
        """
        ε-贪婪策略选择动作。

        以概率 ε 随机选择动作（探索），以概率 1-ε 选择 Q 值最大的动作（利用）。

        参数:
            state_idx: 当前状态的索引

        返回:
            action: 选择的动作索引 (0-3)
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择任意动作
            action = np.random.randint(self.n_actions)         # 均匀随机采样
        else:
            # 利用：选择当前 Q 值最高的动作
            action = np.argmax(self.q_table[state_idx])        # argmax 贪婪选择
        return action

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ):
        """
        执行 Q-Learning 的 TD 更新。

        更新公式:
            Q(s,a) ← Q(s,a) + α × (r + γ × max_{a'} Q(s',a') - Q(s,a))

        如果下一状态是终止状态（done=True），则 TD 目标不包含未来价值项。

        参数:
            state_idx: 当前状态索引 s
            action: 执行的动作 a
            reward: 获得的奖励 r
            next_state_idx: 下一状态索引 s'
            done: 是否到达终止状态
        """
        current_q = self.q_table[state_idx, action]            # 当前 Q(s, a) 值

        if done:
            # 终止状态：TD 目标 = r（没有下一状态，未来价值为 0）
            td_target = reward                                  # TD 目标 = 即时奖励
        else:
            # 非终止状态：TD 目标 = r + γ × max_{a'} Q(s', a')
            max_next_q = np.max(self.q_table[next_state_idx])   # max_{a'} Q(s', a')
            td_target = reward + self.gamma * max_next_q        # TD 目标

        td_error = td_target - current_q                        # TD 误差 δ
        # Q-Learning 更新规则
        self.q_table[state_idx, action] += self.alpha * td_error  # Q(s,a) += α × δ

    def decay_epsilon(self):
        """
        衰减探索率 ε。

        每次 episode 结束时调用，让 Agent 逐渐从探索转向利用。
        ε = max(ε_min, ε × decay)
        """
        self.epsilon = max(self.epsilon_min,                   # 不低于最小探索率
                           self.epsilon * self.epsilon_decay)   # 指数衰减

    def reset(self):
        """
        重置 Agent 的 Q-Table 和探索率，用于多次实验。
        """
        self.q_table = np.zeros((self.n_states, self.n_actions))  # 清零 Q 表
        self.epsilon = self.epsilon_init                           # 重置探索率


# ============================================================================
# 第三部分：训练循环
# ============================================================================

def train_agent(
    env: GridWorld,
    agent: QLearningAgent,
    n_episodes: int = 2000,
    max_steps: int = 500,
    record_history: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    训练 Q-Learning Agent。

    每个 episode 从起点开始，执行动作直到到达终点、踩到陷阱或超过最大步数。

    参数:
        env: 网格世界环境
        agent: Q-Learning Agent
        n_episodes: 训练的 episode 总数
        max_steps: 每个 episode 的最大步数（防止无限循环）
        record_history: 是否记录训练历史（奖励、路径等）
        verbose: 是否打印训练进度

    返回:
        history: 包含 episode_rewards, episode_lengths, epsilon_history,
                q_table_snapshots, optimal_path 的字典
    """
    # ---- 初始化训练记录 ----
    episode_rewards = []                                       # 每个 episode 的总奖励
    episode_lengths = []                                       # 每个 episode 的步数
    epsilon_history = []                                       # 每个 episode 的 ε 值
    q_table_snapshots = {}                                     # Q 表快照（在特定 episode 保存）
    # 记录 snapshot 的 episode 编号
    snapshot_episodes = [0, 50, 200, 500, n_episodes - 1]      # 哪些 episode 保存快照
    # 用于判断收敛：最近 N 个 episode 的平均奖励
    recent_rewards = []                                        # 滑动窗口
    window_size = 100                                          # 窗口大小
    converged_episode = None                                   # 收敛 episode 编号

    if verbose:
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║       s19 Q-Learning — GridWorld 训练开始                        ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"\n  环境: {env.size}×{env.size} 网格, "
              f"起点={env.start}, 终点={env.goal}")
        print(f"  陷阱: {env.traps}")
        print(f"  超参数: α={agent.alpha}, γ={agent.gamma}, "
              f"ε_init={agent.epsilon_init}, ε_decay={agent.epsilon_decay}")
        print(f"  训练 episodes: {n_episodes}, max_steps/episode: {max_steps}")
        print()

    start_time = time.time()                                   # 记录训练开始时间

    for ep in range(n_episodes):
        state = env.reset()                                    # 重置环境，回到起点
        state_idx = env.get_state_index(state)                 # 获取状态索引
        total_reward = 0                                       # 累计本 episode 的奖励
        steps = 0                                              # 本 episode 的步数
        path = [state]                                         # 记录路径（当前 episode）

        for step in range(max_steps):
            # ---- 选择动作 ----
            action = agent.choose_action(state_idx)            # ε-贪婪选择动作

            # ---- 执行动作 ----
            next_state, reward, done = env.step(action)        # 与环境交互
            next_state_idx = env.get_state_index(next_state)   # 下一状态索引

            # ---- Q-Learning 更新 ----
            agent.update(state_idx, action, reward,            # TD 更新 Q 表
                        next_state_idx, done)

            # ---- 记录 ----
            total_reward += reward                             # 累计奖励
            steps += 1                                         # 步数 +1
            path.append(next_state)                            # 记录路径
            state_idx = next_state_idx                         # 状态转移

            if done:
                break                                           # 到达终止状态，结束 episode

        # ---- Episode 结束后更新 ----
        agent.decay_epsilon()                                  # 衰减探索率
        episode_rewards.append(total_reward)                   # 记录总奖励
        episode_lengths.append(steps)                          # 记录步数
        epsilon_history.append(agent.epsilon)                  # 记录 ε 值

        # ---- 滑动窗口均值 ----
        recent_rewards.append(total_reward)
        if len(recent_rewards) > window_size:
            recent_rewards.pop(0)                               # 保持窗口大小

        # ---- 检测收敛 ----
        if (converged_episode is None
            and len(recent_rewards) >= window_size
            and np.mean(recent_rewards) > 0                    # 平均奖励大于 0 视为收敛
            and ep > 500):                                     # 至少训练 500 个 episode
            converged_episode = ep                              # 标记收敛 episode

        # ---- 保存 Q 表快照 ----
        if record_history and ep in snapshot_episodes:
            q_table_snapshots[ep] = agent.q_table.copy()       # 深拷贝 Q 表

        # ---- 打印进度 ----
        if verbose and (ep + 1) % 200 == 0:
            avg_reward = np.mean(recent_rewards)               # 最近 100 episode 的平均奖励
            print(f"  Episode {ep+1:4d}/{n_episodes}: "
                  f"ε={agent.epsilon:.3f}, "
                  f"avg_reward(100ep)={avg_reward:7.2f}, "
                  f"steps={steps:3d}")

    training_time = time.time() - start_time                   # 训练耗时

    # ---- 提取最优策略路径 ----
    optimal_path = extract_optimal_path(env, agent)            # 从起点按照 argmax Q 走

    if verbose:
        print(f"\n  ✓ 训练完成! 耗时: {training_time:.2f} 秒")
        if converged_episode is not None:
            print(f"  ✓ 约在第 {converged_episode} 个 episode 收敛")
        print(f"  ✓ 最优路径长度: {len(optimal_path)} 步")
        print(f"  ✓ 最终 ε = {agent.epsilon:.4f}")

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "epsilon_history": epsilon_history,
        "q_table_snapshots": q_table_snapshots,
        "optimal_path": optimal_path,
        "converged_episode": converged_episode,
        "training_time": training_time,
    }


def extract_optimal_path(env: GridWorld, agent: QLearningAgent) -> List[Tuple[int, int]]:
    """
    按照训练好的 Q 表提取最优路径。
    从起点出发，每一步选择 argmax Q(s,a)，直到到达终点或超过最大步数。

    参数:
        env: 网格世界环境
        agent: 已训练的 Q-Learning Agent

    返回:
        path: 最优路径上的状态坐标列表
    """
    state = env.start                                          # 从起点开始
    path = [state]                                             # 路径初始化
    visited = set()                                            # 访问过的状态集合（防循环）
    max_steps = env.size * env.size                            # 最大步数 = 状态总数

    for _ in range(max_steps):
        state_idx = env.get_state_index(state)                 # 当前状态索引
        if state_idx in visited:
            break                                               # 检测到循环，停止
        visited.add(state_idx)                                 # 标记已访问

        # 选择 Q 值最大的动作（纯利用，ε=0）
        action = np.argmax(agent.q_table[state_idx])           # argmax Q(s, a)
        dr, dc = env.action_deltas[action]                     # 获取偏移
        new_r = state[0] + dr                                  # 新行
        new_c = state[1] + dc                                  # 新列

        # 边界检查
        if 0 <= new_r < env.size and 0 <= new_c < env.size:
            state = (new_r, new_c)                              # 更新位置
        path.append(state)                                     # 记录路径

        if state == env.goal or state in env.traps:
            break                                               # 到达终点或陷阱，停止

    return path


# ============================================================================
# 第四部分：可视化
# ============================================================================

def plot_qvalue_heatmap(
    env: GridWorld,
    agent: QLearningAgent,
    episode_label: str,
    ax: plt.Axes,
    title: str = "Q-Value Heatmap",
):
    """
    Draw Q-value heatmap — display the max Q-value color for each grid cell.

    Uses Q-table data: higher Q-value cells are warmer (red), lower are cooler (blue).

    Parameters:
        env: GridWorld environment
        agent: Q-Learning Agent (or its Q-table snapshot)
        episode_label: episode label
        ax: matplotlib axes
        title: Chart title
    """
    # 提取每个状态的最大 Q 值作为该状态的"价值"
    if isinstance(agent, QLearningAgent):
        q_table = agent.q_table                                 # 当前 Q 表
    else:
        q_table = agent                                         # 直接传入的 Q 表快照

    # 计算每个状态的 max Q 值
    value_grid = np.max(q_table, axis=1).reshape(env.size, env.size)  # (size, size)

    # 设置 Q 值的颜色映射范围
    vmin = min(0, np.min(value_grid))                          # 下限至少为 0（或更低）
    vmax = max(1, np.max(value_grid))                          # 上限至少为 1

    # 绘制热力图
    im = ax.imshow(value_grid, cmap='RdYlBu_r',                # 红=高Q值, 蓝=低Q值
                   origin='upper', vmin=vmin, vmax=vmax,
                   aspect='equal')
    # 在每个格子中添加最大 Q 值文本
    for r in range(env.size):
        for c in range(env.size):
            val = value_grid[r, c]                             # 该状态的最大 Q 值
            if val != 0:
                ax.text(c, r, f'{val:.1f}', ha='center',      # 显示 Q 值
                       va='center', fontsize=6,
                       color='white' if abs(val) > vmax * 0.5 else 'black')

    # 标记起点、终点和陷阱
    ax.plot(env.start[1], env.start[0], 'go',                  # green dot = start
            markersize=10, label='Start')
    ax.plot(env.goal[1], env.goal[0], 'r*',                    # red star = goal
            markersize=15, label=f'Goal (+{env.goal_reward})')
    for trap in env.traps:
        ax.plot(trap[1], trap[0], 'kx', markersize=12,         # black x = trap
                mew=2, label='Trap' if trap == env.traps[0] else "")

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xticklabels(range(env.size))
    ax.set_yticklabels(range(env.size))
    ax.legend(loc='upper left', fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8, label='max Q(s,a)')   # 颜色条


def plot_optimal_policy(
    env: GridWorld,
    agent: QLearningAgent,
    ax: plt.Axes,
    title: str = "最优策略 pi*(s) = argmax_a Q(s,a)",
):
    """
    绘制最优策略图 —— 在每个格子上用箭头标明最优动作方向。

    策略: π*(s) = argmax_a Q(s, a)

    参数:
        env: 网格世界环境
        agent: 已训练的 Q-Learning Agent
        ax: matplotlib 坐标轴
        title: 图表标题
    """
    # 动作方向对应的箭头偏移
    arrow_deltas = {
        0: (0, -0.3),    # 上: 箭头朝上 (dx=0, dy<0)
        1: (0, 0.3),     # 下: 箭头朝下 (dx=0, dy>0)
        2: (-0.3, 0),    # 左: 箭头朝左 (dx<0, dy=0)
        3: (0.3, 0),     # 右: 箭头朝右 (dx>0, dy=0)
    }

    # 绘制网格背景
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()                                          # 让 (0,0) 在左上角

    # 绘制网格线
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)       # 水平线
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)       # 竖直线

    # 在每个格子上绘制最优动作箭头
    for r in range(env.size):
        for c in range(env.size):
            state_idx = env.get_state_index((r, c))            # 状态索引
            best_action = np.argmax(agent.q_table[state_idx])  # 该状态的最优动作
            q_val = agent.q_table[state_idx, best_action]      # 对应的 Q 值

            # 跳过终点和陷阱（这些是终止状态）
            if (r, c) == env.goal or (r, c) in env.traps:
                continue

            dy = -arrow_deltas[best_action][1]                 # imshow 从顶向下，y 方向要取反
            dx = arrow_deltas[best_action][0]                  # x 方向不变

            # 箭头颜色：Q 值越高越绿，越低越红
            color = 'green' if q_val > 0 else 'red'             # 正向/负向动作
            alpha = min(1.0, abs(q_val) / 50)                  # 透明度反映 Q 值的大小
            ax.arrow(c, r, dx, dy, head_width=0.15,            # 绘制箭头
                    head_length=0.15, fc=color, ec=color,
                    alpha=max(0.3, alpha), lw=2)

    # 标记特殊格子
    ax.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start S')     # green start
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=18, label=f'Goal G')      # red star goal
    for i, trap in enumerate(env.traps):
        ax.plot(trap[1], trap[0], 'ks', markersize=14, label=f'Trap X{i+1}')     # black square trap

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Column (col)')
    ax.set_ylabel('Row (row)')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(False)                                             # 关闭自动网格


def plot_training_rewards(
    episode_rewards: List[float],
    window_size: int = 50,
    title: str = "Training Reward Curve",
    ax: Optional[plt.Axes] = None,
):
    """
    绘制训练过程中的 Episode 奖励曲线。

    同时显示原始奖励（浅色）和滑动平均奖励（深色）。

    参数:
        episode_rewards: 每个 episode 的总奖励列表
        window_size: 滑动平均窗口大小
        title: 图表标题
        ax: matplotlib 坐标轴（如果为 None，则创建新图）
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    episodes = np.arange(len(episode_rewards))                 # episode 编号
    rewards = np.array(episode_rewards)                        # 转为 numpy 数组

    # 计算滑动平均
    if len(rewards) >= window_size:
        smoothed = np.convolve(rewards,                        # 卷积实现滑动平均
                              np.ones(window_size) / window_size,
                              mode='valid')
        smooth_episodes = np.arange(window_size - 1, len(rewards))

        ax.plot(smooth_episodes, smoothed, 'b-',               # blue solid = smoothed reward
                linewidth=2, label=f'Moving Avg (window={window_size})')

    ax.plot(episodes, rewards, 'lightblue', alpha=0.3,         # light blue = raw reward
            linewidth=0.5, label='Raw Reward')

    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5,      # y=0 reference line
              label='y=0 (Break-even)')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Total Reward', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_q_vs_episodes(
    q_snapshots: Dict[int, np.ndarray],
    env: GridWorld,
    title_prefix: str = "Q-Value Heatmap Evolution",
):
    """
    绘制多个 episode 的 Q 值热力图快照，展示学习过程。

    参数:
        q_snapshots: {episode: q_table_array} 字典
        env: 网格世界环境
        title_prefix: 标题前缀
    """
    n_snapshots = len(q_snapshots)
    if n_snapshots == 0:
        return

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 5))
    if n_snapshots == 1:
        axes = [axes]                                           # 处理单轴情况

    for ax, (ep, q_table) in zip(axes, q_snapshots.items()):
        # 提取该快照中每个状态的 max Q 值
        value_grid = np.max(q_table, axis=1).reshape(env.size, env.size)
        vmin = min(-1, np.min(value_grid))
        vmax = max(1, np.max(value_grid))

        im = ax.imshow(value_grid, cmap='RdYlBu_r',            # 热力图
                       origin='upper', vmin=vmin, vmax=vmax, aspect='equal')
        for r in range(env.size):
            for c in range(env.size):
                val = value_grid[r, c]
                if abs(val) > 0.5:
                    ax.text(c, r, f'{val:.0f}', ha='center',   # 显示整数 Q 值
                           va='center', fontsize=6,
                           color='white' if abs(val) > vmax * 0.4 else 'black')

        # 标记特殊格子
        ax.plot(env.start[1], env.start[0], 'go', markersize=8)
        ax.plot(env.goal[1], env.goal[0], 'r*', markersize=12)
        for trap in env.traps:
            ax.plot(trap[1], trap[0], 'kx', markersize=10, mew=2)

        ax.set_title(f'Episode {ep}\nε={0.995**ep:.3f}',       # 估算该 episode 的 ε
                    fontsize=10, fontweight='bold')
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'qvalue_heatmap_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] Q 值热力图演化已保存至 images/qvalue_heatmap_evolution.png")


def plot_path_on_grid(
    env: GridWorld,
    path: List[Tuple[int, int]],
    ax: plt.Axes,
    title: str = "Agent Optimal Path",
):
    """
    在网格世界上绘制 Agent 的移动路径。

    参数:
        env: 网格世界环境
        path: 路径列表 [(row, col), ...]
        ax: matplotlib 坐标轴
        title: 图表标题
    """
    # 创建网格背景
    grid = np.zeros((env.size, env.size))                      # 空网格
    im = ax.imshow(grid, cmap='Greys', vmin=0, vmax=1,         # 浅灰背景
                   origin='upper', aspect='equal', alpha=0.1)

    # 绘制路径线
    path_rows = [p[0] for p in path]                           # 路径的行坐标
    path_cols = [p[1] for p in path]                           # 路径的列坐标
    ax.plot(path_cols, path_rows, 'b-', linewidth=2,           # blue line connecting path
            alpha=0.7, label=f'Path ({len(path)} steps)')
    ax.plot(path_cols, path_rows, 'bo', markersize=5, alpha=0.5)  # blue dot node marker

    # Mark start point
    ax.plot(env.start[1], env.start[0], 'go', markersize=12,
            label=f'Start ({env.start[0]},{env.start[1]})')
    # Mark goal point
    ax.plot(env.goal[1], env.goal[0], 'r*', markersize=18,
            label=f'Goal ({env.goal[0]},{env.goal[1]})')
    # Mark traps
    for i, trap in enumerate(env.traps):
        ax.plot(trap[1], trap[0], 'ks', markersize=14,
                label=f'Trap {i+1}')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Column (col)')
    ax.set_ylabel('Row (row)')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_epsilon_comparison(
    results: Dict[str, Dict],
    title: str = "Comparison of Different ε Strategies",
):
    """
    Compare the training effects of different ε decay strategies.

    Parameters:
        results: dict {label: history_dict}
        title: Chart title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']      # color scheme

    # ---- Subplot 1: ε decay curve ----
    for (label, history), color in zip(results.items(), colors):
        axes[0, 0].plot(history['epsilon_history'],
                       color=color, linewidth=2, label=label)
    axes[0, 0].set_xlabel('Episode', fontsize=9)
    axes[0, 0].set_ylabel('ε (Exploration Rate)', fontsize=9)
    axes[0, 0].set_title('Exploration Rate Decay Curve', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # ---- Subplot 2: Reward curve comparison ----
    for (label, history), color in zip(results.items(), colors):
        rewards = np.array(history['episode_rewards'])
        if len(rewards) >= 100:
            smoothed = np.convolve(rewards,
                                  np.ones(100) / 100, mode='valid')
            axes[0, 1].plot(np.arange(99, len(rewards)), smoothed,
                          color=color, linewidth=2, label=label)
    axes[0, 1].set_xlabel('Episode', fontsize=9)
    axes[0, 1].set_ylabel('Avg Reward (window=100)', fontsize=9)
    axes[0, 1].set_title('Training Reward Comparison', fontsize=11, fontweight='bold')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # ---- Subplot 3: Episode length curve ----
    for (label, history), color in zip(results.items(), colors):
        lengths = np.array(history['episode_lengths'])
        if len(lengths) >= 100:
            smoothed = np.convolve(lengths.astype(float),
                                  np.ones(100) / 100, mode='valid')
            axes[1, 0].plot(np.arange(99, len(lengths)), smoothed,
                          color=color, linewidth=2, label=label)
    axes[1, 0].set_xlabel('Episode', fontsize=9)
    axes[1, 0].set_ylabel('Steps per Episode', fontsize=9)
    axes[1, 0].set_title('Episode Length Comparison', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    # ---- Subplot 4: Training time vs final performance ----
    labels = list(results.keys())
    times = [r['training_time'] for r in results.values()]
    final_rewards = [np.mean(results[l]['episode_rewards'][-100:])
                     for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = axes[1, 1].bar(x - width/2, times, width, label='Training Time (s)',
                           color='#2E86AB')
    axes[1, 1].set_xlabel('Strategy', fontsize=9)
    axes[1, 1].set_ylabel('Training Time (s)', fontsize=9, color='#2E86AB')
    ax2 = axes[1, 1].twinx()
    bars2 = ax2.bar(x + width/2, final_rewards, width,
                    label='Final Avg Reward', color='#F18F01')
    ax2.set_ylabel('Final Avg Reward (last 100ep)', fontsize=9, color='#F18F01')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=7)
    axes[1, 1].set_title('Training Time vs Final Performance', fontsize=11, fontweight='bold')

    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'epsilon_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] ε 策略对比图已保存至 images/epsilon_comparison.png")


def plot_learning_rate_comparison(
    results: Dict[str, Dict],
    title: str = "不同学习率 α 对比",
):
    """
    对比不同学习率对 Q-Learning 训练效果的影响。

    参数:
        results: 字典 {label: history_dict}
        title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    for (label, history), color in zip(results.items(), colors):
        rewards = np.array(history['episode_rewards'])
        if len(rewards) >= 100:
            smoothed = np.convolve(rewards,
                                  np.ones(100) / 100, mode='valid')
            ax.plot(np.arange(99, len(rewards)), smoothed,
                   color=color, linewidth=2, label=f'α={label}')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Avg Reward (window=100)', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'learning_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 学习率对比图已保存至 images/learning_rate_comparison.png")


# ============================================================================
# 第五部分：主程序
# ============================================================================

def main():
    """
    主程序：演示 Q-Learning 在 GridWorld 上的完整训练流程。

    流程：
    1. 创建 GridWorld 环境和 Q-Learning Agent
    2. 训练 Agent
    3. 可视化 Q 值热力图演化、训练奖励曲线、最优策略
    4. 对比不同 ε 衰减策略和学习率的效果
    """
    print("\n" + "=" * 70)
    print("    s19 强化学习入门：MDP 与 Q-Learning — 完整演示")
    print("=" * 70)

    # ========================================================================
    # 实验 1: 基础训练与可视化
    # ========================================================================
    print("\n【实验 1】基础 Q-Learning 训练\n")

    # ---- 1.1 创建环境 ----
    env = GridWorld(
        size=10,                                                # 10×10 网格
        start=(0, 0),                                           # 左上角起点
        goal=(9, 9),                                            # 右下角终点
        traps=[(3, 3), (5, 5), (7, 7)],                        # 对角线上的 3 个陷阱
        step_reward=-0.1,                                       # 每步罚 0.1 鼓励最短路径
        goal_reward=100.0,                                      # 到达终点奖励 100
        trap_reward=-50.0,                                      # 踩到陷阱罚 50
    )

    # ---- 1.2 创建 Agent ----
    n_states = env.size * env.size                              # 状态总数: 100
    n_actions = env.action_space                                # 动作总数: 4
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,                                              # 学习率 α
        gamma=0.95,                                             # 折扣因子 γ
        epsilon_init=1.0,                                       # 初始 ε=1.0 (100% 探索)
        epsilon_min=0.01,                                       # 最小 ε=0.01
        epsilon_decay=0.995,                                    # 每次 episode 乘 0.995
    )

    # ---- 1.3 训练 ----
    history = train_agent(
        env=env,
        agent=agent,
        n_episodes=2000,                                        # 训练 2000 个 episode
        max_steps=500,                                          # 每个 episode 最多 500 步
        verbose=True,
    )

    # ---- 1.4 可视化 ----
    print("\n[可视化] 生成图片...")

    # -- 可视化 1: Q 值热力图演化 --
    plot_q_vs_episodes(history['q_table_snapshots'], env,
                       title_prefix='Q-Value Heatmap Evolution')

    # -- Viz 2: Overview (training reward + optimal policy + path) --
    fig = plt.figure(figsize=(16, 5))

    # Subplot 2a: Training reward curve
    ax1 = fig.add_subplot(1, 3, 1)
    plot_training_rewards(history['episode_rewards'], ax=ax1,
                         title='Training Reward Curve (Moving Avg window=50)')

    # Subplot 2b: Optimal policy
    ax2 = fig.add_subplot(1, 3, 2)
    plot_optimal_policy(env, agent, ax=ax2,
                       title='Optimal Policy pi* (arrow=best action)')

    # Subplot 2c: Optimal path
    ax3 = fig.add_subplot(1, 3, 3)
    plot_path_on_grid(env, history['optimal_path'], ax=ax3,
                     title=f'Optimal Path ({len(history["optimal_path"])} steps)')

    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'training_results_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] 训练结果总览已保存至 images/training_results_overview.png")

    # ========================================================================
    # 实验 2: 不同 ε 衰减策略对比
    # ========================================================================
    print("\n【实验 2】不同 ε 衰减策略对比\n")

    epsilon_configs = {
        "快速衰减 (decay=0.99)": {
            "epsilon_decay": 0.99,                              # 快速衰减
            "description": "ε 衰减快 → 早期转向利用，可能陷入次优"
        },
        "中等衰减 (decay=0.995)": {
            "epsilon_decay": 0.995,                             # 中等衰减（默认）
            "description": "平衡探索与利用"
        },
        "慢速衰减 (decay=0.999)": {
            "epsilon_decay": 0.999,                             # 慢速衰减
            "description": "ε 衰减慢 → 长时间探索，收敛较慢但稳定"
        },
    }

    epsilon_results = {}
    for label, config in epsilon_configs.items():
        print(f"  训练: {label} (decay={config['epsilon_decay']})")
        env_test = GridWorld(size=10, start=(0,0), goal=(9,9),
                            traps=[(3,3),(5,5),(7,7)],
                            step_reward=-0.1, goal_reward=100.0, trap_reward=-50.0)
        agent_test = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=0.1,
            gamma=0.95,
            epsilon_init=1.0,
            epsilon_min=0.01,
            epsilon_decay=config['epsilon_decay'],
        )
        result = train_agent(env_test, agent_test,
                            n_episodes=2000, max_steps=500,
                            verbose=False)
        final_reward = np.mean(result['episode_rewards'][-100:])
        print(f"    完成: 最终平均奖励={final_reward:.2f}, 耗时={result['training_time']:.1f}s")
        epsilon_results[label] = result

    plot_epsilon_comparison(epsilon_results, title='Comparison of Different ε Strategies')

    # ========================================================================
    # 实验 3: 不同学习率对比
    # ========================================================================
    print("\n【实验 3】不同学习率 α 对比\n")

    alpha_configs = {
        "0.05": 0.05,                                           # 小学习率
        "0.1": 0.1,                                             # 默认
        "0.3": 0.3,                                             # 中等
        "0.5": 0.5,                                             # 较大学习率
    }

    alpha_results = {}
    for label, alpha in alpha_configs.items():
        print(f"  训练: α={alpha}")
        env_test = GridWorld(size=10, start=(0,0), goal=(9,9),
                            traps=[(3,3),(5,5),(7,7)],
                            step_reward=-0.1, goal_reward=100.0, trap_reward=-50.0)
        agent_test = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,                                        # 不同的学习率
            gamma=0.95,
            epsilon_init=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
        )
        result = train_agent(env_test, agent_test,
                            n_episodes=2000, max_steps=500,
                            verbose=False)
        final_reward = np.mean(result['episode_rewards'][-100:])
        print(f"    完成: 最终平均奖励={final_reward:.2f}, 耗时={result['training_time']:.1f}s")
        alpha_results[label] = result

    plot_learning_rate_comparison(alpha_results,
                                  title='Effect of Different Learning Rates α on Q-Learning')

    # ========================================================================
    # 最终总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)
    print(f"  ✓ 环境: {env.size}×{env.size} 网格, {len(env.traps)} 个陷阱")
    print(f"  ✓ Agent: Q-Learning, α={agent.alpha}, γ={agent.gamma}")
    print(f"  ✓ 最终 ε = {agent.epsilon:.4f}")
    print(f"  ✓ 最优路径长度: {len(history['optimal_path'])} 步")
    if history['converged_episode'] is not None:
        print(f"  ✓ 收敛于 Episode {history['converged_episode']}")
    print(f"  ✓ 最终 100 episode 平均奖励: "
          f"{np.mean(history['episode_rewards'][-100:]):.2f}")
    print(f"\n  核心要点:")
    print(f"  1. Q-Learning 通过 TD 更新: Q(s,a) += α(r + γ·maxQ(s',a') - Q(s,a))")
    print(f"  2. ε-贪婪策略平衡探索与利用")
    print(f"  3. 奖励信号从目标状态向起点反向传播（价值传播）")
    print(f"  4. ε 衰减过快 → 探索不足；过慢 → 收敛慢")
    print(f"  5. α 太大 → 不稳定；太小 → 学习慢")
    print(f"\n  局限性:")
    print(f"  • 表格方法仅适用于离散小状态空间")
    print(f"  • 无法在状态间泛化（两个相似状态需要分别学习）")
    print(f"  • 下一节将用神经网络 (DQN) 突破这些限制")
    print("=" * 70)

    # ---- 展示效果：打印前几个状态的最优 Q 值 ----
    print("\n【最优 Q 值示例 (起点附近)】")
    print("-" * 50)
    for r in range(3):                                         # 前 3 行
        for c in range(3):                                     # 前 3 列
            state_idx = env.get_state_index((r, c))
            q_vals = agent.q_table[state_idx]                  # 该状态的 4 个 Q 值
            best_action = np.argmax(q_vals)                    # 最优动作
            print(f"  状态 ({r},{c}): "
                  f"Q=[{q_vals[0]:6.2f}, {q_vals[1]:6.2f}, "
                  f"{q_vals[2]:6.2f}, {q_vals[3]:6.2f}], "
                  f"最佳动作: {env.action_names[best_action]} "
                  f"(Q={q_vals[best_action]:.2f})")
    print("-" * 50)

    print("\n  所有图片已保存至 images/ 目录")
    print("  运行完成！\n")


if __name__ == "__main__":
    main()
