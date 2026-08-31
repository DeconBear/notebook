# -*- coding: utf-8 -*-
from __future__ import annotations
"""
s20 深度强化学习：DQN 与 Policy Gradient — 演示代码
======================================================
功能：
  1. 从零实现 DQN (Deep Q-Network) 用于 CartPole-v1
     - Q-network (全连接网络)
     - 经验回放缓冲区
     - 目标网络定期更新
     - ε-贪婪探索（带衰减）
  2. 从零实现 REINFORCE (策略梯度) 用于 CartPole-v1
     - Policy network 输出动作概率分布
     - Monte Carlo 回报计算
  3. 对比两种算法的训练效率和最终性能
  4. 可视化训练奖励曲线、若干 episode 的 CartPole 动画

每个函数都有中文 docstring，每行逻辑代码都有中文注释。
运行方式：在 s20_deep_rl/ 目录下执行 python code/demo.py
依赖: pip install gymnasium numpy matplotlib torch
"""

import numpy as np
import matplotlib.pyplot as plt
# 中文字体配置
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
from collections import deque, namedtuple
from typing import List, Tuple, Deque, Optional
import time

# 尝试导入 gymnasium (>=0.26)，如果失败则尝试导入 gym
GYM_AVAILABLE = False
try:
    import gymnasium as gym                                     # 新版 Gym API
    GYM_NEW = True
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym                                              # 旧版 Gym API
        GYM_NEW = False
        GYM_AVAILABLE = True
    except ImportError:
        print("[警告] gymnasium 和 gym 均未安装，跳过 RL 环境演示")
        print("  安装: pip install gymnasium")

import torch                                                    # PyTorch 深度学习框架
import torch.nn as nn                                           # 神经网络模块
import torch.nn.functional as F                                 # 激活函数等
import torch.optim as optim                                     # 优化器

# GPU 自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("（未检测到 GPU，使用 CPU 运行。如有 GPU，请安装 CUDA 版 PyTorch 以获得加速）")

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================================
# 第一部分：工具函数
# ============================================================================

# 经验回放中的单条经验: (状态, 动作, 奖励, 下一状态, 是否终止)
Experience = namedtuple('Experience',                          # 命名元组，方便访问
                       ['state', 'action', 'reward',           # s_t, a_t, r_t
                        'next_state', 'done'])                 # s_{t+1}, done


def set_seed(seed: int = 42):
    """
    设置所有随机种子，保证实验可复现。

    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)                                        # NumPy 随机种子
    torch.manual_seed(seed)                                     # PyTorch CPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)                        # PyTorch GPU 随机种子


# ============================================================================
# 第二部分：经验回放缓冲区 (Experience Replay Buffer)
# ============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区 —— DQN 的关键组件之一。

    存储最近的 N 条经验，支持随机采样 mini-batch，
    从而打破序列相关性、提高数据效率。

    属性:
        buffer: 双端队列，存储 Experience 命名元组
        capacity: 缓冲区最大容量
    """

    def __init__(self, capacity: int = 10000):
        """
        初始化经验回放缓冲区。

        参数:
            capacity: 缓冲区最大容量（最多存储多少条经验）
        """
        self.buffer: Deque[Experience] = deque(maxlen=capacity)  # 双端队列，自动丢弃最旧经验

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        存储一条经验到缓冲区中。

        当缓冲区满时，最旧的经验会自动被丢弃（FIFO）。

        参数:
            state: 当前状态 s_t
            action: 执行的动作 a_t
            reward: 获得的奖励 r_t
            next_state: 下一状态 s_{t+1}
            done: 是否终止
        """
        exp = Experience(state, action, reward, next_state, done)  # 创建经验元组
        self.buffer.append(exp)                                    # 存入缓冲区（FIFO）

    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区中随机采样一个 mini-batch。

        参数:
            batch_size: 采样数量

        返回:
            states: 批量状态，shape (batch_size, state_dim)
            actions: 批量动作，shape (batch_size,)
            rewards: 批量奖励，shape (batch_size,)
            next_states: 批量下一状态，shape (batch_size, state_dim)
            dones: 批量终止标志，shape (batch_size,)
        """
        # 随机选取 batch_size 个索引（不放回抽样）
        indices = np.random.choice(len(self.buffer), batch_size,  # 从 [0, len(buffer)) 中采样
                                   replace=False)
        # 按索引提取对应经验
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            exp = self.buffer[i]                                   # 获取第 i 条经验
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)

        # 转为 numpy 数组并堆叠
        return (
            np.array(states, dtype=np.float32),                    # (batch, state_dim)
            np.array(actions, dtype=np.int64),                     # (batch,)
            np.array(rewards, dtype=np.float32),                   # (batch,)
            np.array(next_states, dtype=np.float32),               # (batch, state_dim)
            np.array(dones, dtype=np.float32),                     # (batch,)
        )

    def __len__(self) -> int:
        """返回缓冲区中当前存储的经验数量。"""
        return len(self.buffer)


# ============================================================================
# 第三部分：DQN 网络结构
# ============================================================================

class QNetwork(nn.Module):
    """
    深度 Q 网络 —— 用全连接神经网络近似 Q 函数 Q_θ(s, a)。

    输入: 状态向量 s (CartPole 为 4 维: 位置, 速度, 角度, 角速度)
    输出: 每个可能动作的 Q 值 [Q(s, a_1), ..., Q(s, a_n)]
          (CartPole 有 2 个动作: 左推/右推)

    架构: 输入层 → 128 (ReLU) → 128 (ReLU) → 输出层 (n_actions)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        """
        初始化 Q 网络。

        参数:
            state_dim: 状态维度（CartPole 为 4）
            n_actions: 动作数量（CartPole 为 2）
            hidden_dim: 隐藏层节点数
        """
        super(QNetwork, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(state_dim, hidden_dim)              # 输入 → 隐藏层 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)             # 隐藏层 1 → 隐藏层 2
        self.fc3 = nn.Linear(hidden_dim, n_actions)              # 隐藏层 2 → 输出 (Q 值)

        # ---- 初始化权重以改善训练稳定性 ----
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,                # Kaiming 初始化
                                       nonlinearity='relu')      # 配合 ReLU 使用
                nn.init.constant_(m.bias, 0)                     # 偏置初始化为 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 状态 → Q 值。

        参数:
            x: 批量状态，shape (batch, state_dim)

        返回:
            q_values: 批量 Q 值，shape (batch, n_actions)
        """
        x = F.relu(self.fc1(x))                                  # 第 1 层 + ReLU
        x = F.relu(self.fc2(x))                                  # 第 2 层 + ReLU
        q_values = self.fc3(x)                                   # 第 3 层 (线性输出 Q 值)
        return q_values


# ============================================================================
# 第四部分：DQN Agent
# ============================================================================

class DQNAgent:
    """
    DQN Agent —— 整合 Q 网络、目标网络、经验回放和 ε-贪婪探索。

    Agent 在每一步:
        1. 用 ε-贪婪策略选择动作
        2. 将经验存入回放缓冲区
        3. 从缓冲区采样 mini-batch 并训练在线网络
        4. 定期将在线网络权重复制到目标网络
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 0.001,                                       # 学习率
        gamma: float = 0.99,                                     # 折扣因子
        epsilon_init: float = 1.0,                               # 初始探索率
        epsilon_min: float = 0.01,                               # 最小探索率
        epsilon_decay: float = 0.995,                            # 探索率衰减
        buffer_capacity: int = 10000,                            # 回放缓冲区容量
        batch_size: int = 64,                                    # mini-batch 大小
        target_update_freq: int = 100,                           # 目标网络更新频率
        device: str = 'cpu',                                     # 设备 (cpu/cuda)
    ):
        """
        初始化 DQN Agent。

        参数:
            state_dim: 状态维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子 γ
            epsilon_init: 初始探索率 ε
            epsilon_min: 最小探索率
            epsilon_decay: ε 每步衰减因子
            buffer_capacity: 经验回放缓冲区大小
            batch_size: mini-batch 大小
            target_update_freq: 每隔多少步更新目标网络
            device: 运行设备
        """
        self.state_dim = state_dim                               # 状态维度
        self.n_actions = n_actions                               # 动作数量
        self.gamma = gamma                                       # 折扣因子 γ
        self.epsilon = epsilon_init                              # 当前探索率 ε
        self.epsilon_min = epsilon_min                           # 最小探索率
        self.epsilon_decay = epsilon_decay                       # 探索率衰减因子
        self.batch_size = batch_size                             # mini-batch 大小
        self.target_update_freq = target_update_freq             # 目标网络更新频率
        self.device = device                                     # 设备
        self.step_count = 0                                      # 全局步数计数器

        # ---- 网络 ----
        self.q_network = QNetwork(state_dim, n_actions).to(device)       # 在线网络 Q_θ
        self.target_network = QNetwork(state_dim, n_actions).to(device)  # 目标网络 Q_θ⁻
        self.target_network.load_state_dict(                            # 初始化目标网络 = 在线网络
            self.q_network.state_dict())
        self.target_network.eval()                                       # 目标网络仅为评估模式

        # ---- 优化器 ----
        self.optimizer = optim.Adam(self.q_network.parameters(),  # Adam 优化器
                                   lr=lr)

        # ---- 经验回放缓冲区 ----
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)  # 经验回放

        # ---- 损失函数 ----
        self.loss_fn = nn.MSELoss()                              # 均方误差损失

        # ---- 记录 ----
        self.loss_history = []                                   # 记录每次训练的损失

    def choose_action(self, state: np.ndarray) -> int:
        """
        ε-贪婪策略选择动作。

        以概率 ε 随机探索，以概率 1-ε 选择 Q 值最高的动作（利用）。

        参数:
            state: 当前状态，shape (state_dim,)

        返回:
            action: 选择的动作 (0 或 1)
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)           # 探索: 随机动作
        else:
            # 利用: 选择 Q 值最大的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
            with torch.no_grad():                                # 不计算梯度（纯推理）
                q_values = self.q_network(state_tensor)          # 前向传播得到 Q 值
            action = q_values.argmax(dim=1).item()               # argmax 选择最优动作
        return action

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        处理一个环境交互步：存储经验、训练网络、更新 ε。

        参数:
            state: 当前状态 s_t
            action: 执行的动作 a_t
            reward: 获得的奖励 r_t
            next_state: 下一状态 s_{t+1}
            done: 是否终止
        """
        # ---- 1. 存储经验到回放缓冲区 ----
        self.replay_buffer.push(state, action, reward, next_state, done)

        # ---- 2. 当缓冲区中有足够经验时，采样训练 ----
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()                                   # 执行一次梯度更新

        # ---- 3. 更新步数计数器 ----
        self.step_count += 1                                     # 全局步数 +1

    def _train_step(self):
        """
        从经验回放缓冲区采样一个 mini-batch，执行一步 DQN 训练。

        DQN 损失:
            L(θ) = E[(r + γ·max Q_θ⁻(s',a') - Q_θ(s,a))²]

        梯度只通过在线网络 Q_θ 传播，目标网络 Q_θ⁻ 被冻结。
        """
        # ---- 采样 mini-batch ----
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # ---- 转为 PyTorch Tensor ----
        states_t = torch.FloatTensor(states).to(self.device)          # (batch, state_dim)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch, 1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (batch, 1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)  # (batch, state_dim)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)   # (batch, 1)

        # ---- 计算当前 Q(s, a) ----
        current_q = self.q_network(states_t).gather(1, actions_t)  # 在线网络输出，取对应动作的 Q 值

        # ---- 计算 TD 目标 y = r + γ·max_{a'} Q_θ⁻(s', a') ----
        with torch.no_grad():                                    # 目标网络被视为常数
            next_q = self.target_network(next_states_t)           # 目标网络 Q_θ⁻(s', :)
            max_next_q = next_q.max(dim=1, keepdim=True)[0]      # max_{a'} Q_θ⁻(s', a')
            # TD 目标: 如果终止态 (done=1)，未来价值为 0
            td_target = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        # ---- 计算损失 L = MSE(td_target, current_q) ----
        loss = self.loss_fn(current_q, td_target)

        # ---- 反向传播更新在线网络 ----
        self.optimizer.zero_grad()                                # 清零梯度
        loss.backward()                                           # 反向传播
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()                                     # 更新参数

        # ---- 记录损失 ----
        self.loss_history.append(loss.item())

        # ---- 定期更新目标网络 θ⁻ ← θ ----
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict())                    # 直接复制在线网络权重

        # ---- 衰减探索率 ----
        self.epsilon = max(self.epsilon_min,                     # 不低于最小探索率
                          self.epsilon * self.epsilon_decay)     # 指数衰减


# ============================================================================
# 第五部分：Policy Network (REINFORCE)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    策略网络 —— REINFORCE 的核心，输出动作的概率分布 π_θ(a|s)。

    输入: 状态向量 s (4 维)
    输出: 每个动作的 log 概率 (用于数值稳定性) 和 softmax 概率

    架构: 输入层 → 128 (ReLU) → 128 (ReLU) → 输出层 (n_actions, log_softmax)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        """
        初始化策略网络。

        参数:
            state_dim: 状态维度
            n_actions: 动作数量
            hidden_dim: 隐藏层节点数
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)              # 输入 → 隐藏层 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)             # 隐藏层 1 → 隐藏层 2
        self.fc3 = nn.Linear(hidden_dim, n_actions)              # 隐藏层 2 → 输出 (logits)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输出动作的 softmax 概率。

        参数:
            x: 批量状态，shape (batch, state_dim)

        返回:
            probs: 动作概率分布 π_θ(a|s)，shape (batch, n_actions)
        """
        x = F.relu(self.fc1(x))                                  # 第 1 层 + ReLU
        x = F.relu(self.fc2(x))                                  # 第 2 层 + ReLU
        logits = self.fc3(x)                                     # 输出 logits
        probs = F.softmax(logits, dim=-1)                        # softmax → 概率分布
        return probs

    def get_action_and_log_prob(
        self,
        state: np.ndarray,
    ) -> Tuple[int, torch.Tensor]:
        """
        根据策略网络选择一个动作，同时返回该动作的 log 概率。
        log 概率将在 REINFORCE 更新中用于计算策略梯度。

        参数:
            state: 当前状态 numpy 数组，shape (state_dim,)

        返回:
            action: 采样的动作 (0 或 1)
            log_prob: 该动作在策略下的 log 概率，标量 Tensor
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)          # (1, state_dim)
        probs = self.forward(state_t)                            # (1, n_actions)
        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(probs)     # 类别分布
        action = action_dist.sample()                            # 采样一个动作
        log_prob = action_dist.log_prob(action)                  # 该动作的 log π(a|s)
        return action.item(), log_prob


class REINFORCEAgent:
    """
    REINFORCE Agent —— 基于策略梯度的强化学习方法。

    与 DQN 不同，REINFORCE:
    - 不维护 Q 值或价值函数
    - 在 episode 结束时用 Monte Carlo 回报一次性更新
    - 直接优化策略网络 π_θ
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 0.001,                                       # 学习率
        gamma: float = 0.99,                                     # 折扣因子 γ
        device: str = 'cpu',
    ):
        """
        初始化 REINFORCE Agent。

        参数:
            state_dim: 状态维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
            device: 运行设备
        """
        self.gamma = gamma                                       # 折扣因子
        self.device = device                                     # 设备
        self.n_actions = n_actions                               # 动作数量

        # ---- 策略网络 ----
        self.policy_network = PolicyNetwork(state_dim, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # ---- 存储一个 episode 的 (log_prob, reward) ----
        self.saved_log_probs = []                                # log π(a_t|s_t) 列表
        self.saved_rewards = []                                  # r_t 列表
        self.episode_reward = 0                                  # 当前 episode 累计奖励

    def choose_action(self, state: np.ndarray) -> int:
        """
        根据当前策略采样一个动作，并存储 log 概率。

        参数:
            state: 当前状态

        返回:
            action: 采样的动作
        """
        action, log_prob = self.policy_network.get_action_and_log_prob(state)
        self.saved_log_probs.append(log_prob)                    # 存储 log 概率
        return action

    def store_reward(self, reward: float):
        """
        存储即时奖励（每步调用）。

        参数:
            reward: 环境返回的奖励
        """
        self.saved_rewards.append(reward)                        # 存储奖励
        self.episode_reward += reward                            # 累计奖励

    def finish_episode(self):
        """
        Episode 结束时调用，执行 REINFORCE 策略梯度更新。

        REINFORCE 的梯度:
            ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]

        其中 G_t 是从时间步 t 开始的折扣累计回报（Return）。

        返回:
            loss: 该 episode 的策略梯度损失值
        """
        # ---- 计算回报 G_t (从后往前递推) ----
        returns = []                                             # 每个时间步的回报 G_t
        G = 0                                                    # 累计回报，初始为 0
        # 从最后一个时间步开始反向计算
        for r in reversed(self.saved_rewards):
            G = r + self.gamma * G                               # G_t = r_t + γ·G_{t+1}
            returns.insert(0, G)                                 # 插入到列表头部（保持时间顺序）

        # ---- 标准化回报 (减少方差的关键技巧) ----
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 零均值单位方差

        # ---- 计算策略梯度损失 ----
        # L(θ) = -Σ_t log π_θ(a_t|s_t) · G_t
        # 负号是因为 PyTorch 做梯度下降，而 REINFORCE 是梯度上升
        policy_loss = []
        for log_prob, G_t in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G_t)                  # 每个时间步的损失
        loss = torch.cat(policy_loss).sum()                      # 总损失 (标量)

        # ---- 反向传播更新策略网络 ----
        self.optimizer.zero_grad()                                # 清零梯度
        loss.backward()                                           # 反向传播
        self.optimizer.step()                                     # 更新参数

        # ---- 清空缓存，准备下一个 episode ----
        episode_reward = self.episode_reward
        self.saved_log_probs = []
        self.saved_rewards = []
        self.episode_reward = 0

        return loss.item(), episode_reward


# ============================================================================
# 第六部分：训练函数
# ============================================================================

def train_dqn(
    env: gym.Env,
    agent: DQNAgent,
    n_episodes: int = 500,
    render_every: int = 50,
    verbose: bool = True,
) -> List[float]:
    """
    使用 DQN 算法训练 Agent。

    参数:
        env: Gym 环境
        agent: DQN Agent
        n_episodes: 训练 episode 数
        render_every: 每隔多少 episode 渲染一次
        verbose: 是否打印进度

    返回:
        episode_rewards: 每个 episode 的总奖励列表
    """
    episode_rewards = []                                         # 记录每个 episode 的奖励
    recent_rewards = deque(maxlen=100)                           # 最近 100 个 episode 的奖励

    if verbose:
        print("\n" + "-" * 50)
        print("  [DQN] 训练开始...")
        print("-" * 50)

    start_time = time.time()

    for ep in range(n_episodes):
        # 重置环境
        if GYM_NEW:
            state, _ = env.reset()                               # gymnasium 返回 (obs, info)
        else:
            state = env.reset()                                   # 旧版 gym 返回 obs
        state = np.array(state, dtype=np.float32)
        total_reward = 0                                         # 累计奖励
        done = False

        while not done:
            # ---- 选择动作 ----
            action = agent.choose_action(state)                  # ε-贪婪选择

            # ---- 执行动作 ----
            if GYM_NEW:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated                   # 终止或截断
            else:
                next_state, reward, done, _ = env.step(action)

            # 如果 episode 被截断 (truncated)，但奖励为正，仍然视为好结果
            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            # ---- 存储经验并训练 ----
            agent.step(state, action, reward, next_state, done)

            state = next_state                                   # 状态转移

            # ---- 渲染 (可选) ----
            if ep % render_every == 0 and ep > 0:
                if hasattr(env, 'render'):
                    try:
                        env.render()
                    except Exception:
                        pass                                    # 忽略渲染错误

        # ---- Episode 结束 ----
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)

        if verbose and (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"  Episode {ep+1:4d}/{n_episodes}: "
                  f"reward={total_reward:6.1f}, "
                  f"avg100={avg_reward:6.1f}, "
                  f"ε={agent.epsilon:.3f}")

    training_time = time.time() - start_time

    if verbose:
        avg100 = np.mean(recent_rewards) if recent_rewards else 0
        print(f"\n  [DQN] 训练完成! 耗时: {training_time:.1f}秒")
        print(f"  [DQN] 最后 100 episode 平均奖励: {avg100:.1f}")

    return episode_rewards


def train_reinforce(
    env: gym.Env,
    agent: REINFORCEAgent,
    n_episodes: int = 500,
    verbose: bool = True,
) -> List[float]:
    """
    使用 REINFORCE 算法训练 Agent。

    参数:
        env: Gym 环境
        agent: REINFORCE Agent
        n_episodes: 训练 episode 数
        verbose: 是否打印进度

    返回:
        episode_rewards: 每个 episode 的总奖励列表
    """
    episode_rewards = []                                         # 记录奖励
    recent_rewards = deque(maxlen=100)                           # 最近 100 个 episode

    if verbose:
        print("\n" + "-" * 50)
        print("  [REINFORCE] 训练开始...")
        print("-" * 50)

    start_time = time.time()

    for ep in range(n_episodes):
        if GYM_NEW:
            state, _ = env.reset()
        else:
            state = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        done = False

        while not done:
            # ---- 选择动作 (REINFORCE: 随机采样，无 ε-贪婪) ----
            action = agent.choose_action(state)

            # ---- 执行动作 ----
            if GYM_NEW:
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            else:
                next_state, reward, done, _ = env.step(action)

            next_state = np.array(next_state, dtype=np.float32)
            total_reward += reward

            # ---- 存储奖励 (log 概率已在 choose_action 中存储) ----
            agent.store_reward(reward)

            state = next_state

        # ---- Episode 结束: 执行 REINFORCE 更新 ----
        loss, ep_reward = agent.finish_episode()
        episode_rewards.append(ep_reward)
        recent_rewards.append(total_reward)

        if verbose and (ep + 1) % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"  Episode {ep+1:4d}/{n_episodes}: "
                  f"reward={total_reward:6.1f}, "
                  f"avg100={avg_reward:6.1f}, "
                  f"loss={loss:.3f}")

    training_time = time.time() - start_time

    if verbose:
        avg100 = np.mean(recent_rewards) if recent_rewards else 0
        print(f"\n  [REINFORCE] 训练完成! 耗时: {training_time:.1f}秒")
        print(f"  [REINFORCE] 最后 100 episode 平均奖励: {avg100:.1f}")

    return episode_rewards


# ============================================================================
# 第七部分：可视化
# ============================================================================

def plot_training_comparison(
    dqn_rewards: List[float],
    reinforce_rewards: List[float],
    title: str = "DQN vs REINFORCE — CartPole-v1 Training Comparison",
):
    """
    绘制 DQN 和 REINFORCE 的训练奖励对比曲线。

    参数:
        dqn_rewards: DQN 的 episode 奖励列表
        reinforce_rewards: REINFORCE 的 episode 奖励列表
        title: 图表标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 子图 1: 原始奖励 ----
    ax = axes[0]
    dqn_eps = np.arange(len(dqn_rewards))
    rf_eps = np.arange(len(reinforce_rewards))

    ax.plot(dqn_eps, dqn_rewards, alpha=0.3, color='#2E86AB',    # DQN raw reward
            linewidth=0.5)
    ax.plot(rf_eps, reinforce_rewards, alpha=0.3, color='#F18F01',  # REINFORCE raw reward
            linewidth=0.5)

    # Moving average
    window = 50
    if len(dqn_rewards) >= window:
        dqn_smooth = np.convolve(dqn_rewards,
                                np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window-1, len(dqn_rewards)), dqn_smooth,
               'b-', linewidth=2, label=f'DQN (Moving Avg)')

    if len(reinforce_rewards) >= window:
        rf_smooth = np.convolve(reinforce_rewards,
                               np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window-1, len(reinforce_rewards)), rf_smooth,
               'orange', linewidth=2, label=f'REINFORCE (Moving Avg)')

    # Mark CartPole max score 500 (env auto-truncates)
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.5,
              label='Max Score (500)')

    ax.set_xlabel('Episode', fontsize=10)
    ax.set_ylabel('Total Reward', fontsize=10)
    ax.set_title('Training Reward (Raw + Moving Avg)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 2: Cumulative average ----
    ax2 = axes[1]
    dqn_cumavg = np.cumsum(dqn_rewards) / (np.arange(len(dqn_rewards)) + 1)
    rf_cumavg = np.cumsum(reinforce_rewards) / (np.arange(len(reinforce_rewards)) + 1)

    ax2.plot(dqn_eps, dqn_cumavg, 'b-', linewidth=2,
            label=f'DQN Cum. Avg')
    ax2.plot(rf_eps, rf_cumavg, 'orange', linewidth=2,
            label=f'REINFORCE Cum. Avg')

    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Cumulative Avg Reward', fontsize=10)
    ax2.set_title('Cumulative Average Reward', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'dqn_vs_reinforce.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] DQN vs REINFORCE 对比图已保存至 images/dqn_vs_reinforce.png")


def plot_dqn_loss(loss_history: List[float]):
    """
    绘制 DQN 训练过程中的损失曲线。

    参数:
        loss_history: 每次训练的损失值列表
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(loss_history, 'b-', linewidth=0.5, alpha=0.5)
    # 滑动平均
    if len(loss_history) > 100:
        smooth = np.convolve(loss_history,
                           np.ones(100) / 100, mode='valid')
        ax.plot(np.arange(99, len(loss_history)), smooth,
               'r-', linewidth=2, label='Moving Avg (window=100)')

    ax.set_xlabel('Training Steps', fontsize=10)
    ax.set_ylabel('Loss (MSE)', fontsize=10)
    ax.set_title('DQN Training Loss Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'dqn_loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] DQN 损失曲线已保存至 images/dqn_loss_curve.png")


def plot_reinforce_policy(
    policy_network: PolicyNetwork,
    title: str = "REINFORCE 策略网络决策边界",
):
    """
    可视化 REINFORCE 策略网络在 CartPole 状态空间中的决策。

    固定 Cart 位置和速度，在 CartPole 的 (角度, 角速度) 平面上绘制
    每个动作的概率，展示策略的决策边界。

    参数:
        policy_network: 训练好的策略网络
        title: 图表标题
    """
    policy_network.eval()                                       # 评估模式

    # 在状态空间的 (角度, 角速度) 平面上采样
    n_points = 50
    # CartPole 典型范围: angle ∈ [-0.2, 0.2], angular_velocity ∈ [-1.5, 1.5]
    angles = np.linspace(-0.2, 0.2, n_points)                    # 角度范围
    ang_vels = np.linspace(-1.5, 1.5, n_points)                  # 角速度范围
    AA, VV = np.meshgrid(angles, ang_vels)                       # 网格

    probs = np.zeros((n_points, n_points))                       # 动作 1 的概率
    for i in range(n_points):
        for j in range(n_points):
            # 构造状态向量 (cart_pos=0, cart_vel=0, angle, ang_vel)
            state = np.array([0.0, 0.0, AA[i, j], VV[i, j]], dtype=np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                p = policy_network(state_t)                      # (1, n_actions)
            probs[i, j] = p[0, 1].item()                         # 动作 1 (右推) 的概率

    # 绘制热力图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.contourf(AA, VV, probs, levels=20, cmap='RdYlBu',   # red=action1, blue=action0
                    alpha=0.8)
    # Add 0.5 probability contour (decision boundary)
    ax.contour(AA, VV, probs, levels=[0.5], colors='k',         # black line = decision boundary
              linewidths=2, linestyles='--')
    plt.colorbar(im, ax=ax, label='P(action=1 | state)')

    ax.set_xlabel('Pole Angle (rad)', fontsize=10)
    ax.set_ylabel('Pole Angular Velocity (rad/s)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(_IMAGES, 'reinforce_policy_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[可视化] REINFORCE 策略热力图已保存至 images/reinforce_policy_heatmap.png")


# ============================================================================
# 第八部分：主程序
# ============================================================================

def main():
    """
    主程序：训练 DQN 和 REINFORCE 在 CartPole-v1 上，可视化并对比结果。
    """
    print("\n" + "=" * 70)
    print("    s20 深度强化学习: DQN 与 Policy Gradient — 完整演示")
    print("=" * 70)

    if not GYM_AVAILABLE:
        print("[跳过] Gym 环境不可用，无法运行 RL 训练演示。")
        print("安装: pip install gymnasium")
        return

    set_seed(42)                                                 # 固定随机种子

    # ---- 创建环境 ----
    print("\n[环境] 创建 CartPole-v1...")
    if GYM_NEW:
        env = gym.make('CartPole-v1')
    else:
        env = gym.make('CartPole-v1')

    state_dim = env.observation_space.shape[0]                   # 4: [位置, 速度, 角度, 角速度]
    n_actions = env.action_space.n                               # 2: 左推 / 右推
    print(f"  状态维度: {state_dim}, 动作数量: {n_actions}")

    # ---- 训练参数 ----
    N_EPISODES = 500                                             # 训练 episode 数
    device = DEVICE      # 使用全局设备配置
    print(f"  设备: {device}")

    # ========================================================================
    # 实验 1: DQN 训练
    # ========================================================================
    print("\n" + "=" * 50)
    print("【实验 1】DQN 训练")
    print("=" * 50)

    dqn_agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=0.001,
        gamma=0.99,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        device=device,
    )

    dqn_rewards = train_dqn(
        env=env,
        agent=dqn_agent,
        n_episodes=N_EPISODES,
        render_every=100,
        verbose=True,
    )

    # ========================================================================
    # 实验 2: REINFORCE 训练
    # ========================================================================
    print("\n" + "=" * 50)
    print("【实验 2】REINFORCE 训练")
    print("=" * 50)

    # 重新创建环境（新的随机种子）
    if GYM_NEW:
        env.close()
        env = gym.make('CartPole-v1')
    else:
        env.close()
        env = gym.make('CartPole-v1')

    reinforce_agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr=0.001,
        gamma=0.99,
        device=device,
    )

    reinforce_rewards = train_reinforce(
        env=env,
        agent=reinforce_agent,
        n_episodes=N_EPISODES,
        verbose=True,
    )

    env.close()                                                  # 关闭环境

    # ========================================================================
    # 实验 3: 可视化与对比
    # ========================================================================
    print("\n" + "=" * 50)
    print("【实验 3】可视化与对比")
    print("=" * 50)

    # ---- 3.1 训练奖励对比 ----
    plot_training_comparison(dqn_rewards, reinforce_rewards)

    # ---- 3.2 DQN 损失曲线 ----
    plot_dqn_loss(dqn_agent.loss_history)

    # ---- 3.3 REINFORCE 策略可视化 ----
    plot_reinforce_policy(reinforce_agent.policy_network)

    # ========================================================================
    # 最终总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("【总结】")
    print("=" * 70)

    dqn_avg100 = np.mean(dqn_rewards[-100:]) \
        if len(dqn_rewards) >= 100 else np.mean(dqn_rewards)
    rf_avg100 = np.mean(reinforce_rewards[-100:]) \
        if len(reinforce_rewards) >= 100 else np.mean(reinforce_rewards)

    print(f"\n  CartPole-v1 最高分: 500 (达到后 episode 自动截断)")
    print(f"  DQN        — 最后 100 ep 平均: {dqn_avg100:.1f}")
    print(f"  REINFORCE  — 最后 100 ep 平均: {rf_avg100:.1f}")

    dqn_reached = sum(1 for r in dqn_rewards[-100:] if r >= 475)
    rf_reached = sum(1 for r in reinforce_rewards[-100:] if r >= 475)
    print(f"  DQN        — 近 100 ep 中达到 475+: {dqn_reached}/100")
    print(f"  REINFORCE  — 近 100 ep 中达到 475+: {rf_reached}/100")

    print(f"\n  【DQN 核心机制】")
    print(f"  - 用神经网络 Q_θ 近似 Q 函数，处理连续状态")
    print(f"  - 经验回放: 随机采样打破相关性，提高数据效率")
    print(f"  - 目标网络: 冻结 TD 目标，稳定训练")
    print(f"  - Off-policy: 可以用旧策略产生的数据训练")
    print(f"\n  【REINFORCE 核心机制】")
    print(f"  - 直接学习策略 π_θ，输出动作概率分布")
    print(f"  - Monte Carlo 回报: 用完整 episode 的累计奖励")
    print(f"  - 回报标准化: 减去均值除以标准差，降方差")
    print(f"  - On-policy: 只能用当前策略产生的数据")
    print(f"\n  【对比总结】")
    print(f"  - DQN 样本效率更高（经验回放重复利用），但只适合离散动作")
    print(f"  - REINFORCE 方差较高但能处理连续动作，策略直接可解释")
    print(f"  - Actor-Critic (A2C/A3C) 结合两者优点，是实际应用的主力")
    print(f"\n  所有图片已保存至 images/ 目录")
    print("=" * 70)

    print("\n  运行完成！\n")


if __name__ == "__main__":
    main()
