# -*- coding: utf-8 -*-
"""
s20 深度强化学习：DQN 与 Policy Gradient — 练习代码
=====================================================
请完成以下 TODO 任务，巩固对 DQN 和 REINFORCE 核心机制的理解。

每个 TODO 都有详细的指示和预期输出描述。
建议先阅读 README.md 和 demo.py，再尝试独立补全代码。
运行方式：在 s20_deep_rl/ 目录下执行 python code/exercise.py
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# 辅助数据结构
# ============================================================================

# 经验元组: (状态, 动作, 奖励, 下一状态, 是否终止)
Experience = Tuple[np.ndarray, int, float, np.ndarray, bool]


# ============================================================================
# 辅助网络: 简单 Q 网络
# ============================================================================

class SimpleQNetwork(nn.Module):
    """
    简单的 Q 网络 —— 用于练习中的 DQN 部分。
    输入 4 维状态 → 输出 2 维 Q 值 [Q(s,0), Q(s,1)]。
    """
    def __init__(self, state_dim: int = 4, n_actions: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)              # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)             # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions)              # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# TODO 1: 实现经验回放缓冲区
# ============================================================================
class ReplayBuffer:
    """
    TODO 1: 实现经验回放缓冲区。

    需要完成两个方法:
        - push(): 存储一条经验到缓冲区
        - sample(): 随机采样一个 mini-batch

    缓冲区是一个定长的双端队列，当存满时自动丢弃最旧的经验（FIFO）。
    """

    def __init__(self, capacity: int = 10000):
        """
        初始化缓冲区。

        参数:
            capacity: 缓冲区最大容量
        """
        # TODO: 初始化双端队列，设置 maxlen=capacity
        self.buffer = None  # ← TODO: deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        存储一条经验 (s, a, r, s', done) 到缓冲区。

        参数:
            state: 当前状态，shape (state_dim,)
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # TODO: 创建经验元组并加入缓冲区
        # 提示: 可以直接用 tuple: (state, action, reward, next_state, done)
        # 提示: self.buffer.append(experience)
        pass  # ← TODO: 实现存储逻辑

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        从缓冲区随机采样 batch_size 条经验。

        参数:
            batch_size: 需要采样的经验数量（必须 ≤ 缓冲区当前大小）

        返回:
            states: shape (batch_size, state_dim)
            actions: shape (batch_size,)
            rewards: shape (batch_size,)
            next_states: shape (batch_size, state_dim)
            dones: shape (batch_size,)
        """
        # TODO: 实现随机采样
        # 提示 1: 使用 np.random.choice(len(self.buffer), batch_size, replace=False)
        #         来选取 batch_size 个随机索引
        # 提示 2: 遍历选中的索引，提取对应经验
        # 提示 3: 返回各自的 numpy 数组

        # (下面每个变量需要从缓冲区中提取对应字段)
        states = None        # ← TODO: 提取所有状态的 numpy 数组
        actions = None       # ← TODO: 提取所有动作的 numpy 数组
        rewards = None       # ← TODO: 提取所有奖励的 numpy 数组
        next_states = None   # ← TODO: 提取所有下一状态的 numpy 数组
        dones = None         # ← TODO: 提取所有终止标志的 numpy 数组

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回当前缓冲区的经验数量。"""
        if self.buffer is None:
            return 0
        return len(self.buffer)


# ---- 测试 TODO 1 ----
def test_replay_buffer():
    """测试经验回放缓冲区的存储和采样功能。"""
    print("=" * 60)
    print("TODO 1 测试: 经验回放缓冲区")
    print("=" * 60)

    buf = ReplayBuffer(capacity=100)

    if buf.buffer is None:
        print("  TODO 未完成: buffer 未初始化，请补全 __init__")
        return

    # 存入 10 条测试经验
    for i in range(10):
        state = np.array([float(i * 4 + j) for j in range(4)])  # 4 维状态
        buf.push(state, i % 2, float(i), state + 1.0, i == 9)   # 每步奖励 = i

    if len(buf) != 10:
        print(f"  测试 1 [存储]: 期望 buffer 大小=10, 实际={len(buf)}")
        print("  TODO 未完成: push 方法可能未正确实现")
        return
    else:
        print(f"  ✓ 测试 1 [存储]: buffer 大小=10, 正确!")

    # 随机采样 4 条经验
    try:
        result = buf.sample(4)
        if all(v is not None for v in result):
            states, actions, rewards, next_states, dones = result
            print(f"  ✓ 测试 2 [采样]:")
            print(f"    states.shape     = {states.shape}      (期望: (4, 4))")
            print(f"    actions.shape    = {actions.shape}       (期望: (4,))")
            print(f"    rewards.shape    = {rewards.shape}       (期望: (4,))")
            print(f"    next_states.shape= {next_states.shape}      (期望: (4, 4))")
            print(f"    dones.shape      = {dones.shape}         (期望: (4,))")
            print(f"    actions 范围: {actions.min()}-{actions.max()} (期望: 0-1)")
        else:
            print("  TODO 未完成: sample 返回了 None 值")
    except Exception as e:
        print(f"  测试 2 [采样] 出错: {e}")

    print()


# ============================================================================
# TODO 2: 实现 DQN 损失计算（含目标网络）
# ============================================================================
def compute_dqn_loss(
    q_network: nn.Module,
    target_network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    TODO 2: 实现 DQN 的损失计算。

    DQN 损失公式:
        L(θ) = E[(r + γ·max_{a'} Q_θ⁻(s', a') - Q_θ(s, a))²]

    其中:
        - Q_θ(s, a): 在线网络的输出 (当前估计)
        - Q_θ⁻(s', a'): 目标网络的输出 (稳定目标)
        - 如果 done=True, TD 目标 = r (没有未来价值)
        - 如果 done=False, TD 目标 = r + γ·max_{a'} Q_θ⁻(s', a')

    参数:
        q_network: 在线网络 Q_θ
        target_network: 目标网络 Q_θ⁻
        states: 批量状态，(batch, state_dim)
        actions: 批量动作，(batch,) 或 (batch, 1)
        rewards: 批量奖励，(batch,) 或 (batch, 1)
        next_states: 批量下一状态，(batch, state_dim)
        dones: 批量终止标志，(batch,) 或 (batch, 1)
        gamma: 折扣因子 γ

    返回:
        loss: 标量损失值 (MSE)
    """
    # TODO: 补全 DQN 损失计算

    # 步骤 1: 确保维度正确
    if actions.dim() == 1:
        actions = actions.unsqueeze(1)                            # (batch,) → (batch, 1)
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)                            # (batch,) → (batch, 1)
    if dones.dim() == 1:
        dones = dones.unsqueeze(1)                                # (batch,) → (batch, 1)

    # 步骤 2: 计算当前 Q 值 Q_θ(s, a)
    # 提示: 使用 q_network(states) 获取所有动作的 Q 值，然后用 gather 取对应动作
    all_q_values = None  # ← TODO: q_network(states)  # (batch, n_actions)
    current_q = None     # ← TODO: all_q_values.gather(1, actions)  # (batch, 1)

    # 步骤 3: 计算 TD 目标
    # 提示: 使用 target_network(next_states) 获取下一状态的 Q 值
    #       取 max: .max(dim=1, keepdim=True)[0]
    #       TD 目标 = rewards + gamma * max_next_q * (1 - dones)
    #       注意: 需要在 torch.no_grad() 下计算，目标网络不计算梯度
    with torch.no_grad():                                        # 目标网络无梯度
        next_q = None  # ← TODO: target_network(next_states)  # (batch, n_actions)
        max_next_q = None  # ← TODO: next_q.max(dim=1, keepdim=True)[0]  # (batch, 1)
        td_target = None  # ← TODO: rewards + gamma * max_next_q * (1 - dones)  # (batch, 1)

    # 步骤 4: 计算 MSE 损失
    loss = None  # ← TODO: F.mse_loss(current_q, td_target)

    return loss


# ---- 测试 TODO 2 ----
def test_dqn_loss():
    """测试 DQN 损失计算。"""
    print("=" * 60)
    print("TODO 2 测试: DQN 损失计算 (含目标网络)")
    print("=" * 60)

    # 创建两个相同的网络
    q_net = SimpleQNetwork(state_dim=4, n_actions=2, hidden_dim=32)
    target_net = SimpleQNetwork(state_dim=4, n_actions=2, hidden_dim=32)
    target_net.load_state_dict(q_net.state_dict())               # 初始时相同

    # 创建测试数据: batch_size=3
    states = torch.randn(3, 4)                                   # 3 个样本，4 维状态
    actions = torch.tensor([0, 1, 0])                            # 三个不同的动作
    rewards = torch.tensor([1.0, -1.0, 0.5])
    next_states = torch.randn(3, 4)
    dones = torch.tensor([0.0, 0.0, 1.0])                        # 第 3 个是终止态

    loss = compute_dqn_loss(q_net, target_net, states, actions,
                           rewards, next_states, dones, gamma=0.99)

    if loss is None:
        print("  TODO 未完成，请补全 compute_dqn_loss 函数")
    else:
        print(f"  ✓ 损失计算成功!")
        print(f"    损失值 = {loss.item():.6f}")
        print(f"    由于 Q_net = Target_net 且 Q 值初始接近零,")
        print(f"    损失应近似为 r² 的均值: (1²+(-1)²+0.5²)/3 ≈ {((1+1+0.25)/3):.4f}")
        expected_approx = (1 + 1 + 0.25) / 3
        if abs(loss.item() - expected_approx) < 5.0:
            print(f"    ✓ 损失值在合理范围内!")

        # 检查梯度流
        loss.backward()
        has_grad = any(p.grad is not None for p in q_net.parameters())
        print(f"    在线网络是否接收到梯度: {'✓ 是' if has_grad else '✗ 否'}")
        has_target_grad = any(p.grad is not None for p in target_net.parameters())
        print(f"    目标网络是否没有被更新: {'✓ 是 (正确)' if not has_target_grad else '✗ 否 (目标网络不应有梯度)'}")

    print()


# ============================================================================
# TODO 3: 实现 REINFORCE 策略梯度更新
# ============================================================================
def reinforce_update(
    policy_network: nn.Module,
    optimizer: optim.Optimizer,
    log_probs: List[torch.Tensor],
    rewards: List[float],
    gamma: float = 0.99,
) -> float:
    """
    TODO 3: 实现 REINFORCE 的策略梯度更新。

    REINFORCE 更新公式:
        θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · G_t

    其中 G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... 是从步 t 起的折扣累计回报。

    实现步骤:
        1. 从最后一个时间步往前，计算每个步的折扣回报 G_t
        2. 对回报做标准化（减均值/除标准差），以降低梯度方差
        3. 计算策略损失: L = -Σ_t log π_θ(a_t|s_t) · G_t
        4. 反向传播并更新参数

    参数:
        policy_network: 策略网络 π_θ
        optimizer: 优化器
        log_probs: 每个步的 log π_θ(a_t|s_t)，Tensor 列表
        rewards: 每个步的奖励 r_t，float 列表
        gamma: 折扣因子 γ

    返回:
        loss_val: 策略梯度的损失值 (float)
    """
    # TODO: 补全 REINFORCE 更新

    # 步骤 1: 计算折扣回报 G_t (从后往前遍历)
    # 提示:
    #   G = 0
    #   returns = []
    #   for r in reversed(rewards):
    #       G = r + gamma * G
    #       returns.insert(0, G)
    returns = []  # ← TODO: 计算每个步的折扣累计回报

    # 步骤 2: 将回报转为 Tensor 并标准化
    # 提示: 如果 len(returns) > 1, 则 (returns - returns.mean()) / (returns.std() + 1e-8)
    returns_t = None  # ← TODO: torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns_t = None  # ← TODO: 标准化

    # 步骤 3: 计算策略梯度损失 L = -Σ log_prob · G_t
    # 提示: 遍历 zip(log_probs, returns_t)，累加 -log_prob * G_t
    #       负号是因为 PyTorch 做梯度下降，而 REINFORCE 是梯度上升
    policy_loss = None  # ← TODO: 初始化为 0
    for log_prob, G_t in zip(log_probs, returns_t):
        policy_loss = None  # ← TODO: 累加 -log_prob * G_t

    # 步骤 4: 反向传播并更新
    optimizer.zero_grad()                                        # 清零梯度
    policy_loss.backward()                                       # 反向传播
    optimizer.step()                                             # 更新参数

    return policy_loss.item()                                    # 返回损失值


# ---- 测试 TODO 3 ----
def test_reinforce_update():
    """测试 REINFORCE 策略梯度更新。"""
    print("=" * 60)
    print("TODO 3 测试: REINFORCE 策略梯度更新")
    print("=" * 60)

    # 创建策略网络
    policy_net = SimpleQNetwork(state_dim=4, n_actions=2, hidden_dim=32)
    # 保存初始权重用于对比
    initial_params = [p.clone().detach() for p in policy_net.parameters()]
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    # 模拟一个 episode: 5 步
    # 前 4 步都是小奖励（保持平衡），最后一步大奖励
    rewards = [-0.1, -0.1, -0.1, -0.1, 10.0]                   # 总共 5 步

    # 模拟每一步的 log 概率（假设选择了动作 0 或 1）
    log_probs = []
    for i in range(5):
        state_t = torch.randn(1, 4)                              # 随机状态
        probs = policy_net(state_t)                              # (1, 2) Q 值
        # 用 softmax 把 Q 值转为概率（虽然这不是严格正确的策略网络，但用于测试足够）
        action_probs = F.softmax(probs, dim=1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)

    loss_val = reinforce_update(policy_net, optimizer, log_probs, rewards, gamma=0.99)

    if loss_val is None:
        print("  TODO 未完成，请补全 reinforce_update 函数")
    else:
        print(f"  ✓ REINFORCE 更新成功! 损失值 = {loss_val:.4f}")

        # 检查权重是否变化（梯度是否传播）
        changed = False
        for p_init, p_now in zip(initial_params, policy_net.parameters()):
            if not torch.allclose(p_init, p_now):
                changed = True
                break
        if changed:
            print(f"  ✓ 策略网络权重已更新!")
        else:
            print(f"  ✗ 策略网络权重未变化 (梯度可能未正确传播)")

        print(f"\n  逻辑验证:")
        print(f"    由于最后一步奖励 10.0 远大于前几步的 -0.1,")
        print(f"    最后一步的 G_t 应该最大 (≈10.0)")
        print(f"    策略梯度应该主要增加最后一步所选动作的概率")
        print(f"    同时，前几步的 G_t 较小 (≈0.1-0.2)，影响较小")

    print()


# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   s20 深度强化学习: DQN 与 Policy Gradient — 动手练习       ║")
    print("║   请依次完成 TODO 1, 2, 3                                    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    test_replay_buffer()
    test_dqn_loss()
    test_reinforce_update()

    print("=" * 60)
    print("所有测试完成！请检查输出结果。")
    print("如有未通过的测试，请回到对应的 TODO 部分补全代码。")
    print("=" * 60)
    print()
    print("提示: 完成 TODO 后，运行 demo.py 查看 CartPole 上的完整训练效果。")
    print("  python code/demo.py")
    print()
