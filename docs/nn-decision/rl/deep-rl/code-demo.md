---
title: "s20 深度强化学习：DQN 与 Policy Gradient — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s20 深度强化学习：DQN 与 Policy Gradient — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/deep-rl/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/rl/deep-rl/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库做什么

```python
import numpy as np                          # 数值计算（环境交互中状态的 numpy 数组操作）
import matplotlib.pyplot as plt             # 绘制训练奖励曲线、损失曲线、策略热力图
from collections import deque, namedtuple   # deque: 经验回放缓冲区（FIFO）; namedtuple: 结构化经验元组
import gymnasium as gym                     # OpenAI Gymnasium 环境（CartPole-v1）
import torch                                # 深度学习框架
import torch.nn as nn                       # nn.Linear, nn.MSELoss 等神经网络组件
import torch.nn.functional as F             # F.relu, F.softmax 等激活函数
import torch.optim as optim                 # Adam 优化器
```

**关键说明**：
- `gymnasium`（>=0.26）是新版 Gym API，与旧版 `gym` 不同：`env.reset()` 返回 `(obs, info)` 元组而非单独的 `obs`
- `namedtuple('Experience', ['state','action','reward','next_state','done'])` 将经验定义为结构化对象，比普通 tuple 更具可读性
- `deque(maxlen=capacity)` 的 `maxlen` 参数自动实现 FIFO 淘汰——存入新经验时，最旧的经验被自动丢弃

### 第2步：环境/数据准备 — 为什么选 CartPole-v1？

CartPole-v1 是强化学习的 "Hello World"：
- **状态空间**：4 维连续向量 —— `[cart位置, cart速度, 杆角度, 杆角速度]`，值域均为连续实数
- **动作空间**：2 个离散动作 —— 左推(0) 或 右推(1)
- **奖励函数**：每存活一步 +1，最高 500 步（episode 在第 500 步自动截断）
- **为什么用它**：它足够复杂以展示 DQN 的价值（连续状态不能用 Q-Table），又足够简单让算法在几分钟内收敛

```python
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n               # 2
```

### 第3步：经验回放缓冲区（Experience Replay Buffer）— DQN 核心组件一

**数学背景**：监督学习中样本独立同分布（i.i.d.），但强化学习的经验是**序列相关**的——连续的状态转移高度关联。如果直接按顺序训练神经网络：

$$
\mathcal{L}(\theta) = \frac{1}{|B|}\sum_{(s,a,s') \in B}\left(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2
$$

其中 $B$ 如果只包含连续的经验，梯度方向会被最近的经验主导，导致训练不稳定甚至发散。

**经验回放的解决方案**：维护一个容量为 $N$（默认 10000）的 FIFO 缓冲区 $\mathcal{D} = \{(s, a, r, s')\}$，每次更新时从中**均匀随机采样** mini-batch。这带来三个好处：
1. **打破相关性** —— 随机采样使 batch 中的经验来自不同时刻，近似 i.i.d.
2. **提高数据效率** —— 每条经验可被多次学习，而非用过即弃
3. **平滑学习** —— 避免策略被最近的几次交互主导

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)     # maxlen 实现自动 FIFO 淘汰

    def push(self, state, action, reward, next_state, done):
        # 使用 namedtuple 结构化存储
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # 逐字段提取并堆叠为 numpy 数组
        states = np.array([self.buffer[i].state for i in indices], dtype=np.float32)
        actions = np.array([self.buffer[i].action for i in indices], dtype=np.int64)
        rewards = np.array([self.buffer[i].reward for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer[i].next_state for i in indices], dtype=np.float32)
        dones = np.array([self.buffer[i].done for i in indices], dtype=np.float32)
        return states, actions, rewards, next_states, dones
```

**关键细节**：`deque(maxlen=capacity)` 的 FIFO 淘汰意味着缓冲区始终保留**最近**的经验。当策略逐步改进时，旧经验（早期随机探索产生的）会被自然淘汰，缓冲区始终包含相对新鲜的交互数据。

### 第4步：Q-Network 设计 — 从 Q-Table 到函数逼近

**核心思想**：Q-Table 用表格存储 $Q(s, a)$，每个状态-动作对需要单独学习。DQN 用一个神经网络 $Q_\theta(s)$ 来**同时输出所有动作的 Q 值**：

$$
Q_\theta(s) = [Q_\theta(s, a_1), Q_\theta(s, a_2), \ldots, Q_\theta(s, a_n)]
$$

这样一次前向传播就能得到所有候选动作的 Q 值，时间复杂度从 $O(n)$ 降到 $O(1)$（n=动作数量）。

**为什么选这个架构**（两层隐藏层，每层 128 个神经元）：
- CartPole 状态只有 4 维，不需要 CNN 这类复杂结构
- 两个隐藏层提供足够的非线性表达能力来近似 Q 函数
- ReLU 激活函数避免梯度消失，Kaiming 初始化确保前向传播时信号不衰减

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        self.fc1 = nn.Linear(state_dim, hidden_dim)    # 4 → 128
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 128 → 128
        self.fc3 = nn.Linear(hidden_dim, n_actions)    # 128 → 2 (Q值)
        # Kaiming 初始化：保持 ReLU 前向传播的方差
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)     # 最后一层无激活函数（Q值可以是任意实数）
        return q_values
```

**为什么最后一层不加激活函数**：Q 值本质上是期望累计奖励，可以是任意实数值（正或负）。如果用了 ReLU，Q 值被限制在 $[0,+\infty)$，意味着"坏动作"的 Q 值无法表示为负数，限制了表达能力。

### 第5步：DQN Agent — 整合在线网络、目标网络与探索策略

#### 5.1 目标网络（Target Network）— DQN 核心组件二

**问题**：Q-Learning 的 TD 目标 $y = r + \gamma \max_{a'} Q(s', a')$ 中，如果用同一个网络计算 $\max Q(s',a')$ 和 $Q(s,a)$，更新会同时改变预测和目标——类似追着自己尾巴跑的狗。

**解决方案**：维护两个结构完全相同但参数不同的网络：
- **在线网络 $Q_\theta$**：每步更新，用于选择动作和计算当前 Q 值
- **目标网络 $Q_{\theta^-}$**：每 $C$ 步（默认 100）才从在线网络复制权重，用于计算 TD 目标中的 $\max_{a'} Q(s', a')$

```python
# 初始化：目标网络 = 在线网络
self.target_network.load_state_dict(self.q_network.state_dict())
self.target_network.eval()   # 目标网络始终处于评估模式（无 dropout/batch norm）

# 每 C 步更新一次
if self.step_count % self.target_update_freq == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**为什么目标网络使训练稳定**：TD 目标 $y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$ 中的 $Q_{\theta^-}$ 被冻结 $C$ 步不变，相当于在 $C$ 步内优化一个固定的回归目标。这避免了目标随着预测一起振荡的问题。

#### 5.2 ε-贪婪探索策略

RL 的核心困境：**探索 (exploration) vs 利用 (exploitation)**。
- 纯利用（$\epsilon=0$）：永远选当前认为最好的动作，但可能错过了更好的策略
- 纯探索（$\epsilon=1$）：完全随机采样，学不到任何东西

ε-贪婪策略在两者间折中，且探索率**随时间衰减**：

```python
# 探索率指数衰减
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
# epsilon_decay=0.995: 经过 1000 步后 ε ≈ 1.0*0.995^1000 ≈ 0.007

def choose_action(self, state):
    if np.random.random() < self.epsilon:
        return np.random.randint(self.n_actions)        # 探索：随机动作
    else:
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()            # 利用：选 Q 值最大
```

**为什么用衰减**：训练初期策略是随机的，需要大量探索来收集经验（$\epsilon \approx 1.0$）。随着策略变好，逐渐减少探索，更多地依赖学到的策略（$\epsilon \to 0.01$）。

#### 5.3 DQN 损失函数与训练步骤

**数学公式**：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \cdot \max_{a'} Q_{\theta^{-}}(s', a') \cdot (1-d) - Q_{\theta}(s, a) \right)^2 \right]
$$

其中 $d \in \{0, 1\}$ 是终止标志（done）。当 $d=1$ 时，TD 目标退化为 $r$（因为终止态没有后续状态，未来 Q 值为 0）。

```python
def _train_step(self):
    # 1. 从回放缓冲区采样 mini-batch
    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

    # 2. 计算当前 Q 值：只取实际执行动作对应的 Q 值
    current_q = self.q_network(states_t).gather(1, actions_t)
    # gather(dim=1, index=actions_t) 的作用：
    #   self.q_network(states_t) 输出 (batch, n_actions)
    #   gather 从中取出每行第 actions_t[i] 列的 Q 值 → (batch, 1)

    # 3. 计算 TD 目标 y = r + γ·max Q_θ⁻(s',a')
    with torch.no_grad():   # 目标网络在计算图中被视为常数
        next_q = self.target_network(next_states_t)
        max_next_q = next_q.max(dim=1, keepdim=True)[0]
        td_target = rewards_t + self.gamma * max_next_q * (1 - dones_t)
        # (1 - dones_t): 如果终止, 未来价值 = 0

    # 4. 均方误差损失
    loss = self.loss_fn(current_q, td_target)   # nn.MSELoss()

    # 5. 反向传播（仅更新在线网络）
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 防止梯度爆炸
    self.optimizer.step()
```

**为什么用 MSE 而非交叉熵**：DQN 是一个回归问题——我们想让 $Q_\theta(s, a)$ 尽可能接近 TD 目标数值 $y$，而非分类。MSE 是回归问题的自然选择。

**梯度裁剪的作用**：$\max Q_{\theta^-}(s',a')$ 可能导致较大的梯度，裁剪将其限制在 1.0 以内，防止一次更新将网络参数推到极端值。

### 第6步：REINFORCE — 策略梯度方法

#### 6.1 Policy Network 设计

与 DQN 的 Q-Network 关键区别：
- DQN 输出 **Q 值**（实数，越大越好）
- REINFORCE 输出 **动作概率分布** $\pi_\theta(a|s)$（和为 1）

```python
class PolicyNetwork(nn.Module):
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits, dim=-1)   # softmax → 概率分布
        return probs
```

**为什么最后一层用 softmax**：策略 $\pi_\theta(a|s)$ 必须是一个合法的概率分布（所有动作概率非负，且和为 1）。Softmax 将任意实数 logits 转换为概率。

#### 6.2 动作采样与 log 概率

```python
def get_action_and_log_prob(self, state):
    probs = self.forward(state_t)                              # π_θ(a|s)
    action_dist = torch.distributions.Categorical(probs)       # 类别分布
    action = action_dist.sample()                               # 按概率采样
    log_prob = action_dist.log_prob(action)                     # log π_θ(a|s)
    return action.item(), log_prob
```

**为什么要记录 $\log \pi_\theta(a_t|s_t)$**：策略梯度定理的核心公式是：

$$
\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 t 开始的折扣累计回报。$\log \pi_\theta$ 告诉我们应该朝哪个方向调整参数，$G_t$ 决定调整的幅度和方向。

#### 6.3 REINFORCE 更新：Monte Carlo 回报

与 DQN 每步更新不同，REINFORCE 在 episode **结束后**一次性更新：

```python
def finish_episode(self):
    # 1. 从后往前计算折扣累计回报 G_t
    returns = []
    G = 0
    for r in reversed(self.saved_rewards):
        G = r + self.gamma * G           # G_t = r_t + γ·G_{t+1}
        returns.insert(0, G)

    # 2. 标准化回报（降方差的关键技巧）
    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 3. 策略梯度损失 L = -Σ log π · G
    policy_loss = []
    for log_prob, G_t in zip(self.saved_log_probs, returns):
        policy_loss.append(-log_prob * G_t)    # 负号：PyTorch 做梯度下降，REINFORCE 是梯度上升
    loss = torch.cat(policy_loss).sum()

    # 4. 反向传播
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

**为什么标准化回报**：不同 episode 的累计回报可能相差巨大（有的 episode 拿了 20 分，有的拿了 200 分）。标准化使所有 episode 的回报变为零均值单位方差：
- $G_t > 0$（比平均好）→ 增加对应动作的概率
- $G_t < 0$（比平均差）→ 减少对应动作的概率

这大幅降低了梯度方差，加速收敛。

**为什么用负号**：策略梯度定理告诉我们沿 $\nabla_\theta \log \pi_\theta \cdot G_t$ 的方向**上升**（梯度 ascent）。但 PyTorch 的优化器默认做梯度**下降**。加负号将梯度上升转化为梯度下降问题。

### 第7步：训练循环 — DQN vs REINFORCE 的关键区别

| 维度 | DQN | REINFORCE |
|------|-----|-----------|
| 更新时机 | 每步更新（off-policy） | Episode 结束后更新（on-policy） |
| 可复用历史数据 | 是（经验回放） | 否（只能用当前策略的数据） |
| 探索方式 | ε-贪婪（超参数控制） | 自然随机（策略分布采样） |
| 损失函数 | MSE(Q值, TD目标) | $-\sum \log\pi \cdot G_t$ |
| 收敛速度 | 较快（bootstrapping） | 较慢（Monte Carlo） |

### 第8步：可视化 — 三张图理解训练效果

1. **训练奖励对比图**（`dqn_vs_reinforce.png`）：原始奖励 + 滑动平均 + 累计平均，对比两种算法的收敛速度和最终性能
2. **DQN 损失曲线**（`dqn_loss_curve.png`）：MSE 损失随训练步数的变化，可以判断网络是否在学习（但注意：DQN 的损失不一定单调下降，因为 TD 目标本身也在变化）
3. **REINFORCE 策略热力图**（`reinforce_policy_heatmap.png`）：在 (杆角度, 角速度) 平面上展示策略的决策边界——黑色虚线是 $\pi(\text{右推}|s)=0.5$ 的分界线

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| Q-Network | 输入状态，输出所有动作的 Q 值（函数逼近而非查表） | `QNetwork` 类 |
| 经验回放 | 随机采样历史经验打破序列相关性 | `ReplayBuffer` 类 |
| 目标网络 | 冻结 TD 目标计算，每 100 步更新一次 | `target_network` / `target_update_freq` |
| ε-贪婪 | 以 ε 概率随机探索，以 1-ε 概率利用 | `choose_action()` |
| 策略网络 | 输出动作的概率分布 $\pi_\theta(a\mid s)$ | `PolicyNetwork` 类 |
| 回报 G_t | 从 t 时刻到 episode 结束的折扣累计奖励 | `finish_episode()` 中的 `returns` |
| 回报标准化 | 减均值除标准差，降低梯度方差 | `returns = (returns - mean) / (std + 1e-8)` |
| 策略梯度损失 | $L = -\sum \log\pi_\theta(a_t\mid s_t) \cdot G_t$ | `policy_loss.append(-log_prob * G_t)` |

## 完整代码

<<< @/nn-decision/rl/deep-rl/code/demo.py
