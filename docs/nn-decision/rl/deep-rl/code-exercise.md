---
title: "s20 深度强化学习：DQN 与 Policy Gradient — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s20 深度强化学习：DQN 与 Policy Gradient — exercise.py 练习指南

<a href="/notebook/code/nn-decision/rl/deep-rl/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO 任务，理解 DQN 和 REINFORCE 的核心实现细节：
1. 实现经验回放缓冲区的存储和采样
2. 实现 DQN 损失函数（含目标网络）
3. 实现 REINFORCE 策略梯度更新

## 预备知识

在开始前请确保理解：
- DQN 的 TD 目标公式：$y = r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a')$（终止态时 $y = r$）
- 目标网络 $Q_{\theta^-}$ 的作用：冻结 TD 目标的计算，防止训练振荡
- 经验回放的核心机制：FIFO 缓冲区 + 均匀随机采样
- REINFORCE 的更新公式：$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
- 折扣回报 $G_t$ 的递推计算：$G_t = r_t + \gamma \cdot G_{t+1}$（从后往前算）

## 任务清单

### TODO 1：实现经验回放缓冲区（`ReplayBuffer` 类）

**任务**：补全 `push()` 和 `sample()` 两个方法。

**push() 提示**：
- 将参数 `(state, action, reward, next_state, done)` 打包成一个 `tuple` 或 `Experience` 元组
- 用 `self.buffer.append(experience)` 存入 `deque`（FIFO，超容量时自动丢弃最旧的）

**push() 预期行为**：
- 存入 10 条经验后，`len(buffer)` 应该等于 10
- 存入超过 capacity 条经验后，旧的经验被自动淘汰

**sample() 提示**：
- 用 `np.random.choice(len(self.buffer), batch_size, replace=False)` 选取随机索引
- 遍历索引，提取对应经验的名字段
- 分别用 `np.array(...)` 堆叠成批量数组

**sample() 预期输出**：
```
states.shape      = (batch_size, state_dim)
actions.shape     = (batch_size,)
rewards.shape     = (batch_size,)
next_states.shape = (batch_size, state_dim)
dones.shape       = (batch_size,)
```

### TODO 2：实现 DQN 损失计算（`compute_dqn_loss` 函数）

**任务**：实现 DQN 的 TD 损失：

$$
\mathcal{L} = \text{MSE}\left(Q_\theta(s, a),\; r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a') \cdot (1-d)\right)
$$

**实现步骤**：
1. 用 `q_network(states)` 计算所有动作的 Q 值
2. 用 `.gather(1, actions)` 取实际执行动作的 Q 值 → `current_q`
3. 在 `torch.no_grad()` 下用 `target_network(next_states)` 计算目标 Q 值
4. 取 `.max(dim=1, keepdim=True)[0]` → `max_next_q`
5. TD 目标 = `rewards + gamma * max_next_q * (1 - dones)`
6. 损失 = `F.mse_loss(current_q, td_target)`

**关键易错点**：
- **必须在 `torch.no_grad()` 下计算目标网络**的输出，否则目标网络也会被反向传播更新
- **`(1 - dones)` 的维度**：dones 是 `(batch, 1)`，`1 - dones` 让终止态的 TD 目标 = reward（未来价值为 0）
- **`gather(dim=1, index=actions)` 的使用**：从所有动作的 Q 值中精确选取实际执行的那个动作的 Q 值

**预期输出**：
```
损失值 ≈ (1²+(-1)²+0.5²)/3 ≈ 0.75（当 Q 值初始接近零时）
在线网络接收到梯度：是
目标网络没被更新：是（正确，目标网络不应有梯度）
```

### TODO 3：实现 REINFORCE 策略梯度更新（`reinforce_update` 函数）

**任务**：实现完整的 REINFORCE 更新流程。

**实现步骤**：
1. **计算折扣回报 $G_t$**：从后往前遍历 `rewards` 列表
   ```python
   G = 0
   returns = []
   for r in reversed(rewards):
       G = r + gamma * G
       returns.insert(0, G)    # 插入头部保持时间顺序
   ```

2. **标准化回报**：`returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)`

3. **计算策略损失**：
   ```python
   policy_loss = 0    # 初始化为 0（或 torch.tensor(0.0)）
   for log_prob, G_t in zip(log_probs, returns_t):
       policy_loss = policy_loss - log_prob * G_t    # 累加负号
   ```

4. **反向传播更新**：
   ```python
   optimizer.zero_grad()
   policy_loss.backward()
   optimizer.step()
   ```

**关键理解点**：
- **为什么要累加**：REINFORCE 在一个 episode 的所有时间步上做**一次**更新——将所有步的策略梯度累加后统一更新
- **负号的作用**：PyTorch 做梯度下降，REINFORCE 是梯度上升（沿梯度方向增加目标 $J(\theta)$）。取负将上升转为下降
- **标准化只在 len>1 时做**：单个时间步无法计算标准差

**预期输出**：
```
最后一步的 G_t 最大（≈ 10.0），前面步骤的 G_t 较小（≈ 0.1-0.2）
由于最后一步奖励 10.0 远大于前几步的 -0.1，
策略梯度应该主要增加最后一步所选动作的概率
策略网络权重已更新
```

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察：
1. DQN 和 REINFORCE 在 CartPole-v1 上的训练奖励收敛过程
2. DQN 的损失曲线（注意到损失不会单调下降——因为 TD 目标本身在变化）
3. REINFORCE 的策略决策边界热力图


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/deep-rl/code/exercise.py`
