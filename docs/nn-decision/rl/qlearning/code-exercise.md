---
title: "s19 强化学习入门：MDP 与 Q-Learning — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s19 强化学习入门：MDP 与 Q-Learning — exercise.py 练习指南

<a href="/notebook/code/nn-decision/rl/qlearning/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写 Q-Learning 的三个核心组件，建立对强化学习中"试错学习"的深刻直觉。完成后你将能够：

1. 独立写出 Q-Learning 的 TD 更新规则
2. 实现 $\varepsilon$-贪婪策略的动作选择
3. 理解动态环境（障碍物变化）对强化学习的影响

## 预备知识

- **Q-Learning 更新公式**：$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a)]$
- **MDP 五元组**：$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$
- **探索 vs 利用**：$\varepsilon$-贪婪策略在随机探索和贪婪利用之间权衡
- **Model-Free**：不需要知道环境模型 $P(s'|s,a)$

---

## 任务清单

### 练习 1：实现 Q-Learning 的 TD 更新规则

**目标**：补全 `q_learning_update()` 函数，这是 Q-Learning 算法的核心。

**核心公式**：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \underbrace{\left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]}_{\text{TD 误差 } \delta}
$$

**TODO 步骤**：

```python
def q_learning_update(q_table, state_idx, action, reward,
                      next_state_idx, done, alpha=0.1, gamma=0.95):
    # 步骤 1: 获取当前 Q(s, a)
    current_q = q_table[state_idx, action]

    # 步骤 2: 计算 TD 目标
    if done:
        # 终止状态：未来价值为 0，TD 目标只有即时奖励
        td_target = reward
    else:
        # 非终止状态：即时奖励 + 打折后的最佳未来价值
        max_next_q = np.max(q_table[next_state_idx])   # max_{a'} Q(s', a')
        td_target = reward + gamma * max_next_q

    # 步骤 3: 计算 TD 误差
    td_error = td_target - current_q

    # 步骤 4: 更新 Q 值
    q_table[state_idx, action] = current_q + alpha * td_error
    return q_table
```

**逐行详解**：

1. **`current_q = q_table[state_idx, action]`**：当前对 $(s, a)$ 的估值。初始时 Q 表全为零，Agent 什么都不知道。

2. **`done` 分支**：如果下一状态是终止状态（终点/陷阱），未来没有更多奖励，TD 目标就是即时奖励 $r$。这个分支确保了到达终点时 $(s, a)$ 的 Q 值直接反映终点奖励。

3. **`max_next_q = np.max(q_table[next_state_idx])`**：取下一状态所有可能动作的 Q 值中的最大值。这体现了 Q-Learning 的 **off-policy** 特性——我们用最优策略来选择下一动作（$\arg\max$），但实际执行的动作由行为策略（$\varepsilon$-贪婪）决定。

4. **`td_error = td_target - current_q`**：TD 误差告诉 Agent "你的估计偏离目标多少"：
   - 正 TD 误差 $\delta > 0$：当前估计太低，需要提高
   - 负 TD 误差 $\delta < 0$：当前估计太高，需要降低

5. **`q_table += alpha * td_error`**：朝 TD 目标的方向走一小步。步长由 $\alpha$ 控制。

**验证测试**：代码提供了两个测试用例——

**测试 1 — 非终止状态**：
- 初始 Q 表全零，state_idx=0, action=1, reward=5.0, alpha=0.1, gamma=0.9
- 更新后：$Q(0,1) = 0 + 0.1 \times (5.0 + 0.9 \times 0 - 0) = 0.5$

**测试 2 — 终止状态**：
- Q(1,0) 初始为 1.0, reward=10.0, done=True, alpha=0.5
- 更新后：$Q(1,0) = 1.0 + 0.5 \times (10.0 - 1.0) = 5.5$

**预期输出**：
```
TODO 1 测试: Q-Learning 的 TD 更新规则
  测试 1 [非终止状态]:
    更新前 Q(0,1) = 0.0
    更新后 Q(0,1) = 0.5000
    预期: 0.0 + 0.1 × (5.0 + 0.9 × 0 - 0.0) = 0.5
    ✓ 测试通过!
  测试 2 [终止状态]:
    更新前 Q(1,0) = 1.0
    更新后 Q(1,0) = 5.5000
    预期: 1.0 + 0.5 × (10.0 - 1.0) = 5.5
    ✓ 测试通过!
```

---

### 练习 2：实现 $\varepsilon$-贪婪动作选择

**目标**：补全 `epsilon_greedy_action()` 函数，实现探索与利用的权衡。

**核心规则**：
$$
a_t = \begin{cases}
\text{随机动作} & \text{以概率 } \varepsilon \\
\arg\max_a Q(s, a) & \text{以概率 } 1 - \varepsilon
\end{cases}
$$

**TODO 步骤**：

```python
def epsilon_greedy_action(q_table, state_idx, epsilon):
    if np.random.random() < epsilon:
        # 探索：随机选择任一动作
        action = np.random.randint(q_table.shape[1])
    else:
        # 利用：选择 Q 值最大的动作
        # 注意：当多个动作 Q 值相同时，应随机选择一个（而非总是第一个）
        q_vals = q_table[state_idx]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]   # 所有最大值动作的索引
        action = np.random.choice(best_actions)         # 随机选一个
    return action
```

**关键要点**：

1. **`np.random.random() < epsilon`**：`random.random()` 返回 `[0, 1)` 之间的均匀随机数。当它小于 $\varepsilon$ 时进入探索分支。

2. **`np.random.randint(q_table.shape[1])`**：从 $[0, n\_actions-1]$ 之间均匀随机选择。所有动作被选中的概率相等——这是"无偏"探索。

3. **最大值平局处理**：当多个动作的 Q 值相同（都是最大值）时，`np.argmax` 只返回第一个——这会造成系统性偏差。使用 `np.where(q_vals == max_q)[0]` 找出**所有**最大值的索引，然后 `np.random.choice` 随机选一个。这确保当 Q 值相同时（如初始全零），探索分支仍能均匀覆盖所有动作。

**验证测试**：
- **$\varepsilon=0$（纯利用）**：100 次选择应全部为最大 Q 值动作
- **$\varepsilon=1$（纯探索）**：500 次选择应覆盖全部 4 个动作
- **$\varepsilon=0.5$（混合）**：约 50% 利用（选动作 1 或 3），50% 探索（可能选动作 0 或 2）

**预期输出**：
```
TODO 2 测试: ε-贪婪动作选择
  测试 1 [ε=0]: 100 次选择结果={1, 3}
    ✓ 测试通过! (全部为最大 Q 值动作)
  测试 2 [ε=1]: 500 次选择的动作集合={0, 1, 2, 3}
    ✓ 测试通过! (覆盖了全部 4 个动作)
  测试 3 [ε=0.5]: 利用=~500, 探索=~500
```

---

### 练习 3：添加动态环境特征——移动障碍物

**目标**：实现动态变化的障碍物机制，理解强化学习在**非稳态环境**中的挑战。

**核心思想**：训练过程中，环境的奖励函数/转移概率发生变化——Agent 不仅要学习最优策略，还要适应环境的改变。这更接近真实世界（市场变化、用户偏好变化等）。

**TODO 步骤**：

```python
def add_moving_obstacle(env, cur_episode):
    periodic_idx = cur_episode // 100     # 每 100 episode 更换一次配置

    obstacle_configs = [
        [(1, 1), (3, 3)],   # 配置 0: 对角线障碍
        [(1, 3), (3, 1)],   # 配置 1: 反对角线障碍
        [(2, 2)],            # 配置 2: 中心障碍
        [(1, 2), (2, 1), (3, 2)],  # 配置 3: 密集障碍
    ]

    idx = periodic_idx % len(obstacle_configs)
    return obstacle_configs[idx]
```

**周期性变化的设计**：每 100 个 episode 换一次障碍物布局。这创建了一个"适应-变化-再适应"的循环：
- Episode 0-99：障碍在 (1,1) 和 (3,3)
- Episode 100-199：障碍换到 (1,3) 和 (3,1)
- Episode 200-299：障碍换到 (2,2)
- Episode 300-399：障碍换到 (1,2), (2,1), (3,2)
- Episode 400+：循环回到配置 0

**对 Agent 的影响**：
- 每次障碍物改变时，Agent 已学到的 Q 值可能不再适用
- Agent 需要通过新的探索来更新 Q 值
- 这测试了 Agent 的**适应能力**——能否快速从环境变化中恢复

**障碍物设计原则**：
1. 不能与起点 (0,0) 或终点 (4,4) 重合（否则 episode 无法正常开始或结束）
2. 不同配置间有足够差异——测试 Agent 是否能适应不同的环境布局
3. 配置 3 障碍物最多，路径选择最受限——最困难的场景

**预期输出**：
```
TODO 3 测试: 动态障碍物
  Episode    0 → 障碍物: [(1, 1), (3, 3)]
  Episode   50 → 障碍物: [(1, 1), (3, 3)]
  Episode  100 → 障碍物: [(1, 3), (3, 1)]
  Episode  199 → 障碍物: [(1, 3), (3, 1)]
  Episode  300 → 障碍物: [(2, 2)]
  Episode  399 → 障碍物: [(2, 2)]
  ✓ 周期性正确! Episode 0 和 100 配置相同
```

---

## 三个练习的关系

| 练习 | Q-Learning 组件 | 在算法中的位置 |
|------|----------------|-------------|
| 练习 1: TD 更新 | 学习规则 | 每一步交互后更新 Q 表 |
| 练习 2: ε-贪婪 | 行为策略 | 每次选择动作时调用 |
| 练习 3: 动态障碍 | 环境设计 | 改变 MDP 的 $R$ 和 $P$ |

这三个组件构成了 Q-Learning 的完整循环：用 ε-贪婪选择动作（练习 2）→ 执行动作并获得奖励 → 用 TD 更新改进 Q 表（练习 1）→ 面对动态变化的环境（练习 3）。

## 检查要点

运行 `python exercise.py`，确认：
- [ ] 练习 1 两个测试用例均通过（非终止状态 0.5, 终止状态 5.5）
- [ ] 练习 2 ε=0 只选最优动作，ε=1 覆盖所有动作
- [ ] 练习 3 障碍物周期性正确，相同周期内配置一致

完成练习后，返回 demo.py 观察完整的 GridWorld 环境、Q-Learning Agent 和消融实验——这些都是对练习中三个核心概念的深入应用。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/qlearning/code/exercise.py`
