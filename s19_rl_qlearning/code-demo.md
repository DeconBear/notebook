---
title: "s19 强化学习入门：MDP 与 Q-Learning — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s19 强化学习入门：MDP 与 Q-Learning — demo.py 代码详解

<a href="../code/s19_rl_qlearning/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s19_rl_qlearning/code
python demo.py
```

**依赖**：`numpy`, `matplotlib`

---

## 代码逐段详解

### 第1步：GridWorld 环境 — 强化学习的三要素

GridWorld（网格世界）是强化学习中最经典的测试环境。Agent 在一个二维网格中移动，目标是到达终点并获得正向奖励，同时避免陷阱带来的负向惩罚。

```python
class GridWorld:
    def __init__(self, size=10, start=(0,0), goal=(9,9),
                 traps=[(3,3), (5,5), (7,7)],
                 step_reward=-0.1, goal_reward=100.0, trap_reward=-50.0):
        self.action_deltas = [(-1,0), (1,0), (0,-1), (0,1)]  # 上/下/左/右
```

**环境配置对应 MDP 五元组**：

| MDP 元素 | GridWorld 实现 | 设计原因 |
|---------|---------------|---------|
| $\mathcal{S}$（状态空间） | 100 个离散状态（10x10 网格）, $s = (row, col)$ | 离散状态使 Q-Table 可行 |
| $\mathcal{A}$（动作空间） | 4 个离散动作：{上, 下, 左, 右} | 最简单的导航动作集 |
| $P(s'|s, a)$（状态转移） | 确定性转移（只有边界检查） | 简化问题，聚焦算法 |
| $R(s, a)$（奖励函数） | 终点 +100, 陷阱 -50, 每步 -0.1 | 引导 Agent 学习最短安全路径 |
| $\gamma$（折扣因子） | 0.95 | 对未来奖励适度打折 |

**奖励设计的关键**：
- **步数惩罚 -0.1**：鼓励 Agent 走最短路径。如果没有步数惩罚，Agent 可以任意绕路，只要最终到达终点。
- **终点奖励 +100**：正向信号，远大于步数惩罚的累计值（最优路径约 18 步 = -1.8），确保到达终点的策略优于中途徘徊。
- **陷阱惩罚 -50**：足够强烈，让 Agent 宁可绕远路也不要冒险。这比步数惩罚大三数量级，确保了安全性优先。

#### 1.1 `step()` 方法

```python
def step(self, action):
    dr, dc = self.action_deltas[action]         # 获取偏移量
    new_r = self.state[0] + dr
    new_c = self.state[1] + dc

    # 边界检查：如果移出网格，留在原地
    if 0 <= new_r < self.size and 0 <= new_c < self.size:
        self.state = (new_r, new_c)

    # 判断奖励和终止条件
    if self.state == self.goal:
        reward = self.goal_reward                 # +100
        done = True                               # episode 结束
    elif self.state in self.traps:
        reward = self.trap_reward                 # -50
        done = True
    else:
        reward = self.step_reward                 # -0.1
        done = False
    return self.state, reward, done
```

**`self.action_deltas` 的设计**：四个动作的偏移量 `[(-1,0), (1,0), (0,-1), (0,1)]` 对应上/下/左/右。这种编码方式简单高效，通过索引直接获取行和列的增量。

**边界处理**：如果 Agent 试图移出网格（如从 (0,0) 向上移），行为无效且 Agent 留在原地。这在强化学习中称为"absorbing boundary"——尝试无效动作不会导致 episode 终止，但会产生步数惩罚（因为 step 数增加了），从而让 Agent 学会不要撞墙。

#### 1.2 `get_state_index()` — 状态编码

```python
def get_state_index(self, state):
    return state[0] * self.size + state[1]       # 行优先编码: index = row * 10 + col
```

将二维坐标 $(row, col)$ 映射为一维索引（0 到 99）。这是 Q-Table 索引的基础——Q-Table 是一个二维 numpy 数组，第一维是状态索引。

---

### 第2步：Q-Learning Agent — "试错学习"的核心

#### 2.1 Q-Table 初始化

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_init=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = np.zeros((n_states, n_actions))   # Q 表全零初始化
        self.epsilon = epsilon_init                       # 初始 100% 探索
```

**Q-Table 的形状**：对于 10x10 网格（100 个状态）和 4 个动作，Q-Table 是 $100 \times 4$ 的矩阵：
- `q_table[0]`：状态 0（即 (0,0)）下 4 个动作的 Q 值
- `q_table[0, 3]`：在起点执行"右"动作的 Q 值

**初始化为零**：Q-Learning 表格方法通常将 Q 表初始化为零（或小的随机值）。初始化为零意味着 Agent 对环境的估值一开始是"中性"的——它认为任何动作在任何状态下的期望奖励都是 0。随着交互进行，Q 值逐渐收敛到真实值。

#### 2.2 $\varepsilon$-贪婪动作选择

这是强化学习**探索 vs 利用**（Exploration vs Exploitation）权衡的最佳体现：

$$
a_t = \begin{cases}
\text{随机动作} & \text{以概率 } \varepsilon \\
\arg\max_a Q(s, a) & \text{以概率 } 1 - \varepsilon
\end{cases}
$$

```python
def choose_action(self, state_idx):
    if np.random.random() < self.epsilon:
        action = np.random.randint(self.n_actions)    # 探索：随机选
    else:
        action = np.argmax(self.q_table[state_idx])   # 利用：选 Q 值最大的
    return action
```

**为什么 $\varepsilon$ 从 1.0 开始？** 初始 Q 表全为零，如果 $\varepsilon=0$，Agent 会一直选择第一个动作（`argmax` 在平局时返回第一个索引），永远无法探索其他动作。$\varepsilon=1.0$ 确保训练初期完全是随机探索，让 Agent 全面接触环境。

**`argmax` 在平局时的行为**：当多个动作的 Q 值相同时（如初始化时全为零），`np.argmax` 返回**第一个**最大值的索引。这意味着纯利用（$\varepsilon=0$）时 Agent 会有一个系统性偏差——总是选动作 0（上）。这也更凸显了探索的必要性。

#### 2.3 Q-Learning 的 TD 更新 — 算法的核心

这是整个 demo 中最重要的一行代码：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ \underbrace{r + \gamma \cdot \max_{a'} Q(s', a')}_{\text{TD 目标}} - \underbrace{Q(s, a)}_{\text{当前估计}} \right]
$$

```python
def update(self, state_idx, action, reward, next_state_idx, done):
    current_q = self.q_table[state_idx, action]         # Q(s,a)

    if done:
        td_target = reward                               # 终止状态：未来价值为 0
    else:
        max_next_q = np.max(self.q_table[next_state_idx]) # max_{a'} Q(s', a')
        td_target = reward + self.gamma * max_next_q      # r + γ·max Q(s',a')

    td_error = td_target - current_q                     # TD 误差 δ
    self.q_table[state_idx, action] += self.alpha * td_error  # Q += α·δ
```

**逐项解释**：

1. **`current_q`**：当前对 $(s, a)$ 的价值估计。这是更新前的"旧认识"。

2. **`td_target`（TD 目标）**：我们**认为** $(s, a)$ 应该值多少。这是由即时奖励 $r$ 加上打折后的未来最佳价值组成。注意这里使用了 $\max_{a'} Q(s', a')$ 而非实际执行的下一动作的 Q 值——这是 **off-policy** 性质的核心体现：我们用最优策略的价值来更新当前策略。

3. **`td_error`（TD 误差 $\delta$）**：TD 目标与当前估计的差距。正值表示"之前低估了这个动作"，负值表示"之前高估了"。

4. **`q_table += alpha * td_error`**：朝着 TD 目标的方向走一小步（步长由 $\alpha$ 控制）。

**终止状态的特殊处理**：当 $s'$ 是终止状态（终点或陷阱），没有"未来"，所以 TD 目标就是即时奖励 $r$（$\gamma \cdot 0 = 0$）。这是合理的——到达终点后的"未来"没有更多奖励。

**Off-Policy 的含义**：在更新时，我们用 $\max_{a'} Q(s', a')$ 来估算 TD 目标——这是**最优策略**的行为（总是选 Q 值最大的动作）。但 Agent **实际**执行的动作是由 $\varepsilon$-贪婪策略决定的（可能包含随机探索）。这种"学习最优策略，但用任意策略收集数据"的特性使得 Q-Learning 是 off-policy 的。

#### 2.4 $\varepsilon$ 衰减

```python
def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min,                # 不低于最小探索率
                       self.epsilon * self.epsilon_decay) # 指数衰减
```

**指数衰减**：$\varepsilon_t = \varepsilon_0 \times \text{decay}^t$。这是一种"退火"（annealing）策略：
- Episode 0: $\varepsilon = 1.0 \times 0.995^0 = 1.0$（完全探索）
- Episode 100: $\varepsilon = 1.0 \times 0.995^{100} \approx 0.606$
- Episode 500: $\varepsilon = 1.0 \times 0.995^{500} \approx 0.082$
- Episode 2000: $\varepsilon = \max(0.01, 1.0 \times 0.995^{2000}) \approx 0.01$（几乎纯利用）

**`epsilon_min` 保底**：即使训练到最后，也保留 1% 的探索概率。这是为了防止 Agent 陷入局部最优——万一学到的"最优"策略其实不是全局最优，保留少量探索机会能让 Agent 有机会发现更好的策略。

---

### 第3步：训练循环

```python
def train_agent(env, agent, n_episodes=2000, max_steps=500):
    for ep in range(n_episodes):
        state = env.reset()                              # 回到起点
        state_idx = env.get_state_index(state)

        for step in range(max_steps):
            action = agent.choose_action(state_idx)       # ε-贪婪选动作
            next_state, reward, done = env.step(action)   # 与环境交互
            next_state_idx = env.get_state_index(next_state)
            agent.update(state_idx, action, reward,       # TD 更新
                        next_state_idx, done)
            state_idx = next_state_idx

            if done:
                break                                     # 到达终点或陷阱
        agent.decay_epsilon()                             # 衰减探索率
```

**一个 episode 的生命周期**：
1. 从起点出发
2. 每一步：选择动作 → 执行 → 获得奖励 → 更新 Q 表
3. 直到：到达终点、踩到陷阱或步数超过 max_steps
4. Episode 结束，衰减 $\varepsilon$

**`max_steps=500`** 作为安全网：如果 Agent 陷入循环或永远找不到终点，episode 不会无限进行。

#### 3.1 收敛检测

```python
if (converged_episode is None
    and len(recent_rewards) >= window_size
    and np.mean(recent_rewards) > 0
    and ep > 500):
    converged_episode = ep
```

当最近 100 个 episode 的平均奖励**首次**大于 0 时，认为 Agent 已收敛。奖励大于 0 意味着 Agent 找到了终点（+100 的终点奖励超过了步数惩罚和可能的陷阱惩罚）。

---

### 第4步：提取最优策略

训练完成后，Agent 的 Q-Table 已学会每个状态的最优动作。最优策略的提取方式很简单——在每个状态选择 Q 值最大的动作：

```python
def extract_optimal_path(env, agent):
    state = env.start
    path = [state]
    for _ in range(max_steps):
        state_idx = env.get_state_index(state)
        action = np.argmax(agent.q_table[state_idx])    # 纯利用，ε=0
        dr, dc = env.action_deltas[action]
        state = (state[0]+dr, state[1]+dc)
        path.append(state)
        if state == env.goal or state in env.traps:
            break
    return path
```

注意这里用 `argmax` 而非 $\varepsilon$-贪婪——因为我们已经训练完了，不再需要探索。最优路径应该直接展示"Agent 学到了什么"。

**价值传播的直观理解**：在训练初期，只有终点附近状态的 Q 值被更新（因为只有它们能直接获得终点奖励）。随着更多 episode 的进行，这些状态的 Q 值通过贝尔曼备份逐渐传播回更早的状态——这就是 Q-Learning 中"奖励信号像涟漪一样从终点扩散回起点"的直觉。

---

### 第5步：可视化

#### 5.1 Q 值热力图演化

绘制不同 episode 的 Q 值热力图，展示学习过程：
- Episode 0：全零（初始状态）
- Episode 50：终点附近开始出现高 Q 值（但范围有限）
- Episode 200：价值传播到网格中部
- Episode 500：几乎所有状态都有了合理的 Q 值
- Episode 1999：最优 Q 值趋于稳定

#### 5.2 最优策略可视化

在每个格子上用箭头标明最优动作方向，箭头的颜色和透明度由 Q 值大小决定：
- 绿色箭头：正向 Q 值（这个动作朝向奖励）
- 红色箭头：负向 Q 值（这个动作应避免）
- 箭头透明度：Q 值的绝对值越大，箭头越不透明

#### 5.3 训练奖励曲线

```python
def plot_training_rewards(episode_rewards, window_size=50):
    smoothed = np.convolve(rewards, np.ones(window_size)/window_size,
                          mode='valid')
    ax.plot(smooth_episodes, smoothed, 'b-', linewidth=2)      # 滑动平均
    ax.plot(episodes, rewards, 'lightblue', alpha=0.3)         # 原始奖励
```

**`np.convolve` 实现滑动平均**：将原始奖励序列与一个全为 1/window_size 的窗口做卷积，等价于每 window_size 个值的算术平均。滑动平均值更平滑，便于观察训练趋势——理论上应该从负值逐渐上升并趋于平稳。

---

### 第6步：消融实验 — $\varepsilon$ 衰减策略对比

```python
epsilon_configs = {
    "快速衰减 (decay=0.99)":    {"decay": 0.99},
    "中等衰减 (decay=0.995)":   {"decay": 0.995},
    "慢速衰减 (decay=0.999)":   {"decay": 0.999},
}
```

| 策略 | $\varepsilon$ 下降速度 | Episode 1000 时的 $\varepsilon$ | 效果 |
|------|----------------------|--------------------------|------|
| 快速衰减 | 快 | $\approx 0.00004$ | 早期转向利用，但可能陷入次优策略 |
| 中等衰减 | 中 | $\approx 0.0067$ | 平衡探索与利用 |
| 慢速衰减 | 慢 | $\approx 0.368$ | 长时间探索，收敛慢但更可能找到全局最优 |

**为什么快速衰减可能不好？** 如果探索不足，Agent 可能在学习初期偶然发现一条"还不错"的路径后就停止探索，永远无法发现更优的路径。这就是**探索-利用困境**的具体表现。

### 第7步：消融实验 — 学习率 $\alpha$ 对比

```python
alpha_configs = {"0.05": 0.05, "0.1": 0.1, "0.3": 0.3, "0.5": 0.5}
```

| 学习率 | 更新幅度 | 效果 |
|--------|---------|------|
| $\alpha=0.05$ | 小步 | 学习慢但稳定，不易震荡 |
| $\alpha=0.1$ | 适步（默认） | 平衡 |
| $\alpha=0.3$ | 大步 | 学习快但可能不稳定 |
| $\alpha=0.5$ | 非常大 | 容易震荡，Q 值可能在最优值附近剧烈波动 |

---

## 关键概念速查表

| 概念 | 公式 | 一句话 |
|------|------|--------|
| MDP | $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$ | 五元组形式化决策问题 |
| Q 函数 | $Q^{\pi}(s,a) = \mathbb{E}_{\pi}[\sum \gamma^k r_{t+k+1} | s_t=s, a_t=a]$ | 在状态 s 执行动作 a 后的期望累计奖励 |
| TD 更新 | $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max Q(s',a') - Q(s,a)]$ | Q-Learning 的核心更新规则 |
| TD 目标 | $r + \gamma \cdot \max_{a'} Q(s', a')$ | 理想情况下 $(s,a)$ 应该值多少 |
| TD 误差 | $\delta = r + \gamma\max Q(s',a') - Q(s,a)$ | 目标与实际估计的差距 |
| 学习率 $\alpha$ | 控制每次更新步长 | $\alpha=1$ 完全替换，$\alpha=0$ 不学习 |
| 折扣因子 $\gamma$ | 控制对未来的重视程度 | $\gamma=0$ 鼠目寸光，$\gamma=1$ 远见卓识 |
| $\varepsilon$-贪婪 | 以概率 $\varepsilon$ 随机，$1-\varepsilon$ 选最优 | 最简单的探索策略 |
| $\varepsilon$ 衰减 | $\varepsilon_t = \max(\varepsilon_{\min}, \varepsilon_0 \cdot \text{decay}^t)$ | 从探索逐渐转向利用 |
| Off-Policy | 用 $\max_{a'} Q(s',a')$ 更新，而非实际执行的 $a'$ | 学习最优策略，用任意策略收集数据 |
| Model-Free | 不需要知道 $P(s'|s,a)$ | 从交互中学习，不需要环境模型 |
| Q-Table | `np.zeros((n_states, n_actions))` | 存储每个状态-动作对的估计价值 |
| 价值传播 | 奖励从终点向起点反向传播 | Q-Learning 的收敛机制 |

---

## 完整代码

<<< @/snippets/s19_rl_qlearning/demo.py
