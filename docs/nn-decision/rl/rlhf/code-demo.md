---
title: "s21 RLHF：当强化学习遇见大模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s21 RLHF：当强化学习遇见大模型 — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/rlhf/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/rl/rlhf/code
python demo.py
```

**重要说明**：PPO 的裁剪与 GAE 已在 [PPO](/nn-decision/rl/ppo/) 讲过。本 demo 只是把同一套公式套在玩具语言模型上，再对比 DPO。完整 RLHF 需要数百 GPU 天；这里用小词汇表 LSTM，CPU 可跑完。

## 代码逐段详解

### 第1步：导入库 — 每个库做什么

```python
import torch                          # 深度学习框架
import torch.nn as nn                 # nn.Linear, nn.LSTM, nn.Embedding 等
import torch.nn.functional as F       # F.log_softmax, F.logsigmoid, F.cross_entropy 等
from torch.distributions import Categorical  # 类别分布 —— 采样动作和计算 log 概率
```

**关键引入**：`torch.distributions.Categorical` 是 RLHF 实现的核心工具。它将策略网络输出的 softmax 概率包装为概率分布对象，支持：
- `.sample()`：按分布采样 token
- `.log_prob(token)`：返回所选 token 的 log 概率 $\log \pi_\theta(a|s)$

### 第2步：玩具语言模型 — 模拟 LLM 在 RLHF 中的角色

**设计理念**：在真实 RLHF 中，策略 $\pi_\theta$ 是一个大语言模型（如 GPT）。在本 demo 中，我们用一个小型 LSTM 来模拟其核心行为——输入 token 序列，输出下一个 token 的概率分布。

**RLHF 的形式化**：
- **状态 $s_t$**：prompt + 已生成 token $(x, y_{<t})$
- **动作 $a_t$**：下一个 token $y_t$（从词汇表 $\mathcal{V}$ 中选择）
- **策略 $\pi_\theta(a_t|s_t)$**：LM 本身 —— 给定上下文，输出下一个 token 的概率
- **轨迹 $\tau$**：完整生成序列 $(x, y_1, y_2, \ldots, y_T)$

$$
\pi_\theta(a_t|s_t) = P_\theta(y_t | x, y_{<t})
$$

```python
class ToyLanguageModel(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, hidden_dim=128, num_layers=2):
        self.embedding = nn.Embedding(vocab_size, embed_dim)     # Token → 向量
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)         # 隐藏状态 → vocab 分布
```

**架构说明**（Embedding → LSTM → Linear → Softmax）：
- `nn.Embedding(30, 64)` 将 30 个 token 映射为 64 维连续向量
- `nn.LSTM(64, 128, 2)` 两层 LSTM 处理序列，输出 128 维隐藏状态
- `nn.Linear(128, 30)` 将隐藏状态投影回词汇表空间，产生 logits

**词汇表设计**：30 个 token = 4 个特殊 token + 26 个字母（a-z）。token 索引为：PAD=0, BOS=1, EOS=2, UNK=3, a=4, ..., z=29。

#### 关键方法：`get_log_probs()` — 计算序列中每个 token 的 log 概率

```python
def get_log_probs(self, input_ids):
    logits, _ = self.forward(input_ids)          # (batch, seq_len, vocab_size)
    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]                    # 后 seq_len-1 个位置作为标签
    log_probs = log_probs_all.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return log_probs                              # (batch, seq_len-1)
```

**为什么 `input_ids[:, 1:]` 是标签**：在自回归语言模型中，位置 $t$ 的输出应该预测位置 $t+1$ 的 token。因此 `logits[:, :-1, :]`（位置 0 到 T-1 的输出）的标签是 `input_ids[:, 1:]`（位置 1 到 T 的 token）。

**为什么用 `F.log_softmax` 而非 `F.softmax` 后取 log**：数值稳定性。`log_softmax` 在内部做了数值稳定处理（减去最大值），避免了 log(非常小的数) 造成的下溢。

#### 关键方法：`generate()` — 自回归生成

```python
def generate(self, prompt, max_len, temperature=1.0):
    for _ in range(max_len):
        logits, hidden = self.forward(generated[:, -1:], hidden)
        logits = logits.squeeze(1) / temperature    # 温度缩放
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        next_token = dist.sample()
        log_probs.append(dist.log_prob(next_token))
        if next_token.item() == EOS_TOKEN:
            break
    return generated, log_probs
```

**温度参数 $\tau$ 的作用**：$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$
- $\tau = 1.0$：原始分布（自然采样）
- $\tau < 1.0$：分布更尖锐，高概率 token 更可能被选中（更确定、更保守）
- $\tau > 1.0$：分布更平坦，低概率 token 更多机会被选中（更随机、更具创造性）

### 第3步：基于规则的奖励模型 — 模拟 RLHF 的 RM

在真实 RLHF 中，奖励模型 $R_\phi(x, y)$ 是一个训练好的神经网络，接受 (prompt, response) 并输出标量分数。本 demo 使用规则来模拟——核心目的是展示 RL 流程，而非追求真实奖励质量。

**奖励规则**（总分范围约 $[-3, 8]$）：
1. **长度奖励**：高斯形状，最优长度 $\approx 15$ 个字符：$R_{\text{len}} = 2.0 \cdot \exp\left(-\frac{(l-15)^2}{50}\right)$
2. **多样性奖励**：独特字符比例 × 3.0，鼓励使用更多不同字母：$R_{\text{div}} = 3.0 \cdot \frac{\text{unique}}{\text{total}}$
3. **连贯性奖励**：元音-辅音交替模式，最大 3.0 分
4. **短序列惩罚**：$\max(0, 3.0 - \text{length})$，太短扣分
5. **重复惩罚**：连续相同 token 每次加 0.5 罚分，上限 2.0

**为什么需要多样性和连贯性**：在真实 RLHF 中，人类标注者会偏好信息丰富、结构清晰的回复。这些规则是对人类偏好的粗略模拟——多样性对应"不重复说废话"，连贯性对应"逻辑流畅"。

### 第4步：PPO 实现 — RLHF 的核心强化学习环节

#### 4.1 PPO Agent 架构 — 四个模型

```python
class PPOAgent:
    def __init__(self, policy, ref_model, value_network, ...):
        self.policy = policy                # Actor: 策略 π_θ（正在被训练）
        self.ref_model = ref_model          # 参考模型 π_ref（冻结的 SFT 模型）
        self.value_network = value_network  # Critic: 价值函数 V_ψ(s)
```

**为什么需要 4 个模型**：
- **Actor $\pi_\theta$**：正在优化的策略 —— 唯一被更新的"主角"
- **Critic $V_\psi$**：估计状态价值，用于计算优势函数 —— 也需要训练
- **Reference Model $\pi_{\text{ref}}$**：冻结的初始 SFT 模型，用于计算 KL 惩罚
- **Reward Model $R_\phi$**：在本 demo 中是规则模型

#### 4.2 GAE 优势估计 — 平衡偏差与方差

**数学公式**：

$$
\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t) \quad \text{(TD 误差)}
$$

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \cdot \delta_{t+l}
$$

**递推实现**（从后往前）：

```python
def compute_gae(self, rewards, values, next_value, dones):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
        else:
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages
```

**GAE 参数 $\lambda$ 的含义**：
- $\lambda = 0$：只用单步 TD 误差（$\hat{A}_t = \delta_t$），低方差但高偏差（依赖不准确的 $V$）
- $\lambda = 1$：Monte Carlo 回报（$\hat{A}_t = G_t - V(s_t)$），无偏但高方差
- $\lambda = 0.95$（默认）：在偏差和方差间取折中

#### 4.3 PPO 裁剪目标 — 防止策略突变

**核心公式**：

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right) \right]
$$

**代码实现**：

```python
def ppo_update(self, old_log_probs, advantages, returns, states, actions, values, ref_log_probs):
    # 1. 计算概率比 r_t(θ)
    new_log_probs = self.policy.get_log_probs(actions)
    log_ratio = new_lp - old_lp.detach()       # log r_t(θ)
    ratio = torch.exp(log_ratio)                # r_t(θ)

    # 2. PPO 裁剪损失
    surr1 = ratio * adv                         # 未裁剪目标
    surr2 = torch.clamp(ratio, 1-ε, 1+ε) * adv  # 裁剪后目标
    policy_loss = -torch.min(surr1, surr2).mean()  # 取 min 确保保守更新

    # 3. KL 惩罚：防止奖励黑客
    kl_div = (new_lp - ref_lp).mean()            # KL(π_θ || π_ref)
    policy_loss = policy_loss + self.kl_coef * kl_div
```

**裁剪机制的直观理解**：
- **当 $\hat{A}_t > 0$（好动作）**：想增加概率，但最多允许 $r_t(\theta) \leq 1+\varepsilon$（防止过度自信）
- **当 $\hat{A}_t < 0$（坏动作）**：想降低概率，但最多允许 $r_t(\theta) \geq 1-\varepsilon$（防止过度惩罚）
- **取 $\min$ 的关键**：确保无论 advantage 符号如何，都不会因为更新幅度过大而获得更高的代理目标——这实现了"保守更新"

#### 4.4 KL 惩罚 — 防止奖励黑客的核心机制

$$
R_{\text{total}} = R_{\phi}(x, y) - \beta \cdot \text{KL}(\pi_\theta \parallel \pi_{\text{ref}})
$$

**为什么需要 KL 惩罚**：没有它，策略可能学会"奖励黑客"——找到让奖励模型打高分但实际无意义的回复模式。例如：
- 奖励模型可能偏好长句子 → 策略学会无限重复字母
- 奖励模型可能偏好某些特定词汇 → 策略滥用这些词

KL 惩罚像一根"橡皮筋"，把策略拉向初始模型——允许策略偏离一点来适应人类偏好，但不允许完全脱离预训练期间学到的语言能力。

```python
kl_div = self.compute_kl_divergence(new_lp, ref_lp)
# KL(π_θ || π_ref) ≈ mean(log π_θ - log π_ref)
policy_loss = policy_loss + self.kl_coef * kl_div  # β=0.1
```

#### 4.5 Value Network — Critic 的设计

```python
class ValueNetwork(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)   # 输出标量 V(s)
```

**输入**：LM embedding 层的输出（状态嵌入）—— 而非原始 token ID。这保证了 Critic 看到的表示与 Actor 看到的表示在同一个语义空间。

**输出**：标量 $V(s)$，表示从当前状态开始的期望累计奖励。Critic 用 MSE 损失训练：

$$
\mathcal{L}_V(\psi) = \text{MSE}(V_\psi(s_t), R_t^{\text{target}})
$$

其中 $R_t^{\text{target}} = \hat{A}_t + V(s_t)_{\text{old}}$ 是通过 GAE 估计的累计回报。

### 第5步：DPO 实现 — 绕过奖励模型的直接偏好优化

#### 5.1 DPO 的数学原理

DPO（Rafailov et al., 2023）的起点是 Bradley-Terry 偏好模型下的一个关键观察：最优策略可以反推出奖励函数：

$$
R^*(x, y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)
$$

将这代入偏好模型的损失函数后，$Z(x)$ 项被抵消，得到只依赖 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 的损失函数——**不需要显式训练奖励模型**。

#### 5.2 DPO 损失函数

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma\left( \beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

**代码实现**：

```python
def compute_dpo_loss(policy, ref_model, prompt, y_w, y_l, beta=0.1):
    # 1. 计算当前策略的 log 概率
    total_log_p_w = policy.get_log_probs(y_w).sum()    # log π_θ(y_w)
    total_log_p_l = policy.get_log_probs(y_l).sum()    # log π_θ(y_l)

    # 2. 计算参考模型的 log 概率（冻结，不计算梯度）
    with torch.no_grad():
        ref_log_p_w = ref_model.get_log_probs(y_w).sum()
        ref_log_p_l = ref_model.get_log_probs(y_l).sum()

    # 3. 计算对数比率
    log_ratio_w = total_log_p_w - ref_log_p_w    # log(π_θ/π_ref) for y_w
    log_ratio_l = total_log_p_l - ref_log_p_l    # log(π_θ/π_ref) for y_l

    # 4. DPO 损失
    diff = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(diff)                    # -log σ(diff)
    return loss
```

**直观解释**：
- 如果策略更偏好 $y_w$（好回复）且更不偏好 $y_l$（差回复），差值 $( \log\pi_w/\pi_{\text{ref},w} - \log\pi_l/\pi_{\text{ref},l} )$ 变大
- sigmoid 接近 1 → `logsigmoid` 接近 0 → 损失小（好）
- 如果策略偏好错误方向，"差值"变小或为负 → sigmoid 接近 0 → `-log(接近0)` → 损失大（差）

#### 5.3 偏好对生成

```python
def generate_dpo_preference_pair(policy, reward_model, prompt, n_candidates=4):
    # 用当前策略生成 n_candidates 个候选回复
    candidates, scores = [], []
    for _ in range(n_candidates):
        gen, _ = policy.generate(prompt, MAX_SEQ_LEN, temperature=1.0)
        score = reward_model.score(gen_tokens)
        candidates.append(gen)
        scores.append(score)

    # 选得分最高和最差的作为偏好对
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    return candidates[best_idx], candidates[worst_idx]
```

**为什么生成多个候选**：在真实 RHLF 中，标注者对同一个 prompt 的多个回复进行排序（如选 K=4 个中的最好和最差）。这里通过 reward_model 模拟这个过程。

### 第6步：训练循环 — 三阶段流水线

#### 阶段 1：SFT 预训练

```python
# 语言模型训练标准：交叉熵损失
loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

这是标准的自回归语言模型训练——最大化 $\log P(\text{response} | \text{prompt})$。

#### 阶段 2：PPO 训练循环

每个 episode 的流程：
1. 采样一个 prompt → 编码为 token
2. 用当前策略生成回复 → 得到轨迹（log_probs, rewards, values, states）
3. 用 RM 打分 → 构造奖励序列（中间步 = 0，最后步 = RM 分数）
4. 计算 GAE 优势 → 标准化
5. 执行 PPO 更新（裁剪损失 + KL 惩罚 + Critic MSE）

**关键：奖励只在最后一步**。LLM 的自回归生成中，中间 token 没有即时奖励——PPO 的 Critic $V(s_t)$ 通过"预测"最终奖励来引导 Actor。

#### 阶段 3：DPO 训练循环

每步的流程：
1. 采样一个 prompt
2. 用当前策略生成 4 个候选回复 → RM 打分 → 选出 ($y_w$, $y_l$)
3. 计算 DPO 损失 → 反向传播更新

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| RLHF 形式化 | 状态=已生成token序列, 动作=下一个token, 奖励=RM打分 | `ToyLanguageModel.generate()` |
| SFT | 监督微调 —— 交叉熵学习语言模式 | `pretrain_policy()` |
| PPO 裁剪目标 | $\min(r\hat{A}, \text{clip}(r,1-\varepsilon,1+\varepsilon)\hat{A})$ | `PPOAgent.ppo_update()` |
| KL 惩罚 | 防止策略偏离初始模型太远导致奖励黑客 | `compute_kl_divergence()` |
| GAE | 平衡偏差方差的优势估计，$\lambda=0.95$ | `compute_gae()` |
| DPO 损失 | 绕过 RM，直接从偏好数据优化策略 | `compute_dpo_loss()` |
| $F.\!logsigmoid$ | 数值稳定的 $\log\sigma(x)$，避免 softmax 溢出 | `-F.logsigmoid(diff)` |
| 温度参数 $\tau$ | 控制采样随机性，$\tau<1$ 更确定 | `generate(temperature=1.0)` |

### DPO vs PPO 对比

| 维度 | PPO | DPO |
|------|-----|-----|
| 需要奖励模型 | 是 | 否 |
| 需要在线采样 | 是（每步用当前策略生成） | 否（纯离线） |
| 维持模型数 | 4 个（Actor, Critic, Ref, RM） | 2 个（Policy, Ref） |
| 训练稳定性 | 需要仔细调参 | 较稳定（类似分类任务） |
| 理论最优性 | 依赖 RM 质量 | Bradley-Terry 下最优 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/rlhf/code/demo.py`
