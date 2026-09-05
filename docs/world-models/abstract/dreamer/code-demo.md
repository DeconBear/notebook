---
title: "wm03 Dreamer — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# Dreamer — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/dreamer/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/dreamer/code
python demo.py
```

`main()` **先**跑倒立摆（与 [PETS](/world-models/abstract/pets/)、[LeWM](/world-models/abstract/lewm/) 同一套物理），**再**短跑离散走廊对照。CPU 约 2–3 分钟。不是 DreamerV3：无像素、无 RSSM 随机状态、无离散潜变量。要看 $h_t/s_t$ 先验后验，先读 [RSSM 代码讲解](/world-models/abstract/rssm/code-demo)。本章补的是 RSSM 之后那一跳：**世界模型冻住，在想象轨迹上更新 Actor-Critic**。

## 代码逐段详解

### 第1步：导入与种子

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

和 RSSM 章相同：`nn` 搭网络，`F` 提供 `mse_loss` / `one_hot` / `softplus` / `binary_cross_entropy_with_logits`，`optim.Adam` 更新。倒立摆 Actor 是连续力矩，走廊 Actor 是离散左右，两套头。

`set_seed(42)` 同时钉 NumPy 和 PyTorch。走廊附录里会再 `set_seed(42)` 一次，让 REINFORCE 基线和环境从同一随机源开始，比较才公平。

---

### 第2步：倒立摆环境 `Pendulum` — 观测为什么是 $\cos\theta,\sin\theta$

```python
def obs(self):
    return np.array(
        [np.cos(self.theta), np.sin(self.theta), self.omega],
        dtype=np.float32,
    )
```

$\theta$ 在 $-\pi$ 和 $\pi$ 处数值突变，网络会以为「跳了一大步」。$(\cos\theta,\sin\theta)$ 在圆上连续，和「角是周期的」一致。第三维角速度 $\omega$。`float32`：与 `nn.Linear` 对齐（见 RSSM 讲解）。

```python
def _wrap_pi(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
```

**语法 `%`**：浮点取模。把角折回 $(-\pi,\pi]$。积分 $\theta\leftarrow\theta+\Delta t\cdot\omega$ 会无限增大，不折回来奖励里的 $\theta^2$ 会炸。

```python
u = float(np.clip(action, -1.0, 1.0)) * self.max_torque
```

Actor 输出 $[-1,1]$，再乘 `max_torque=2` 变成真实力矩。**`np.clip`**：超出区间就钉在边界，防止策略刚开始乱输出把仿真打飞。

奖励：

```python
reward = np.exp(-8.0 * theta ** 2) * np.exp(-0.05 * omega ** 2) - 0.01 * (u / max_torque) ** 2
```

直立、角速度小 → 接近 1；力矩大有小惩罚。`**` 是平方。这是稠密奖励，玩具上比「只有直立才给 1」好学。

`reset` 把 $\theta,\omega$ 抽在 0 附近：演示「稳住」，不是从最低点摆起来（那需要更长训练）。

---

### 第3步：`PendulumWorld` — 确定性潜动力学（相对 RSSM 简化了什么）

```python
self.encoder = nn.Sequential(nn.Linear(3, 32), nn.ELU(), nn.Linear(32, z_dim))
self.dynamics = nn.Sequential(nn.Linear(z_dim + 1, 48), nn.ELU(), nn.Linear(48, z_dim))
self.reward_head = nn.Linear(z_dim, 1)
```

- **encoder**：观测 $o\to z$。没有后验高斯，相当于「确定性 RSSM 的 $h$」，随机性被拿掉以便 CPU 快训。
- **dynamics 输入 `z_dim + 1`**：潜状态拼上**一个**标量动作。`torch.cat([z, action], dim=-1)` 要求 `action` 的最后一维是 1，所以收集数据时 `traj['act'].append([action])` 带个括号，变成长度为 1 的列表，堆成 `(N, 1)`。
- **reward_head**：从 $z_{t+1}$ 预测 $r_t$。想象时环境不在，奖励只能由模型给。

```python
def imagine_step(self, z, action):
    z_next = self.dynamics(torch.cat([z, action], dim=-1))
    reward = self.reward_head(z_next).squeeze(-1)
    return z_next, reward
```

**`.squeeze(-1)`**：shape `(B, 1)` → `(B,)`。Critic/MSE 要 1 维向量，多出来的尺寸 1 会广播出错或警告。

**设计取舍**：真 Dreamer 的 `imagine_step` 走 RSSM 先验采样。这里动力学是确定性 MLP，想象轨迹不可随机分叉，但对「冻住模型、在 $z$ 里滚 horizon 步」这条主链足够。

---

### 第4步：`GaussianActor` / `PendulumCritic`

```python
h = self.net(z)                    # (B, 2)
mu = torch.tanh(h[..., :1])        # 第一列，压到 [-1,1]
std = F.softplus(h[..., 1:2]) + 0.08
```

- **语法 `h[..., :1]`**：`...` 表示「前面所有维都保留」。无论 `h` 是 `(2,)` 还是 `(B, 2)`，都取最后一维的第 0 个，并保持长度为 1，方便和 `z` 拼接。写成 `h[:, 0]` 会丢掉最后那维 1，`cat` 对不齐。
- **`tanh`**：力矩均值必须在 $[-1,1]$。
- **`softplus + 0.08`**：标准差恒正，且不小于 0.08，避免策略方差塌成 0、不再探索。

Critic：`Linear → ELU → Linear(1)`，输出 $V(z)$。`.squeeze(-1)` 同上。

---

### 第5步：收集真实轨迹 `collect_pendulum_episodes`

```python
with torch.no_grad():
    z = world.encode(obs_t)
    mu, std = actor(z)
    a = torch.clamp(mu + noise * torch.randn_like(mu), -1.0, 1.0)
```

- **`unsqueeze(0)`**：观测是 `(3,)`，Linear 要 `(B, 3)`，加一维 batch。
- **探索**：均值加高斯噪声，`noise` 随训练轮次从 0.4 降到 0.08（见 `train_dreamer_pendulum`）。不是学方差去采样（评估时只用 `mu`），收集数据时人为加噪，保证回放多样。
- **`with torch.no_grad()`**：交互不算梯度。真交互只写 replay，反传发生在后面「训世界模型 / 想象 AC」两段。

字典 `traj` 的 value 都是 list，`extend` 多条 episode 后变成一大袋转移，供 `train_pendulum_world` 随机抽 mini-batch。这是最简 replay：没有优先级、没有序列采样。

---

### 第6步：拟合世界模型 `train_pendulum_world`

```python
z = world.encode(obs_t[idx])
z_tgt = world.encode(next_t[idx]).detach()
z_pred, r_pred = world.imagine_step(z, act_t[idx])
loss = F.mse_loss(z_pred, z_tgt) + F.mse_loss(r_pred, rew_t[idx])
```

- **目标 $z'$ 用 encoder(next_obs)**，不是让动力学去拟合原始 $o'$。潜空间一致，想象时滚的是 $z$。
- **`.detach()`**：不要让「目标编码器」的梯度穿过 $z_{\text{tgt}}$ 去和预测抢。否则 encoder 可以同时改目标和预测，loss 假降。真 Dreamer 还有 stop-gradient / EMA；这里 detach 是最小实现。
- 两项 MSE：动力学 + 奖励。没有 KL（因为没有先验后验）。

`np.random.choice(n, size=min(64, n), replace=False)`：转移少于 64 就全用。`min` 防止 $n<64$ 时报错。

---

### 第7步：想象里的 $V_\lambda$ — Dreamer V1 的回报估计

```python
def _pendulum_v_lambda(rewards, values, bootstrap, gamma=0.99, lam=0.95):
    gae = torch.zeros_like(bootstrap)
    next_v = bootstrap
    for t in reversed(range(horizon)):
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        out.insert(0, gae + values[t])
        next_v = values[t]
    return torch.stack(out)
```

从后往前扫。`bootstrap` 是想象终点再往后的 $V(z_H)$，没有真实终止时用它补尾。

- **`delta`**：TD 残差 $r+\gamma V_{t+1}-V_t$。
- **`gae`**：广义优势。$\lambda=0.95$ 介于 1-step TD 和 Monte Carlo 之间：偏差小一点、方差也不至于爆。
- **`gae + values[t]`**：把优势加回 $V_t$ 得到 $V_\lambda$ 目标，给 Critic 回归，也给 Actor 当「这段想象值多少」。
- **`list.insert(0, ...)`**：每次插到开头，扫完顺序就是 $t=0\ldots H-1$。用 `append` 再 `reverse` 也行。
- **`reversed(range(horizon))`**：从 `horizon-1` 数到 `0`。

Critic：`mse_loss(values_t, v_lam.detach())`。目标 detach，避免 $V_\lambda$ 里的 bootstrap 和当前 $V$ 拧成一团。

Actor（第二段再想象一遍）：

```python
actor_loss = -v_lam.mean()
```

**直通梯度**：最大化想象回报的均值。`PendulumWorld` 的参数**不在** `actor_opt` 里，所以 `backward` 不会改动力学（即使没有 `no_grad`，Adam 也碰不到它们）。走廊版则显式 `with torch.no_grad(): imagine_step`，两种写法一个意思：策略更新不准把世界模型当「作弊通道」去改奖励头。

`torch.clamp(v_lam, -15, 15)`：想象初期模型很瞎，$V_\lambda$ 可能极大，夹一下防一步把 Actor 打飞。

`mu + std * randn`：想象时按当前高斯采样力矩，再 `clamp` 到 $[-1,1]$。和收集数据的「均值+外加 noise」略有不同：这里用网络自己的 `std`。

---

### 第8步：外循环 `train_dreamer_pendulum`

```python
for it in range(n_iters):          # 默认 14
    episodes = collect_pendulum_episodes(...)   # 真交互
    wm_loss = train_pendulum_world(...)         # 只改 world
    imagine_train_pendulum_ac(...)              # 只改 actor/critic
    eval_pendulum(...)                          # 只用 mu，无噪声
```

这就是正文说的 Dreamer 三件套。评估回报画在 `dreamer_pendulum.png` 左轴，$|θ|$ 在右轴。

`all_obs[-400:]`：想象起点从最近 400 帧真实观测里抽。太旧的观测分布和当前策略不一致。

---

### 第9步：走廊环境 `CorridorBalance` — 离散附录

11 个格子，奇数所以有唯一中心。动作 0 左 / 1 右。贴边 reward=-1 且 `done=True`；否则

$$
r = 1 - \frac{|\text{pos}-\text{center}|}{\text{center}}
$$

越靠中间越高。`one_hot(pos)`：长度为 11 的向量，当前位置为 1。**语法 `obs[pos] = 1.0`**：整数当下标。世界模型的 `obs_dim=11`。

这是为了好画：价值热力图一排 11 个数，策略柱状图「该往哪推回中心」。

---

### 第10步：走廊世界模型与离散 Actor

`TinyWorldModel` 比摆多一个 **done 头**：`done_head` 输出 logit，想象时用 `sigmoid` 当「这一步该停的概率」，回报不再往后传（`cont = 1 - dones`）。摆的 `done` 恒为 False，所以没这头。

```python
act_oh = F.one_hot(act_t, num_classes=n_actions).float()
```

离散动作是整数 0/1，动力学网要向量。**`one_hot`** 得到 0/1 矩阵，`.float()` 才能和 `z` 的 float 拼接。`nn.Linear(latent + n_actions, ...)`。

`Actor.act`：

```python
dist = torch.distributions.Categorical(logits=logits)
action = dist.sample()
return action.item(), dist.log_prob(action)
```

- **`Categorical(logits=...)`**：内部做 softmax，按概率抽样。
- **`.item()`**：0 维张量 → Python `int`，才能 `env.step(action)`。
- **`log_prob`**：策略梯度要 $\log\pi(a|z)$。走廊想象循环用它乘 advantage。

---

### 第11步：走廊的想象 AC（和摆的差别）

```python
with torch.no_grad():
    z_next, r_pred, d_logit = world.imagine_step(z.detach(), act_oh)
```

世界模型完全冻住，`z.detach()` 切断「策略 → $z$」的梯度（本实现动力学不可导回 Actor，和摆的直通不同）。回报用 Monte Carlo：

```python
G = critic(z).detach()
for t in reversed(range(horizon)):
    G = rewards[t] + gamma * (1.0 - dones[t].detach()) * G
```

再 `advantage = return - V`，`actor_loss = -(log_prob * advantage).mean()`。这是带基线的 REINFORCE，在**想象**轨迹上算，不是在真实 episode 上算——同一公式，数据源换成世界模型。

`train_world_model` 里 `binary_cross_entropy_with_logits(d_logit, done)`：数值上比先 sigmoid 再 BCE 稳。`done` 是 0/1 float。

---

### 第12步：REINFORCE 基线在比什么

`train_reinforce_baseline`：**没有**世界模型。每条真实 episode 算折扣回报，归一化后 $-log\pi\cdot G$。样本效率应当差：每次更新都要真走环境。图 `dreamer_vs_reinforce.png` 把 REINFORCE 的横轴线性缩放到 Dreamer 的 iter 数，只是视觉对齐，不是「同样交互次数」的严格横轴。看趋势：想象多练能否用更少真实步数把回报拉起来。

评估走廊策略时 `logits.argmax`：贪婪，不采样。

`F.softmax(logits, dim=-1)[0, 1]`：batch 里第 0 个样本、动作「向右」的概率，画柱状图。中心左侧应 $P(\text{右})>0.5$，右侧相反，才会把 agent 推回中间。

---

### 第13步：`main` 读什么图

| 文件 | 对应哪段代码 |
|------|----------------|
| `dreamer_pendulum.png` | `train_dreamer_pendulum` 评估曲线 |
| `dreamer_vs_reinforce.png` | 走廊 Dreamer vs REINFORCE |
| `dreamer_value_heatmap.png` | `plot_value_heatmap`，$V(z(\text{pos}))$ |
| `dreamer_policy_bars.png` | 各格 $P(\text{向右})$ |

若摆的 $|θ|$ 降得不明显：玩具步数少、动力学是确定性 MLP，属于演示上限，不是实现抄错。完整 Dreamer 用 RSSM + 更长交互。

---

### 关键概念速查表

| 概念 | 本章怎么落地 |
|------|----------------|
| 真交互只教模型 | `collect_*` + `train_*_world` |
| 想象里更新策略 | `imagine_train_pendulum_ac` / `imagine_and_train_actor_critic` |
| 冻住世界模型 | 不进 `actor_opt`，或 `no_grad` + `detach` |
| $V_\lambda$ | `_pendulum_v_lambda`，摆用；走廊用 MC return |
| `one_hot` / `Categorical` | 离散动作 |
| `tanh`+`softplus` | 连续力矩均值与方差 |
| `.squeeze(-1)` / `[..., :1]` | 对齐形状，避免 `cat` 失败 |
| `.item()` | 张量变 Python 标量再交给 `env.step` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/dreamer/code/demo.py`
