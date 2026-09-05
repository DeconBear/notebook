---
title: "PPO：近端策略优化 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# PPO：近端策略优化 — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/ppo/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/rl/ppo/code
python demo.py
```

CPU 即可（`DEVICE = cpu`）。先画四张正文示意图，再在「走廊平衡」上各训 90 次迭代：REINFORCE vs PPO-Clip + GAE。重点看 `ppo-02-clip-curves.png`（$\hat A>0$ 时目标在 $1+\varepsilon$ 封顶）和 `ppo_vs_reinforce.png`（回报更稳、clip fraction）。本文件不是完整 OpenAI Baselines PPO：没有 mini-batch 打乱、没有价值裁剪，只把比率、裁剪和 GAE 跑通。

## 代码逐段详解

### 第1步：导入与超参

```python
from torch.distributions import Categorical
CLIP_EPS = 0.2
GAMMA = 0.97
LAM = 0.95
PPO_EPOCHS = 4
ENTROPY_COEF = 0.02
VF_COEF = 0.5
```

- **`os` / `_IMAGES_DIR`**：脚本相对路径写到章节 `images/`，从哪启动都对。
- **`numpy`**：环境、GAE 的 NumPy 递推。**`torch.nn` / `F` / `optim`**：Actor-Critic、MSE、Adam。
- **`Categorical`**：离散策略。给 logits，内部做 Softmax；`.sample()` 按概率抽动作，`.log_prob(a)` 是 $\log\pi(a\mid s)$。
- **`CLIP_EPS=0.2`**：正文 $\varepsilon$。比率锁在 $[0.8,1.2]$。
- **`LAM=0.95`**：GAE 的 $\lambda$，介于单步 TD 和整条 Monte Carlo 之间。
- **`PPO_EPOCHS=4`**：同一批 on-policy 数据反复更新 4 次——靠裁剪兜底，REINFORCE 不敢这么干。
- **`set_seed`**：NumPy（环境）和 PyTorch（权重、`sample`）两套随机源都要钉死。

`_box` / `_arrow` / `draw_*` 只负责示意图，算法在 `CorridorBalance` 到 `train_algo`。

---

### 第2步：`CorridorBalance` — 为什么走廊就够

奇数格一维走廊，中心奖励最高，走出两端失败（$r=-1$ 并 `done`）。动作 `0=左`、`1=右`。PPO 要学的是：**别被一次野更新把「待在中间」的策略踢飞**。格子少，CPU 上能把两种算法并排跑完。

```python
def reset(self):
    self.pos = int(np.clip(self.center + np.random.randint(-1, 2), 0, self.n_pos - 1))
```

起点在中心附近抖一格，避免永远从正中开局。**`np.clip(x, 0, n-1)`**：把数限制在闭区间。

```python
self.pos += -1 if action == 0 else 1
if self.pos < 0 or self.pos >= self.n_pos:
    return self.pos, -1.0, True
reward = 1.0 - dist / self.center
```

越靠近中心 `reward` 越接近 $1$。**`assert n_pos % 2 == 1`**：保证有唯一中心。`max_steps` 到了也 `done`，否则一条轨迹无限走。

---

### 第3步：`onehot` + `ActorCritic` — 离散格子怎么喂给 Linear

位置是整数 $0\ldots 8$，`nn.Linear` 要的是向量。最简单的编码：

```python
def onehot(pos, n=N_POS):
    x = torch.zeros(n, device=DEVICE)
    x[int(pos)] = 1.0
    return x
```

第 `pos` 维置 1，其余 0。**不是** embedding 表，没有可学的查找向量——格子就 9 个，one-hot 够了。`device=DEVICE` 让零向量和权重住同一块设备。

```python
self.body = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh())
self.pi = nn.Linear(hidden, n_act)
self.v = nn.Linear(hidden, 1)
```

共享躯干，两个头：策略 logits（2 维）、价值标量。**`squeeze(-1)`**：`v` 的形状 `(B,1)` 挤成 `(B,)`，后面才能和 GAE 回报逐元素减。

**`super().__init__()`**：继承 `nn.Module` 必须先调父类，否则 `parameters()` 是空的，Adam 更新不到。

---

### 第4步：`collect_batch` — 用旧策略采数据

```python
net.eval()
with torch.no_grad():
    x = onehot(pos).unsqueeze(0)
    logits, v = net(x)
    dist = Categorical(logits=logits)
    a = dist.sample()
    logp = dist.log_prob(a)
```

- **`unsqueeze(0)`**：`(9,)` → `(1,9)`，假装 batch=1，因为 `Linear` 习惯看见最前面一维是 batch。
- **`net.eval()` + `no_grad`**：采样不算梯度、不建图。旧的 $\log\pi$ 要当常数，后面 `old_logp.detach()` 再钉一次。
- **`.item()`**：0 维张量 → Python `int`/`float`，才能传给 `env.step`。
- **`last_v = 0.0`**：走廊在 `done` 时结束，bootstrap 价值当 0（失败或超时都没有「下一状态还要活」）。

`torch.stack(obs)` 把 list 堆成 `(T, 9)`。**`stack` 新增一维**；这里每步已经是长度为 9 的向量，堆完第一条维是时间。

---

### 第5步：`compute_gae` — 和 Dreamer 的 $V_\lambda$ 同一套递推

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad
\hat A_t=\sum_{\ell\ge 0}(\gamma\lambda)^\ell\delta_{t+\ell}
$$

```python
for t in reversed(range(t_len)):
    next_v = last_value if t == t_len - 1 else values[t + 1]
    next_nonterminal = 0.0 if dones[t] else 1.0
    delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
    gae = delta + gamma * lam * next_nonterminal * gae
    adv[t] = gae
returns = adv + values
```

- **`reversed(range(t_len))`**：从最后一步往回扫。后面的 GAE 已经算好，才能折进前一步。
- **`next_nonterminal`**：若本步 `done`，下一状态不存在，$V_{t+1}$ 与后续 GAE 都乘 0，避免把下一条轨迹的价值串进来（batch 是多回合首尾相接的）。
- **`returns = adv + values`**：$\hat A_t + V_t$ 就是 $V_\lambda$ 风格的回报目标，拿去训 Critic。

[Dreamer 的 `_pendulum_v_lambda`](/world-models/abstract/dreamer/code-demo) 是同一递推：先 `delta`，再 `gae = delta + γλ gae`，最后 `gae + values[t]`。区别：想象轨迹通常不在中途 `done`，所以没有 `next_nonterminal`；Dreamer 用 $V_\lambda$ 当 Actor 要最大化的标量，PPO 用 $\hat A$ 去乘比率再裁剪。$\lambda=0$ 退回单步 TD（低方差高偏差），$\lambda=1$ 接近 Monte Carlo。

---

### 第6步：`ppo_update` — 比率、裁剪、`min`

重要性采样比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}=\exp\big(\log\pi_\theta-\log\pi_{\mathrm{old}}\big)
$$

```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
old_logp = batch['logp'].detach()
ratio = torch.exp(logp - old_logp)
surr1 = ratio * adv_t
surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_t
policy_loss = -torch.min(surr1, surr2).mean()
```

- **标准化 $\hat A$**：减均值除标准差。量纲一，Adam 的 lr 才对得上不同回合。`+1e-8` 防一批优势全相同。
- **`.detach()`**：旧 logπ 当常数。不脱离的话梯度会流回「采样时的图」——但采样已在 `no_grad` 里，这里是双保险。
- **`exp(logp - old_logp)`**：在对数域做除法，比直接 $\pi_{\mathrm{new}}/\pi_{\mathrm{old}}$ 稳。
- **`torch.clamp(r, 1-ε, 1+ε)`**：把比率锁进盒子。
- **`min(surr1, surr2)`**：不能靠把 $r$ 推得更极端来刷分。$\hat A>0$ 时 $r>1+\varepsilon$ 不再加分；$\hat A<0$ 时 $r<1-\varepsilon$ 不再减分。
- **前面的负号**：我们要**最大化** $L^{\mathrm{CLIP}}$，PyTorch 的 `backward` 最小化，所以取负。

```python
value_loss = F.mse_loss(v, ret_t)
entropy = dist.entropy().mean()
loss = policy_loss + VF_COEF * value_loss - ENTROPY_COEF * entropy
nn.utils.clip_grad_norm_(net.parameters(), 1.0)
```

Critic 拟合 GAE 回报。熵奖励（前面是减号）防止过早变成确定性左右。**`clip_grad_norm_`**：全体梯度 L2 超过 1 就等比缩小；函数名末尾 `_` 表示原地改 `.grad`。

**`clip_frac`**：`((ratio-1).abs() > clip).float().mean()` —— 有多少样本的比率已经越出盒子。太高说明步子仍偏野，或 $\varepsilon$ 太紧。

同一 `obs`/`acts` 循环 `PPO_EPOCHS` 次：网络变了，`logp` 变了，`ratio` 会离开 1，裁剪开始咬合。

---

### 第7步：`reinforce_update` — 对照「无裁剪、无 Critic」

```python
for t in reversed(range(len(rews))):
    if dones[t]:
        g = 0.0
    g = rews[t] + GAMMA * g
    returns[t] = g
loss = -(dist.log_prob(batch['acts']) * ret).mean() - ENTROPY_COEF * dist.entropy().mean()
```

整条折扣回报 $G_t$ 当权重，**每批只 `step` 一次**，没有 `clamp`、没有 $V$。`dones[t]` 时把 $g$ 清零，同样是为了多回合拼接。优势用 $G$ 减均值除标准差，没有减 $V$，方差通常比 GAE 大。

---

### 第8步：`train_algo` 与图

```python
set_seed(SEED + (0 if kind == 'ppo' else 1))
batch = collect_batch(env, net, EPISODES_PER_ITER)
if kind == 'ppo':
    info = ppo_update(net, opt, batch)
```

`kind=='ppo'` 和 `'reinforce'` 用不同 `set_seed` 偏移，避免两条曲线撞同一随机性。每次迭代：用当前 $\pi$ 采 12 回合 → 更新 → 记 `mean(ep_rets)`。PPO 额外记 `clip_frac`。采完之后 $\theta_{\mathrm{old}}$ 就是这一拍的网络；下一迭代再 `collect_batch`，数据永远是新的 on-policy。

示意图：`draw_trust_region`（REINFORCE → TRPO → 裁剪）、`draw_clip_curves`（两条 $L^{\mathrm{CLIP}}$）、`draw_gae`（$\lambda$ 滑动尺）、`draw_loop`（采样 → GAE → K 次更新 → 旧策略跟着走）。训练图左：两条回报；右：越出 $[1\pm\varepsilon]$ 的比例。走廊太简单，PPO 不必显著更高分，图要看的是**更稳**——裁剪后的曲线少尖刺。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| one-hot | 格子 $i$ → 第 $i$ 维为 1 | `onehot` |
| 比率 | $r=\exp(\log\pi-\log\pi_{\mathrm{old}})$ | `torch.exp(logp - old_logp)` |
| 裁剪 | $\mathrm{clip}(r,1-\varepsilon,1+\varepsilon)$ | `torch.clamp(ratio, 0.8, 1.2)` |
| $L^{\mathrm{CLIP}}$ | $\min(r\hat A,\,\mathrm{clip}(r)\hat A)$ | `-torch.min(surr1, surr2).mean()` |
| GAE | 从后往前折 $\delta$ | `compute_gae` |
| $V_\lambda$ | $\hat A+V$，Dreamer 同源 | `returns = adv + values` |
| 熵奖励 | 防止策略过早变尖 | `- ENTROPY_COEF * entropy` |
| K epoch | 同批数据反复用 | `for _ in range(epochs)` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/ppo/code/demo.py`
