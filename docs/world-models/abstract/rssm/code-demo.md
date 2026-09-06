---
title: "wm02 经典起源与 RSSM — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm02 经典起源与 RSSM — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/rssm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/rssm/code
python demo.py
```

CPU 即可（`DEVICE = cpu`）。大约一两分钟：生成 300 条 2D 质点轨迹，训练简化 RSSM，画出训练曲线、想象 rollout、开环 vs 闭环误差。本文件不是完整 PlaNet：没有像素编码器、没有 CEM 规划，只把 RSSM 的 $h_t / s_t$、先验/后验和「做梦」rollout 跑通。

## 代码逐段详解

### 第1步：导入 — 每个名字在后面干什么

```python
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

- **`os`**：拼出 `images/` 路径。`os.path.dirname(os.path.abspath(__file__))` 是「当前 `.py` 所在目录」，再 `join(..., '..', 'images')` 写到章节图目录，无论从哪启动脚本路径都对。
- **`numpy`**：环境仿真、数据集。`np.zeros((T, 2))` 先开好形状再填；`rng.normal(0, σ, size=2)` 给 2 维噪声。
- **`matplotlib.rcParams['font.sans-serif']`**：中文字体列表，缺第一个就试下一个。**`axes.unicode_minus = False`**：否则负号会画成方块。
- **`typing.Tuple, List`**：标注返回值，例如 `-> Tuple[np.ndarray, np.ndarray, np.ndarray]` 表示三个数组。运行时不检查，给人看的。
- **`torch`**：张量与自动微分。`torch.from_numpy` 把 NumPy 数组转成模型能吃的张量（共享内存，改一边另一边也会变，训练前通常已经拷好）。
- **`torch.nn`**：`nn.Module`、`nn.Linear`、`nn.GRUCell`、`nn.Sequential`。
- **`torch.nn.functional as F`**：无状态函数：`F.mse_loss`、`F.softplus`。和 `nn.MSELoss()` 模块版功能类似，这里当函数调用更短。
- **`torch.optim`**：`Adam`。RSSM 参数用它更新。

```python
DEVICE = torch.device('cpu')
```

**`torch.device`**：告诉张量住在 CPU 还是 GPU。本章模型极小，强制 CPU，避免没装 CUDA 的机器报错。

```python
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
```

两套随机源都要钉死：NumPy 管环境噪声，PyTorch 管权重初始化和 `randn_like`。只设一个，轨迹或梯度仍会对不齐。

---

### 第2步：玩具环境 `ToyPointEnv` — 为什么不直接用真实机器人

RSSM 要学的是：**观测残缺（只有带噪位置），动力学由动作驱动，且每条轨迹参数不同**。2D 质点够小，能在 CPU 上训 250 epoch，同时逼模型必须用 $a_t$，不能背一条固定圆。

真实状态是位置 + 速度；模型只看到

$$
o_t = \text{pos}_t + \varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\sigma_{\text{obs}}^2)
$$

目标点绕圆转，PD 控制器给出加速度 $a_t$：

$$
a_t = k_p(\text{target}-\text{pos}) + k_d(\text{target\_vel}-\text{vel})
$$

**为什么用 PD 当「策略」？** 本章不学 Actor，只要「有一条合理的动作序列」。PD 保证质点大致跟着圆走，RSSM 的任务是：**看见 $(o_{1:t}, a_{1:t})$ 后，还能闭眼把后面的 $o$ 想出来**。

#### `rollout` 里的循环

```python
for t in range(T):
    time = t * self.dt
    target = r * np.array([np.cos(w * time + phi), np.sin(w * time + phi)])
    target_vel = r * w * np.array([-np.sin(w * time + phi), np.cos(w * time + phi)])
    action = self.k_p * (target - pos) + self.k_d * (target_vel - vel)
    true_pos[t] = pos
    obs[t] = pos + rng.normal(0, self.obs_noise_std, size=2)
    actions[t] = action
    vel = vel + self.dt * (action - self.damping * vel) + noise
    pos = pos + self.dt * vel
```

- **`np.array([...])`**：把两个标量收成 shape `(2,)` 的向量，才能和 `pos` 做减法。
- **先记 `true_pos[t] = pos` 再更新 `pos`**：这一帧的观测对应**当前**位置；动作也是在当前状态上算的，下一步才积分。顺序反了，动作和观测会对不齐。
- **`rng` 而不是 `np.random`**：把 `RandomState` 传进来，每条轨迹独立、可复现；全局 `np.random` 会被别处打乱。

#### `generate_dataset`：每条轨迹换一组 $(r,\omega,\phi)$

```python
r = rng.uniform(0.5, 1.4)
w = rng.uniform(0.6, 1.6) * rng.choice([-1, 1])
phi = rng.uniform(0, 2 * np.pi)
```

- **`uniform(a, b)`**：在区间里均匀抽样。半径、角速度都不固定。
- **`choice([-1, 1])`**：顺时针或逆时针。模型不能假设「永远往左转」。
- 返回三个数组，shape 都是 `(n_traj, T, 2)`。第三维是 $x,y$。

**语法：`dtype=np.float32`**。PyTorch 默认权重是 float32；NumPy 默认 float64。不转的话 `torch.from_numpy` 会得到 float64 模型，和 `nn.Linear` 的 float32 对不上，前向会报类型错误。

---

### 第3步：`RSSM` 四个积木 — 为什么要拆 $h$ 和 $s$

普通 VAE：一步 $x\to z\to\hat x$，没有时间。普通 RNN：只有确定性隐状态，说不清「我不确定」。RSSM 两套状态：

| 符号 | 代码 | 角色 |
|------|------|------|
| $h_t$ | `h`，长度 `deter_dim=32` | GRU 记住的长期趋势（圆有多快、往哪转） |
| $s_t$ | `s`，长度 `stoch_dim=4` | 高斯采样的随机状态（噪声、模糊） |
| 先验 $p(s_t\mid h_t)$ | `prior_net` | **不看** $o_t$，做梦时用 |
| 后验 $q(s_t\mid h_t,o_t)$ | `posterior_net` | **看** $o_t$，训练时当老师 |
| 解码 | `decoder` | $(h_t,s_t)\to\hat o_t$ |

```python
self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)
```

**`nn.GRUCell`**：单步 GRU，不是整段 `nn.GRU`。这是**类的实例**（一台带权重的机器），不是 \(h_t\)。类 vs 实例、以及 `Linear` / `Sequential` / Cell 的用法见 [编程基础 · torch.nn](/programming/nn/)。输入是「上一步 $s$ 和上一步 $a$ 拼起来」，隐状态是 $h$。公式：

$$
h_t = \mathrm{GRU}(h_{t-1},\, [s_{t-1}; a_{t-1}])
$$

为什么输入不是 $o_{t-1}$？观测有噪声，先压进 $s$，再让 GRU 吃 $s$ 和 $a$，确定性记忆少被噪声直接污染。

```python
self.prior_net = nn.Sequential(
    nn.Linear(deter_dim, hidden_dim), nn.ELU(),
    nn.Linear(hidden_dim, 2 * stoch_dim),
)
```

- **`nn.Sequential`**：按列表顺序接模块。前向就是 `x → Linear → ELU → Linear`。
- **输出 `2 * stoch_dim`**：一半当均值、一半当 $\log\sigma$ 的原料，后面 `_split_mean_std` 切开。一个网络出两套参数，比两个独立 Linear 少一层重复。
- **`nn.ELU()`**：指数线性单元，负区不截死（不像 ReLU），世界模型里常用，梯度更稳。

后验网的输入维是 `deter_dim + obs_dim`：把 $h_t$ 和 $o_t$ **拼在一起**再映射。解码器输入是 `deter_dim + stoch_dim`。

```python
super().__init__()
```

**语法**：`RSSM` 继承 `nn.Module`。必须先调父类构造，PyTorch 才会登记子模块（`self.gru` 等），否则 `model.parameters()` 是空的，Adam 更新不到权重。

---

### 第4步：`_split_mean_std` / `softplus` / 重参数化

```python
mean, log_std = torch.chunk(x, 2, dim=-1)
std = F.softplus(log_std) + 1e-3
```

- **`torch.chunk(x, 2, dim=-1)`**：沿最后一维切成两块。`dim=-1` 表示「无论前面有没有 batch 维，都切最后一维」。shape `(B, 8)` → 两个 `(B, 4)`。
- **不能直接用 `exp(log_std)` 当标准差？** 可以，但 `softplus(x)=\log(1+e^x)` 更不容易在 `log_std` 很大时爆。`+ 1e-3` 保证 $\sigma$ 不落到 0（后面 KL 要除 $\sigma_p^2$）。

```python
eps = torch.randn_like(mean)
return mean + std * eps
```

这是 VAE 的**重参数化**。若写成 `Normal(mean, std).sample()`，采样路径不可导，`mean`/`std` 收不到重建损失的梯度。改成 $\mu+\sigma\varepsilon$，$\varepsilon$ 当常数，梯度能流过 $\mu$ 和 $\sigma$。

**`randn_like(mean)`**：生成和 `mean` 同形状的标准正态。不必手写 `torch.randn(mean.shape)`，device/dtype 也一起对齐。

---

### 第5步：对角高斯 KL — 公式怎么落到一行代码

两个对角高斯（各维独立）的 KL 有闭式，不必 Monte Carlo：

$$
\mathrm{KL}(q\|p)=\sum_i\Bigl[\log\frac{\sigma_{p,i}}{\sigma_{q,i}}+\frac{\sigma_{q,i}^2+(\mu_{q,i}-\mu_{p,i})^2}{2\sigma_{p,i}^2}-\frac12\Bigr]
$$

```python
var_q, var_p = std_q ** 2, std_p ** 2
kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
return kl.sum(dim=-1)
```

- `**` 在 PyTorch 里是逐元素幂，和 NumPy 一样。
- **`sum(dim=-1)`**：对 `stoch_dim` 求和，每个样本一个标量，shape `(batch,)`。不要对 batch 维也 `sum`，后面还要 `.mean()` 做平均。
- **顺序是 KL(后验 $\|$ 先验)**：老师是 $q$（看见观测），学生是 $p$（只靠 $h$）。把 $p$ 拉向 $q$，想象时闭眼才能猜得准。反了就变成「把后验掐成先验」，后验坍缩。

---

### 第6步：`forward` — 教师强制的一步时序

训练时**每一步都喂真实 $o_t$** 算后验（teacher forcing），不是用自己的预测当下一步输入。否则早期预测很烂，误差会把 GRU 训崩。

```python
batch, T, _ = obs_seq.shape
h = torch.zeros(batch, self.deter_dim, device=obs_seq.device)
s = torch.zeros(batch, self.stoch_dim, device=obs_seq.device)
prev_action = torch.zeros(batch, act_seq.shape[-1], device=obs_seq.device)
```

- **`obs_seq.shape` 解包**：`(batch, T, obs_dim)`。`_` 表示「这个数我不用」。
- **`device=obs_seq.device`**：零张量和数据住同一块设备。数据在 CPU，零却在 CUDA，会报错。
- 第 0 步没有「上一动作」，用全零当 $a_{-1}$。

循环里每一步：

1. `h = self.gru(torch.cat([s, prev_action], dim=-1), h)`  
   **`torch.cat(..., dim=-1)`**：在特征维拼接。$(B,4)$ 和 $(B,2)$ → $(B,6)$，正好是 GRUCell 的 input_size。
2. `prior(h)`：闭眼猜 $s_t$。
3. `posterior(h, obs_seq[:, t])`：  
   **语法 `obs_seq[:, t]`**：所有 batch、第 $t$ 个时间步，得到 `(batch, obs_dim)`。`:` 是「这一维全要」。
4. `s = reparameterize(post_mean, post_std)`：训练用后验采样。想象阶段才改用先验。
5. `decode(h, s)` 对 $o_t$ 做 MSE。
6. `prev_action = act_seq[:, t]`：这一步的动作留给下一步 GRU。

```python
recon_loss = recon_loss + F.mse_loss(obs_pred, obs_seq[:, t], reduction='none').sum(-1).mean()
```

- **`reduction='none'`**：不立刻平均，保留每个样本、每个坐标的误差，shape `(B, 2)`。
- **`.sum(-1)`**：两个坐标加起来，变成每个样本一个标量。
- **`.mean()`**：对 batch 平均。再在循环外 `/ T`，得到「逐步平均的重建」。

KL 同样逐步 `.mean()` 再 `/ T`。最后 `torch.stack(obs_preds, dim=1)` 把长度为 $T$ 的 list 堆成 `(B, T, 2)`。**`stack` 会新增一维**；`cat` 是沿已有维拼接。这里要「时间维」，用 `stack(..., dim=1)`。

---

### 第7步：`imagine` — 热启动后闭眼做梦

```python
@torch.no_grad()
def imagine(self, context_obs, context_act, future_act):
```

**`@torch.no_grad()`**：装饰器。整个函数里不算梯度、不建计算图。评估做梦不需要反传，省内存也避免误更新。

两段循环：

**热启动 $C$ 步**（代码里 `context_len=10`）：和训练一样走 GRU + **后验**，但

```python
s = post_mean   # 不用采样
```

评估要看均值轨迹，采样噪声会让图画得抖。

**想象 $K$ 步**：不再读 `obs`，只

```python
prior_mean, prior_std = self.prior(h)
s = prior_mean
obs_pred = self.decode(h, s)
prev_action = future_act[:, t]
```

动作从哪来？测试时用数据集里**真实未来动作** `future_act`。本章验证的是「动力学是否想得准」，不是同时学策略。Dreamer 那章才在想象里让 Actor 自己出 $a$。

---

### 第8步：`filter_rollout` — 对照「一直睁眼」

每步都用后验均值解码。没有「未来不可用观测」的约束，误差几乎不该随步数飙。拿它和 `imagine` 比，才能说「开环变差是因为闭眼，不是因为解码器本身废了」。

---

### 第9步：训练循环 — free-nats、Adam、梯度裁剪

```python
idx = np.random.choice(n_train, batch_size, replace=False)
obs_batch = obs_train[idx]
```

- **`choice(..., replace=False)`**：无放回抽样一个 mini-batch。`replace=True` 可能抽到重复轨迹。
- 整数数组当索引：`obs_train[idx]` 是高级索引，取出那些行。

```python
kl_penalty = torch.clamp(kl_loss, min=free_nats)
loss = recon_loss + kl_beta * kl_penalty
```

**`torch.clamp(x, min=m)`**：小于 $m$ 的提到 $m$。KL 很小时不再加压，允许后验保留大约 `free_nats=0.5` nats 的信息。没有这截断，KL 项容易把 $q$ 掐成 $p$，随机状态废掉（posterior collapse）。Dreamer 论文里的 free bits / KL balancing 是同一家族。

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
optimizer.step()
```

- **`zero_grad()`**：清掉上一步梯度。PyTorch 默认累加，不清会串 epoch。
- **`clip_grad_norm_`**：全体参数梯度的 L2 范数若 $>5$，等比缩小。RNN/GRU 容易梯度爆炸，裁一下。函数名末尾 `_` 表示原地改 `.grad`。
- **`.item()`**：把 0 维张量变成 Python `float`，才能存进 list 画曲线。

---

### 第10步：三张图在验证什么

1. **`rssm_training_loss.png`**：重建应下降；KL 会先升（后验开始用随机维）再可能稳住。
2. **`rssm_imagination_rollout.png`**：灰线真实位置，蓝线热启动段，红虚线闭眼预测。对得上说明先验 + 动作已经抓住圆周动力学。
3. **`rssm_rollout_error_growth.png`**：
   - `np.linalg.norm(..., axis=-1)`：每个时间步的 2D 欧氏距离。
   - `.mean(axis=0)`：对测试轨迹平均，得到「第 $k$ 步的平均误差」。
   - 开环红线往上翘，闭环蓝线平：长视野仍会漂，所以 Dreamer 想象 horizon 通常只取十几步，而不是几百步。

`plot_imagination_rollout` 里 `zorder`：数字越大越画在上面，叉标记才不会被线挡住。`set_aspect('equal')` 让 $x,y$ 比例尺相同，圆看起来才是圆。

---

### 第11步：`main` 数据怎么切

```python
true_pos_all, obs_all, act_all = generate_dataset(n_traj=300, T=40, seed=42)
n_train = 250
obs_train = torch.from_numpy(obs_all[:n_train])
obs_test = torch.from_numpy(obs_all[n_train:])
```

**切片 `[:250]` / `[250:]`**：前 250 条训练，后 50 条测试。测试集的圆周参数训练时没见过，才能说「不是背训练轨迹」。

`RSSM(obs_dim=2, act_dim=2, deter_dim=32, stoch_dim=4)`：观测/动作都是平面向量；随机状态只有 4 维——故意压得很低，强迫 $h$ 承担「圆的规则」，$s$ 只补不确定的那一点。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| $h_t$ | 确定性记忆 | `nn.GRUCell` |
| 先验 | 闭眼 $p(s\|h)$ | `prior_net`，`imagine` 里用 |
| 后验 | 睁眼 $q(s\|h,o)$ | `posterior_net`，`forward` 里采样 |
| 重参数化 | $\mu+\sigma\varepsilon$ | `reparameterize` |
| KL | 让先验学会后验 | `kl_divergence` + `clamp(..., min=free_nats)` |
| `[:, t]` | 取时间步 $t$ | 所有序列循环 |
| `torch.cat(dim=-1)` | 拼特征 | GRU / 后验 / 解码的输入 |
| `@torch.no_grad()` | 评估不算梯度 | `imagine` / `filter_rollout` |
| teacher forcing | 训练逐步用真 $o$ | `forward` 循环 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/rssm/code/demo.py`
