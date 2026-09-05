---
title: "PETS — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# PETS — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/pets/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/pets/code
python demo.py
```

CPU 约 1–2 分钟。`main()` 先跑一维质点（看清 CEM 收缩、MPC 只执行第一拍），再跑倒立摆直立稳定。不依赖 Gymnasium。这不是完整 PETS 论文：倒立摆闭环用岭回归 bootstrap 集成（`LinearEnsemble`），`ProbMLP` 写在文件里但**闭环没用它**——注释写明是为赶 CPU 时间。

和 [Dreamer](/world-models/abstract/dreamer/code-demo) 的差别：PETS **不训练策略网**，每一步用学到的动力学做 CEM 规划；Dreamer 冻住模型、在想象里更新 Actor。

## 代码逐段详解

### 第1步：超参数 — 每个数字管哪一段

```python
DT = 0.08          # 质点积分步长
HORIZON = 12       # CEM 计划长度
N_SAMPLE = 80      # 每轮 CEM 抽多少条动作序列
N_ELITE = 12       # 留下得分最高的几条，用来更新 μ、σ
CEM_ITERS = 6      # CEM 收缩轮数
N_ENSEMBLE = 5     # 集成成员数（PE）
N_PARTICLE = 8     # 每条动作序列用几颗粒子估回报（TS）
```

**设计**：`N_ELITE / N_SAMPLE ≈ 15%`，太少方差估不准，太多精英里掺进差轨迹，均值收不紧。`N_ENSEMBLE=5` 够画出「模型之间意见不合」，再多 CPU 浪费在重复岭回归上。

---

### 第2步：真实质点 `true_step` — 模型要拟合的东西

```python
def true_step(pos, vel, acc):
    vel = vel + DT * (acc - DAMPING * vel) / MASS
    pos = pos + DT * vel
    vel = vel + 0.01 * np.random.randn()
    return pos, vel
```

牛顿：$a=(u-c v)/m$，再欧拉积分。最后一行过程噪声：没有它，确定性线性模型会「过自信」；有了它，集成残差方差 `var` 才不为 0，TS 采样才有意义。

**语法 `np.random.randn()`**：标准正态标量。后面 `randn(2)` 是长度为 2 的向量。

---

### 第3步：`fit_ensemble` — 为什么预测 $\Delta s$ 而不是 $s'$

```python
x = np.stack([t[0] for t in transitions])  # (N, 3) pos,vel,acc
y = np.array([t[1] for t in transitions])  # (N, 2) dpos,dvel
idx = np.random.randint(0, n_data, size=n_data)  # 有放回，bootstrap
xb = np.concatenate([xb, np.ones((len(xb), 1))], axis=1)  # 偏置列
a = xb.T @ xb + 1e-2 * np.eye(xb.shape[1])
w = np.linalg.solve(a, xb.T @ yb)
```

- **残差目标** $y=\Delta s$：下一步 = 当前 + 预测增量。位置量级大时直接回归 $s'$ 容易学成恒等映射。
- **`randint(..., size=n_data)`**：有放回抽样，每个成员看到略不同的数据集 → 参数分歧，这就是 **ensemble PE**（认知不确定）。
- **拼全 1 列**：线性模型 $w$ 含偏置，否则过原点。
- **岭回归** $X^\top X+\lambda I$：$\lambda=10^{-2}$ 防止 $X^\top X$ 接近奇异时 `solve` 炸掉。`np.linalg.solve` 比先求逆再乘更稳。
- **`resid.var`**：各维残差方差，当 **aleatoric** 噪声，`predict_step` 里再采样。`maximum(..., 1e-4)` 防止方差估成 0。

**语法 `list[t[0] for t in transitions]`**：列表推导。`np.stack` 把 list of arrays 叠成二维。

---

### 第4步：`predict_step` — PE 怎么用

```python
b = np.random.randint(len(models)) if member is None else member
mean = feat @ w
noise = np.random.randn(2) * np.sqrt(var)
```

`feat = [pos, vel, acc, 1]` 必须和拟合时的 4 列一致。`@` 是矩阵乘；`w` 形状 `(4, 2)`，得到 `(2,)` 的 $\Delta$ pos、$\Delta$ vel。

`member is None`：每步重抽成员（短视 PE）。传入固定 `member`：整条轨迹绑死一个成员（**TS∞**），避免每步换模型把轨迹拧成「平均世界」——PETS 论文认为 TS∞ 对规划更诚实。

---

### 第5步：`rollout_return` 与 CEM

```python
r -= (pp - target) ** 2 + 0.02 * a ** 2
elite = seqs[np.argsort(scores)[-N_ELITE:]]
mu = 0.7 * elite.mean(axis=0) + 0.3 * mu
std = 0.7 * elite.std(axis=0) + 0.3 * std + 0.05
```

回报是负代价：离目标远、加速度大，分数低。CEM 要**最大化** `scores`，所以 `argsort` 取**最后** `N_ELITE` 个（最大的）。

- **`0.7 * elite + 0.3 * old`**：动量，防止一轮坏采样把 $\mu$ 拽飞。
- **`std + 0.05`**：地板，避免方差塌成 0 后不再探索。

**语法 `np.argsort(scores)[-N_ELITE:]`**：升序索引的末尾 = 最大。`seqs[那些行]` 高级索引。

`cem_plan` 里 `seqs = mu + std * noise`，`noise` 形状 `(N_SAMPLE, HORIZON)`：**广播**后每条候选是一整段未来加速度。

MPC 只执行 `mu[0]`（第一拍），下一步重新规划。这就是「开环计划、闭环执行」：模型错了，下一步还能改。

---

### 第6步：倒立摆观测与奖励

```python
return np.array([np.cos(self.theta), np.sin(self.theta), self.omega], ...)
```

$\theta$ 在 $\pm\pi$ 处跳变，$(\cos,\sin)$ 在圆上连续。与 Dreamer/LeWM 同一套物理，方便对照三条路径。

`pendulum_reward` **只从观测算**（规划时没有真 $\theta$）：

```python
cos_t, omega = states[:, 0], states[:, 2]
return cos_t * np.exp(-0.05 * omega ** 2) - 0.05 * (1.0 - cos_t)
```

`cosθ=1` 直立。CEM 在模型想象的状态上累加这个函数，不调用 `env.step` 的指数奖励——两边形状类似即可，不必逐项相等。

`reset(swing_up=False)` 从直立附近出发：本 demo 做**稳住**，不做从最低点摆起。

---

### 第7步：`LinearEnsemble.sample_next` — 批量粒子

```python
mask = model_idx == b
next_s[mask] = states[mask] + mean + noise
nrm = np.sqrt(next_s[:, 0] ** 2 + next_s[:, 1] ** 2) + 1e-8
next_s[:, 0] /= nrm
next_s[:, 1] /= nrm
```

一次前向喂 `N×P` 个粒子。`model_idx` 标记每个粒子绑哪个成员。循环 5 个成员，用布尔 mask 只更新属于自己的行——比 Python 里 for 每个粒子调一次岭回归快。

**归一化 $\cos^2+\sin^2$**：线性残差会把点推离单位圆，不拉回来后续 `cos` 奖励失真。`1e-8` 防除零。

---

### 第8步：`cem_plan_pendulum` 的张量把戏

```python
particles = np.tile(state, (N, P, 1))     # 每条序列 × 每条粒子 复制当前观测
boot = np.random.randint(0, B, size=(N, P))  # TS∞：粒子×成员 开局抽一次，整段 horizon 不变
flat_a = np.repeat(acts[:, t], P)         # 同一动作复制给 P 个粒子
```

- **`np.tile(state, (N,P,1))`**：`state` 是 `(3,)`，结果 `(N, P, 3)`。
- **`np.repeat(acts[:, t], P)`**：动作序列第 $t$ 拍，每个样本重复 P 次，对齐 flatten 后的粒子。

`returns += pendulum_reward(ns).reshape(N, P).mean(axis=1)`：先对粒子平均（TS 估期望），再累加 horizon。返回 `mean[0]`：只执行第一拍。

---

### 第9步：外循环何时重新拟合

质点：每 8 步 `fit_ensemble(trans[-200:])`，只用最近 200 条，分布跟着当前策略走。

摆：每个 trial 开始 `ens.fit` 一次，trial 内边走边往 buffer 里塞，**下一** trial 才用上新数据。`|θ|>0.8` 则 `reset`，避免倒下后的转移把线性模型带偏（直立附近才近似线性）。

---

### 关键概念速查表

| 概念 | 本章落地 |
|------|----------|
| 集成 PE | bootstrap 岭回归，成员间分歧 |
| 残差方差 | `var`，逐步采样 |
| TS∞ | `member` 整条 rollout 固定 |
| CEM | 采样 → 精英 → 收缩 μ、σ |
| MPC | 只执行计划第 0 步 |
| `argsort[-k:]` | 取最大 k 个 |
| 观测 $(cos,sin,ω)$ | 角周期 |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/pets/code/demo.py`
