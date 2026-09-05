---
title: "GRPO：组相对策略优化 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# GRPO：组相对策略优化 — demo.py 代码详解

<a href="/notebook/code/nn-decision/rl/grpo/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/rl/grpo/code
python demo.py
```

CPU 即可。36 道 $a+b$ 口算题，答案 $0\ldots 12$。每道题采 $G=8$ 个答案：无基线 REINFORCE 用绝对 0/1 奖励；GRPO 用组内 z-score 当 $\hat A$，再套 PPO 的 clip，并加一项对冻结参考策略的 KL。看左图正确率、右图「一组全对/全错被跳过」的比例。本文件没有 LLM、没有 token 级损失，只把「组相对优势」和裁剪跑通。

## 代码逐段详解

### 第1步：导入与题库

```python
N_A, N_B = 6, 6
N_ANS = N_A + N_B + 1    # 0 .. 12
N_Q = N_A * N_B          # 36
GROUP = 8
CLIP_EPS = 0.2
KL_BETA = 0.02
```

- **`N_ANS=13`**：和 $0+0$ 到 $5+5$ 的合法和一致，动作空间刚好盖住所有正确答案，没有「超出范围」的干扰项。
- **`GROUP=8`**：同一 prompt 采 8 条。太小则 $\sigma$ 估不准；太大则每步太贵。真 GRPO 往往 $G=4\sim 16$。
- **`CLIP_EPS`**：和 [PPO](/nn-decision/rl/ppo/code-demo) 同一套盒子。
- **`KL_BETA`**：把 $\pi$ 拴在参考策略旁，对应正文里「别离 SFT / 旧策略太远」。这里参考是**初始化那一刻冻结的 logits**。
- **`Categorical`**：离散答案上的策略。`.sample()` / `.log_prob` / `kl_divergence` 都靠它。

```python
def build_bank():
    for a in range(N_A):
        for b in range(N_B):
            qs.append(a * N_B + b)
            gold.append(a + b)
```

题目编号 $q=a\cdot N_B+b$，标签是 $a+b$。**不是**把 $(a,b)$ 拼成向量——后面 `AnswerPolicy` 直接用整数下标查表。`np.array` 方便和 `argmax` 比正确率。

---

### 第2步：`AnswerPolicy` — 「按题查表的小 LLM」

```python
class AnswerPolicy(nn.Module):
    def __init__(self, n_q=N_Q, n_ans=N_ANS):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_q, n_ans))

    def dist(self, q_idx):
        return Categorical(logits=self.logits[q_idx])
```

- **`nn.Parameter`**：一张 `(36, 13)` 的表，每道题一组 logits。全零起步 = 对 13 个答案均匀。没有 MLP，梯度直接改这张表——对应「小 LLM 的最后一层分类头」，把表示学习剥掉。
- **`self.logits[q_idx]`**：`q_idx` 是 Python `int` 时取出长度为 13 的 1 维张量；`Categorical` 在这 13 维上做 Softmax。
- **必须 `super().__init__()`**：否则这张表进不了 `parameters()`，`SGD` 更新不到。

`accuracy`：`argmax(dim=-1)` 贪心取每题最可能的答案，和 `gold` 比均值。评估不算梯度（`no_grad`）。

---

### 第3步：`group_advantages` — 组内 z-score，没有 $V_\phi$

PPO 的 $\hat A$ 来自 GAE + Critic。GRPO 换成：同一题的 $G$ 个标量奖励，减组均值、除组标准差。

$$
\hat A_i=\frac{r_i-\mu}{\sigma},\qquad
\mu=\frac{1}{G}\sum_j r_j
$$

```python
r = np.asarray(rewards, dtype=np.float64)
std = r.std()
if std < 1e-6:
    return np.zeros_like(r, dtype=np.float32)
return ((r - r.mean()) / (std + eps)).astype(np.float32)
```

- **比组内平均好** → $\hat A>0$，提高这些答案的概率；差的压低。这就是「相对」：全员 0 分或全员 1 分时，没有谁比平均更好。
- **`std < 1e-6`**：方差塌掉，整组优势为 0。后面 `grpo_step` 直接跳过 `backward`——没信号就别更新，避免 $0/0$ 附近的噪声梯度。
- **`float64` 算、`float32` 回**：和 PPO 里 GAE 的写法一样，先用更宽的浮点减均值。
- **没有 `compute_gae`，没有 `self.v`**：长思维链上训 $V$ 又贵又不稳，组采样反正都要做，基线就用组均值。

**同一组数，两种优势差在哪。** 设 $G=4$，奖励 $[1,0,1,0]$。$\mu=0.5$，样本标准差 $0.5$，z-score 约为 $[+1,-1,+1,-1]$：两个对的往上推、两个错的往下压，幅度对称。REINFORCE 则对的乘 $+1$、错的乘 $0$——只推对的，不主动压错的，也没有「比这道题的平均好多少」这把尺。若一组全是 $[1,1,1,1]$，z-score 全 0，GRPO 跳过；REINFORCE 仍会对四个 $\log\pi$ 各乘 1 再平均，把已经对的题继续顶尖。

和 [PPO 的 GAE](/nn-decision/rl/ppo/code-demo) 对照：那边 $\hat A_t$ 是时间轴上的多步 TD；这里 $\hat A_i$ 是**并列的 $G$ 个样本**之间的相对分。裁剪公式一字不差。同一组里 $\hat A$ 有正有负时，`min` 与 `clamp` 仍按样本各自生效：好答案的 $r$ 不能靠冲出 $1+\varepsilon$ 刷分，坏答案的 $r$ 也不能无底下降。

---

### 第4步：`grpo_step` — 先用旧策略采样，再 clip + KL

```python
dist_old = Categorical(logits=policy.logits[q_idx].detach())
for _ in range(group):
    a = dist_old.sample()
    answers.append(int(a.item()))
    logp_old.append(dist_old.log_prob(a))
    rewards.append(1.0 if answers[-1] == int(gold) else 0.0)
```

- **`.detach()`**：采样时的 logits 当常数。验证器是规则：猜中 `gold` 得 1，否则 0。这就是示意图里的「可验证奖励」，没有奖励模型。
- **`int(a.item())`**：张量 → Python 整数，才能和 `gold` 比、才能再 `torch.tensor(answers)`。

```python
adv = group_advantages(rewards)
if np.allclose(adv, 0):
    return float(np.mean(rewards)), True
```

`allclose` 把「全零优势」判定为跳过。返回的 `True` 累进 `skipped`，画右图。

有方差才更新：

```python
dist = policy.dist(q_idx)
new_lp = dist.log_prob(answers_t)
ratio = torch.exp(new_lp - old_lp)
surr1 = ratio * adv_t
surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t
clip_loss = -torch.min(surr1, surr2).mean()
```

和 PPO 章的 `ppo_update` 同一套：$r=\exp(\Delta\log\pi)$，`clamp` 进 $[1\pm\varepsilon]$，`min` 后取负（最大化 $L^{\mathrm{CLIP}}$）。差别只有 `adv_t` 的来源：这里是组内 z-score，不是 GAE。

**`old_lp = torch.stack(logp_old).detach()`**：把 $G$ 个标量 logπ 堆成向量。采样循环里每个 `log_prob` 仍连着 `dist_old`；再 `detach` 一次，保证比率的旧半边没有梯度。

```python
ref = Categorical(logits=ref_logits[q_idx])
kl = torch.distributions.kl.kl_divergence(dist, ref)
loss = clip_loss + KL_BETA * kl
```

离散 Categorical 的 KL 有闭式，不必 Monte Carlo。`ref_logits` 在 `train` 开头 `detach().clone()`，**整段训练不更新**——相当于「SFT 参考」。没有这项，clip 仍限制一步跨多远，但多步之后可以漂到只会背当前这几题。

`opt` 是 **SGD 不是 Adam**：表很小，固定步长更直观。`zero_grad` → `backward` → `step` 一次，对应论文里对一组样本的一拍（玩具没有再套 K 个 epoch）。

---

### 第5步：`reinforce_step` — 绝对 0/1，无基线、无 clip

```python
for _ in range(group):
    a = dist.sample()
    r = 1.0 if int(a.item()) == int(gold) else 0.0
    loss = loss - dist.log_prob(a) * r
loss = loss / group
```

同一组采样次数，但权重是原始 $r\in\{0,1\}$：猜错的项贡献 0，等于扔掉；猜对的按 $\nabla\log\pi$ 推一把。没有减均值，全错的题梯度全是 0（和 GRPO 一样没信号），全对的题却会**所有答案都乘 1** 去推——而它们已经对了，还在按绝对奖励加码，方差大、也容易把概率顶得过尖。没有 `clamp`，一步可以离开旧策略很远。

注意：这里 `dist` **没有**先 `detach` 再采样，采样与损失共用当前图，是经典 REINFORCE；GRPO 则显式分开 $\pi_{\mathrm{old}}$ 与 $\pi_\theta$。

---

### 第6步：`train` — 每步随机抽一题

```python
ref_logits = policy.logits.detach().clone()
opt = optim.SGD(policy.parameters(), lr=LR)
i = int(np.random.randint(0, len(qs)))
```

120 步，每步均匀抽一道题。`kind` 用不同 seed 偏移。`skip_frac = skipped / (step+1)` 是**累计**跳过比例：训练后期更多题被学对，一组 $G$ 个全 1 的机会上升，右图会往上爬——这不是故障，是「没方差就没信号」。

`draw_group` / `draw_vs_ppo` / `draw_verifier` / `draw_roadmap` 是正文框图。训练图：左正确率，右跳过比例。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 组采样 | 同一 $q$ 抽 $G$ 个答案 | `for _ in range(group)` |
| 组优势 | $\hat A_i=(r_i-\mu)/\sigma$ | `group_advantages` |
| 零方差 | 全对或全错 → 没梯度 | `std<1e-6` / `allclose` → `skip` |
| 裁剪 | 与 PPO 同一 $L^{\mathrm{CLIP}}$ | `clamp` + `min` |
| KL | $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ | `kl_divergence(dist, ref)` |
| 参考策略 | 初始化冻结的 logits | `ref_logits = ...clone()` |
| REINFORCE | 绝对 $r$，无基线无 clip | `reinforce_step` |
| 可验证奖励 | 对=1 错=0 | `answers[-1] == gold` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/rl/grpo/code/demo.py`
