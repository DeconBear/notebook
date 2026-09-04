---
title: "s21 RLHF：当强化学习遇见大模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s21 RLHF：当强化学习遇见大模型 — exercise.py 练习指南

<a href="/notebook/code/nn-decision/rl/rlhf/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO，在 **LLM 记号**下再用一遍已经学过的零件：
1. PPO 裁剪 —— 公式见 [PPO 练习](/nn-decision/rl/ppo/code-exercise)，这里换成 token 概率比
2. DPO 损失 —— 本章新内容，绕过显式奖励模型
3. GAE —— 公式见 PPO 章，这里用在稀疏的序列末尾奖励上

## 预备知识

- PPO 裁剪目标：$\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t)\right]$
- 概率比：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$
- DPO 损失：$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta\cdot\left(\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right)$
- GAE 递推：$\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

## 任务清单

### TODO 1：实现 PPO 裁剪目标（`ppo_clipped_objective` 函数）

**任务**：实现 $\mathcal{L}^{\text{CLIP}} = \mathbb{E}[\min(r \cdot \hat{A},\; \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \cdot \hat{A})]$

**实现步骤**：
1. `surr1 = ratio * advantage` —— 未裁剪的原始目标
2. `clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)` —— 限制比率
3. `surr2 = clipped_ratio * advantage` —— 裁剪后目标
4. `objective = torch.min(surr1, surr2).mean()` —— 取 min 确保保守更新

**关键理解**：取 `min` 是 PPO 的精髓。无论 advantage 是正还是负，`min` 都确保不会因为过大的 $r_t(\theta)$ 而获得更高的目标值——这驱动策略做"小步"更新。

**预期输出**（clip_epsilon=0.2）：
```
测试 1 [正优势 Â > 0]:
  ratio = [0.5, 1.0, 1.5, 2.0, 3.0]
  objective ≈ 1.2200
  预期: (1.0 + 1.0 + 1.2 + 1.2 + 1.2) / 5 = 1.22
  第3-5个 ratio > 1.2 被裁剪为 1.2

测试 2 [负优势 Â < 0]:
  min 选择更负的目标值，裁剪防止 ratio 过度降低
  ratio < 0.8 时被裁剪到 0.8
```

### TODO 2：实现 DPO 损失函数（`dpo_loss` 函数）

**任务**：实现 $\mathcal{L}_{\text{DPO}} = -\log\sigma(\beta \cdot (\log(\pi_\theta(y_w)/\pi_{\text{ref}}(y_w)) - \log(\pi_\theta(y_l)/\pi_{\text{ref}}(y_l))))$

**实现步骤**：
1. `log_ratio_w = log_p_w - ref_log_p_w` —— 偏好回复的对数比率
2. `log_ratio_l = log_p_l - ref_log_p_l` —— 不偏好回复的对数比率
3. `diff = beta * (log_ratio_w - log_ratio_l)` —— 加权差值
4. `loss = -F.logsigmoid(diff)` —— DPO 损失

**为什么用 `F.logsigmoid` 而非 `-torch.log(torch.sigmoid(diff))`**：`F.logsigmoid` 在内部做了数值稳定处理。当 `diff` 很小时，`sigmoid(diff)` 接近 0，`log(接近0)` 会下溢；`F.logsigmoid` 用 `-softplus(-diff)` 的方式避免了此问题。

**预期输出**：
```
测试 1 [策略偏好 y_w (正确方向)]:
  log_ratio_w = -2 - (-3) = +1, log_ratio_l = -5 - (-3) = -2
  diff = 0.1 * (1 - (-2)) = 0.3
  loss ≈ -log σ(0.3) ≈ 0.555   (损失小 = 好)

测试 2 [策略偏好 y_l (错误方向)]:
  log_ratio_w = -5 - (-3) = -2, log_ratio_l = -2 - (-3) = +1
  diff = 0.1 * (-2 - 1) = -0.3
  loss ≈ -log σ(-0.3) ≈ 0.854   (损失大 = 差)

loss(正确方向) < loss(错误方向) ✓
```

### TODO 3：实现 GAE 优势估计（`compute_gae` 函数）

**任务**：实现递推形式的 GAE 计算。

**核心递推公式**（从后往前）：
$$
\hat{A}_t = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

**实现步骤**：
```python
for t in reversed(range(T)):
    # 1. 确定 V(s_{t+1})
    if t == T - 1:
        next_v = next_value      # 最后一步：终止后 value=0 或 last_value
    else:
        next_v = values[t + 1]   # 一般情况

    # 2. 计算 TD 误差
    delta = rewards[t] + gamma * next_v - values[t]

    # 3. 递推 GAE
    gae = delta + gamma * gae_lambda * gae

    # 4. 存储
    advantages[t] = gae
```

**关键参数**：
- `next_value`：如果 episode 终止（done=True），$V(s_T) = 0$；否则 $V(s_T) = \text{values}[-1]$
- `gae_lambda`：$\lambda$ 控制偏差-方差折中（0=低方差高偏差，1=高方差低偏差）

**预期输出**（5 步 episode，最后一步奖励=10.0，其他=0.0）：
```
GAE 优势: 大约 [6.99, 7.45, 7.94, 8.46, 9.0]
最后一步优势 ≈ 9.0（因为 δ₄ = 10 + 0 - 1 = 9.0）
所有优势 > 0: 是（GAE 将未来奖励反向传播到了前几步）
```

## 完成后的验证

全部三个 TODO 通过测试后，运行 `python code/demo.py` 观察：
1. PPO 训练过程中的 RM 分数上升（策略在变好）
2. KL 散度的变化（策略偏离了多少）
3. 策略熵的变化（探索程度是否在下降）
4. DPO 的偏好边际（y_w - y_l 的分数差距）是否在扩大

## 完整代码

<<< @/nn-decision/rl/rlhf/code/exercise.py
