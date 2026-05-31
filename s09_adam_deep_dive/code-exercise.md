---
title: "s09 Adam深度解析与训练实战 — exercise.py"
---

# s09 Adam深度解析与训练实战 — exercise.py 练习指南

<a href="../code/s09_adam_deep_dive/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过亲手补全 Adam 的偏差修正、AdamW 的解耦权重衰减、学习率 warmup 调度器，以及诊断 NaN loss 故障，掌握 Adam 内部机制的每一个关键细节。

## 预备知识

建议先阅读 s08 和 s09 的 index.md，确保理解：

| 概念 | 公式 | 说明 |
|------|------|------|
| Adam 一阶矩 | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | 梯度的指数滑动平均（方向） |
| Adam 二阶矩 | $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ | 梯度平方的指数滑动平均（尺度） |
| 偏差修正 | $\hat{m}_t = m_t / (1 - \beta_1^t)$ | 补偿从零初始化的偏差 |
| AdamW | $\theta - \alpha \cdot \text{Adam} - \alpha \lambda \theta$ | 权重衰减与自适应更新解耦 |
| Warmup | $\alpha_t = \alpha_{\text{target}} \cdot t / t_w$ | 训练初期线性增加学习率 |

---

## 任务清单

### 任务1：实现 Adam 的偏差修正

**描述**：补全 `AdamBiasCorrectionExercise.step()` 和 `get_correction_factors()` 方法。当前的 `step()` 缺少偏差修正，导致训练初期步长偏小、收敛缓慢。

**偏差修正的数学**：

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**为什么需要修正？** $m_t$ 从 0 初始化。第一步 $m_1 = (1-\beta_1) g_1 = 0.1 g_1$（设 $\beta_1=0.9$）——只保留了梯度的 10%。除以 $1-\beta_1^t = 1-0.9=0.1$ 后，$\hat{m}_1 = g_1$，完美补偿。

**提示**：
- 修正因子随时间增长：$t=1$ 时 $1-0.9^1=0.1$（大幅修正），$t=100$ 时 $1-0.9^{100} \approx 0.99997$（几乎无需修正）
- 对于 $\beta_2=0.999$，修正持续更久：$t=1000$ 时 $1-0.999^{1000} \approx 0.632$，仍有明显修正
- 用**修正后的** $\hat{m}_t$ 和 $\hat{v}_t$ 做参数更新，而不是原始的 $m_t$ 和 $v_t$

**`get_correction_factors()` 补充任务**：返回 $(1-\beta_1^t, 1-\beta_2^t)$ 的值。这个函数让你能观察到修正因子随时间的变化——越接近 1 说明偏差越小。

**期望输出**：
- 前 3 步显示"大幅修正"（因子 < 0.5）
- 第 4-6 步显示"小幅修正"
- 第 7 步以后显示"接近无修正"

---

### 任务2：实现 AdamW 的解耦权重衰减

**描述**：补全 `AdamWExercise.step()` 方法中的两个 TODO——Adam 自适应更新和独立的权重衰减。

**AdamW 的更新公式**：

$$
\theta_{t+1} = \underbrace{\theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam 自适应更新}} - \underbrace{\alpha \lambda \theta_t}_{\text{独立权重衰减}}
$$

**与 Adam+L2 的本质区别**：

- **Adam + L2**：把 $\lambda \theta$ 加到梯度里，然后梯度被 $\sqrt{\hat{v}}$ 缩放 → 不同参数受到的正则化强度不同
- **AdamW**：先做 Adam 自适应更新，再独立扣掉 $\alpha \lambda \theta$ → 所有参数受到相同比例的正则化

**提示**：
- Adam 部分（$m_t, v_t, \hat{m}_t, \hat{v}_t$，自适应更新）照常实现
- 关键是：在自适应更新之后，再加一行 `param -= self.lr * self.weight_decay * param`
- 注意权重衰减与 $\hat{m}_t$、$\hat{v}_t$ 完全无关——它独立地缩小每个参数

**期望输出**：
- Adam（无衰减）训练后的 $w$ 接近目标值 5.0
- AdamW（weight_decay=0.1）训练后的 $w$ 显著小于 5.0 —— 因为权重衰减持续把参数拉向 0

---

### 任务3：实现学习率 Warmup 调度

**描述**：补全 `WarmupSchedulerExercise.step()` 方法。

**Warmup 的线性增加公式**：

$$
\alpha_t = \begin{cases}
\alpha_{\text{target}} \cdot \dfrac{t}{t_{\text{warmup}}} & \text{if } t \leq t_{\text{warmup}} \\[1em]
\alpha_{\text{target}} & \text{if } t > t_{\text{warmup}}
\end{cases}
$$

**为什么需要 warmup？**
1. 模型参数刚开始是随机的，梯度方向不可靠
2. Adam 的 $m_t$ 和 $v_t$ 从 0 开始，需要时间积累
3. 特别是 $\beta_2=0.999$ 的二阶矩，需要几百步才能建立可靠的尺度估计
4. 在 Transformer 训练中，没有 warmup 几乎必定导致训练初期 loss 爆炸

**提示**：
- `self.current_step += 1` 递增计数
- `progress = self.current_step / self.warmup_steps`——线性插值比例
- 确保 `step > warmup_steps` 时 `lr = target_lr`，不要继续增大或减小

**期望输出**：
- Warmup 阶段（10 步内）：学习率从 0 线性增长到 0.001, 0.002, ..., 0.01
- 第 11 步起：学习率稳定在 0.01

---

### 任务4：诊断 — 调试 NaN Loss

**描述**：分析训练日志，选择最可能的原因和解决方案。

**场景**：
```
使用 Adam，lr=0.1（很大！）
Loss: 2.3 → 1.8 → 1.2 → 0.9 → NaN
梯度范数: 0.5 → 2.1 → 8.7 → 53.4 → NaN
```

**分析线索**：
- Loss 在逐步下降但突然变成 NaN → **不是**数据缺失问题（否则一开始就会 NaN）
- 梯度范数每步都在翻倍增长（0.5 → 2.1 → 8.7 → 53.4）→ **梯度爆炸**
- lr=0.1 对 Adam 来说偏大很多（Adam 的常用 lr 是 0.001）→ **学习率过大**

**正确答案**：
- **原因**：A. 学习率过大导致梯度爆炸
- **方案**：D. 以上全部（A + B + C）——降低学习率（核心）+ 开启梯度裁剪（保险）+ 检查数据（排除异常值）

**实战建议**：
1. 最优先：降低学习率到 0.001（Adam 默认值）
2. 保险措施：添加梯度裁剪 `max_norm=1.0`
3. 监控：每个 batch 打印梯度范数，观察是否在健康范围（$10^{-4}$ 到 $10^1$）

---

### 关键概念速查

| 任务 | 核心公式 | 最容易错的地方 |
|------|---------|--------------|
| TODO 1: 偏差修正 | $\hat{m} = m / (1-\beta_1^t)$ | 忘记 `**` 指数运算，写成 `1 - beta1 * t` |
| TODO 2: AdamW | +独立的一行 `param -= lr * wd * param` | 把权重衰减写到梯度里（那就变成 Adam+L2 了） |
| TODO 3: Warmup | $lr = target * step / warmup\_steps$ | 忘记在 warmup 结束后固定 lr |
| TODO 4: 诊断 | 观察梯度范数增长趋势 | 太快下结论，不分析数据规律 |

## 完整代码

<<< @/snippets/s09_adam_deep_dive/exercise.py
