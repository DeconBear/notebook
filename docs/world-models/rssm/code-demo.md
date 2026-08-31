---
title: "wm02 经典起源与 RSSM — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm02 经典起源与 RSSM — demo.py 代码详解

<a href="/notebook/code/world-models/rssm/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/rssm/code
python demo.py
```

## 代码逐段详解

### 第1步：玩具环境 `ToyPointEnv`

环境是一个 2D 质点，由 PD 控制器驱动去追踪旋转目标点：

$$
\text{target}(t) = r\cdot[\cos(\omega t+\phi),\ \sin(\omega t+\phi)]
$$

$$
a_t = k_p(\text{target}-\text{pos}) + k_d(\text{target\_vel}-\text{vel})
$$

模型只能看到带噪声的位置 $o_t = \text{pos}_t + \text{noise}$，看不到真实速度和圆周参数。每条轨迹的 $(r,\omega,\phi)$ 都随机采样，逼迫模型真正利用动作序列区分不同动力学。

### 第2步：RSSM 的四个组件

```python
self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)   # h_t
self.prior_net = ...      # h_t → N(μ_prior, σ_prior)
self.posterior_net = ...  # (h_t, o_t) → N(μ_post, σ_post)
self.decoder = ...        # (h_t, s_t) → ô_t
```

**为什么要拆成先验和后验两个网络**：
- 训练时用后验（看到了观测，估计更准）采样 $s_t$，提供高质量监督
- 想象时用先验（看不到观测）采样，模拟真实部署场景
- 两者通过 KL 散度对齐，让先验逐渐学会"闭眼猜准"

### 第3步：重参数化与 KL

```python
z = mean + std * torch.randn_like(mean)   # 重参数化：梯度可传到 mean/std
```

对角高斯 KL 有解析解（见 index.md 第三节），代码里对 `stoch_dim` 求和后返回每个样本一个标量。

### 第4步：训练循环中的 free-nats

```python
kl_penalty = torch.clamp(kl_loss, min=free_nats)
loss = recon_loss + kl_beta * kl_penalty
```

`free_nats` 允许模型"免费"使用最多这么多信息量的随机状态；只有超出部分才计入损失。这是防止 posterior collapse 的经典技巧。

### 第5步：`imagine()` —— 在潜空间做梦

1. **热启动**：用前 $C$ 步真实观测的后验更新 $(h,s)$
2. **想象**：之后完全脱离观测，只用先验均值 + 未来动作序列连续 rollout $K$ 步
3. 每步解码得到预测观测，用于与真实轨迹对比

### 第6步：开环 vs 闭环误差对比

- **开环想象**：只用先验 → 误差随步数增长
- **闭环滤波**：每步用后验 → 误差几乎平坦

这直接验证了 wm01 的核心直觉：潜空间做梦可行，但长视野仍需注意误差累积（Dreamer 通常用 15 步左右的想象视野）。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 确定性状态 $h_t$ | GRU 承载的长期记忆 | `self.gru` |
| 先验 / 后验 | 不看 / 看观测的 $s_t$ 分布 | `prior()` / `posterior()` |
| 重参数化 | $z=\mu+\sigma\varepsilon$，保持可微 | `reparameterize()` |
| Free-nats | KL 下限裁剪，防后验坍缩 | `torch.clamp(kl, min=...)` |
| 想象 rollout | 热启动后纯先验多步预测 | `imagine()` |

## 完整代码

<<< @/world-models/rssm/code/demo.py
