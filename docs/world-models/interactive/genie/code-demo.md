---
title: "wm06 Genie — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm06 Genie — demo.py 代码详解

<a href="/notebook/code/world-models/interactive/genie/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/interactive/genie/code
python demo.py
```

## 代码逐段详解

### 第1步：Genie 架构示意图

`plot_genie_architecture()` 手绘训练阶段（视频→分词器→LAM→动态模型）与推理阶段（用户指定潜在动作→逐帧生成）两段流程，输出 `wm06-01-genie.png`。

### 第2步：网格世界数据

```python
def make_transition_dataset(n_transitions, size=6):
    for i in range(n_transitions):
        pos = (np.random.randint(1, size-1), np.random.randint(1, size-1))  # 只用内部格子
        action = np.random.randint(0, 4)
        new_pos = step_grid(pos, action, size)
        ...
```

**为什么只用内部格子？** 边界上存在动作歧义——在第 0 行时"向上"和"原地不动"效果相同，同一个 $(frame_t, frame_{t+1})$ 对应两种真实动作。内部格子保证 4 个动作 ↔ 4 种转移效果一一对应，方便观察"潜在动作发现"是否成功。

**关键点**：`true_actions` 只在最终评估时使用，**从不进入训练循环**。

### 第3步：动作编码器

```python
class ActionEncoder(nn.Module):
    def forward(self, frame_t, frame_tp1):
        x = torch.cat([frame_t.view(B, -1), frame_tp1.view(B, -1)], dim=1)
        return self.net(x)   # (B, latent_dim) 连续潜向量
```

编码器看到的是**帧对**，而不是带标签的动作。它必须从"两帧之间发生了什么变化"中自行提取动作信息——这正是 Genie LAM 的核心设定。

### 第4步：向量量化（VQ）+ 直通估计器

```python
dist = torch.cdist(z.unsqueeze(1), self.codebook.unsqueeze(0)).squeeze(1)
code_idx = dist.argmin(dim=1)
z_q = self.codebook[code_idx]
z_q_st = z + (z_q - z).detach()   # 直通估计器
```

$$
z_q^{\text{st}} = z + \mathrm{sg}[z_q - z]
$$

前向用量化值，反向梯度直接传给 $z$，绕过不可微的 $\arg\min$。另加：

$$
\mathcal{L}_{\text{VQ}} = \|z_q - \mathrm{sg}[z]\|^2 + 0.25\, \|z - \mathrm{sg}[z_q]\|^2
$$

（码本损失 + 承诺损失）

### 第5步：动态模型与总损失

```python
logits = self.dynamics(frame_t, z_q)          # 预测智能体新位置的分布
recon_loss = F.cross_entropy(logits, target_idx)
loss = recon_loss + vq_loss
```

玩具规模下，"预测下一帧"被简化为"预测智能体在哪个格子"的分类问题。真实 Genie 会对每个空间 token 做自回归/掩码生成，但教学要点相同：**训练信号只有重建/预测损失，没有动作标签**。

### 第6步：对齐评估

训练结束后，统计混淆矩阵：每个潜在码 $k$ 最常对应哪个真实动作。若对齐准确率接近 100%，说明模型无监督地"发明"了与真实动作语义一致的离散码表。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| 潜在动作 | 从帧对推断的离散码，解释"发生了什么变化" | `ActionEncoder` + `VectorQuantizer` |
| 直通估计器 | 前向量化、反向恒等，使 VQ 可微 | `z + (z_q - z).detach()` |
| 承诺损失 | 强迫编码器输出靠近码字，防止码字被忽略 | `0.25 * MSE(z, z_q.detach())` |
| 交互式生成 | 推理时直接查码表指定潜在码，驱动世界演化 | `plot_rollout_demo` |

## 完整代码

<<< @/world-models/interactive/genie/code/demo.py
