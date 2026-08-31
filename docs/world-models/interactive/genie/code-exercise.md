---
title: "wm06 Genie — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm06 Genie — exercise.py 练习指南

<a href="/notebook/code/world-models/interactive/genie/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO，理解 Genie 潜在动作模型的核心实现：
1. 实现网格世界一步转移 `step_grid()`
2. 实现向量量化层 `VectorQuantizer.forward()`（含直通估计器）
3. 实现下一帧预测总损失 `compute_next_frame_loss()`

## 预备知识

- 潜在动作模型的训练信号只有"预测下一帧"，不使用动作标签
- VQ 的 $\arg\min$ 不可微，需用直通估计器：$z_q^{\text{st}} = z + \mathrm{sg}[z_q - z]$
- 码本损失让码字靠近编码器输出，承诺损失让编码器输出靠近码字

## 任务清单

### TODO 1：`step_grid`

动作编码：0=上(行-1), 1=下(行+1), 2=左(列-1), 3=右(列+1)。越界用 `max`/`min` 裁剪到 $[0, size-1]$。

```python
r, c = pos
if action == 0:   r = max(0, r - 1)
elif action == 1: r = min(size - 1, r + 1)
elif action == 2: c = max(0, c - 1)
elif action == 3: c = min(size - 1, c + 1)
return (r, c)
```

### TODO 2：`VectorQuantizer.forward`

```python
dist = torch.cdist(z.unsqueeze(1), self.codebook.unsqueeze(0)).squeeze(1)
code_idx = dist.argmin(dim=1)
z_q = self.codebook[code_idx]
vq_loss = F.mse_loss(z_q, z.detach()) + 0.25 * F.mse_loss(z, z_q.detach())
z_q_st = z + (z_q - z).detach()
return z_q_st, code_idx, vq_loss
```

**易错点**：忘记 `.detach()` 会导致码本损失/承诺损失方向错误，或直通估计器失效。

### TODO 3：`compute_next_frame_loss`

```python
z = action_encoder(frame_t, frame_tp1)
z_q, code_idx, vq_loss = vq(z)
logits = dynamics(frame_t, z_q)
target_idx = frame_tp1.view(B, -1).argmax(dim=1)
recon_loss = F.cross_entropy(logits, target_idx)
total_loss = recon_loss + vq_loss
```

## 完成后的验证

运行 `python code/exercise.py`：
1. TODO1 自检全部通过
2. 训练后下一帧预测准确率应明显上升（>60%）
3. 量化层应使用到多个不同码字（不是坍缩到 1 个）

## 完整代码

<<< @/world-models/interactive/genie/code/exercise.py
