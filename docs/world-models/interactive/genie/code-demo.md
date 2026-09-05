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

PyTorch、CPU。6×6 网格上无监督发现 4 个离散潜在动作。真实动作标签**只用于最后评估**。图：`wm06-01-genie.png`、`genie_training_curves.png`、`genie_latent_action_alignment.png`、`genie_rollout_demo.png`。

## 代码逐段详解

### 第1步：内部格子，避免动作歧义

```python
pos = (np.random.randint(1, size - 1), np.random.randint(1, size - 1))
```

`randint(1, size-1)` 是半开区间，得到 $1\ldots 4$。贴边时「往墙上走」和「原地」同一对帧，一个 $(x_t,x_{t+1})$ 会对应两种真动作，LAM 无法辨识。内部四向位移一一对应。`true_actions` 写入数据集，**从不进 `forward` / `loss`**。

`make_grid_frame`：one-hot 平面，智能体格为 1。`step_grid`：0 上 1 下 2 左 3 右，越界 `max`/`min` 贴墙。

---

### 第2步：`ActionEncoder` — 只看帧对

```python
x = torch.cat([frame_t.view(B, -1), frame_tp1.view(B, -1)], dim=1)
return self.net(x)   # (B, latent_dim)
```

`view(B,-1)` 把 $6\times6$ 拉成 36 维，两帧拼接 72 维，MLP 出 8 维连续 $z$。没有动作整数输入。差异里应藏着「往哪走」。

---

### 第3步：VQ + 直通 — 前向量化、反向当恒等

码表 `codebook` 是 `(K, d)` 的 `nn.Parameter`，$K=4$。

$$
k=\arg\min_j\|z-e_j\|_2,\quad z_q=e_k
$$

```python
dist = torch.cdist(z.unsqueeze(1), self.codebook.unsqueeze(0)).squeeze(1)
code_idx = dist.argmin(dim=1)
z_q = self.codebook[code_idx]
```

- **`z.unsqueeze(1)`**：`(B,d)→(B,1,d)`；`codebook.unsqueeze(0)`：`(K,d)→(1,K,d)`。`cdist` 得到 `(B,K)` 两两欧氏距离。
- **`argmin(dim=1)`**：每行最近码。这一步 **不可微**。

直通估计器：

$$
z_q^{\mathrm{st}}=z+\mathrm{sg}[z_q-z]
$$

前向值等于 $z_q$（因为 $z+(z_q-z)=z_q$），反向 $\partial z_q^{\mathrm{st}}/\partial z=I$，梯度绕过 $\arg\min$ 传到编码器。

```python
z_q_st = z + (z_q - z).detach()
```

`.detach()` 就是 stop-gradient。解码器吃的是 `z_q_st`，不是裸 `z_q`（否则编码器收不到重建梯度）。

VQ 损失（码本 + 承诺）：

$$
\mathcal{L}_{\mathrm{VQ}}=\|z_q-\mathrm{sg}[z]\|^2+0.25\,\|z-\mathrm{sg}[z_q]\|^2
$$

```python
codebook_loss = F.mse_loss(z_q, z.detach())      # 码字去追编码器
commitment_loss = F.mse_loss(z, z_q.detach())    # 编码器去承诺靠近码字
vq_loss = codebook_loss + 0.25 * commitment_loss
```

- **`z.detach()` 进码本损失**：更新码字时当 $z$ 为常数。
- **`z_q.detach()` 进承诺损失**：更新编码器时当码字为常数。两边不要抢着对同一条边求梯度，否则会抖。
- **`0.25`**：承诺项较弱，避免编码器被码表拖死、重建学不好。

---

### 第4步：动态模型是「下一格分类」，不是像素解码

```python
logits = self.dynamics(frame_t, z_q)          # (B, 36)
target_idx = frames_tp1.reshape(n, -1).argmax(axis=1)
recon_loss = F.cross_entropy(logits, target_idx)
loss = recon_loss + vq_loss
```

one-hot 帧的 `argmax` 就是智能体格子编号。玩具里「预测下一帧」= 36 类分类。真实 Genie 对空间 token 做自回归；教学要点相同：**唯一监督是预测准不准，没有动作交叉熵**。

`LatentActionWorldModel.forward`：`encoder → vq → dynamics`，返回 `logits, code_idx, vq_loss`。

---

### 第5步：对齐评估与交互式指定码

`evaluate_latent_action_alignment`：混淆矩阵 `confusion[k, a] += 1`，每行 `argmax` 得到「码 $k$ 最常对应的真动作」，再算最优匹配准确率。接近 100% 说明 4 个码自发对齐上/下/左/右（排列可以任意）。

`plot_rollout_demo`：**推理时绕过编码器**，直接 `model.vq.codebook[k]` 当动作向量喂给 `dynamics`。这就是「手柄上的离散键」：用户指定码，世界走一步。蓝格起点、红格预测落点。

`frame_t = torch.from_numpy(frame).float().unsqueeze(0)`：加 batch 维，`Linear` 才接受。`pred_pos_idx // size` 与 `% size` 把 0…35 拆回 `(行, 列)`。语法 `//` 是整数除，不要写成 `/` 再 `int`（负索引时行为不同，这里下标非负）。

训练循环 `np.random.choice(n_data, batch_size, replace=False)`：无放回 mini-batch。`true_actions` 传进 `train_latent_action_model` 的签名但函数体**从未读取**——留下是为了调用处接口整齐，不要误以为交叉熵用了动作标签。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| 帧对编码器 | $(x_t,x_{t+1})\to z$ | `ActionEncoder` |
| `cdist` + `argmin` | 最近码 | `VectorQuantizer` |
| 直通 | $z+\mathrm{sg}[z_q-z]$ | `z + (z_q-z).detach()` |
| 码本损失 | $\|z_q-\mathrm{sg}[z]\|^2$ | `mse(z_q, z.detach())` |
| 承诺损失 | $0.25\|z-\mathrm{sg}[z_q]\|^2$ | `0.25 * mse(z, z_q.detach())` |
| 动态模型 | $x_t,z_q\to$ 下一格 logits | `DynamicsModel` |
| 无动作标签 | 损失里没有 `true_actions` | `train_latent_action_model` |
| 交互 | 直接查码表 | `codebook[k]` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/interactive/genie/code/demo.py`
