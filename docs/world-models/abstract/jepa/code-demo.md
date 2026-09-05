---
title: "wm05 JEPA — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm05 JEPA / V-JEPA — demo.py 代码详解

<a href="/notebook/code/world-models/abstract/jepa/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/abstract/jepa/code
python demo.py
```

PyTorch、CPU。先画架构示意，再训 300 步玩具 I-JEPA（合成形状图），最后在倒立摆火柴杆帧上跑 **`TinyActionJEPA`**（动作条件的表征预测）。一两分钟。图包括 `wm05-01-jepa.png`、`jepa_mask_example.png`、`jepa_loss_curve.png`、`jepa_prediction_error_map.png`、`jepa_pendulum_frames.png`。

## 代码逐段详解

### 第1步：合成图 + `patchify`

每张图 1–2 个圆/方，再加 $\sigma=0.03$ 像素噪声——**可预测的是形状，不可预测的是噪声**。

```python
patches = images.reshape(n, n_h, patch_size, n_w, patch_size)
patches = patches.transpose(0, 1, 3, 2, 4).reshape(n, n_h * n_w, patch_size * patch_size)
```

`20\times20`、`patch_size=4` → $5\times5=25$ 个 patch。`transpose(0,1,3,2,4)` 把「每个 patch 的两个空间维」挨到一起再展平。返回 `(patches, n_h, n_w)`。

---

### 第2步：`PatchEncoder` — `gather` 位置编码

```python
x = self.proj(patches)
pos = torch.gather(self.pos_embed.expand(B, -1, -1), 1,
                   patch_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
x = x + pos
return self.encoder(x)
```

- **`pos_embed` 形状 `(1, n_patches, embed_dim)`**：每个格子一个可学习向量。
- **`torch.gather(..., 1, index)`**：按 `patch_idx` 从 25 个位置里抽出当前这 $K$ 个可见（或全部）patch 的编码。`unsqueeze(-1).expand` 把下标扩到与 embedding 同宽。
- 一层 `TransformerEncoder`（`batch_first=True`）：patch 之间做自注意力。上下文编码器**只看见可见 patch**。

`Predictor` 把「上下文表征」和「掩码 token + 目标位置编码」拼成一条序列，Transformer 后再 `head`，**只取最后 $K_{\mathrm{tgt}}$ 个输出**（`out[:, -K_tgt:, :]`）。`ctx_tokens = context_repr + ctx_pos * 0.0`：源码写明上下文已含位置，这里乘 0 只是对齐形状。

---

### 第3步：EMA 目标编码器

$$
\theta_{\mathrm{tgt}}\leftarrow m\,\theta_{\mathrm{tgt}}+(1-m)\theta_{\mathrm{ctx}},\quad m=0.996
$$

```python
@torch.no_grad()
def ema_update(target, context, momentum=0.996):
    for p_t, p_c in zip(target.parameters(), context.parameters()):
        p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)
```

- **`mul_` / `add_`**：原地改 `.data`。目标参数 `requires_grad_(False)`，从不 `backward`。
- 初始 `tgt_encoder.load_state_dict(ctx_encoder.state_dict())`：两边从同一点出发。
- 靶子慢慢跟着学生走，但始终滞后——BYOL/JEPA 防表征坍缩的标准手段。

---

### 第4步：一次 JEPA 前向

`sample_mask`：在 25 个下标里切一段连续区间当目标（`mask_ratio=0.4`），其余当上下文。真实 I-JEPA 用多个矩形块；这里是「一段索引」。

```python
ctx_repr = ctx_encoder(ctx_patches, context_idx)
with torch.no_grad():
    full_repr = tgt_encoder(batch_patches, full_idx)
    tgt_repr = torch.gather(full_repr, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
pred_repr = predictor(ctx_repr, context_idx, target_idx)
loss = F.mse_loss(pred_repr, tgt_repr)
```

目标编码器看**整图**，再 `gather` 出被遮挡位置的表征当回归靶。损失是表征 L2，**不是像素重建**。只把 `ctx_encoder` 和 `predictor` 放进 Adam。

热力图用 `1 - cosine_similarity`，不是训练时的 MSE，读图时注意量纲不同。

---

### 第5步：倒立摆火柴杆 — `TinyActionJEPA`

这是文件后半段独立的一条线，不是上面那个 Transformer。

`render_stick(theta)`：从中心沿角度画一条折线像素，再加噪声。`_wrap_pi` 把角折回 $(-\pi,\pi]$。

`collect_pendulum_frames`：随机力矩 $a\in[-1,1]$，用

$$
\ddot\theta=\frac{3g}{2L}\sin\theta+\frac{3}{m L^2}u
$$

欧拉积分，得到 `(帧_t, 动作, 帧_{t+1})`。$|\theta|>\pi/2$ 时重置，避免倒穿。

```python
class TinyActionJEPA(nn.Module):
    def __init__(self, img_dim=400, z_dim=16):
        self.enc = nn.Sequential(nn.Linear(img_dim, 64), nn.ELU(), nn.Linear(64, z_dim))
        self.pred = nn.Sequential(nn.Linear(z_dim + 1, 32), nn.ELU(), nn.Linear(32, z_dim))
    def predict(self, z, a):
        return self.pred(torch.cat([z, a], dim=-1))
```

- **`img_dim=400`**：$20\times20$ 展平。没有 patch、没有 Transformer。
- **预测器吃 `z` 和标量动作**（`z_dim+1`），对应 V-JEPA 2 里「动作条件」的最小版。
- 目标侧只用 **`tgt_enc = TinyActionJEPA(...).enc`**，一份独立的编码器权重，EMA 系数改成 `0.99` / `0.01`（与图像 JEPA 的 0.996 不同）。

```python
z = model.encode(xt[idx])
with torch.no_grad():
    z_tgt = tgt_enc(xn[idx])
z_hat = model.predict(z, at[idx])
loss_lat = F.mse_loss(z_hat, z_tgt)
loss_pix = F.mse_loss(xn[idx], xt[idx])   # 对照：把当前帧当下一帧
```

`loss_pix` **不反传**，只记录「复制当前像素当预测」有多差。表征 MSE 应对准下一帧语义；像素复制 MSE 几乎是常数（噪声+杆动了）。右图两条曲线对照这一点。

---

### 关键概念速查表

| 概念 | 数学 / 直觉 | 代码 |
|------|-------------|------|
| patchify | 切块展平 | `transpose` + `reshape` |
| `gather` | 按索引取位置编码 | `PatchEncoder.forward` |
| 掩码 token | 只告诉「预测哪」 | `Predictor.mask_token` |
| EMA | $\theta_t\leftarrow m\theta_t+(1-m)\theta_c$ | `ema_update` |
| 表征损失 | $\|\hat z-z_{\mathrm{tgt}}\|^2$ | `F.mse_loss(pred_repr, tgt_repr)` |
| `TinyActionJEPA` | 帧→$z$，$(z,a)\to\hat z'$ | 倒立摆段 |
| `loss_pix` | 像素复制对照，不训练 | `mse(xn, xt)` |
| `_wrap_pi` | 角度折回 | `(angle+π)%2π - π` |

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/jepa/code/demo.py`
