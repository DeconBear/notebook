---
title: "wm05 JEPA — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm05 JEPA / V-JEPA — demo.py 代码详解

<a href="/notebook/code/world-models/jepa/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/world-models/jepa/code
python demo.py
```

依赖 PyTorch（CPU 即可），训练 300 步的玩具 JEPA，约十几秒完成。

## 代码逐段详解

### 第1步：合成图像数据集 `make_synthetic_images()`

```python
images[i][mask] = val
images[i] += np.random.normal(0, 0.03, size=(size, size)).astype(np.float32)
```

每张图像上画 1~2 个随机形状（圆形/方形），再叠加一点像素级噪声。这模拟了真实图像的核心特性——**存在可预测的语义结构（形状、位置），也存在不可预测的高频细节（噪声）**。JEPA 的核心主张就是：好的表征应该只保留前者、自动过滤后者。

### 第2步：分块 `patchify()`

```python
patches = images.reshape(n, n_h, patch_size, n_w, patch_size)
patches = patches.transpose(0, 1, 3, 2, 4).reshape(n, n_h * n_w, patch_size * patch_size)
```

Vision Transformer / I-JEPA 的标准操作：把图像切成不重叠的小块，每个 patch 拉平成一个向量。`transpose` 把 `patch_size` 两个维度换到一起，`reshape` 最终把每个 patch 拉平。

### 第3步：上下文编码器与预测器

```python
class PatchEncoder(nn.Module):
    def forward(self, patches, patch_idx):
        x = self.proj(patches)
        pos = torch.gather(self.pos_embed.expand(...), 1, patch_idx.unsqueeze(-1).expand(...))
        x = x + pos
        return self.encoder(x)
```

`PatchEncoder` 把 patch 像素向量投影到 embedding 维度，加上（按位置索引 `gather` 出来的）位置编码，再过一层 Transformer 自注意力，让 patch 之间交换信息。`Predictor` 结构类似，但输入是"上下文表征 + 目标位置的共享掩码 token"——**预测器从不直接看到被遮挡 patch 的像素内容**，只能靠上下文推断该位置在语义上"应该"是什么表征。

### 第4步：EMA 更新目标编码器 —— 防止表征坍缩的关键

```python
@torch.no_grad()
def ema_update(target, context, momentum=0.996):
    for p_t, p_c in zip(target.parameters(), context.parameters()):
        p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)
```

目标编码器的参数**永远不通过反向传播更新**，只通过这个 EMA（指数滑动平均）公式缓慢跟随上下文编码器。这是 JEPA/BYOL/DINO 一类自监督方法防止"表征坍缩"（模型学到输出常数向量也能让损失最小）的关键设计——目标始终"移动"，但移动得足够慢，为预测任务提供一个稳定又不断演化的靶子。

### 第5步：训练循环 —— 组装一次完整的 JEPA 前向计算

```python
ctx_repr = ctx_encoder(ctx_patches, context_idx)
with torch.no_grad():
    full_repr = tgt_encoder(batch_patches, full_idx)
    tgt_repr = torch.gather(full_repr, 1, target_idx.unsqueeze(-1).expand(-1, -1, embed_dim))
pred_repr = predictor(ctx_repr, context_idx, target_idx)
loss = F.mse_loss(pred_repr, tgt_repr)
```

**易错点**：目标编码器的前向计算必须包在 `torch.no_grad()` 内——它只提供回归目标，不参与梯度更新（梯度更新完全通过 EMA 完成）。损失是预测表征与目标表征之间的 **L2 距离，而不是任何像素级的重建误差**——这正是 JEPA 名字的含义。

### 第6步：可视化 —— 掩码示例、损失曲线、逐 patch 预测误差热力图

`plot_prediction_error_map()` 用余弦距离衡量每个 patch 位置的平均预测误差，画成热力图——可以直观看到哪些位置的语义结构更容易/更难预测。

### 关键概念速查表

| 概念 | 一句话解释 | 代码位置 |
|------|-----------|---------|
| patchify | 图像切分为不重叠的 patch 向量 | `patchify()` |
| 上下文/目标编码器 | 前者可训练，后者 EMA 跟随+停梯度 | `PatchEncoder`, `ema_update()` |
| 掩码 token | 预测器输入中代表"待预测位置"的可学习向量 | `Predictor.mask_token` |
| 表征空间损失 | L2 回归损失定义在表征上，不是像素上 | `train_toy_jepa()` |

## 完整代码

<<< @/world-models/jepa/code/demo.py
