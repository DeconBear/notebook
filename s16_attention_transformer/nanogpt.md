---
title: "s16 — nanoGPT: 从零训练 GPT-2"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# nanoGPT: 从零训练一个真正的 GPT

基于 [Andrej Karpathy 的 nanoGPT](https://github.com/karpathy/nanoGPT)，从零实现完整的 GPT-2 架构，在莎士比亚文本上训练并生成文本。

<a href="../code/s16_attention_transformer/nanogpt.py" target="_blank" download>Download nanogpt.py</a>

## 运行方式

```bash
cd s16_attention_transformer/code
python nanogpt.py              # CPU 训练（~15 分钟）
python nanogpt.py --gpu        # GPU 训练（如有）
python nanogpt.py --generate   # 仅加载已有模型生成文本
```

**无需下载外部数据**：代码内置了莎士比亚剧本片段作为训练数据。如果想用自己的数据，只需在 `code/` 目录下放置一个 `input.txt` 文件。

**设备自适应**：
- CPU 模式：batch_size=16, 500 次迭代，约 15 分钟
- GPU 模式：batch_size=64, 5000 次迭代
- 训练过程中自动保存最佳模型为 `nanogpt_model.pt`

---

## 架构详解

nanoGPT 实现了完整的 GPT-2 风格 Transformer，包含以下核心组件：

### 1. Q/K/V 的合并投影

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # 将 Q、K、V 三个投影合并为一个大矩阵
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        B, T, C = x.size()                              # batch, seq_len, embed_dim
        qkv = self.c_attn(x)                            # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)         # 拆分为 Q, K, V

        # 拆分为多头: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)                              # 输出投影
        return y
```

**`self.c_attn` 的合并设计**：将 Q/K/V 三个投影矩阵合并为一个 `(C, 3C)` 的大矩阵，一次 `nn.Linear` 调用同时输出 Q、K、V。这在 GPU 上更高效——只需要一次数据加载和一次大矩阵乘法，而非三次小矩阵乘法。

**`qkv.split(self.n_embd, dim=2)`**：将 `(B, T, 3C)` 沿 dim=2 均分为 3 个 `(B, T, C)` 张量，分别对应 Q、K、V。

**因果掩码存储为 buffer**：

```python
self.register_buffer("bias",
    torch.tril(torch.ones(config.block_size, config.block_size))
          .view(1, 1, config.block_size, config.block_size))
```

因为因果掩码在整个训练过程中不变，预先计算并注册为 buffer 避免了每次 forward 都重新创建。

### 2. GELU 激活的 MLP

```python
class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)        # 升维 4x
        x = self.gelu(x)        # GELU 激活
        x = self.c_proj(x)      # 降维回原维度
        return x
```

**GELU vs ReLU**：GPT-2 使用 GELU（Gaussian Error Linear Unit）而非 ReLU。GELU 是平滑的、非单调的激活函数：
- 在正值区域行为类似 ReLU
- 在负值区域不完全截断为零（允许少量负值通过）
- 零点附近的平滑过渡让梯度流动更顺畅

**4x 扩展比**：`d_ff = 4 * d_model` 是 Transformer 的标准配置。这个 "bottleneck" 结构提供了足够的容量来存储 FFN 层的"知识"——约 2/3 的 Transformer 参数都在 FFN 层中。

### 3. Pre-LN Transformer Block

```python
class Block(nn.Module):
    def __init__(self, config):
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # Pre-LN: LN → Attention → Add
        x = x + self.mlp(self.ln_2(x))      # Pre-LN: LN → MLP → Add
        return x
```

**Pre-LN 的好处**：LayerNorm 放在子层**之前**（而非之后），使得残差连接中只有"干净的"子层输出加回恒等路径。这样做的好处是训练更稳定——梯度在反向传播时，残差连接提供的恒等路径不受 LayerNorm 影响。

### 4. Token + Position 嵌入

```python
class GPT(nn.Module):
    def __init__(self, config):
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),   # Position Embedding
            ...
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight Tying: 输入嵌入和输出投影共享权重
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # Token 嵌入 + 位置嵌入 → 元素级相加
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)          # (B, T, C)
        pos_emb = self.transformer.wpe(pos)          # (T, C) → 广播到 (B, T, C)
        x = tok_emb + pos_emb

        # 通过所有 Transformer Blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)                 # 最终 LayerNorm

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])     # 推理时只取最后一个位置
            loss = None
        return logits, loss
```

**可学习位置嵌入 vs 正弦位置编码**：nanoGPT 使用可学习的位置嵌入（`nn.Embedding(block_size, n_embd)`），而非 demo.py 中使用的正弦位置编码。GPT 系列模型都使用可学习位置嵌入——虽然不能像正弦编码那样外推到更长序列，但在训练长度范围内的效果通常更好。

**Weight Tying（权重绑定）**：`self.transformer.wte.weight = self.lm_head.weight`——输入嵌入矩阵和输出投影矩阵共享权重。这是 Transformer 语言模型的标准做法：
- 减少了参数量（vocab_size * n_embd 个参数）
- 两个矩阵的语义空间一致——输入时映射 token→向量，输出时映射向量→token 概率

**推理优化**：`if targets is not None` 分支在训练时对所有位置计算 logits（因为需要计算每个位置的 loss）；推理时只取 `x[:, [-1], :]`（最后一个位置的隐藏状态）做预测，减少计算量。

### 5. 自回归文本生成

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 如果序列太长，截取最后 block_size 个 token
        idx_cond = idx if idx.size(1) <= self.config.block_size \
                   else idx[:, -self.config.block_size:]

        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature   # 温度缩放

        # Top-k 采样：只从概率最高的 k 个 token 中选
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Top-k 采样**：只保留概率最高的 k 个 token，将其余 token 的 logits 设为 $-\infty$（概率为 0），然后从这 k 个候选 token 中按概率采样。这可以防止模型采样到极低概率的"垃圾"token，提高生成质量。top_k=40 是 GPT-2 论文中的默认值。

### 6. 参数分组与 AdamW

```python
def configure_optimizers(self, learning_rate, weight_decay, device_type):
    decay_params = []
    no_decay_params = []
    for name, param in self.named_parameters():
        if param.requires_grad:
            # 只有 2D 及以上（权重矩阵）的参数应用 weight decay
            if len(param.shape) >= 2 and 'ln' not in name and 'bias' not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)    # bias 和 LayerNorm 参数不衰减

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                  betas=(0.9, 0.95))
```

**为什么 bias 和 LayerNorm 不用 weight decay？** Weight decay 的作用是正则化（防止过拟合），但 bias 和 LayerNorm 参数通常维度很低（1D），过拟合风险小。对所有参数应用 weight decay 会抑制这些参数的学习，降低模型性能。GPT-2/3 论文中都采用了这种分组的做法。

**AdamW vs Adam**：AdamW 将 weight decay 与梯度更新解耦——传统的 Adam 将 L2 正则化隐含在梯度中，导致自适应学习率与正则化相互作用；AdamW 直接对权重做衰减，效果更好。

### 7. 学习率调度：预热 + 余弦衰减

```python
if step < max_iters * 0.1:
    lr = learning_rate * (step / (max_iters * 0.1))       # 线性预热
else:
    progress = (step - max_iters * 0.1) / (max_iters * 0.9)
    lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))  # 余弦衰减
```

**预热（Warmup）**：训练前 10% 的步数中，学习率从 0 线性增长到目标值。Transformer 训练初期梯度方差大，直接从高学习率开始容易导致训练不稳定。预热让模型先用小步更新，等参数稳定后再加大步长。

**余弦衰减**：学习率按照余弦函数从目标值衰减到接近 0。余弦衰减比线性衰减更平滑，能在训练后期保持有效的学习率。

---

## nanoGPT vs demo.py 对比

| 维度 | demo.py | nanoGPT |
|------|---------|---------|
| 模型规模 | ~50k-80k 参数 | ~15M 参数（可配置） |
| 训练数据 | 合成短文本（中文） | 完整莎士比亚剧本（英文） |
| 训练时间 | ~30 秒 | ~15 分钟（CPU） |
| 生成质量 | 字符级乱码 | 有意义的仿莎士比亚文本 |
| 架构完整性 | 简化版 | 完整 GPT-2 |
| Position Encoding | 正弦编码 | 可学习位置嵌入 |
| 激活函数 | 用于 demo 的 EncoderBlock | GELU |
| Weight Tying | 有（token_embed = lm_head） | 有 |
| 学习率调度 | 恒定 lr | 预热 + 余弦衰减 |
| 参数分组 | 无 | 区分 weight decay / no decay |
| 适用场景 | 理解 Transformer 原理 | 体验真正的 GPT 训练 |

---

## 关键概念速查表

| 概念 | 在 NanoGPT 中的实现 | 作用 |
|------|-------------------|------|
| 因果掩码 | `torch.tril(torch.ones(...))` 注册为 buffer | 防止看到未来 token |
| Pre-LN | `x = x + attn(ln_1(x))` | 训练稳定性，梯度路径更干净 |
| Weight Tying | `wte.weight = lm_head.weight` | 减少参数，语义空间一致 |
| 参数分组 | bias/LN 不用 weight decay | 提升性能 |
| AdamW | `optim.AdamW` with `betas=(0.9, 0.95)` | 解耦 weight decay 和自适应学习率 |
| 学习率预热 | 前 10% 步数线性增长 lr | 训练初期稳定性 |
| 余弦衰减 | `lr = 0.5 * lr * (1 + cos(pi * progress))` | 平滑衰减到 0 |
| Top-k 采样 | 只从概率最高的 k 个 token 采样 | 提升生成质量 |
| 温度 | `logits / temperature` | 控制生成随机性 |
| GELU | `nn.GELU()` | 平滑版 ReLU，梯度更好 |
| QKV 合并 | `nn.Linear(C, 3*C)` | 一次矩阵乘法计算 Q,K,V |

## 完整代码

```python
<<< @/snippets/s16_attention_transformer/nanogpt.py
```
