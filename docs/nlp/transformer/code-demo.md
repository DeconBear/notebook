---
title: "s16 Attention与Transformer — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s16 Attention与Transformer — demo.py 代码详解

<a href="/notebook/code/nlp/transformer/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nlp/transformer/code
python demo.py
```

**依赖**：`numpy`, `torch`, `matplotlib`, `seaborn`

---

## 代码逐段详解

### 第1步：导入库

```python
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns       # 注意力热力图可视化
```

---

### 第2步：缩放点积注意力 — 一切的基础

**核心公式**（Vaswani et al., 2017）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

这是所有 Transformer 架构的基石。无论 BERT、GPT 还是 ViT，其自注意力计算都是这个公式的某种变体。

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)                                          # 每个头的维度
    scores = torch.matmul(Q, K.transpose(-2, -1))             # QK^T: (..., seq_q, seq_k)
    scores = scores / math.sqrt(d_k)                          # 除以 √d_k 缩放
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))      # 掩码位置设为 -∞
    attn_weights = F.softmax(scores, dim=-1)                  # softmax 归一化
    output = torch.matmul(attn_weights, V)                    # 加权求和
    return output
```

**逐行解释**：

1. **`K.transpose(-2, -1)`**：将 K 的最后两维转置。如果 Q 是 `(..., seq_q, d_k)`，K 转置后为 `(..., d_k, seq_k)`，那么 `Q @ K^T` 的结果是 `(..., seq_q, seq_k)` —— 每个查询位置对所有键位置的得分。

2. **`/ math.sqrt(d_k)`**：缩放因子的核心作用——当 $d_k$ 较大时（如 64 或 128），点积 $q \cdot k = \sum q_i k_i$ 的方差约为 $d_k$。如果不缩放，点积值过大，softmax 会输出几乎 one-hot 的分布（梯度接近 0），模型将无法学习。除以 $\sqrt{d_k}$ 将方差控制在 1，保持 softmax 在梯度良好的区域。

3. **`masked_fill(mask, float('-inf'))`**：掩码位置填入 $-\infty$，经过 softmax 后权重为 0。用在因果掩码（屏蔽未来位置）和 padding 掩码（屏蔽填充 token）。

4. **`F.softmax(scores, dim=-1)`**：沿最后一个维度（即 Key 维度）做 softmax。这保证了对于每个 Query 位置，它对所有 Key 位置的注意力权重之和为 1。

5. **`torch.matmul(attn_weights, V)`**：用注意力权重对 Value 做加权求和。直观上，这是"根据每个位置的关联程度，从 Value 中提取信息"。

---

### 第3步：多头自注意力 — `MultiHeadSelfAttention`

**为什么要多头？** 单头注意力只能捕捉一种关系模式（如句法依赖），但语言中有多种类型的关系需要同时建模——共指关系、语义关联、局部短语结构等。多头注意力通过并行运行 $h$ 组独立的 Q/K/V 投影，让不同头关注不同类型的模式。

**核心公式**：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$，通常 $d_k = d_v = d_{\text{model}} / h$。

#### 3.1 初始化：Q/K/V 投影矩阵

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads          # 每个头的维度

        # Q、K、V 的线性投影（所有头合并在一个矩阵中）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)   # 输出投影
```

**设计选择**：
- `bias=False`：原始 Transformer 中 QKV 投影不带偏置，简化了计算
- `d_k = d_model // num_heads`：总维度 d_model 被均匀分配给 h 个头，保证多头和单头的计算量大致相当
- `W_O`（输出投影）：将所有头的输出拼接后，通过一个线性层融合，学习如何组合不同头的信息

#### 3.2 前向传播：拆分-计算-合并

```python
def forward(self, x, mask=None):
    batch_size, seq_len, _ = x.shape

    # 1. 线性投影：每个 token 的向量分别通过 Q、K、V 投影
    Q = self.W_Q(x)    # (batch, seq_len, d_model)
    K = self.W_K(x)
    V = self.W_V(x)

    # 2. 拆分为多头：reshape → transpose
    #    (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
    Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    # 3. 缩放点积注意力（每个头独立计算）
    attn_output = scaled_dot_product_attention(Q, K, V, mask)
    # (batch, num_heads, seq_len, d_k)

    # 4. 合并多头：transpose → reshape
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, self.d_model
    )
    # 5. 最终输出投影
    output = self.W_O(attn_output)
    return output, attn_weights
```

**`view + transpose` 的奥秘**：这是多头注意力的关键实现技巧。

- `Q.view(batch, seq_len, num_heads, d_k)`：将 d_model=512 的向量"折叠"为 `num_heads=8` 个 `d_k=64` 的小向量。
- `.transpose(1, 2)`：交换 seq_len 和 num_heads 维度，使得后续矩阵乘法在**每个头上独立进行**。
- 合并时：`.transpose(1, 2).contiguous().view(batch, seq_len, d_model)`——将 8 个头的 64 维输出拼接回 512 维。

**为什么这样设计？** 如果不拆分维度而用 for 循环对每个头分别计算，代码更直观但速度慢数十倍。将"多头"信息编码到张量维度中，可以利用 GPU 的批量矩阵乘法一次性完成所有头的计算。

---

### 第4步：Feed-Forward Network（FFN）

Transformer Block 的第二个子层是 Position-wise FFN——对每个位置独立应用同一个两层全连接网络：

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

维度变化：$d_{\text{model}} \to 4 \times d_{\text{model}} \to d_{\text{model}}$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        self.linear1 = nn.Linear(d_model, d_ff)       # 先升维 4x
        self.linear2 = nn.Linear(d_ff, d_model)       # 再降维回去
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))
```

**为什么先升维再降维？** 4x 扩展比提供了更大的容量来存储"知识"。FFN 被比喻为 Transformer 的"知识存储"——注意力负责"查找"相关信息，FFN 负责对查找结果进行非线性变换。升维给了 FFN 足够的表达能力来学习复杂的特征变换。

**GELU vs ReLU**：原始 Transformer 用 ReLU，但 GPT-2 及之后的模型普遍用 GELU（Gaussian Error Linear Unit）。GELU 是平滑版的 ReLU，在零点附近不是硬截断而是平滑过渡，梯度流动更好。

---

### 第5步：Transformer Encoder Block — Pre-LN 风格

```
输入 x
  ↓
LayerNorm(x)           ← Pre-LN: 归一化在注意力之前
  ↓
Multi-Head Self-Attention
  ↓
Dropout → Add(x)       ← 残差连接
  ↓
LayerNorm(...)
  ↓
FFN
  ↓
Dropout → Add(...)     ← 残差连接
  ↓
输出
```

```python
class TransformerEncoderBlock(nn.Module):
    def forward(self, x, mask=None):
        # 子层 1: 自注意力 + 残差
        residual = x
        x_norm = self.norm1(x)                        # Pre-LN
        attn_out, attn_weights = self.self_attn(x_norm, mask)
        x = residual + self.dropout(attn_out)          # 残差连接

        # 子层 2: FFN + 残差
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights
```

**Post-LN vs Pre-LN**：原始 Transformer 论文用的是 Post-LN（LayerNorm 在加法之后），但后来的实践（GPT-2+）发现 Pre-LN（LayerNorm 在子层之前）训练更稳定，梯度流动更顺畅。这是因为 Pre-LN 下残差路径没有经过 LayerNorm，梯度的反向传播路径更"干净"。

**残差连接为什么关键？** 残差连接（$x + \text{Sublayer}(x)$）让梯度可以通过恒等路径直通底层。没有残差连接，训练几十层的 Transformer 几乎不可能——梯度会随着层数消散。有了残差连接，即使注意力或 FFN 部分的梯度消失，恒等路径仍然可以向底层传递梯度信号。

---

### 第6步：正弦位置编码 — `SinusoidalPositionEncoding`

自注意力天然对位置不敏感——打乱输入序列中词的顺序，注意力输出只会在顺序上不同，但值的集合完全相同。位置编码将位置信息注入序列。

**公式**：
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

```python
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)          # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)    # 偶数维度: sin
        pe[:, 1::2] = torch.cos(position * div_term)    # 奇数维度: cos
        self.register_buffer('pe', pe.unsqueeze(0))      # (1, max_len, d_model)
```

**关键设计分析**：

- **不同频率的正弦波**：$10000^{2i/d_{\text{model}}}$ 产生从 $1$ 到 $10000$ 的波长范围。低维（小 i）对应高频（短波长），高维（大 i）对应低频（长波长）。这让模型能从多个粒度感知位置——高频维度区分相邻位置，低频维度捕捉远距离位置关系。

- **为什么用 sin/cos 而不是可学习嵌入？** 正弦编码可以外推到训练时未见过的序列长度（因为函数是确定性的），且相邻位置的编码具有线性关系：$PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数，有助于模型学习相对位置。

- **`register_buffer`**：将张量注册为 buffer（而非 Parameter），意味着它随模型保存/加载但不参与梯度更新。

---

### 第7步：Mini-GPT — Decoder-only Transformer

```python
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, max_len=128, dropout=0.1):
        # 词嵌入 + 位置编码
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len)

        # 堆叠多个 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终 LayerNorm + 语言模型头
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定（Weight Tying）
        self.lm_head.weight = self.token_embed.weight

        # 缓存因果掩码
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )
```

#### 7.1 因果掩码 — `causal_mask`

因果掩码是一个上三角矩阵（对角线以上为 True/1），用于防止位置 $i$ 关注位置 $j$（当 $j > i$ 时）：

```
对于 seq_len=4:
[[False, True,  True,  True],    ← 位置0只能看到自己
 [False, False, True,  True],    ← 位置1能看到0和1
 [False, False, False, True],    ← 位置2能看到0,1,2
 [False, False, False, False]]   ← 位置3能看到全部
```

`torch.triu(..., diagonal=1)` 保留对角线以上（不含对角线）的元素，对角线及以下置 0。mask 位置为 True 时会被 `masked_fill` 设为 $-\infty$。

#### 7.2 前向传播

```python
def forward(self, x, return_attn=False):
    batch_size, seq_len = x.shape
    # 获取因果掩码，塑形为 (1, 1, seq_len, seq_len) 以便广播到 (batch, num_heads, seq_len, seq_len)
    mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)

    # 词嵌入（缩放）+位置编码
    x_emb = self.token_embed(x) * math.sqrt(self.d_model)
    x_emb = self.pos_encoding(x_emb)

    # 通过所有 Transformer Block
    hidden = x_emb
    for block in self.blocks:
        hidden, attn_weights = block(hidden, mask)

    # 最终 LayerNorm + LM Head
    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)         # (batch, seq_len, vocab_size)
    return logits, attn_maps
```

**`* math.sqrt(self.d_model)`**：这是原始 Transformer 论文中的一个小技巧。嵌入向量的初始方差很小（通常从 $\mathcal{N}(0, 1)$ 初始化），乘以 $\sqrt{d_{\text{model}}}$ 让嵌入的尺度与位置编码的尺度匹配，防止位置编码"淹没"了语义信息。

**权重绑定 `self.lm_head.weight = self.token_embed.weight`**：输入嵌入层的权重和输出投影层的权重共享。这在 GPT 系列中是标准做法——输入时将一个 token 映射为向量，输出时将向量映射回 token 概率分布，共享权重减少了参数量，且两边的语义空间是一致的。

#### 7.3 自回归文本生成

```python
def generate(self, seed_tokens, max_new_tokens=50, temperature=0.8):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 取最后 max_len 个 token 作为输入（序列过长时截断）
            input_seq = torch.tensor([tokens[-128:]], device=device)
            logits, _ = self.forward(input_seq)
            # 取最后一个位置的 logits，除以温度
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
    return tokens
```

**逐 token 生成**：每次生成一个新 token 后，将其追加到序列末尾，然后用更新后的序列预测下一个。这是标准的自回归（autoregressive）生成方式。

**因果注意力的效率**：虽然每次只预测最后一个位置，但前向传播时仍需计算所有位置的自注意力。不过因果掩码确保位置 $i$ 只能看到 $j \le i$ 的位置，这恰好符合自回归的需求。

---

### 第8步：$\sqrt{d_k}$ 缩放实验

代码的最后一部分是一个关键的对照实验：对比不同 $d_k$ 下有无缩放的 softmax 分布差异。

**核心指标**：

| 指标 | 含义 | 理想值 |
|------|------|--------|
| 平均熵 | softmax 分布的均匀程度，熵越高分布越均匀 | 有缩放时熵较高 |
| 最大注意力权重 | 最受关注的位置权重，越高越集中（饱和） | 有缩放时较低 |

**实验发现**：当 $d_k = 256$ 时，无缩放的 softmax 几乎完全饱和（某个位置的权重 $\approx 1$，其余 $\approx 0$），平均最大注意力权重接近 1.0。有缩放后，分布更均匀，多个位置都能获得有意义的注意力权重。

这个实验直接验证了 $\sqrt{d_k}$ 缩放的数学原理：$\text{Var}(q \cdot k) = d_k$，因此标准差 $\sigma = \sqrt{d_k}$，除以 $\sqrt{d_k}$ 将标准差归一化为 1。

---

### 第9步：注意力热力图可视化

代码绘制了两类热力图：
1. **各层平均注意力**（所有头的平均）：展示信息流如何随层加深而变化——浅层倾向于关注局部邻域，深层倾向于关注全局或语义相关的 token。
2. **最后一层的多头对比**：展示不同注意力头关注的不同模式——有的头可能关注相邻词，有的头可能关注语法结构（如主语-谓语）。

---

## 关键概念速查表

| 概念 | 公式/描述 | 作用 |
|------|----------|------|
| Scaled Dot-Product Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ | 全局信息聚合 |
| Q (Query) | $Q = XW^Q$ | "我在找什么？" |
| K (Key) | $K = XW^K$ | "作为信息源，我是什么？" |
| V (Value) | $V = XW^V$ | "如果被选中，我提供什么信息？" |
| 多头注意力 | $\text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$ | 并行捕捉多种关系模式 |
| $\sqrt{d_k}$ 缩放 | 除以 $\sqrt{d_k}$ | 防止 softmax 饱和 |
| 因果掩码 | 上三角 $-\infty$ | 防止看到未来 |
| 位置编码 | $\sin/\cos$ 或可学习嵌入 | 注入位置信息 |
| Pre-LN | LayerNorm 在子层之前 | 训练更稳定 |
| 残差连接 | $x + \text{Sublayer}(x)$ | 梯度直通底层 |
| FFN | $\text{GELU}(xW_1+b_1)W_2+b_2$ | 位置独立非线性变换 |
| 权重绑定 | $\text{token\_embed.weight} = \text{lm\_head.weight}$ | 减少参数，语义一致 |

---

## 完整代码

<<< @/nlp/transformer/code/demo.py
