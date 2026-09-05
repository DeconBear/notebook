---
title: "s16 Attention与Transformer — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s16 Attention与Transformer — exercise.py 练习指南

<a href="/notebook/code/applied/nlp/transformer/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写三个 Transformer 核心组件，建立对注意力机制的深刻直觉。完成后你将能够：

1. 独立写出缩放点积注意力公式的完整 PyTorch 实现
2. 创建因果掩码矩阵，理解其在自回归生成中的作用
3. 实现正弦位置编码，理解为什么需要给序列注入位置信息

## 预备知识

- **Self-Attention 公式**：$\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$
- **Q/K/V 的含义**：Query（查询）、Key（键）、Value（值）——来自信息检索的"字典查询"范式
- **因果掩码**：在自回归生成中，位置 $t$ 只能关注 $0, 1, ..., t$，不能看 $t+1, t+2, ...$
- **位置编码的必要性**：自注意力天然对位置不敏感，需要显式注入位置信息

---

## 任务清单

### 练习 1：实现缩放点积注意力

**目标**：补全 `scaled_dot_product_attention()` 函数，这是所有 Transformer 变体的核心。

**核心公式**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**张量形状**：
- `Q`：`(batch, seq_len, d_k)` — Query 矩阵
- `K`：`(batch, seq_len, d_k)` — Key 矩阵
- `V`：`(batch, seq_len, d_v)` — Value 矩阵（d_v 可以不同于 d_k）
- `mask`：`(batch, seq_len, seq_len)` — 布尔掩码，True 的位置需屏蔽

**TODO 步骤**：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    # 步骤 1: 计算 QK^T — 注意力得分矩阵
    # Q: (batch, seq_len, d_k), K^T: (batch, d_k, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))   # (batch, seq_len, seq_len)

    # 步骤 2: 缩放 — 除以 √d_k
    scores = scores / math.sqrt(d_k)

    # 步骤 3: 应用掩码 — 被 mask 的位置设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 步骤 4: Softmax 归一化 — 每行的权重和为 1
    attn_weights = F.softmax(scores, dim=-1)         # (batch, seq_len, seq_len)

    # 步骤 5: 加权求和 Value
    output = torch.matmul(attn_weights, V)            # (batch, seq_len, d_v)
    return output
```

**关键提示**：
- `K.transpose(-2, -1)` 转置 K 的最后两维以计算矩阵乘法，这是最易出错的地方
- 必须除以 `math.sqrt(d_k)`（不是 `math.sqrt(seq_len)` 或 `d_k`），这是 Transformer 能稳定训练的关键细节
- `masked_fill` 将 True 位置设为 $-\infty$（而非 0），因为 softmax 后 $-\infty \to 0$
- `F.softmax(scores, dim=-1)` 沿最后一维（Key 维度）做归一化

**验证方法**：softmax 每一行的权重之和应约为 1.0。

**预期输出**：
```
[练习1] 注意力输出形状: torch.Size([2, 5, 4]) (期望: [2, 5, 4])
        softmax 行求和: 最小值=1.000000, 最大值=1.000000 (期望: 都≈1.0)
```

---

### 练习 2：实现因果掩码（Causal Mask）

**目标**：创建用于自回归生成的上三角布尔掩码矩阵。

**核心概念**：在自回归模型中，$t$ 时刻的输出只能依赖 $t$ 及之前的信息。因果掩码确保位置 $i$ 不能关注位置 $j$（当 $j > i$ 时）。

**掩码矩阵示例（seq_len=3）**：
```
[[False, True,  True],   ← 位置0: 只能看自己
 [False, False, True],   ← 位置1: 能看0和1
 [False, False, False]]  ← 位置2: 能看0,1,2
```

其中 False 表示可以关注，True 表示需要屏蔽（将设为 $-\infty$）。

**TODO 步骤**：

```python
def create_causal_mask(seq_len):
    # 方法：创建全1矩阵，保留上三角部分（对角线以上）
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool),
        diagonal=1                    # 从对角线以上开始保留，对角线本身不保留
    )
    return mask
```

**关键提示**：
- `torch.triu(input, diagonal=k)` 保留第 k 条对角线及以上的元素
- `diagonal=0` 保留对角线及以上；`diagonal=1` 只保留对角线**以上**（不含对角线）
- 这里我们用 `diagonal=1`，使得对角线（位置 $i$ 关注位置 $i$ 本身）不被屏蔽
- 返回 dtype 为 bool，与 `masked_fill` 的语义一致（True=屏蔽）

**验证**：对于 seq_len=4，验证 mask[0,0]=False（能看到自己），mask[0,3]=True（不能看未来），mask[3,0]=False（能看到过去）。

**预期输出**：
```
[练习2] 因果掩码正确性: True (期望: True)
```

---

### 练习 3：实现正弦位置编码

**目标**：实现原始 Transformer 使用的正弦/余弦位置编码。

**核心公式**：
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

其中 $pos$ 是位置索引（0 到 max_len-1），$i$ 是维度索引（0 到 d_model/2-1）。

**TODO 步骤**：

```python
def sinusoidal_position_encoding(max_len, d_model):
    # 步骤 1: 创建位置索引 (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # 步骤 2: 计算分母 div_term
    # 10000^(2i/d_model) = exp(2i * (-log(10000) / d_model))
    # 使用 exp + log 的形式避免数值问题
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    # 步骤 3: 填充偶数维度为 sin，奇数维度为 cos
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数列: sin
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数列: cos
    return pe
```

**关键提示**：
- `position.unsqueeze(1)` 将 (max_len,) 变为 (max_len, 1)，以便与 div_term (d_model/2,) 广播
- `position * div_term` 的结果形状为 (max_len, d_model/2)，广播机制自动复制 div_term 到所有位置
- `pe[:, 0::2]` 取所有行、所有偶数列；`pe[:, 1::2]` 取所有奇数列
- `10000^{2i/d_{\text{model}}}$ 通过 `exp(log)` 变换计算：$\exp(2i \cdot (-\log(10000)/d_{\text{model}}))$

**为什么用不同频率？** 低维度（小 i）对应高频正弦波，区分相邻位置；高维度（大 i）对应低频正弦波，捕捉远距离位置关系。不同频率的组合让模型能在多个尺度上感知位置。

**验证**：
- 位置编码的值域在 [-1, 1]（sin 和 cos 的范围）
- 相邻位置的编码应该有差异（否则位置编码无意义）

**预期输出**：
```
[练习3] 位置编码形状: torch.Size([10, 16]) (期望: [10, 16])
        值域: [-1.000, 1.000] (期望: [-1.0, 1.0])
        相邻位置平均差异: XXXXXX (期望: > 0)
```

---

## 三个练习的关联

| 练习 | 对应 Transformer 组件 | 在架构中的位置 |
|------|---------------------|-------------|
| 练习 1: 缩放点积注意力 | Attention 核心计算 | 每个注意力头的内部 |
| 练习 2: 因果掩码 | Causal Mask | 施加在注意力得分矩阵上 |
| 练习 3: 正弦位置编码 | Position Encoding | 加到词嵌入上，在 Block 之前 |

学习了这三个组件后，组合它们就能构建一个完整的 Transformer Block。`demo.py` 中的 `TransformerEncoderBlock` 就是将练习 1 的注意力（包含练习 2 的因果掩码）与 FFN、LayerNorm、残差连接组合在一起的完整实现。

## 检查要点

运行 `python exercise.py`，确认：
- [ ] 练习 1 输出形状正确，softmax 每行和为 1.0
- [ ] 练习 2 因果掩码矩阵与期望一致（下三角 False，上三角 True）
- [ ] 练习 3 位置编码值域在 [-1, 1]，相邻位置有差异

全部通过后，返回 demo.py 查看完整的 Mini-GPT 实现，观察这些组件如何被整合为一个可训练的文本生成模型。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/nlp/transformer/code/exercise.py`
