---
title: "s15 序列模型 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s15 序列模型 — exercise.py 练习指南

<a href="/notebook/code/applied/nlp/sequence-models/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写 RNN、LSTM、GRU 核心组件的代码，建立对循环神经网络前向传播和门控机制的深刻直觉。完成三个练习后，你将能够：

1. 独立写出 RNN 的时间步循环前向传播
2. 理解 LSTM 遗忘门在梯度传播中的作用
3. 掌握 GRU 更新门的线性插值机制

## 预备知识

在开始练习前，请确保你已经理解以下概念（详见 index.md 和 demo.py 详解）：

- **RNN 公式**：$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
- **BPTT**：梯度沿时间反向传播，连乘导致梯度消失
- **sigmoid 函数**：$\sigma(x) = \frac{1}{1+e^{-x}}$，输出范围 $(0, 1)$
- **LSTM 三个门**：遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$
- **GRU 双门**：重置门 $r_t$、更新门 $z_t$

## 任务清单

### 练习 1：实现 RNN 前向传播（手动循环时间步）

**目标**：补全 `rnn_forward()` 函数，实现 RNN 的逐时间步前向传播。

**核心公式**：
$$
h_t = \tanh(x_t W_{ih}^\top + h_{t-1} W_{hh}^\top + b)
$$

**输入张量形状**：
- `x`：`(batch, seq_len, input_size)` — 输入序列
- `W_ih`：`(hidden_size, input_size)` — 输入→隐藏权重
- `W_hh`：`(hidden_size, hidden_size)` — 隐藏→隐藏权重
- `b`：`(hidden_size,)` — 偏置
- `h0`：`(batch, hidden_size)` — 初始隐藏状态，默认全零

**TODO 步骤**：
```python
for t in range(seq_len):
    x_t = x[:, t, :]                      # 取出第 t 步的输入，形状 (batch, input_size)
    h = torch.tanh(
        x_t @ W_ih.T                        # (batch, input_size) @ (input_size, hidden_size) → (batch, hidden_size)
        + h @ W_hh.T                        # (batch, hidden_size) @ (hidden_size, hidden_size) → (batch, hidden_size)
        + b                                 # (hidden_size,) 广播到 (batch, hidden_size)
    )
    all_h.append(h)                       # 保存当前隐藏状态
```

**关键提示**：
- `@` 是 PyTorch 矩阵乘法运算符（等价于 `torch.matmul`）
- `W_ih` 的形状是 `(hidden_size, input_size)`，需要用 `.T` 转置后与输入相乘
- `b` 的维度是 `(hidden_size,)`，会被 PyTorch 自动广播到 `(batch, hidden_size)`
- 最后用 `torch.stack(all_h, dim=1)` 将所有时间步的隐藏状态堆叠起来

**预期输出**：
```
[练习1] RNN 前向传播输出形状: (2, 5, 3) (期望: (2, 5, 3))
```

---

### 练习 2：实现 LSTM 遗忘门的计算

**目标**：补全 `lstm_forget_gate()` 函数，实现遗忘门的计算。

**核心公式**：
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

其中 $[h_{t-1}, x_t]$ 表示在 dim=1（特征维度）上拼接，$\sigma$ 是 sigmoid 函数。

**TODO 步骤**：
```python
# 1. 在 dim=1 上拼接 h_prev 和 x_t
combined = torch.cat([h_prev, x_t], dim=1)   # (batch, hidden_size + input_size)

# 2. 线性变换：W_f @ combined^T → (batch, hidden_size)
#    使用 F.linear(combined, W_f, b_f) 或 manual matmul
gate = F.linear(combined, W_f, b_f)           # (batch, hidden_size)

# 3. sigmoid 激活
f_t = torch.sigmoid(gate)                     # (batch, hidden_size)，值域 [0,1]
```

**关键提示**：
- `torch.cat([h_prev, x_t], dim=1)` 在特征维度拼接
- `F.linear(input, weight, bias)` 等价于 `input @ weight.T + bias`
- `torch.sigmoid()` 将任意实数映射到 (0, 1)，值越大表示"越不想遗忘"
- $f_t \approx 0$：遗忘该维度的信息；$f_t \approx 1$：保留该维度的信息

**理解遗忘门的直觉**：在读取一句长文本时，读到句号后可能需要"遗忘"前面句子的部分细节；读到新的主语时需要"遗忘"前一个主语的信息。

**预期输出**：
```
[练习2] 遗忘门输出形状: torch.Size([4, 8]), 范围在[0,1]: True (期望: True)
```

---

### 练习 3：实现 GRU 更新门

**目标**：补全 `gru_update_gate()` 函数，实现 GRU 更新门的计算。

**核心公式**：
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

更新门的作用是控制 $h_t$ 在多大程度上保留旧状态 $h_{t-1}$、在多大程度上采用新候选 $\tilde{h}_t$：

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**TODO 步骤**（与练习 2 的遗忘门实现几乎相同）：
```python
combined = torch.cat([h_prev, x_t], dim=1)
z_t = torch.sigmoid(F.linear(combined, W_z, b_z))
return z_t
```

**关键提示**：
- GRU 的更新门 $z_t$ 同时做了 LSTM 中遗忘门和输入门的工作
- $z_t \to 0$：$h_t \approx h_{t-1}$（保留历史，不更新）
- $z_t \to 1$：$h_t \approx \tilde{h}_t$（用新信息替换）
- **GRU 比 LSTM 少一个门**，参数更少，训练更快

**预期输出**：
```
[练习3] 更新门输出形状: torch.Size([4, 8]), 范围在[0,1]: True (期望: True)
```

---

## 三个模型的参数量对比

完成练习后，可以计算一下三种模型的参数量差异：

| 模型 | 线性变换数 | 参数量公式 | 门控机制 |
|------|----------|-----------|---------|
| RNN | 2 | $d_h(d_h + d_x)$ | 无（一个 tanh） |
| LSTM | 4（合并为1个大矩阵） | $4d_h(d_h + d_x)$ | 遗忘+输入+输出 |
| GRU | 2（门合并为1，候选独立） | $3d_h(d_h + d_x)$ | 重置+更新 |

## 检查要点

完成所有 TODO 后，运行 `python exercise.py`，确认：
- [ ] 练习 1 输出形状为 `(2, 5, 3)`
- [ ] 练习 2 遗忘门值域在 [0, 1] 内
- [ ] 练习 3 更新门值域在 [0, 1] 内

全部通过后，返回 demo.py 对照参考实现，理解每个细胞完整的 forward 逻辑。

## 完整代码

<<< @/applied/nlp/sequence-models/code/exercise.py
