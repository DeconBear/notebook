---
title: "s15 序列模型 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s15 序列模型 — demo.py 代码详解

<a href="/notebook/code/nlp/sequence-models/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/nlp/sequence-models/code
python demo.py
```

**依赖**：`numpy`, `torch`, `matplotlib`（均为标准 PyTorch 生态）

**GPU 说明**：代码自动检测 CUDA / MPS / CPU。CPU 模式下减少训练轮数以加快演示速度。

---

## 代码逐段详解

### 第1步：导入库 — 每个库的作用

```python
import numpy as np               # 数值计算
import math                      # 数学函数
from collections import Counter  # 统计字符频率（未直接使用，预留）
import torch                     # 深度学习核心框架
import torch.nn as nn            # 神经网络模块（nn.Linear, nn.Embedding 等）
import torch.nn.functional as F  # 函数式接口（softmax, cross_entropy 等）
import torch.optim as optim      # 优化器（Adam）
from torch.utils.data import Dataset, DataLoader  # 数据加载
import matplotlib.pyplot as plt  # 绘图：训练曲线对比
```

**关键设计**：demo.py 完全从零实现 RNN/LSTM/GRU 细胞，不使用 `torch.nn.RNN` 等内置模块。这样做是为了让你看清门控机制的每一步计算，建立对循环网络的深刻直觉。

---

### 第2步：从零实现 RNN 细胞 — `MyRNNCell`

RNN 的核心公式是同一个细胞在时间上反复调用：

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

其中：
- $h_{t-1} \in \mathbb{R}^{d_h}$：上一时刻的隐藏状态（"记忆"）
- $x_t \in \mathbb{R}^{d_x}$：当前时刻的输入
- $W_h \in \mathbb{R}^{d_h \times d_h}$：隐藏到隐藏的权重（循环连接，是 RNN 的"核心"）
- $W_x \in \mathbb{R}^{d_h \times d_x}$：输入到隐藏的权重
- $\tanh$：激活函数，将值压缩到 $(-1, 1)$ 防止数值爆炸

```python
class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)  # W_x: 输入→隐藏
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)  # W_h: 隐藏→隐藏(含偏置)

    def forward(self, x, h_prev):
        return torch.tanh(self.W_ih(x) + self.W_hh(h_prev))
```

**逐行解释**：
- `nn.Linear(input_size, hidden_size, bias=False)`：创建一个线性层 $y = xW^\top$（PyTorch 的 `nn.Linear` 内部存储的是转置后的权重矩阵）
- `self.W_ih(x)`：计算 $x_t W_{ih}^\top$，将输入从 `input_size` 维映射到 `hidden_size` 维
- `self.W_hh(h_prev)`：计算 $h_{t-1} W_{hh}^\top$，将上一隐藏状态再投影一次
- `torch.tanh(...)`：逐元素应用 tanh，输出范围 $(-1, 1)$

**设计选择**：为什么 `W_ih` 的 `bias=False`？因为偏置已包含在 `W_hh` 中，两个都加偏置会导致冗余。实践中可以根据需要调整。

---

### 第3步：从零实现 LSTM 细胞 — `MyLSTMCell`

LSTM 通过引入**细胞状态** $c_t$ 和三个**门**（遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$）来解决 RNN 的梯度消失问题。

#### 3.1 核心公式回顾

**遗忘门**：
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**输入门**：
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**候选细胞状态**：
$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**细胞状态更新（LSTM 最核心的创新）**：
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**输出门**：
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**隐藏状态输出**：
$$
h_t = o_t \odot \tanh(c_t)
$$

其中 $\sigma$ 是 sigmoid（输出 0~1），$\odot$ 是逐元素乘法。

#### 3.2 代码实现与逐行解释

```python
class MyLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 四个线性变换合并为一个矩阵：输出维度 = 4 * hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x, h_prev, c_prev):
        # 拼接输入和上一隐藏状态
        combined = torch.cat([h_prev, x], dim=1)        # (batch, input_size+hidden_size)
        gates = self.W(combined)                         # (batch, 4*hidden_size)
        f_gate, i_gate, c_tilde, o_gate = gates.chunk(4, dim=1)  # 拆分为4组

        f = torch.sigmoid(f_gate)           # 遗忘门: [0,1]，控制丢弃哪些旧信息
        i = torch.sigmoid(i_gate)           # 输入门: [0,1]，控制写入哪些新信息
        c_tilde = torch.tanh(c_tilde)       # 候选细胞状态: [-1,1]，新信息的候选内容
        c = f * c_prev + i * c_tilde        # 细胞状态更新: 加法路径！
        o = torch.sigmoid(o_gate)           # 输出门: [0,1]，控制暴露哪些信息
        h = o * torch.tanh(c)               # 隐藏状态: 过滤后的细胞状态
        return h, c
```

**关键设计分析**：

1. **四个门合并为一个矩阵**：`W` 的输出维度是 `4 * hidden_size`，一次矩阵乘法同时计算 $f, i, \tilde{c}, o$。这比四个独立的 `nn.Linear` 更高效（只需一次大矩阵乘法和一次内存读取）。

2. **`torch.cat([h_prev, x], dim=1)`**：将上一隐藏状态和当前输入拼接。dim=1 表示在特征维度拼接（batch 维度是 dim=0）。拼接后的向量维度为 `input_size + hidden_size`。

3. **`gates.chunk(4, dim=1)`**：将 `(batch, 4*hidden_size)` 的张量沿 dim=1 均匀切成 4 块，每块 `hidden_size` 维，分别对应 $f, i, \tilde{c}, o$。

4. **`torch.sigmoid()` 用于门**：sigmoid 输出 $(0, 1)$，天然适合做"门控开关"——值为 0 表示完全关闭（信息不通过），值为 1 表示完全打开（信息全部通过）。

5. **`torch.tanh()` 用于候选状态**：tanh 输出 $(-1, 1)$，作为信息的内容编码。注意这里用了"c_tilde 变量覆盖"（先获得门控值，再用 tanh 处理），这是合法的因为 `chunk` 返回的是视图。

6. **$c_t = f \odot c_{t-1} + i \odot \tilde{c}_t$（加法更新）**：这是 LSTM 解决梯度消失的关键。因为 $\frac{\partial c_t}{\partial c_{t-1}} = f_t$（当 $f_t \approx 1$ 时梯度无损传播），梯度可以在时间上"直通"而无需经过 tanh 等非线性压缩。

7. **$h_t = o_t \odot \tanh(c_t)$**：输出门 $o_t$ 决定将细胞状态的哪些部分暴露为隐藏状态。$\tanh(c_t)$ 将细胞状态值压缩到 $(-1, 1)$。

---

### 第4步：从零实现 GRU 细胞 — `MyGRUCell`

GRU（Cho et al., 2014）将 LSTM 的三个门精简为两个，去掉了独立的细胞状态 $c_t$：

**重置门**：
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$

**更新门**：
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$

**候选隐藏状态**：
$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t])
$$

**最终隐藏状态（线性插值）**：
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

```python
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 重置门和更新门的线性变换合并
        self.W_rz = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=True)
        # 候选隐藏状态的线性变换
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

    def forward(self, x, h_prev):
        combined = torch.cat([h_prev, x], dim=1)
        rz = self.W_rz(combined)
        r_gate, z_gate = rz.chunk(2, dim=1)
        r = torch.sigmoid(r_gate)   # 重置门: 控制忽略多少历史信息
        z = torch.sigmoid(z_gate)   # 更新门: 控制保留历史 vs 写入新信息

        # 候选隐藏状态 — 重置门过滤后的历史 + 当前输入
        combined_reset = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # 最终状态 — z 做线性插值（同时做了 LSTM 遗忘门+输入门的工作）
        h = (1 - z) * h_prev + z * h_tilde
        return h
```

**关键设计分析**：

1. **`z_t` 同时做了 LSTM 的遗忘门和输入门**：当 $z_t \to 0$，$h_t \approx h_{t-1}$（保留全部历史）；当 $z_t \to 1$，$h_t \approx \tilde{h}_t$（完全更新）。

2. **`r_t \odot h_{t-1}`**：重置门控制"在计算候选状态时，多少旧信息需要被忽略"。$r_t \to 0$ 表示完全重置，$h_{t-1}$ 的影响被抹去，只依赖 $x_t$。

3. **参数量对比**：GRU 的参数量约为 LSTM 的 3/4（GRU: $3d_h(d_h+d_x)$ vs LSTM: $4d_h(d_h+d_x)$），但效果通常与 LSTM 相当。

---

### 第5步：字符级语言模型 — `CharRNNLM`

语言模型的任务是：给定前文，预测下一个字符。

#### 5.1 数据准备

```python
def build_char_vocab(text):
    chars = sorted(set(text))                    # 去重排序获得所有字符
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)
```

**训练数据格式**：将完整文本切成重叠的 (输入序列, 下一字符) 对。例如文本 "ABCDEFG"，seq_len=3：
- 样本 0：输入 "ABC" → 目标 "D"
- 样本 1：输入 "BCD" → 目标 "E"
- 依此类推...

```python
class CharSeqDataset(Dataset):
    def __init__(self, text, seq_length=30):
        # 将整个文本转为索引序列
        self.data = [self.char_to_idx[ch] for ch in text]
        # 构建样本对
        self.samples = []
        for i in range(0, len(self.data) - seq_length):
            input_seq = self.data[i:i + seq_length]       # 前30个字符
            target_char = self.data[i + seq_length]       # 第31个字符
            self.samples.append((input_seq, target_char))
```

#### 5.2 模型结构

```python
class CharRNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, cell_type='lstm'):
        self.embed = nn.Embedding(vocab_size, embed_dim)     # 字符→稠密向量
        # 根据 cell_type 选择细胞
        if cell_type == 'rnn':
            self.cell = MyRNNCell(embed_dim, hidden_size)
        elif cell_type == 'lstm':
            self.cell = MyLSTMCell(embed_dim, hidden_size)
        elif cell_type == 'gru':
            self.cell = MyGRUCell(embed_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)  # 隐藏状态→词汇表
```

**`nn.Embedding(vocab_size, embed_dim)`**：将离散的字符索引（如 0, 1, 2, ...）映射为 `embed_dim` 维的稠密向量。这比 one-hot 编码更紧凑，且能学习字符之间的语义关系。

**`nn.Linear(hidden_size, vocab_size)`**：将 hidden_size 维的隐藏状态投影到 vocab_size 维的 logits 空间，每个维度对应一个字符的"得分"。

#### 5.3 前向传播：沿时间步循环

```python
def forward(self, x, h_prev=None, c_prev=None):
    batch_size, seq_len = x.shape
    # 初始化隐藏状态为零向量
    if h_prev is None:
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

    outputs = []
    for t in range(seq_len):                    # 逐时间步处理
        x_t = self.embed(x[:, t])               # (batch, embed_dim) — 嵌入当前字符
        if self.cell_type == 'lstm':
            h, c = self.cell(x_t, h, c)         # LSTM: 更新 h 和 c
        else:
            h = self.cell(x_t, h)               # RNN/GRU: 只更新 h
        logits = self.output_proj(h)            # (batch, vocab_size) — 每个字符的得分
        outputs.append(logits)

    outputs = torch.stack(outputs, dim=1)        # (batch, seq_len, vocab_size)
    return outputs, h, c
```

**关键点**：
- 循环 `for t in range(seq_len)` 实现了 RNN 的"时间展开"（unrolling）。同一套参数在每一时间步被复用，这是 RNN 能处理变长序列的根本原因。
- `torch.stack(outputs, dim=1)` 将 seq_len 个 `(batch, vocab_size)` 张量堆叠为 `(batch, seq_len, vocab_size)`。
- 隐藏状态 $h$ 在时间步之间传递——它携带了历史信息。

#### 5.4 训练：预测下一个字符

```python
def train_char_lm(model, dataset, epochs=30, lr=0.005):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            outputs, _, _ = model(inputs)                      # 前向
            loss = criterion(outputs[:, -1, :], targets)       # 只取最后一个时间步
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
```

**`outputs[:, -1, :]`**：只取最后一个时间步的输出作为预测。因为目标是"读完 30 个字符后，预测第 31 个字符"。

**梯度裁剪 `clip_grad_norm_`**：RNN 训练中梯度容易爆炸（BPTT 中连乘的 $\prod \frac{\partial h_t}{\partial h_{t-1}}$ 可能远大于 1）。梯度裁剪将梯度向量的模长限制在 `max_norm` 以内，是训练 RNN 的标准操作。

#### 5.5 文本生成：自回归采样

```python
def generate_text(model, dataset, seed_text, gen_length=50, temperature=0.8):
    with torch.no_grad():
        for _ in range(gen_length):
            logits = outputs[0, -1, :] / temperature  # 温度缩放
            probs = F.softmax(logits, dim=-1)          # 转为概率分布
            next_idx = torch.multinomial(probs, 1)     # 按概率采样（非贪心）
            indices.append(next_idx)
```

**温度参数 `temperature`** 控制生成的随机性：
- `temperature < 1`（如 0.5）：概率分布更尖锐，模型更倾向于选高分 token → 输出更确定但可能重复
- `temperature > 1`（如 1.5）：概率分布更平坦，更多低分 token 也有机会被选中 → 输出更多样但可能不合理
- `temperature = 0`：等价于贪心解码（每次都选概率最高的 token）

**`torch.multinomial(probs, 1)`** 按概率采样而非取 argmax。这让生成具有多样性——即使模型的"最优"选择是某个词，采样也可能选择次优的词，产生更有趣的文本。

---

### 第6步：序列分类 — `SentimentRNN`

分类模型与语言模型的区别在于，我们只关心序列末尾的隐藏状态（它聚合了全部序列信息）：

```python
class SentimentRNN(nn.Module):
    def forward(self, x):
        for t in range(seq_len):
            x_t = self.embed(x[:, t])
            if self.cell_type == 'lstm':
                h, c = self.cell(x_t, h, c)
            else:
                h = self.cell(x_t, h)
        # 取最后一个隐藏状态做分类
        return self.classifier(h)   # (batch, 1) — 二分类 logits
```

**`BCEWithLogitsLoss`**：将 sigmoid 和二分类交叉熵合二为一，数值上更稳定。等价于 `sigmoid(logits)` 后计算 binary cross-entropy。

---

### 第7步：实验结果与对比

训练完成后，代码自动生成两张对比图：
- **`rnn_lstm_gru_loss_comparison.png`**：三个模型的训练损失曲线。预期 LSTM 和 GRU 的收敛速度快于 RNN，最终损失也更低。
- **`rnn_lstm_gru_classification_accuracy.png`**：三个模型的分类准确率。预期 LSTM > GRU > RNN。

**为什么 LSTM/GRU 优于 RNN？** 根本原因是梯度传播路径的差异：

| 模型 | 关键梯度路径 | 梯度消失风险 |
|------|------------|------------|
| RNN | $\frac{\partial h_t}{\partial h_{t-1}}$ 含 $\tanh'$ | 指数衰减 |
| LSTM | $\frac{\partial c_t}{\partial c_{t-1}} = f_t \approx 1$ | 几乎无损 |
| GRU | $h_t = (1-z)h_{t-1} + z\tilde{h}_t$ | 与 LSTM 类似的加法路径 |

---

## 关键概念速查表

| 概念 | 公式 | 一句话 |
|------|------|--------|
| RNN 隐藏状态 | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ | 当前输入+历史记忆的加权组合 |
| LSTM 遗忘门 | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ | 决定丢弃哪些旧细胞状态信息 |
| LSTM 输入门 | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ | 决定写入哪些新信息 |
| LSTM 输出门 | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ | 决定暴露哪些信息到隐藏状态 |
| LSTM 细胞状态 | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | 信息高速公路，梯度可无损传播 |
| GRU 重置门 | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ | 忽略多少历史信息 |
| GRU 更新门 | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ | 历史 vs 新信息的插值系数 |
| GRU 隐藏状态 | $h_t = (1-z_t)h_{t-1} + z_t\tilde{h}_t$ | LSTM 的精简版 |
| 梯度裁剪 | $\|g\| \leftarrow \min(1, \text{max\_norm}/\|g\|) \cdot g$ | 防止梯度爆炸 |
| 温度采样 | $P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$ | 控制生成随机性 |
| 字符嵌入 | $\text{Embed}(idx)$ → $d_{\text{embed}}$ 维向量 | 离散字符→连续向量 |

---

## 完整代码

<<< @/nlp/sequence-models/code/demo.py
