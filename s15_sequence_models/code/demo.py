# -*- coding: utf-8 -*-
"""
s15 序列模型 demo：RNN / LSTM / GRU 从零实现与对比
====================================================
本文件从零实现了 RNN、LSTM、GRU 三个序列模型，并通过两个任务
展示它们的实际表现：
  任务1：字符级语言模型（文本生成）
  任务2：序列分类（中文情感分析模拟）

运行方式：在 s15_sequence_models 目录下执行 `python code/demo.py`
依赖：numpy, torch, matplotlib
"""

import numpy as np
import math
from collections import Counter
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：从零实现 RNN、LSTM、GRU 细胞
# ============================================================

class MyRNNCell(nn.Module):
    """
    RNN 细胞的手动实现。
    h_t = tanh(W_hh @ h_{t-1} + W_ih @ x_t + b)
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        参数：
            input_size: 输入特征维度 d_x
            hidden_size: 隐藏状态维度 d_h
        """
        super().__init__()
        self.hidden_size = hidden_size
        # 输入到隐藏的线性变换
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        # 隐藏到隐藏的线性变换
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        RNN 前向传播。

        参数：
            x: 当前输入 (batch, input_size)
            h_prev: 上一时刻隐藏状态 (batch, hidden_size)
        返回：
            h: 当前隐藏状态 (batch, hidden_size)
        """
        return torch.tanh(self.W_ih(x) + self.W_hh(h_prev))


class MyLSTMCell(nn.Module):
    """
    LSTM 细胞的手动实现，包含遗忘门、输入门、输出门。
    细胞状态 c_t 以加法方式更新，解决梯度消失问题。
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        参数：
            input_size: 输入特征维度 d_x
            hidden_size: 隐藏状态维度 d_h
        """
        super().__init__()
        self.hidden_size = hidden_size
        # 四个线性变换合并到一个矩阵中以提高效率：W·[h_{t-1}, x_t]
        # 输出维度 = 4 * hidden_size (f, i, c̃, o 各 hidden_size)
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=True)

    def forward(
        self, x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LSTM 前向传播。

        参数：
            x: 当前输入 (batch, input_size)
            h_prev: 上一时刻隐藏状态 (batch, hidden_size)
            c_prev: 上一时刻细胞状态 (batch, hidden_size)
        返回：
            h: 当前隐藏状态 (batch, hidden_size)
            c: 当前细胞状态 (batch, hidden_size)
        """
        # 拼接输入和上一隐藏状态
        combined = torch.cat([h_prev, x], dim=1)  # (batch, input_size + hidden_size)
        # 一次矩阵乘法计算所有门的值
        gates = self.W(combined)  # (batch, 4 * hidden_size)
        # 拆分为四个门
        f_gate, i_gate, c_tilde, o_gate = gates.chunk(4, dim=1)
        # 遗忘门 — 控制丢弃哪些旧信息
        f = torch.sigmoid(f_gate)       # (batch, hidden_size)
        # 输入门 — 控制写入哪些新信息
        i = torch.sigmoid(i_gate)       # (batch, hidden_size)
        # 候选细胞状态 — 新信息的内容
        c_tilde = torch.tanh(c_tilde)   # (batch, hidden_size)
        # 细胞状态更新 — 加法方式：c_t = f ⊙ c_{t-1} + i ⊙ c̃
        c = f * c_prev + i * c_tilde
        # 输出门 — 控制暴露哪些信息到 h_t
        o = torch.sigmoid(o_gate)       # (batch, hidden_size)
        # 隐藏状态输出
        h = o * torch.tanh(c)
        return h, c


class MyGRUCell(nn.Module):
    """
    GRU 细胞的手动实现，包含重置门和更新门。
    相比 LSTM 去掉了独立的细胞状态，用双门机制简化设计。
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        参数：
            input_size: 输入特征维度 d_x
            hidden_size: 隐藏状态维度 d_h
        """
        super().__init__()
        self.hidden_size = hidden_size
        # 重置门和更新门的线性变换
        self.W_rz = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=True)
        # 候选隐藏状态的线性变换
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        GRU 前向传播。

        参数：
            x: 当前输入 (batch, input_size)
            h_prev: 上一时刻隐藏状态 (batch, hidden_size)
        返回：
            h: 当前隐藏状态 (batch, hidden_size)
        """
        combined = torch.cat([h_prev, x], dim=1)
        # 重置门 r 和更新门 z
        rz = self.W_rz(combined)
        r_gate, z_gate = rz.chunk(2, dim=1)
        r = torch.sigmoid(r_gate)  # 重置门 — 控制忽略多少历史
        z = torch.sigmoid(z_gate)  # 更新门 — 控制保留多少历史 vs 写入多少新信息
        # 候选隐藏状态 h̃ = tanh(W_h · [r ⊙ h_{t-1}, x])
        combined_reset = torch.cat([r * h_prev, x], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        # 最终隐藏状态：z 控制历史(1-z)和新信息(z)的线性插值
        h = (1 - z) * h_prev + z * h_tilde
        return h


# ============================================================
# 第二部分：字符级语言模型（文本生成）
# ============================================================

# 训练文本：使用一个简单的中文文本作为训练数据
TEXT_CORPUS = """
人工智能是计算机科学的一个重要分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
深度学习是机器学习的一个分支，它通过多层神经网络来学习数据的特征表示。
循环神经网络是一种用于处理序列数据的神经网络，它可以捕捉时间序列中的依赖关系。
长短期记忆网络通过门控机制解决了传统循环神经网络的梯度消失问题。
门控循环单元是长短期记忆网络的一种简化变体，它使用更少的门控机制达到了类似的效果。
自然语言处理是人工智能的一个重要应用领域，它研究如何让计算机理解和使用人类语言。
"""


def build_char_vocab(text: str) -> Tuple[dict, dict, int]:
    """
    构建字符级词汇表。

    参数：
        text: 原始文本
    返回：
        char_to_idx: 字符→索引映射
        idx_to_char: 索引→字符映射
        vocab_size: 词汇表大小
    """
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


class CharSeqDataset(Dataset):
    """
    字符级序列数据集：每个样本是一段长度为 seq_length 的字符序列，
    目标是下一个字符。
    """

    def __init__(self, text: str, seq_length: int = 30):
        """
        参数：
            text: 原始文本字符串
            seq_length: 输入序列长度
        """
        self.seq_length = seq_length
        self.char_to_idx, self.idx_to_char, self.vocab_size = build_char_vocab(text)
        # 将整个文本转为索引序列
        self.data = [self.char_to_idx[ch] for ch in text]
        # 构建 (输入序列, 目标字符) 对
        self.samples = []
        for i in range(0, len(self.data) - seq_length):
            input_seq = self.data[i:i + seq_length]
            target_char = self.data[i + seq_length]
            self.samples.append((input_seq, target_char))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class CharRNNLM(nn.Module):
    """
    字符级 RNN 语言模型。
    支持切换到 RNN、LSTM、GRU 三种细胞类型。
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, cell_type: str = 'lstm'):
        """
        参数：
            vocab_size: 字符集大小
            embed_dim: 字符嵌入维度
            hidden_size: 隐藏状态维度
            cell_type: 'rnn' | 'lstm' | 'gru'
        """
        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 根据 cell_type 选择细胞类型
        if cell_type == 'rnn':
            self.cell = MyRNNCell(embed_dim, hidden_size)
        elif cell_type == 'lstm':
            self.cell = MyLSTMCell(embed_dim, hidden_size)
        elif cell_type == 'gru':
            self.cell = MyGRUCell(embed_dim, hidden_size)
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")
        # 输出投影：hidden_size → vocab_size
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: torch.Tensor, h_prev=None, c_prev=None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：处理整个输入序列。

        参数：
            x: 输入序列 (batch, seq_len)
            h_prev: 初始隐藏状态 (可选的 batch 维初始化)
            c_prev: 初始细胞状态（仅 LSTM）
        返回：
            outputs: 每个时间步的 logits (batch, seq_len, vocab_size)
            h: 最终隐藏状态
            c: 最终细胞状态（仅 LSTM，否则 None）
        """
        batch_size, seq_len = x.shape
        # 初始化隐藏状态
        if h_prev is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_prev
        c = c_prev
        if self.cell_type == 'lstm' and c is None:
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []
        for t in range(seq_len):
            # 嵌入当前字符
            x_t = self.embed(x[:, t])  # (batch, embed_dim)
            # 循环细胞前向
            if self.cell_type == 'lstm':
                h, c = self.cell(x_t, h, c)
            else:
                h = self.cell(x_t, h)
            # 投影到词汇表空间
            logits = self.output_proj(h)  # (batch, vocab_size)
            outputs.append(logits)

        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)
        return outputs, h, c


def train_char_lm(model: CharRNNLM, dataset: CharSeqDataset, epochs: int = 30, lr: float = 0.005):
    """
    训练字符级语言模型。

    参数：
        model: CharRNNLM 模型
        dataset: 字符序列数据集
        epochs: 训练轮数
        lr: 学习率
    返回：
        loss_history: 每个 epoch 的平均损失
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            # outputs: (batch, seq_len, vocab_size), targets: (batch,)
            # 取最后一个时间步的输出预测下一个字符
            loss = criterion(outputs[:, -1, :], targets)
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history


def generate_text(
    model: CharRNNLM,
    dataset: CharSeqDataset,
    seed_text: str,
    gen_length: int = 50,
    temperature: float = 0.8,
):
    """
    用训练好的语言模型生成文本。

    参数：
        model: 训练好的 CharRNNLM
        dataset: 数据集（用于获取词汇表映射）
        seed_text: 种子文本（生成的起始字符序列）
        gen_length: 要生成的字符数
        temperature: 温度参数（越小越确定，越大越随机）
    返回：
        generated: 生成的完整文本
    """
    device = next(model.parameters()).device
    model.eval()
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # 将种子文本转为索引
    indices = [char_to_idx.get(ch, 0) for ch in seed_text]
    generated = seed_text

    h, c = None, None
    with torch.no_grad():
        for _ in range(gen_length):
            # 取最后 seq_length 个字符作为输入
            input_seq = indices[-30:] if len(indices) >= 30 else indices
            x = torch.tensor([input_seq], dtype=torch.long, device=device)
            outputs, h, c = model(x, h_prev=h, c_prev=c)
            # 取最后一个时间步的 logits
            logits = outputs[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # 按概率采样下一个字符
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)
            generated += idx_to_char[next_idx]
    return generated


# 数据集
dataset = CharSeqDataset(TEXT_CORPUS, seq_length=30)
print(f"[字符LM] 词汇表大小: {dataset.vocab_size}, 训练样本数: {len(dataset)}")
print()

# 训练三种模型并记录损失,用于对比
loss_histories = {}
models_trained = {}
cell_types = ['rnn', 'lstm', 'gru']

print("[字符LM 训练] 对比 RNN / LSTM / GRU")
print("=" * 60)
for ct in cell_types:
    print(f"\n--- 训练 {ct.upper()} ---")
    model = CharRNNLM(dataset.vocab_size, embed_dim=32, hidden_size=64, cell_type=ct)
    history = train_char_lm(model, dataset, epochs=50, lr=0.005)
    loss_histories[ct] = history
    models_trained[ct] = model

# 绘制训练损失曲线对比
plt.figure(figsize=(10, 5))
colors = {'rnn': '#E53935', 'lstm': '#1E88E5', 'gru': '#43A047'}
for ct in cell_types:
    plt.plot(loss_histories[ct], color=colors[ct], linewidth=1.5, label=ct.upper())
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("RNN vs LSTM vs GRU 字符语言模型训练损失对比", fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/rnn_lstm_gru_loss_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[可视化] 训练损失对比图已保存至 images/rnn_lstm_gru_loss_comparison.png")

# 文本生成演示
print("\n" + "=" * 60)
print("[文本生成 Demo] 用不同模型生成文本")
print("=" * 60)
seed = "人工智能"
for ct in cell_types:
    generated = generate_text(models_trained[ct], dataset, seed, gen_length=40, temperature=0.8)
    print(f"\n[{ct.upper()}] 种子: 「{seed}」")
    print(f"  生成: {generated}")

# ============================================================
# 第三部分：序列分类（中文情感分析模拟）
# ============================================================

# 模拟中文评论情感分析数据
review_data = [
    ("这个产品质量非常好 我很喜欢", 1),
    ("客服态度恶劣 商品有瑕疵 太失望了", 0),
    ("物流很快 包装也很精美 好评", 1),
    ("用了几次就坏了 质量太差 不值得购买", 0),
    ("性价比很高 推荐大家购买 非常好用", 1),
    ("完全不值这个价钱 被图片骗了 差评", 0),
    ("已经第二次购买了 质量稳定 满意", 1),
    ("外观好看但是不耐用 中评", 0),
    ("功能强大 操作简单 老人家也能用", 1),
    ("收到货就发现坏了 退货还麻烦", 0),
    ("比实体店便宜多了 正品无疑", 1),
    ("做工粗糙 和描述不符 上当了", 0),
    ("用了一个月 感觉不错 值得入手", 1),
    ("发货特别慢 等了一周才到 体验不好", 0),
    ("颜色很正 大小合适 满意的一次购物", 1),
    ("有异味 不敢用 联系客服也不回复", 0),
    ("买给父母的 他们很喜欢 功能齐全", 1),
    ("刚买就降价了 还不给保价 生气", 0),
    ("材质很好 手感不错 下次还来", 1),
    ("安装复杂 说明书太简陋 体验差", 0),
]

# 构建字符级词汇表用于分类
all_chars = sorted(set(''.join([r[0] for r in review_data])))
char2idx_cls = {ch: i for i, ch in enumerate(all_chars)}
idx2char_cls = {i: ch for i, ch in enumerate(all_chars)}
V_cls = len(all_chars)


class ReviewDataset(Dataset):
    """情感分析数据集"""

    def __init__(self, data, char2idx, max_len=30):
        self.data = data
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        indices = [self.char2idx.get(ch, 0) for ch in text]
        # 填充或截断到 max_len
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


class SentimentRNN(nn.Module):
    """
    基于 RNN/LSTM/GRU 的序列分类模型。
    序列末尾的隐藏状态用于分类。
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, cell_type='lstm'):
        super().__init__()
        self.cell_type = cell_type
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if cell_type == 'rnn':
            self.cell = MyRNNCell(embed_dim, hidden_size)
        elif cell_type == 'lstm':
            self.cell = MyLSTMCell(embed_dim, hidden_size)
        elif cell_type == 'gru':
            self.cell = MyGRUCell(embed_dim, hidden_size)
        # 分类头：取最后时刻的隐藏状态进行分类
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: (batch, seq_len) 字符索引序列
        返回：
            logits: (batch, 1) 二分类 logits
        """
        batch_size, seq_len = x.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.cell.hidden_size, device=x.device) if self.cell_type == 'lstm' else None

        for t in range(seq_len):
            x_t = self.embed(x[:, t])
            if self.cell_type == 'lstm':
                h, c = self.cell(x_t, h, c)
            else:
                h = self.cell(x_t, h)
        # 取最后的隐藏状态做分类
        return self.classifier(h)


def train_sentiment_model(
    model: SentimentRNN, train_loader: DataLoader, epochs: int = 50, lr: float = 0.01
):
    """训练情感分类模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs).squeeze(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 准确率
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        history.append({'loss': total_loss / len(train_loader), 'acc': correct / total})
    return history


# 训练分类模型
print("\n" + "=" * 60)
print("[序列分类] 中文评论情感分析")
print("=" * 60)

review_dataset = ReviewDataset(review_data, char2idx_cls, max_len=30)
review_loader = DataLoader(review_dataset, batch_size=4, shuffle=True)

cls_histories = {}
for ct in cell_types:
    print(f"\n--- 训练 {ct.upper()} 分类器 ---")
    cls_model = SentimentRNN(V_cls, embed_dim=16, hidden_size=32, cell_type=ct)
    history = train_sentiment_model(cls_model, review_loader, epochs=40, lr=0.01)
    cls_histories[ct] = history
    final_acc = history[-1]['acc']
    print(f"  最终准确率: {final_acc:.2%}")

# 绘制分类准确率曲线
plt.figure(figsize=(10, 5))
for ct in cell_types:
    accs = [h['acc'] for h in cls_histories[ct]]
    plt.plot(accs, color=colors[ct], linewidth=1.5, label=f"{ct.upper()} (最终: {accs[-1]:.2%})")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("准确率", fontsize=12)
plt.title("RNN vs LSTM vs GRU 情感分类准确率对比", fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/rnn_lstm_gru_classification_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[可视化] 分类准确率对比图已保存至 images/rnn_lstm_gru_classification_accuracy.png")

# ============================================================
# 第四部分：总结
# ============================================================

print("\n" + "=" * 60)
print("[总结] RNN → LSTM → GRU 核心要点")
print("=" * 60)
print("""
  RNN:   h_t = tanh(W_h h_{t-1} + W_x x_t)   — 简单但梯度消失
  LSTM:  三扇门(遗忘/输入/输出) + 细胞状态 c_t  — 加法梯度路径
  GRU:   双扇门(重置/更新) + 合并 h_t 和 c_t  — 精简但有效

  梯度消失根源: ∂h_t/∂h_{t-1} 含有 tanh' ≤ 1，多次连乘 → 指数衰减
   LSTM 的解法: c_t = f ⊙ c_{t-1} + ... → ∂c_t/∂c_{t-1} = f ≈ 1

  下一章 [s16 Attention 与 Transformer]:
    注意力机制用全局关注取代了逐个时间步传递，
    彻底解决了长距离依赖问题，并极大加速了训练。
""")
print("所有 demo 运行完成！图表已保存至 images/ 目录。")
