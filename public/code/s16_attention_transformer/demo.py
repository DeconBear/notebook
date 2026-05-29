# -*- coding: utf-8 -*-
"""
s16 Attention 与 Transformer demo：从零实现注意力与 Mini-GPT
============================================================
本文件从零实现了 Transformer 的核心组件：
  1. 缩放点积注意力（Scaled Dot-Product Attention）
  2. 多头自注意力（Multi-Head Self-Attention）
  3. Transformer 编码器 Block（Attention + FFN + LayerNorm + 残差）
  4. Mini-GPT（decoder-only）用于字符级文本生成
  5. 注意力可视化（热力图）
  6. √d_k 缩放效果对比

运行方式：在 s16_attention_transformer 目录下执行 `python code/demo.py`
依赖：numpy, torch, matplotlib, seaborn
"""

import numpy as np
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# GPU 自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("（未检测到 GPU，使用 CPU 运行。如有 GPU，请安装 CUDA 版 PyTorch 以获得加速）")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================
# 第一部分：从零实现 Attention 核心组件
# ============================================================

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    缩放点积注意力：Attention(Q, K, V) = softmax(QK^T / √d_k) V

    参数：
        Q: Query 矩阵 (..., seq_len_q, d_k)
        K: Key 矩阵 (..., seq_len_k, d_k)
        V: Value 矩阵 (..., seq_len_v, d_v)，seq_len_v = seq_len_k
        mask: 注意力掩码 (..., seq_len_q, seq_len_k)，True 的位置将被设为 -inf
    返回：
        output: 加权后的 Value (..., seq_len_q, d_v)
    """
    d_k = Q.size(-1)
    # QK^T: 计算 Query 和 Key 的点积
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    # 缩放：除以 √d_k，防止点积过大导致 softmax 饱和
    scores = scores / math.sqrt(d_k)
    # 应用掩码（如因果掩码）：掩码位置设为负无穷
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    # softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)  # (..., seq_len_q, seq_len_k)
    # 加权求和 Value
    output = torch.matmul(attn_weights, V)    # (..., seq_len_q, d_v)
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）。
    不使用 nn.MultiheadAttention，完全从零实现。

    参数：
        d_model: 模型维度（输入/输出维度）
        num_heads: 注意力头数
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性投影（合并所有头到一个矩阵中）
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # (d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        参数：
            x: 输入序列 (batch, seq_len, d_model)
            mask: 注意力掩码 (batch, 1, seq_len, seq_len) 或 broadcastable
        返回：
            output: 多头注意力输出 (batch, seq_len, d_model)
            attn_weights: 注意力权重（用于可视化，第一个头的平均）
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影得到 Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # 拆分为多头：reshape 为 (batch, seq_len, num_heads, d_k) → transpose
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 现在形状: (batch, num_heads, seq_len, d_k)

        # 缩放点积注意力（mask 已在 MiniGPT.forward 中塑形为 (1, 1, seq_len, seq_len)，可直接广播）
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, seq_len, d_k)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        # 最终线性投影
        output = self.W_O(attn_output)
        output = self.dropout(output)

        # 为可视化保存第一个头的注意力权重
        # 计算用于可视化的注意力权重
        with torch.no_grad():
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)

        return output, attn_weights


class FeedForward(nn.Module):
    """
    Transformer 的 Position-wise Feed-Forward Network。
    FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2
    维度变化：d_model → 4*d_model → d_model
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    一个完整的 Transformer Encoder Block：
    Input → LayerNorm → Multi-Head Self-Attention → Add (残差)
    → LayerNorm → FFN → Add (残差) → Output
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        # LayerNorm（Pre-LN 风格）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        前向传播。

        参数：
            x: 输入 (batch, seq_len, d_model)
            mask: 注意力掩码
        返回：
            output: 输出 (batch, seq_len, d_model)
            attn_weights: 注意力权重（用于可视化）
        """
        # 子层 1: 自注意力 + 残差（Pre-LN: LN 在注意力之前）
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, mask)
        x = residual + self.dropout(attn_out)

        # 子层 2: FFN + 残差
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))
        return x, attn_weights


# ============================================================
# 第二部分：位置编码
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    """
    正弦位置编码。
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 计算分母: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer（不参与训练，但随模型保存）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码加到输入上"""
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 第三部分：Mini-GPT（Decoder-only Transformer）
# ============================================================

class MiniGPT(nn.Module):
    """
    一个微型的 GPT 风格模型（Decoder-only Transformer）。
    用于字符级文本生成。

    参数：
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: Transformer Block 层数
        max_len: 最大序列长度
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # 词嵌入 + 位置编码
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionEncoding(d_model, max_len)
        self.dropout_embed = nn.Dropout(dropout)
        # 堆叠 Transformer Block
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # 最终 LayerNorm + 输出投影
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重绑定：将 lm_head 的权重与 token_embed 共享
        self.lm_head.weight = self.token_embed.weight

        # 缓存因果掩码，避免每次前向都重新计算
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        )

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        前向传播。

        参数：
            x: 输入 token 序列 (batch, seq_len)
            return_attn: 是否返回注意力权重（用于可视化）
        返回：
            logits: (batch, seq_len, vocab_size)
            attn_maps: 各层的注意力权重列表（可选）
        """
        batch_size, seq_len = x.shape
        # 获取因果掩码 — 塑形为 (1, 1, seq_len, seq_len) 以便与 (batch, num_heads, seq_len, seq_len) 广播
        mask = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        # 词嵌入 + 位置编码
        x_emb = self.token_embed(x) * math.sqrt(self.d_model)  # 缩放嵌入
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout_embed(x_emb)
        # 通过所有 Transformer Block
        attn_maps = [] if return_attn else None
        hidden = x_emb
        for block in self.blocks:
            hidden, attn_weights = block(hidden, mask)
            if return_attn:
                attn_maps.append(attn_weights)
        # 最终 LayerNorm + LM Head
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)  # (batch, seq_len, vocab_size)
        return logits, attn_maps

    def generate(
        self, seed_tokens: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8
    ) -> List[int]:
        """
        自回归生成文本。

        参数：
            seed_tokens: 初始 token 序列 (1, seed_len)
            max_new_tokens: 生成的最大 token 数
            temperature: 温度参数
        返回：
            generated: 生成的 token 序列
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = seed_tokens.tolist()[0] if seed_tokens.dim() > 1 else seed_tokens.tolist()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 取最后 max_len 个 token 作为输入
                input_seq = torch.tensor([tokens[-128:]], dtype=torch.long, device=device)
                logits, _ = self.forward(input_seq)
                # 取最后一个位置的 logits
                next_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)
        return tokens


# ============================================================
# 第四部分：训练与可视化
# ============================================================

# 使用一个简单的中文语料训练字符级模型
TEXT = """
深度学习的核心是神经网络，它通过多层非线性变换来学习数据的层次化特征表示。
反向传播算法是训练神经网络的关键技术，它利用链式法则高效地计算损失函数对每个参数的梯度。
卷积神经网络擅长处理图像数据，它通过局部连接和权重共享来捕捉空间结构信息。
循环神经网络适合处理序列数据，它通过隐藏状态来记忆历史信息。
注意力机制是Transformer架构的核心，它允许模型在处理序列时动态地关注不同位置的信息。
Transformer架构完全基于注意力机制，抛弃了传统的循环和卷积结构。
自然语言处理是人工智能的重要领域，它致力于让计算机理解和生成人类语言。
计算机视觉使机器能够像人类一样看懂图像和视频。
强化学习通过试错和奖励机制让智能体学会最优决策策略。
生成对抗网络通过生成器和判别器的对抗训练来生成逼真的数据样本。
知识图谱以结构化的方式表示实体之间的语义关系。
迁移学习利用在大规模数据上预训练的模型来提升小样本任务的性能。
大语言模型通过在海量文本上预训练，展现出惊人的语言理解和生成能力。
"""


def build_dataset(text: str, seq_len: int = 30):
    """构建字符级训练数据集"""
    chars = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = [char2idx[ch] for ch in text]
    # 构建 (input_seq, target_seq) 对
    samples = []
    for i in range(0, len(data) - seq_len):
        input_seq = data[i:i + seq_len]
        target_seq = data[i + 1:i + seq_len + 1]  # 目标序列是输入右移一位
        samples.append((input_seq, target_seq))
    return samples, char2idx, idx2char, len(chars)


class MiniGPTDataset(Dataset):
    """Mini-GPT 训练数据集"""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# 构建数据集
samples, char2idx, idx2char, vocab_size = build_dataset(TEXT, seq_len=30)
dataset = MiniGPTDataset(samples)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f"[数据集] 词汇表大小: {vocab_size}, 训练样本数: {len(dataset)}")

# 创建模型
model = MiniGPT(
    vocab_size=vocab_size,
    d_model=64,
    num_heads=4,
    num_layers=3,
    d_ff=256,
    max_len=128,
    dropout=0.1,
)
print(f"[模型] 参数量: {sum(p.numel() for p in model.parameters()):,}")
device = DEVICE
model = model.to(device)
print(f"[设备] {device}")

# 训练
print("\n[训练] 开始训练 Mini-GPT...")
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
loss_history = []

for epoch in range(80):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        # logits: (batch, seq_len, vocab_size), targets: (batch, seq_len)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/80, Loss: {avg_loss:.4f}")

# 绘制训练损失
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='#1E88E5', linewidth=1.5)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Mini-GPT Character Language Model Training Loss", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'minigpt_loss_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] 训练损失图已保存")

# ============================================================
# 第五部分：注意力可视化
# ============================================================

# 取一个样本输入，可视化注意力权重
sample_text = "深度学习是人工智能的重要领域"
sample_tokens = torch.tensor([[char2idx.get(ch, 0) for ch in sample_text]], dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    _, attn_maps = model(sample_tokens, return_attn=True)

# 绘制各层的注意力热力图（取第一个头）
fig, axes = plt.subplots(1, len(attn_maps), figsize=(5 * len(attn_maps), 4))
if len(attn_maps) == 1:
    axes = [axes]

for layer_idx, attn in enumerate(attn_maps):
    # attn: (batch, num_heads, seq_len, seq_len)
    # 取第一个样本、所有头的平均值
    avg_attn = attn[0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
    ax = axes[layer_idx]
    im = ax.imshow(avg_attn, cmap='YlOrRd', aspect='auto')
    # 标注因果掩码的下三角区域
    ax.set_xticks(range(len(sample_text)))
    ax.set_xticklabels(list(sample_text), fontsize=9, rotation=45)
    ax.set_yticks(range(len(sample_text)))
    ax.set_yticklabels(list(sample_text), fontsize=9)
    ax.set_title(f"Layer {layer_idx+1} Average Attention", fontsize=11)
plt.suptitle("Mini-GPT Attention Weight Heatmaps by Layer (Causal Mask Lower Triangle)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'attention_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] 注意力热力图已保存至 images/attention_heatmap.png")

# 多头注意力对比（取最后一层）
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
last_attn = attn_maps[-1][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
for head_idx in range(min(4, last_attn.shape[0])):
    ax = axes[head_idx]
    im = ax.imshow(last_attn[head_idx], cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(sample_text)))
    ax.set_xticklabels(list(sample_text), fontsize=8, rotation=45)
    ax.set_yticks(range(len(sample_text)))
    ax.set_yticklabels(list(sample_text), fontsize=8)
    ax.set_title(f"Attention Head {head_idx+1}", fontsize=11)
plt.suptitle("Last Layer: 4 Attention Heads Showing Different Focus Patterns", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'multihead_attention_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] 多头注意力模式图已保存")

# ============================================================
# 第六部分：√d_k 缩放效果对比
# ============================================================

print("\n" + "=" * 60)
print("[实验] √d_k 缩放对 softmax 的影响")
print("=" * 60)

# 模拟不同 d_k 下的点积得分分布
torch.manual_seed(42)
seq_len = 16
for d_k in [4, 16, 64, 256]:
    Q_test = torch.randn(1, seq_len, d_k)
    K_test = torch.randn(1, seq_len, d_k)
    # 不缩放
    scores_no_scale = torch.matmul(Q_test, K_test.transpose(-2, -1))
    # 缩放
    scores_scaled = scores_no_scale / math.sqrt(d_k)

    # 计算 softmax 分布的熵（熵越高，分布越均匀）
    probs_no_scale = F.softmax(scores_no_scale, dim=-1)
    probs_scaled = F.softmax(scores_scaled, dim=-1)

    # 计算每行的平均熵
    entropy_no = -(probs_no_scale * torch.log(probs_no_scale + 1e-9)).sum(dim=-1).mean().item()
    entropy_scaled = -(probs_scaled * torch.log(probs_scaled + 1e-9)).sum(dim=-1).mean().item()
    # 最大注意力权重（越大越集中）
    max_attn_no = probs_no_scale.max(dim=-1)[0].mean().item()
    max_attn_scaled = probs_scaled.max(dim=-1)[0].mean().item()

    print(f"\nd_k = {d_k:3d}:")
    print(f"  无缩放: 平均熵 = {entropy_no:.4f}, 最大注意力 = {max_attn_no:.4f}")
    print(f"  有缩放: 平均熵 = {entropy_scaled:.4f}, 最大注意力 = {max_attn_scaled:.4f}")

# 大 d_k 下的对比柱状图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
d_k_values = [4, 16, 64, 256]
entropy_no_list, entropy_scaled_list = [], []
max_no_list, max_scaled_list = [], []

for d_k in d_k_values:
    Q_test = torch.randn(1, seq_len, d_k)
    K_test = torch.randn(1, seq_len, d_k)
    scores_no = torch.matmul(Q_test, K_test.transpose(-2, -1))
    scores_sc = scores_no / math.sqrt(d_k)
    probs_no = F.softmax(scores_no, dim=-1)
    probs_sc = F.softmax(scores_sc, dim=-1)
    entropy_no_list.append(-(probs_no * torch.log(probs_no + 1e-9)).sum(dim=-1).mean().item())
    entropy_scaled_list.append(-(probs_sc * torch.log(probs_sc + 1e-9)).sum(dim=-1).mean().item())
    max_no_list.append(probs_no.max(dim=-1)[0].mean().item())
    max_scaled_list.append(probs_sc.max(dim=-1)[0].mean().item())

x = np.arange(len(d_k_values))
w = 0.35
axes[0].bar(x - w/2, entropy_no_list, w, label='No Scaling', color='#E53935', alpha=0.8)
axes[0].bar(x + w/2, entropy_scaled_list, w, label='With Scaling 1/sqrt(d_k)', color='#1E88E5', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"d_k={v}" for v in d_k_values])
axes[0].set_ylabel("Softmax Average Entropy", fontsize=11)
axes[0].set_title("Entropy (Higher = More Uniform)", fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x - w/2, max_no_list, w, label='No Scaling', color='#E53935', alpha=0.8)
axes[1].bar(x + w/2, max_scaled_list, w, label='With Scaling 1/sqrt(d_k)', color='#1E88E5', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"d_k={v}" for v in d_k_values])
axes[1].set_ylabel("Average Max Attention Weight", fontsize=11)
axes[1].set_title("Max Attention (Lower = Less Saturated)", fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle("Effect of 1/sqrt(d_k) Scaling on Softmax Saturation: Without Scaling, Small d_k is More Uniform, Large d_k is Extremely Concentrated", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'dk_scaling_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n[可视化] √d_k 缩放效果对比图已保存")

# ============================================================
# 第七部分：文本生成
# ============================================================

print("\n" + "=" * 60)
print("[文本生成] Mini-GPT 生成演示")
print("=" * 60)

seeds = ["深度", "自然", "注意", "计算"]
for seed in seeds:
    seed_tokens = torch.tensor([[char2idx.get(ch, 0) for ch in seed]], dtype=torch.long, device=device)
    generated = model.generate(seed_tokens, max_new_tokens=30, temperature=0.8)
    gen_text = ''.join([idx2char.get(t, '?') for t in generated])
    print(f"  种子「{seed}」→ {gen_text}")

print("\n所有 demo 运行完成！图表已保存至 images/ 目录。")
print("\n" + "=" * 60)
print("[核心要点]")
print("=" * 60)
print("""
  1. Self-Attention = softmax(QK^T / √d_k) V
  2. QKV: Query(查)、Key(被查)、Value(值) —— 字典查询范式
  3. Multi-Head: 并行多组注意力，捕捉不同类型的语言关系
  4. 因果掩码: 上三角为 -inf，防止模型'偷看'未来
  5. Pre-LN: LayerNorm 在子层之前，训练更稳定
  6. √d_k 缩放: 没有它，大 d_k 时 softmax 会饱和→梯度消失
  7. 残差连接: 让深层 Transformer 梯度直通底层
""")
