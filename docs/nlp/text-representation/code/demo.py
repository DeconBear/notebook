# -*- coding: utf-8 -*-
"""
s14 文本表示 demo：TF-IDF + word2vec Skip-gram
==============================================
本文件从零实现了两大文本表示方法：
  1. TF-IDF 向量化 — 统计词频 × 逆文档频率
  2. word2vec Skip-gram + 负采样 — 用神经网络学习稠密词向量

并通过 t-SNE 可视化、近义词查询、类比推理等实验，
直观对比两种方法的表达能力。

运行方式：在 s14_text_representation 目录下执行 `python code/demo.py`
依赖：numpy, torch, matplotlib, scikit-learn, scipy
"""

import numpy as np
import math
import random
from collections import Counter
from typing import List, Dict, Tuple, Set

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

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES, exist_ok=True)

# ============================================================
# 第一部分：中文语料库
# ============================================================

CORPUS = [
    # 体育类
    "足球比赛在工人体育场举行 观众热情高涨",
    "篮球运动员在训练中表现出色 投篮命中率很高",
    "游泳选手在奥运会中打破世界纪录 获得金牌",
    "乒乓球是中国最受欢迎的运动项目之一",
    "马拉松选手在雨中坚持跑到终点 令人感动",
    # 科技类
    "人工智能技术正在改变各行各业的运作方式",
    "机器学习模型在图像识别任务中超越了人类水平",
    "深度学习需要大量的数据和计算资源进行训练",
    "自然语言处理是人工智能的重要研究方向",
    "计算机视觉技术可以帮助自动驾驶汽车识别路况",
    # 教育类
    "大学教授正在讲授机器学习的基础理论知识",
    "学生在图书馆认真复习准备期末考试",
    "教育部发布了新的课程改革方案",
    "中小学生的课外辅导负担需要进一步减轻",
    "在线教育平台为偏远地区学生提供了优质课程",
    # 经济类
    "股票市场今天大幅上涨 投资者信心回升",
    "央行降低了基准利率以刺激经济增长",
    "房地产市场调控政策持续发力 房价趋于稳定",
    "国际贸易摩擦对全球经济造成了不确定性",
    "企业数字化转型成为经济高质量发展的关键",
    # 医疗类
    "医生建议人们定期体检以预防疾病",
    "新型疫苗的研发为疫情防控带来了希望",
    "心理健康问题越来越受到社会的关注",
    "中药在现代医学中的应用研究取得进展",
    "医院引进了先进的医疗设备提升诊疗水平",
]


def tokenize(text: str) -> List[str]:
    """
    对中文文本进行简单分词（按字符切分，实际项目中建议使用 jieba 等分词工具）。

    参数：
        text: 输入的中文文本字符串
    返回：
        分词后的词列表
    """
    # 去掉空格并按字符切分（简化的中文分词）
    text = text.replace(" ", "")
    return list(text)


# 对语料库进行分词
tokenized_corpus = [tokenize(doc) for doc in CORPUS]
# 构建词汇表
all_words = [word for doc in tokenized_corpus for word in doc]
vocab = sorted(set(all_words))
word_to_idx = {w: i for i, w in enumerate(vocab)}  # 词 → 索引映射
idx_to_word = {i: w for i, w in enumerate(vocab)}  # 索引 → 词映射
V = len(vocab)  # 词汇表大小

print(f"[语料统计] 文档数: {len(CORPUS)}, 词汇表大小: {V}")
print(f"[语料统计] 总词数: {len(all_words)}")
print()

# ============================================================
# 第二部分：TF-IDF 从零实现
# ============================================================

def compute_tf(doc_tokens: List[str]) -> Dict[str, float]:
    """
    计算单篇文档中每个词的词频 TF(w, d) = count(w, d) / total_words_in_doc

    参数：
        doc_tokens: 一篇文档的分词列表
    返回：
        dict: {词: TF值}
    """
    counter = Counter(doc_tokens)
    total = len(doc_tokens)
    if total == 0:
        return {}
    return {word: count / total for word, count in counter.items()}


def compute_idf(tokenized_docs: List[List[str]]) -> Dict[str, float]:
    """
    计算逆文档频率 IDF(w) = log(N / df(w))，其中 N 是总文档数，df(w) 是包含 w 的文档数。

    参数：
        tokenized_docs: 所有文档的分词列表
    返回：
        dict: {词: IDF值}
    """
    N = len(tokenized_docs)
    idf = {}
    for doc in tokenized_docs:
        # 文档内去重——每个词只算一次（df 是包含该词的文档数）
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1
    # IDF = log(N / df)，加 1 平滑防止分母为 0
    for word in idf:
        idf[word] = math.log((N + 1) / (idf[word] + 1)) + 1
    return idf


# 计算全语料库的 IDF
idf_scores = compute_idf(tokenized_corpus)

# 构建 TF-IDF 矩阵：形状 (N_docs, V)
tfidf_matrix = np.zeros((len(CORPUS), V))
for doc_idx, doc_tokens in enumerate(tokenized_corpus):
    tf = compute_tf(doc_tokens)
    for word, tf_val in tf.items():
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            tfidf_matrix[doc_idx, word_idx] = tf_val * idf_scores.get(word, 0)

print("=" * 60)
print("[TF-IDF Demo] 每篇文档的 Top-3 关键词:")
print("=" * 60)
for doc_idx, doc in enumerate(CORPUS):
    # 获取当前文档的 TF-IDF 向量
    doc_vec = tfidf_matrix[doc_idx]
    # 取出得分最高的 3 个词的索引
    top_indices = np.argsort(doc_vec)[::-1][:3]
    top_words = [f"{idx_to_word[i]}({doc_vec[i]:.3f})" for i in top_indices if doc_vec[i] > 0]
    print(f"  文档{doc_idx+1}: {doc[:25]}...")
    print(f"      关键词: {', '.join(top_words)}")
print()

# 可视化：TF-IDF 热力图（部分词汇）
print("[TF-IDF 可视化] 正在绘制热力图...")
fig, ax = plt.subplots(figsize=(16, 6))
# 选择 TF-IDF 总分最高的 30 个词来展示
word_totals = tfidf_matrix.sum(axis=0)
top_word_indices = np.argsort(word_totals)[::-1][:30]
top_words_viz = [idx_to_word[i] for i in top_word_indices]
top_tfidf = tfidf_matrix[:, top_word_indices]
im = ax.imshow(top_tfidf.T, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(CORPUS)))
ax.set_xticklabels([f"D{i+1}" for i in range(len(CORPUS))], rotation=45, fontsize=8)
ax.set_yticks(range(len(top_words_viz)))
ax.set_yticklabels(top_words_viz, fontsize=8)
ax.set_xlabel("Document ID", fontsize=12)
ax.set_ylabel("Keywords", fontsize=12)
ax.set_title("TF-IDF Heatmap: Documents x Keywords (Brighter = More Important)", fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='TF-IDF Score')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'tfidf_heatmap_demo.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[TF-IDF 可视化] 热力图已保存至 images/tfidf_heatmap_demo.png")
print()

# ============================================================
# 第三部分：word2vec Skip-gram + 负采样
# ============================================================

# ---------- 3.1 构建训练数据 ----------

def build_skipgram_pairs(
    tokenized_docs: List[List[str]],
    window_size: int = 2,
) -> List[Tuple[int, int]]:
    """
    构建 Skip-gram 训练对：(中心词索引, 上下文词索引)

    参数：
        tokenized_docs: 所有文档的分词列表
        window_size: 上下文窗口大小（单侧）
    返回：
        (中心词, 上下文词) 索引对的列表
    """
    pairs = []
    for doc in tokenized_docs:
        indices = [word_to_idx[w] for w in doc if w in word_to_idx]
        for i, center in enumerate(indices):
            # 遍历窗口内的上下文词
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:  # 不包含中心词自身
                    pairs.append((center, indices[j]))
    return pairs


# 构建训练对
skipgram_pairs = build_skipgram_pairs(tokenized_corpus, window_size=2)
print(f"[Skip-gram] 共生成 {len(skipgram_pairs)} 个训练对")
print()

# 计算词频的 3/4 次方作为负采样分布（word2vec 论文推荐的噪声分布）
word_freq = Counter(all_words)
word_freq_pow = {w: count ** 0.75 for w, count in word_freq.items()}
total_pow = sum(word_freq_pow.values())
noise_dist = np.array([word_freq_pow.get(w, 0) / total_pow for w in vocab])


class SkipGramDataset(Dataset):
    """
    Skip-gram 训练数据集，每个样本为 (中心词, 上下文词, 负样本列表)
    """

    def __init__(self, pairs: List[Tuple[int, int]], num_neg: int = 5, noise_dist: np.ndarray = None):
        """
        参数：
            pairs: (中心词, 上下文词) 列表
            num_neg: 每个正样本配多少个负样本
            noise_dist: 负采样概率分布
        """
        self.pairs = pairs
        self.num_neg = num_neg
        self.noise_dist = noise_dist
        self.V = len(noise_dist) if noise_dist is not None else 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, pos_context = self.pairs[idx]
        # 负采样：从噪声分布中随机采样，排除正样本
        neg_samples = []
        while len(neg_samples) < self.num_neg:
            neg = np.random.choice(self.V, p=self.noise_dist)
            if neg != pos_context and neg != center:
                neg_samples.append(neg)
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(pos_context, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
        )


# ---------- 3.2 Skip-gram 模型 ----------

class SkipGramNegSampling(nn.Module):
    """
    Skip-gram 模型 with Negative Sampling.

    参数：
        vocab_size: 词汇表大小 V
        embed_dim: 词向量维度 d（通常 50~300）
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super().__init__()
        # 输入嵌入矩阵 W (V × d) —— 这就是训练后我们要保留的词向量
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 输出嵌入矩阵 W' (V × d) —— 辅助矩阵，训练后可丢弃
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 参数初始化：小随机值
        self.in_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)
        self.out_embeddings.weight.data.uniform_(-0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, center_words: torch.Tensor, context_words: torch.Tensor, neg_words: torch.Tensor):
        """
        前向计算：负采样损失。

        参数：
            center_words: 中心词索引，shape (batch,)
            context_words: 正样本上下文词索引，shape (batch,)
            neg_words: 负样本词索引，shape (batch, num_neg)
        返回：
            loss: 负采样损失（标量）
        """
        batch_size = center_words.size(0)
        # 查表获取向量
        v_center = self.in_embeddings(center_words)       # (batch, d)
        u_pos = self.out_embeddings(context_words)          # (batch, d)
        u_neg = self.out_embeddings(neg_words)              # (batch, num_neg, d)

        # 正样本得分：v_center · u_pos → sigmoid → log
        pos_score = torch.sum(v_center * u_pos, dim=1)     # (batch,)
        pos_loss = F.logsigmoid(pos_score).mean()           # -log σ(v·u_pos)，取负是因为 log_sigmoid

        # 负样本得分：v_center · u_neg → sigmoid(-score) → log
        neg_score = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1).mean()           # -log σ(-v·u_neg)

        # 总损失 = -(正样本损失 + 负样本损失)
        return -(pos_loss + neg_loss)


# ---------- 3.3 训练 ----------

def train_skipgram(
    model: SkipGramNegSampling,
    dataloader: DataLoader,
    epochs: int = 50,
    lr: float = 0.01,
    device: torch.device = None,
):
    """
    训练 Skip-gram 模型。

    参数：
        model: SkipGramNegSampling 模型实例
        dataloader: 训练数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    返回：
        loss_history: 每个 epoch 的平均损失列表
    """
    if device is None:
        device = DEVICE
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print(f"[Skip-gram 训练] 设备: {device}, Epochs: {epochs}, LR: {lr}")
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (center, pos, neg) in enumerate(dataloader):
            center = center.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            optimizer.zero_grad()
            loss = model(center, pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        loss_history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return loss_history


# 准备训练数据
dataset = SkipGramDataset(skipgram_pairs, num_neg=5, noise_dist=noise_dist)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 创建并训练模型
embed_dim = 64  # 词向量维度
model = SkipGramNegSampling(V, embed_dim=embed_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_history = train_skipgram(model, dataloader, epochs=80, lr=0.005, device=device)

# 训练完成后，提取输入嵌入矩阵作为最终的词向量
word_vectors = model.in_embeddings.weight.data.cpu().numpy()  # (V, embed_dim)
print(f"[Skip-gram] 词向量矩阵形状: {word_vectors.shape}")
print()

# ---------- 3.4 训练损失曲线 ----------

plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='#2196F3', linewidth=1.5)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Skip-gram Negative Sampling Training Loss Curve", fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'skipgram_loss_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[可视化] 训练损失曲线已保存至 images/skipgram_loss_curve.png")

# ============================================================
# 第四部分：词向量分析
# ============================================================

# ---------- 4.1 t-SNE 降维可视化 ----------

from sklearn.manifold import TSNE

print("[t-SNE] 正在进行降维可视化...（可能需要几秒）")
# 选择出现频率最高的 100 个词进行可视化
word_counts = Counter(all_words)
top_words = [w for w, _ in word_counts.most_common(100) if w in word_to_idx]
top_indices = [word_to_idx[w] for w in top_words]
top_vectors = word_vectors[top_indices]

# t-SNE 降维到 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(top_words) - 1), max_iter=500)
vectors_2d = tsne.fit_transform(top_vectors)

plt.figure(figsize=(16, 14))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='steelblue', alpha=0.6, s=50)
# 标注每个词的标签
for i, word in enumerate(top_words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                 fontsize=9, alpha=0.85,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.title("word2vec Embedding t-SNE Visualization (Top-100 Frequent Words)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(_IMAGES, 'word2vec_tsne.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[t-SNE] 可视化已保存至 images/word2vec_tsne.png")
print()

# ---------- 4.2 近义词查询 ----------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。

    参数：
        v1, v2: 两个同维度向量
    返回：
        余弦相似度，范围 [-1, 1]
    """
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return dot / norm


def find_nearest_neighbors(
    query_word: str,
    word_vectors: np.ndarray,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    查找与查询词最相似的 top_k 个词。

    参数：
        query_word: 查询词
        word_vectors: 词向量矩阵 (V, d)
        word_to_idx: 词到索引的映射
        idx_to_word: 索引到词的映射
        top_k: 返回最近邻数量
    返回：
        [(词, 余弦相似度), ...] 列表
    """
    if query_word not in word_to_idx:
        return []
    query_idx = word_to_idx[query_word]
    query_vec = word_vectors[query_idx]
    # 计算与所有词的余弦相似度
    similarities = []
    for i in range(len(word_vectors)):
        if i != query_idx:
            sim = cosine_similarity(query_vec, word_vectors[i])
            similarities.append((idx_to_word[i], sim))
    # 按相似度降序排列
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# 演示：查询若干个词的近邻
query_words = ["学", "球", "医", "教", "机", "经"]
print("=" * 60)
print("[近义词查询] word2vec 余弦相似度 Top-5")
print("=" * 60)
for qw in query_words:
    if qw in word_to_idx:
        neighbors = find_nearest_neighbors(qw, word_vectors, word_to_idx, idx_to_word, top_k=5)
        print(f"  「{qw}」的近邻: {', '.join([f'{w}({s:.3f})' for w, s in neighbors])}")
    else:
        print(f"  「{qw}」不在词汇表中")
print()

# ---------- 4.3 类比推理 ----------

def word_analogy(
    a: str, b: str, c: str,
    word_vectors: np.ndarray,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    词类比推理：a - b + c ≈ ?  (如 国王 - 男人 + 女人 ≈ 女王)

    参数：
        a, b, c: 三个类比词
        word_vectors: 词向量矩阵
        word_to_idx, idx_to_word: 词-索引映射
        top_k: 返回 top-k 结果
    返回：
        [(词, 余弦相似度), ...] 列表
    """
    if a not in word_to_idx or b not in word_to_idx or c not in word_to_idx:
        return []
    # 计算类比向量
    result_vec = word_vectors[word_to_idx[a]] - word_vectors[word_to_idx[b]] + word_vectors[word_to_idx[c]]
    # 排除 a, b, c 本身
    exclude = {word_to_idx[a], word_to_idx[b], word_to_idx[c]}
    # 计算所有词的余弦相似度
    similarities = []
    for i in range(len(word_vectors)):
        if i not in exclude:
            sim = cosine_similarity(result_vec, word_vectors[i])
            similarities.append((idx_to_word[i], sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# 注：由于语料库较小，类比推理效果有限。这里演示方法。
# 在大规模语料上，king-man+woman≈queen 这种效果才会显著。
print("=" * 60)
print("[类比推理演示] word2vec 向量运算")
print("=" * 60)

# 尝试几个类比：由于语料小，结果可能不好，但展示方法
analogy_triples = [
    ("足", "篮", "游"),   # 足球 - 篮球 + 游泳 ≈ ?
]
for a, b, c in analogy_triples:
    results = word_analogy(a, b, c, word_vectors, word_to_idx, idx_to_word, top_k=5)
    if results:
        print(f"  {a} - {b} + {c} ≈  ?")
        for word, sim in results:
            print(f"    → {word} (相似度: {sim:.4f})")
print()

# ---------- 4.4 TF-IDF vs word2vec 相似度对比 ----------

print("=" * 60)
print("[对比] TF-IDF vs word2vec 词相似度")
print("=" * 60)

# 用 TF-IDF 计算"文档"间的相似度（不是词之间的）
# word2vec 可以计算词之间的相似度
# 这展示了两种方法的本质区别

# TF-IDF 文档相似度
def tfidf_cosine_similarity(doc1_idx: int, doc2_idx: int, tfidf_matrix: np.ndarray) -> float:
    """计算两篇文档的 TF-IDF 向量余弦相似度"""
    v1 = tfidf_matrix[doc1_idx]
    v2 = tfidf_matrix[doc2_idx]
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return dot / norm


# 显示部分文档之间的 TF-IDF 相似度
print("\nTF-IDF 文档相似度矩阵 (部分):")
for i in range(0, 10, 2):
    for j in range(i + 1, min(i + 3, len(CORPUS))):
        sim = tfidf_cosine_similarity(i, j, tfidf_matrix)
        print(f"  Doc{i+1} vs Doc{j+1}: {sim:.4f}  |  {CORPUS[i][:20]}... <-> {CORPUS[j][:20]}...")

print("\nword2vec 词级别相似度（可以计算任意两个词的相似度）:")
# 对比"足"与"篮"（体育类应相似）vs "足"与"医"（不相关）
for w1, w2 in [("足", "篮"), ("学", "习"), ("足", "医"), ("机", "器")]:
    if w1 in word_to_idx and w2 in word_to_idx:
        sim = cosine_similarity(word_vectors[word_to_idx[w1]], word_vectors[word_to_idx[w2]])
        print(f"  sim('{w1}', '{w2}') = {sim:.4f}")

print()
print("=" * 60)
print("[核心对比总结]")
print("=" * 60)
print("  TF-IDF:   基于统计计数的稀疏表示，忽略词序，适合文档级任务")
print("  word2vec: 基于上下文的稠密表示，捕获语义，适合词级任务")
print("  TF-IDF 给出「文档向量」用于文档检索/分类")
print("  word2vec 给出「词向量」用于近义词查询/类比推理")
print()
print("所有 demo 运行完成！图表已保存至 images/ 目录。")
