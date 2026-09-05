---
title: "s14 文本表示 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s14 文本表示 — demo.py 代码详解

<a href="/notebook/code/applied/nlp/text-representation/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/applied/nlp/text-representation/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 —— 每个库是做什么的

```python
import numpy as np                     # TF-IDF 矩阵、向量化计算
import math                            # log 运算（IDF 公式需要）
from collections import Counter        # 词频统计

import torch
import torch.nn as nn                  # Embedding 层（word2vec 输入/输出矩阵）
import torch.nn.functional as F        # logsigmoid（负采样损失）
import torch.optim as optim            # Adam 优化器
from torch.utils.data import Dataset, DataLoader  # Mini-batch 训练

import matplotlib.pyplot as plt        # TF-IDF 热力图、t-SNE 可视化
from sklearn.manifold import TSNE      # 高维词向量 → 2D 可视化
```

| 库 | 在此 demo 中的角色 |
|---|---|
| `numpy` | TF-IDF 矩阵存储和运算 |
| `collections.Counter` | 词频统计 `{"足球": 3, "比赛": 2, ...}` |
| `torch.nn.Embedding` | word2vec 的输入嵌入矩阵 $W$ 和输出嵌入矩阵 $W'$ |
| `torch.nn.functional.logsigmoid` | 负采样损失中的 $\log\sigma(\cdot)$ |
| `sklearn.manifold.TSNE` | 高维词向量降维到 2D 可视化 |

### 第2步：中文语料库 —— 25 篇文档，5 个主题

```python
CORPUS = [
    # 体育类（5篇）
    "足球比赛在工人体育场举行 观众热情高涨",
    "篮球运动员在训练中表现出色 投篮命中率很高",
    ...
    # 科技类（5篇）
    "人工智能技术正在改变各行各业的运作方式",
    ...
    # 教育类（5篇）、经济类（5篇）、医疗类（5篇）
]
```

**为什么用中文？** 对于中文读者，中文语料更直观，可以看到 TF-IDF 如何在中文文本上提取关键词（每个汉字都被视为一个 token）。实际项目中应使用 jieba 等分词工具，此处使用逐字切分是为了简化，避免依赖额外库。

**分词函数**：

```python
def tokenize(text: str) -> List[str]:
    text = text.replace(" ", "")  # 去空格
    return list(text)             # 逐字切分
```

### 第3步：TF-IDF —— 从零实现统计文本表示

TF-IDF 由两个独立的统计量相乘得到。

#### 3.1 TF（词频）：一个词在一篇文档中出现的频率

$$
\text{TF}(w, d) = \frac{c(w, d)}{\sum_{w'} c(w', d)}
$$

```python
def compute_tf(doc_tokens):
    counter = Counter(doc_tokens)  # {"足球": 1, "比赛": 1, ...}
    total = len(doc_tokens)
    return {word: count / total for word, count in counter.items()}
```

**为什么用相对频率而不是绝对频次？** 长文档中所有词的绝对频次都更高。如果不做归一化，长文档的 TF-IDF 向量在数值上会"淹没"短文档，导致文档相似度被文档长度主导而非内容。

#### 3.2 IDF（逆文档频率）：一个词在整个语料库中的"稀有程度"

$$
\text{IDF}(w) = \log\frac{N}{df(w)}
$$

其中 $N$ 是总文档数，$df(w)$ 是包含词 $w$ 的文档数。

```python
def compute_idf(tokenized_docs):
    N = len(tokenized_docs)
    idf = {}
    for doc in tokenized_docs:
        for word in set(doc):  # 每篇文档中每个词只计一次！
            idf[word] = idf.get(word, 0) + 1  # df(w)

    for word in idf:
        idf[word] = math.log((N + 1) / (idf[word] + 1)) + 1  # 平滑版公式
    return idf
```

**为什么用 `set(doc)`？** IDF 关心的是**包含该词的文档数量**，而不是在文档中出现了几次。`df(w)` 是文档频率（Document Frequency），不是词频（Term Frequency）。

**平滑的作用**：
- 原始公式：$\log(N / df(w))$
- 平滑公式：$\log((N+1)/(df(w)+1)) + 1$

平滑避免了当 $df(w) = N$（词在所有文档中出现）时 $\log(1) = 0$ 导致的零权重。平滑后即使"万能词"也有一个小正值。

**IDF 的含义**：

| 词 | 出现在几篇文档 | IDF（约） | 解释 |
|-----|-------------|----------|------|
| "的" | 25 篇（全部） | $\approx 1$ | 高频但无信息量，IDF 低 |
| "足球" | 1 篇 | $\approx 3$ | 稀有词，IDF 高，区分力强 |

#### 3.3 TF-IDF 矩阵构建

```python
# 构建 (N_docs × V) 的 TF-IDF 矩阵
tfidf_matrix = np.zeros((len(CORPUS), V))
for doc_idx, doc_tokens in enumerate(tokenized_corpus):
    tf = compute_tf(doc_tokens)
    for word, tf_val in tf.items():
        word_idx = word_to_idx[word]
        tfidf_matrix[doc_idx, word_idx] = tf_val * idf_scores[word]
```

**矩阵解读**：
- 行 = 文档（25 行），列 = 词汇（~250 列）
- `tfidf_matrix[3, 15]` = 文档 3 中词汇 15 的 TF-IDF 得分
- 大部分元素为 0（每篇文档只包含少量词）——典型的高维稀疏矩阵

**TF-IDF 关键词提取**：对每篇文档取 TF-IDF 得分最高的 3 个词，这些词就是该文档的"关键词"：

```python
for doc_idx, doc in enumerate(CORPUS):
    doc_vec = tfidf_matrix[doc_idx]
    top_indices = np.argsort(doc_vec)[::-1][:3]  # 降序取前3
    top_words = [idx_to_word[i] for i in top_indices]
```

### 第4步：word2vec Skip-gram + 负采样 —— 从零实现

#### 4.1 Skip-gram 训练数据构建

**Skip-gram 的核心思想**：给定中心词 $w_t$，预测它周围的上下文词。

```
窗口大小为 2 的示例:
句子:  "今天 天气 非常 好"
窗口:  [中心=天气, 上下文={今天, 非常, 好}]  →  3 个训练对
```

```python
def build_skipgram_pairs(tokenized_docs, window_size=2):
    pairs = []
    for doc in tokenized_docs:
        indices = [word_to_idx[w] for w in doc]
        for i, center in enumerate(indices):
            # 遍历窗口内的上下文词
            for j in range(max(0, i - window_size),
                           min(len(indices), i + window_size + 1)):
                if i != j:  # 不包含中心词自身
                    pairs.append((center, indices[j]))
    return pairs
```

**为什么 Skip-gram 对罕见词效果好？** CBOW 将上下文词向量取平均后预测中心词，这个平均操作会"抹平"罕见词的独特信息。Skip-gram 直接用中心词预测每个上下文词，每个词对都独立处理，罕见词的表示不会被平均化。

#### 4.2 负采样 —— 让训练变得可行

**为什么需要负采样？** Skip-gram 的输出层是一个大小为 $V$ 的 softmax。词汇表 $V$ 可能有几万到几十万——每一步训练都要做 $V$-way 的 softmax，计算量巨大。

**负采样的巧思**：把"$V$ 分类"变成"$K+1$ 个二分类"（通常 $K=5$）。
- 正样本：$(w_t, w_c)$ —— $w_c$ 是 $w_t$ 的真实上下文
- 负样本：$(w_t, w_{\text{rand}})$ —— 随机采样的 $K$ 个词

**负采样分布**：使用词频的 $3/4$ 次方作为采样概率：

$$
P_n(w) = \frac{\text{freq}(w)^{0.75}}{\sum_{w'} \text{freq}(w')^{0.75}}
$$

```python
word_freq_pow = {w: count ** 0.75 for w, count in word_freq.items()}
```

**为什么是 $3/4$ 次方？** 纯词频分布下，高频词（如"的"、"了"）几乎包揽所有负采样，模型学会了"高频词不是上下文"的偏见。$3/4$ 次方降低了高频词的过度优势，提高了中低频词的采样率，更均衡的负样本分布产生更好的词向量。

#### 4.3 Skip-gram 模型

```python
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim=100):
        # 输入嵌入矩阵 W (V × d) —— 训练后保留为词向量！
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 输出嵌入矩阵 W' (V × d) —— 辅助矩阵，训练后可丢弃
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)
```

**两个嵌入矩阵的区别**：

| 矩阵 | 形状 | 含义 | 作用 |
|------|------|------|------|
| `in_embeddings` | $V \times d$ | 每行是词的**输入向量** $v_w$ | 训练后保留**这个**作为词向量 |
| `out_embeddings` | $V \times d$ | 每行是词的**输出向量** $u_w$ | 辅助训练，通常丢弃 |

**为什么保留 `in_embeddings` 而不是两者的平均？** 两种做法都有人用。保留输入向量是最常见的做法（Gensim 的 word2vec 即如此）。在 GloVe 中会将输入和输出向量求和。

**负采样损失函数**：

$$
\mathcal{L} = -\log \sigma(v_{w_t} \cdot u_{w_c}) - \sum_{i=1}^{K} \log \sigma(-v_{w_t} \cdot u_{w_i})
$$

```python
def forward(self, center_words, context_words, neg_words):
    v_center = self.in_embeddings(center_words)   # (batch, d)
    u_pos = self.out_embeddings(context_words)     # (batch, d)
    u_neg = self.out_embeddings(neg_words)         # (batch, K, d)

    # 正样本：希望 v_center · u_pos 很大 → sigmoid 接近 1 → log loss 小
    pos_score = torch.sum(v_center * u_pos, dim=1)  # (batch,)
    pos_loss = F.logsigmoid(pos_score).mean()       # -log σ(v·u_pos)

    # 负样本：希望 v_center · u_neg 很小 → sigmoid(-score) 接近 1 → log loss 小
    neg_score = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)  # (batch, K)
    neg_loss = F.logsigmoid(-neg_score).sum(dim=1).mean()           # -log σ(-v·u_neg)

    return -(pos_loss + neg_loss)  # 总的负采样损失
```

**损失函数的直觉**：
- 第一项 `F.logsigmoid(pos_score)`：正样本得分高 $\to$ sigmoid 值大 $\to$ log 值接近 0 $\to$ 损失小。即：**鼓励中心词和正确的上下文词相似**。
- 第二项 `F.logsigmoid(-neg_score)`：负样本得分低 $\to$ sigmoid 值大 $\to$ log 值接近 0 $\to$ 损失小。即：**鼓励中心词和随机的非上下文词不相似**。

### 第5步：词向量分析

#### 5.1 t-SNE 降维可视化

```python
# 选择频率最高的 100 个词
top_words = [w for w, _ in word_counts.most_common(100)]

# t-SNE 降维到 2D
tsne = TSNE(n_components=2, perplexity=30)
vectors_2d = tsne.fit_transform(top_vectors)
```

**为什么用 t-SNE 而不是 PCA？** PCA 是线性降维，只保留全局方差最大的方向。t-SNE 是非线性降维，专注于保持局部邻域结构——对词向量来说，我们关心的是"哪些词聚在一起"，这正是 t-SNE 的强项。`perplexity=30` 是平衡局部和全局结构的常用值。

**预期结果**："球"和"篮"（体育类）、"学"和"习"（教育类）在 t-SNE 投影中应该靠得很近。

#### 5.2 近义词查询（余弦相似度）

```python
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

**为什么用余弦相似度而不是欧氏距离？** 词向量的长度（幅度）通常受词频影响——高频词向量更长。余弦相似度只关心方向（角度），消除了词频对相似度判断的干扰。在 NLP 中，$\cos(v_{\text{足球}}, v_{\text{篮球}}) \approx 0.8$ 比 $\|v_{\text{足球}} - v_{\text{篮球}}\| \approx 0.5$ 更有意义。

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} \in [-1, 1]
$$

#### 5.3 类比推理

```python
def word_analogy(a, b, c, word_vectors, word_to_idx, idx_to_word):
    # a - b + c ≈ ?
    result_vec = word_vectors[a_idx] - word_vectors[b_idx] + word_vectors[c_idx]
    # 找与 result_vec 最相似的词
```

**数学原理**：

$$
\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}
$$

向量运算 $\vec{v}_a - \vec{v}_b$ 捕捉了"从 a 到 b 的语义方向"，加上 $\vec{v}_c$ 就是将这个方向应用于 c。

> **注意**：由于本 demo 语料库较小（25 篇文档），类比推理的效果有限。这个向量运算的特性需要在大规模语料（数十亿词）上才显著体现。代码展示了方法本身，在实际项目中可替换为大语料。

### 第6步：TF-IDF vs word2vec —— 全面对比

代码最后通过"文档级 TF-IDF 相似度"和"词级 word2vec 相似度"的对比，展示了两种方法的本质差异。

| 对比维度 | TF-IDF | word2vec |
|---------|--------|----------|
| **表示粒度** | 文档级（一篇文档 = 一个向量） | 词级（每个词 = 一个向量） |
| **向量类型** | 稀疏高维（|V|维，大多为0） | 稠密低维（通常 100-300 维） |
| **语义能力** | 仅统计频率 | 通过上下文学习语义关系 |
| **词序** | 完全忽略 | 通过上下文窗口部分保留 |
| **相似度** | 文档相似度（共享关键词） | 词语义相似度（共享上下文） |
| **典型应用** | 文档检索、关键词提取 | 近义词查询、类比推理、下游模型初始化 |

### 关键概念速查表

| 概念 | 公式 | 代码对应 |
|------|------|---------|
| TF | $c(w,d) / \sum c(w',d)$ | `compute_tf()` |
| IDF | $\log(N / df(w))$ | `compute_idf()` |
| TF-IDF | $\text{TF} \times \text{IDF}$ | `tfidf_matrix[d,i]` |
| Skip-gram | 中心词 → 上下文词 | `build_skipgram_pairs()` |
| 负采样 | $K$ 个随机词 vs 正样本 | `SkipGramDataset` |
| 负采样损失 | $-\log\sigma(v\cdot u_+) - \sum\log\sigma(-v\cdot u_-)$ | `SkipGramNegSampling.forward()` |
| 噪声分布 | $P_n(w) \propto \text{freq}(w)^{0.75}$ | `word_freq_pow` |
| 余弦相似度 | $\mathbf{a}\cdot\mathbf{b} / \|\mathbf{a}\|\|\mathbf{b}\|$ | `cosine_similarity()` |
| 词类比 | $\vec{v}_a - \vec{v}_b + \vec{v}_c \approx \vec{v}_?$ | `word_analogy()` |
| 分布式假设 | 词的含义由上下文决定 | word2vec 的理论基础 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/nlp/text-representation/code/demo.py`
