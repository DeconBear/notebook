---
title: "s14 文本表示 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s14 文本表示 — exercise.py 练习指南

<a href="/notebook/code/nlp/text-representation/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写 TF-IDF 和 word2vec 的核心算法，深入理解文本表示的两种范式：

1. **实现 IDF 计算** —— 理解逆文档频率如何量化词的区分能力
2. **实现负采样损失** —— 理解 Skip-gram 训练的核心损失函数
3. **实现余弦相似度** —— 向量相似度的标准度量

## 预备知识

- **TF-IDF**：$\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)$
  - $\text{TF}(w, d) = \frac{c(w,d)}{\sum_{w'} c(w',d)}$
  - $\text{IDF}(w) = \log\frac{N}{df(w)}$
- **Skip-gram**：给定中心词 $w_t$，预测上下文词 $w_{t+j}$
- **负采样损失**：$\mathcal{L} = -\log\sigma(v_{w_t}\cdot u_{w_c}) - \sum_{i=1}^{K}\log\sigma(-v_{w_t}\cdot u_{w_i})$
- **余弦相似度**：$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$

## 任务清单

### 练习 1：实现 TF-IDF 的 IDF 计算

**任务**：在 `compute_idf(tokenized_docs)` 中实现 IDF 计算。

**公式**：

$$
\text{IDF}(w) = \log\frac{N}{df(w)}
$$

其中：
- $N$：总文档数
- $df(w)$：包含词 $w$ 的文档数（Document Frequency，不是词频）

**步骤提示**：

```
1. 统计每个词出现在多少篇文档中（每篇文档内部去重）
2. 对每个词，IDF = log(N / df(w))
3. 可选平滑：IDF = log((N+1) / (df(w)+1)) + 1
```

**代码框架**：

```python
def compute_idf(tokenized_docs):
    N = len(tokenized_docs)
    df = {}
    for doc in tokenized_docs:
        for word in set(doc):  # 去重：每篇文档每个词只计一次
            df[word] = df.get(word, 0) + 1

    idf = {}
    for word, doc_freq in df.items():
        idf[word] = math.log((N + 1) / (doc_freq + 1)) + 1  # 平滑版
    return idf
```

**关键细节：`set(doc)` 的去重作用**：

考虑词"学习"出现在 3 篇文档中：
- 在文档 1 中出现 5 次
- 在文档 3 中出现 2 次
- 在文档 5 中出现 1 次

`df("学习") = 3`（它出现在 3 篇文档中），不是 $5+2+1=8$。IDF 关心的是**文档频率**（多少篇文档），不是**词频**（出现了多少次）。

**测试用例**：

```python
test_docs = [
    ["机器", "学习", "是", "人工智能", "的", "分支"],
    ["深度", "学习", "需要", "大量", "数据"],
    ["机器", "学习", "模型", "需要", "训练"],
]
# 预期："的"的IDF最高（只出现在1篇文档中）
#        "学习"的IDF最低（出现在3篇文档中）
```

### 练习 2：实现负采样损失函数

**任务**：在 `negative_sampling_loss(v_center, u_pos, u_neg)` 中实现 Skip-gram 负采样损失。

**公式**：

$$
\mathcal{L} = -\log \sigma(v_{\text{center}} \cdot u_{\text{pos}}) - \sum_{i=1}^{K} \log \sigma(-v_{\text{center}} \cdot u_{\text{neg}, i})
$$

其中：
- $v_{\text{center}}$：中心词的输入向量
- $u_{\text{pos}}$：正样本上下文词的输出向量
- $u_{\text{neg}, i}$：第 $i$ 个负样本词的输出向量
- $\sigma$：sigmoid 函数 $\sigma(x) = \frac{1}{1+e^{-x}}$

**为什么用 `F.logsigmoid` 而不是 `F.log(F.sigmoid(...))`？** `log_sigmoid` 在数值上更稳定。当 $x$ 很小时，$\sigma(x) \to 0$，$\log\sigma(x) \to -\infty$，直接计算会导致 NAN。`logsigmoid` 在实现中用 $\log(1+e^{-x})^{-1} = -\log(1+e^{-x})$ 避免了数值溢出。

**步骤提示**：

```
1. 计算正样本得分: pos_score = sum(v_center * u_pos, dim=1)  → (batch,)
2. 正样本损失: pos_loss = F.logsigmoid(pos_score).mean()     → 标量
3. 计算负样本得分: neg_score = (u_neg @ v_center.unsqueeze(-1)).squeeze(-1)  → (batch, K)
4. 负样本损失: neg_loss = F.logsigmoid(-neg_score).sum(dim=1).mean()  → 标量
5. 总损失: loss = -(pos_loss + neg_loss)
```

**代码框架**：

```python
def negative_sampling_loss(v_center, u_pos, u_neg):
    # 正样本：希望 v·u_pos 很大 → σ(v·u_pos) ≈ 1 → log loss 小
    pos_score = torch.sum(v_center * u_pos, dim=1)        # (batch,)
    pos_loss = F.logsigmoid(pos_score).mean()              # mean (batch,)

    # 负样本：希望 v·u_neg 很小 → σ(-v·u_neg) ≈ 1 → log loss 小
    neg_score = torch.bmm(u_neg, v_center.unsqueeze(2)).squeeze(2)  # (batch, K)
    neg_loss = F.logsigmoid(-neg_score).sum(dim=1).mean()           # sum over K, mean over batch

    return -(pos_loss + neg_loss)
```

**损失函数的直觉**：
- 好的中心词向量 $v$ 应该与正样本上下文词向量 $u_+$ 的内积**很大**（$\sigma \to 1$，$\log\sigma \to 0$）
- 同时与负样本词向量 $u_-$ 的内积**很小**（$\sigma(-x) \to 1$，$\log\sigma(-x) \to 0$）

这本质上是在训练向量空间中：拉近与相关词的距离，推远与不相关词的距离。

**`torch.bmm` 维度说明**：
- `u_neg` 形状：`(batch, K, d)`
- `v_center.unsqueeze(2)` 形状：`(batch, d, 1)`
- `torch.bmm(u_neg, v_center.unsqueeze(2))` 做了 `(batch, K, d) × (batch, d, 1)` = `(batch, K, 1)`
- `.squeeze(2)` 去掉最后一维：`(batch, K)`

这计算了每个 batch 样本中，K 个负样本与中心词的批量内积。

### 练习 3：实现余弦相似度

**任务**：在 `cosine_similarity(v1, v2)` 中实现余弦相似度计算。

**公式**：

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \sqrt{\sum_i b_i^2}}
$$

**步骤提示**：

```
1. dot = np.dot(v1, v2)                     # 分子: 点积
2. norm = np.linalg.norm(v1) * np.linalg.norm(v2)  # 分母: 范数之积
3. 如果 norm == 0: 返回 0.0                    # 防止除零
4. 返回 dot / norm
```

**代码框架**：

```python
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return dot / norm
```

**测试用例**：

| v1 | v2 | 预期 cos | 直观含义 |
|-----|-----|---------|----------|
| `[1,2,3]` | `[1,2,3]` | 1.0 | 完全相同 |
| `[1,2,3]` | `[-1,-2,-3]` | -1.0 | 完全相反 |
| `[1,2,3]` | `[1,0,0]` | $\approx$ 0.267 | 部分重叠 |

**余弦相似度的几何理解**：

- `cos = 1`：两个向量指向完全相同方向（但长度可能不同）
- `cos = 0`：两个向量正交（互不相关）
- `cos = -1`：两个向量指向完全相反方向

对于词向量，$v_{\text{足球}}$ 和 $v_{\text{篮球}}$ 的余弦相似度通常在 0.6-0.9，而 $v_{\text{足球}}$ 和 $v_{\text{医学}}$ 的余弦相似度接近 0。

**为什么余弦相似度在 NLP 中比欧氏距离更好？** 词向量的长度与词频相关——高频词的向量更长。如果用欧氏距离，高频词的空间位置影响了所有距离计算。余弦相似度只关心方向，消除了频率偏差，更准确地反映了语义相似性。

## 完整代码

<<< @/nlp/text-representation/code/exercise.py
