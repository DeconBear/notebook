---
title: "s17 预训练范式 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s17 预训练范式 — exercise.py 练习指南

<a href="/notebook/code/applied/nlp/pretrained/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过动手实现 BERT 分类头、MLM 遮盖策略和多义词嵌入对比，建立对预训练-微调范式的实际理解。完成后你将能够：

1. 在任意预训练 BERT 上添加分类头并计算损失
2. 理解并实现 MLM 的 80%-10%-10% 遮盖策略
3. 理解 BERT 上下文嵌入如何解决多义词问题

## 预备知识

- **BERT 架构**：Encoder-only，双向自注意力，输入三种嵌入的和（Token + Segment + Position）
- **BERT 的特殊 token**：`[CLS]=101`, `[SEP]=102`, `[MASK]=103`, `[PAD]=0`
- **MLM 遮盖策略**：选中 15% 的 token → 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变
- **为什么不全用 `[MASK]`**：防止预训练-微调不匹配（微调时输入中没有 `[MASK]` token）

---

## 任务清单

### 练习 1：为 BERT 构建分类头并计算损失

**目标**：补全 `BertForClassification` 类，在预训练 BERT 之上添加分类头。

**核心思路**：BERT 输出的 `[CLS]` token 的隐藏向量聚合了整个句子的信息。分类头的任务是将这个向量映射为类别 logits。

**BERT 前向流程**：
```
输入文本 → Tokenizer → input_ids (batch, seq_len)
  → BERT → last_hidden_state (batch, seq_len, hidden_size)
  → 取 [CLS] = last_hidden_state[:, 0, :] → (batch, hidden_size)
  → 分类头 Linear(hidden_size, num_labels) → logits (batch, num_labels)
  → CrossEntropyLoss(logits, labels) → 标量损失
```

**TODO 步骤**：

**步骤 A — 创建分类头**：
```python
class BertForClassification(nn.Module):
    def __init__(self, bert_backbone, num_labels):
        self.bert = bert_backbone
        hidden_size = self.bert.config.hidden_size   # 通常 768
        self.classifier = nn.Linear(hidden_size, num_labels)  # ← TODO
```

**步骤 B — 提取 [CLS] 向量**：
```python
outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
# outputs.last_hidden_state: (batch, seq_len, hidden_size)
cls_embedding = outputs.last_hidden_state[:, 0, :]   # ← TODO: 第一个 token
```

`[:, 0, :]` 的含义：取所有 batch、第 0 个 token（即 `[CLS]`）、所有 hidden_size 维度。

**步骤 C — 分类与损失计算**：
```python
logits = self.classifier(cls_embedding)                # ← TODO: (batch, num_labels)
if labels is not None:
    loss = F.cross_entropy(logits, labels)              # ← TODO
    return loss, logits
```

**为什么用 `[CLS]` 而不是平均所有 token？** `[CLS]` 是一个特殊的聚合 token，它在预训练期间（NSP 任务）被训练来编码整个输入序列的语义。Attention 机制让 `[CLS]` 能够从所有其他 token 中收集信息。

**预期理解**：`[CLS]`向量 → Linear(hidden, num_labels) → CrossEntropy。这就是 BERT 微调的标准范式。

```
[练习1] BERT 分类头结构已定义
        核心步骤: [CLS]向量 → Linear(hidden, num_labels) → CrossEntropy
```

---

### 练习 2：实现 MLM 的随机遮盖

**目标**：补全 `random_mask_tokens()` 函数，实现 BERT 预训练中使用的 MLM 遮盖策略。

**遮盖策略总结**：
- 随机选中 15% 的 token（不包括特殊 token）
- 选中的 token 中：
  - **80%** 替换为 `[MASK]` — 让模型学会从上下文推断
  - **10%** 替换为随机 token — 防止模型"看到 [MASK] 就无脑复制"
  - **10%** 保持原样 — 让模型学会"这个位置可能不需要改变"
- 特殊 token（`[CLS]`, `[SEP]`, `[PAD]`）永不被遮盖

**TODO 步骤**：

```python
def random_mask_tokens(input_ids, vocab_size, mask_token_id,
                       special_tokens=None, mask_prob=0.15):
    labels = torch.full_like(input_ids, -100)        # -100 在 CrossEntropyLoss 中被忽略
    masked_input_ids = input_ids.clone()

    # 步骤 1: 确定哪些位置需要被遮盖
    prob_matrix = torch.rand(input_ids.shape)        # (batch, seq_len) 的 [0,1) 随机数
    # 需要遮盖的位置 = 随机概率 < mask_prob 且 不是特殊 token
    mask_positions = (prob_matrix < mask_prob)
    for sp in special_tokens:
        mask_positions = mask_positions & (input_ids != sp)

    # 步骤 2: 对每个被选中的位置，按 80-10-10 策略替换
    for i in range(input_ids.size(0)):
        for j in range(input_ids.size(1)):
            if mask_positions[i, j]:
                labels[i, j] = input_ids[i, j]          # 记录真实 token
                rand_val = torch.rand(1).item()
                if rand_val < 0.8:                       # 80% → [MASK]
                    masked_input_ids[i, j] = mask_token_id
                elif rand_val < 0.9:                     # 10% → 随机 token
                    masked_input_ids[i, j] = torch.randint(
                        0, vocab_size, (1,)
                    ).item()
                # else: 10% 保持原样（不修改）

    return masked_input_ids, labels
```

**关键设计**：`labels` 中非遮盖位置的值为 `-100`，这是 PyTorch `CrossEntropyLoss` 的特殊约定——当 `ignore_index=-100` 时，对应位置的损失不参与计算。这样模型只在被遮盖的位置上计算损失，与 BERT 预训练的做法一致。

**预期输出**：
```
[练习2] MLM 遮盖测试:
  原始:     [101, 123, 456, 789, 102, 0, 0]
  遮盖后:   [101, 103, 456, 789, 102, 0, 0]  (示例)
  标签:     [-100, 123, -100, -100, -100, -100, -100]
  注意: 101([CLS]), 102([SEP]), 0([PAD]) 不应被遮盖
```

---

### 练习 3：比较多义词在不同上下文中的 BERT 嵌入

**目标**：理解 BERT 的上下文嵌入如何解决 Word2Vec 等静态嵌入的多义词问题。

**核心概念**：Word2Vec/GloVe 等静态嵌入给每个词一个固定的向量——"苹果"只有一个向量表示，无论它出现在"吃苹果"还是"苹果公司"的上下文中。BERT 的不同在于：同一个词的嵌入会随上下文动态变化。

**TODO 步骤**：

```python
def compare_polysemous_embeddings(sentence_pairs, get_bert_embedding):
    similarities = []
    for sent1, sent2 in sentence_pairs:
        emb1 = get_bert_embedding(sent1)   # 获取句子 1 中目标词的 BERT 嵌入
        emb2 = get_bert_embedding(sent2)   # 获取句子 2 中目标词的 BERT 嵌入
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        similarities.append(sim)
    return similarities
```

**预期现象**：
- "苹果很好吃"中的"苹果" vs "苹果发布了新手机"中的"苹果" → 余弦相似度**低**（不同语义）
- "苹果很好吃"中的"苹果" vs "红富士苹果很甜"中的"苹果" → 余弦相似度**高**（相同语义）
- "苹果很好吃"中的"苹果" vs "香蕉很好吃"中的"香蕉" → 可能比前一对的相似度**更高或更低**，取决于模型是否学到了"苹果和香蕉都是水果"这种类别知识

**这为什么重要？** 多义词处理是 NLP 的核心挑战之一。BERT 的上下文嵌入让下游任务（情感分析、问答等）能区分"bank（银行）"和"bank（河岸）"——这对理解句子含义至关重要。

```
[练习3] 多义词嵌入对比（需要实际的 BERT 模型才能测试）
         预期: '苹果很好吃'中的'苹果' ≠ '苹果发布了新手机'中的'苹果'
```

---

## 三个练习的关系

| 练习 | 对应概念 | BERT 预训练/微调中的位置 |
|------|---------|----------------------|
| 练习 1: 分类头 | BERT 微调 | 在预训练模型顶部加任务相关层 |
| 练习 2: MLM 遮盖 | BERT 预训练 | 预训练阶段的核心训练目标 |
| 练习 3: 上下文嵌入 | BERT 表示能力 | BERT 的独特优势 |

## 检查要点

运行 `python exercise.py`，确认：
- [ ] 练习 1 分类头结构正确（Linear + CrossEntropy）
- [ ] 练习 2 特殊 token 不被遮盖，遮盖位置按 80-10-10 策略处理
- [ ] 练习 3 理解同一词在不同上下文中的向量差异

完成练习后，返回 demo.py 观察这些概念在完整 pipeline 中的应用——BERT 分类微调（练习 1 的概念）、MLM 演示（练习 2 的概念）和上下文嵌入对比（练习 3 的概念）。

## 完整代码

<<< @/applied/nlp/pretrained/code/exercise.py
