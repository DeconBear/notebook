---
title: "s17 预训练范式 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s17 预训练范式 — demo.py 代码详解

<a href="/notebook/code/applied/nlp/pretrained/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/applied/nlp/pretrained/code
python demo.py
```

**依赖**：`torch`, `transformers`, `matplotlib`

**首次运行**：会自动从 HuggingFace Hub 下载模型文件。使用的模型包括：
- `prajjwal1/bert-tiny`（最小 BERT，约 4MB，2 层 128 维）
- `bert-base-chinese`（BERT 中文模型，用于 MLM 演示）
- `uer/gpt2-chinese-cluecorpussmall`（中文 GPT-2）

---

## 代码逐段详解

### 第1步：核心导入 — `transformers` 库的关键类

```python
from transformers import (
    AutoTokenizer,                    # 自动选择对应模型的 tokenizer
    AutoModelForSequenceClassification,  # BERT + 分类头
    AutoModelForMaskedLM,             # BERT MLM（掩码预测）
    AutoModelForCausalLM,             # GPT 自回归语言模型
    Trainer,                          # HuggingFace 的高层训练 API
    TrainingArguments,                # 训练超参数配置
    pipeline,                         # 一行代码完成推理的便捷 API
)
```

**`AutoTokenizer` vs 专用 Tokenizer**：`AutoTokenizer.from_pretrained("bert-base-chinese")` 会自动识别模型类型并加载对应的 tokenizer。不需要手动指定 `BertTokenizer` 或 `GPT2Tokenizer`。

**`AutoModelFor*` 系列**：HuggingFace 的 "Auto" 类会根据 checkpoint 名称自动选择正确的模型架构。`AutoModelForSequenceClassification` 会在预训练 BERT 顶部自动添加一个分类头（pooler + dropout + linear）。

---

### 第2步：BERT 文本分类微调

#### 2.1 数据格式

```python
train_data = [
    ("这个产品质量非常好，我很满意", 1),  # 正面评论 → 标签 1
    ("客服态度恶劣，完全不解决问题", 0),  # 负面评论 → 标签 0
    ...
]
```

每条数据是一个 (文本, 标签) 对。标签 1=正面，0=负面。训练集 24 条，验证集 6 条——这是典型的微调场景：**标注数据很少，但预训练模型已经"懂"语言，只需少量数据即可适应特定任务**。

#### 2.2 Tokenizer 编码

```python
class SentimentDataset(Dataset):
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            text,
            truncation=True,           # 超过 max_len 则截断
            padding='max_length',      # 不足 max_len 则填充到 max_len
            max_length=128,
            return_tensors='pt',       # 返回 PyTorch 张量
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
```

**tokenizer 输出的三个关键字段**：

| 字段 | 形状 | 含义 |
|------|------|------|
| `input_ids` | (batch, max_len) | 每个 token 的词汇表索引，包括 `[CLS]`, `[SEP]`, `[PAD]` |
| `attention_mask` | (batch, max_len) | 1=真实 token，0=填充 token。Attention 计算时忽略 0 的位置 |
| `token_type_ids` | (batch, max_len) | 0=句子 A，1=句子 B（单句分类时全为 0，此处未使用） |

**`truncation=True, padding='max_length'`**：保证所有样本的输入长度一致（都是 max_len），这是批处理的要求。截断丢弃超出部分，填充补齐不足部分。

#### 2.3 HuggingFace Trainer — 高层训练 API

```python
training_args = TrainingArguments(
    output_dir="./bert_sentiment_checkpoints",
    num_train_epochs=4,                        # 微调只需少量 epoch
    per_device_train_batch_size=4,
    eval_strategy="epoch",                     # 每个 epoch 评估一次
    load_best_model_at_end=True,               # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",
    report_to="none",                          # 不上传到 wandb
)

trainer = Trainer(
    model=bert_cls,                            # 预训练 BERT + 分类头
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,           # 自定义评估函数
)
trainer.train()
```

**为什么微调只需 2-4 个 epoch？** 预训练模型已经学到了通用的语言知识（语法、语义、常识），微调只需将这些知识"调整"到特定任务。epoch 数过多反而会导致过拟合——模型会"忘记"预训练学到的通用知识（catastrophic forgetting）。

**`Trainer` 的设计哲学**：HuggingFace 的 Trainer 封装了训练循环、梯度累积、混合精度训练、分布式训练、日志记录、模型保存等底层细节。对于标准的微调任务，只需配置参数即可，无需手写训练循环。

#### 2.4 预测新样本

```python
for text in test_texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_cls(**inputs).logits          # (1, 2) — 正/负类的未归一化得分
    probs = F.softmax(logits, dim=-1)               # (1, 2) — 转为概率
    pred = torch.argmax(logits, dim=-1).item()      # 0 或 1
```

**`**inputs` 字典解包**：将 `{'input_ids': ..., 'attention_mask': ...}` 作为关键字参数传入模型。等价于 `bert_cls(input_ids=..., attention_mask=...)`。

**softmax 将 logits 转为概率**：`probs = F.softmax(logits, dim=-1)` 得到 `[P(负面), P(正面)]`。

---

### 第3步：BERT MLM — 掩码预测演示

MLM（Masked Language Model）是 BERT 预训练的核心任务——随机遮盖部分 token，让模型从上下文预测被遮盖的词。

#### 3.1 使用 Pipeline API

```python
mlm_pipeline = pipeline(
    "fill-mask",                    # 任务类型：掩码填充
    model=mlm_model,                # 预训练的 BERT MLM 模型
    tokenizer=mlm_tokenizer,
    device=0 if DEVICE.type == 'cuda' else -1,
)
```

**`pipeline("fill-mask", ...)`** 将模型加载、tokenization、前向传播、结果解析封装为一个函数调用。输入带 `[MASK]` 的文本，输出最可能的填充词。

#### 3.2 预测被遮盖的词

```python
mlm_examples = [
    "今天天气真[MASK]，适合出去郊游。",
    "这个手机拍照效果很[MASK]，我非常满意。",
    "深度学习是人工智能的一个重要[MASK]。",
]
for text in mlm_examples:
    results = mlm_pipeline(text, top_k=3)
    # results[0] = {'score': 0.85, 'token_str': '好', 'sequence': '今天天气真好...'}
```

**MLM 的威力**：BERT 能准确预测不同上下文中的 `[MASK]`——在"天气真[MASK]"的上下文中填"好"，在"效果很[MASK]"的上下文中填"好"或"棒"。这展示了 BERT **双向理解**的能力——它能同时利用左右两侧的上下文信息。

**注意 `[MASK]` token 的特殊性**：`[MASK]` 是 BERT 词表中一个特殊的 token（id=103）。预训练期间模型学会了：当看到 `[MASK]` 时，需要预测其原始词汇。但在微调阶段，输入中没有 `[MASK]`——BERT 使用了 80%-10%-10% 的替换策略来弥合这个 gap。

---

### 第4步：GPT-2 文本生成 — 对比 BERT

#### 4.1 为什么 BERT 不能生成文本？

BERT 是 Encoder-only 架构，使用**双向自注意力**——每个 token 可以同时看到左右的 token。这意味着 BERT 无法按顺序一个接一个地生成 token（因为它天然需要"看到全部"才能做预测）。

GPT 是 Decoder-only 架构，使用**因果自注意力**（causal mask）——每个 token 只能看到它之前的 token。这让 GPT 天然支持自回归生成：给定前文，预测下一个 token，然后将其追加到序列中，重复此过程。

#### 4.2 GPT-2 生成参数

```python
outputs = gpt_model.generate(
    **inputs,
    max_new_tokens=30,            # 最多生成 30 个新 token
    temperature=0.8,              # 温度：<1 更确定，>1 更随机
    do_sample=True,               # 采样而非贪心解码
    top_p=0.9,                    # nucleus sampling：累积概率阈值
    repetition_penalty=1.1,       # >1 抑制重复，<1 鼓励重复
    pad_token_id=gpt_tokenizer.pad_token_id,
)
```

**`top_p`（Nucleus Sampling）**：从概率最高的 token 开始累加，当累积概率达到 `top_p`（如 0.9）时停止，只从这组 token 中采样。与 top-k 采样相比，top-p 能根据概率分布动态调整候选集大小。

**`repetition_penalty`**：对已经出现过的 token 施加惩罚（logits 降低），防止模型陷入重复循环（如"我爱你我爱你我爱你..."）。值 >1 表示惩罚重复。

**`do_sample=True`**：使用概率采样而非贪心解码。如果 `do_sample=False`，等价于 temperature=0（每次选概率最高的 token），生成结果是确定性的，缺乏多样性。

---

### 第5步：上下文嵌入 — BERT vs Word2Vec

代码通过一个巧妙的实验展示 BERT 的核心优势——**上下文相关的嵌入**：

```python
test_sentences = [
    "我喜欢吃苹果，特别是红富士苹果",       # 两句中的"苹果"都是水果→相似
    "苹果公司发布了最新的iPhone手机",       # 这句中的"苹果"是公司→与水果不同
    "我在超市买了三个苹果",
    "苹果的股价今天上涨了百分之五",
]
```

**Word2Vec 的问题**：无论"苹果"出现在什么上下文中，其词向量完全相同。模型无法区分"吃苹果（水果）"和"苹果公司（科技公司）"。

**BERT 的解决方案**：BERT 的嵌入是**上下文相关**的——同一个词在不同句子（或同一句的不同位置）中有不同的向量表示。代码通过计算同一句中两个"苹果"的余弦相似度来验证：当其处于相同语义上下文时（都是水果），嵌入相似度高；跨语义上下文时，嵌入会不同。

**计算余弦相似度**：
$$
\text{cosine\_similarity}(v_1, v_2) = \frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}
$$

`F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))` 计算两个向量的夹角余弦，值域 [-1, 1]。1 表示完全相同，0 表示正交，-1 表示完全相反。

---

### 第6步：回退机制 — 当模型下载失败时

代码中包含了健壮的回退逻辑：如果 HuggingFace 模型下载失败（如无网络），会创建一个 `TinyFallbackClassifier`（微型 Transformer 分类器），确保 demo 在任何环境下都能运行。

```python
class TinyFallbackClassifier(nn.Module):
    def __init__(self, vocab_size=1000, num_labels=2):
        self.embedding = nn.Embedding(vocab_size, 32)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32, nhead=2, ...),
            num_layers=2
        )
        self.classifier = nn.Linear(32, num_labels)
```

这是一个简单的 Encoder-only 模型（类似于微型 BERT），用于演示微调流程。虽然效果远不如真正的 BERT，但它能让你理解"预训练模型 + 分类头 → 微调"的完整 pipeline。

---

## 关键概念速查表

| 概念 | 公式/描述 | 关键点 |
|------|----------|--------|
| MLM (掩码语言模型) | 预测 `[MASK]` 位置的原词 | BERT 的双向理解能力来源 |
| CLM (因果语言模型) | $P(x_t|x_{<t})$ | GPT 的自回归生成基础 |
| Tokenizer | 文本→input_ids, attention_mask | 分词+编码+填充+截断 |
| `[CLS]` token | 句子级别的聚合表示 | BERT 分类任务用它的向量 |
| `[SEP]` token | 句子分隔符 | 分隔不同的句子/段落 |
| `[MASK]` token | 被遮盖的 token | MLM 的预测目标 |
| attention_mask | 1=真实 token, 0=padding | 让注意力忽略填充位置 |
| 微调 epoch 数 | 通常 2-4 | 预训练模型只需少量调整 |
| top-p sampling | 累积概率阈值 | 动态确定候选 token 集合 |
| repetition_penalty | 降低重复 token 的 logits | 防止生成循环重复文本 |
| 上下文嵌入 | 同一词在不同上下文中向量不同 | BERT 解决多义词问题 |
| Pipeline API | `pipeline("fill-mask", model)` | 一行代码完成推理 |

---


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/applied/nlp/pretrained/code/demo.py`
