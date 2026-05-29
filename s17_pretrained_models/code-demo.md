---
title: "s17 预训练范式 — demo.py"
---

# s17 预训练范式 — demo.py

<a href="../code/s17_pretrained_models/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

使用 HuggingFace transformers 库演示预训练模型的核心用法和本质差异：BERT 文本分类微调（中文情感二分类，展示"理解"能力）、BERT 掩码预测（MLM 能力展示）、GPT-2 文本生成（自回归生成，展示"创作"能力）。通过三个任务直观对比 Encoder-only (BERT) 和 Decoder-only (GPT) 架构在能力上的根本区别，并展示 BERT 上下文嵌入如何解决传统 word2vec 无法区分的多义词问题。

## Run It

```bash
cd s17_pretrained_models/code
python demo.py
```

## Key Output

- BERT 情感分类微调：24 条中文评论训练 / 6 条验证，二分类（正/负）
- 使用 prajjwal1/bert-tiny（最小 BERT，2 层 128 维，约 4MB）
- 微调 4 个 epoch，验证准确率约 80-95%
- BERT MLM 演示：在 5 个例句中预测 [MASK] 位置的最可能词
- GPT-2 生成演示：从 3 个提示词各生成 30 个 token 的续写文本

<<< @/snippets/s17_pretrained_models/demo.py
