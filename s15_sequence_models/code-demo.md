---
title: "s15 序列模型 — demo.py"
---

# s15 序列模型 — demo.py

<a href="../code/s15_sequence_models/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零手动实现 RNN、LSTM 和 GRU 三个循环神经网络的细胞单元（不使用 PyTorch 内置 RNN 模块），并在两个任务上对比它们的实际表现：任务1为字符级语言模型（给定前缀生成后续文本），任务2为序列分类（中文评论情感二分类）。通过训练损失对比和分类准确率对比，直观展示 LSTM/GRU 的门控机制如何解决 RNN 的梯度消失问题。

## Run It

```bash
cd s15_sequence_models/code
python demo.py
```

## Key Output

- 任务1（字符级语言模型）：词汇表约 200 字符，训练样本约 150 条
- 任务1 RNN/LSTM/GRU 最终训练损失：RNN > GRU > LSTM（LSTM 和 GRU 收敛效果相近，均优于 RNN）
- 任务1 文本生成：三个模型从种子"人工智能"各生成 40 字符，LSTM/GRU 连贯性明显优于 RNN
- 任务2（情感分类）：20 条中文评论（10 正 10 负），最终准确率 LSTM > GRU > RNN

<<< @/snippets/s15_sequence_models/demo.py
