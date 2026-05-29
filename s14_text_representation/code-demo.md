---
title: "s14 文本表示 — demo.py"
---

# s14 文本表示 — demo.py

<a href="../code/s14_text_representation/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现两种经典文本表示方法：TF-IDF（词频-逆文档频率，基于统计的稀疏表示）和 word2vec Skip-gram + 负采样（基于神经网络的稠密词向量）。在中文语料库（25 篇文档，覆盖体育/科技/教育/经济/医疗 5 个主题）上进行演示，通过 TF-IDF 热力图、t-SNE 词向量可视化、近义词查询和类比推理直观对比两种方法的表达能力差异。

## Run It

```bash
cd s14_text_representation/code
python demo.py
```

## Key Output

- 语料库：25 篇中文文档，5 个主题（体育/科技/教育/经济/医疗各 5 篇）
- 词汇表大小：约 250+ 个独特汉字
- TF-IDF：为每篇文档提取 Top-3 关键词
- Skip-gram 训练：窗口大小 2，负采样 5 个，词向量维度 64，训练 80 个 epoch
- 近义词查询：同一领域的词余弦相似度显著高于跨领域词对

<<< @/snippets/s14_text_representation/demo.py
