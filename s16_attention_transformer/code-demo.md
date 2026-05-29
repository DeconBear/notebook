---
title: "s16 Attention与Transformer — demo.py"
---

# s16 Attention与Transformer — demo.py

<a href="../code/s16_attention_transformer/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现 Transformer 的核心组件：缩放点积注意力（Scaled Dot-Product Attention）、多头自注意力（Multi-Head Self-Attention）、Position-wise FFN、LayerNorm + 残差连接的 Encoder Block，并构建一个 Mini-GPT（decoder-only Transformer）用于字符级文本生成。额外演示了注意力热力图可视化（因果掩码下三角）、多头注意力模式差异，以及 sqrt(d_k) 缩放对 softmax 饱和程度的关键影响。

## Run It

```bash
cd s16_attention_transformer/code
python demo.py
```

## Key Output

- Mini-GPT 配置：d_model=64, num_heads=4, num_layers=3, d_ff=256
- 参数量：约 50,000-80,000（微型模型，用于演示）
- 训练：80 个 epoch，训练损失从约 3.0 降至 1.0 以下
- 文本生成：从种子"深度"、"自然"、"注意"、"计算"各生成 30 个字符
- sqrt(d_k) 缩放实验：无缩放时 d_k=256 的 softmax 极度饱和；有缩放后分布更均匀

<<< @/snippets/s16_attention_transformer/demo.py
