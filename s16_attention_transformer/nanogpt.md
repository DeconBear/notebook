---
title: "s16 — nanoGPT: 从零训练 GPT-2"
---

# nanoGPT: 从零训练一个真正的 GPT

基于 [Andrej Karpathy 的 nanoGPT](https://github.com/karpathy/nanoGPT)，从零实现完整的 GPT-2 架构，
在莎士比亚文本上训练并生成文本。

## What This Code Does

- 实现完整的 GPT-2 Transformer（因果自注意力、LayerNorm、FFN、残差连接）
- 内置莎士比亚剧本作为训练语料（无需下载外部数据）
- CPU 上 500 次迭代即可看到效果，GPU 上 5000 次迭代可生成高质量文本
- 支持模型保存/加载和温度控制、top-k 采样

## Run It

```bash
cd s16_attention_transformer/code
python nanogpt.py           # CPU 训练（~15 分钟）
python nanogpt.py --gpu     # GPU 训练（如有）
python nanogpt.py --generate # 仅生成文本（需已有训练好的模型）
```

## Architecture

| 组件 | 实现 |
|------|------|
| Token Embedding | `nn.Embedding(vocab_size, n_embd)` |
| Position Embedding | `nn.Embedding(block_size, n_embd)` |
| CausalSelfAttention | Q/K/V 投影 + 因果 mask + 多头拆分/合并 |
| MLP | Linear → GELU → Linear（4× 扩展比） |
| Block | Pre-LN → Attention → residual → Pre-LN → MLP → residual |
| Output | LayerNorm → Linear(vocab_size) |
| Weight Tying | token embedding 和 lm_head 共享权重 |

## Key Output

- **训练输出**: 每 100 步打印 train/val loss
- **生成输出**: 3 段文本（不同 prompt 和温度）
- **保存**: 最佳模型保存为 `nanogpt_model.pt`

## nanoGPT vs demo.py

| | demo.py | nanoGPT |
|------|---------|---------|
| 模型规模 | ~2M 参数 | ~15M 参数 |
| 训练数据 | 合成短文本 | 完整莎士比亚剧本 |
| 训练时间 | ~30 秒 | ~15 分钟 |
| 生成质量 | 字符级乱码 | 有意义的仿莎士比亚文本 |
| 架构完整性 | 简化版 | 完整 GPT-2 |
| 适用场景 | 理解 Transformer 原理 | 体验真正的 GPT 训练 |

<a href="../code/s16_attention_transformer/nanogpt.py" target="_blank" download>Download nanogpt.py</a>

```python
<<< @/snippets/s16_attention_transformer/nanogpt.py
```
