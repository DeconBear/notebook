---
title: "s21 RLHF：当强化学习遇见大模型 — demo.py"
---

# s21 RLHF：当强化学习遇见大模型 — demo.py

<a href="../code/s21_rlhf/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

在学术教学用的简化玩具环境中实现 RLHF 核心流程：创建小型 LSTM 语言模型（词汇表 30，a-z + 特殊 token），构建基于规则的奖励模型，从零实现 PPO（含裁剪目标 L_CLIP、GAE 优势估计、KL 惩罚 R_total = R_RM - beta*KL）和 DPO（最小化偏好损失），并对比两种方法的训练稳定性与生成样本质量。

## Run It

```bash
cd s21_rlhf/code
python demo.py
```

## Key Output

- 玩具语言模型: LSTM (embed_dim=64, hidden=128, vocab=30)，max_seq_len=20
- SFT 预训练：用合成数据训练基础语言模型
- PPO 训练：每个 epoch 打印 reward/KL 散度/策略熵/策略损失
- DPO 训练：每个 epoch 打印 DPO loss，展示直接从偏好对数据学习的效率
- 对比总结：PPO vs DPO 的训练稳定性、最终生成样本质量对比

<<< @/snippets/s21_rlhf/demo.py
