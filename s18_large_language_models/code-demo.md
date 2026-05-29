---
title: "s18 大语言模型 — demo.py"
---

# s18 大语言模型 — demo.py

<a href="../code/s18_large_language_models/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

演示大语言模型的四个核心概念：Scaling Law 幂律下降规律（Kaplan + Chinchilla 最优配比 D=20N）、涌现行为的 sigmoid 相位转变模拟（小模型随机+跨阈值跃升）、DPO 直接偏好优化的损失函数实现与数值示例、以及 LoRA 低秩微调的概念演示（参数对比 + 模拟训练）。

## Run It

```bash
cd s18_large_language_models/code
python demo.py
```

## Key Output

- Kaplan Scaling Law 公式 L(N,D) = a/N^alpha + b/D^beta + c 的计算示例
- 涌现行为模拟：4 个涌现任务（3位加减法、多语言翻译、CoT推理、指令遵循）和 2 个非涌现任务（情感分析、词性标注）的准确率曲线
- DPO 损失数值对比：正确偏好场景 Loss 较小，错误偏好场景 Loss 较大
- LoRA 参数量对比：全参数 16.8M vs LoRA(r=16) 131K，减少 128x

<<< @/snippets/s18_large_language_models/demo.py
