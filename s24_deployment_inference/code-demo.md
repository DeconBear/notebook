---
title: "s24 模型部署与推理优化 — demo.py"
---

# s24 模型部署与推理优化 — demo.py

<a href="../code/s24_deployment_inference/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

用纯 NumPy 演示大模型推理的两个核心优化技术：KV Cache（模拟自回归生成中 Key/Value 矩阵的缓存复用，对比有/无缓存在不同序列长度下的 FLOPs 和耗时差异）和模型量化（FP32 到 INT8 的对称量化过程，展示压缩率、量化前后权重分布及精度损失 MSE/MAE）。

## Run It

```bash
cd s24_deployment_inference/code
python demo.py
```

## Key Output

- KV Cache 基准测试：不同序列长度下有/无 cache 的推理耗时对比及加速比
- 序列越长加速比越大：无 cache 时每步重算全部历史 K/V（O(n^2)），有 cache 时只算当前 token（O(n)）
- 量化统计：原始 FP32 权重的 min/max/mean/std，量化后 INT8 值的统计
- 压缩率：FP32 (4 bytes) vs INT8 (1 byte)，理论压缩比 4x
- 量化误差：MSE（均方误差）和 MAE（平均绝对误差）数值展示

<<< @/snippets/s24_deployment_inference/demo.py
