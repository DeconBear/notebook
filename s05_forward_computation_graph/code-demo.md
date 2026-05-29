---
title: "s05 前向传播与计算图 — demo.py"
---

# s05 前向传播与计算图 — demo.py

<a href="../code/s05_forward_computation_graph/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

用纯NumPy构建一个3层MLP（3->4->4->1），展示完整的前向传播过程。包括He参数初始化、逐层线性变换+非线性激活（ReLU/Sigmoid）、中间值缓存（cache）、张量形状追踪，以及网络结构图、激活函数对比图和激活值分布演变图的可视化。

## Run It

```bash
cd s05_forward_computation_graph/code
python demo.py
```

## Key Output

- 网络结构：输入3维 -> 隐藏层1(4神经元, ReLU) -> 隐藏层2(4神经元, ReLU) -> 输出层(1神经元, Sigmoid)
- 总参数量：3*4+4 + 4*4+4 + 4*1+1 = 41个参数
- 输入：32个样本的小批量，每个样本3个特征
- 前向传播输出：每个样本一个标量预测值（Sigmoid输出，范围[0,1]）

<<< @/snippets/s05_forward_computation_graph/demo.py
