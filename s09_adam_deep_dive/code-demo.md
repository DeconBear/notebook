---
title: "s09 Adam深度解析与训练实战 — demo.py"
---

# s09 Adam深度解析与训练实战 — demo.py

<a href="../code/s09_adam_deep_dive/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现Adam、AdamW和SGD+Momentum优化器的完整NumPy版本，在MNIST子集上训练小型MLP（784->128->64->10），对比三者的训练性能和收敛速度。额外演示偏差修正的效果（有vs无）、Warmup+Cosine Decay学习率调度、以及梯度裁剪对大学习率下梯度爆炸的抑制。

## Run It

```bash
cd s09_adam_deep_dive/code
python demo.py
```

## Key Output

- 数据集：MNIST子集，5000训练/1000验证，28x28灰度图，10类别
- 网络结构：784输入 -> 128隐藏(ReLU) -> 64隐藏(ReLU) -> 10输出(Softmax)，共约 110K 参数
- Adam 收敛最快，AdamW 在验证集上泛化略优于 Adam（权重衰减的解耦设计），SGD+Momentum 收敛最慢
- 偏差修正效果：无偏差修正时早期步长过大训练不稳定；有偏差修正则学习率从0平滑启动

<<< @/snippets/s09_adam_deep_dive/demo.py
