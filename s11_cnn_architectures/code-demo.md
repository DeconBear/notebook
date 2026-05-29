---
title: "s11 经典CNN架构演进 — demo.py"
---

# s11 经典CNN架构演进 — demo.py

<a href="../code/s11_cnn_architectures/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

使用 PyTorch 从零构建 ResNet-18（含 BasicBlock 残差块），在 CIFAR-10 上训练，并与同深度的 Plain CNN（无跳跃连接）进行对比实验。通过训练损失曲线、测试准确率曲线和逐层梯度范数分布，直观展示残差连接的梯度高速公路效应如何缓解深度网络的退化问题。

## Run It

```bash
cd s11_cnn_architectures/code
python demo.py
```

## Key Output

- 数据集：CIFAR-10，50000 训练 / 10000 测试，32x32 RGB，10 类
- ResNet-18 参数量：约 11.17M；Plain CNN 参数量：约 11.17M（参数量相同，仅架构不同）
- ResNet-18 最终测试准确率显著高于 Plain CNN（残差连接使深度网络更容易优化）
- ResNet 的平均梯度范数明显大于 Plain CNN，证明跳跃连接能有效保持深层梯度流动

<<< @/snippets/s11_cnn_architectures/demo.py
