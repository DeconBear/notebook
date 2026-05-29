---
title: "s10 CNN核心原理 — demo.py"
---

# s10 CNN核心原理 — demo.py

<a href="../code/s10_cnn_fundamentals/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

使用纯 NumPy 从零实现 Conv2d（通过 im2col 矩阵乘法加速）和 MaxPool2d（通过 as_strided 视图技巧），构建一个简单 CNN（Conv->ReLU->Pool->Conv->ReLU->Pool->FC->Softmax）在 MNIST 上训练。同时展示卷积核可视化、逐层特征图可视化、感受野递推计算，以及卷积网络的参数效率优势。

## Run It

```bash
cd s10_cnn_fundamentals/code
python demo.py
```

## Key Output

- MNIST 数据集：60000 训练 / 10000 测试，28x28 灰度图像
- 模型结构：Conv(1->8, 3x3) -> ReLU -> MaxPool(2) -> Conv(8->16, 3x3) -> ReLU -> MaxPool(2) -> FC(784, 10)
- 总参数量：约 8,190（卷积+FC），同等全连接网络需约 100,000+，参数节省约 12 倍
- 训练 3 个 epoch 后的测试准确率：约 70-85%

<<< @/snippets/s10_cnn_fundamentals/demo.py
