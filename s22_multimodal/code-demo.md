---
title: "s22 多模态模型 — demo.py"
---

# s22 多模态模型 — demo.py

<a href="../code/s22_multimodal/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

加载 OpenAI 预训练的 CLIP ViT-B/32 模型（首次运行自动下载约 600MB），演示零样本图像分类（无需任何训练即可识别任意自定义类别）、图文相似度计算（图像与多段文本描述的余弦相似度得分矩阵）、以及 CLIP 嵌入空间的 PCA 降维可视化（展示图像和文本在共享 512 维空间中的分布关系）。

## Run It

```bash
cd s22_multimodal/code
python demo.py
```

## Key Output

- 环境自动检测 CUDA/MPS/CPU 设备并据此选择运行模式
- 零样本分类：对测试图像输出 Top-5 预测类别及概率分数
- 图文相似度：计算图像与多段文本描述之间的余弦相似度矩阵并打印
- PCA 可视化：将图像和文本嵌入向量降至 2D 并绘制在同一共享空间中

<<< @/snippets/s22_multimodal/demo.py
