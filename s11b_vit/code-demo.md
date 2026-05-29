---
title: "s11b Vision Transformer — vit_demo.py"
---

# s11b Vision Transformer — vit_demo.py

<a href="../code/s11b_vit/vit_demo.py" target="_blank" download>Download vit_demo.py</a>

## What This Code Does

使用 PyTorch 从零实现 Vision Transformer（ViT），包含 Patch Embedding、多头自注意力、Transformer Encoder Block 和 MLP 分类头。在 CIFAR-10 上训练，与从零训练的 ResNet-18 以及预训练 ViT-B/16（微调）进行准确率对比。通过准确率柱状图和训练曲线图，直观展示 ViT 在数据有限时的小样本局限性以及预训练带来的巨大收益。

## Run It

```bash
cd s11b_vit/code
python vit_demo.py
```

## Key Output

- 数据集：CIFAR-10，50000 训练 / 10000 测试，统一 resize 到 224x224，10 类
- GPU 模式：SimpleViT (embed_dim=384, depth=8, heads=6) 约 10 epochs
- CPU 模式：轻量 ViT (embed_dim=192, depth=4, heads=3) 约 3 epochs
- 预训练 ViT-B/16 微调后准确率远超从零训练模型
- ResNet-18 在小数据集上仍有竞争力（CNN 的归纳偏置优势）

<<< @/snippets/s11b_vit/vit_demo.py
