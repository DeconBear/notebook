---
title: "s13 图像生成 — demo.py"
---

# s13 图像生成 — demo.py

<a href="../code/s13_image_generation/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

使用 PyTorch 分别实现 GAN（生成对抗网络）和 VAE（变分自编码器），在 MNIST 上训练并生成数字图像。对比两种生成方法：GAN 通过对抗训练生成较锐利的图像，VAE 通过编码-解码+KL 散度约束学习结构化潜空间。展示生成样本、VAE 重建、潜空间 t-SNE 可视化和训练曲线对比。

## Run It

```bash
cd s13_image_generation/code
python demo.py
```

## Key Output

- 命令行打印设备信息、GAN 各 epoch 的 D Loss 和 G Loss、VAE 各 epoch 的总损失/重构损失/KL 散度
- 生成 4 张图：GAN 生成样本（4x4 网格）、VAE 重建对比（原始 vs 重建）、VAE 潜空间 t-SNE 投影（按类别着色）、训练曲线对比

<<< @/snippets/s13_image_generation/demo.py
