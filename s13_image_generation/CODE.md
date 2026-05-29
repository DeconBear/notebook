# s13 图像生成 -- 代码说明

## 概述
使用 PyTorch 分别实现 GAN（生成对抗网络）和 VAE（变分自编码器），在 MNIST 上训练并生成数字图像。对比两种生成方法：GAN 通过对抗训练生成较锐利的图像，VAE 通过编码-解码+KL 散度约束学习结构化潜空间。展示生成样本、VAE 重建、潜空间 t-SNE 可视化和训练曲线对比。

## 运行方法
```bash
cd s13_image_generation/code
python demo.py
```

## 依赖
- torch
- torchvision
- matplotlib
- numpy
- scikit-learn（可选：用于 t-SNE 潜空间可视化）

## 预期输出
命令行打印设备信息、GAN 各 epoch 的 D Loss 和 G Loss、VAE 各 epoch 的总损失/重构损失/KL 散度、以及最终总结对比。生成 4 张图。

## 代码结构
- `load_mnist()` / `to_image()` -- 数据加载和图像转换工具
- `class Generator` -- GAN 生成器：FC(128->256->512->1024->784)+BN+ReLU->Tanh，输出 (1,28,28)
- `class Discriminator` -- GAN 判别器：FC(784->512->256->1)+LeakyReLU->Sigmoid
- `train_gan()` -- GAN 训练循环（交替训练 D 和 G）
- `class VAE` -- 变分自编码器：encoder 输出 mu/logvar，重参数化采样 z，decoder 重建
- `vae_loss()` -- VAE 损失 = BCE 重构损失 + KL 散度（解析解）
- `train_vae()` -- VAE 训练循环
- `visualize_generated_samples()` -- GAN 生成图像网格可视化
- `visualize_vae_reconstructions()` -- VAE 原始 vs 重建对比（上下两行）
- `visualize_vae_latent_space()` -- VAE 潜空间 t-SNE 2D 投影（按数字类别着色）
- `plot_training_curves()` -- 三合一对比图（GAN 损失、VAE 损失、文字对比说明）
- `main()` -- 主流程

## 输出文件
图片保存在 `s13_image_generation/images/` 目录：
- `gan_samples.png` -- GAN 生成的 16 张数字图像（4x4 网格）
- `vae_reconstructions.png` -- VAE 原始图像（上）vs 重建图像（下）对比
- `vae_latent_space.png` -- VAE 潜空间的 t-SNE 2D 投影（10 类颜色区分）
- `training_curves.png` -- GAN/VAE 训练曲线与 GAN vs VAE 文字对比说明
