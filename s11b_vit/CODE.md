# s11b Vision Transformer — 代码说明与运行报告

## 程序做了什么
使用 PyTorch 从零实现 Vision Transformer（ViT），包含 Patch Embedding、多头自注意力、Transformer Encoder Block 和 MLP 分类头。在 CIFAR-10 上训练，与从零训练的 ResNet-18 以及预训练 ViT-B/16（微调）进行准确率对比。通过准确率柱状图和训练曲线图，直观展示 ViT 在数据有限时的小样本局限性以及预训练带来的巨大收益。

## 运行方法
```bash
cd s11b_vit/code
python vit_demo.py
```

## 运行结果

### 输出摘要
- 数据集：CIFAR-10，50000 训练 / 10000 测试，统一 resize 到 224x224，10 类
- GPU 模式：SimpleViT (embed_dim=384, depth=8, heads=6) 约 10 epochs；CPU 模式：轻量 ViT (embed_dim=192, depth=4, heads=3) 约 3 epochs
- 参数量：SimpleViT 根据配置不同约 5-90M；ResNet-18 约 11.2M；ViT-B/16 约 86M
- SimpleViT（从零训练）在 CIFAR-10 上准确率有限，因为 ViT 需要海量数据才能充分发挥
- 预训练 ViT-B/16 微调后准确率远超从零训练模型，展示迁移学习的威力
- ResNet-18 内置的卷积归纳偏置使其在小数据集上仍具竞争力

### 生成图表

#### 图表 1: 模型准确率对比柱状图
![ViT准确率对比](./images/vit_accuracy_comparison.png)
**说明了什么：** 横向对比各模型在 CIFAR-10 测试集上的最终准确率。预训练 ViT 微调后显著领先，ResNet-18 从零训练通常优于 SimpleViT（小数据下 CNN 的归纳偏置是优势），SimpleViT 从零训练效果有限。

#### 图表 2: 训练曲线对比
![ViT训练曲线](./images/vit_training_curves.png)
**说明了什么：** 两张子图分别展示 (1) 训练 Loss 下降曲线 和 (2) 测试准确率上升曲线。可以观察到 ViT 的 Loss 下降通常比 CNN 慢（缺乏归纳偏置），但预训练模型微调的收敛速度极快且起点很高。

## 代码结构
- `get_cifar10_loaders()` — CIFAR-10 数据加载，resize 到 224x224 + ImageNet 归一化
- `class PatchEmbedding` — 图像切 Patch：Conv2d(kernel_size=patch_size, stride=patch_size) 实现
- `class MultiHeadAttention` — 多头缩放点积自注意力：QKV 投影 → 分头 → Attention → 拼接 → 输出投影
- `class TransformerBlock` — ViT 编码器块：Pre-Norm → MSA → 残差 → Pre-Norm → MLP → 残差
- `class SimpleViT` — 完整 ViT 模型：Patch Embedding + CLS Token + 位置编码 + N 层 Transformer + 分类头
- `load_pretrained_vit()` — 加载预训练 ViT-B/16（优先 torchvision，回退 timm）
- `load_pretrained_resnet()` — 加载预训练 ResNet-18
- `train_one_epoch()` / `evaluate()` — 单轮训练/评估工具
- `train_full()` — 完整训练流程（含 AdamW + CosineAnnealingLR）
- `quick_finetune()` — 预训练模型微调（分类头大 lr，backbone 小 lr）
- `plot_accuracy_comparison()` — 模型准确率柱状图
- `plot_training_curves()` — 训练 Loss + 准确率曲线对比图
- `count_parameters()` — 统计模型参数量

## 运行环境
- Python 依赖: torch, torchvision, matplotlib, numpy（timm 可选，用于回退加载预训练 ViT）
- 硬件需求: GPU 推荐（ViT 训练计算量大，10 epoch 约 15-30 分钟）；CPU 可用（自动降为轻量模式，3 epoch 约 5-15 分钟）
- 预计运行时间: GPU 15-30 分钟 / CPU 5-15 分钟
- 预训练模型首次运行时需下载（约 330MB ViT-B/16 + 45MB ResNet-18）
