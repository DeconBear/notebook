---
title: "s11b Vision Transformer — vit_demo.py"
---

# s11b Vision Transformer — vit_demo.py 代码详解

<a href="../code/s11b_vit/vit_demo.py" target="_blank" download>Download vit_demo.py</a>

## 运行方式

```bash
cd s11b_vit/code
python vit_demo.py
```

## 代码逐段详解

### 第1步：导入库 —— 每个库是做什么的

```python
import torch
import torch.nn as nn           # Transformer 需要 Linear, LayerNorm, Dropout
import torch.nn.functional as F  # softmax, cross_entropy 等
import torch.optim as optim      # AdamW 优化器
import torchvision               # 数据集（Imagenette/CIFAR-100）、预训练 ViT
import torchvision.transforms as transforms
```

| 库 | 在此 demo 中的角色 |
|---|---|
| `torch.nn` | 构建 PatchEmbedding、MultiHeadAttention、TransformerBlock |
| `torch.optim` | AdamW（解耦权重衰减的 Adam）+ 余弦退火调度 |
| `torchvision` | Imagenette 下载 + 预训练 ViT-B/16 加载 |

**设备自适应**：CPU 模式下使用轻量 ViT（`embed_dim=192, depth=4`），3 个 epoch 快速演示。GPU 模式下使用较大模型（`embed_dim=384, depth=8`），10 个 epoch 得到有意义的结果。

### 第2步：数据加载 —— Imagenette 数据集

```python
def get_imagenette_loaders(batch_size: int, img_size: int = 224):
```

**为什么用 Imagenette 而不是 CIFAR-10？** ViT 的 patch embedding 需要输入分辨率能被 patch_size 整除。原始 ViT 设计为 $224 \times 224$ 输入，切分成 $14 \times 14 = 196$ 个 patch（每个 $16 \times 16$）。CIFAR-10 的 $32 \times 32$ 太小，需要大改架构。Imagenette 是 ImageNet 的 10 类子集（tench, springer, cassette player 等），图像原始尺寸足够大，resize 到 $224 \times 224$ 后可以直接适配标准 ViT。

**三级回退策略**：
1. Imagenette（$\sim$1.5GB，自动下载）—— 首选
2. CIFAR-100 —— 回退方案1
3. 合成随机数据 —— 回退方案2（仅演示流程）

**数据增强**：

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),            # 统一尺寸
    transforms.RandomHorizontalFlip(),         # 水平翻转
    transforms.ColorJitter(0.2, 0.2),         # 颜色抖动：亮度±20%，对比度±20%
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 统计值
                         std=[0.229, 0.224, 0.225]),
])
```

> **颜色抖动（ColorJitter）的关键性**：ViT 没有 CNN 的色彩归纳偏置，color jitter 作为数据增强强迫 ViT 学习颜色不变的表示，在小数据集上尤为重要。

### 第3步：Patch Embedding —— 图像变成词序列

这是 ViT 中唯一与 NLP 不同的组件，是连接两个世界的"桥梁"。

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        # Conv2d 的 kernel_size 和 stride 都设为 patch_size
        # 这等价于"均匀切分 + 线性投影"两个操作
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):  # x: (B, 3, 224, 224)
        x = self.proj(x)           # (B, D, 14, 14) — 每个 16×16 patch → D维向量
        x = x.flatten(2)           # (B, D, 196)  — 展平空间维度
        x = x.transpose(1, 2)      # (B, 196, D) — 变成序列格式
        return x
```

**为什么用 Conv2d 而不是手动切分？** 数学上等价，但 Conv2d 更高效：

$$
\text{PatchEmbed}(\mathbf{I}) = \text{Conv2d}(C_{\text{in}}=3, C_{\text{out}}=D, \text{kernel}=P, \text{stride}=P)(\mathbf{I})
$$

卷积核以步长 $P$ 在图像上滑动，每次覆盖 $P \times P$ 的区域并输出一个 $D$ 维向量。由于 stride = kernel_size，各窗口**互不重叠**，恰好实现了"切 patch + 线性投影"。

**维度变化跟踪**：

```
输入:  (B, 3, 224, 224)     # 一张 224×224 RGB 图
 ↓ Conv2d(3→D, kernel=16, stride=16)
      (B, D, 14, 14)         # 14×14 个 patch，每个 D 维
 ↓ flatten(2)
      (B, D, 196)            # 196 个 token
 ↓ transpose(1,2)
      (B, 196, D)            # token 序列，shape 与 NLP 的 (B, seq_len, d_model) 一致
```

### 第4步：多头自注意力（MSA）—— ViT 的大脑

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False):
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子 1/√(d_k)

        # Q、K、V 合并在一个 Linear 中计算，比分别做三次 Linear 更高效
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)  # 多头拼接后的输出投影
```

**为什么 Q、K、V 合并在一个 Linear 中？** 矩阵乘法 $X W_{\text{qkv}}$ 的计算量是 $O(N \cdot D \cdot 3D)$。分开做三次 $X W_Q$、$X W_K$、$X W_V$ 需要三次 $O(N \cdot D^2)$。合并后只需一次大矩阵乘法和 reshape，内存访问更连续，GPU 利用更充分。

**前向传播 —— 注意力计算的代码实现**：

```python
def forward(self, x):  # x: (B, N, D)
    B, N, C = x.shape

    # 1. 线性变换得到 Q、K、V
    qkv = self.qkv(x)                              # (B, N, 3D)
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, num_heads, N, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]              # 各 (B, num_heads, N, head_dim)

    # 2. 缩放点积注意力
    attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
    attn = attn.softmax(dim=-1)                    # 沿 key 维度 softmax
    attn = self.attn_drop(attn)

    # 3. 加权求和 + 多头拼接
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)                               # (B, N, D)
    return x
```

**数学对应**：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

- `q @ k.transpose(-2, -1)` 计算的是 $\mathbf{Q}\mathbf{K}^\top$，产生 $(N, N)$ 的注意力矩阵
- `* self.scale` 除以 $\sqrt{d_k}$ —— **为什么要缩放？** 当 $d_k$ 很大时（如 64），$\mathbf{Q}\mathbf{K}^\top$ 的元素值可能很大，softmax 会进入饱和区（梯度几乎为零）。缩放使输入 softmax 的值保持在合理范围。
- `softmax(dim=-1)` 对每一行（每个 query 对所有 key）归一化，$\sum_j \alpha_{ij} = 1$
- `attn @ v` 用注意力权重对 value 加权求和

**ViT 中自注意力的意义**：与 CNN 不同，ViT 的每个 patch 在第一层就能"看到"所有其他 patch（全局感受野）。注意力矩阵 $\alpha_{ij}$ 编码了"第 $i$ 个 patch 应该关注第 $j$ 个 patch 的程度"。

### 第5步：Transformer 编码器块

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)   # Pre-Norm: MSA 前归一化
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)   # Pre-Norm: MLP 前归一化
        # MLP: D → 4D → D，GELU 激活
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-Norm 残差: MSA
        x = x + self.mlp(self.norm2(x))    # Pre-Norm 残差: MLP
        return x
```

**Pre-Norm vs Post-Norm**：原始 Transformer（Vaswani et al., 2017）使用 Post-Norm（残差后再 LayerNorm）。ViT 使用 Pre-Norm（LayerNorm 放在子层之前），这在训练更深的 Transformer 时更稳定，梯度流动更好。

**为什么用 GELU 而不是 ReLU？** GELU (Gaussian Error Linear Unit) 是 BERT 和 ViT 的标准激活函数：

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
$$

GELU 在 $x=0$ 附近是光滑的，不像 ReLU 那样有尖锐的不可导点。在 Transformer 中，GELU 通常比 ReLU 带来更好的收敛性和最终精度。

**为什么 MLP 有 4 倍扩展率？** `mlp_ratio=4` 意味着 MLP 的隐藏层是 `dim * 4`。这提供了足够的容量让 Transformer 在注意力聚合全局信息后，对每个 token 的表示做非线性变换。这个比例在原始 Transformer 论文中确定，在 ViT 和 BERT 中保持了一致。

### 第6步：完整 SimpleViT —— 组装所有组件

```python
class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # 2. CLS Token：可学习的分类 token，放在序列开头
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Position Embedding：可学习的 1D 位置编码
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 4. Transformer 编码器（L 层）
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 5. 分类头
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
```

**CLS Token 的工作原理**：`[CLS]` 是一个可学习的参数向量，形状为 $(D,)$。在每一层自注意力中，它与所有 patch token 交互，逐步聚合整个图像的信息。最终分类时只取 `[CLS]` 的输出。这个设计与 BERT 的 `[CLS]` 完全相同。

**为什么 1D 位置编码而不是 2D？** 直觉上图像有行列结构，应该用 2D 编码。但 ViT 的实验表明 1D 和 2D 效果几乎一样——Transformer 足够强大，可以从 1D 序列中学会推 2D 空间关系。训练完成后，相近位置的编码向量确实更相似，说明模型自动"领悟"了 2D 结构。

**完整前向传播**：

```python
def forward(self, x):  # x: (B, 3, 224, 224)
    B = x.shape[0]

    # 1. Patch Embedding: (B, C, H, W) → (B, N, D)
    x = self.patch_embed(x)

    # 2. 追加 CLS Token: (B, 1, D) + (B, N, D) → (B, N+1, D)
    cls_tokens = self.cls_token.expand(B, -1, -1)  # 复制到 batch 维度
    x = torch.cat((cls_tokens, x), dim=1)          # 在序列维度拼接

    # 3. 加上位置编码: (B, N+1, D) + (1, N+1, D) → (B, N+1, D)
    x = x + self.pos_embed  # 广播加法

    # 4. Dropout → Transformer 编码器
    x = self.pos_drop(x)
    x = self.blocks(x)

    # 5. LayerNorm → 取 CLS Token → 分类
    x = self.norm(x)
    x = self.head(x[:, 0])  # x[:, 0] 是 CLS token 的最终输出

    return x
```

**流程图示**：

```
Image (224×224)  →  PatchEmbed  →  196 Tokens
                                      + [CLS] Token  =  197 Tokens
                                      + Pos Embed
                                      ↓
                                  × L 层 TransformerEncoder
                                      ↓
                                  取 [CLS] Token  →  MLP Head  →  分类
```

### 第7步：预训练模型加载与微调

```python
def load_pretrained_vit(num_classes=10):
    # 方案1: torchvision (≥0.13)
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # ImageNet-1K 预训练
    in_features = model.heads.head.in_features                  # 768
    model.heads.head = nn.Linear(in_features, num_classes)      # 替换分类头
    return model
```

**为什么预训练 ViT 在 Imagenette 上需要微调？** 预训练模型学到的特征来自 ImageNet-1K 的 1000 类。Imagenette 虽然只取了其中的 10 类，但分类头不同——需要替换最后一层并微调。微调时学习率要小（backbone `lr×0.1`，head `lr`），避免破坏预训练权重。

**微调的参数分组策略**：

```python
head_params = []  # 分类头：正常学习率
body_params = []  # backbone：1/10 学习率
for name, param in model.named_parameters():
    if 'head' in name or 'fc' in name:
        head_params.append(param)
    else:
        body_params.append(param)

optimizer = optim.AdamW([
    {'params': body_params, 'lr': lr * 0.1},  # backbone 慢速更新
    {'params': head_params, 'lr': lr},          # 分类头正常更新
], weight_decay=0.05)
```

**为什么 AdamW 而不是 SGD？** Transformer 对优化器敏感。SGD 在 Transformer 上通常不如自适应优化器。AdamW（Adam with decoupled Weight Decay）是当前训练 Transformer 的标准选择：
- Adam 部分：自适应学习率，处理不同参数的梯度尺度差异
- Weight Decay 解耦：正则化项直接作用于参数（$\theta = \theta - \eta \cdot \lambda \theta$），而不是通过梯度，效果更好

### 第8步：可视化 —— 准确率对比与训练曲线

代码生成两张图：

1. **准确率柱状图**：SimpleViT（从零）、ResNet-18（从零）、ViT-B/16（预训练+微调）的对比
2. **训练曲线**：Training Loss 和 Test Accuracy 随 epoch 的变化

**预期结果解读**：

| 模型 | 预期准确率 | 原因 |
|------|-----------|------|
| SimpleViT（从零） | 60-75% | 缺少归纳偏置 + 数据量不足 |
| ResNet-18（从零） | 85-92% | CNN 的局部性先验在小数据集上优势明显 |
| ViT-B/16（预训练+微调） | 95%+ | 预训练弥补了数据需求，微调适配目标域 |

> 这个对比验证了 ViT 论文的核心发现：**归纳偏置 = 数据效率 vs 灵活性上限的 trade-off。** CNN 的局部性和平移等变性在小数据上是优势；ViT 抛弃这些偏置后，需要大规模预训练才能发挥潜力。

### 关键概念速查表

| 概念 | 公式/说明 | 代码对应 |
|------|----------|---------|
| Patch Embedding | $\text{Conv2d}(3, D, \text{kernel}=P, \text{stride}=P)$ | `PatchEmbedding` 类 |
| 自注意力 | $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ | `MultiHeadAttention.forward()` |
| CLS Token | 可学习参数 $\mathbf{x}_{\text{class}} \in \mathbb{R}^{D}$ | `self.cls_token = nn.Parameter(...)` |
| Pre-Norm 残差 | $x + \text{MSA}(\text{LN}(x))$ | `TransformerBlock.forward()` |
| GELU | $x \cdot \Phi(x)$ | `nn.GELU()` |
| 位置编码 | 可学习 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ | `self.pos_embed` |
| AdamW | Adam + 解耦权重衰减 | `optim.AdamW(...)` |
| 微调参数分组 | backbone 用 `lr×0.1`，head 用 `lr` | `quick_finetune()` |

## 完整代码

<<< @/snippets/s11b_vit/vit_demo.py
