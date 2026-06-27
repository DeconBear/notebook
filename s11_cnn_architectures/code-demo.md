---
title: "s11 经典CNN架构演进 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s11 经典CNN架构演进 — demo.py 代码详解

<a href="../code/s11_cnn_architectures/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s11_cnn_architectures/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 —— 每个库是做什么的

```python
import torch                          # PyTorch 核心框架，提供张量计算和自动求导
import torch.nn as nn                 # 神经网络模块（Linear, Conv2d, BatchNorm等）
import torch.nn.functional as F       # 函数式接口（relu, cross_entropy等，无状态）
import torch.optim as optim           # 优化器（SGD, Adam）和学习率调度器
import torchvision                    # 视觉库，提供预训练模型和数据集（CIFAR-10）
import torchvision.transforms as transforms  # 数据预处理/增强管道

# GPU 自动检测：CUDA（NVIDIA） > MPS（Apple Silicon） > CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
```

| 库 | 角色 |
|---|---|
| `torch` | 张量计算引擎 + 自动求导 |
| `torch.nn` | 有状态（含可学习参数）的网络层 |
| `torch.nn.functional` | 无状态的函数（激活函数、池化等） |
| `torch.optim` | 优化器（SGD/Adam）+ 学习率衰减策略 |
| `torchvision` | 视觉数据集 + 预训练模型 |
| `torchvision.transforms` | 图像增广管道（裁剪、翻转、归一化） |

### 第2步：数据准备 —— CIFAR-10 数据集

```python
def get_cifar10_loaders(batch_size: int = 128):
    # 训练数据增强：随机裁剪+填充（抗过拟合）、水平翻转、归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 32x32 图像先填充到 40x40，再随机裁回 32x32
        transforms.RandomHorizontalFlip(),         # 50% 概率水平翻转
        transforms.ToTensor(),                     # PIL Image → Tensor，值域 [0,1]
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 各通道均值
                             (0.2023, 0.1994, 0.2010)),  # CIFAR-10 各通道标准差
    ])
    # 测试数据：仅归一化，不做增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(...),
    ])
```

**为什么使用数据增强？** CIFAR-10 只有 50000 张训练图。随机裁剪和翻转在不增加存储的前提下增大了有效数据量，迫使模型学习平移/翻转不变性，是 AlexNet 时代延续至今的标准做法。

**为什么用这些均值和标准差？** 它们是整个 CIFAR-10 训练集各通道的统计量。归一化后每个通道的输入分布接近 $\mathcal{N}(0, 1)$，梯度更稳定，收敛更快。

**回退机制**：如果 CIFAR-10 下载失败（如网络问题），代码自动创建合成数据确保 demo 可运行。CPU 模式下将训练集缩减到 1000 样本以在合理时间内完成演示。

### 第3步：ResNet BasicBlock —— 残差学习的核心

这是 ResNet 最关键的组件。让我们逐行剖析：

```python
class BasicBlock(nn.Module):
    expansion = 1  # BasicBlock 不改变通道数倍数（对比 Bottleneck 的 expansion=4）

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        # 第一个 3×3 卷积：输入 in_planes 通道 → 输出 planes 通道
        # padding=1 保持空间尺寸不变（输入 H×W → 输出 H×W，stride=1 时）
        # bias=False 因为后面紧跟 BatchNorm（BN 自带 bias 项，卷积的 bias 浪费）
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)   # 批归一化：稳定训练、加速收敛

        # 第二个 3×3 卷积：planes → planes（同维度卷积）
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 跳跃连接（Skip Connection / Identity Shortcut）
        # 如果输入输出维度不匹配，用 1×1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),  # 1×1 卷积仅用于维度对齐
                nn.BatchNorm2d(planes),
            )
```

**为什么 bias=False？** 卷积层的 bias 为每个输出通道增加一个可学习偏置。但 BatchNorm 的计算中包含 $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$，其中 $\beta$ 本身就是偏置项。卷积后紧接 BN 时，卷积的 bias 会被 BN 的 $\beta$ 完全替代，设置 bias=False 减少无意义的参数。

**为什么 shortcut 用 1×1 卷积？** 当 `stride != 1`（空间尺寸减半）或 `in_planes != planes`（通道数变化）时，identity $x$ 的维度和主路径输出 $F(x)$ 不一致，无法直接相加。$1 \times 1$ 卷积在保持空间信息的前提下完成通道和尺寸的对齐，计算开销极小。

**前向传播 —— 残差公式的实现**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = self.shortcut(x)          # 跳跃连接分支

    out = self.conv1(x)                  # Conv 3×3
    out = self.bn1(out)                  # BatchNorm
    out = F.relu(out)                    # ReLU 激活

    out = self.conv2(out)                # Conv 3×3
    out = self.bn2(out)                  # BatchNorm —— 注意：这里还没加 ReLU！

    out += identity                      # 🔑 残差加法 H(x) = F(x) + x
    out = F.relu(out)                    # 加法之后才做 ReLU

    return out
```

> **关键细节：残差加法在第二个 BN 之后、第二个 ReLU 之前。** 为什么？如果把 ReLU 放在加法之前（即 $F(x)$ 做完 ReLU 再加 $x$），$F(x)$ 的输出被钳制在 $\ge 0$，限制了残差映射的表达能力。加法后再 ReLU 让 $F(x) + x$ 整体通过非线性，保留了更丰富的特征。

**数学对应**：

$$
y = \text{ReLU}(\text{BN}(\text{Conv}(\text{BN}(\text{Conv}(x)))) + x)
$$

即：

$$
y = \sigma(\mathcal{F}(x, \{W_i\}) + x)
$$

其中 $\sigma$ 是 ReLU，$\mathcal{F}$ 是两层卷积+BN 组成的残差函数。

**梯度高速公路效应**：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}
= \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)
$$

那个 **"+1"** 是残差连接对梯度传播的核心贡献：即使 $\frac{\partial \mathcal{F}}{\partial x}$ 非常小（梯度衰减），梯度也能通过恒等路径无损传播。这就是为什么 ResNet 可以训练 152 层而不退化。

### 第4步：Bottleneck —— 深层 ResNet 的效率秘诀

```python
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道 = planes * 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        # 第1层: 1×1 降维
        #   in_planes → planes（如 256 → 64），大幅削减计算量
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第2层: 3×3 卷积（主要特征提取）
        #   planes → planes，在低维度下计算
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 第3层: 1×1 升维
        #   planes → planes * 4（如 64 → 256），恢复高维度
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
```

**为什么 Bottleneck 先降维再升维？** 直接做 $256 \to 256$ 的 $3 \times 3$ 卷积，FLOPs 为 $9 \times 256 \times 256 = 589,824$。先 $1 \times 1$ 降到 $64$，做 $3 \times 3$ 卷积 $9 \times 64 \times 64 = 36,864$，再 $1 \times 1$ 升回 $256$ 需 $64 \times 256 = 16,384$。总计 $1 \times 1$ 的 $256 \times 64 \times 2 = 32,768$，加上 $3 \times 3$ 的 $36,864$，约 $70,000$ —— **节省了 88% 的计算量**，而表达能力几乎不损失（$1 \times 1$ 卷积学习通道间的线性组合）。

Bottleneck 的形状像一个沙漏：两头粗（高通道数），中间细（低通道数）。这被用于 ResNet-50/101/152。

### 第5步：完整 ResNet —— 组装所有组件

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        self.in_planes = 64  # 初始通道数

        # 初始卷积：3通道 → 64通道，stride=1 适配 CIFAR-10（32×32 无需大下采样）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4个残差层，通道逐层翻倍，空间逐层减半
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)  # 64→64,  32×32保持
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 64→128, 32→16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 128→256, 16→8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 256→512, 8→4

        # 全局平均池化：将 4×4 特征图压缩为 1×1，替代传统的大全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
```

**ResNet 的通道-空间 trade-off**：每经过一个 layer，通道数翻倍、空间尺寸减半，总信息量基本守恒（$64 \times 32 \times 32 \approx 256 \times 8 \times 8$）。这与 CNN 的通用设计哲学一致：浅层关注"哪里"（大空间尺寸、少通道），深层关注"是什么"（小空间尺寸、多通道）。

**`_make_layer` 方法**：

```python
def _make_layer(self, block, planes, num_blocks, stride):
    layers = []
    # 第一个 block 可能需要下采样
    layers.append(block(self.in_planes, planes, stride))
    self.in_planes = planes * block.expansion  # 更新全局通道数
    # 后续 block 保持尺寸
    for _ in range(1, num_blocks):
        layers.append(block(self.in_planes, planes, stride=1))
    return nn.Sequential(*layers)
```

**Kaiming 初始化**：

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
```

**为什么用 Kaiming 初始化而非 Xavier？** Xavier（Glorot）初始化假设激活函数是线性的（或 tanh），在 ReLU 下会导致输出方差逐层衰减。Kaiming（He）初始化专门针对 ReLU 设计——它考虑了 ReLU 将一半输入置零的特性，使得正向传播的方差保持稳定：

$$
\text{Var}(W) = \frac{2}{\text{fan\_out}}
$$

`mode='fan_out'` 表示用输出神经元数（而非输入）来缩放，适合反向传播梯度的稳定。

### 第6步：Plain CNN —— 对照模型

```python
class PlainBlock(nn.Module):
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # ⚠️ 这里没有 out += x —— 缺少了跳跃连接！
        out = F.relu(out)
        return out
```

PlainCNN 的结构与 ResNet-18 完全一致，唯一的区别是：**去掉了所有的跳跃连接（shortcut）**。参数量相同，但优化难度截然不同。这个对照实验展示了：

- **Plain CNN**：深度增加 $\to$ 梯度逐渐消失 $\to$ 训练误差上升（退化问题）
- **ResNet-18**：梯度通过 shortcut 无损传播 $\to$ 深度越大效果越好

### 第7步：训练流程 —— 梯度范数监控

```python
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()          # 清零梯度（PyTorch 默认累积梯度）
        outputs = model(inputs)        # 前向传播
        loss = criterion(outputs, targets)  # 交叉熵损失
        loss.backward()                # 反向传播，计算梯度
        optimizer.step()               # 更新参数
```

**为什么用 SGD + Momentum 而不是 Adam？** 对于 ResNet 训练，SGD with Momentum（$\text{momentum}=0.9$）配合 CosineAnnealing 学习率调度是经过验证的最佳实践。Adam 虽然收敛快，但在图像分类任务的最优精度上通常不如精心调参的 SGD。

**梯度范数计算**：

```python
def compute_gradient_norms(model):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()  # L2 范数
    return grad_norms
```

梯度范数 $\|\nabla_W \mathcal{L}\|_2$ 是衡量梯度流动的指标。ResNet 的梯度范数显著大于 Plain CNN，尤其是在深层——这是"梯度高速公路"的数值证据。

### 第8步：可视化 —— 对比分析

代码生成三张图：

1. **训练 Loss 曲线**：ResNet vs Plain CNN 的 loss 下降速度
2. **测试准确率曲线**：残差连接带来的精度提升
3. **梯度范数对比**：ResNet 梯度范数更大，证明 shortcut 维持了有效梯度

### 关键概念速查表

| 概念 | 公式 | 代码对应 |
|------|------|---------|
| 残差块 | $y = \mathcal{F}(x) + x$ | `out = bn2(conv2(...)); out += identity` |
| 梯度高速公路 | $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot (1 + \frac{\partial \mathcal{F}}{\partial x})$ | `loss.backward()` 自动传播 |
| BatchNorm | $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | `nn.BatchNorm2d(planes)` |
| Kaiming 初始化 | $\text{Var}(W) = \frac{2}{\text{fan\_out}}$ | `nn.init.kaiming_normal_` |
| Bottleneck | 1×1降维 → 3×3卷积 → 1×1升维 | `Bottleneck` 类 |
| 退化问题 | 层数增加但训练误差不降反升 | PlainCNN 的糟糕表现 |
| 网络退化 | 不是过拟合（测试误差也上升），是优化困难 | ResNet 解决了这个问题 |

## 完整代码

<<< @/snippets/s11_cnn_architectures/demo.py
