---
title: "s11 经典CNN架构演进 — exercise.py"
---

# s11 经典CNN架构演进 — exercise.py 练习指南

<a href="../code/s11_cnn_architectures/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写 ResNet 的核心组件，深入理解残差学习机制：

1. **理解 BasicBlock 的前向传播** —— 残差公式 $y = F(x) + x$ 在代码中如何落地
2. **理解 BatchNorm 的位置** —— 为什么 Conv $\to$ BN $\to$ ReLU 这个顺序是标准做法
3. **构建完整 ResNet** —— 从 BasicBlock 组装出 ResNet-34
4. **手动估算参数量** —— 不靠 `torch.summary`，用公式计算不同 ResNet 变体的参数

## 预备知识

- **残差连接**：$y = \mathcal{F}(x) + x$，其中 $\mathcal{F}$ 是两层 $3 \times 3$ 卷积
- **BatchNorm**：对 mini-batch 内各通道做归一化 $\frac{x - \mu_B}{\sigma_B} \cdot \gamma + \beta$
- **ReLU**：$\text{ReLU}(x) = \max(0, x)$，最常用的 CNN 激活函数
- **Kaiming 初始化**：适用于 ReLU 的权重初始化，$\text{std} = \sqrt{2 / \text{fan\_out}}$

## 任务清单

### 练习 1：实现 BasicBlock 的前向传播（含跳跃连接）

**任务**：在 `BasicBlockExercise.forward()` 中实现完整的残差前向传播。

**残差块公式**：

$$
\text{out} = \text{ReLU}(\text{BN}(\text{Conv3x3}(\text{BN}(\text{Conv3x3}(x)))) + \text{shortcut}(x))
$$

**步骤提示**：

```
1. identity = self.shortcut(x)                    # 跳跃连接分支（恒等或投影）
2. out = F.relu(self.bn1(self.conv1(x)))          # Conv→BN→ReLU（第一层）
3. out = self.bn2(self.conv2(out))                # Conv→BN（第二层，不加ReLU！）
4. out += identity                                 # 🔑 残差加法 H(x) = F(x) + x
5. out = F.relu(out)                              # 加法后才做 ReLU
6. return out
```

**关键细节**：第二步的 ReLU 为什么不放在加法之后？因为两个卷积中间需要一个非线性来增强表达能力。但第二个 BN 之后不能加 ReLU，否则 $F(x)$ 被截断到 $\ge 0$，限制了残差的表达范围。

**TODO 提示**：还需要补全 `__init__` 中 shortcut 的条件构建：

```python
if stride != 1 or in_planes != planes:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes),
    )
```

**预期输出**：

- 输入 `(2, 64, 32, 32)` → 输出 `(2, 128, 32, 32)` (stride=1)
- 输入 `(2, 64, 32, 32)` → 输出 `(2, 128, 16, 16)` (stride=2)

### 练习 2：为残差块添加 BatchNorm 并理解其位置

**任务**：补全 `ResidualBlockWithBN` 的构建和前向传播。

**BatchNorm 在残差块中的正确位置**：

```
x → Conv → BN → ReLU → Conv → BN → + shortcut → ReLU
```

**思考题**：如果把 BN 放在加法之后会有什么问题？

> **答案**：shortcut 分支输出 $x$ 的分布和主路径输出 $F(x)$ 的分布在加法后混合，对这个混合结果做 BN 会破坏恒等映射的"纯度"。更重要的是，shortcut 路径没有经过 BN，而主路径经过了 BN，两者在 BN 后的分布已经不同——再加一次 BN 并没有消除而是放大了这种不一致。正确的做法是两个分支各自做完处理后直接相加。

**TODO 提示**：

```python
self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
self.bn1 = nn.BatchNorm2d(out_channels)
self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
self.bn2 = nn.BatchNorm2d(out_channels)

# shortcut：维度匹配时恒等，不匹配时 1×1 Conv + BN
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
else:
    self.shortcut = nn.Identity()
```

### 练习 3：构建 ResNet-34

**任务**：参考 demo.py 中的 `ResNet` 类，完成 `ResNet34`。

**ResNet-34 的 block 配置**：`[3, 4, 6, 3]`（共 16 个 BasicBlock，34 层）

| Layer | 输出大小 (CIFAR-10) | 通道数 | Block 数 |
|-------|---------------------|--------|----------|
| conv1  | 32×32 | 64 | — |
| layer1 | 32×32 | 64 | 3 |
| layer2 | 16×16 | 128 | 4 |
| layer3 | 8×8 | 256 | 6 |
| layer4 | 4×4 | 512 | 3 |
| avgpool+FC | 1×1 | 512→10 | — |

**对比 ResNet-18**：`[2, 2, 2, 2]`。ResNet-34 在 layer2 和 layer3 增加了深度，总层数从 18 增加到 34。

**TODO 提示**：

```python
self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
self.bn1 = nn.BatchNorm2d(64)
self.layer1 = self._make_layer(64,  3, stride=1)
self.layer2 = self._make_layer(128, 4, stride=2)
self.layer3 = self._make_layer(256, 6, stride=2)
self.layer4 = self._make_layer(512, 3, stride=2)
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(512, num_classes)
```

**`_make_layer` 方法**：

```python
def _make_layer(self, planes, num_blocks, stride):
    layers = []
    layers.append(BasicBlockExercise(self.in_planes, planes, stride))
    self.in_planes = planes  # 更新全局通道数
    for _ in range(1, num_blocks):
        layers.append(BasicBlockExercise(self.in_planes, planes, stride=1))
    return nn.Sequential(*layers)
```

**forward 方法**：`conv1→bn1→relu → layer1→layer2→layer3→layer4 → avgpool→flatten→fc`

**预期输出**：输入 `(1, 3, 32, 32)` → 输出 `(1, 10)`，参数量约 21.28M。

### 练习 4：分析参数量和 FLOPs

**任务**：手动计算三种 ResNet 的参数量。

**BasicBlock 参数量公式**（忽略 BN 的 $\gamma, \beta$）：

- 两个 $3 \times 3$ 卷积：`in_planes × planes × 9 + planes × planes × 9`
- 如果 shortcut 不是恒等：再加上 `in_planes × planes × 1`（$1 \times 1$ 卷积）

**Bottleneck 参数量公式**：

- 三个卷积：`in_planes × planes × 1 + planes × planes × 9 + planes × (4×planes) × 1`
- 如果 shortcut 不是恒等：再加上 `in_planes × (4×planes) × 1`

**各 ResNet 的配置**：

| 模型 | Block 类型 | num_blocks | 通道数序列 | 参数量（约） |
|------|-----------|------------|-----------|-------------|
| ResNet-18 | BasicBlock | [2,2,2,2] | 64→128→256→512 | 11.17M |
| ResNet-34 | BasicBlock | [3,4,6,3] | 64→128→256→512 | 21.28M |
| ResNet-50 | Bottleneck | [3,4,6,3] | 256→512→1024→2048 | 23.52M |

**提示**：别忘了初始卷积 `conv1`（`3 × 64 × 9 = 1,728`）和最后的全连接层 `fc`（`512 × num_classes`）。

**期望值**（供校验）：

```python
{
    "resnet18": 11173962,  # ~11.17M
    "resnet34": 21282122,  # ~21.28M
    "resnet50": 23520842,  # ~23.52M (with expansion=4)
}
```

## 完整代码

<<< @/snippets/s11_cnn_architectures/exercise.py
