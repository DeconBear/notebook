# -*- coding: utf-8 -*-
"""
s11 经典架构演进 demo：ResNet-18 从零实现与训练
================================================
使用 PyTorch 从零构建 ResNet-18，在 CIFAR-10 上训练，
与同深度的普通 CNN（无跳跃连接）进行对比。

运行方式：python demo.py（从 s11_cnn_architectures/code/ 目录运行）
依赖：torch, torchvision, matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import defaultdict

# ============================================================
# 第 1 部分：ResNet 组件实现
# ============================================================

class BasicBlock(nn.Module):
    """
    ResNet 基本残差块（BasicBlock）

    结构: Conv3x3 → BN → ReLU → Conv3x3 → BN → + skip → ReLU

    用于 ResNet-18 和 ResNet-34。
    每个卷积后都有 BatchNorm，通过 skip connection 缓解梯度消失。
    """

    expansion = 1  # BasicBlock 不扩展通道数

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        初始化 BasicBlock

        参数:
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 步长（用于下采样）
        """
        super(BasicBlock, self).__init__()

        # ---------- 第一个 3×3 卷积 ----------
        # padding=1 保持空间尺寸，stride 控制下采样
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化，加速收敛

        # ---------- 第二个 3×3 卷积 ----------
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # ---------- 跳跃连接（Skip Connection）----------
        # 当输入输出维度不匹配时（stride != 1 或通道变化），
        # 需要 1×1 卷积调整 shortcut 的维度和尺寸
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 主路径 F(x) + 跳跃连接 x

        参数:
            x: 输入张量，形状 (N, in_planes, H, W)
        返回:
            out: 输出张量，形状 (N, planes, H/stride, W/stride)
        """
        # ---------- 主路径：Conv → BN → ReLU → Conv → BN ----------
        identity = self.shortcut(x)  # 跳跃连接（恒等或投影）

        out = self.conv1(x)          # 3×3 卷积
        out = self.bn1(out)          # 批归一化
        out = F.relu(out)            # ReLU 激活

        out = self.conv2(out)        # 3×3 卷积
        out = self.bn2(out)          # 批归一化

        # ---------- 残差连接：F(x) + x ----------
        out += identity               # H(x) = F(x) + x
        out = F.relu(out)             # 最后的 ReLU

        return out


class Bottleneck(nn.Module):
    """
    ResNet 瓶颈残差块（Bottleneck）

    结构: Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN → + skip → ReLU

    用于 ResNet-50/101/152。通过 1×1 卷积先降维再升维，降低计算量。
    """

    expansion = 4  # Bottleneck 将通道扩展 4 倍

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        初始化 Bottleneck Block

        参数:
            in_planes: 输入通道数
            planes: 瓶颈通道数（中间层的通道数，不算 expansion）
            stride: 步长
        """
        super(Bottleneck, self).__init__()

        # 1×1 降维: in_planes → planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3×3 卷积: planes → planes（主要计算发生在这里）
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1×1 升维: planes → planes*expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 瓶颈 + 跳跃连接"""
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))  # 1×1 降维
        out = F.relu(self.bn2(self.conv2(out)))  # 3×3 卷积
        out = self.bn3(self.conv3(out))  # 1×1 升维（不加 ReLU）

        out += identity
        out = F.relu(out)

        return out


# ============================================================
# 第 2 部分：ResNet 完整模型
# ============================================================

class ResNet(nn.Module):
    """
    ResNet 完整模型

    支持 ResNet-18, 34, 50, 101, 152 各变体。

    架构:
        - 初始卷积: 3×3 Conv, 64通道, stride=1 (适配 CIFAR)
        - 4 个 layer，每个 layer 包含若干个 block
        - 全局平均池化 → 全连接
    """

    def __init__(self, block: nn.Module, num_blocks: list,
                 num_classes: int = 10):
        """
        初始化 ResNet

        参数:
            block: 残差块类型（BasicBlock 或 Bottleneck）
            num_blocks: 每个 layer 的 block 数量，如 [2,2,2,2] 对应 ResNet-18
            num_classes: 分类类别数（CIFAR-10 为 10）
        """
        super(ResNet, self).__init__()
        self.in_planes = 64  # 初始通道数

        # ---------- 初始卷积层 ----------
        # CIFAR-10 图像较小 (32×32)，使用 stride=1, padding=1，不做大幅下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ---------- 4 个残差层 ----------
        # 每个 layer 的 planes 逐层翻倍，stride 在第二个 layer 开始使用 2 实现下采样
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ---------- 分类头 ----------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化 → 1×1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ---------- 权重初始化 ----------
        self._initialize_weights()

    def _make_layer(self, block: nn.Module, planes: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        构建一个残差层（包含多个 block）

        参数:
            block: 残差块类型
            planes: 该层的输出通道数（BasicBlock）或瓶颈通道数（Bottleneck）
            num_blocks: 该层包含的 block 数量
            stride: 第一个 block 的步长（用于下采样）
        返回:
            Sequential 包装的残差层
        """
        layers = []
        # 第一个 block 可能进行下采样
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion

        # 后续 block 保持尺寸
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming 初始化：适用于 ReLU 的权重初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ResNet 前向传播

        参数:
            x: 输入图像，形状 (N, 3, 32, 32)
        返回:
            out: 分类 logits，形状 (N, num_classes)
        """
        # 初始卷积 + BN + ReLU
        x = F.relu(self.bn1(self.conv1(x)))

        # 4 个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化 + 分类
        x = self.avgpool(x)  # (N, 512*expansion, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (N, 512*expansion)
        x = self.fc(x)
        return x


def ResNet18(num_classes: int = 10) -> ResNet:
    """构建 ResNet-18（BasicBlock × 2×4=8 个，2+2+2+2）"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes: int = 10) -> ResNet:
    """构建 ResNet-34（BasicBlock × 3+4+6+3=16 个）"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


# ============================================================
# 第 3 部分：无跳跃连接的普通 CNN（对照模型）
# ============================================================

class PlainBlock(nn.Module):
    """
    普通卷积块（无跳跃连接）

    结构与 BasicBlock 相同但去掉了 skip connection。
    用于展示深度网络不加残差连接时的退化现象。
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 无 shortcut！这就是与 BasicBlock 的唯一区别

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（无残差连接）

        参数:
            x: 输入张量
        返回:
            out: 输出张量
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 注意：这里没有 out += x（去掉了跳跃连接）
        out = F.relu(out)
        return out


class PlainCNN(nn.Module):
    """
    与 ResNet-18 同深度的普通 CNN（无跳跃连接）

    架构完全相同，只是将 BasicBlock 替换为 PlainBlock。
    """

    def __init__(self, num_blocks: list, num_classes: int = 10):
        super(PlainCNN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        """构建普通卷积层"""
        layers = []
        # 第一个 block 需要调整维度（因为去掉 shortcut 后需要手动匹配）
        if stride != 1 or self.in_planes != planes:
            layers.append(nn.Conv2d(self.in_planes, planes, kernel_size=1,
                                     stride=stride, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU())
        layers.append(PlainBlock(planes, planes, stride=1))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# 第 4 部分：训练与评估工具
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TrainingLogger:
    """训练日志记录器：记录 loss、准确率、梯度范数"""

    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.grad_norms = []  # 每层的梯度范数

    def update(self, train_loss: float, train_acc: float,
               test_acc: float, grad_norms: dict = None):
        """记录一轮训练的数据"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        if grad_norms is not None:
            self.grad_norms.append(grad_norms)


def compute_gradient_norms(model: nn.Module) -> dict:
    """
    计算模型各层的梯度范数（用于诊断梯度消失/爆炸）

    返回:
        grad_norms: {层名: 梯度的 L2 范数}
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


def train_one_epoch(model: nn.Module, train_loader, optimizer,
                    criterion, device: torch.device) -> tuple:
    """
    训练一个 epoch

    参数:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备

    返回:
        (平均 loss, 准确率)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, test_loader, device: torch.device) -> float:
    """
    在测试集上评估模型

    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 计算设备

    返回:
        测试准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def get_cifar10_loaders(batch_size: int = 128):
    """
    加载 CIFAR-10 数据集

    返回:
        train_loader, test_loader, classes
    """
    # 训练数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 随机裁剪 + 填充
        transforms.RandomHorizontalFlip(),         # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10 均值
                             (0.2023, 0.1994, 0.2010)),  # CIFAR-10 标准差
    ])

    # 测试数据：仅归一化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True,
        transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root='../data', train=False, download=True,
        transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


# ============================================================
# 第 5 部分：可视化工具
# ============================================================

def plot_training_comparison(logger_resnet: TrainingLogger,
                              logger_plain: TrainingLogger,
                              save_dir: str):
    """
    绘制 ResNet vs Plain CNN 的训练对比图

    生成三张子图：(1) 训练 Loss, (2) 测试准确率, (3) 梯度范数均值
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(logger_resnet.train_losses) + 1)

    # 子图 1: 训练 Loss
    ax = axes[0]
    ax.plot(epochs, logger_resnet.train_losses, 'b-o', label='ResNet-18',
            linewidth=2, markersize=4)
    ax.plot(epochs, logger_plain.train_losses, 'r-s', label='Plain CNN',
            linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('训练 Loss')
    ax.set_title('训练 Loss 对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 2: 测试准确率
    ax = axes[1]
    ax.plot(epochs, logger_resnet.test_accs, 'b-o', label='ResNet-18',
            linewidth=2, markersize=4)
    ax.plot(epochs, logger_plain.test_accs, 'r-s', label='Plain CNN',
            linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('测试准确率 (%)')
    ax.set_title('测试准确率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图 3: 平均梯度范数（取所有层梯度的均值）
    ax = axes[2]
    if logger_resnet.grad_norms and logger_plain.grad_norms:
        resnet_avg_grads = [np.mean(list(g.values()))
                            for g in logger_resnet.grad_norms]
        plain_avg_grads = [np.mean(list(g.values()))
                           for g in logger_plain.grad_norms]
        ax.plot(epochs, resnet_avg_grads, 'b-o', label='ResNet-18',
                linewidth=2, markersize=4)
        ax.plot(epochs, plain_avg_grads, 'r-s', label='Plain CNN',
                linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('平均梯度范数')
        ax.set_title('梯度范数对比（残差连接保持梯度流动）')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'resnet_vs_plain.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 训练对比图已保存到 {save_path}")


def plot_gradient_distribution(grad_norms: dict, epoch: int,
                                save_dir: str, prefix: str = ""):
    """
    绘制单轮中各层梯度范数的分布图（诊断哪层梯度消失）

    参数:
        grad_norms: {层名: 范数}
        epoch: 当前 epoch
        save_dir: 保存目录
        prefix: 文件名前缀（区分 ResNet 和 Plain）
    """
    names = list(grad_norms.keys())
    values = list(grad_norms.values())

    if len(values) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(names)))
    # 简化层名（只显示最后一部分）
    short_names = [n.split('.')[-1] for n in names]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('梯度 L2 范数')
    ax.set_title(f'{prefix} Epoch {epoch} — 逐层梯度范数分布')
    ax.axhline(y=0, color='red', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = os.path.join(save_dir, f'grads_{prefix}_epoch{epoch:02d}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()


# ============================================================
# 第 6 部分：主训练流程
# ============================================================

def main():
    """主函数：训练 ResNet-18 和 Plain CNN，对比分析"""
    print("=" * 60)
    print("s11 经典架构演进 Demo")
    print("ResNet-18 vs Plain CNN 对比实验")
    print("=" * 60)

    # ---------- 设备选择 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---------- 加载数据 ----------
    print("\n[1/5] 加载 CIFAR-10 数据集...")
    batch_size = 128
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size)
    print(f"  训练集: 50000 张, 测试集: 10000 张")
    print(f"  类别: {classes}")

    # ---------- 构建模型 ----------
    print("\n[2/5] 构建模型...")
    n_epochs = 10  # demo 模式训练 10 个 epoch

    resnet = ResNet18(num_classes=10).to(device)
    plain_cnn = PlainCNN([2, 2, 2, 2], num_classes=10).to(device)

    print(f"  ResNet-18 参数量: {count_parameters(resnet):,}")
    print(f"  Plain CNN 参数量: {count_parameters(plain_cnn):,}")

    # ---------- 优化器和损失函数 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.1,
                                 momentum=0.9, weight_decay=5e-4)
    optimizer_plain = optim.SGD(plain_cnn.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)

    # 学习率调度：每 30 epoch 降低到原来的 0.1（这里简化）
    scheduler_resnet = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_resnet, T_max=n_epochs
    )
    scheduler_plain = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_plain, T_max=n_epochs
    )

    # ---------- 训练 ----------
    print("\n[3/5] 训练 ResNet-18...")
    logger_resnet = TrainingLogger()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            resnet, train_loader, optimizer_resnet, criterion, device
        )
        test_acc = evaluate(resnet, test_loader, device)
        scheduler_resnet.step()

        # 记录梯度范数
        grad_norms = compute_gradient_norms(resnet)

        logger_resnet.update(train_loss, train_acc, test_acc, grad_norms)

        print(f"  Epoch {epoch:2d}/{n_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    print("\n[4/5] 训练 Plain CNN（无跳跃连接）...")
    logger_plain = TrainingLogger()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            plain_cnn, train_loader, optimizer_plain, criterion, device
        )
        test_acc = evaluate(plain_cnn, test_loader, device)
        scheduler_plain.step()

        grad_norms = compute_gradient_norms(plain_cnn)
        logger_plain.update(train_loss, train_acc, test_acc, grad_norms)

        print(f"  Epoch {epoch:2d}/{n_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    # ---------- 可视化 ----------
    print("\n[5/5] 生成可视化对比...")
    output_dir = "../images"
    os.makedirs(output_dir, exist_ok=True)

    # 训练过程对比图
    plot_training_comparison(logger_resnet, logger_plain, output_dir)

    # 最后一轮的梯度分布
    if logger_resnet.grad_norms:
        plot_gradient_distribution(
            logger_resnet.grad_norms[-1],
            epoch=n_epochs, save_dir=output_dir,
            prefix="resnet"
        )
    if logger_plain.grad_norms:
        plot_gradient_distribution(
            logger_plain.grad_norms[-1],
            epoch=n_epochs, save_dir=output_dir,
            prefix="plain"
        )

    # ---------- 总结 ----------
    print("\n" + "=" * 60)
    print("训练总结:")
    print(f"  ResNet-18    最终测试准确率: {logger_resnet.test_accs[-1]:.2f}%")
    print(f"  Plain CNN    最终测试准确率: {logger_plain.test_accs[-1]:.2f}%")
    print(f"  ResNet 优势:   {logger_resnet.test_accs[-1] - logger_plain.test_accs[-1]:.2f}%")

    if logger_resnet.grad_norms:
        resnet_avg_grad = np.mean(list(logger_resnet.grad_norms[-1].values()))
        plain_avg_grad = np.mean(list(logger_plain.grad_norms[-1].values()))
        print(f"  ResNet 平均梯度范数: {resnet_avg_grad:.6f}")
        print(f"  Plain 平均梯度范数: {plain_avg_grad:.6f}")
        print(f"  (ResNet 的梯度范数更大，说明残差连接有效)
")

    print("=" * 60)
    print("Demo 完成！查看 ../images/ 目录下的可视化结果。")
    print("=" * 60)


if __name__ == "__main__":
    main()
