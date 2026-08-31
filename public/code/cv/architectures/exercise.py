# -*- coding: utf-8 -*-
"""
s11 经典架构演进 练习
=====================
完成以下 TODO 练习来加深对 ResNet 架构的理解。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# ============================================================
# 练习 1：实现 BasicBlock 前向传播（含跳跃连接）
# ============================================================

class BasicBlockExercise(nn.Module):
    """
    TODO: 完成 BasicBlock 的前向传播实现

    残差块公式: out = ReLU( Conv→BN→ReLU→Conv→BN(x) + shortcut(x) )

    这是 ResNet 最核心的计算单元。你需要理解：
    1. 主路径: 两个 3×3 卷积，每个后面跟 BN
    2. 跳跃连接: 如果 in_planes != planes 或 stride != 1，用 1×1 卷积对齐
    3. 残差加法: F(x) + identity 在 ReLU 之前
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        初始化 BasicBlock

        参数:
            in_planes: 输入通道数
            planes: 输出通道数（中间和目标通道都是它）
            stride: 步长（下采样用）
        """
        super(BasicBlockExercise, self).__init__()

        # 主路径的两个卷积和 BN
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 跳跃连接：如果维度不匹配，需要用 1x1 卷积调整
        self.shortcut = nn.Sequential()
        # TODO: 当 in_planes != planes 或 stride != 1 时，
        #       添加 1×1 Conv + BN 来匹配维度
        # if stride != 1 or in_planes != planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(???, ???, kernel_size=1, stride=???, bias=False),
        #         nn.BatchNorm2d(???),
        #     )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: 实现残差块的前向传播

        参数:
            x: 输入张量，形状 (N, in_planes, H, W)
        返回:
            out: 输出张量，形状 (N, planes, H//stride, W//stride)

        步骤提示:
        1. identity = self.shortcut(x)  # 恒等映射或投影
        2. out = F.relu(self.bn1(self.conv1(x)))  # 第一个 Conv → BN → ReLU
        3. out = self.bn2(self.conv2(out))  # 第二个 Conv → BN（暂不加 ReLU）
        4. out += identity  # 跳跃连接加法: H(x) = F(x) + x
        5. out = F.relu(out)  # 最后的 ReLU
        6. return out
        """
        # TODO: 写下你的代码
        pass


# ============================================================
# 练习 2：为残差块添加 BatchNorm 并理解其位置
# ============================================================

class ResidualBlockWithBN(nn.Module):
    """
    TODO: 分析 BatchNorm 在残差块中的正确位置

    BN 通常放在卷积之后、ReLU 之前，即 Conv → BN → ReLU。
    请完成以下 block 的构建，回答：为什么 BN 要放在加法之前而不是之后？

    残差块中 BN 的正确位置:
        x → Conv → BN → ReLU → Conv → BN → + shortcut → ReLU

    如果把 BN 放在加法之后会有什么问题？
    (提示: 考虑 shortcut 路径的 BN 对恒等映射的影响)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlockWithBN, self).__init__()

        # TODO: 补全以下构建代码
        self.conv1 = None  # TODO: 3×3 Conv, in_channels→out_channels
        self.bn1 = None    # TODO: BatchNorm2d(out_channels)
        self.conv2 = None  # TODO: 3×3 Conv, out_channels→out_channels
        self.bn2 = None    # TODO: BatchNorm2d(out_channels)

        # TODO: 跳跃连接（维度匹配时用恒等，不匹配时用 1×1 Conv + BN）
        self.shortcut = nn.Identity()  # 占位，替换为正确的 shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: 实现带正确 BN 位置的前向传播
        """
        pass


# ============================================================
# 练习 3：构建 ResNet-34
# ============================================================

def build_resnet34(num_classes: int = 10) -> nn.Module:
    """
    TODO: 参考 ResNet-18 的构建方式，完成 ResNet-34

    ResNet-34 使用 BasicBlock，每个 layer 的 block 数量为:
        [3, 4, 6, 3]  (总计 16 个残差块，34 层)

    对比 ResNet-18 的 [2, 2, 2, 2]:
        - layer1: 2→3 个 block，通道数 64
        - layer2: 2→4 个 block，通道数 128
        - layer3: 2→6 个 block，通道数 256
        - layer4: 2→3 个 block，通道数 512

    需要实现的组件:
        1. 初始卷积: Conv2d(3, 64, kernel=3, stride=1, padding=1, bias=False) + BN + ReLU
        2. _make_layer 方法: 构建一个包含多个 BasicBlock 的 layer
        3. 全局平均池化: AdaptiveAvgPool2d((1, 1))
        4. 分类头: Linear(512, num_classes)

    提示: 参考 demo.py 中的 ResNet 类实现
    """

    class ResNet34(nn.Module):
        def __init__(self):
            super(ResNet34, self).__init__()
            self.in_planes = 64  # 初始通道数

            # TODO: 初始卷积
            self.conv1 = None  # Conv2d(3, 64, 3, 1, 1)
            self.bn1 = None    # BatchNorm2d(64)

            # TODO: 4 个残差层
            # self.layer1 = self._make_layer(64, 3, stride=1)
            # self.layer2 = self._make_layer(128, 4, stride=2)
            # self.layer3 = self._make_layer(256, 6, stride=2)
            # self.layer4 = self._make_layer(512, 3, stride=2)

            # TODO: 全局平均池化 + 全连接
            self.avgpool = None  # AdaptiveAvgPool2d((1, 1))
            self.fc = None       # Linear(512, num_classes)

        def _make_layer(self, planes: int, num_blocks: int,
                        stride: int) -> nn.Sequential:
            """
            TODO: 构建一个残差层

            参数:
                planes: 该层的输出通道数
                num_blocks: 该层包含的 block 个数
                stride: 第一个 block 的步长

            返回:
                nn.Sequential 包装的残差层

            提示:
                1. 第一个 block 使用给定 stride 和 in_planes→planes
                2. 更新 self.in_planes = planes * BasicBlock.expansion (= planes)
                3. 后续 block 使用 stride=1
                4. 层层包装到 layers 列表中
            """
            layers = []
            # TODO: 添加第一个 block (stride 可能需要下采样)
            # TODO: 添加剩余 num_blocks-1 个 block (stride=1)
            return nn.Sequential(*layers)

        def forward(self, x):
            """TODO: 实现前向传播"""
            # x = F.relu(self.bn1(self.conv1(x)))
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)
            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            # x = self.fc(x)
            # return x
            pass

    return ResNet34()


# ============================================================
# 练习 4：分析参数量和 FLOPs
# ============================================================

def analyze_resnet_parameters() -> dict:
    """
    TODO: 计算并返回不同 ResNet 变体的参数量

    基于以下信息（忽略 BN 参数，近似计算）:

    BasicBlock (in_planes, planes, stride=1):
        参数 = in_planes*planes*9 + planes*planes*9

    BasicBlock 带下采样 (stride=2 或 in_planes != planes):
        参数 = 上述 + shortcut 1×1 Conv 的 in_planes*planes*1

    各 ResNet 的 block 配置:
        ResNet-18: [2, 2, 2, 2], 通道数 [64, 128, 256, 512]
        ResNet-34: [3, 4, 6, 3], 通道数 [64, 128, 256, 512]
        ResNet-50: [3, 4, 6, 3], 通道数 [256, 512, 1024, 2048] (用 Bottleneck)

    返回:
        dict: {"resnet18": 参数数量, "resnet34": 参数数量, "resnet50": 参数数量}

    提示: 也要记得加上初始 conv1 (3*64*9) 和最后的 fc (512*num_classes)
    """
    # TODO: 逐层计算参数量
    params = {
        "resnet18": 0,  # 期望值: ~11.17M
        "resnet34": 0,  # 期望值: ~21.28M
        "resnet50": 0,  # 期望值: ~23.52M (含 bottleneck expansion)
    }
    return params


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("s11 经典架构演进 — 练习测试")
    print("=" * 50)

    # ---- 测试练习 1：BasicBlock 前向传播 ----
    print("\n[练习 1] BasicBlock 前向传播测试：")
    try:
        block = BasicBlockExercise(64, 128, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print(f"  期望输出形状: torch.Size([2, 128, 32, 32])")
        print(f"  {'✓ 通过' if out.shape == (2, 128, 32, 32) else '✗ 失败'}")
    except Exception as e:
        print(f"  测试异常: {e}")
        print("  (请完成 forward 方法中的 TODO)")

    # ---- 测试下采样情况 ----
    print("\n  BasicBlock 下采样测试 (stride=2)：")
    try:
        block = BasicBlockExercise(64, 128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print(f"  期望输出形状: torch.Size([2, 128, 16, 16])")
        print(f"  {'✓ 通过' if out.shape == (2, 128, 16, 16) else '✗ 失败'}")
    except Exception as e:
        print(f"  测试异常: {e}")

    # ---- 测试练习 3：ResNet-34 构建 ----
    print("\n[练习 3] ResNet-34 构建测试：")
    try:
        model = build_resnet34(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ResNet-34 总参数量: {total_params:,}")
        print(f"  期望约: ~21,280,000")
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {out.shape}")
        print(f"  期望输出形状: torch.Size([1, 10])")
        print(f"  {'✓ 通过' if out.shape == (1, 10) else '✗ 失败'}")
    except Exception as e:
        print(f"  测试异常: {e}")
        print("  (请完成 build_resnet34 中的 TODO)")

    # ---- 练习 4：参数计算 ----
    print("\n[练习 4] 参数量分析：")
    param_estimates = analyze_resnet_parameters()
    for name, count in param_estimates.items():
        print(f"  {name}: {count:,} 参数")

    print("\n" + "=" * 50)
    print("完成所有练习后，运行 demo.py 查看完整的训练对比实验。")
    print("=" * 50)
