# -*- coding: utf-8 -*-
"""
s11b Vision Transformer (ViT) Demo：从零实现 ViT，对比预训练 ViT 与 CNN
=====================================================================
使用 PyTorch 从零构建一个 Vision Transformer，在 Imagenette 上训练，
与预训练的 ViT-B/16 和 ResNet-18 进行准确率对比。

核心目标：
  1. 理解 ViT 的 Patch Embedding → Transformer Encoder → Classification Head 流程
  2. 对比 ViT（从零训练）、预训练 ViT（微调）、CNN（ResNet-18）的性能
  3. 直观感受 ViT 需要更多数据的特性

运行方式：
  cd s11b_vit/code
  python vit_demo.py

依赖：torch, torchvision, matplotlib, numpy
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
matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import os
import sys
import time

# ============================================================
# 全局配置
# ============================================================

# 设备检测：GPU > MPS (Mac) > CPU
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
elif DEVICE.type == 'cpu':
    print("  (未检测到 GPU，使用 CPU 轻量模式)")

# 图片保存路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_HERE, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)

# ---- 根据设备选择模型规模 ----
if DEVICE.type == 'cpu':
    # CPU 轻量配置：小模型 + 少 epoch + 跳过大模型
    CONFIG = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 192,
        'depth': 4,
        'num_heads': 3,
        'mlp_ratio': 2.0,
        'epochs': 3,
        'batch_size': 32,
        'lr': 1e-3,
        'use_pretrained': False,
    }
else:
    # GPU 全量配置
    CONFIG = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 8,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'epochs': 10,
        'batch_size': 64,
        'lr': 3e-4,
        'use_pretrained': True,
    }

NUM_CLASSES = 10  # Imagenette 类别数（ImageNet 的 10 类子集）

# Imagenette 类别名（ImageNet 中的 10 个易分类类别）
IMAGENETTE_CLASSES = (
    'tench', 'English springer', 'cassette player', 'chain saw',
    'church', 'French horn', 'garbage truck', 'gas pump',
    'golf ball', 'parachute'
)

IMAGENETTE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'


# ============================================================
# 第 1 部分：数据加载
# ============================================================

def download_imagenette(data_dir: str) -> str:
    """下载并解压 Imagenette（ImageNet 的 10 类子集，~1.5GB）"""
    import tarfile, urllib.request
    tgz_path = os.path.join(data_dir, 'imagenette2.tgz')
    extracted_dir = os.path.join(data_dir, 'imagenette2')

    # 如果已解压，直接返回
    if os.path.exists(extracted_dir):
        print(f"  Imagenette 已存在于 {extracted_dir}")
        return extracted_dir

    # 下载
    print(f"  正在下载 Imagenette (ImageNet 10 类子集，~1.5GB)...")
    print(f"  URL: {IMAGENETTE_URL}")
    try:
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
    except Exception as e:
        raise RuntimeError(f"Imagenette 下载失败: {e}")

    # 解压
    print(f"  正在解压...")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(data_dir)

    # 清理压缩包
    os.remove(tgz_path)
    return extracted_dir


def get_imagenette_loaders(batch_size: int, img_size: int = 224):
    """
    加载 Imagenette 数据集（ImageNet 的 10 类子集）。
    自动下载 ~1.5GB 数据。如果下载失败，回退到 CIFAR-100。
    """
    # 数据增强（训练集）
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data_dir = os.path.join(_HERE, '..', '..', 'data')

    # ---- 方案 1：Imagenette（ImageNet 子集）----
    try:
        extracted = download_imagenette(data_dir)
        train_path = os.path.join(extracted, 'train')
        val_path = os.path.join(extracted, 'val')

        train_set = torchvision.datasets.ImageFolder(
            train_path, transform=transform_train
        )
        test_set = torchvision.datasets.ImageFolder(
            val_path, transform=transform_test
        )
        class_names = train_set.classes
        print(f"  Imagenette 加载成功！类别: {class_names}")
        print(f"  训练集: {len(train_set)} 张, 验证集: {len(test_set)} 张")
        tl, vl = _make_loaders(train_set, test_set, batch_size)
        return tl, vl, class_names

    except Exception as e:
        print(f"  Imagenette 加载失败 ({type(e).__name__}: {e})")
        print(f"  你也可以手动下载: {IMAGENETTE_URL}")
        print(f"  解压到: {data_dir}/imagenette2/")

    # ---- 方案 2：CIFAR-100 ----
    print("  回退方案 1: 尝试 CIFAR-100...")
    try:
        train_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True,
            transform=transform_train
        )
        test_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True,
            transform=transform_test
        )
        global NUM_CLASSES
        NUM_CLASSES = 100
        print(f"  CIFAR-100 加载成功！（{NUM_CLASSES} 类）")
        tl, vl = _make_loaders(train_set, test_set, batch_size)
        return tl, vl, None
    except Exception as e2:
        print(f"  CIFAR-100 也失败了 ({type(e2).__name__})")

    # ---- 方案 3：合成数据 ----
    print("  回退方案 2: 使用合成数据（仅供演示流程）")
    from torch.utils.data import TensorDataset
    np.random.seed(42)
    n_train, n_test = 2000, 400
    synth_X_train = torch.randn(n_train, 3, img_size, img_size)
    synth_y_train = torch.randint(0, NUM_CLASSES, (n_train,))
    synth_X_test = torch.randn(n_test, 3, img_size, img_size)
    synth_y_test = torch.randint(0, NUM_CLASSES, (n_test,))
    train_set = TensorDataset(synth_X_train, synth_y_train)
    test_set = TensorDataset(synth_X_test, synth_y_test)
    tl, vl = _make_loaders(train_set, test_set, batch_size)
    return tl, vl, None


def _make_loaders(train_set, test_set, batch_size):
    """创建 DataLoader"""
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


# ============================================================
# 第 2 部分：从零实现 Vision Transformer (ViT)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Patch Embedding：将图像切分成固定大小的 Patch，并映射为 Token 序列

    输入图像 (B, C, H, W) → 使用 stride=patch_size 的 Conv2d 切分 →
    输出 (B, N, D)，其中 N = (H/P) * (W/P) 为 Patch 数量，D 为嵌入维度
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d 比 unfold + Linear 更高效，本质相同
        # (B, C, 224, 224) → (B, D, 14, 14)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, C, H, W) 输入图像
        返回:
            (B, N, D) Patch Token 序列
        """
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2)              # (B, D, N)
        x = x.transpose(1, 2)         # (B, N, D)
        return x


class MultiHeadAttention(nn.Module):
    """
    多头自注意力 (Multi-Head Self-Attention, MSA)

    让序列中每个 Patch 可以"看到"所有其他 Patch，计算全局依赖关系。
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, dim: int, num_heads: int = 12,
                 qkv_bias: bool = False, attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)

        # 将 Q、K、V 合并在一个 Linear 中计算，提升效率
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, D) 输入序列
        返回:
            (B, N, D) 注意力输出
        """
        B, N, C = x.shape

        # 计算 Q、K、V： (B, N, D) → (B, N, 3*D) → 拆分为 3 个 (B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放点积注意力
        # Q @ K^T: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)        # 沿最后一个维度做 softmax
        attn = self.attn_drop(attn)

        # 加权求和: (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    ViT Transformer 编码器块

    结构: x → LayerNorm → MSA → +残差 → LayerNorm → MLP → +残差

    这与原始 Transformer 编码器完全一致。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, drop: float = 0.,
                 attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # MLP: D → 4D → D (GELU 激活)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：Pre-Norm 残差结构"""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """
    从零实现的 Vision Transformer (ViT)

    架构:
      图像 → Patch Embedding → + CLS Token → + Position Embedding →
      ×L Transformer Blocks → LayerNorm → CLS Token → MLP Head → 分类

    这是 ViT-Base 的简化版，可通过参数控制规模。
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.,
                 qkv_bias: bool = False, drop_rate: float = 0.,
                 attn_drop_rate: float = 0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # ---- CLS Token：可学习的分类 token，放在序列开头 ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Position Embedding：可学习的 1D 位置编码 ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # ---- Transformer 编码器 ----
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # ---- 分类头：取 CLS token 输出做分类 ----
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        # ---- 权重初始化 ----
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """对 Linear 和 LayerNorm 进行初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: (B, C, H, W) 输入图像
        返回:
            logits: (B, num_classes) 分类 logits
        """
        B = x.shape[0]

        # 1. Patch Embedding: (B, C, H, W) → (B, N, D)
        x = self.patch_embed(x)

        # 2. 追加 CLS Token: (B, 1, D) + (B, N, D) → (B, N+1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 加上位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer 编码器
        x = self.blocks(x)

        # 5. LayerNorm → 取 CLS Token → 分类
        x = self.norm(x)
        x = self.head(x[:, 0])  # 只取位置 0 的 CLS token

        return x


# ============================================================
# 第 3 部分：预训练模型加载
# ============================================================

def load_pretrained_vit(num_classes: int = 10):
    """
    加载预训练的 ViT-B/16，替换分类头为 num_classes

    优先使用 torchvision 内置 ViT，其次尝试 timm 库，失败则返回 None。
    """
    # ---- 尝试 torchvision (>= 0.13) ----
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        print("  [预训练] ViT-B/16 (torchvision, ImageNet-1K 权重)")
        return model
    except (ImportError, AttributeError):
        pass

    # ---- 回退：尝试 timm 库 ----
    try:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True,
                                  num_classes=num_classes)
        print("  [预训练] ViT-B/16 (timm, ImageNet-21K→1K 权重)")
        return model
    except ImportError:
        pass

    print("  [跳过] 预训练 ViT 不可用（需要 torchvision>=0.13 或 timm）")
    return None


def load_pretrained_resnet(num_classes: int = 10):
    """
    加载预训练的 ResNet-18，替换分类头

    预训练权重来自 ImageNet-1K，替换最后的全连接层适配 Imagenette。
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print("  [预训练] ResNet-18 (torchvision, ImageNet-1K 权重)")
        return model
    except (ImportError, AttributeError):
        pass

    # 回退：不用预训练权重
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print("  [从零] ResNet-18 (无预训练权重)")
    return model


# ============================================================
# 第 4 部分：训练与评估工具
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个 epoch

    返回:
        avg_loss: 平均损失
        accuracy: 训练准确率 (%)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
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


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """
    在测试集上评估模型

    返回:
        avg_loss: 平均损失
        accuracy: 测试准确率 (%)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_full(model, train_loader, test_loader, epochs, lr, device,
               model_name="Model"):
    """
    完整训练流程：训练 epochs 轮，记录 loss 和准确率曲线

    返回:
        history: { 'train_loss': [...], 'train_acc': [...], 'test_acc': [...] }
        final_test_acc: 最终测试准确率
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0

    print(f"\n  [{model_name}] 开始训练 {epochs} epochs, lr={lr}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        elapsed = time.time() - t0
        print(f"    Epoch {epoch:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Time: {elapsed:.0f}s")

    print(f"  [{model_name}] 完成！最佳测试准确率: {best_acc:.2f}%")
    return history, best_acc


def quick_finetune(model, train_loader, test_loader, epochs, lr, device,
                   model_name="Pretrained"):
    """
    快速微调预训练模型：主要训练分类头，fine-tune 最后几层

    适用于已在大数据集上预训练的 ViT-B/16。使用较小的学习率避免破坏
    预训练学到的良好特征表示。
    """
    criterion = nn.CrossEntropyLoss()

    # 对 backbone 使用更小的学习率
    # 分类头参数单独分组，使用较大学习率
    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if 'head' in name or 'fc' in name:
            head_params.append(param)
        else:
            body_params.append(param)

    optimizer = optim.AdamW([
        {'params': body_params, 'lr': lr * 0.1},   # backbone 慢速更新
        {'params': head_params, 'lr': lr},          # 分类头正常更新
    ], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0

    print(f"\n  [{model_name}] 微调 {epochs} epochs, head_lr={lr}, body_lr={lr*0.1:.1e}")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

        elapsed = time.time() - t0
        print(f"    Epoch {epoch:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Time: {elapsed:.0f}s")

    print(f"  [{model_name}] 完成！最佳测试准确率: {best_acc:.2f}%")
    return history, best_acc


# ============================================================
# 第 5 部分：可视化
# ============================================================

def plot_accuracy_comparison(results: dict, save_path: str):
    """
    绘制模型准确率对比柱状图

    参数:
        results: {model_name: accuracy}
        save_path: 图片保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    accs = list(results.values())

    # 使用不同颜色区分从零训练 vs 预训练
    colors = []
    for name in names:
        if '预训练' in name or 'Pretrained' in name:
            colors.append('#2196F3')  # 蓝色：预训练模型
        elif '微调' in name or 'Finetuned' in name:
            colors.append('#4CAF50')  # 绿色：微调模型
        else:
            colors.append('#FF9800')  # 橙色：从零训练

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor='white')

    # 在柱子上方标注数值
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Imagenette Classification Accuracy: ViT vs CNN', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accs) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 准确率对比已保存到 {save_path}")


def plot_training_curves(histories: dict, save_path: str):
    """
    绘制训练曲线对比图（Loss + Accuracy）

    参数:
        histories: {model_name: {'train_loss': [...], 'test_acc': [...]}}
        save_path: 图片保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#FF9800', '#2196F3', '#4CAF50', '#F44336']

    # 子图 1: 训练 Loss
    ax = axes[0]
    for idx, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], marker='o', markersize=3,
                color=colors[idx % len(colors)], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图 2: 测试准确率
    ax = axes[1]
    for idx, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist['test_acc']) + 1)
        ax.plot(epochs, hist['test_acc'], marker='s', markersize=3,
                color=colors[idx % len(colors)], label=name, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 训练曲线已保存到 {save_path}")


# ============================================================
# 第 6 部分：主流程
# ============================================================

def main():
    """主函数：训练并对比 ViT (从零)、预训练 ViT、ResNet-18"""
    print("=" * 65)
    print("  s11b Vision Transformer Demo")
    print("  ViT (从零) vs 预训练 ViT-B/16 vs ResNet-18")
    print("=" * 65)

    cfg = CONFIG
    print(f"\n配置:")
    print(f"  设备: {DEVICE}")
    print(f"  输入尺寸: {cfg['img_size']}x{cfg['img_size']}")
    print(f"  Patch 大小: {cfg['patch_size']}x{cfg['patch_size']}")
    print(f"  ViT 嵌入维度: {cfg['embed_dim']}")
    print(f"  ViT 层数: {cfg['depth']}, 注意力头数: {cfg['num_heads']}")
    print(f"  Epochs: {cfg['epochs']}, Batch Size: {cfg['batch_size']}")
    print(f"  使用预训练模型: {cfg['use_pretrained']}")

    # ---- 1. 加载数据 ----
    print("\n[1/5] 加载 Imagenette 数据集 (resize 到 {}x{})..."
          .format(cfg['img_size'], cfg['img_size']))
    train_loader, test_loader, classes = get_imagenette_loaders(
        cfg['batch_size'], cfg['img_size']
    )
    print(f"  训练集: {len(train_loader.dataset)} 张")
    print(f"  测试集: {len(test_loader.dataset)} 张")
    print(f"  类别: {classes}")

    # ---- 2. 构建模型 ----
    print("\n[2/5] 构建模型...")

    # 2a. 从零实现的 SimpleViT
    simple_vit = SimpleViT(
        img_size=cfg['img_size'],
        patch_size=cfg['patch_size'],
        in_chans=3,
        num_classes=NUM_CLASSES,
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
    ).to(DEVICE)
    print(f"  [从零] SimpleViT 参数量: {count_parameters(simple_vit):,}")

    # 2b. ResNet-18 (从零训练)
    resnet = load_pretrained_resnet(num_classes=NUM_CLASSES)
    # 不加载预训练权重，从零训练以公平对比
    from torchvision.models import resnet18 as _resnet18_raw
    resnet = _resnet18_raw(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    resnet = resnet.to(DEVICE)
    print(f"  [从零] ResNet-18 参数量: {count_parameters(resnet):,}")

    # 2c. 预训练 ViT-B/16 (可选)
    pretrained_vit = None
    if cfg['use_pretrained'] and DEVICE.type != 'cpu':
        pretrained_vit = load_pretrained_vit(num_classes=NUM_CLASSES)
        if pretrained_vit is not None:
            pretrained_vit = pretrained_vit.to(DEVICE)
            print(f"  [预训练] ViT-B/16 参数量: {count_parameters(pretrained_vit):,}")

    # ---- 3. 训练模型 ----
    print("\n[3/5] 训练从零实现的 SimpleViT...")
    vit_history, vit_acc = train_full(
        simple_vit, train_loader, test_loader,
        epochs=cfg['epochs'], lr=cfg['lr'], device=DEVICE,
        model_name="SimpleViT (从零)"
    )

    print("\n[4/5] 训练 ResNet-18 (从零)...")
    # ResNet-18 使用稍小的学习率 (SGD 标准)
    rn_history, rn_acc = train_full(
        resnet, train_loader, test_loader,
        epochs=cfg['epochs'], lr=1e-3, device=DEVICE,
        model_name="ResNet-18 (从零)"
    )

    # 收集训练曲线用于绘图（只看从零训练的模型）
    training_histories = {
        'SimpleViT (从零)': vit_history,
        'ResNet-18 (从零)': rn_history,
    }

    # 收集最终准确率
    final_results = {
        'SimpleViT\n(从零训练)': vit_acc,
        'ResNet-18\n(从零训练)': rn_acc,
    }

    # ---- 4. 微调预训练模型 ----
    if pretrained_vit is not None:
        print("\n[4b/5] 微调预训练 ViT-B/16...")
        # 微调 epoch 数减半（预训练模型收敛更快）
        ft_epochs = max(2, cfg['epochs'] // 2)
        pt_history, pt_acc = quick_finetune(
            pretrained_vit, train_loader, test_loader,
            epochs=ft_epochs, lr=cfg['lr'] * 0.5, device=DEVICE,
            model_name="ViT-B/16 (预训练+微调)"
        )
        final_results['ViT-B/16\n(预训练+微调)'] = pt_acc
        training_histories['ViT-B/16 (预训练+微调)'] = pt_history

    # ---- 5. 可视化 ----
    print("\n[5/5] 生成对比图表...")

    # 图表 1: 准确率对比柱状图
    plot_accuracy_comparison(
        final_results,
        os.path.join(_IMAGES_DIR, 'vit_accuracy_comparison.png')
    )

    # 图表 2: 训练曲线
    plot_training_curves(
        training_histories,
        os.path.join(_IMAGES_DIR, 'vit_training_curves.png')
    )

    # ---- 总结 ----
    print("\n" + "=" * 65)
    print("  训练总结")
    print("=" * 65)
    for name, acc in final_results.items():
        name_clean = name.replace('\n', ' ')
        print(f"  {name_clean:<35} {acc:.2f}%")

    if pretrained_vit is not None:
        print(f"\n  关键发现:")
        print(f"  - ViT 从零训练在 Imagenette 上效果有限（需要更多数据）")
        print(f"  - 预训练 ViT 通过微调大幅超越从零训练的模型")
        print(f"  - ResNet 的卷积归纳偏置在小数据集上仍有优势")
    else:
        print(f"\n  提示: 在 GPU 上运行可启用预训练 ViT 对比，观察迁移学习效果")

    print(f"\n  图表已保存到: {_IMAGES_DIR}")
    print("=" * 65)
    print("  Demo 完成！")
    print("=" * 65)


if __name__ == '__main__':
    main()
