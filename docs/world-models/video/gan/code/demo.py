# -*- coding: utf-8 -*-
"""
路径一 · GAN demo：MNIST 上的小型生成对抗网络
===========================================================
只训练 GAN。VAE 见同路径下一章。

运行方式：python demo.py
依赖：torch, torchvision, matplotlib, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# GPU 自动检测
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("（未检测到 GPU，使用 CPU 运行。如有 GPU，请安装 CUDA 版 PyTorch 以获得加速）")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import os
import time

# 图片保存目录：固定为本章节的 images/ 目录（相对于本脚本的 ../images/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGES_DIR = os.path.join(_SCRIPT_DIR, '..', 'images')
os.makedirs(_IMAGES_DIR, exist_ok=True)


# ============================================================
# 第 0 部分：通用工具
# ============================================================

def load_mnist(batch_size: int = 128) -> DataLoader:
    """
    加载 MNIST 数据集

    参数:
        batch_size: 批大小
    返回:
        train_loader: 训练数据加载器（仅含图像，不需要标签）
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # MNIST 图像是 [0,1]，GAN 的 tanh 输出是 [-1,1]
        # 因此将图像归一化到 [-1, 1]
        transforms.Normalize((0.5,), (0.5,)),
    ])

    try:
        train_set = torchvision.datasets.MNIST(
            root='../data', train=True, download=True,
            transform=transform
        )
    except Exception as e:
        print(f"[警告] MNIST 下载失败 ({e})，使用合成数据")
        # 回退：合成 28x28 单通道图像
        from torch.utils.data import TensorDataset
        np.random.seed(42)
        synth_X = torch.rand(10000, 1, 28, 28) * 2 - 1  # [-1, 1]
        synth_y = torch.randint(0, 10, (10000,))
        train_set = TensorDataset(synth_X, synth_y)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True
    )

    return train_loader


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    将张量转换为可显示的图像数组

    参数:
        tensor: 形状 (C, H, W) 或 (N, C, H, W)，值范围 [-1, 1]
    返回:
        img: numpy 数组，值范围 [0, 1]
    """
    # 反归一化: [-1, 1] → [0, 1]
    img = (tensor.detach().cpu().numpy() + 1) / 2.0
    img = np.clip(img, 0, 1)

    if img.ndim == 4 and img.shape[1] == 1:
        img = img[:, 0, :, :]  # (N, 1, H, W) → (N, H, W)
    elif img.ndim == 3 and img.shape[0] == 1:
        img = img[0]  # (1, H, W) → (H, W)
    return img


# ============================================================
# 第 1 部分：GAN —— 生成对抗网络
# ============================================================

class Generator(nn.Module):
    """
    GAN 生成器

    将随机噪声 z 映射为一张 28×28 的 MNIST 图像。

    架构: FC(128→256)→BN→ReLU → FC(256→512)→BN→ReLU → FC(512→784)→Tanh
    最后 reshape 为 (1, 28, 28)，Tanh 输出范围 [-1, 1] 与归一化的图像匹配。
    """

    def __init__(self, latent_dim: int = 128):
        """
        初始化生成器

        参数:
            latent_dim: 输入噪声 z 的维度
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # ---------- 构建全连接网络：逐步放大维度 ----------
        self.model = nn.Sequential(
            # Block 1: latent_dim → 256
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),          # BN 稳定训练，加速收敛
            nn.ReLU(inplace=True),
            # Block 2: 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Block 3: 512 → 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # Block 4: 1024 → 784 (MNIST 像素数)
            nn.Linear(1024, 784),
            nn.Tanh(),  # 输出范围 [-1, 1]，与 MNIST 归一化一致
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        前向传播：噪声 → 图像

        参数:
            z: 随机噪声，形状 (N, latent_dim)
        返回:
            img: 生成的图像，形状 (N, 1, 28, 28)，值范围 [-1, 1]
        """
        img = self.model(z)         # (N, 784)
        img = img.view(-1, 1, 28, 28)  # reshape 为图像形状
        return img


class Discriminator(nn.Module):
    """
    GAN 判别器

    判断输入图像是真实图像（来自 MNIST）还是生成器伪造的假图像。

    架构: FC(784→512)→LeakyReLU → FC(512→256)→LeakyReLU → FC(256→1)→Sigmoid
    输出一个 [0, 1] 的标量，表示图像为真的概率。
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Block 1: 784 → 512
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),  # 使用 LeakyReLU 防止 dead neurons
            # Block 2: 512 → 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3: 256 → 1
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出概率 [0, 1]
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        前向传播：图像 → 真实性概率

        参数:
            img: 输入图像，形状 (N, 1, 28, 28) 或 (N, 784)
        返回:
            validity: 图像为真的概率，形状 (N, 1)
        """
        # 展平图像: (N, 1, 28, 28) → (N, 784)
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def train_gan(dataloader: DataLoader, device: torch.device,
              n_epochs: int = 50, latent_dim: int = 128) -> dict:
    """
    训练 GAN

    参数:
        dataloader: MNIST 数据加载器
        device: 计算设备
        n_epochs: 训练轮数
        latent_dim: 潜变量维度

    返回:
        history: 包含每 epoch 的 G loss 和 D loss
    """
    print(f"\n  {'='*50}")
    print(f"  训练 GAN (epochs={n_epochs})")
    print(f"  {'='*50}")

    # ---------- 初始化模型 ----------
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    print(f"  Generator 参数: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator 参数: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ---------- 损失函数和优化器 ----------
    adversarial_loss = nn.BCELoss()  # 二元交叉熵损失

    # 两个独立的优化器（交替训练）
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # ---------- 训练循环 ----------
    history = {"g_loss": [], "d_loss": []}
    fixed_noise = torch.randn(16, latent_dim, device=device)  # 用于定期可视化

    for epoch in range(1, n_epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)

            # 创建标签（真实=1，假=0）
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ========== 训练判别器 D ==========
            optimizer_D.zero_grad()

            # 真实图像的损失：D(real_img) → 1
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_labels)

            # 假图像的损失：D(G(z)) → 0
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)  # 生成假图像
            fake_pred = discriminator(fake_imgs.detach())  # detach() 防止梯度传回 G
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)

            # 判别器的总损失 = 真实损失 + 假损失
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # ========== 训练生成器 G ==========
            optimizer_G.zero_grad()

            # 生成器的目标：让判别器认为假图像是真的 D(G(z)) → 1
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            gen_pred = discriminator(gen_imgs)  # 注意：这里不用 detach()
            g_loss = adversarial_loss(gen_pred, real_labels)  # 目标是"真实"

            g_loss.backward()
            optimizer_G.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        # 记录 epoch 平均损失
        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    return history, generator, discriminator


# ============================================================
# 第 2 部分：VAE —— 变分自编码器
# ============================================================

def visualize_generated_samples(generator, device, latent_dim,
                                 save_path: str, n_samples: int = 16):
    """
    可视化 GAN 生成的图像样本

    参数:
        generator: 训练好的 GAN 生成器
        device: 计算设备
        latent_dim: 潜变量维度
        save_path: 保存路径
        n_samples: 生成的样本数
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        samples = generator(z)
        samples = to_image(samples)  # (N, H, W)

    # 排列为网格
    ncols = 4
    nrows = int(np.ceil(n_samples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            ax.imshow(samples[i], cmap='gray')
        ax.axis('off')

    plt.suptitle('GAN Generated Digits', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] GAN 生成的样本已保存到 {save_path}")



def plot_gan_curves(history, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["g_loss"], label="G loss")
    ax.plot(history["d_loss"], label="D loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("GAN 训练曲线")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_path}")


def main():
    print("=" * 60)
    print("路径一 · GAN Demo")
    print("=" * 60)
    device = DEVICE
    print(f"计算设备: {device}")
    train_loader = load_mnist(128)
    if device.type == "cpu":
        n_gan_epochs = 2
        n_train_subset = 1000
        print("[配置] CPU：GAN 2 epochs, 1000 样本")
        if hasattr(train_loader.dataset, "data"):
            train_loader.dataset.data = train_loader.dataset.data[:n_train_subset]
            if hasattr(train_loader.dataset, "targets"):
                train_loader.dataset.targets = train_loader.dataset.targets[:n_train_subset]
    else:
        n_gan_epochs = 30
    gan_history, generator, discriminator = train_gan(
        train_loader, device, n_epochs=n_gan_epochs, latent_dim=128
    )
    visualize_generated_samples(
        generator, device, latent_dim=128,
        save_path=os.path.join(_IMAGES_DIR, "gan_samples.png"),
    )
    plot_gan_curves(gan_history, os.path.join(_IMAGES_DIR, "training_curves.png"))
    print(f"最终 G={gan_history['g_loss'][-1]:.4f} D={gan_history['d_loss'][-1]:.4f}")
    print(f"图片目录 {_IMAGES_DIR}")


if __name__ == "__main__":
    main()
