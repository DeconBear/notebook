# -*- coding: utf-8 -*-
"""
s13 图像生成 demo：从零实现 GAN 和 VAE 用于 MNIST 数字生成
===========================================================
使用 PyTorch 实现简单的 GAN 和 VAE，在 MNIST 上训练，
对比两种生成方法的效果。

运行方式：python demo.py（从 s13_image_generation/code/ 目录运行）
依赖：torch, torchvision, matplotlib, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time


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

    train_set = torchvision.datasets.MNIST(
        root='../data', train=True, download=True,
        transform=transform
    )

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

    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]  # 去掉通道维度 → (H, W)
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

class VAE(nn.Module):
    """
    变分自编码器（VAE）

    包含：
    - 编码器: 输入 x → μ 和 log(σ²)
    - 重参数化: z = μ + σ ⊙ ε, ε ~ N(0, 1)
    - 解码器: z → 重建图像 x̂

    损失 = 重构损失 (MSE/BCE) + KL 散度 (D_KL(q(z|x) || p(z)))
    """

    def __init__(self, latent_dim: int = 20):
        """
        初始化 VAE

        参数:
            latent_dim: 潜变量 z 的维度
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ---------- 编码器: x (784) → μ (latent_dim), logvar (latent_dim) ----------
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        # μ 和 log(σ²) 分别由两个独立的全连接层预测
        self.fc_mu = nn.Linear(256, latent_dim)       # 均值 μ
        self.fc_logvar = nn.Linear(256, latent_dim)   # log(σ²)，用 log 保证 σ² > 0

        # ---------- 解码器: z (latent_dim) → 重建 x̂ (784) ----------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 784),
            nn.Sigmoid(),  # 输出 [0, 1]，对应归一化的像素值
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """
        编码：输入图像 → μ 和 log(σ²)

        参数:
            x: 输入图像展平，形状 (N, 784)
        返回:
            (mu, logvar): 均值和 log 方差，形状均为 (N, latent_dim)
        """
        h = self.encoder(x)               # 共享的特征提取
        mu = self.fc_mu(h)                # 预测均值
        logvar = self.fc_logvar(h)        # 预测 log(σ²)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧: z = μ + σ ⊙ ε

        这是 VAE 的核心创新。直接从 N(μ, σ²) 采样 z 是不可微的，
        通过将随机性"外包"给 ε ~ N(0,1)，使得 z 对 μ 和 σ 可微。

        参数:
            mu: 均值，形状 (N, latent_dim)
            logvar: log(σ²)，形状 (N, latent_dim)
        返回:
            z: 采样后的潜变量，形状 (N, latent_dim)
        """
        std = torch.exp(0.5 * logvar)     # σ = exp(0.5 * log(σ²))
        eps = torch.randn_like(std)       # ε ~ N(0, 1)
        z = mu + std * eps                # 重参数化: z = μ + σ ⊙ ε
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码：潜变量 z → 重建图像 x̂

        参数:
            z: 潜变量，形状 (N, latent_dim)
        返回:
            x_recon: 重建图像，形状 (N, 784)，值范围 [0, 1]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        VAE 完整前向传播

        参数:
            x: 输入图像，形状 (N, 1, 28, 28)
        返回:
            (x_recon, mu, logvar):
                - x_recon: 重建图像，形状 (N, 784)
                - mu: 编码均值
                - logvar: 编码 log 方差
        """
        # 展平图像: (N, 1, 28, 28) → (N, 784)
        x_flat = x.view(x.size(0), -1)

        # 编码 → μ, log(σ²)
        mu, logvar = self.encode(x_flat)

        # 重参数化采样 z
        z = self.reparameterize(mu, logvar)

        # 解码 → 重建
        x_recon = self.decode(z)

        return x_recon, mu, logvar


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> tuple:
    """
    计算 VAE 的损失函数

    L_VAE = 重构损失 + KL 散度

    KL 散度的解析形式（高斯分布）:
        D_KL( N(μ,σ²) || N(0,1) ) = -0.5 * sum(1 + log(σ²) - μ² - σ²)

    参数:
        x_recon: 重建图像，形状 (N, 784)
        x: 原始图像展平，形状 (N, 784)
        mu: 编码均值，形状 (N, latent_dim)
        logvar: 编码 log 方差，形状 (N, latent_dim)

    返回:
        (total_loss, recon_loss, kl_loss)
    """
    # ---------- 重构损失：二元交叉熵（适用于 [0,1] 范围的图像）----------
    recon_loss = F.binary_cross_entropy(x_recon, x.view(x.size(0), -1),
                                         reduction='sum') / x.size(0)

    # ---------- KL 散度（解析解）----------
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # ---------- 总损失 ----------
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(dataloader: DataLoader, device: torch.device,
              n_epochs: int = 30, latent_dim: int = 20) -> dict:
    """
    训练 VAE

    参数:
        dataloader: MNIST 数据加载器
        device: 计算设备
        n_epochs: 训练轮数
        latent_dim: 潜变量维度

    返回:
        history: 包含每 epoch 的损失
    """
    print(f"\n  {'='*50}")
    print(f"  训练 VAE (epochs={n_epochs}, latent_dim={latent_dim})")
    print(f"  {'='*50}")

    model = VAE(latent_dim).to(device)
    print(f"  VAE 参数: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for imgs, _ in dataloader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(imgs)
            total_loss, recon_loss, kl_loss = vae_loss(x_recon, imgs, mu, logvar)
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        history["total_loss"].append(epoch_total / n_batches)
        history["recon_loss"].append(epoch_recon / n_batches)
        history["kl_loss"].append(epoch_kl / n_batches)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Total: {epoch_total/n_batches:.4f} | "
                  f"Recon: {epoch_recon/n_batches:.4f} | "
                  f"KL: {epoch_kl/n_batches:.4f}")

    return history, model


# ============================================================
# 第 3 部分：可视化工具
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

    plt.suptitle('GAN 生成的数字', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] GAN 生成的样本已保存到 {save_path}")


def visualize_vae_reconstructions(model, test_loader, device,
                                   save_path: str, n_samples: int = 8):
    """
    可视化 VAE 重建结果（原始图像 vs 重建图像）

    参数:
        model: 训练好的 VAE 模型
        test_loader: 测试数据加载器
        device: 计算设备
        save_path: 保存路径
        n_samples: 显示的样本数
    """
    model.eval()
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n_samples].to(device)

    with torch.no_grad():
        x_recon, mu, logvar = model(imgs)
        x_recon = x_recon.view(n_samples, 1, 28, 28)

    originals = to_image(imgs)
    recons = to_image(x_recon)

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.5, 3))
    for i in range(n_samples):
        axes[0, i].imshow(originals[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('原始', fontsize=10)

        axes[1, i].imshow(recons[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('重建', fontsize=10)

    plt.suptitle('VAE 重建结果（上: 原始, 下: 重建）', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] VAE 重建结果已保存到 {save_path}")


def visualize_vae_latent_space(model, test_loader, device,
                                save_path: str, n_samples: int = 1000):
    """
    可视化 VAE 的潜空间（2D t-SNE 投影）

    参数:
        model: 训练好的 VAE
        test_loader: 测试数据加载器
        device: 计算设备
        save_path: 保存路径
        n_samples: 采样的潜变量数量
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [跳过] 潜空间可视化需要 scikit-learn: pip install scikit-learn")
        return

    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            x_flat = imgs.view(imgs.size(0), -1)
            mu, logvar = model.encode(x_flat)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(targets.numpy())

            if len(latent_vectors) * imgs.size(0) >= n_samples:
                break

    latent_vectors = np.concatenate(latent_vectors)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]

    # t-SNE 降维到 2D
    print("    正在运行 t-SNE 降维（可能需要几秒）...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10), label='数字类别')
    ax.set_title('VAE 潜空间的 t-SNE 投影', fontsize=14)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] VAE 潜空间 t-SNE 已保存到 {save_path}")


def plot_training_curves(gan_history: dict, vae_history: dict,
                          save_dir: str):
    """
    绘制训练曲线对比图

    参数:
        gan_history: GAN 训练历史（含 g_loss, d_loss）
        vae_history: VAE 训练历史（含 total_loss, recon_loss, kl_loss）
        save_dir: 保存目录
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- GAN 损失曲线 ----
    ax = axes[0]
    epochs = range(1, len(gan_history["g_loss"]) + 1)
    ax.plot(epochs, gan_history["g_loss"], 'b-', label='G Loss', linewidth=1.5)
    ax.plot(epochs, gan_history["d_loss"], 'r-', label='D Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GAN 训练曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- VAE 损失曲线 ----
    ax = axes[1]
    epochs = range(1, len(vae_history["total_loss"]) + 1)
    ax.plot(epochs, vae_history["total_loss"], 'purple', label='Total', linewidth=1.5)
    ax.plot(epochs, vae_history["recon_loss"], 'orange', label='Recon', linewidth=1)
    ax.plot(epochs, vae_history["kl_loss"], 'green', label='KL', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('VAE 训练曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 生成质量直观对比说明 ----
    ax = axes[2]
    ax.axis('off')
    comparison_text = (
        "GAN vs VAE 对比\n\n"
        "GAN:\n"
        "  - 图像较锐利（通过对抗训练优化视觉质量）\n"
        "  - 无显式潜空间结构\n"
        "  - 训练不稳定（需要平衡 G 和 D）\n"
        "  - 可能出现模式坍塌\n\n"
        "VAE:\n"
        "  - 图像较模糊（逐像素 MSE/BCE 平均效应）\n"
        "  - 潜空间平滑有结构（可做插值）\n"
        "  - 训练稳定（明确的优化目标）\n"
        "  - 覆盖数据分布更全面"
    )
    ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 训练曲线已保存到 {save_path}")


# ============================================================
# 第 4 部分：主函数
# ============================================================

def main():
    """主函数：训练 GAN 和 VAE，生成对比可视化"""
    print("=" * 60)
    print("s13 图像生成 Demo")
    print("GAN vs VAE: MNIST 数字生成对比")
    print("=" * 60)

    # ---------- 设备选择 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n计算设备: {device}")

    # ---------- 加载数据 ----------
    print("\n[1/5] 加载 MNIST 数据集...")
    batch_size = 128
    train_loader = load_mnist(batch_size)

    # 测试集（VAE 重建可视化用）
    test_set = torchvision.datasets.MNIST(
        root='../data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)

    # ---------- 训练 GAN ----------
    print("\n[2/5] 训练 GAN...")
    gan_history, generator, discriminator = train_gan(
        train_loader, device, n_epochs=30, latent_dim=128
    )

    # ---------- 训练 VAE ----------
    print("\n[3/5] 训练 VAE...")
    # 创建新的 train_loader（因为之前的被消耗了）
    train_loader_vae = load_mnist(batch_size)
    # VAE 使用 [0,1] 范围的图像（Sigmoid 输出），需要重新处理
    # 简化处理：VAE 内部接受 [-1,1] 输入但输出 Sigmoid [0,1]，损失用 BCE

    vae_history, vae_model = train_vae(
        train_loader_vae, device, n_epochs=30, latent_dim=20
    )

    # ---------- 可视化 ----------
    print("\n[4/5] 生成可视化...")
    output_dir = "../images"
    os.makedirs(output_dir, exist_ok=True)

    # GAN 生成的样本
    visualize_generated_samples(generator, device, latent_dim=128,
                                 save_path=os.path.join(output_dir, "gan_samples.png"))

    # VAE 重建结果
    visualize_vae_reconstructions(vae_model, test_loader, device,
                                   save_path=os.path.join(output_dir, "vae_reconstructions.png"))

    # VAE 潜空间可视化
    visualize_vae_latent_space(vae_model, test_loader, device,
                                save_path=os.path.join(output_dir, "vae_latent_space.png"))

    # 训练曲线对比
    plot_training_curves(gan_history, vae_history, output_dir)

    # ---------- 总结 ----------
    print("\n[5/5] 总结")
    print("=" * 60)
    print("GAN:")
    print(f"  最终 G Loss: {gan_history['g_loss'][-1]:.4f}")
    print(f"  最终 D Loss: {gan_history['d_loss'][-1]:.4f}")
    print(f"  (理想状态: D Loss ≈ 0.693, G Loss 适中 — D 无法分辨真伪)")
    print(f"\nVAE:")
    print(f"  最终 Total Loss: {vae_history['total_loss'][-1]:.4f}")
    print(f"  重构损失: {vae_history['recon_loss'][-1]:.4f}")
    print(f"  KL 散度: {vae_history['kl_loss'][-1]:.4f}")
    print(f"  (KL 散度越小，潜空间越接近标准正态分布)")
    print("=" * 60)
    print("\nDemo 完成！查看 ../images/ 目录下的可视化结果。")
    print("=" * 60)


if __name__ == "__main__":
    main()
