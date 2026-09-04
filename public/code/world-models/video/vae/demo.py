# -*- coding: utf-8 -*-
"""
路径一 · VAE demo：MNIST 变分自编码器
===========================================================
只训练 VAE。GAN 见同路径上一章。

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
    # MNIST 图像经过 Normalize((0.5,),(0.5,)) 后值域为 [-1, 1]，
    # 而 VAE 解码器的 Sigmoid 输出为 [0, 1]，BCE 要求 target ∈ [0,1]，
    # 因此需要将目标图像从 [-1, 1] 反归一化回 [0, 1]
    x_target = (x.view(x.size(0), -1) + 1) / 2.0  # [-1, 1] → [0, 1]
    recon_loss = F.binary_cross_entropy(x_recon, x_target,
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
            axes[0, i].set_ylabel('Original', fontsize=10)

        axes[1, i].imshow(recons[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=10)

    plt.suptitle('VAE Reconstruction (Top: Original, Bottom: Reconstructed)', fontsize=14)
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
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    ax.set_title('t-SNE Projection of VAE Latent Space', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] VAE 潜空间 t-SNE 已保存到 {save_path}")



def plot_vae_curves(history, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["total_loss"], label="total")
    ax.plot(history["recon_loss"], label="recon")
    ax.plot(history["kl_loss"], label="KL")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.set_title("VAE 训练曲线")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {save_path}")


def main():
    print("=" * 60)
    print("路径一 · VAE Demo")
    print("=" * 60)
    device = DEVICE
    train_loader = load_mnist(128)
    try:
        test_set = torchvision.datasets.MNIST(
            root="../data", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]),
        )
    except Exception as e:
        print(f"[警告] {e}，合成测试集")
        from torch.utils.data import TensorDataset
        test_set = TensorDataset(
            torch.rand(1000, 1, 28, 28) * 2 - 1,
            torch.randint(0, 10, (1000,)),
        )
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=0)
    if device.type == "cpu":
        n_epochs = 2
        n_train_subset = 1000
        if hasattr(train_loader.dataset, "data"):
            train_loader.dataset.data = train_loader.dataset.data[:n_train_subset]
            if hasattr(train_loader.dataset, "targets"):
                train_loader.dataset.targets = train_loader.dataset.targets[:n_train_subset]
        print("[配置] CPU：VAE 2 epochs")
    else:
        n_epochs = 30
    vae_history, vae_model = train_vae(train_loader, device, n_epochs=n_epochs, latent_dim=20)
    visualize_vae_reconstructions(
        vae_model, test_loader, device,
        save_path=os.path.join(_IMAGES_DIR, "vae_reconstructions.png"),
    )
    visualize_vae_latent_space(
        vae_model, test_loader, device,
        save_path=os.path.join(_IMAGES_DIR, "vae_latent_space.png"),
    )
    plot_vae_curves(vae_history, os.path.join(_IMAGES_DIR, "vae_training_curves.png"))
    print(
        f"Total={vae_history['total_loss'][-1]:.4f} "
        f"recon={vae_history['recon_loss'][-1]:.4f} "
        f"KL={vae_history['kl_loss'][-1]:.4f}"
    )
    print(f"图片目录 {_IMAGES_DIR}")


if __name__ == "__main__":
    main()
