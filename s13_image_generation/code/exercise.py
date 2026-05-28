# -*- coding: utf-8 -*-
"""
s13 图像生成 练习
==================
完成以下 TODO 练习来加深对 GAN 和 VAE 核心算法的理解。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# 练习 1：实现 GAN 判别器的损失函数
# ============================================================

def gan_discriminator_loss(d_real_pred: torch.Tensor,
                            d_fake_pred: torch.Tensor) -> torch.Tensor:
    """
    TODO: 实现 GAN 判别器的损失函数

    GAN 的判别器 D 有两个目标:
    1. 对于真实图像 x，D(x) 应该接近 1（判定为真）
    2. 对于生成图像 G(z)，D(G(z)) 应该接近 0（判定为假）

    因此判别器的损失为:
        L_D = -E[log D(x)] - E[log(1 - D(G(z)))]

    或者等价的 BCE 形式:
        L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)

    参数:
        d_real_pred: 判别器对真实图像的输出，形状 (N, 1)，期望接近 1
        d_fake_pred: 判别器对假图像的输出，形状 (N, 1)，期望接近 0

    返回:
        loss: 判别器的总损失（标量）

    提示:
    1. 使用 F.binary_cross_entropy 或手动计算
    2. real_target 应为全 1 张量，fake_target 应为全 0 张量
    3. 两种实现可选:
       a) loss = 0.5 * (BCE(d_real, ones) + BCE(d_fake, zeros))
       b) loss = -torch.mean(torch.log(d_real) + torch.log(1 - d_fake))
    """
    # TODO: 创建目标标签
    # real_labels = torch.ones_like(???)  # 全 1，形状与 d_real_pred 相同
    # fake_labels = torch.zeros_like(???) # 全 0，形状与 d_fake_pred 相同

    # TODO: 计算二元交叉熵损失
    # real_loss = F.binary_cross_entropy(???, ???)  # 真实图像损失
    # fake_loss = F.binary_cross_entropy(???, ???)  # 假图像损失

    # TODO: 返回总损失（两者的平均或求和）
    # return (real_loss + fake_loss) / 2

    return torch.tensor(0.0)  # 替换为你的实现


def gan_generator_loss(d_fake_pred: torch.Tensor) -> torch.Tensor:
    """
    TODO: 实现 GAN 生成器的损失函数

    生成器 G 的目标: 让判别器认为生成的图像是真的
    D(G(z)) 应该接近 1

    标准 GAN 损失:
        L_G = -E[log D(G(z))]

    实际实现中常用:
        L_G = BCE(D(G(z)), 1)  # 让 D(G(z)) 接近 1

    参数:
        d_fake_pred: 判别器对生成图像的输出，形状 (N, 1)

    返回:
        loss: 生成器的损失（标量）

    提示:
    1. 目标标签是全 1 张量
    2. 使用 F.binary_cross_entropy
    """
    # TODO: 创建全 1 的目标标签
    # target = torch.ones_like(???)

    # TODO: 计算 BCE 损失
    # loss = F.binary_cross_entropy(???, ???)

    return torch.tensor(0.0)  # 替换为你的实现


# ============================================================
# 练习 2：实现 VAE 的重参数化技巧
# ============================================================

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    TODO: 实现 VAE 的重参数化技巧

    重参数化将采样操作变为可微:
        z = μ + σ ⊙ ε
    其中:
        - σ = exp(0.5 * logvar) （因为 logvar = log(σ²)）
        - ε ~ N(0, 1)

    为什么需要重参数化？
    如果直接从 N(μ, σ²) 采样 z，这个采样操作是不可微的，
    梯度无法从 decoder 传回 encoder 的 μ 和 σ。
    重参数化将随机性"外包"给 ε，使 μ 和 σ 成为可微的部分。

    参数:
        mu: 编码器预测的均值，形状 (N, latent_dim)
        logvar: 编码器预测的 log(σ²)，形状 (N, latent_dim)

    返回:
        z: 采样后的潜变量，形状 (N, latent_dim)

    提示:
    1. std = torch.exp(0.5 * logvar)   # σ = e^(0.5*log(σ²))
    2. eps = torch.randn_like(std)     # ε ~ N(0, 1)
    3. z = mu + std * eps              # 重参数化
    """
    # TODO: 计算标准差 σ
    # std = ???

    # TODO: 从标准正态分布采样 ε
    # eps = ???

    # TODO: 重参数化: z = μ + σ ⊙ ε
    # z = ???

    return None  # 替换为你的实现


# ============================================================
# 练习 3：计算 VAE 的 KL 散度
# ============================================================

def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    TODO: 计算 VAE 的 KL 散度 D_KL(q(z|x) || p(z))

    假设:
    - q(z|x) = N(z; μ, σ²)  (编码器的后验分布)
    - p(z) = N(z; 0, 1)     (先验分布，标准正态)

    两个高斯分布之间的 KL 散度有解析解:
        D_KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ_j (1 + log(σ_j²) - μ_j² - σ_j²)

    参数:
        mu: 编码器输出的均值，形状 (N, latent_dim)
        logvar: 编码器输出的 log(σ²)，形状 (N, latent_dim)

    返回:
        kl: KL 散度值（标量，对 batch 取平均）

    提示:
    1. 逐元素计算: 1 + logvar - mu^2 - exp(logvar)
    2. 对 latent_dim 维度求和
    3. 对 batch 取平均
    4. 乘以 -0.5

    KL 散度的直觉:
    - 如果 μ=0 且 σ²=1（后验 = 先验），KL = 0
    - 如果 μ 远离 0 或 σ² 远不等于 1，KL 变大
    - KL 项起正则化作用：让潜空间保持平滑，接近标准正态
    """
    # TODO: 计算每个样本的 KL 散度
    # 逐元素: 1 + logvar - mu^2 - exp(logvar)
    # kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    # 对 latent_dim 求和，对 batch 取平均
    # kl = -0.5 * torch.sum(kl_element, dim=1).mean()

    return torch.tensor(0.0)  # 替换为你的实现


# ============================================================
# 练习 4：解释 GAN 训练中的模式坍塌
# ============================================================

def explain_mode_collapse():
    """
    TODO: 用文字结合代码逻辑解释 GAN 的模式坍塌（Mode Collapse）问题

    请在下方写出你对模式坍塌的理解（中文回答以下问题）:

    1. 什么是模式坍塌？给出一个具体的例子
    2. 从优化的角度，为什么 GAN 容易发生模式坍塌？
    3. 至少列举两种缓解模式坍塌的方法

    将你的回答写在下方字符串中并返回。
    """
    explanation = """
    请在此处写下你对模式坍塌的理解，回答上述三个问题。

    (提示示例)
    1. 模式坍塌是指...
    2. 从优化角度看...
    3. 缓解方法:
       a) ...
       b) ...
    """
    return explanation.strip()


# ============================================================
# 练习 5：对比 GAN 和 VAE 的损失函数设计理念
# ============================================================

def compare_gan_vae_objectives():
    """
    TODO: 从数学和直觉两个角度对比 GAN 和 VAE 的损失函数设计

    GAN 的目标: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]
    VAE 的目标: max E[log p(x|z)] - D_KL(q(z|x) || p(z))

    请分析:
    1. 为什么 GAN 的损失会导致锐利但可能有模式坍塌的图像？
    2. 为什么 VAE 的损失会导致模糊但覆盖完整的图像？
    """
    comparison = """
    请在此处写出你的分析。
    """
    return comparison.strip()


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("s13 图像生成 — 练习测试")
    print("=" * 50)

    # ---- 测试练习 1：GAN 损失函数 ----
    print("\n[练习 1] GAN 损失函数测试：")

    # 模拟输出
    d_real = torch.tensor([[0.9], [0.8], [0.95]])  # 较好的真实判别
    d_fake = torch.tensor([[0.1], [0.2], [0.05]])  # 较好的假判别

    d_loss = gan_discriminator_loss(d_real, d_fake)
    g_loss = gan_generator_loss(d_fake)

    print(f"  判别器损失: {d_loss.item():.4f}")
    print(f"  生成器损失: {g_loss.item():.4f}")
    print(f"  (判别器损失小 = D 能区分真假; 生成器损失大 = G 还需改进)")
    print(f"  (期望 d_loss ≈ 0.5~1.0, g_loss ≈ 2~3)")

    # ---- 测试练习 2：重参数化 ----
    print("\n[练习 2] 重参数化技巧测试：")

    mu = torch.tensor([[0.5, -0.3], [0.0, 0.8]])
    logvar = torch.tensor([[0.1, 0.2], [-0.5, 0.0]])

    z = reparameterize(mu, logvar)

    if z is not None:
        print(f"  μ:\n{mu}")
        print(f"  logvar:\n{logvar}")
        print(f"  σ:\n{torch.exp(0.5 * logvar)}")  # 期望标准差
        print(f"  采样 z:\n{z}")
        print(f"  z 的形状: {z.shape}")
        print(f"  (z 应该在 μ 附近随机波动)")
    else:
        print("  请完成 reparameterize 实现")

    # ---- 测试练习 3：KL 散度 ----
    print("\n[练习 3] KL 散度测试：")

    # 测试 1: 后验 = 先验 → KL ≈ 0
    mu1 = torch.zeros(10, 5)    # μ = 0
    logvar1 = torch.zeros(10, 5)  # log(σ²) = 0 → σ² = 1
    kl1 = compute_kl_divergence(mu1, logvar1)
    print(f"  μ=0, σ²=1 (后验=先验): KL = {kl1.item():.6f} (期望 ≈ 0)")

    # 测试 2: 后验偏离先验 → KL > 0
    mu2 = torch.ones(10, 5) * 2.0  # μ 远离 0
    logvar2 = torch.ones(10, 5) * 1.0  # σ² = e¹ ≈ 2.72
    kl2 = compute_kl_divergence(mu2, logvar2)
    print(f"  μ=2, σ²≈2.72 (后验≠先验): KL = {kl2.item():.4f} (期望 > 0)")

    # ---- 练习 4 和 5：概念题 ----
    print("\n[练习 4] 模式坍塌解释：")
    print(explain_mode_collapse())

    print("\n[练习 5] GAN vs VAE 损失对比：")
    print(compare_gan_vae_objectives())

    print("\n" + "=" * 50)
    print("完成所有练习后，运行 demo.py 查看完整的生成对比实验。")
    print("=" * 50)
