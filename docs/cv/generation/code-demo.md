---
title: "s13 图像生成 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s13 图像生成 — demo.py 代码详解

<a href="/notebook/code/cv/generation/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/cv/generation/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库与全局配置 —— 每个库是做什么的

```python
import torch
import torch.nn as nn            # GAN 的 Linear+BN、VAE 的 Encoder/Decoder
import torch.nn.functional as F  # BCE、MSE、relu、sigmoid 等损失和激活函数
import torch.optim as optim      # Adam 优化器（GAN 和 VAE 都用 Adam）
import torchvision               # MNIST 数据集
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```

| 库 | 在此 demo 中的角色 |
|---|---|
| `torch.nn` | 构建 Generator、Discriminator、VAE Encoder/Decoder |
| `torch.optim` | Adam 优化器（GAN 用 `lr=0.0002, betas=(0.5, 0.999)`） |
| `torchvision` | MNIST 数据集（28x28 手写数字） |

**为什么 MNIST？** 手写数字生成是图像生成模型的"Hello World"。28x28 灰度图维度低、结构简单，可以让 GAN 和 VAE 在 CPU 上也能快速训练出可见效果。同时 MNIST 的 10 个类别（0-9）提供了清晰的聚类结构，便于分析 VAE 潜空间的 t-SNE 分布。

**数据归一化策略**：

```python
transforms.Normalize((0.5,), (0.5,))  # [0,1] → [-1,1]
```

| 模型 | 输出激活 | 输出范围 | 目标范围 | 匹配？ |
|------|---------|---------|---------|--------|
| GAN Generator | Tanh | $[-1, 1]$ | $[-1, 1]$ | 完美匹配 |
| VAE Decoder | Sigmoid | $[0, 1]$ | $[-1, 1]$ (输入) | **需要转换！** |

> **VAE 的特别注意**：VAE 解码器用 Sigmoid 输出 $[0, 1]$，但 MNIST 数据被归一化到 $[-1, 1]$。在计算 VAE 损失时，代码将目标反归一化回 $[0, 1]$ 再用 BCE 损失——详见第4步。

### 第2步：GAN —— 生成对抗网络

GAN 由两个网络组成：**生成器 $G$** 和 **判别器 $D$**，它们在一个极小极大博弈中对抗训练。

**GAN 的数学形式**：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

- 判别器 $D$ 想最大化 $V$：真图为 1，假图为 0
- 生成器 $G$ 想最小化 $V$：让 $D(G(z))$ 接近 1（以假乱真）

#### 2.1 生成器 Generator —— 从噪声到图像

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        self.model = nn.Sequential(
            # Block 1: 128 → 256
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),          # BN 稳定训练
            nn.ReLU(inplace=True),
            # Block 2: 256 → 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # Block 3: 512 → 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            # Block 4: 1024 → 784 (28×28)
            nn.Linear(1024, 784),
            nn.Tanh(),  # 输出 [-1, 1]
        )
```

**为什么用全连接而不是转置卷积？** 对于 MNIST 这种 28x28 的小图像，全连接层足以生成合理的结果（参数约 1M）。转置卷积（DCGAN）在大图像上效果更好，但全连接版本更简洁，适合教学。反卷积版本通常能生成更平滑的纹理。

**为什么 Generator 用 BatchNorm 而 Discriminator 不用？** BatchNorm 在生成器中非常关键——它防止生成器的输出分布漂移，确保各层的激活保持在合理范围。判别器不需要 BN（甚至 BN 可能有害），因为判别器的任务是对单张图像做判断，BN 引入的 mini-batch 统计依赖会干扰逐样本判断。

**为什么最后一层用 Tanh？** Tanh 输出范围 $[-1, 1]$，与 MNIST 的归一化范围一致。如果用 Sigmoid（输出 $[0, 1]$），需要改变数据预处理，且梯度在两端更平缓。

#### 2.2 判别器 Discriminator —— 真假判断

```python
class Discriminator(nn.Module):
    def __init__(self):
        self.model = nn.Sequential(
            # Block 1: 784 → 512
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),        # LeakyReLU 防止死神经元
            # Block 2: 512 → 256
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # Block 3: 256 → 1
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出概率 [0, 1]
        )
```

**为什么判别器用 LeakyReLU 而不是 ReLU？** LeakyReLU 在负半轴保留了微小斜率（$\alpha=0.2$），避免了 ReLU 的"死神经元"问题——当 ReLU 输入恒为负时，梯度为零，神经元永远无法恢复。在 GAN 训练中，判别器的梯度质量直接影响生成器的学习，死神经元是 GAN 训练失败的常见原因之一。

$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0.2x & \text{if } x \le 0 \end{cases}
$$

#### 2.3 GAN 训练循环 —— 交替优化

正文 [GAN 交替训练示意](/cv/generation/) 对应这里的实现节奏：先 D 后 G。

```python
for epoch in range(n_epochs):
    for imgs, _ in dataloader:
        batch_size = imgs.size(0)
        real_imgs = imgs.to(device)

        real_labels = torch.ones(batch_size, 1, device=device)   # 真实=1
        fake_labels = torch.zeros(batch_size, 1, device=device)   # 虚假=0

        # ===== 1. 训练判别器 D =====
        optimizer_D.zero_grad()

        # D(真实图像) → 应该接近 1
        real_pred = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, real_labels)

        # D(生成图像) → 应该接近 0
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_pred = discriminator(fake_imgs.detach())  # ⚠️ detach() 防止梯度传回 G
        d_fake_loss = adversarial_loss(fake_pred, fake_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ===== 2. 训练生成器 G =====
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        gen_pred = discriminator(gen_imgs)  # 注意：这次不用 detach()
        g_loss = adversarial_loss(gen_pred, real_labels)  # G 的目标：让 D 判为真

        g_loss.backward()
        optimizer_G.step()
```

**关键代码细节**：

1. **`fake_imgs.detach()`**：训练判别器时，生成的假图必须从计算图中切断。否则梯度会通过 `fake_imgs` 流回生成器，导致我们在训练 D 的同时误改了 G 的参数。

2. **生成器的损失目标**：$L_G = \text{BCE}(D(G(z)), 1)$。注意这里用的目标标签是 **1（真实）**——生成器想"骗过"判别器，让判别器认为生成图像是真的。

3. **交替训练**：每轮先训 D 再训 G。理论上应该是 D 训 k 步 G 训 1 步（原始 GAN 论文的推荐），但实践中 1:1 交替训练通常效果不错。

**GAN 训练的直观理解**：

| 角色 | 目标 | 策略 |
|------|------|------|
| 判别器 D | 区分真假 | 给真图高分，给假图低分 |
| 生成器 G | 以假乱真 | 让 D 给生成图高分 |
| 理想平衡点 | $D(x) = \frac{1}{2}$ | D 完全无法区分，G 生成完美 |

### 第3步：VAE —— 变分自编码器

VAE 的学习目标与 GAN 完全不同：它不是对抗博弈，而是最大化数据的**证据下界（ELBO）**。管线与 AE/VAE 潜空间对比见正文 [VAE 一节](/cv/generation/)。

**VAE 的数学核心**：

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重构损失（越大越好）}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{KL 散度（越小越好）}}
$$

#### 3.1 VAE 网络结构

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        # --- 编码器：x → μ 和 log(σ²) ---
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)       # 均值 μ
        self.fc_logvar = nn.Linear(256, latent_dim)   # log(σ²) —— 为什么是 log？

        # --- 解码器：z → 重建 x̂ ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),        # Sigmoid 输出 [0, 1]
        )
```

**为什么编码器输出 `logvar` 而不是 $\sigma^2$ 直接？** 方差 $\sigma^2$ 必须为正，如果网络直接输出 $\sigma^2$，我们需要加一个激活函数（如 softplus）来保证正值。输出 $\log(\sigma^2)$ 不需要任何约束——它可以取任意实数值，然后通过 $\exp$ 恢复为正的 $\sigma^2$：

$$
\sigma^2 = \exp(\log(\sigma^2)), \quad \sigma = \exp(0.5 \cdot \log(\sigma^2))
$$

这在数学上比约束式输出更干净，训练更稳定。

#### 3.2 重参数化技巧 —— VAE 的关键创新

如果直接从 $\mathcal{N}(\mu, \sigma^2)$ 采样 $z$，采样操作不可微，梯度无法从 decoder 传回 encoder。重参数化将采样分解为确定性部分 + 随机噪声：

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})
$$

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)    # σ = e^(0.5 * log(σ²))
    eps = torch.randn_like(std)      # ε ~ N(0, 1)
    z = mu + std * eps               # z = μ + σ ⊙ ε
    return z
```

**梯度流分析**：

$$
\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \varepsilon
$$

$\mu$ 和 $\sigma$ 直接出现在 $z$ 的计算中，梯度可以毫无障碍地传回 encoder。$\varepsilon$ 是一个与模型参数无关的随机变量，反向传播时被当作常数。

#### 3.3 VAE 损失函数

```python
def vae_loss(x_recon, x, mu, logvar):
    # ---- 重构损失：二元交叉熵 ----
    x_target = (x.view(x.size(0), -1) + 1) / 2.0  # [-1,1] → [0,1]
    recon_loss = F.binary_cross_entropy(x_recon, x_target, reduction='sum') / x.size(0)

    # ---- KL 散度（解析解）----
    # 对于 q = N(μ, σ²), p = N(0, 1):
    # D_KL(q || p) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss
```

**KL 散度解析公式的推导**：

对于两个 $d$ 维高斯分布 $q = \mathcal{N}(\mu, \Sigma)$ 和 $p = \mathcal{N}(0, I)$：

$$
\begin{aligned}
D_{\text{KL}}(q \| p) &= \frac{1}{2} \left[ \text{tr}(\Sigma) + \mu^\top \mu - d - \log \det(\Sigma) \right] \\
&= \frac{1}{2} \sum_{j=1}^{d} \left[ \sigma_j^2 + \mu_j^2 - 1 - \log(\sigma_j^2) \right] \\
&= -\frac{1}{2} \sum_{j=1}^{d} \left[ 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right]
\end{aligned}
$$

**KL 散度的作用**：
- 如果 $\mu = 0$ 且 $\sigma^2 = 1$（后验 = 先验），KL = 0
- 如果 $\mu$ 远离 0 或 $\sigma^2$ 远离 1，KL >> 0
- KL 正则化迫使潜空间保持结构化和平滑——两个相似的 $x$ 映射到相近的 $z$

#### 3.4 VAE 训练循环

```python
for imgs, _ in dataloader:
    imgs = imgs.to(device)
    optimizer.zero_grad()

    x_recon, mu, logvar = model(imgs)             # 前向传播
    total_loss, recon_loss, kl_loss = vae_loss(   # 计算损失
        x_recon, imgs, mu, logvar
    )
    total_loss.backward()                          # 梯度可以穿过重参数化
    optimizer.step()
```

VAE 训练比 GAN 稳定得多——只有一个优化器，一个损失函数，没有博弈过程。

### 第4步：可视化 —— 四种对比图

**GAN 生成样本**：从随机噪声生成 16 张数字图像，排列成 4x4 网格。观察：
- 生成质量：数字是否清晰可辨
- 多样性：是否覆盖了 0-9 全部数字（还是模式坍塌只生成少数几个）

**VAE 重建对比**：上方原始图像 vs 下方 VAE 重建。典型情况是重建比原始模糊——这是 VAE 的"均值化效应"：逐像素 BCE 损失倾向于预测像素值的条件期望，导致模糊。

**VAE 潜空间 t-SNE**：将测试集图像通过 VAE 编码器得到的 $\mu$ 向量做 t-SNE 投影到 2D。如果潜空间结构良好，不同数字类别应该在投影空间中形成清晰的聚类。

**训练曲线对比**：三张子图：
1. GAN 的 G Loss 和 D Loss 曲线 —— 理想状态是两个 loss 在合理范围内波动，不收敛也不发散
2. VAE 的总损失 / 重构损失 / KL 散度 —— 重构损失下降、KL 散度上升后稳定
3. 文字对比总结 —— GAN vs VAE 的特性

### 关键概念速查表

| 概念 | 公式 | 代码对应 |
|------|------|---------|
| GAN 极小极大 | $\min_G \max_D V(D,G)$ | 交替训练 D 和 G |
| D 的损失 | $-\log D(x) - \log(1-D(G(z)))$ | `BCE(d_real_pred, 1) + BCE(d_fake_pred, 0)` |
| G 的损失 | $-\log D(G(z))$ | `BCE(gen_pred, 1)` |
| VAE ELBO | $\mathbb{E}_q[\log p(x\|z)] - D_{KL}(q\|p)$ | `recon_loss + kl_loss` |
| 重参数化 | $z = \mu + \sigma \odot \varepsilon$ | `mu + std * eps` |
| KL 散度（高斯） | $-\frac{1}{2}\sum[1+\log\sigma^2-\mu^2-\sigma^2]$ | `-0.5 * torch.sum(1 + logvar - mu^2 - exp(logvar))` |
| 模式坍塌 | G 只生成少数模式 | 生成样本多样性低 |
| LeakyReLU | $\max(0.2x, x)$ | `nn.LeakyReLU(0.2)` |

## 完整代码

<<< @/cv/generation/code/demo.py
