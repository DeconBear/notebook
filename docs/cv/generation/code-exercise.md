---
title: "s13 图像生成 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s13 图像生成 — exercise.py 练习指南

<a href="/notebook/code/cv/generation/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过手写 GAN 和 VAE 的核心损失函数与算法组件，深入理解两种生成范式：

1. **实现 GAN 判别器/生成器损失** —— 理解对抗博弈的优化目标
2. **实现 VAE 重参数化技巧** —— 理解如何让采样可微
3. **实现 KL 散度** —— 理解潜空间正则化的数学
4. **分析模式坍塌** —— 理解 GAN 最常见的失败模式
5. **对比 GAN vs VAE 的设计理念** —— 两种范式各有何优势和代价

## 预备知识

- **GAN 目标**：$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$
- **VAE 目标**：最大化 ELBO = $\mathbb{E}[\log p(x|z)] - D_{KL}(q(z|x)\|p(z))$
- **BCE 损失**：$\text{BCE}(\hat{y}, y) = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$
- **KL 散度（高斯）**：$\frac{1}{2} \sum [\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2]$
- **重参数化**：$z = \mu + \sigma \odot \varepsilon, \varepsilon \sim \mathcal{N}(0,1)$

## 任务清单

### 练习 1：实现 GAN 判别器和生成器的损失函数

#### 1a. 判别器损失

**任务**：实现 `gan_discriminator_loss(d_real_pred, d_fake_pred)`。

**数学目标**：
- 对真实图像：$D(x)$ 应接近 1，损失为 $-\log D(x)$
- 对生成图像：$D(G(z))$ 应接近 0，损失为 $-\log(1 - D(G(z)))$
- 总损失：$L_D = -\frac{1}{2}[\mathbb{E}\log D(x) + \mathbb{E}\log(1 - D(G(z)))]$

**代码框架**：

```python
def gan_discriminator_loss(d_real_pred, d_fake_pred):
    real_labels = torch.ones_like(d_real_pred)   # 全 1
    fake_labels = torch.zeros_like(d_fake_pred)  # 全 0

    real_loss = F.binary_cross_entropy(d_real_pred, real_labels)
    fake_loss = F.binary_cross_entropy(d_fake_pred, fake_labels)

    return (real_loss + fake_loss) / 2  # 取平均
```

**BCE 的数学**：

$$
\text{BCE}(\hat{y}, 1) = -\log(\hat{y}), \quad \text{BCE}(\hat{y}, 0) = -\log(1-\hat{y})
$$

因此 `BCE(D(x), 1) + BCE(D(G(z)), 0)` 等价于 $-\log D(x) - \log(1 - D(G(z)))$。

#### 1b. 生成器损失

**任务**：实现 `gan_generator_loss(d_fake_pred)`。

**数学目标**：让判别器认为生成的图像是真的，即 $D(G(z)) \to 1$。

**为什么不是 $-\log(1 - D(G(z)))$？** 原始 GAN 论文确实使用这个公式，但实践中改用 $-\log D(G(z))$：
- 当 $D(G(z)) \approx 0$（生成器还很差），$-\log(1 - D(G(z))) \approx 0$，梯度几乎为零——**梯度消失**
- 使用 $-\log D(G(z))$ 时，$D(G(z)) \approx 0$ 给出非常大的梯度，帮助生成器快速改进
- 这个修改被称为 **non-saturating GAN loss**，是现代 GAN 训练的标准做法

**代码框架**：

```python
def gan_generator_loss(d_fake_pred):
    target = torch.ones_like(d_fake_pred)  # 全 1
    loss = F.binary_cross_entropy(d_fake_pred, target)
    return loss
```

**测试用例**：

```python
d_real = torch.tensor([[0.9], [0.8], [0.95]])  # D 正确识别真图
d_fake = torch.tensor([[0.1], [0.2], [0.05]])  # D 正确识别假图

d_loss = gan_discriminator_loss(d_real, d_fake)  # D 做得好 → loss 小
g_loss = gan_generator_loss(d_fake)               # G 做得差 → loss 大（~2.3）
```

### 练习 2：实现 VAE 的重参数化技巧

**任务**：实现 `reparameterize(mu, logvar)`。

**为什么需要重参数化？** 如果写 `z = torch.normal(mu, std)`，这个采样操作不可微——`mu` 和 `std` 是模型参数，但我们无法计算 $\frac{\partial z}{\partial \mu}$ 用于反向传播。重参数化将随机性"外包"给 $\varepsilon$：

$$
z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})
$$

此时 $\frac{\partial z}{\partial \mu} = 1$，$\frac{\partial z}{\partial \sigma} = \varepsilon$，梯度可以正常传播。

**代码框架**：

```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)    # σ = e^(0.5 * log(σ²))
    eps = torch.randn_like(std)      # ε ~ N(0, 1)
    z = mu + std * eps               # z = μ + σ ⊙ ε
    return z
```

**测试用例**：给定 $\mu$ 和 $\log(\sigma^2)$，采样得到的 $z$ 应该在 $\mu$ 附近随机波动，波动幅度由 $\sigma$ 控制。

### 练习 3：计算 VAE 的 KL 散度

**任务**：实现 `compute_kl_divergence(mu, logvar)`。

**公式**（两个高斯分布的 KL 散度解析解）：

$$
D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2\mathbf{I}) \| \mathcal{N}(\mathbf{0}, \mathbf{I})) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

**代码框架**：

```python
def compute_kl_divergence(mu, logvar):
    # 逐元素计算: 1 + log(σ²) - μ² - σ²
    kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
    # 对 latent_dim 求和，对 batch 取平均，乘以 -0.5
    kl = -0.5 * torch.sum(kl_element, dim=1).mean()
    return kl
```

**测试用例**：

| $\mu$ | $\log(\sigma^2)$ | $\sigma^2$ | KL散度 | 解释 |
|-------|-----------------|-----------|--------|------|
| 0 | 0 | 1 | $\approx 0$ | 后验 = 先验 $\mathcal{N}(0,1)$ |
| 2 | 1 | $e^1 \approx 2.72$ | $> 0$ | 后验远离先验 |

**KL 散度的直觉**：它是正则化项，防止编码器将所有输入映射到截然不同的 $z$ 区域（这会导致潜空间支离破碎，插值毫无意义）。KL 项越大，潜空间越接近标准正态分布，越平滑和结构良好。

### 练习 4：解释 GAN 训练中的模式坍塌

**任务**：用文字回答三个问题，写入 `explain_mode_collapse()` 返回的字符串。

**1. 什么是模式坍塌（给出具体例子）？**

> 模式坍塌（Mode Collapse）指 GAN 的生成器学会了"作弊"——不管输入什么 $z$，都输出几乎相同的少数几张图像。例如：训练一个生成 MNIST 数字的 GAN，最终无论输入什么噪声，都只生成"1"和"7"（或更极端——只生成某一种"1"）。生成的分布只覆盖了真实分布（0-9）的少数模式。

**2. 从优化角度，为什么 GAN 容易发生模式坍塌？**

> GAN 的优化是 $\min_G \max_D V(D,G)$。如果 $G$ 发现生成某几种模式足够"骗过"当前 $D$，它就没有动力去探索其他模式。$D$ 学会识别这些模式后，$G$ 跳转到另一组模式——而不是学习覆盖所有模式。这就产生了模式间的"旋转门"效应。本质原因是：GAN 的损失只惩罚"生成质量差"（不够真），不直接惩罚"多样性不足"（覆盖不全）。

**3. 至少两种缓解方法**

> a) **Minibatch Discrimination**：让判别器在同一 batch 内比较不同样本的相似度——如果所有样本都很像，给出惩罚信号。
> b) **Wasserstein GAN（WGAN）**：用 Wasserstein 距离替代 JS 散度，提供更平滑的梯度信号，大幅缓解模式坍塌。WGAN-GP 通过梯度惩罚实现了稳定的训练。
> c) **Unrolled GAN**：展开优化步骤——生成器考虑判别器"未来会怎样学"，选择更具前瞻性的策略。

### 练习 5：对比 GAN 和 VAE 的损失函数设计理念

**任务**：分析 `compare_gan_vae_objectives()` 中的两个问题。

**1. 为什么 GAN 的损失导致锐利但可能有模式坍塌的图像？**

> GAN 的判别器学习区分真假——这是一个**感知质量**的判断。生成器不需要逐像素匹配训练数据，只需要"看起来真"。因此 GAN 能生成锐利、细节丰富的图像。但生成器可以"偷懒"——只生成最能骗过当前判别器的少数模式。GAN 的损失中没有显式的"多样性"惩罚，模式坍塌是结构性问题。

**2. 为什么 VAE 的损失导致模糊但覆盖完整的图像？**

> VAE 的逐像素重构损失（BCE/MSE）是一个"保守"的损失——当输入存在多种可能的重构时，最优预测是它们的**期望值**（即平均值）。例如一个像素在训练集中有时是白色有时是黑色，VAE 最优输出是灰色——这就是 VAE 图像模糊的根源。但 VAE 的 KL 散度正则化强制潜空间保持连续和平滑，确保了采样时能覆盖整个数据分布（多样性好）。

| 特性 | GAN | VAE |
|------|-----|-----|
| 图像质量 | 锐利（对抗性目标优化视觉质量） | 模糊（逐像素损失的均值化效应） |
| 多样性 | 低（模式坍塌风险） | 高（KL 约束覆盖分布） |
| 训练稳定性 | 差（博弈可能不收敛） | 好（单一优化目标） |
| 潜空间 | 无可解释结构 | 平滑、可插值、结构良好 |

## 完整代码

<<< @/cv/generation/code/exercise.py
