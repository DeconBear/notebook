---
title: "扩散模型：从 DDPM 到视频世界模拟器"
order: 30
---
# 扩散模型：把生成写成逐步去噪

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 路径一的第三块积木，也是今天视频世界模型的默认引擎。GAN 一次出图、VAE 一次解码；扩散把「从噪声回到数据」拆成一条马尔可夫链。Sora 类系统、NVIDIA Cosmos 的扩散变体、Stable Diffusion，都是这条链在像素或 [VAE](/world-models/video/vae/) 潜空间上的加长版。

---

## 一、核心思想：前向加噪，反向去噪

灵感来自非平衡热力学。两个过程成对出现：

**前向过程 $q$**（不用学）：向干净样本逐步加高斯噪声，经过 $T$ 步变成近乎纯噪声。

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\big)
$$

$\beta_t$ 是噪声调度，经典 DDPM 从 $\beta_1\approx 10^{-4}$ 线性增到 $\beta_T\approx 0.02$。令 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^t\alpha_s$，可以一步跳到任意 $t$：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

**反向过程 $p_\theta$**（要学）：从 $x_T\sim\mathcal{N}(0,I)$ 出发，逐步恢复。

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\, \mu_\theta(x_t,t),\, \Sigma_\theta(x_t,t)\big)
$$

![扩散模型——前向加噪与反向去噪](./images/13-03-diffusion-forward-reverse.png)

> **图解说明**：左向加噪是固定的高斯核；右向每一步由网络预测「这一层噪声里藏着什么」。

---

## 二、DDPM：训练塌成一个回归

Ho, Jain & Abbeel（2020）的关键简化：不必直接拟合 $\mu_\theta$，改成预测所加的噪声 $\epsilon$。损失变成

$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\Big[
\big\|
\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\, t)
\big\|^2
\Big]
$$

翻译成训练循环：

1. 抽一张 $x_0$、一个时间步 $t$、一团 $\epsilon$；
2. 合成 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$；
3. 网络看 $(x_t,t)$，回归 $\epsilon$；
4. 这就是 MSE。稳定、可并行（每个 $t$ 独立），不像 GAN 要交替两个网络。

采样（祖先采样）仍是逐步的：

1. $x_T\sim\mathcal{N}(0,I)$
2. 对 $t=T,\ldots,1$：
   $$
   x_{t-1}
   =
   \frac{1}{\sqrt{\alpha_t}}
   \Big(
   x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
   \Big)
   + \sigma_t z
   $$
3. 输出 $x_0$

$T$ 常取 1000，这就是「扩散慢」的来源。**DDIM**（Song et al., 2021）把反向过程改成非马尔可夫的确定性（或少随机）轨迹，可用 50 步左右换质量，是工程加速的第一刀。后续还有蒸馏（一致性模型）、整流流（Rectified Flow / Flow Matching）等，教学上先抓住：都是「噪声 ↔ 数据」的连续或离散路径，网络学速度场或噪声。

本章 `demo.py` 在一维双峰混合上跑一个极小 DDPM：你可以看见前向把两个峰糊成一团高斯，反向再把峰「揭」回来——不必上 U-Net 也能建立时间步直觉。

---

## 三、Stable Diffusion：VAE 潜空间里跑扩散

Rombach et al.（2022）的 Latent Diffusion 把三件事焊在一起：

1. **潜空间扩散**：先用 **VAE** 把图像压到低维 $z$，再对 $z$ 做 DDPM。算力大约降一个数量级——VAE 不是被淘汰，而是被嵌进流水线。
2. **文本条件**：CLIP（或后续 T5）把 prompt 编成向量，经 **交叉注意力** 注入去噪 U-Net / DiT。
3. **Classifier-Free Guidance（CFG）**：同时训有条件和无条件去噪，推理时
   $$
   \hat\epsilon_\theta(x_t,c)
   =
   \epsilon_\theta(x_t,\varnothing)
   + w\cdot\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)
   $$
   $w>1$ 加强文本服从，过大则过饱和、多样性下降。

对路径一的含义：你现在具备了「压缩（VAE）+ 去噪先验（扩散）+ 条件（文本/动作）」三件套。视频世界模型只是把 $x$ 从一张图换成时空潜变量，把 $c$ 从句子换成句子或动作序列。

---

## 四、从图像扩散到视频世界模型

把 $x_0$ 想成一段 clip（或 clip 的 VAE 潜变量），前向过程沿**时间轴与空间轴**同时加噪。工程上常见三条路：

| 路线 | 做法 | 代表直觉 |
|------|------|----------|
| 3D U-Net | 在 2D 去噪网上加时间卷积/注意力 | 早期文生视频 |
| 潜空间视频 VAE + DiT | 时空 patch 进 Transformer，条件走 AdaLN / 交叉注意力 | Sora 技术报告叙事、大量开源 DiT 视频 |
| 自回归 token | 离散 tokenizer + next-token（或 next-chunk） | 另一类 Cosmos / VideoPoet 变体 |

**DiT（Diffusion Transformer）**：不再用卷积 U-Net，而把带噪潜变量切成 patch，用 Transformer 预测噪声。缩放曲线更像语言模型，这是「世界基础模型」能吃互联网视频的原因之一。

当模型开始满足下面三条，社区就把它叫做**视频世界模型**，而不只是「会做特效的生成器」：

1. 较长时间内物体身份不太丢；
2. 大致遵守直观物理（碰撞、重力「看起来对」）；
3. 接受文本、相机或动作条件，能在生成的未来里做想象。

这正是下一章 [Sora / Cosmos](/world-models/video/sora/) 的主题。请带着本章的公式去读：那些系统几乎都是 $\epsilon_\theta(x_t,t,c)$ 的超大规模实例。路径四的警告在这里最响——纯旁观视频优化的是 $P(o_{t+1}\mid o_{\le t})$，不是 $P(\cdot\mid do(a))$。

---

## 五、三种范式收束

| 特性 | GAN | VAE | Diffusion |
|------|-----|-----|-----------|
| **生成质量** | 高（锐利） | 较低（模糊） | 极高（锐利+多样） |
| **多样性** | 低（模式坍塌） | 高 | 高 |
| **训练稳定性** | 极不稳定 | 稳定 | 稳定 |
| **采样速度** | 快（一次前向） | 快 | 慢（数百步→数十步） |
| **潜空间** | 无显式（StyleGAN 除外） | 有结构 | 噪声空间；LDM 借用 VAE 潜空间 |
| **理论基础** | 博弈论 | 变分推断 | 随机微分方程 / 得分匹配 |

![生成模型对比——GAN vs VAE vs Diffusion](./images/13-04-generative-models-comparison.png)

- **GAN**：要速度、要锐度（实时特效、部分判别器辅助）。
- **VAE**：要可插值的瓶子、要当 tokenizer。
- **Diffusion**：要当前的生成质量上限，并作为视频 WM 主干。

---

## 六、小结

| 概念 | 一句话 |
|------|--------|
| 前向 / 反向 | 固定加噪 $q$ / 学去噪 $p_\theta$ |
| $\mathcal{L}_{\mathrm{simple}}$ | 预测 $\epsilon$，MSE 回归 |
| DDIM | 少步、更确定的采样轨迹 |
| Latent Diffusion | VAE 压缩后再扩散 + 文本交叉注意力 |
| CFG | 有条件与无条件之差，放大文本服从 |
| DiT / 视频扩散 | 同一去噪目标，状态变成时空潜变量 |
| 与世界模型 | 条件 $c$ 含动作时，生成器即（昂贵的）模拟器 |

> 下一章把这套机器当成模拟器来用：[视频生成式世界模型](/world-models/video/sora/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/world-models/video/diffusion/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/world-models/video/diffusion/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*. [[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)]
2. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. *ICLR*. [[arXiv:2010.02502](https://arxiv.org/abs/2010.02502)]
3. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*. [[arXiv:2112.10752](https://arxiv.org/abs/2112.10752)]
4. Peebles, W. & Xie, S. (2023). Scalable Diffusion Models with Transformers (DiT). *ICCV*. [[arXiv:2212.09748](https://arxiv.org/abs/2212.09748)]
5. Brooks, T., et al. (2024). Video generation models as world simulators.（Sora 叙事）
