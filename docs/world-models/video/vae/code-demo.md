---
title: "VAE — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# VAE — demo.py 代码详解

<a href="/notebook/code/world-models/video/vae/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/vae/code
python demo.py
```

有 GPU/MPS 会自动用。MNIST 下载失败则用合成噪声图。相关：[GAN](/world-models/video/gan/code-demo)、[RSSM 重参数](/world-models/abstract/rssm/code-demo)。

## 代码逐段详解

### 归一化与解码器对不齐

`Normalize((0.5,), (0.5,))` 把像素收到 $[-1,1]$（为了和 GAN 共用 loader）。解码器 **Sigmoid** 输出 $[0,1]$，所以 BCE 前必须 `(x+1)/2` 折回。这是接口包袱。

`x.view(N, -1)`：`-1` 自动算 784。`to_image`：`detach().cpu().numpy()` — GPU 张量不能直接转 NumPy。

### $\mu$、logvar、重参数

```python
std = torch.exp(0.5 * logvar)
z = mu + std * torch.randn_like(std)
```

采样对 $\mu,\sigma$ 可微。`fc_logvar` 保证 $\sigma^2=\exp(\mathrm{logvar})>0$。

### 损失

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
```

标准正态先验的解析 KL。`reduction='sum'` 再除 batch，避免像素平均把重构压没、KL 占满（后验瞬间贴先验）。

## 源码位置

`docs/world-models/video/vae/code/demo.py`
