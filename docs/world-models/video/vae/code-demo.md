---
title: "VAE — demo.py"
---

# VAE — demo.py

<a href="/notebook/code/world-models/video/vae/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/vae/code
python demo.py
```

编码器输出 $\mu,\log\sigma^2$，重参数采样 $z$，解码重构；损失 = 重构 + KL。看 `vae_reconstructions.png` 与 `vae_latent_space.png`。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/video/vae/code/demo.py`
