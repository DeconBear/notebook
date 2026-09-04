---
title: "GAN — demo.py"
---

# GAN — demo.py

<a href="/notebook/code/world-models/video/gan/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/gan/code
python demo.py
```

CPU 上默认 2 个 epoch、1000 张 MNIST（下载失败则用合成图）。先更新判别器再更新生成器：假图 `detach()` 后再进 $D$，生成器一步则不 `detach`，对应正文的交替训练。

样张写入 `images/gan_samples.png`，曲线写入 `images/training_curves.png`。

## 完整代码

<<< @/world-models/video/gan/code/demo.py
