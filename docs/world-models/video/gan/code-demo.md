---
title: "GAN — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# GAN — demo.py 代码详解

<a href="/notebook/code/world-models/video/gan/demo.py" target="_blank" download>Download demo.py</a>

```bash
cd docs/world-models/video/gan/code
python demo.py
```

## 代码逐段详解

生成器最后 **Tanh**，与 $[-1,1]$ 图像对齐。判别器用 **LeakyReLU(0.2)**：负半轴漏一点梯度，减轻 dead ReLU。

两个 Adam：`betas=(0.5, 0.999)` 是 DCGAN 常用，比默认 $\beta_1=0.9$ 动量更小，对抗训练少振荡。

```python
fake_pred = discriminator(fake_imgs.detach())  # 更新 D，梯度不进 G
...
gen_pred = discriminator(gen_imgs)             # 更新 G，梯度穿过 D
g_loss = adversarial_loss(gen_pred, real_labels)
```

假图进 D 时 **`detach`**：这一步只训练「认假」。G 那一步不 detach，标签却是「真」——骗 D。`BCELoss` 吃 Sigmoid 后的概率。

`fixed_noise`：同一批 $z$ 定期可视化，才能看出样本是在变好还是在跳。

## 源码位置

`docs/world-models/video/gan/code/demo.py`
