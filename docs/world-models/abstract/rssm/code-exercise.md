---
title: "wm02 经典起源与 RSSM — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# wm02 经典起源与 RSSM — exercise.py 练习指南

<a href="/notebook/code/world-models/abstract/rssm/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过补全三个 TODO，掌握 RSSM 训练中最核心的三个数学组件：
1. 对角高斯 KL 散度
2. 重参数化采样
3. Free-nats 裁剪

## 预备知识

- 对角高斯 KL 的解析公式（见 index.md 第三节）
- 重参数化技巧：$z = \mu + \sigma \varepsilon,\ \varepsilon\sim\mathcal{N}(0,I)$
- Free-nats：$\text{penalty} = \max(\text{KL},\ \text{free\_nats})$

## 任务清单

### TODO 1：对角高斯 KL 散度

**公式**：
$$
\text{KL}=\sum_i\left[\log\frac{\sigma_{p,i}}{\sigma_{q,i}}+\frac{\sigma_{q,i}^2+(\mu_{q,i}-\mu_{p,i})^2}{2\sigma_{p,i}^2}-\frac12\right]
$$

**提示**：
```python
var_q, var_p = std_q ** 2, std_p ** 2
kl_per_dim = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5
return kl_per_dim.sum(dim=-1)
```

**预期**：相同分布 KL≈0；$\mathcal{N}(0,1)\,\|\,\mathcal{N}(1,1)$ 的 KL≈0.5。

### TODO 2：重参数化采样

**提示**：`eps = torch.randn_like(mean); return mean + std * eps`

**关键验证点**：对 `z.sum().backward()` 后，`mean.grad` 和 `std.grad` 都不能是 `None`——这正是重参数化相比普通采样的优势。

### TODO 3：Free-nats 裁剪

**提示**：`return torch.clamp(kl_loss, min=free_nats)`

**预期**：
- `kl=0.2, free_nats=0.5` → `0.5`
- `kl=0.5` → `0.5`
- `kl=1.2` → `1.2`

## 完成后的验证

全部 TODO 通过后，运行 `python code/demo.py` 观察 RSSM 训练曲线、想象 rollout 轨迹、以及开环/闭环误差对比。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/world-models/abstract/rssm/code/exercise.py`
