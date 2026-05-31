---
title: "ml10 蒙特卡洛方法 — exercise.py"
---

# ml10 蒙特卡洛方法 — exercise.py 练习指南

<a href="../code/ml10_monte_carlo/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现 MC 积分、重要性采样权重计算、MH 单步更新，从代码层面深入理解蒙特卡洛方法和 MCMC 的数学原理与实现细节。

## 预备知识

- MC 积分公式：$\int_a^b f(x)dx \approx (b-a) \cdot \frac{1}{N}\sum_{i=1}^N f(x_i)$，$x_i \sim U(a,b)$
- 收敛率 $O(1/\sqrt{N})$：误差减半需要 4 倍样本
- 重要性权重：$w_i = p(x_i) / q(x_i)$，用 log-space 计算防数值溢出
- MH 接受率：$\alpha = \min(1, p(x')/p(x))$（对称提议的特例）
- 细致平衡条件是 MCMC 的理论基石

## 任务清单

### 任务1：实现 MC 积分 `mc_integral(f, a, b, N)`

- **公式**：$\hat{I} = (b-a) \cdot \frac{1}{N} \sum_{i=1}^N f(x_i)$
- **标准误差**：$SE = (b-a) \cdot \sigma_f / \sqrt{N}$，其中 $\sigma_f$ 是 $f(x_i)$ 的标准差
- **实现步骤**：
  1. 采样 $N$ 个均匀分布点：`np.random.uniform(a, b, N)`
  2. 计算估计值：`(b-a) * f(x).mean()`
  3. 计算 SE：`(b-a) * np.std(f(x)) / np.sqrt(N)`
- **验证**：对 $\int_0^1 x^2 dx = 1/3$，估计值应在真实值的 3 SE 之内

### 任务2：重要性采样权重 `importance_sampling_weights(samples, p_logpdf, q_logpdf)`

- **log-space 计算**：防止 $p/q$ 的比值在极端样本点处上溢/下溢
  - $\log w_i = \log p(x_i) - \log q(x_i)$
  - $w_i = \exp(\log w_i - \max_j \log w_j)$（减去最大值 = 数值稳定化）
  - $w_i = w_i / \sum_j w_j$（归一化）
- **有效样本量**：$ESS = (\sum w_i)^2 / \sum w_i^2 = 1 / \sum w_i^2$（归一化后）
- ESS 衡量的是重要性权重的"均匀程度"——权重越均匀，ESS 越接近 $N$

### 任务3：MH 单步更新 `mh_step(x_current, target_logpdf, proposal_std)`

- **对称随机游走提议**：$x' = x^{(t)} + \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \sigma^2)$
- **对数接受率**：$\log \alpha = \log p(x') - \log p(x^{(t)})$（对称提议下）
- **接受/拒绝**：比较 $\log(\text{rand}()) < \log \alpha$
- **为什么用 log？** 在高维空间中 $p(x)$ 的值可能极小，在浮点数下可能下溢到 0。使用对数避免了这个问题。

### 任务4：MC vs 解析解对比 `compare_mc_vs_analytic()`

- **目标**：验证误差 $\propto 1/\sqrt{N}$
- **方法**：对 $\int_0^\pi \sin(x) dx = 2$，用不同 $N$ 计算 MC 估计
- **预期**：$N$ 增大 100 倍时，平均误差约减小 10 倍（$\sqrt{100} = 10$）
- **多次重复**：对每个 $N$ 做 10 次独立运行取平均误差可以平滑随机性

## 完整代码

<<< @/snippets/ml10_monte_carlo/exercise.py
