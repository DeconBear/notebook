---
title: "ml10 蒙特卡洛方法 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# ml10 蒙特卡洛方法 — demo.py 代码详解

<a href="/notebook/code/ml/advanced/monte-carlo/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/ml/advanced/monte-carlo/code
python demo.py
```

## 代码逐段详解

### 第1步：蒙特卡洛估计 $\pi$

```python
def estimate_pi_mc(N=10000):
    points = np.random.uniform(-1, 1, size=(N, 2))       # U(-1,1)^2 均匀采样
    inside = (points[:, 0]**2 + points[:, 1]**2) <= 1     # 判断是否在单位圆内
    pi_est = 4.0 * inside.sum() / N                        # π ≈ 4 × 比例
```

数学原理：

$$
\pi = 4 \times \frac{\text{圆内点数}}{N} \approx 4 \times \mathbb{P}(x^2 + y^2 \le 1)
$$

这是一个从几何直觉到统计估计的优雅转换：单位圆的面积是 $\pi$，外接正方形的面积是 $4$，所以圆内点的比例 $\times 4$ 就是 $\pi$ 的估计。

**收敛性**：右侧的收敛图展示了估计值随 $N$ 的变化。注意前几百个样本波动很大，但随着 $N$ 增大，曲线逐渐稳定在 $\pi \approx 3.1416$ 附近——误差按照 $O(1/\sqrt{N})$ 的速率衰减。

### 第2步：重要性采样 —— 尾部概率估计

```python
def importance_sampling_demo():
    # 提议分布: N(5, 1)
    samples_is = np.random.randn(N) + 5
    # 重要性权重: p(x)/q(x) = phi(x) / phi(x-5)
    log_weights = -0.5 * (samples_is**2 - (samples_is - 5)**2)
    weights = np.exp(log_weights)
    # 加权估计: (1/N) sum w(x_i) * I(x_i > 5)
    is_estimates = np.cumsum(weights * (samples_is > 5)) / np.arange(1, N + 1)
```

这是重要性采样最经典的演示案例。目标是估计 $\mathbb{P}(X > 5)$ 其中 $X \sim \mathcal{N}(0,1)$，真实概率约为 $2.87 \times 10^{-7}$。

**普通 MC 的灾难**：从 $\mathcal{N}(0,1)$ 采样 10000 次，落在 $x>5$ 区域的数学期望只有 $10000 \times 2.87 \times 10^{-7} \approx 0.003$ 个样本——几乎肯定一个都没有。估计值是 $0$，完全无意义。

**重要性采样的解法**：从 $\mathcal{N}(5,1)$ 采样，99.9% 的样本都落在目标区域附近。然后通过权重 $w(x) = p(x)/q(x)$ 来修正——权重本身已经编码了"这个样本来自 $q$ 而非 $p$"的修正因子。

**方差缩减**：重要性采样的方差远小于普通 MC（在本次设置中通常缩减 100 倍以上）。

### 第3步：Metropolis-Hastings 采样

```python
def metropolis_hastings(n_iter=5000, proposal_std=1.0):
    for t in range(n_iter):
        # 提议: x' ~ N(x_current, proposal_std^2 I)
        x_proposal = x_current + rng.randn(d) * proposal_std
        # 接受率: α = min(1, p(x')/p(x))
        log_alpha = log_p_proposal - log_p_current
        alpha = min(1.0, np.exp(log_alpha))
        # 接受/拒绝
        if rng.rand() < alpha:
            x_current = x_proposal
```

MH 算法的三个关键部分：

1. **提议分布（Proposal）**：这里使用对称的随机游走提议 $q(x'|x) = \mathcal{N}(x' | x, \sigma^2 \mathbf{I})$。因为是**对称的**（$q(x'|x) = q(x|x')$），接受率简化为 $\alpha = \min(1, p(x')/p(x))$。

2. **接受率（Acceptance Ratio）**：$\alpha = \min(1, p(x')/p(x^{(t)}))$。如果新点概率更高（$p(x') > p(x)$），$\alpha = 1$（总是接受，向高概率区域移动）。如果新点概率更低，以相应较小的概率接受（保证在低概率区域也能"逃离"）。

3. **细致平衡**：接受/拒绝机制保证了 $\pi(x) T(x \to x') = \pi(x') T(x' \to x)$，从而目标分布是链的平稳分布。

**提议标准差的选择**：$\sigma_{\text{proposal}}$ 是关键超参数。太大则接受率过低（提议的点离当前太远，几乎都被拒绝）；太小则链移动缓慢（acceptance 高但探索效率低）。经验法则：调节 $\sigma$ 使得接受率在 20-50% 之间。

### 第4步：Gibbs 采样二元正态

```python
def gibbs_sampling_bivariate_normal(n_iter=5000, rho=0.8):
    for t in range(n_iter):
        # x1 | x2 ~ N(rho * x2, 1 - rho^2)
        x1 = rng.normal(rho * x2, np.sqrt(1 - rho**2))
        # x2 | x1 ~ N(rho * x1, 1 - rho^2)
        x2 = rng.normal(rho * x1, np.sqrt(1 - rho**2))
```

二元正态分布的 Gibbs 采样的优雅之处在于条件分布有解析形式：

$$
x_1 | x_2 \sim \mathcal{N}(\rho x_2, 1 - \rho^2)
$$
$$
x_2 | x_1 \sim \mathcal{N}(\rho x_1, 1 - \rho^2)
$$

注意：
- 接受率恒为 1（因为是直接从条件分布采样，MH 接受率 $\alpha = 1$）
- 当 $\rho$ 接近 1 时（强相关），Gibbs 效率下降——因为每次只能沿坐标轴移动，在强相关的狭窄山谷中只能"之字形"爬行
- 自相关图（ACF）显示 Gibbs 产生的样本存在自相关性——lag 越大相关性越低，收敛到零。高自相关意味着有效样本量（ESS）远小于总迭代数

### 第5步：MCMC 诊断

代码中包含了几个关键的诊断工具：
- **Trace Plot**：展示采样值随迭代的变化，用于判断 burn-in 和链的混合
- **接受率时序**：展示 MH 接受率的滑动平均，应在 0.2-0.5 之间稳定
- **自相关函数**：展示样本间的相关性，显示 thinning 的必要性
- **有效样本量（ESS）**：$ESS = N / (1 + 2\sum_{k} \rho_k)$，量化了自相关造成的"样本浪费"

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| MC 积分 | $\hat{I}_N = \frac{1}{N}\sum f(x_i)$ | `estimate_pi_mc()` | 收敛率 $O(1/\sqrt{N})$ |
| 重要性采样 | $\sum w_i f(x_i)$, $w_i = p/q$ | `importance_sampling_demo()` | 减少方差的关键技术 |
| MH 算法 | $\alpha = \min(1, p(x')/p(x))$ | `metropolis_hastings()` | 提议 + 接受/拒绝 |
| 细致平衡 | $p(x)T(x \to x') = p(x')T(x' \to x)$ | 隐含在接受率中 | 平稳分布的充分条件 |
| Gibbs 采样 | $x_i \sim p(x_i | \mathbf{x}_{-i})$ | `gibbs_sampling_*()` | 条件分布采样，$\alpha=1$ |
| Burn-in | 丢弃前 N 个样本 | `n_burnin` 参数 | 避免初始值偏差 |
| ESS | $N/(1+2\sum\rho_k)$ | ACF 计算 | 考虑自相关的有效样本数 |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/ml/advanced/monte-carlo/code/demo.py`
