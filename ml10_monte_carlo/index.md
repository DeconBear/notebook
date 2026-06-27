# ml10 蒙特卡洛方法

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。


> 用随机数解决确定性问题——从估算 $\pi$ 到采样复杂分布，蒙特卡洛方法是大数定律在计算中的优雅应用。

---

## 一、什么是蒙特卡洛方法？

**蒙特卡洛方法（Monte Carlo Methods）** 是一类基于**随机采样**的数值计算方法。其核心思想极为简洁：用大量随机样本来近似难以解析计算的量。

蒙特卡洛方法得名于摩纳哥的蒙特卡洛赌场——一个以随机性闻名的地方。该方法在 1940 年代由 Stanislaw Ulam 和 John von Neumann 在洛斯阿拉莫斯国家实验室（美国核武器项目）正式发展，用于模拟中子在核反应堆中的随机运动。

> **一句话概括**：当解析解不存在或太复杂时，我们就用"投飞镖"来求解。

---

## 二、蒙特卡洛积分

### 2.1 基本思想

目标是计算积分：

$$
I = \int_{\Omega} f(\mathbf{x}) \, d\mathbf{x}
$$

蒙特卡洛方法将其改写为期望形式：

$$
I = \int_{\Omega} p(\mathbf{x}) \frac{f(\mathbf{x})}{p(\mathbf{x})} \, d\mathbf{x}
= \mathbb{E}_{\mathbf{x} \sim p}\left[ \frac{f(\mathbf{x})}{p(\mathbf{x})} \right]
$$

然后用样本均值来近似期望：

$$
\hat{I}_N = \frac{1}{N} \sum_{i=1}^{N} \frac{f(\mathbf{x}_i)}{p(\mathbf{x}_i)}, \quad \mathbf{x}_i \sim p(\mathbf{x})
$$

### 2.2 收敛性：大数定律

蒙特卡洛估计的收敛由**大数定律（LLN）**保证：

$$
\lim_{N \to \infty} \hat{I}_N = I \quad （以概率 1 收敛）
$$

更重要的是**中心极限定理（CLT）**告诉我们误差的分布：

$$
\sqrt{N}(\hat{I}_N - I) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

其中 $\sigma^2 = \text{Var}[f(\mathbf{x})/p(\mathbf{x})]$。这意味着：

$$
\text{误差} \sim O\left(\frac{1}{\sqrt{N}}\right)
$$

> **蒙特卡洛的"诅咒"**：要想误差减半，需要 4 倍的样本。$O(1/\sqrt{N})$ 的收敛速度与维度无关——这是 MC 在高维积分中的核心优势（规则网格法（如梯形法）在高维中的误差是 $O(N^{-2/d})$，随维度 $d$ 恶化）。

![蒙特卡洛积分示意图：左侧画一个 2D 任意形状区域，内部散落 N 个随机点（红色在区域内，蓝色在区域外），标注"面积 ≈ (红点数/总点数) × 矩形面积"。右侧是估计值 vs N 的收敛曲线，真实值用虚线标出，三根不同颜色的曲线（3 次独立运行）都趋近虚线](./images/ml10-01-monte-carlo-integration.png)

---

## 三、重要性采样（Importance Sampling）

### 3.1 为什么需要重要性采样？

原始 MC 从均匀分布采样在 $p(x)$ 下，当 $f(x)$ 在某些区域值很大而采样概率很低时，方差会非常大——绝大多数样本集中在 $f(x)$ 小的区域，浪费了计算资源。

**重要性采样**的解决方案：用一个"更好"的提议分布（proposal distribution）$q(x)$ 来采样，然后通过权重修正：

$$
I = \int f(x) dx = \int \frac{f(x)}{q(x)} q(x) dx
= \mathbb{E}_{x \sim q}\left[ \frac{f(x)}{q(x)} \right]
$$

### 3.2 最优提议分布

方差最小时的 $q(x)$ 是什么？可以证明最优提议分布为：

$$
q^*(x) \propto |f(x)|
$$

这意味着：**在 $f(x)$ 绝对值大的地方多采样**——把计算资源集中在"重要的"区域。这就是"重要性采样"名称的由来。

在实践中，我们通常选择与 $|f(x)|$ 形状相近但易于采样的分布作为 $q(x)$。

### 3.3 示例：尾部概率估计

假设要估计 $\mathbb{P}(X > 5)$ 其中 $X \sim \mathcal{N}(0, 1)$。

- **原始 MC**：从 $\mathcal{N}(0, 1)$ 采样 10000 次，约只有 3 个样本落在 $>5$ 的区域——估计极不稳定
- **重要性采样**：从 $\mathcal{N}(5, 1)$ 采样，大量样本集中在目标区域，通过权重 $p(x)/q(x)$ 修正——估计高效且精确

![重要性采样对比图：左图展示被积函数 f(x)（蓝色曲线）和原始采样分布 p(x)（灰色虚线），大部分采样点（底部刻度线）集中在 f(x)≈0 的区域（浪费）；右图展示最优提议分布 q(x)（红色虚线，形状与 f(x) 相近），采样点集中在 f(x) 大的区域，每个点的大小表示权重 f(x)/q(x)](./images/ml10-02-importance-sampling.png)

---

## 四、拒绝采样（Rejection Sampling）

### 4.1 核心思想

目标是从复杂分布 $p(x)$ 采样，但我们只知道 $p(x)$ 的非归一化形式 $\tilde{p}(x) = Z p(x)$（$Z$ 未知）。

拒绝采样的思路：用一个容易采样的提议分布 $q(x)$ 来"包围"目标分布：

1. 选择一个 $q(x)$ 和常数 $M$，使得 $M q(x) \ge \tilde{p}(x)$ 对所有 $x$ 成立
2. 从 $q(x)$ 采样一个候选点 $x^*$
3. 以概率 $\frac{\tilde{p}(x^*)}{M q(x^*)}$ 接受 $x^*$，否则拒绝并回到步骤 2

接受的样本服从 $p(x)$。

**几何直觉**：$M q(x)$ 是目标分布 $\tilde{p}(x)$ 的"上包络"。在 $M q(x)$ 曲线下均匀撒点，落在 $\tilde{p}(x)$ 曲线下的点被接受。

### 4.2 拒绝采样的局限

- **维度的诅咒**：在高维空间中，接受率 $\propto 1/M$，而 $M$ 通常随维度指数增长
- **需要好的提议分布**：如果 $q(x)$ 与 $p(x)$ 形状差异大，接受率极低
- 不适合高维复杂分布——这也是为什么 MCMC 取代了拒绝采样成为主流

---

## 五、MCMC：马尔可夫链蒙特卡洛

### 5.1 核心思想

当 $p(x)$ 是高维复杂分布时，直接采样不可能，拒绝采样也因接受率过低而失败。**MCMC** 的巧妙思路：不试图直接从 $p(x)$ 独立采样，而是构建一条马尔可夫链，使其**平稳分布（stationary distribution）**恰好等于 $p(x)$。然后运行这条链足够长时间（burn-in），收集到的样本就近似服从 $p(x)$。

MCMC 的两个关键问题：
1. **如何构建这样的链？** —— Metropolis-Hastings / Gibbs Sampling
2. **链"混合"需要多长时间？** —— burn-in + thinning

### 5.2 Metropolis-Hastings 算法

Metropolis-Hastings（MH）是最通用的 MCMC 算法。核心是**提议 + 接受/拒绝**两步：

**步骤 1：提议（Proposal）**

从提议分布 $q(x' | x^{(t)})$ 中采样候选点 $x'$。常用的是**随机游走提议**：$x' = x^{(t)} + \varepsilon$，$\varepsilon \sim \mathcal{N}(0, \sigma^2)$。

**步骤 2：接受/拒绝（Accept/Reject）**

计算接受率：

$$
\alpha = \min\left(1, \frac{p(x') q(x^{(t)} | x')}{p(x^{(t)}) q(x' | x^{(t)})}\right)
$$

以概率 $\alpha$ 接受 $x'$（$x^{(t+1)} = x'$），否则保留当前状态（$x^{(t+1)} = x^{(t)}$）。

**为什么接受率是这个形式？** 这是为了保证**细致平衡条件（detailed balance）**：

$$
p(x) \cdot T(x \to x') = p(x') \cdot T(x' \to x)
$$

其中 $T(x \to x') = q(x' | x) \cdot \alpha(x, x')$ 是转移核。细致平衡是平稳分布为 $p(x)$ 的充分条件。

> **直觉**：接受率 $\alpha$ 是目标分布在新旧状态之间的比值修正——如果新状态比旧状态"更可能"（$p(x') > p(x^{(t)})$），大概率接受；如果新状态更不可能，以相应较小的概率接受。这保证了链在 $p(x)$ 高的区域"停留更久"，在低的区域"快速离开"。

![Metropolis-Hastings 算法示意图：2D 目标分布等高线（双峰混合高斯），叠加一条马尔可夫链的轨迹。链从远处开始（标注"起始点"），逐步移动到高概率区域（标注"Burn-in 期"），然后在两个峰之间来回跳跃（标注"收敛后采样"）。右下角小图展示轨迹的 x1 坐标序列，burn-in 后明显稳定](./images/ml10-03-metropolis-hastings.png)

### 5.3 Gibbs 采样

Gibbs 采样是 MH 的一种特殊形式——提议分布就是目标分布的条件分布，**接受率恒为 1**。

对于 $d$ 维随机变量 $\mathbf{x} = (x_1, \dots, x_d)$，Gibbs 采样逐维更新：

$$
x_1^{(t+1)} \sim p(x_1 | x_2^{(t)}, x_3^{(t)}, \dots, x_d^{(t)}) \\
x_2^{(t+1)} \sim p(x_2 | x_1^{(t+1)}, x_3^{(t)}, \dots, x_d^{(t)}) \\
\vdots \\
x_d^{(t+1)} \sim p(x_d | x_1^{(t+1)}, x_2^{(t+1)}, \dots, x_{d-1}^{(t+1)})
$$

每次只更新一个维度，但使用的是该变量在给定其他所有变量**最新值**下的条件分布。

**Gibbs 的优势**：不需要设计提议分布，不需要调接受率——只要知道条件分布就能采样。当条件分布容易采样时（如共轭先验下的后验分布），Gibbs 非常高效。

**Gibbs 的局限**：当变量强相关时，Gibbs 的"只沿坐标轴移动"导致链在参数空间中走得很慢（随机游走行为严重），混合效率低。

---

## 六、经典应用

### 6.1 估计 $\pi$

最简单的 MC 应用：在 $[-1, 1]^2$ 的正方形中均匀随机撒点，落在单位圆内的比例 $\times 4 \approx \pi$：

$$
\pi \approx 4 \times \frac{\#\{ (x,y) : x^2 + y^2 \le 1 \}}{N}
$$

### 6.2 估计复杂积分

对于没有解析形式的多维积分，MC 是默认工具。例如贝叶斯推断中的边缘似然（marginal likelihood）：

$$
p(\mathcal{D}) = \int p(\mathcal{D} | \theta) p(\theta) \, d\theta
$$

这个积分通常没有闭式解，MC 是计算它的标准方法。

### 6.3 采样复杂分布

很多统计模型的后验分布 $p(\theta | \mathcal{D})$ 没有标准形式，无法直接采样。MCMC（特别是 MH 和 Gibbs）是贝叶斯推断的计算引擎。

---

## 七、MCMC 诊断与实用技巧

| 要点 | 说明 |
|------|------|
| **Burn-in** | 丢弃链前期的样本（链尚未收敛到平稳分布），通常丢弃前 10-50% |
| **Thinning** | 每隔若干步取一个样本来减少自相关，$k$ 步 thinning 保留每 $k$ 个样本 |
| **多链（Multiple Chains）** | 运行多条从不同初始点出发的链，比较链间和链内方差（$\hat{R}$ 统计量） |
| **有效样本量（ESS）** | 考虑自相关后的"等效独立样本数"，$ESS = N / (1 + 2\sum \rho_k)$ |
| **Trace Plot** | 绘制采样值 vs 迭代次数的时序图，检查是否有趋势或漂移 |

> **实战建议**：运行至少 3 条链，检查 $\hat{R} < 1.1$（链间方差 ≈ 链内方差）。如果 trace plot 显示明显的趋势或"卡住"，说明链未混合——调整提议分布的标准差（MH）或重新参数化模型（Gibbs）。

---

## 本章总结

| 概念 | 一句话 |
|------|--------|
| 蒙特卡洛积分 | 用样本均值 $\frac{1}{N}\sum f(x_i)/p(x_i)$ 近似积分 |
| 收敛率 | $O(1/\sqrt{N})$，与维度无关 —— MC 在高维中的核心优势 |
| 重要性采样 | 用提议分布 $q(x)$ 替换 $p(x)$，加权修正，降低方差 |
| 最优提议分布 | $q^*(x) \propto \|f(x)\|$ —— 在 $f$ 大的地方多采样 |
| 拒绝采样 | $Mq(x)$ 包围 $\tilde{p}(x)$，接受落在下方的点 |
| MCMC | 构建平稳分布为 $p(x)$ 的马尔可夫链来间接采样 |
| Metropolis-Hastings | 提议 $x' \sim q(\cdot\|x)$，以 $\min(1, \frac{p(x')q(x\|x')}{p(x)q(x'\|x)})$ 接受 |
| 细致平衡 | $p(x)T(x \to x') = p(x')T(x' \to x)$，平稳分布的充分条件 |
| Gibbs 采样 | 从条件分布 $p(x_i \| \mathbf{x}_{-i})$ 逐维采样，接受率恒为 1 |
| Burn-in | 丢弃前期未收敛的样本 |
| 多链诊断 | 比较链间/链内方差（$\hat{R} < 1.1$），确保混合收敛 |

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="../code/ml10_monte_carlo/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="../code/ml10_monte_carlo/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *J. Chem. Phys.*, 21(6), 1087-1092. [[doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)]
2. Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109. [[doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)]
3. Geman, S. & Geman, D. (1984). Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images. *IEEE TPAMI*, 6(6), 721-741. [[doi:10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596)]
4. Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods (2nd ed.). *Springer*. [[doi:10.1007/978-1-4757-4145-2](https://doi.org/10.1007/978-1-4757-4145-2)]
5. Gelman, A. & Rubin, D. B. (1992). Inference from Iterative Simulation Using Multiple Sequences. *Statistical Science*, 7(4), 457-472. [[doi:10.1214/ss/1177011136](https://doi.org/10.1214/ss/1177011136)] ($\hat{R}$ 统计量)
