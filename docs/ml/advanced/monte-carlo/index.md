---
title: "ml10 蒙特卡洛方法"
order: 50
legacyPaths:
  - /ml10_monte_carlo/
---
# ml10 蒙特卡洛方法

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 用随机数解决确定性问题——从估算 $\pi$ 到采样复杂分布。独立同分布采样且方差有限时，RMSE 按 $1/\sqrt N$ 衰减；指数与维度无关，但方差常数仍可能随维度恶化。`demo.py` 会画投点估面积、重要性采样和 MH 轨迹。$\pi$ 的投针/投点公式在折叠里。

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

以上要求采样密度在被积函数非零处为正；独立同分布采样且 $\mathbb E_p|f/p|<\infty$ 时可应用大数定律。下面的 CLT 与 $N^{-1/2}$ RMSE 结论还要求方差有限。

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

> **图示待补**：当前积分示意图是 1×1 占位文件，暂不展示。请参考本节公式与下方估算 $\pi$ 的数字例。

> **几何直觉**：区域内点数占比 × 包围盒面积 ≈ 区域面积；独立同分布采样且方差有限时，RMSE 按 $1/\sqrt N$ 衰减。

**数字例（估 $\pi$）。** 在 $[-1,1]^2$ 均匀投 $N$ 点，圆内比例 $\hat p$ 满足 $\pi\approx 4\hat p$。$N=100$ 时常见 $3.0\sim 3.3$；$N=10^4$ 大概到小数点后两位。标准差 $\sqrt{p(1-p)/N}\cdot 4$，要再准一位大约 $N\times 100$。

::: details 逐步推导：从期望到 $O(N^{-1/2})$ 以及重要性采样权重（点击展开）

$I=\mathbb E_{x\sim p}[f(x)/p(x)]$。样本均值无偏。方差 $\sigma^2/N$，故 RMSE 是 $1/\sqrt N$。网格法在 $d$ 维是 $N^{-r/d}$，高维时 MC 反超。

重要性：$I=\mathbb E_{q}[f/q]$。权重 $w=f/q$（或 $p/q$ 若估期望）。$q$ 要盖住 $f$ 大的地方，否则少数巨大权重主导，有效样本量 $n_{\mathrm{eff}}=(\sum w)^2/\sum w^2$ 崩掉。MH：从 $q(x'\mid x)$ 提议，接受率 $\min\bigl(1,\frac{\tilde p(x')q(x\mid x')}{\tilde p(x)q(x'\mid x)}\bigr)$；$\tilde p$ 不用归一化。Burn-in 丢掉开头，因为还没进平稳分布。

:::

---

::: details 把误差、置信区间和稀有事件方差逐步算清楚

令 $Y_i=f(X_i)/q(X_i)$，$X_i$ 独立同分布来自 $q$。若 $q>0$ 覆盖 $f\ne0$ 的区域，且 $\int |f|<\infty$，则
$\mathbb E_qY_i=\int(f/q)q=\int f=I$，所以 $\hat I_N=N^{-1}\sum_iY_i$ 无偏，强大数定律给出一致性。
若还满足 $\mathbb E_qY_i^2<\infty$，独立性消去协方差项：

$
\operatorname{Var}(\hat I_N)
=\frac1{N^2}\sum_{i=1}^N\operatorname{Var}(Y_i)
=\frac{\sigma_Y^2}{N},\qquad
\operatorname{RMSE}(\hat I_N)=\frac{\sigma_Y}{\sqrt N}.
$

用 $s_Y^2=(N-1)^{-1}\sum_i(Y_i-\bar Y)^2$ 估计方差时，CLT 的近似区间为
$\hat I_N\pm1.96s_Y/\sqrt N$；它要求渐近近似已经合理，不能把“公式算得出”当作“覆盖率有保证”。MCMC 样本相关时，方差还包含自协方差，不能直接套独立公式。

**稀有事件反例。** 本例 $Y_i=\mathbf1\{X_i>5\}$，$p=\Pr(X>5)\approx2.87\times10^{-7}$。
$\operatorname{Var}(\hat p)=p(1-p)/N$；$N=10^4$ 时，相对标准差
$\sqrt{(1-p)/(Np)}\approx18.7$，即约 $1870\%$。
零命中的概率 $(1-p)^N\approx e^{-Np}\approx0.9971$。因此几乎总会看到样本均值与样本方差都为零，二者都不能证明真实概率为零。
零命中时，一侧 $95\%$ 二项上界由 $(1-p_{\rm upper})^N=0.05$ 给出：
$p_{\rm upper}=1-0.05^{1/N}\approx3/N$，比目标概率大约三个数量级。
代码比较方差时使用已知 $p$ 的理论 MC 方差，避免零命中的假象。

**重要性权重为什么能纠偏。** 从 $q=\mathcal N(5,1)$ 采样时，
$\log w=\log p(x)-\log q(x)=-5x+12.5$；
$Y=w(X)\mathbf1\{X>5\}$ 才是用于取平均和计算样本方差的量。
约一半提议落在 $x>5$，但每个点的权重都很小，平均后仍恢复原分布的极小尾概率。
已知归一化密度时应除以 $N$；改成除以 $\sum w_i$ 是自归一化重要性采样，通常有限样本有偏，是另一种估计器。

:::

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

- **原始 MC**：从 $\mathcal{N}(0, 1)$ 采样 10000 次，预期命中数仅约 $10000\times2.87\times10^{-7}=0.00287$，通常一次也命不中——估计极不稳定
- **重要性采样**：从 $\mathcal{N}(5, 1)$ 采样，大量样本集中在目标区域，通过权重 $p(x)/q(x)$ 修正——估计高效且精确

> **图示待补**：当前重要性采样示意图是 1×1 占位文件，暂不展示。核心是从 $q$ 采样，再用 $f/q$（估计期望时为 $p/q$）修正权重。

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
2. **链"混合"需要多长时间？** —— 用多链、轨迹图、$\hat R$ 与 ESS 诊断；burn-in 与 thinning 本身不保证混合

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

对 $x'\ne x$，转移部分为 $T(x \to x') = q(x' | x) \cdot \alpha(x, x')$；完整转移核还包括拒绝提议后留在 $x$ 的概率质量。细致平衡是平稳分布为 $p(x)$ 的充分条件。

> **直觉**：对于本例的对称随机游走提议，$\alpha=\min(1,p(x')/p(x))$：密度更高的候选点总被接受，密度更低的点按相应概率接受。一般的非对称提议还必须包含 $q(x\mid x')/q(x'\mid x)$，不能只比较目标密度。

> **图示待补**：当前 MH 示意图是 1×1 占位文件，暂不展示。采样轨迹和收敛诊断应由实际运行结果验证。

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
| **Burn-in** | 丢弃初期受起点影响较大的样本；丢弃固定比例不能保证收敛 |
| **Thinning** | 每 $k$ 步保留一个样本以节省存储；同等迭代预算下通常会损失信息，不能替代混合诊断 |
| **多链（Multiple Chains）** | 运行多条从不同初始点出发的链，比较链间和链内方差（$\hat{R}$ 统计量） |
| **有效样本量（ESS）** | 考虑自相关后的"等效独立样本数"，$ESS = N / (1 + 2\sum \rho_k)$ |
| **Trace Plot** | 绘制采样值 vs 迭代次数的时序图，检查是否有趋势或漂移 |

> **实战建议**：从分散初值运行多条链，联合检查 rank-normalized split-$\hat R$（常用参考阈值 $1.01$）、bulk/tail ESS 和轨迹图；单个阈值通过不是收敛证明。明显趋势或“卡住”提示混合不足，可调整 MH 提议或重新参数化。参见 [Stan 的 MCMC 诊断说明](https://mc-stan.org/docs/reference-manual/analysis.html)。

---

## 本章总结

| 概念 | 一句话 |
|------|--------|
| 蒙特卡洛积分 | 用样本均值 $\frac{1}{N}\sum f(x_i)/p(x_i)$ 近似积分 |
| 收敛率 | i.i.d. 且有限方差时 RMSE 为 $O(1/\sqrt N)$；常数仍可能随维度恶化 |
| 重要性采样 | 用提议分布 $q(x)$ 替换 $p(x)$，加权修正，降低方差 |
| 最优提议分布 | $q^*(x) \propto \|f(x)\|$ —— 在 $f$ 大的地方多采样 |
| 拒绝采样 | $Mq(x)$ 包围 $\tilde{p}(x)$，接受落在下方的点 |
| MCMC | 构建平稳分布为 $p(x)$ 的马尔可夫链来间接采样 |
| Metropolis-Hastings | 提议 $x' \sim q(\cdot\|x)$，以 $\min(1, \frac{p(x')q(x\|x')}{p(x)q(x'\|x)})$ 接受 |
| 细致平衡 | $p(x)T(x \to x') = p(x')T(x' \to x)$，平稳分布的充分条件 |
| Gibbs 采样 | 从条件分布 $p(x_i \| \mathbf{x}_{-i})$ 逐维采样，接受率恒为 1 |
| Burn-in | 丢弃前期未收敛的样本 |
| 多链诊断 | 联合检查 $\hat R$、ESS 和轨迹图；不能仅凭阈值证明收敛 |

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/ml/advanced/monte-carlo/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/ml/advanced/monte-carlo/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of State Calculations by Fast Computing Machines. *J. Chem. Phys.*, 21(6), 1087-1092. [[doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)]
2. Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109. [[doi:10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97)]
3. Geman, S. & Geman, D. (1984). Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images. *IEEE TPAMI*, 6(6), 721-741. [[doi:10.1109/TPAMI.1984.4767596](https://doi.org/10.1109/TPAMI.1984.4767596)]
4. Robert, C. P. & Casella, G. (2004). Monte Carlo Statistical Methods (2nd ed.). *Springer*. [[doi:10.1007/978-1-4757-4145-2](https://doi.org/10.1007/978-1-4757-4145-2)]
5. Gelman, A. & Rubin, D. B. (1992). Inference from Iterative Simulation Using Multiple Sequences. *Statistical Science*, 7(4), 457-472. [[doi:10.1214/ss/1177011136](https://doi.org/10.1214/ss/1177011136)] ($\hat{R}$ 统计量)
