---
title: "常见分布"
order: 22
---
# 常见分布：计数、等待、噪声各用哪一张脸

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 上一章用贝叶斯更新信念，但似然 $P(\mathrm{data}\mid\theta)$ 必须先指定「数据长什么样」。本章只放五张最常用的脸：**伯努利 / 二项、泊松、指数、高斯**，并记住各自的期望与方差。前置：[概率与贝叶斯](/math/probability/)。怎么从样本反推参数，见下一章 [估计](/math/estimation/)。

---

## 一、同一套数字特征，不同的生成故事

选分布不是看哪条曲线好看，而是问数据**怎么生成**的。抛一次币、抛 $n$ 次再计数、稀有事件的个数、等到下一辆车、大量微小误差叠加——五张脸对应五种故事。数字特征却是同一套：期望（中心）和方差（分散）。

| 分布 | 故事 | $\mathbb{E}$ | $\mathrm{Var}$ |
|------|------|--------------|----------------|
| Bernoulli$(p)$ | 一次抛币 | $p$ | $p(1-p)$ |
| Binomial$(n,p)$ | $n$ 次独立抛币的正面数 | $np$ | $np(1-p)$ |
| Poisson$(\lambda)$ | 稀有事件的计数 | $\lambda$ | $\lambda$ |
| Exp$(\lambda)$ | 等到下一事件的时间 | $1/\lambda$ | $1/\lambda^2$ |
| $\mathcal{N}(\mu,\sigma^2)$ | 大量微小误差叠加 | $\mu$ | $\sigma^2$ |

$n$ 大、$p$ 小、$np=\lambda$ 固定时，二项的 pmf 会贴向泊松——demo 里 $n=10,p=0.3,\lambda=3$ 已经能看出轮廓像（还没完全贴死：$n=10$ 谈不上「很大」）。指数的无记忆：已经等了 $t$ 秒，再等 $s$ 秒的分布与从头等 $s$ 秒一样。高斯的「默认噪声」地位来自中心极限，详见 [CLT](/math/clt/)。

![常见分布一家人](./images/math-prob-dist.png)

> **图解说明**：左三是离散计数，右二是连续密度。同一套 $\mathbb{E},\mathrm{Var}$；选哪一张脸取决于数据怎么生成，不是看哪条曲线好看。

**保姆级数字例。** 伯努利 $p=0.3$：期望 $0.3$，方差 $0.3\times 0.7=0.21$。抛 $n=10$ 次，二项期望 $np=3$，方差 $2.1$，标准差 $\sqrt{2.1}\approx 1.45$——所以 $K$ 多数落在 $0$ 到 $6$ 附近，极端的 $K=10$ 很稀有。泊松 $\lambda=3$ 期望、方差都是 $3$，比二项略散一点（二项有「最多 $n$ 次」的硬顶，泊松没有）。

**卡点。** 泊松「均值=方差」是诊断：若计数数据的样本方差远大于样本均值（过离散），泊松会低估尾部，该换成负二项。指数的「无记忆」对机械寿命往往不成立（零件会老化）；它对放射性衰变、理想排队更合适。高斯对称、可取负值：等待时间、降雨量不要直接套高斯。

---

## 二、密度公式（代码里就是这几行）

二项：

$$
P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}.
$$

泊松（用对数累加避免 $k!$ 爆）：

$$
P(K=k)=e^{-\lambda}\frac{\lambda^k}{k!}.
$$

高斯与指数：

$$
\phi(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big),
\qquad
f(x)=\lambda e^{-\lambda x}\ (x\ge 0).
$$

C++ `dist.hpp` 与 Python `demo.py` 打印同一组探针：$\mathrm{Binom}(10,0.3)$ 的 $P(K=3)$、$\varphi(0)\approx 0.3989$。

**保姆级数字例（与 demo 探针一致）。**

$$
\begin{aligned}
P_{\mathrm{Binom}}(K=3)
&=\binom{10}{3}(0.3)^3(0.7)^7
=120\times 0.027\times 0.0823543
\approx 0.2668,\\[4pt]
P_{\mathrm{Poisson}}(K=3)
&=e^{-3}\frac{3^3}{3!}
\approx 0.049787\times 4.5
\approx 0.2240,\\[4pt]
\phi(0)
&=\frac{1}{\sqrt{2\pi}}\approx 0.3989.
\end{aligned}
$$

二项在 $k=3$ 处比泊松更高一点：因为 $n=10$ 还不够大，$p=0.3$ 也不算很小，泊松极限只是「轮廓像」。标准正态在 $0$ 处的高度 $0.3989$ 是记住「铃铛有多高」的尺子——$\sigma$ 若改成 $2$，峰高变成 $0.3989/2\approx 0.1995$。

::: details 逐步推导：二项如何从 $n$ 次伯努利加出来，以及 $P(K=3)\approx 0.2668$（点击展开）

一次伯努利：$X_i\in\{0,1\}$，$P(X_i=1)=p$。令 $K=X_1+\cdots+X_n$。$K=k$ 意味着某 $k$ 个位置是 $1$、其余是 $0$。一种具体序列的概率是 $p^k(1-p)^{n-k}$；这样的序列有 $\binom{n}{k}$ 种，所以

$$
P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}.
$$

期望用线性（不必独立）：$\mathbb{E}K=n\mathbb{E}X_1=np$。方差用独立可加：$\mathrm{Var}(K)=np(1-p)$。

代入 $n=10$、$p=0.3$、$k=3$：

$$
\binom{10}{3}=\frac{10\cdot 9\cdot 8}{6}=120,
\quad
0.3^3=0.027,
\quad
0.7^7=0.0823543.
$$

$120\times 0.027=3.24$，$3.24\times 0.0823543\approx 0.2668$。这就是终端打印的 `Binom P(K=3)`。

组合数 $\binom{n}{k}$ 会溢出：C++ 用乘除交错算 $\binom{n}{k}$，泊松 pmf 用 $\log$ 累加 $- \lambda + k\log\lambda -\sum_{i=2}^k\log i$ 再 $\exp$，避免先算 $k!$。Python `math.comb` 在 $n=10$ 上完全安全。

伯努利方差 $p(1-p)$ 在 $p=1/2$ 最大（$=1/4$），在 $p\to 0$ 或 $1$ 时趋于 $0$——很偏的硬币几乎每次都一样，计数几乎不散。这就是「确定事件熵小」在方差上的影子。

:::

::: details 逐步推导：泊松极限 $np=\lambda$ 与指数无记忆（点击展开）

**泊松是二项的稀有事件极限。** 令 $p=\lambda/n$，固定 $k$，令 $n\to\infty$：

$$
\begin{aligned}
P(K=k)
&=\frac{n(n-1)\cdots(n-k+1)}{k!}\Big(\frac{\lambda}{n}\Big)^k\Big(1-\frac{\lambda}{n}\Big)^{n-k}\\
&=\frac{\lambda^k}{k!}\cdot\underbrace{\frac{n}{n}\cdot\frac{n-1}{n}\cdots\frac{n-k+1}{n}}_{\to 1}
\cdot\Big(1-\frac{\lambda}{n}\Big)^{n}\cdot\Big(1-\frac{\lambda}{n}\Big)^{-k}.
\end{aligned}
$$

$(1-\lambda/n)^n\to e^{-\lambda}$，$(1-\lambda/n)^{-k}\to 1$。于是

$$
P(K=k)\to e^{-\lambda}\frac{\lambda^k}{k!}.
$$

demo 用 $n=10$、$p=0.3$、$\lambda=3$：$P_{\mathrm{Binom}}(K=3)\approx 0.2668$ 对 $P_{\mathrm{Poisson}}(K=3)\approx 0.2240$，相对差约 $19\%$——轮廓像，定量还早。经验上 $n\ge 50$、$p\le 0.1$ 才开始「贴」。泊松均值等于方差：$\mathbb{E}K=\mathrm{Var}(K)=\lambda$。

**指数来自泊松过程的等待。** 若事件按速率 $\lambda$ 独立发生，则等第一个事件的时间 $T$ 满足 $P(T>t)=P(\text{这段里 0 个事件})=e^{-\lambda t}$，密度 $f(t)=\lambda e^{-\lambda t}$（$t\ge 0$）。期望 $\mathbb{E}T=1/\lambda$，方差 $1/\lambda^2$。$\lambda=1$ 时密度在 $0$ 处是 $1$，然后按 $e^{-t}$ 下降；负半轴为 $0$——这就是右图那条折线。

**无记忆。** 已经等了 $s$ 秒还没来：

$$
P(T>s+t\mid T>s)
=\frac{P(T>s+t)}{P(T>s)}
=\frac{e^{-\lambda(s+t)}}{e^{-\lambda s}}
=e^{-\lambda t}
=P(T>t).
$$

再等 $t$ 秒的分布，和刚开始等一模一样。几何分布（离散等待）有同样的性质。有老化的寿命模型（Weibull、$\Gamma$）会打破它。

**高斯在 $0$ 处。** $\mu=0$、$\sigma=1$：

$$
\phi(0)=\frac{1}{\sqrt{2\pi}}\approx\frac{1}{2.5066}\approx 0.3989,
$$

与终端 `N(0,1) φ(0)=0.3989` 一致。积分 $\int_{-\infty}^{\infty}\phi=1$ 需要高斯积分 $\int e^{-x^2/2}\,dx=\sqrt{2\pi}$，本章当作已知。大量独立微小误差之和长成这只铃铛，证明见 [CLT](/math/clt/)。

![二项与泊松](./images/dist_discrete.png)

> **图解说明**：$n=10$、$p=0.3$ 的二项柱，对 $\lambda=3$ 的泊松柱。峰都在 $k=3$ 附近，高度和尾巴略有差别。

:::

---

## 三、代码在做什么

左：二项柱与泊松柱并排。右：标准正态铃铛对指数折线（指数在负半轴为 $0$）。

![二项与泊松](./images/dist_discrete.png)

![高斯与指数](./images/dist_continuous.png)

```bash
cd docs/math/distributions/code
python demo.py
g++ -std=c++17 demo.cpp -o dist_demo
```

---

## 四、小结

| 概念 | 一句话 |
|------|--------|
| 伯努利 / 二项 | 抛币与计数；$P(K=3)\approx 0.2668$（$n=10,p=0.3$） |
| 泊松 | 稀有事件；均值=方差；二项的 $np=\lambda$ 极限 |
| 指数 | 等待时间；无记忆；只活在 $x\ge 0$ |
| 高斯 | 对称噪声；$\phi(0)\approx 0.3989$；平方损失的亲戚 |
| 下游 | 似然、GLM、[估计](/math/estimation/) |

> 下一章 [数理统计与估计](/math/estimation/)：有了生成故事，怎么用数据猜 $p$、$\mu$、$\sigma^2$。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/math/distributions/demo.py" target="_blank" download>Download</a> |
| dist.hpp | — | <a href="/notebook/code/math/distributions/dist.hpp" target="_blank" download>Download</a> |
| demo.cpp | — | <a href="/notebook/code/math/distributions/demo.cpp" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/math/distributions/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Blitzstein & Hwang, *Introduction to Probability*
2. Feller, *An Introduction to Probability Theory*（经典叙事）
