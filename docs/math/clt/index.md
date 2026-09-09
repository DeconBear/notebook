---
title: "大数定律与中心极限"
order: 26
---
# 大数定律与中心极限：平均会老实，而且长成铃铛

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [估计](/math/estimation/) 说 $\hat\theta$ 是随机变量。本章回答两个「然后呢」：**大数定律（LLN）**——$n$ 大时 $\bar X$ 贴向 $\mathbb{E}X$；**中心极限定理（CLT）**——贴的方式是 $\sqrt{n}(\bar X-\mu)$ 长成高斯，即使原来的 $X$ 根本不是高斯。这是「用训练集平均代替真实风险」、以及误差条画成 $\pm 1.96\sigma/\sqrt{n}$ 的许可证。前置：[常见分布](/math/distributions/)。

---

## 一、大数定律

$X_i$ i.i.d.，$\mathbb{E}|X_1|<\infty$，则

$$
\bar X_n=\frac1n\sum_{i=1}^n X_i \ \to\ \mu=\mathbb{E}X_1
$$

（依概率；还有几乎处处的强形式，这里不展开）。直觉就一句：**独立噪声在平均里互相抵消，中心留下来。**

抛币 $p=0.6$，运行均值会从乱抖收成一条贴着 $0.6$ 的线——demo 左图抽 $n=400$ 次。它**不**保证分布是高斯，只保证中心找对。

机器学习里把损失的期望换成样本平均，靠的就是 LLN。SGD 的 minibatch 是「小 $n$ 的 LLN」：平均大致朝真梯度，但每次仍抖——抖的幅度大约是 $\sigma/\sqrt{B}$，$B$ 是 batch size。

**保姆级数字例。** 伯努利 $p=0.6$ 的方差 $p(1-p)=0.24$。样本均值的标准差是

$$
\frac{\sigma}{\sqrt{n}}=\frac{\sqrt{0.24}}{\sqrt{400}}\approx\frac{0.490}{20}\approx 0.0245.
$$

所以 $n=400$ 时，$\bar X$ 典型地落在 $0.6\pm 0.02$ 这一带，不会还在 $0$ 和 $1$ 之间乱跳。单次抛币的标准差是 $\sqrt{0.24}\approx 0.49$，平均之后稳了约 $\sqrt{400}=20$ 倍。

**卡点。** LLN 要的是期望存在。Cauchy 分布没有期望，样本均值会继续乱走——「平均一定收敛」不是宇宙定律。相关样本（马尔可夫链、同一篇文章里的相邻词）让有效样本量变小，收敛变慢，公式里的 $n$ 不能按条数傻数。

::: details 逐步推导：为什么 $\mathrm{Var}(\bar X)=\sigma^2/n$，以及 Chebyshev 如何推出 LLN（点击展开）

设 $X_i$ i.i.d.，$\mathbb{E}X_i=\mu$，$\mathrm{Var}(X_i)=\sigma^2<\infty$。均值的期望

$$
\mathbb{E}[\bar X_n]=\frac1n\sum_i\mathbb{E}X_i=\mu
$$

——平均作为估计量是无偏的。方差用独立可加：

$$
\mathrm{Var}\Big(\sum_{i=1}^n X_i\Big)=n\sigma^2,
\qquad
\mathrm{Var}(\bar X_n)=\mathrm{Var}\Big(\frac1n\sum X_i\Big)=\frac1{n^2}\cdot n\sigma^2=\frac{\sigma^2}{n}.
$$

标准差按 $\sqrt{n}$ 变小，不是按 $n$。想把误差再压一半，样本量要乘 $4$——这是蒙特卡洛、A/B 测试、误差条里同一条账。

Chebyshev：对任意随机变量（只要方差有限），$P(|Y-\mathbb{E}Y|\ge\varepsilon)\le\mathrm{Var}(Y)/\varepsilon^2$。把 $Y=\bar X_n$ 代进去：

$$
P\big(|\bar X_n-\mu|\ge\varepsilon\big)\le\frac{\sigma^2}{n\varepsilon^2}.
$$

$n\to\infty$ 时右边 $\to 0$。这就是**依概率收敛**（弱大数定律）的一种证法。界很松：伯努利 $p=0.6$、$n=400$、$\varepsilon=0.05$ 时右边 $=0.24/(400\cdot 0.0025)=0.24$，只告诉你「偏离超过 $0.05$ 的概率 $\le 24\%$」；CLT 用高斯尾会给出紧得多的数。Chebyshev 的价值是：**不需要 $X$ 是高斯，只要方差有限。**

指数分布 $\mathrm{Exp}(\lambda)$ 的期望 $1/\lambda$、方差 $1/\lambda^2$。demo 取 $\lambda=1$，于是 $\mu=1$、$\sigma^2=1$，$n=30$ 时 $\mathrm{Var}(\bar X)=1/30\approx 0.0333$。C++ 用最小 LCG 抽 $\mathrm{Exp}(1)$，就是在核对这一个数。

:::

---

## 二、中心极限

设 $\mathrm{Var}(X_1)=\sigma^2<\infty$，则

$$
\sqrt{n}\,(\bar X_n-\mu)\ \xrightarrow{d}\ \mathcal{N}(0,\sigma^2),
$$

或等价地 $\bar X_n$ 大约是 $\mathcal{N}(\mu,\sigma^2/n)$。原分布可以是指数（右偏、只活在正半轴）：$n=1$ 时直方图还是折线，$n=5$ 开始鼓包，$n=30$ 已经像铃铛。

![中心极限示意](./images/math-stat-clt.png)

> **图解说明**：三档样本量。LLN 是「中心贴过去」；CLT 是「贴的误差长成高斯」。底栏：原分布不必是高斯。

标准化样本均值

$$
Z_n=\frac{\sqrt{n}\,(\bar X_n-\mu)}{\sigma}
$$

在练习里手写。置信区间 $\bar X\pm 1.96\,\sigma/\sqrt{n}$ 就是在假装 $Z_n\approx\mathcal{N}(0,1)$：标准正态约 $95\%$ 的质量落在 $[-1.96,1.96]$。

**保姆级数字例。** $\mathrm{Exp}(1)$ 本身：$P(X>x)=e^{-x}$（$x\ge 0$），右偏，众数在 $0$，均值 $1$。demo 重复 $4000$ 次取平均：

| $n$ | $\bar X$ 的理论标准差 $\sigma/\sqrt{n}$ | 直方图看起来 |
|-----|------------------------------------------|--------------|
| $1$ | $1$ | 还是指数折线 |
| $5$ | $1/\sqrt{5}\approx 0.447$ | 开始鼓包，仍略右偏 |
| $30$ | $1/\sqrt{30}\approx 0.183$ | 已经像铃铛；经验方差应接近 $1/30\approx 0.0333$ |

$n=30$ 不是魔法阈值，只是「偏得不太离谱的分布」的经验尺。偏度极大（比如 Pareto 尾）时，$30$ 远远不够。

**卡点。** CLT 说的是 **$\bar X$ 的分布**，不是「每个 $X_i$ 变成高斯」。你不能把一张右偏的原始直方图平滑成铃铛，然后说「CLT」。另一个卡点：误差条 $\pm 1.96\sigma/\sqrt{n}$ 里的 $\sigma$ 若用样本估，小 $n$ 时应换成 $t$ 分位数；demo 叠的是**已知** $\sigma$ 的理论高斯，用来对照形状。

::: details 逐步推导：从「独立和的方差」到标准化 $Z_n\to\mathcal{N}(0,1)$（点击展开）

**第一步：位置和尺度。** $\bar X_n$ 的均值是 $\mu$，方差是 $\sigma^2/n$。减掉 $\mu$、再除以标准差，得到

$$
Z_n=\frac{\bar X_n-\mu}{\sigma/\sqrt{n}}=\frac{\sqrt{n}\,(\bar X_n-\mu)}{\sigma}.
$$

这样 $\mathbb{E}Z_n=0$、$\mathrm{Var}(Z_n)=1$，无论 $n$ 是多少。CLT 多说的是：**分布形状**也变成标准正态。

**第二步：为什么是高斯（特征函数速写）。** 设 $Y_i=(X_i-\mu)/\sigma$，则 $\mathbb{E}Y_i=0$、$\mathrm{Var}(Y_i)=1$，$Z_n=n^{-1/2}\sum Y_i$。特征函数 $\varphi(t)=\mathbb{E}e^{itY_1}$ 在 $0$ 附近

$$
\varphi(t)=1-\frac{t^2}{2}+o(t^2)
$$

（二阶泰勒，用了均值 $0$、方差 $1$）。于是

$$
\mathbb{E}e^{it Z_n}=\big[\varphi(t/\sqrt{n})\big]^n
=\Big(1-\frac{t^2}{2n}+o(n^{-1})\Big)^n
\to e^{-t^2/2},
$$

右边正是 $\mathcal{N}(0,1)$ 的特征函数。这就是「很多独立、方差有限的小贡献加在一起，只留下均值和方差，更高阶被 $n$ 洗掉」。

**第三步：区间。** 若 $Z_n\approx\mathcal{N}(0,1)$，则

$$
P\big(|Z_n|\le 1.96\big)\approx 0.95
\iff
P\Big(\bar X_n\in\Big[\mu-1.96\frac{\sigma}{\sqrt{n}},\ \mu+1.96\frac{\sigma}{\sqrt{n}}\Big]\Big)\approx 0.95.
$$

对 $\mathrm{Exp}(1)$、$n=30$：$1.96/\sqrt{30}\approx 0.36$，所以 $\bar X$ 大约有 $95\%$ 的机会落在 $[0.64,\,1.36]$。原指数变量 $X$ 自己落在这个区间的概率远没有 $95\%$——**平均**才有这个精度。

指数的偏度是 $2$，CLT 的「还你高斯」对偏度的修正阶是 $1/\sqrt{n}$（Edgeworth）。$n=5$ 时 $2/\sqrt{5}\approx 0.89$，直方图仍能看出右偏；$n=30$ 时 $2/\sqrt{30}\approx 0.37$，肉眼就比较像铃铛了。这与 demo 三张图一致。

![CLT 直方图](./images/clt_hist.png)

> **图解说明**：$\mathrm{Exp}(1)$ 的样本均值，$n=1,5,30$，各 $4000$ 次重复。红线是 $\mathcal{N}(1,1/n)$，不是原指数密度。

:::

---

## 三、代码在做什么

`clt_lln.png`：一条 $p=0.6$ 的运行均值，横轴到 $n=400$。`clt_hist.png`：指数样本均值的三档直方图，叠一条理论高斯。C++ 用最小 LCG 抽 $\mathrm{Exp}(1)$，印 $n=30$ 时 $\mathrm{Var}(\bar X)$ 是否接近 $1/30$。

![LLN 运行均值](./images/clt_lln.png)

![CLT 直方图](./images/clt_hist.png)

```bash
cd docs/math/clt/code
python demo.py
g++ -std=c++17 demo.cpp -o clt_demo
```

---

## 四、小结

| 概念 | 一句话 |
|------|--------|
| LLN | $\bar X\to\mu$，中心找对；方差按 $1/n$ 降 |
| CLT | $\sqrt{n}(\bar X-\mu)$ 变高斯；原分布不必是高斯 |
| $\sigma/\sqrt{n}$ | 均值比单点更稳，稳 $\sqrt{n}$ 倍 |
| $1.96$ | 标准正态 $95\%$ 分位；误差条的来源 |
| 下游 | 误差条、SGD 噪声、[蒙特卡洛](/ml/advanced/monte-carlo/) |

> 下一章 [优化与梯度](/math/optimization/)：把「平均损失」写成 $L(\theta)$，参数怎么下山。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/math/clt/demo.py" target="_blank" download>Download</a> |
| clt.hpp | — | <a href="/notebook/code/math/clt/clt.hpp" target="_blank" download>Download</a> |
| demo.cpp | — | <a href="/notebook/code/math/clt/demo.cpp" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/math/clt/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Durrett, *Probability: Theory and Examples*（定理表述）
2. Blitzstein & Hwang, *Introduction to Probability*（例题向）
