---
title: "数理统计与估计"
order: 24
---
# 数理统计与估计：数据是随机的，估计量是数据的函数

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 概率论假定 $p$、$\mu$ 已知，问样本会长什么样。**数理统计反过来**：样本已经落在桌上，问参数大概是多少。本章只抓三件：**点估计、极大似然、偏差（除以 $n$ 还是 $n-1$）**。前置：[常见分布](/math/distributions/)。区间与「平均为何像高斯」见 [CLT](/math/clt/)。贝叶斯把参数也当成随机变量，仍在 [概率与贝叶斯](/math/probability/)。

---

## 一、样本与估计量

$X_1,\ldots,X_n$ i.i.d. 来自某个带参数 $\theta$ 的分布。**估计量** $\hat\theta=g(X_1,\ldots,X_n)$ 是样本的函数，所以它自己也是随机变量。同一组硬币，再抛一次，$\hat p$ 会变。

常见无脑但好用的选择：

- 样本均值 $\bar X=\frac1n\sum X_i$ 估期望；
- 样本方差 $s^2=\frac1{n-1}\sum(X_i-\bar X)^2$ 估方差。

**无偏**：$\mathbb{E}[\hat\theta]=\theta$。除以 $n-1$ 就是为了让 $s^2$ 无偏——用 $\bar X$ 代替了未知的 $\mu$，少了一个自由度。

![极大似然](./images/math-stat-mle.png)

> **图解说明**：左是数据；中是似然山峰，$\hat p_{\mathrm{MLE}}$ 在山顶；右是高斯云里样本均值对准 $\mu$，并标出 $n-1$。底栏：估计量随数据抖。

---

## 二、极大似然

似然把参数当变量、数据当常数：

$$
L(\theta)=\prod_{i=1}^n f(x_i\mid\theta).
$$

$\hat\theta_{\mathrm{MLE}}=\arg\max_\theta L(\theta)$（通常对 $\log L$ 求导更稳）。

- 伯努利：$L(p)=p^{k}(1-p)^{n-k}$，$k$ 次正面 $\Rightarrow \hat p=k/n=\bar X$。
- 高斯（$\mu,\sigma^2$ 都未知）：$\hat\mu=\bar X$，$\hat\sigma^2_{\mathrm{MLE}}=\frac1n\sum(X_i-\bar X)^2$。注意 **MLE 除以 $n$**，所以对 $\sigma^2$ **有偏**（略偏小）。无偏方差才除以 $n-1$。

平方误差损失对应高斯、固定 $\sigma$ 时的最大似然——这就是「最小二乘从哪来」的统计说法。分类的交叉熵是另一张脸的 MLE，见 [信息论精简](/math/information/)。

---

## 三、代码在做什么

左三张直方图：$\hat p$ 在 $n=5,20,80$ 时越挤越紧（还没到 CLT 的正规形状，那是下一章）。右：同一批 $n=8$ 的高斯样本，MLE 方差整体比真值 4 偏左，无偏版本中心更正。

![伯努利 MLE 随 n](./images/est_bernoulli.png)

![方差：/n vs /(n-1)](./images/est_variance.png)

C++ `mle.hpp` 对硬编码的 10 次抛币印 $\hat p=0.7$，对 $\{1,2,3,4,5\}$ 印高斯 MLE。

```bash
cd docs/math/estimation/code
python demo.py
g++ -std=c++17 demo.cpp -o est_demo
```

---

## 四、小结

| 概念 | 一句话 |
|------|--------|
| 估计量 | 数据的函数，本身随机 |
| 无偏 | 平均瞄得准；不保证单次准 |
| MLE | 让当前数据最像被 $\theta$ 生出来 |
| $n$ vs $n-1$ | MLE 方差偏小；样本方差无偏 |
| 下游 | 回归、EM、[CLT](/math/clt/) |

> 下一章 [大数定律与中心极限](/math/clt/)：为什么 $n$ 大时 $\hat\theta$ 会老实，以及为何误差条常画成高斯。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/math/estimation/demo.py" target="_blank" download>Download</a> |
| mle.hpp | — | <a href="/notebook/code/math/estimation/mle.hpp" target="_blank" download>Download</a> |
| demo.cpp | — | <a href="/notebook/code/math/estimation/demo.cpp" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/math/estimation/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Casella & Berger, *Statistical Inference*
2. Wasserman, *All of Statistics*（短而全）
