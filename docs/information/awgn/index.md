---
title: "高斯信道"
order: 50
---
# 高斯信道：功率对噪声

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [BSC](/information/shannon/) 是离散翻转。真实链路常常是 **加性白高斯噪声**：$Y=X+Z$，$Z\sim\mathcal{N}(0,N)$，发送功率受限 $\mathbb{E}[X^2]\le P$。香农公式（每实数维、每用一次）

$$
C=\frac12\log_2\bigl(1+\tfrac{P}{N}\bigr)=\frac12\log_2(1+\mathrm{SNR}).
$$

输入高斯时达到容量。纠错码仍要满足 $R<C$，只是噪声模型换了；短码例子见 [Hamming](/information/channel-coding/)。

---

## 一、微分熵差

连续变量用微分熵 $h$。高斯噪声下 $h(Y\mid X)=h(Z)$ 与 $X$ 无关。功率约束下 $Y$ 的微分熵在 $X$ 高斯时最大，于是 $I(X;Y)=h(Y)-h(Z)$ 得到上面的 $C$。不必在本章推完微分熵的变分——记住「容量随 SNR 对数涨」即可。

![AWGN 功率对噪声](./images/info-awgn.png)

> **图解说明**：左是 $Y=X+Z$ 两个高斯；中是 $C=\tfrac12\log_2(1+\mathrm{SNR})$；右是容量随 SNR(dB) 上升。BSC 用 $h_2(p)$，这里用功率比。

$\mathrm{SNR}_{\mathrm{dB}}=10\log_{10}(P/N)$。0 dB 时 $C=0.5$ bit；20 dB 时约 $3.3$ bit。

---

## 二、代码在做什么

`demo.py` 左图画 $C(\mathrm{SNR})$；右图在 0 dB 与 10 dB 下各撒 $\pm\sqrt{P}$ 二进制输入加噪声的散点（不是容量实现，只让「云有多糊」看得见）。终端打印两三个 SNR 的容量。

![AWGN 容量与散点](./images/awgn_cap.png)

二进制输入达不到公式里的 $C$（公式假定高斯输入），但散点仍能看出 SNR 高时两团分开。

---

## 三、小结

| 概念 | 一句话 |
|------|--------|
| AWGN | $Y=X+Z$，功率 $\le P$ |
| SNR | $P/N$；dB 是 $10\log_{10}$ |
| $C$ | $\tfrac12\log_2(1+\mathrm{SNR})$ |
| 下游 | 允许失真换更低速率：[率失真](/information/rate-distortion/) |

> 下一章 [率失真](/information/rate-distortion/)。量子侧噪声模型换掉，见 [量子信息](/quantum/overview/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/information/awgn/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/information/awgn/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Cover & Thomas, 高斯信道一章
2. Shannon (1948) 第 IV 部分
