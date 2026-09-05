---
title: "率失真"
order: 60
---
# 率失真：允许错一点，能少传多少

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [Huffman](/information/source-coding/) 是**无损**：$L\ge H$。若允许重构 $\hat X$ 与 $X$ 差一点，速率还可以再低。率失真函数 $R(D)$ 是失真不超过 $D$ 时的最小互信息（也是最小速率）。最简例子：公平比特、汉明失真 $D=P(\hat X\neq X)$，当 $D\le 1/2$ 时

$$
R(D)=1-h_2(D).
$$

$D=0$ 必须 1 bit（无损）；$D=1/2$ 可以什么都不传（瞎猜也是一半对）。这和 BSC 容量 $C=1-h_2(p)$ **公式长得一样**，对偶：那边 $p$ 是信道噪声，这边 $D$ 是你愿意制造的噪声。

---

## 一、$R(D)$ 不是信道

信道编码：世界已经有噪声，你加冗余去对抗。率失真：你**主动**丢掉细节，换更短的描述。JPEG / 语音编码是这条线的工程后代。

![率失真 R(D)](./images/info-rd.png)

> **图解说明**：横轴失真 $D$，纵轴最少速率。无损在左上角 $R(0)=H$；右端 $D$ 大到无信息时 $R=0$。

---

## 二、代码在做什么

`demo.py` 画 $R(D)=1-h_2(D)$。另外用「按概率 $D$ 随机翻转比特当作重构」估计实际汉明失真（应接近 $D$），并在曲线上标 $D=0.05,0.11,0.25$。这不是最优编码器，只是让 $D$ 有操作定义。

![二元汉明率失真](./images/rd_binary.png)

$p=0.11$ 那条竖线提醒：和 BSC 容量曲线是同一函数，读法相反。

---

## 三、小结

| 概念 | 一句话 |
|------|--------|
| $R(D)$ | 失真 $\le D$ 的最小速率 |
| 二元汉明 | $R(D)=1-h_2(D)$（$D\le 1/2$） |
| 无损 | $D=0\Rightarrow R=H$ |
| 对偶 | 与 BSC 容量同一公式 |
| 下游 | [量子信息](/quantum/overview/)；ML 损失见 [精简章](/math/information/) |

> 信息论七章到此。回到 [导论](/information/overview/) 或进 [量子信息](/quantum/overview/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/information/rate-distortion/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/information/rate-distortion/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Cover & Thomas, Rate Distortion Theory
2. Shannon, “Coding theorems for a discrete source with a fidelity criterion”
