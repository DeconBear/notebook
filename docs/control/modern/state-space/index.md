---
title: "状态空间"
order: 10
---
# 状态空间：植物怎样写成矩阵

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 经典控制把植物压成 $G(s)$。现代控制把内部量摊开成向量 $x$，写成 $\dot x=Ax+Bu$。**能控**时可以用 $u=-Kx$ 把闭环极点放到你指定的位置。LQR 是「让代价最小」的那种 $K$，见 [LQR](/control/modern/lqr/)；测不全 $x$ 时见 [卡尔曼](/control/modern/kalman/)。

---

## 一、状态方程

线性时不变：

$$
\dot x = Ax + Bu,\qquad y=Cx.
$$

$x$ 要长到「知道它就能预测下一步」——双积分器 $\ddot p=u$ 取 $x=[p,\dot p]^\top$：

$$
A=\begin{pmatrix}0&1\\0&0\end{pmatrix},\quad
B=\begin{pmatrix}0\\1\end{pmatrix}.
$$

无控制时位置匀速漂。反馈 $u=-Kx$ 把系统变成 $\dot x=(A-BK)x$。

![极点配置](./images/ctrl-pole-placement.png)

> **图解说明**：左是状态反馈回路；中是把开环极点（积分器堆在原点）搬到希望的闭环极点；右是能控性矩阵满秩才能任意搬。

能控性矩阵（单输入，$n=2$）

$$
\mathcal{C}=\begin{bmatrix}B & AB\end{bmatrix}.
$$

$\operatorname{rank}\mathcal{C}=n$ 才能任意配置极点。demo 的双积分器

$$
\mathcal{C}=\begin{bmatrix}0&1\\1&0\end{bmatrix},
$$

秩为 $2$（列是 $B$ 与 $AB$，不是单位阵，但满秩同样能控）。

---

## 二、二阶 companion 的手算 $K$

$A-BK$ 在这个 $(A,B)$ 上是

$$
\begin{pmatrix}0&1\\-k_1&-k_2\end{pmatrix},
$$

特征多项式 $s^2+k_2 s+k_1$。希望极点 $-2,-3$ 则

$$
(s+2)(s+3)=s^2+5s+6 \implies K=[6,\;5].
$$

更高阶用 Ackermann 公式；本章不调用控制工具箱。

---

## 三、代码在做什么

`demo.py` 打印 $\mathcal{C}$ 的秩，用欧拉积分同一初值 $x=[1,0.8]^\top$：无控制 vs $u=-Kx$。左图位置，右图速度。开环匀速漂走；闭环极点在 $-2,-3$，把位置和速度一起按回原点。

![极点配置后的双积分器](./images/pole_place.png)

开环位置直线漂；闭环把位置和速度一起按回去。终端打印 $A-BK$ 的特征值，应接近 $-2,-3$。

---

## 四、小结

| 概念 | 一句话 |
|------|--------|
| $x$ | 预测下一步所需的内部向量 |
| $A,B$ | 自由演化和「$u$ 往哪推」 |
| 能控 | $\operatorname{rank}[B,AB,\ldots]=n$ |
| $u=-Kx$ | 闭环矩阵 $A-BK$ |
| 下游 | [LQR](/control/modern/lqr/) 用 $Q,R$ 代替「先指定极点」 |

> 下一章 [LQR](/control/modern/lqr/)。经典侧对照 [根轨迹](/control/classical/root-locus/)：那边扫一个标量 $K$，这边一次给出向量 $K$。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/control/modern/state-space/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/control/modern/state-space/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Chen, *Linear System Theory and Design*（能控 / 极点配置）
2. 上一章分组首页 [现代控制](/control/modern/)
