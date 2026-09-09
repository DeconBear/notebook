---
title: "线性方程组与秩"
order: 12
---
# 线性方程组与秩：有没有解，解有几个

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 上一章把矩阵看成「捏空间」。这一章问更硬的一句：**给定 $A$ 和 $b$，方程 $Ax=b$ 有没有解？** 几何上是「直线 / 平面交不交」；代数上是「$b$ 能不能写成 $A$ 的各列的线性组合」。学完应能读懂最小二乘为何要写成 $A^\top A\hat x=A^\top b$，以及「秩亏」为什么让训练炸掉。前置：[线性代数直觉](/math/linear-algebra/)。

---

## 一、$Ax=b$ 三种结局

把两元方程看成平面上的直线 $a x + b y = c$：

| 几何 | 代数 | 秩 |
|------|------|----|
| 两线交于一点 | 唯一解 | $\mathrm{rank}(A)=\mathrm{rank}([A\mid b])=n$ |
| 两线平行 | 无解 | $\mathrm{rank}(A) < \mathrm{rank}([A\mid b])$ |
| 两线重合 | 无穷多解 | $\mathrm{rank}(A)=\mathrm{rank}([A\mid b])<n$ |

$n$ 是未知数个数。增广矩阵 $[A\mid b]$ 比 $A$ 多一列 $b$：如果 $b$ 带来了新的独立方向，方程互相打架。

![Ax=b：交点、平行、重合](./images/math-la-systems.png)

> **图解说明**：左是唯一交点；中是平行无交；右把故事抬到三维（三张平面）。底栏那句话是列空间语言：列的线性组合能不能拼出 $b$。

列空间 $\mathrm{Col}(A)$ 就是所有 $Ax$ 能到达的点。**有解 $\Leftrightarrow b\in\mathrm{Col}(A)$。** 秩 $\mathrm{rank}(A)$ 是列空间的维数，也等于非零行数（消元之后）。

**保姆级数字例（与 `demo.py` 三组直线相同）。** 未知数 $n=2$。

1. **唯一解。** $x+y=2$ 与 $x-y=0$。两线斜率 $ -1$ 与 $+1$，交于 $(1,1)$。矩阵
   $$
   A=\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
   \mathrm{rank}(A)=\mathrm{rank}([A\mid b])=2.
   $$
2. **无解。** $x+y=2$ 与 $x+y=3$。平行，永远差 $1$。$\mathrm{rank}(A)=1$，但增广矩阵第二行变成 $[0\ 0\mid 1]$，秩变成 $2$。
3. **无穷多解。** $x+y=2$ 与 $2x+2y=4$。第二行是第一行的两倍，同一条直线。$\mathrm{rank}=1<2$，自由未知数一个：令 $x=t$，则 $y=2-t$。

**卡点。** 「方程个数 = 未知数个数」既不保证有解，也不保证唯一。三张平行平面是三个方程两个……不对，三维里三张平面仍可能无交。真正起决定作用的是秩，不是「几个式子」。另一个卡点：浮点里几乎看不到「精确平行」，两条几乎平行的线会交出一个很远、对扰动极敏感的点——这就是病态。

::: details 逐步推导：三组直线怎样从秩读出唯一解 / 无解 / 无穷解（点击展开）

把直线 $a x+b y=c$ 写成一行 $[a\ b\mid c]$。

**唯一。** 两行 $[1\ 1\mid 2]$ 与 $[1\ -1\mid 0]$。消元：第二行减第一行得 $[0\ -2\mid -2]$，于是 $y=1$，回代 $x=1$。两行都非零且不平行，$\mathrm{rank}=2=n$。

**平行。** $[1\ 1\mid 2]$ 与 $[1\ 1\mid 3]$。消元后第二行 $[0\ 0\mid 1]$——左边全 $0$、右边 $1$，即 $0=1$。$A$ 的两行成比例，秩 $1$；多出来的 $b$ 列让增广秩变成 $2$。列空间语言：$A$ 的两列张成同一条直线（列 $=(1,1)^\top$ 的倍数），$b=(2,3)^\top$ 不在这条线上。

**重合。** $[1\ 1\mid 2]$ 与 $[2\ 2\mid 4]$。第二行 $=2\times$ 第一行，消元后整行变 $0$。$\mathrm{rank}(A)=\mathrm{rank}([A\mid b])=1<2$。解集是一条直线，不是一个点。

一般判据（Rouché–Capelli）：$Ax=b$ 有解 $\Leftrightarrow\mathrm{rank}(A)=\mathrm{rank}([A\mid b])$；有解时，自由未知数个数 $=n-\mathrm{rank}(A)$。

![两条直线的三种结局](./images/systems_lines.png)

> **图解说明**：左交于橙点 $(1,1)$；中平行；右两条线画成一条（虚线叠在实线上）。

:::

---

## 二、高斯消元在干什么

把第 $j$ 列的主元变成 $1$，再用它把同列其他行削成 $0$。部分主元（选绝对值最大的行对调）避免除以接近 $0$ 的数。走完若主元全在，就得到唯一解；某列扫不到主元，秩就掉一档。

三维教科书例子（demo 与 `gauss.hpp` 同一组）：

$$
\begin{pmatrix}
2 & 1 & -1 \\
-3 & -1 & 2 \\
-2 & 1 & 2
\end{pmatrix}
\begin{pmatrix}x\\ y\\ z\end{pmatrix}
=
\begin{pmatrix}8\\ -11\\ -3\end{pmatrix}
\quad\Rightarrow\quad
(x,y,z)=(2,3,-1).
$$

`demo.py` 的 `gauss_solve` 与 `gauss.hpp` 写的是同一套循环。对照 `numpy.linalg.solve` 只为验算，不是再学一个黑盒。

最小二乘是「$b$ 不在列空间里」时的退路：把 $b$ 正交投影到 $\mathrm{Col}(A)$ 上再解。正规方程 $A^\top A\hat x=A^\top b$ 正是在投。线性回归见 [线性回归](/ml/foundations/linear-regression/)。

**卡点。** 主元接近 $0$ 时，这一列几乎已经被前面的列表示了，继续除会把舍入误差放大到爆炸——部分主元能救命，但不能把病态矩阵变好。条件数大的意思是：$b$ 动一点点，$x$ 跑很远。训练里相关特征（两列几乎成比例）就是离散版的这件事。

::: details 逐步推导：$3\times 3$ 消元走到 $(2,3,-1)$，以及正规方程从哪来（点击展开）

增广矩阵：

$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
-3 & -1 & 2 & -11 \\
-2 & 1 & 2 & -3
\end{array}\right].
$$

第一列主元 $2$。$R_2\leftarrow R_2+\frac32 R_1$，$R_3\leftarrow R_3+R_1$：

$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
0 & 1/2 & 1/2 & 1 \\
0 & 2 & 1 & 5
\end{array}\right].
$$

第二列选 $1/2$（若做部分主元会与第三行对调，因为 $|2|>|1/2|$；手算沿用当前顺序，结果相同）。$R_3\leftarrow R_3-4 R_2$：

$$
\left[\begin{array}{ccc|c}
2 & 1 & -1 & 8 \\
0 & 1/2 & 1/2 & 1 \\
0 & 0 & -1 & 1
\end{array}\right].
$$

回代：$-z=1\Rightarrow z=-1$。第二行 $\frac12 y+\frac12(-1)=1\Rightarrow y=3$。第一行 $2x+3-(-1)=8\Rightarrow 2x=4\Rightarrow x=2$。残差 $\|Ax-b\|$ 在精确算术下为 $0$；浮点里应是机器精度量级，demo 会打印这个范数。

**相关矩阵掉一档。** demo 第二张

$$
B=\begin{pmatrix}1&2&3\\2&4&6\\1&1&1\end{pmatrix}
$$

第二行 $=2\times$ 第一行，所以 $\mathrm{rank}(B)=2$（第三行与第一行独立）。`numerical_rank` 数大于阈值的奇异值个数：浮点里「精确相关」几乎不出现，要用阈值。练习就是在数这个个数。

**最小二乘。** 若 $b\notin\mathrm{Col}(A)$，没有 $x$ 使 $Ax=b$。改成最小化 $\|Ax-b\|_2^2$。令 $f(x)=\|Ax-b\|^2=(Ax-b)^\top(Ax-b)$，梯度

$$
\nabla f=2A^\top(Ax-b).
$$

令梯度为 $0$：$A^\top A\hat x=A^\top b$。几何：误差 $b-A\hat x$ 必须垂直于每一列，也就是垂直于 $\mathrm{Col}(A)$——投影的定义。$A^\top A$ 可逆当且仅当 $A$ 列满秩；列相关时正规方程本身秩亏，要改用伪逆 / SVD / 岭回归（对角加 $\lambda I$）。

:::

---

## 三、秩、自由未知数、病态

- 满秩方阵：可逆，唯一解。
- 行线性相关（例如第二行是第一行的两倍）：$\mathrm{rank}<n$，要么无解要么自由变量。
- **数值秩**：奇异值大于阈值的个数。浮点里「精确相关」几乎不出现，要用阈值。练习就是在数这个个数。
- 条件数大：解对 $b$ 的扰动过敏。上一章提过名字，这里会在消元里碰到「主元特别小」。

自由未知数个数 $=n-\mathrm{rank}(A)$（在有解的前提下）。能控性、可观测性矩阵是否满秩，问的也是这件事，见 [状态空间](/control/modern/state-space/)。

---

## 四、代码在做什么

Python 画三组直线，并对手写 $3\times 3$ 消元与 NumPy 对答案。C++ `gauss.hpp` 印同一组解 $(2,3,-1)$，以及一个秩为 $2$ 的相关矩阵 $B$。

![两条直线的三种结局](./images/systems_lines.png)

编译：

```bash
cd docs/math/systems/code
python demo.py
g++ -std=c++17 demo.cpp -o systems_demo
```

Windows 直接跑 `systems_demo.exe`。不要链 Eigen。

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| $Ax=b$ | $b$ 是否在列空间里 |
| 秩 | 独立列（或行）的个数 |
| 高斯消元 | 用主元清列，读出解或秩 |
| 增广矩阵 | 多看 $b$ 这一列，判断有没有打架 |
| 正规方程 | $A^\top A\hat x=A^\top b$：把 $b$ 投影到列空间 |
| 下游 | 最小二乘、可观测性、[特征值](/math/eigen/) |

> 下一章 [特征值与二次型](/math/eigen/)：不解 $Ax=b$，改为找「只被拉伸的方向」。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/math/systems/demo.py" target="_blank" download>Download</a> |
| gauss.hpp | — | <a href="/notebook/code/math/systems/gauss.hpp" target="_blank" download>Download</a> |
| demo.cpp | — | <a href="/notebook/code/math/systems/demo.cpp" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/math/systems/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Strang, G. *Introduction to Linear Algebra*（列空间与消元）
2. Trefethen & Bau, *Numerical Linear Algebra*（主元与条件数）
