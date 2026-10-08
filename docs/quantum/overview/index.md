---
title: "量子信息全景"
order: 10
---
# 量子信息全景：五条专题共用的语言

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 量子信息不是「把电脑换成量子的」一句话，而是一套关于**如何编码、传送、保存、模拟和用量子系统学习**的共同语言。

本领域紧接 **[信息论](/information/)**：经典熵、信道与 [香农容量](/information/shannon/) 说清楚以后，再把比特换成量子比特。线性代数是硬前置，请先有 [向量 / 矩阵 / 内积](/math/linear-algebra/) 的几何直觉。具身与规划见 [控制论](/control/overview/) 与 [世界模型](/world-models/intro/)。

第一遍只抓住：量子比特是归一化复向量；叠加与混合要靠密度矩阵和测量基来区分；不可克隆和退相干是后面每一章的墙。需要把 $H|+\rangle=|0\rangle$ 或不可克隆证一遍时，点开「逐步推导」。

---

## 一、量子信息在问什么？

经典比特是 `{0,1}`。量子比特（qubit）的纯态由二维归一化复向量（态矢量）描述；一般含噪状态则需要密度矩阵。纯态可写为

$$
|\psi\rangle = \alpha|0\rangle + \beta|1\rangle,
\quad
|\alpha|^2 + |\beta|^2 = 1.
$$

测量在计算基上只能得到 0 或 1，概率分别是 $|\alpha|^2$、$|\beta|^2$。**叠加不是「同时是 0 又是 1 的魔法」**，而是：在测量之前，系统由振幅描述；测量之后，坍缩成一个经典结果。

多个量子比特的空间是张量积 $\mathbb{C}^{2}\otimes\mathbb{C}^{2}\otimes\cdots$，维度 $2^n$。这既是算力叙事的来源（状态空间指数大），也是模拟它之所以难的原因。

**保姆级：振幅可以相消，概率不能。** 经典随机是「50% 是 0、50% 是 1」的混合，再做一个 $H$ 门仍然是 50/50。相干叠加 $|+\rangle$ 的相对相位是齐的，同一个 $H$ 能把它收成几乎确定的 $|0\rangle$。demo 就是在画这件事。

---

## 二、五条专题：一张地图

本领域按**你拿量子系统干什么**拆成五章，而不是按公司或芯片名单堆名词：

![量子信息五条专题](./images/qi-01-five-pillars.png)

> **图解说明**：计算研究门与算法；网络把纠缠当成可分发的资源；存储解决光子飞太快、物质相干太短的时间错配；模拟用可控量子系统去跟自然哈密顿量；机器学习把线路嵌进可训练管线。五条路共用量子比特、纠缠和噪声。

| 专题 | 关键问题 | 下一章 |
|------|----------|--------|
| **量子计算** | 门、线路、测量、NISQ vs 容错 | [computing](/quantum/computing/) |
| **量子网络** | 如何把纠缠分发到远处？ | [network](/quantum/network/) |
| **量子存储** | 如何把量子态「按住」一段时间？ | [memory](/quantum/memory/) |
| **量子模拟** | 如何用量子系统模拟量子系统？ | [simulation](/quantum/simulation/) |
| **量子机器学习** | 经典特征如何写进线路并训练？ | [qml](/quantum/qml/) |

阅读顺序建议：全景 → 计算 → 网络 / 存储（可并行）→ 模拟 → 机器学习。QML 章收编了混合量子分类实验（VQNet 核心随该章发布）。

---

## 三、两条贯穿约束

### 3.1 不可克隆

未知量子态不能被可靠地复制成两份相同的未知态（no-cloning）。因此：

- 不能像复制文件那样「备份一个量子比特再测量」；
- 量子密钥分发里，偷听会扰动态，从而留下痕迹；
- 纠错必须绕开「先复制再投票」的经典思路，改用纠缠与稳定子。

::: details 逐步推导：线性复制机为什么不可能（不可克隆）（点击展开）

假设存在一个与未知态无关的酉 $U$，使得对所有 $|\psi\rangle$

$$
U\bigl(|\psi\rangle\otimes|0\rangle\bigr)=|\psi\rangle\otimes|\psi\rangle.
$$

取两个不同的纯态 $|0\rangle$、$|+\rangle$（或任意不正交的一对）。则

$$
U(|0\rangle|0\rangle)=|00\rangle,\qquad U(|+\rangle|0\rangle)=|+\rangle|+\rangle.
$$

也可直接用线性看矛盾：通用复制机还必须满足 $U|10\rangle=|11\rangle$，于是 $U(|+\rangle|0\rangle)=(|00\rangle+|11\rangle)/\sqrt{2}$；但所需输出 $|+\rangle|+\rangle=(|00\rangle+|01\rangle+|10\rangle+|11\rangle)/2$，两者不同。用内积保持可以得到更简短的证明。

酉保持内积：$\langle 0|+\rangle=\langle 00|++\rangle$。左边是 $1/\sqrt{2}$，右边是 $(1/\sqrt{2})^2=1/2$，矛盾。因此这样的 $U$ 不存在。

推论：未知态不能先复印再测量两份取平均；传态是「搬走」不是「复制」；QKD 里 Eve 的拦截-重发过不了完美复印这一关。正交态（已知的计算基）可以复制——那已经是经典比特。

:::

### 3.2 退相干

真实系统会与环境纠缠，相对相位被冲刷。常用两个时间尺度：

- $T_1$：能量弛豫（激发态掉回基态）；
- $T_2$：失相（布洛赫球赤道上的相干先没）。

计算深度、网络距离、存储时间、模拟时长，最后都撞上这两条钟。

![不可克隆与退相干](./images/qi-02-no-cloning.png)

> **图解说明**：左边是禁止的复印机；右边是布洛赫矢量被噪声往球心拽。后面每一章都会回到这两张图。

---

## 四、叠加 vs 混合：demo 在画什么

测量「0 和 1 各一半」有两种完全不同的来源：

- **相干叠加** $|+\rangle=(|0\rangle+|1\rangle)/\sqrt{2}$：有相对相位，再用 $H$ 可以几乎确定地变回 $|0\rangle$；
- **完全混合** $\rho=I/2$：在任何正交测量基下都是 50/50，做 $H$ 也不改变它。混合可来自制备信息未知或与环境纠缠；纯叠加态的测量结果也可以是真正随机的。

计算基下的两个密度矩阵是

$
\rho_+=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix},
\qquad
\rho_{\mathrm{mix}}=\frac12\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$

它们的对角相同，所以测 $Z$ 得到相同概率；非对角不同，所以测 $X$ 能区分统计分布。纯度分别是 $\mathrm{Tr}(\rho_+^2)=1$ 与 $\mathrm{Tr}(\rho_{\mathrm{mix}}^2)=1/2$。注意“混合态没有相干”不是一般结论：$\rho=(3/4)|+\rangle\langle+|+(1/4)|-\rangle\langle-|$ 是混合态，但在计算基下有非零非对角元 $1/4$。相干依赖所选基，纯度则不依赖基。

`demo.py` 用两次测量把这件事画出来。每次 $2000$ 次抽样（`shots`），种子 `42`。纯度 $\mathrm{Tr}(\rho^2)$ 是配套练习：纯态为 1，单比特完全混合为 $1/2$。

![叠加与混合的测量对比](./images/superposition_vs_mixture.png)

> **图像待更新**：绘图脚本已纠正“只有混合才是真随机”的标题表述；现有图片尚未重新生成。请以本节说明为准，并运行 `code/demo.py` 更新标题。

::: details 逐步推导：为什么 $H|+\rangle=|0\rangle$，而 $H$ 对 $I/2$ 毫无办法（点击展开）

Hadamard（与 demo 矩阵一致）

$$
H=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad
H|0\rangle=|+\rangle,\quad H|+\rangle=|0\rangle.
$$

第二式：把 $|+\rangle=(|0\rangle+|1\rangle)/\sqrt{2}$ 乘进去，$H|1\rangle=|-\rangle$，于是 $H|+\rangle=(|+\rangle+|-\rangle)/\sqrt{2}=|0\rangle$。密度矩阵 $\rho_+=|+\rangle\langle +|$ 经 $H\rho H^\dagger$ 变成 $|0\rangle\langle 0|$，测 $Z$ 几乎全是 $0$。

完全混合 $\rho=I/2$。$H(I/2)H^\dagger=I/2$（$H$ 酉），测什么基都是 $1/2$。直方图左栏两种情况都像抛硬币；右栏只有叠加被「收回」。这说明相干叠加与完全混合的区别能通过改变测量基显现，不能用「只有混合才随机」来区分。

不可克隆禁止的是用同一装置完美复制任意未知态；已知的 $|+\rangle$ 可以重复制备。在独立副本上分别测 $X$ 和 $Z$ 能估计各自的统计分布，但不能给同一个量子系统赋予两个不相容可观测量的同时确定值。

:::

---

## 五、和本笔记本其他部分的接口

- **数学**：态矢量、酉门、测量投影，全是线代；变分量子线路的训练还用得到 [梯度](/math/optimization/) 与 [KL / 交叉熵](/math/information/)。经典容量对照见 [信息论](/information/shannon/)。
- **深度学习**：QML 的经典压缩器就是普通网络；对照 [CNN](/applied/cv/cnn/)。
- **科学计算**：量子模拟是 AI4S 的「另一条轴」——不一定用神经网络逼近 PDE，而是让硬件自己演化哈密顿量。

---

## 六、小结

| 概念 | 一句话 |
|------|--------|
| 量子比特 | 归一化复向量；测量给出经典比特 |
| 张量积 | $n$ 比特空间维度 $2^n$ |
| 叠加 vs 混合 | 单一基的直方图可能相同；需换基检查相干与纯度 |
| 不可克隆 | 未知态不能完美复印 |
| 退相干 | $T_1$ 掉能量，$T_2$ 掉相位 |
| 五条专题 | 计算 / 网络 / 存储 / 模拟 / 学习 |

> 下一章 [量子计算](/quantum/computing/)：把 $|0\rangle$、$H$、CNOT 和测量连成一条能跑的线路。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/quantum/overview/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/quantum/overview/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Nielsen, M. A. & Chuang, I. L. *Quantum Computation and Quantum Information*.
2. Wilde, M. M. *Quantum Information Theory*.
3. Preskill, J. *Quantum Computing in the NISQ era and beyond*. *Quantum* (2018).
