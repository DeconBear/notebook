---
title: "量子模拟"
order: 50
---
# 量子模拟：让硬件演化哈密顿量

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> Feynman 的原话精神是：用量子系统去模拟量子系统。经典计算机要存 $2^n$ 个振幅；若硬件自己就是那 $n$ 个自旋，演化可以「长在」物理里。

前置：[量子计算](/quantum/computing/) 的酉演化；[科学计算全景](/science/overview/) 里维度灾难的对照。

---

## 一、问题：多体量子力学难在哪

封闭系统的薛定谔方程

$$
i\hbar\frac{d}{dt}|\psi\rangle = H|\psi\rangle
\quad\Rightarrow\quad
|\psi(t)\rangle = e^{-iHt/\hbar}|\psi(0)\rangle
$$

$H$ 是 $2^n\times 2^n$ 的厄米矩阵（$n$ 个自旋）。精确对角化只对很小的 $n$ 可行。蒙特卡洛会碰上符号问题；张量网络擅长一维……**没有一种经典方法通吃所有相互作用图**。

这和 PINN / 算子学习要逼近的 PDE 不是同一句话，但痛点同类：状态空间太大。

![用量子模拟量子](./images/qi-sim-02-nature.png)

> **图解说明**：左边是分子或晶格，右边是可编程的自旋 / 超导 / 离子阵列。模拟成功的标志是能读出能量、关联函数或动力学，而不是「比特数广告」。

---

## 二、模拟 vs 数字模拟

- **模拟型（analog）**：直接让硬件哈密顿量 $\tilde H(t)$ 贴近目标 $H$。调耦合、磁场，少谈「门」。适合特定晶格。
- **数字型（digital）**：把 $e^{-iHt}$ 编译成基本门序列。通用，但深度随精度与时间增长，NISQ 上很快撞墙。

两者之间还有混合：变分量子本征求解器（VQE）用浅线路猜基态，经典优化器调角度——和 [QML](/quantum/qml/) 共享「参数化线路 + 经典环」。

---

## 三、Trotter：把时间切成能编译的薄片

若 $H=A+B$ 且 $[A,B]\neq 0$，则

$$
e^{-i(A+B)t} \approx \big(e^{-iA t/n}e^{-iB t/n}\big)^n
$$

$n$ 越大，一阶公式的误差大致 $O(t^2/n)$。更高阶 Suzuki 公式能再压误差，但门更多。

![Trotter 切时间](./images/qi-sim-01-trotter.png)

`demo.py` 在 **两个自旋** 的横场 Ising

$$
H = J\,Z\otimes Z + h(X\otimes I + I\otimes X)
$$

上对比「精确 `eigh`」与一阶 Trotter。你应看到：步数增加，失真度在对数图上往下掉。

![Trotter 误差](./images/trotter_error.png)

> **图解说明**：这是数字模拟的最小可运行内核。真实分子还要做费米到自旋的映射（Jordan–Wigner 等），本章不展开。

---

## 四、读出什么才算「模拟成功」

常见观测：

- 基态能量（化学精度是另一场战争）；
- 两点关联、结构因子；
- 淬火后的动力学（Loschmidt echo、扩散）。

NISQ 上的噪声会让长时动力学先假掉。因此实验论文会同时报：系统尺寸、演化时间、以及对照的经典方法（张量网络等）还能不能跟上。

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| 指数维 | $n$ 自旋 → $2^n$ 振幅 |
| 模拟型 | 硬件 $H$ 贴近目标 $H$ |
| 数字型 | 编译 $e^{-iHt}$ 为门 |
| Trotter | 把不对易的 $A+B$ 切成小时间步 |
| VQE | 浅线路 + 经典优化找能量 |

> 下一章 [量子机器学习](/quantum/qml/)：把可训练线路接到经典特征上，并收编 VQNet 混合分类实验。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/quantum/simulation/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/quantum/simulation/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Feynman, R. P. *Simulating physics with computers*. IJTP (1982).
2. Lloyd, S. *Universal quantum simulators*. Science (1996).
3. Childs et al., Trotter error 理论综述；Cerezo et al., VQE 综述 (2021).
