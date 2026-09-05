---
title: "LQR 与最优控制"
order: 20
---
# LQR：状态反馈怎么一次给完增益

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [PID](/control/classical/pid/) 看的是一个误差标量。[状态空间](/control/modern/state-space/) 已经把植物写成 $\dot x=Ax+Bu$。本章只讲：**LQR 怎样用 $Q,R$ 一次给出 $u=-Kx$**。demo 仍把卡尔曼串在同一条双积分器上（只测带噪位置），因为分离原理要同时看见「增益」和「估计」；滤波器本身见 [卡尔曼](/control/modern/kalman/)。倒立摆是图解里的直觉对象。

---

## 一、状态空间

线性时不变植物：

$$
\dot x = Ax + Bu,\qquad y = Cx.
$$

- $x$：$n$ 维内部状态（倒立摆常取 $[p,\dot p,\theta,\dot\theta]^\top$）；
- $u$：控制；
- $y$：能测到的量，往往比 $x$ 短——比如只测位置，速度得估。

![状态空间与 LQR](./images/ctrl-03-state-space.png)

> **图解说明**：上半是 $\dot x=Ax+Bu$、$y=Cx$ 的方块；中段用倒立摆说明「状态可以比输出更长」；下段天平是 LQR：$Q$ 罚状态偏离，$R$ 罚控制能量。

离散时间（demo 用这个，因为电脑按步走）把 $A,B$ 换成一步转移：

$$
x_{k+1} = A x_k + B u_k.
$$

双积分器 $\ddot x=u$ 在步长 $\Delta t$ 下的精确离散是

$$
A=\begin{pmatrix}1&\Delta t\\ 0&1\end{pmatrix},\quad
B=\begin{pmatrix}\tfrac12\Delta t^2\\ \Delta t\end{pmatrix}.
$$

无控制时位置匀速漂走——这就是「为什么需要反馈」的最小例子。机械臂要把这套接到关节，见 [运动学](/robotics/kinematics/) 与 [动力学](/robotics/dynamics/)。

---

## 二、LQR：一次求出增益

无限时域二次代价（连续写法便于记；实现是离散 Riccati）

$$
J=\int_0^\infty \big(x^\top Q x + u^\top R u\big)\,\mathrm{d}t.
$$

最优反馈在线性植物上是 **$u=-Kx$**。$K$ 来自代数 Riccati 的解 $P$：离散情形

$$
K=(R+B^\top P B)^{-1} B^\top P A.
$$

$Q$ 大 → 更在乎把 $x$ 按回去；$R$ 大 → 更吝啬 $u$。没有积分项，也没有人手调 $K_p$：折中被两块权重矩阵写死。前提是 **你得有 $x$**（或一个够好的 $\hat x$）。

---

## 三、卡尔曼：预测再修正

传感器只给 $z=Hx+v$。双积分器上 $H=[1,\;0]$：只测位置。速度藏在状态里，要用滤波器估。

![卡尔曼：预测与更新](./images/ctrl-04-kalman.png)

> **图解说明**：蓝框按模型走一步，$\hat x^-=A\hat x+Bu$，协方差变大；绿框用新息 $z-H\hat x^-$ 和卡尔曼增益 $K_g$ 把估计拉向测量，椭圆变小。$K_g$ 决定信模型还是信测量。

循环是：

1. **预测** $\hat x\leftarrow A\hat x+Bu$，$P\leftarrow APA^\top+Q_n$；
2. **更新** $K_g=PH^\top(HPH^\top+R_n)^{-1}$，再用残差修正 $\hat x$ 与 $P$。

这和 [世界模型](/world-models/intro/) 里「先验动力学 + 用观测纠正隐状态」是同一句人话；RSSM / 卡尔曼只是线性高斯时有闭式。量子侧把「测完如何更新信念」换成另一套语言，见 [量子信息](/quantum/overview/)。

---

## 四、代码在做什么

`demo.py` 三条轨迹同一套离散双积分器：无控制、全状态 LQR（直接用真 $x$）、以及「控制律看 $\hat x$，卡尔曼只吃带噪位置」。左图位置，右图加速度指令；虚线是滤波器对位置的估计。

![LQR 与卡尔曼把双积分器拉回原点](./images/lqr_kalman.png)

无控制会漂；全状态 LQR 最快按回原点；带滤波的 LQR 稍吵、稍慢，但不再需要测速度。增益 $K$ 由迭代离散 Riccati 算出，不调用外部控制工具箱。

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| 状态 $x$ | 预测下一步所需的内部向量 |
| LQR | 二次代价下的 $u=-Kx$；$Q$/$R$ 折中 |
| 卡尔曼 | 预测放大不确定，测量再收缩 |
| 分离原理（直觉） | 线性高斯时，可先滤波再套 LQR |
| 下游 | 机器人、飞行器；学出来的动力学见世界模型 |

> 几何与力矩接到手臂上：[运动学](/robotics/kinematics/)、[动力学](/robotics/dynamics/)。在想象里滚动未来：[世界模型导论](/world-models/intro/)。真机话题与坐标系习惯见 [ROS 2](/ros2/overview/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/control/modern/lqr/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/control/modern/lqr/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Kalman, “A New Approach to Linear Filtering and Prediction Problems” (1960)
2. Anderson & Moore, *Optimal Control*（LQR / Riccati）
3. [PID](/control/classical/pid/)（时域直觉）、[状态空间](/control/modern/state-space/)（$A,B$）
