---
title: "机器人建模方法"
order: 30
---
# 机器人建模方法：DH 把杆件收成矩阵链

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [2R 三角公式](/robotics/kinematics/) 写两根杆刚刚好。六轴工业臂再手写 $\cos(\theta_1+\theta_2+\cdots)$ 会疯。**Denavit–Hartenberg（DH）** 给每一关节四个几何数，齐次矩阵连乘就得到基座到末端。本章只讲标准 DH 的四个参数、一帧 $A_i$、以及平面 3R（$\alpha=d=0$）这条最简单的链。旋转矩阵为什么必须 $\det=1$ 见 [李群](/robotics/lie-groups/)。

---

## 一、为什么要统一坐标系

串联臂是一串刚体。相邻两杆之间只要说清：**沿什么轴转/移多少、两轴之间扭了多少**。DH 的约定是：每个关节配一个坐标系，相邻系之间恰好四个量 $(a_i,\alpha_i,d_i,\theta_i)$。换一套约定（modified DH）数字会变，但「四个标量 + 齐次阵」这件事不变。本章与 demo 用经典顺序：

> 绕 $z$ 转 $\theta$ → 沿 $z$ 移 $d$ → 沿 $x$ 移 $a$ → 绕 $x$ 转 $\alpha$。

![DH 参数](./images/rob-03-dh.png)

> **图解说明**：相邻系 $\{i-1\}$ 与 $\{i\}$ 之间，$a$ 是沿公垂线的杆长，$\alpha$ 是绕该公垂线的扭角，$d$ 是沿 $z$ 的偏置，$\theta$ 是绕 $z$ 的关节角。正方向右手定则。图中矩阵若与教材另一约定不一致，**以代码里的乘法顺序为准**。

---

## 二、一帧齐次变换

标准 DH 对应的 $4\times 4$ 矩阵（与 `dh()` 一致）为

$$
A_i=
\begin{pmatrix}
c_\theta & -s_\theta c_\alpha & s_\theta s_\alpha & a c_\theta \\
s_\theta & c_\theta c_\alpha & -c_\theta s_\alpha & a s_\theta \\
0 & s_\alpha & c_\alpha & d \\
0 & 0 & 0 & 1
\end{pmatrix}.
$$

其中 $c_\theta=\cos\theta$ 等。平面转动关节：$\alpha=0,\;d=0,\;a=$ 杆长，$\theta$ 是唯一变量。此时 $A_i$ 退化成「转 $\theta$ 再沿杆走 $a$」，第三行第三列为 $1$。

整条链：

$$
T_n^{0} = A_1 A_2 \cdots A_n.
$$

左上 $3\times 3$ 是末端姿态，右上 $3\times 1$ 是末端位置。这就是可扩展的正运动学：[运动学](/robotics/kinematics/) 那两行余弦，只是 $n=2$ 时乘开的结果。

---

## 三、平面 3R 在做什么

三根杆 $a=(0.6,0.5,0.35)$，关节角给定后，把每个 $A_i$ 累乘，记下每一帧原点，就得到折线。没有逆解、没有雅可比——本章只证明「DH 链能画出一只手臂」。空间臂 $\alpha\neq 0$ 时 $z$ 轴不再全平行，矩阵第三行才真正用上。

[ROS 2](/ros2/overview/) 的 URDF 用的是另一套「xyz + rpy」而不是 DH 表，但正运动学仍然是齐次变换连乘；读 URDF 时可把每条 `<joint>` 想成一帧 $A_i$。

---

## 四、代码在做什么

`demo.py` 对 $\theta=(0.4,-0.7,0.5)$ 调用 `fk_chain`，打印末端 $xy$ 与旋转块 $R$，并画出三杆折线 `dh_3r.png`。

![DH 平面 3R](./images/dh_3r.png)

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| DH 四元组 | $(a,\alpha,d,\theta)$ 描述相邻系 |
| $A_i$ | 一帧 $4\times 4$；平面臂大量零 |
| 正运动学 | $T=A_1\cdots A_n$ |
| 与 2R 公式 | 三角是 $n=2$ 的展开 |
| 下游 | 标定、URDF、李群上的姿态 |

> 下一章 [李群](/robotics/lie-groups/)：$R$ 不能当 $\mathbb{R}^9$ 随便加，指数映射才是「小转动」的正确加法。机构怎么连成闭链见 [机构学](/robotics/mechanisms/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/modeling/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/modeling/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Denavit & Hartenberg, “A Kinematic Notation…” (1955)
2. Craig, *Introduction to Robotics*（标准 DH 表）
