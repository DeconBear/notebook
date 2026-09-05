---
title: "机器人动力学"
order: 20
---
# 机器人动力学：力矩怎样变成加速度

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> [运动学](/robotics/kinematics/) 告诉你手在哪。动力学告诉你：**要让关节按某个 $\ddot q$ 动，电机该出力矩 $\tau$ 多少**——以及反过来，零力矩时重力会把臂往哪拽。本章用平面 2R、质量集中在杆末端，走拉格朗日，不引入牛顿-欧拉递推。

---

## 一、能量路线 vs 逐杆受力

牛顿定律要在每根杆上画力、铰链反力，再消去约束。杆一多就烦。拉格朗日只写标量：

$$
L = T - V,\qquad
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial \dot q_i}-\frac{\partial L}{\partial q_i}=\tau_i.
$$

$T$ 动能，$V$ 势能，$q$ 广义坐标（这里就是 $\theta_1,\theta_2$）。约束已经被坐标选法吃掉。图解里杆有质心与转动惯量；**demo 把质量放在每根杆末端**，公式更短，结构不变。

![拉格朗日 2R](./images/rob-02-lagrange.png)

> **图解说明**：两关节 $\theta_1,\theta_2$ 与力矩 $\tau$；动能 $T$、势能 $V$、拉格朗日量 $L=T-V$；欧拉-拉格朗日方程按每个广义坐标写一行。重力竖直向下。

约定：demo 里 $\theta=0$ 是**水平**（$\cos\theta=1$ 时重力力矩最大）。这和「竖直悬挂为 0」的画法不同，读公式时先看符号。

---

## 二、标准形：$M(q)\ddot q + H(q,\dot q)=\tau$

整理之后一定能写成

$$
M(q)\,\ddot q + H(q,\dot q) = \tau.
$$

- $M(q)$：**质量矩阵**，对称正定。平面 2R 里 $M$ 只依赖 $\theta_2$（绕基座转一圈，惯性长得一样）。
- $H$：科氏 / 离心（速度二次项）+ 重力。速度为 0 时 $H$ 只剩重力。

正动力学：已知 $\tau$，解 $\ddot q=M^{-1}(\tau-H)$。逆动力学：已知运动，算 $\tau=M\ddot q+H$（控制里更常用）。[现代控制](/control/modern/) 的 $x$ 若包含 $q,\dot q$，线性化后的 $A,B$ 就藏在这个非线性式的切线里。

阻尼在 demo 里额外写成 $-c\dot q$ 加进右端，用来对比「保守系统永远摆」和「有摩擦停下来」。

---

## 三、无主动力矩：自由落体

$\tau=0$ 时手臂是双摆。无阻尼则能量在动能与势能间倒腾，关节角一直晃；有阻尼则末端轨迹螺旋靠近低势能姿态。这不是 bug：拉格朗日在保守力下保能量，必须额外加耗散项才会停。

和 [经典 PID](/control/classical/pid/) 对照：PID 是在植物外面用误差造 $\tau$；植物本身仍是这一套 $M,H$。

---

## 四、代码在做什么

`demo.py` 从初值 $(\theta_1,\theta_2)=(0.3,0.9)$、静止出发，分别积无阻尼与阻尼 $c=1.6$。左图关节角对时间；右图有阻尼时末端 $(x,y)$ 轨迹。终端用末段差分估计残余角速度。

![2R 拉格朗日自由落体](./images/lagrange_2r.png)

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| 拉格朗日 | 用 $T-V$ 推导，不必画铰链内力 |
| $M(q)$ | 惯性；2R 中只靠 $\theta_2$ |
| $H$ | 科氏 / 离心 + 重力 |
| 正 / 逆动力学 | 已知 $\tau$ 求 $\ddot q$ / 已知运动求 $\tau$ |
| 下游 | 计算力矩控制、辨识、仿真 |

> 下一章 [DH 建模](/robotics/modeling/)：把「两根杆的三角公式」换成可往 6 轴扩的齐次链。旋转本身的流形见 [李群](/robotics/lie-groups/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/dynamics/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/dynamics/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Spong, Hutchinson, Vidyasagar, *Robot Modeling and Control*
2. 上一章 [运动学](/robotics/kinematics/)（FK 与本章 `fk` 同一套角）
