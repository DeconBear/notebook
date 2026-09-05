---
title: "旋量代数"
order: 60
---
# 旋量代数：刚体怎样「拧」过去

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> Chasles 定理：刚体任意有限位移等价于绕某轴旋转并沿该轴平移——一条**螺旋**。[李群 $\mathrm{SO}(3)$](/robotics/lie-groups/) 只覆盖转动；加上平移后是 $\mathrm{SE}(3)$，李代数 $\mathfrak{se}(3)$ 里的元素叫 **twist（捻）**。对偶的力和力矩叫 **wrench（力螺）**。本章图解给 6 维全貌；**demo 降到平面 $\mathrm{SE}(2)$**，用指数映射让一块小矩形绕固定瞬心转过去。

---

## 一、捻与力螺

空间刚体瞬时运动用六元组

$$
\xi=\begin{pmatrix}\omega\\ v\end{pmatrix}\in\mathbb{R}^6,
$$

$\omega$ 是角速度方向（转轴），$v$ 是与轴相关的线速度（过原点的参考）。轴上一点 $q$ 处

$$
v_{\text{点}}=\omega\times q + h\omega,
$$

螺距 $h$ 描述「转一弧度沿轴走多远」。$h=0$ 是纯转动，$h=\infty$ 极限是纯平移。

力螺 $F=(f,\tau)$ 与捻通过瞬时功率配对：

$$
P=\omega\cdot\tau + v\cdot f.
$$

![旋量：捻与力螺](./images/rob-06-twist.png)

> **图解说明**：刚体上一条螺旋轴；$\omega$ 沿轴，$v$ 含沿轴平移与因转动引起的线速度。右栏捻 $\xi$、力螺 $F$ 与功率。平面 demo 只保留 $\omega$ 的一个标量和 $v_x,v_y$。

这套语言把 [运动学](/robotics/kinematics/) 的雅可比列写成「各关节旋量」，把 [动力学](/robotics/dynamics/) 的力写成力螺；现代教材（Murray–Li–Sastry）用它统一开链。

---

## 二、平面 $\mathfrak{se}(2)$ 指数映射

平面刚体 $3$ 自由度：转角 + 平移。李代数元素 $\xi=(\omega,v_x,v_y)$。$\omega=0$ 时 $\exp$ 就是沿 $(v_x,v_y)$ 平移。$\omega\neq 0$ 时，转角 $\omega t$，平移由

$$
\begin{pmatrix}x\\ y\end{pmatrix}
=
\frac{1}{\omega}
\begin{pmatrix}
\sin(\omega t) & -(1-\cos(\omega t))\\
1-\cos(\omega t) & \sin(\omega t)
\end{pmatrix}
\begin{pmatrix}v_x\\ v_y\end{pmatrix}
$$

给出（与 `se2_exp` 一致）。齐次矩阵 $T(t)=\exp(t\hat\xi)$ 作用在点的齐次坐标上。

纯转动：若瞬心在 $q$，线速度应满足「绕 $q$ 转」。demo 采用

$$
v=\omega(q_y,-q_x)
$$

（与三维 $\omega\times r$ 差一个平面定向约定，练习里会钉死）。于是矩形会绕黑叉瞬心转，而不是绕原点转。

---

## 三、和 DH、四杆的关系

DH 给的是**有限**相邻位姿；旋量给的是**瞬时**生成元。指数映射把「关节速度 $\dot\theta$ 乘关节旋量」积成有限 $T(\theta)$——这就是指数坐标 / 指数积公式（PoE）的平面缩影。四杆机构每根杆的瞬时运动也是某个 twist；闭链意味着若干旋量的线性相关（自由度与 Gruebler 一致）。

[世界模型](/world-models/intro/) 若要在 SE(2)/SE(3) 上预测下一帧位姿，增量也应走 $\exp$，而不是把 yaw 当普通实数加到 $2\pi$ 外面。

---

## 四、代码在做什么

`demo.py` 指定 $\omega=1.2$、瞬心 $q=(0.8,0.2)$，构造纯转动 $v$，对一块矩形在 $t=0\ldots 1.6$ 取 $7$ 帧，颜色沿 viridis 变化，得到 `screw_se2.png`。

![平面螺旋运动](./images/screw_se2.png)

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| 螺旋 / Chasles | 刚体位移 = 绕轴转 + 沿轴移 |
| twist $\xi$ | 瞬时运动；$(\omega,v)$ |
| wrench | 力与力矩；与 twist 功率对偶 |
| $\mathrm{SE}(2)$ $\exp$ | 平面刚体有限位移 |
| 下游 | PoE 正运动学、力控制、位姿积分 |

> 机器人六章到此收束。建议回到 [运动学](/robotics/kinematics/) 把 2R 再走一遍，或用 [ROS 2](/ros2/overview/) 看真机 TF。控制侧从 [经典 PID](/control/classical/pid/) 接到关节伺服。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/screw/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/screw/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Murray, Li, Sastry, *A Mathematical Introduction to Robotic Manipulation*（旋量与指数积）
2. Lynch & Park, *Modern Robotics*（公开课与教材）
