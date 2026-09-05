---
title: "机器人运动学"
order: 10
---
# 机器人运动学：关节角怎样变成末端坐标

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 运动学不问力，只问几何：**给定关节角，手在哪；给定手要去的点，关节该怎么弯**。本章用平面 2R 手臂把正运动学、肘上/肘下逆解、雅可比奇异一次讲完。力与加速度见 [动力学](/robotics/dynamics/)；把真实串联臂写成矩阵链见 [DH 建模](/robotics/modeling/)。

---

## 一、机器人学六章怎么排

本领域按「你要算哪一类量」拆成六章，而不是按品牌堆手册：

| 章 | 在问什么 |
|----|----------|
| **[运动学](/robotics/kinematics/)**（本页） | 关节角 $\leftrightarrow$ 末端位姿；速度用雅可比 |
| **[动力学](/robotics/dynamics/)** | 力矩 $\leftrightarrow$ 加速度；拉格朗日 $M(q)\ddot q+H=\tau$ |
| **[建模方法](/robotics/modeling/)** | DH 参数把杆件收成齐次变换连乘 |
| **[李群](/robotics/lie-groups/)** | 旋转不是普通向量；$\mathrm{SO}(3)$ 与 $\mathfrak{so}(3)$ |
| **[机构学](/robotics/mechanisms/)** | 杆怎么连、Gruebler 自由度、四杆曲线 |
| **[旋量](/robotics/screw/)** | 刚体瞬时运动：转轴 + 沿轴平移 |

控制回路（PID / LQR）见 [PID](/control/classical/pid/) 与 [LQR](/control/modern/lqr/)。在想象里预演碰撞见 [世界模型](/world-models/intro/)。真机坐标系、TF、关节状态话题从 [ROS 2](/ros2/overview/) 进实验室。

---

## 二、正运动学：角 → 点

平面两根杆，长度 $\ell_1,\ell_2$，关节角 $\theta_1$（相对基座 $x$ 轴）、$\theta_2$（相对第一杆延长线）：

$$
\begin{aligned}
x &= \ell_1\cos\theta_1 + \ell_2\cos(\theta_1+\theta_2),\\
y &= \ell_1\sin\theta_1 + \ell_2\sin(\theta_1+\theta_2).
\end{aligned}
$$

可达区域是圆环：$|\ell_1-\ell_2|\le r\le \ell_1+\ell_2$，其中 $r=\sqrt{x^2+y^2}$。

![平面 2R 正 / 逆运动学](./images/rob-01-2r-fk.png)

> **图解说明**：左图由 $\theta_1,\theta_2$ 推出末端 $(x,y)$，虚线圆是最大/最小伸长；右图同一目标有肘上、肘下两套角。余弦定理先解 $\theta_2$，再用 $\operatorname{atan2}$ 解 $\theta_1$。

正运动学永远唯一；逆运动学在平面 2R 上通常是两个解，边界上合成一个，圈外无解。demo 里 $\ell_1=1$、$\ell_2=0.7$。

---

## 三、逆解：肘上与肘下

余弦定理：

$$
\cos\theta_2=\frac{r^2-\ell_1^2-\ell_2^2}{2\ell_1\ell_2}.
$$

$\sin\theta_2$ 取正或取负，就是肘上 / 肘下。$\theta_1$ 要把「指向目标的方位」扣掉第二杆造成的偏角。代码里用 `arctan2` 而不是 `arccos`，避免象限丢符号。

工业臂还要在连续轨迹上**选哪一支解**（别突然从肘上跳到肘下）。那是路径规划问题；本章只把两支都画出来。

---

## 四、雅可比：关节速度 → 末端速度

$$
\begin{pmatrix}v_x\\ v_y\end{pmatrix}
=J(\theta)
\begin{pmatrix}\dot\theta_1\\ \dot\theta_2\end{pmatrix}.
$$

$J$ 随构型变。两杆共线（伸直 $\theta_2=0$ 或折叠 $\theta_2=\pm\pi$）时 $\det J=0$：末端瞬时少一个可动方向，速度逆解炸。对 2R 有闭式 $\det J=\ell_1\ell_2\sin\theta_2$。

![雅可比与奇异](./images/rob-01b-jacobian.png)

> **图解说明**：上半 $v=J\dot\theta$；下半两杆共线时 $\det J=0$，工作空间内外边界正好是这些奇异构型。奇异附近不要硬求 $J^{-1}$。

---

## 五、代码在做什么

`demo.py` 对目标 $(1.1,0.6)$ 求两支 IK，再 FK 回去核验，并打印 $\det J$。左图工作空间采样 + 两种肘形；右图固定 $\theta_1=0.4$，扫 $\theta_2$ 看行列式过零。

![2R 肘形与奇异](./images/arm_2r.png)

---

## 六、小结

| 概念 | 一句话 |
|------|--------|
| FK | 角 → 末端；唯一 |
| IK | 点 → 角；2R 常两解 |
| 工作空间 | 圆环带，圈外无解 |
| 雅可比 | 速度映射；共线时奇异 |
| 下游 | 动力学、DH、李群、旋量、ROS 2 |

> 下一章 [动力学](/robotics/dynamics/)：没有力矩，手臂在重力下会自己摆。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/kinematics/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/kinematics/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Craig, *Introduction to Robotics*（正逆解与雅可比）
2. Siciliano et al., *Robotics: Modelling, Planning and Control*
