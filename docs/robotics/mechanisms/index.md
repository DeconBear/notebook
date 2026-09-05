---
title: "机构学"
order: 50
---
# 机构学：杆件怎么连、有几个自由度

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 前面几章默认「开链手臂」：一端钉死，另一端自由。[机构学](/robotics/mechanisms/) 问的是更老的机械问题：若干杆用铰链连成**闭链**之后，还能动几下？最经典的答案是平面铰链四杆——自由度 $1$，曲柄转一圈，连杆上一点画出封闭曲线。开链几何仍见 [运动学](/robotics/kinematics/)。

---

## 一、Gruebler：先数自由度

平面刚体每个有 $3$ 个自由度（$x,y,\theta$）。$N$ 个构件（含机架）用 $J_1$ 个单自由度低副（铰链或滑块）连起来，常用公式是

$$
M = 3(N-1) - 2 J_1.
$$

机架被减掉的 $3$ 个自由度已经锁死。四杆：$N=4,\;J_1=4$，于是 $M=3\cdot 3-8=1$。有的教材令 $N$ 只计**活动**构件（此时 $N=3$），写成 $M=3N-2J_1$，数字一样。demo 与练习用第一种。

$M=1$ 的含义：指定曲柄角 $\theta$，其余姿态（在装配模式下）被代数约束钉死。这和开链 2R 的 $M=2$ 不同：闭链少了自由，多了环闭合方程。

![四杆与自由度](./images/rob-05-fourbar.png)

> **图解说明**：机架 $AD$，曲柄（蓝）绕 $A$ 整周转，连杆（绿）带偶联点曲线，摇杆（紫）绕 $D$ 摆。四个铰链。自由度 $1$，一个主动输入决定整机。

---

## 二、曲柄摇杆与环闭合

杆长满足 Grashof 条件且最短杆为曲柄时，得到曲柄摇杆：一侧能转圈，一侧只能摆。demo 取机架 / 曲柄 / 连杆 / 摇杆为 $1.4,\;0.45,\;1.1,\;0.9$。

已知曲柄端 $A$，求连杆另一端 $B$，使 $|B-A|=L_2$ 且 $|B-D|=L_3$——两圆交点。一般有两个交点，对应两种装配（开/交叉）。代码取叉积为正的那一支（法向 $n$ 由 $D-A$ 逆时针转 $90^\circ$）。交点不存在就是这一步的死点或杆长不可达。

偶联点可取连杆上任意固连点；demo 用中点 $P=(A+B)/2$，扫 $\theta\in[0,2\pi)$ 得到封闭曲线。

---

## 三、和开链章节的关系

四杆的「位置分析」就是带约束的运动学。若把摇杆拆掉，剩下开链，[正运动学](/robotics/kinematics/) 立刻多一个自由关节。把瞬时速度写成旋量，见 [旋量代数](/robotics/screw/)。真实机器人里平行四杆、差动、五杆并联都是机构学的后代；[ROS 2](/ros2/overview/) 的 URDF 闭链要用 `mimic` 或单独运动学插件，不会自动解环。

---

## 四、代码在做什么

`demo.py` 先打印 Gruebler 应为 $1$，再扫一圈曲柄，画偶联点轨迹，并在 $\theta\approx 0.7$ 处叠一幅连杆快照 `fourbar.png`。

![四杆偶联点曲线](./images/fourbar.png)

---

## 五、小结

| 概念 | 一句话 |
|------|--------|
| Gruebler | 平面 $M=3(N-1)-2J_1$；四杆 $=1$ |
| 曲柄 / 摇杆 / 连杆 | 整周转 / 摆动 / 平面一般运动 |
| 环闭合 | 两圆相交定 $B$ |
| 偶联曲线 | 连杆固连点扫出的轨迹 |
| 下游 | 并联机构、仿生腿、传动设计 |

> 下一章 [旋量](/robotics/screw/)：瞬时运动不再逐点写速度，而写成「绕某轴转并沿轴挪」。李群上的有限转动见 [SO(3)](/robotics/lie-groups/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/robotics/mechanisms/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/robotics/mechanisms/exercise.py" target="_blank" download>Download</a> |

## 参考

1. Norton, *Design of Machinery*（四杆与 Grashof）
2. McCarthy, *Geometric Design of Linkages*
