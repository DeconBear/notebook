---
title: "机器人学"
---

# 机器人学：关节角、力矩、闭链，各自用哪套几何

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 侧栏 **机器人学** 不是「工业臂说明书摘要」。先分清你在算哪一类量：**手在哪**（运动学）、**角怎么随时间变**（轨迹）、**电机该使多大劲**（动力学）、**杆件坐标系怎么贴**（DH）、**旋转为什么不能当向量加**（李群）、**闭链还能动几下**（机构）、**转加移合成螺旋**（旋量）。点分组标题进本页。伺服回路见 [控制论](/control/)；真机 TF / 关节话题从 [ROS 2](/ros2/overview/) 进实验室。

![机器人学怎么拆](./images/rob-00-map.png)

> **图解说明**：从导论的构型空间，到运动学与轨迹，再到力与建模，最后是旋转流形、闭链与旋量。底栏接到 PID/LQR 与 ROS 2。

| 入口 | 在问什么 |
|------|----------|
| **[机器人学导论](/robotics/overview/)** | 开链 / 闭链、自由度、工作空间 ≠ 构型空间 |
| **[运动学](/robotics/kinematics/)** | 平面 2R：FK / IK / 雅可比 / 奇异 |
| **[轨迹规划](/robotics/trajectory/)** | 关节插值；θ 直线 ≠ 手画直线 |
| **[动力学](/robotics/dynamics/)** | 拉格朗日；$M\ddot q+H=\tau$ |
| **[DH 建模](/robotics/modeling/)** | 四参数、齐次链、平面 3R |
| **[李群](/robotics/lie-groups/)** | $\mathrm{SO}(3)$、Rodrigues、不能欧拉角乱加 |
| **[机构学](/robotics/mechanisms/)** | Gruebler、四杆、偶联曲线 |
| **[旋量](/robotics/screw/)** | twist / $\mathrm{SE}(2)$ 指数映射 |

建议顺序：导论 → 运动学 → 轨迹 → 动力学 → DH → 李群 → 机构 → 旋量。数学前置：[导数](/math/derivative/)、[线性代数](/math/linear-algebra/)、[特征值](/math/eigen/)。
