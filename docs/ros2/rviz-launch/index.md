---
title: "12 RViz 与组合 Launch"
order: 130
---
# 12 · RViz 与组合 Launch

## 目标

用一个 launch 同时拉起 **Gazebo 仿真 + RViz**，形成最小“仿真栈”。

## 概念

真实项目很少手开十几个终端。第 05 课的 Launch 在这里升级为**组合栈**：

```text
sim_stack.launch.py
  ├─ include gazebo.launch.py   (仿真 + bridge)
  └─ rviz2 -d sim.rviz          (看 odom / TF)
```

RViz 负责**可视化**；Gazebo 负责**物理仿真**。两者常一起用。

## 通俗理解

到这里，前面学的东西开始**拼成一台能看的小系统**：

| 你已学的 | 在本课栈里的角色 |
|----------|------------------|
| Topic / cmd_vel | 开车指令 |
| Launch | 一条命令拉起仿真+桥+RViz |
| TF / odom | RViz 里看车在哪 |
| Gazebo | 真正让车在虚拟世界里动 |

`sim_stack.launch.py` 只是把第 11 课的仿真 launch **include** 进来，再加一个 RViz——这就是真实项目里“组合栈”的缩影。

## 常见疑问

**Q：为什么还要 RViz？Gazebo 里不是已经能看见车了吗？**  
Gazebo 看的是仿真场景；RViz 看的是 ROS 侧数据（里程计、TF、以后还有激光/路径）。调试导航时 RViz 更重要。

**Q：Fixed Frame 为什么常用 odom？**  
仿真里车相对里程计坐标系运动；选 `odom` 才能正确显示轨迹/里程计。URDF 课用 `base_link` 是因为那时还没跑起来。

**Q：这算学完 ROS 了吗？**  
算走完入门主线（通信 → 接口 → 空间运动 → 仿真可视化）。后面还有 Nav2 / MoveIt / 硬件驱动等支线。

**Q：下一步更适合我（偏硬件）学什么？**  
可考虑 `ros2_control`、串口桥、micro-ROS；或先用 Nav2 把移动栈跑通再回头接真机。

## 动手

```bash
source /opt/ros/humble/setup.bash
source $ROS_WS/install/setup.bash
ros2 launch simple_robot sim_stack.launch.py
```

另开终端控车：

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.15}}"
```

在 RViz 中 Fixed Frame 选 `odom`，应能看到里程计相关显示（配置见 `sim.rviz`）。

## 关键命令

```bash
ros2 launch simple_robot sim_stack.launch.py
ros2 launch simple_robot display.launch.py    # 仅 URDF（第 10 课）
ros2 launch simple_robot gazebo.launch.py     # 仅仿真（第 11 课）
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`launch/sim_stack.launch.py`](../../../workspaces/ros2-humble/src/simple_robot/launch/sim_stack.launch.py) | 组合栈 |
| [`rviz/sim.rviz`](../../../workspaces/ros2-humble/src/simple_robot/rviz/sim.rviz) | 仿真可视化 |

## 小练习

1. 读 `sim_stack.launch.py`，试着再 `Include` 第 09 课的 `cmd_vel_publisher`（可选作业）。
2. 回顾 00–12：哪些是通信、哪些是描述、哪些是仿真。

## 验证标准（给学员与 AI）

- [ ] （有桌面）`ros2 launch simple_robot sim_stack.launch.py` 同时起仿真与 RViz
- [ ] 能说清 Gazebo vs RViz 的分工
- [ ] 知道 00–12 主线已结束，后续为可选支线

失败时：可分别验证 gazebo launch 与 RViz；headless 以概念理解为主。

## 小结（A→C 收束）

你已具备：

- 自定义接口（msg/srv）
- TF / cmd_vel / URDF
- Gazebo + bridge + RViz 组合启动

之后可选支线：**Nav2**、**MoveIt**、或 **ros2_control / micro-ROS（硬件向）**。

目录：[README](/ros2/overview/)
