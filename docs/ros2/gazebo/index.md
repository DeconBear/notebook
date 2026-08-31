---
title: "11 Gazebo 最小仿真"
order: 120
---
# 11 · Gazebo 最小仿真

## 目标

在 Gazebo Fortress（Ignition Gazebo 6）中加载差速小车，通过 `ros_gz_bridge` 把 ROS `/cmd_vel` 接到仿真。

## 概念

```text
ROS 2 /cmd_vel  --bridge-->  Gazebo DiffDrive 插件  -->  车轮转动
ROS 2 /odom     <--bridge--  仿真里程计
```

本机 Humble 对应 **Gazebo Fortress**。启动入口常用 `ros_gz_sim`。

模型在 `simple_robot/models/simple_diff_robot/`，世界在 `worlds/simple_robot.sdf`。

## 通俗理解

Gazebo 是**带物理的虚拟世界**：重力、碰撞、轮子转动。RViz 只是“看数据的显示器”，不会替你算物理。

ROS 和 Gazebo 本来是两套系统，中间靠 **bridge（桥）** 翻译话题：

```text
你在 ROS 发 /cmd_vel
    → bridge 转成 Gazebo 认识的速度
    → DiffDrive 插件转轮子
    → 里程计再桥回 ROS 的 /odom
```

所以第 09 课学的 `cmd_vel`，在这里第一次真正“开到车”。

本机装的是 **Gazebo Fortress（Ignition Gazebo 6）**，不是很老的 Gazebo Classic；命令常见 `ign gazebo` / `ros_gz_sim`。

## 常见疑问

**Q：本机有 Gazebo 吗？**  
有（若已按本教程装过 `ros-humble-ros-gz`）。可用 `ign gazebo --versions` 查看，应为 6.x。

**Q：Gazebo 和 RViz 有何不同？**  
Gazebo = 仿真物理世界；RViz = 可视化 ROS 数据（TF、点云、路径…）。常一起开。

**Q：bridge 是干什么的？**  
两边话题名/消息类型不同，bridge 做双向（或单向）转换，否则 ROS 节点和仿真车各说各话。

**Q：为什么仿真里车能动，却还没写 PID？**  
DiffDrive 插件内部处理了“按 cmd_vel 转轮”。真机上这层通常换成驱动板 + 速度环。

**Q：无显示器能跑吗？**  
GUI 基本不行。可在有桌面的机器上跑；或以后再学无头less 选项。

## 动手

需要图形界面：

```bash
source /opt/ros/humble/setup.bash
source $ROS_WS/install/setup.bash
ros2 launch simple_robot gazebo.launch.py
```

另开终端发速度：

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.1}}"
```

或用第 09 课节点：

```bash
ros2 run py_learning cmd_vel_publisher --ros-args -p linear_x:=0.15 -p angular_z:=0.2
```

看里程计：

```bash
ros2 topic echo /odom --once
```

## 关键命令

```bash
ros2 launch simple_robot gazebo.launch.py
ros2 topic list | grep -E 'cmd_vel|odom'
ign topic -l          # 看 Gazebo 侧话题（命令也可能是 gz topic）
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`models/simple_diff_robot/model.sdf`](../../../workspaces/ros2-humble/src/simple_robot/models/simple_diff_robot/model.sdf) | 差速插件模型 |
| [`worlds/simple_robot.sdf`](../../../workspaces/ros2-humble/src/simple_robot/worlds/simple_robot.sdf) | 世界 |
| [`launch/gazebo.launch.py`](../../../workspaces/ros2-humble/src/simple_robot/launch/gazebo.launch.py) | 启动仿真 + bridge |

Bridge 映射（简化理解）：

```text
/cmd_vel  ↔  /model/simple_diff_robot/cmd_vel
/odom     ↔  /model/simple_diff_robot/odometry
```

## 小练习

1. 只发前进、只发转向，观察车运动。
2. 打开官方 demo 对比：`ros2 launch ros_gz_sim_demos diff_drive.launch.py`

## 注意

- 无显示器/SSH 无转发时，Gazebo GUI 可能起不来；可稍后在本机桌面试。
- 若找不到模型，确认 launch 已把 `models/` 加入 `IGN_GAZEBO_RESOURCE_PATH`。

## 验证标准（给学员与 AI）

- [ ] 已安装 `ros-humble-ros-gz`（`scripts/check_env.sh` 提示 OK）
- [ ] （有桌面）`ros2 launch simple_robot gazebo.launch.py` 能起仿真
- [ ] 向 `/cmd_vel` 发速度后，车有运动或 `/odom` 有数据

失败时：检查资源路径；无桌面则记录限制并继续概念学习。

## 小结

- 仿真 = 物理世界；bridge = ROS 与仿真的翻译官。
- 下一课：[12 · RViz 与组合 Launch](/ros2/rviz-launch/)
