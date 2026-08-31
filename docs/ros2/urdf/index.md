---
title: "10 URDF 与机器人描述"
order: 110
---
# 10 · URDF 与机器人描述

## 目标

用 URDF 描述一辆简单差速小车，在 RViz 中显示，并用 GUI 拖动关节。

## 概念

**URDF**（Unified Robot Description Format）用 XML 描述 link（刚体）与 joint（关节）。

| 组件 | 作用 |
|------|------|
| `robot_state_publisher` | 读 URDF + 关节角 → 发 TF / `robot_description` |
| `joint_state_publisher_gui` | 手动拖关节（学习用） |
| RViz `RobotModel` | 看三维模型 |

包：`simple_robot`。

## 通俗理解

URDF 是机器人的**说明书/装配图**：有哪些零件（link）、怎么连（joint）、长什么样（visual）、碰撞体积多大。

它本身不会让车跑起来。要“看见”它，通常：

1. `robot_state_publisher` 读说明书 → 算出各坐标系 TF，并发布 `robot_description`
2. RViz 订阅这些信息 → 画出三维模型
3. （学习用）`joint_state_publisher_gui` 让你拖滑条假装转动关节

和第 08 课手写 TF 的关系：URDF + RSP 能**自动**生成一大棵 TF；手写 TF 是为了先建立直觉。

## 常见疑问

**Q：URDF 是仿真吗？**  
不是。URDF 是描述；Gazebo 才是物理仿真（第 11 课用 SDF/插件）。RViz 只是可视化，没有真实物理碰撞。

**Q：为什么要学 URDF？**  
导航、MoveIt、很多驱动都依赖“机器人长什么样、关节叫什么”。没有描述，TF 树和碰撞模型都对不齐。

**Q：Fixed Frame 选错会怎样？**  
RViz 里模型可能飞掉或看不见。本课用 `base_link`。

**Q：没有图形界面怎么办？**  
至少可 `ros2 topic echo /robot_description` 确认描述已加载；完整显示需本机桌面。

## 动手

需要图形界面（本机桌面）：

```bash
source /opt/ros/humble/setup.bash
source $ROS_WS/install/setup.bash
ros2 launch simple_robot display.launch.py
```

在 RViz 中确认 Fixed Frame 为 `base_link`，应看到车体与轮子；拖动 `joint_state_publisher_gui` 滑条，轮子姿态会变。

无 GUI 时至少可检查描述是否加载：

```bash
ros2 topic echo /robot_description --once | head
```

## 关键命令

```bash
ros2 launch simple_robot display.launch.py
check_urdf $(ros2 pkg prefix simple_robot)/share/simple_robot/urdf/simple_robot.urdf
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`urdf/simple_robot.urdf`](../../../workspaces/ros2-humble/src/simple_robot/urdf/simple_robot.urdf) | 模型 |
| [`launch/display.launch.py`](../../../workspaces/ros2-humble/src/simple_robot/launch/display.launch.py) | RSP + JSP GUI + RViz |
| [`rviz/urdf.rviz`](../../../workspaces/ros2-humble/src/simple_robot/rviz/urdf.rviz) | RViz 配置 |

## 小练习

1. 改车体尺寸，重新 build 再 launch。
2. 对比 URDF 里的 `lidar_link` 与第 08 课手写 TF 的思路。

## 验证标准（给学员与 AI）

- [ ] （有桌面）`ros2 launch simple_robot display.launch.py` 能开 RViz 并看到模型
- [ ] 或至少 `ros2 topic echo /robot_description --once` 有 URDF 内容
- [ ] 能区分：URDF=描述，RViz=可视化，不是物理仿真

失败时：无 DISPLAY 则跳过 GUI，用 topic 验证；检查 urdf 是否安装到 share。

## 小结

- URDF = 机器人几何与关节的说明书；RSP 把它变成 TF。
- 下一课：[11 · Gazebo](/ros2/gazebo/)
