---
title: "09 速度控制 cmd_vel"
order: 100
---
# 09 · 速度控制 `cmd_vel`

## 目标

发布 `geometry_msgs/Twist` 到 `/cmd_vel`，理解移动机器人最常用的速度接口。

## 概念

差速小车通常订阅：

```text
/cmd_vel  (geometry_msgs/msg/Twist)
  linear.x   前进速度 (m/s)
  angular.z  转向角速度 (rad/s)
```

本课先**只发话题**（用 `ros2 topic echo` 观察）。到第 11 课接上 Gazebo 后，同一话题就能开车。

## 通俗理解

本课**只讲“怎么发速度指令”**，不讲 PID。

`/cmd_vel` 是上层对底盘说的话：

> “请以线速度 \(v\)、角速度 \(\omega\) 运动。”

谁听、怎么变成电机电流/PWM，是**下一层**的事（差速驱动、`ros2_control`、板子上的速度环/PID）。本课停在“把期望速度发到话题上”。

```text
你的节点  --发布 Twist-->  /cmd_vel  --订阅-->  底盘/仿真插件
                              ↑
                         第 09 课停在这里
```

平面差速车通常只用两个量：

- `linear.x`：前进/后退（m/s），\(x\) 一般是车头方向  
- `angular.z`：绕竖直轴转向（rad/s）

## 常见疑问

**Q：这一节也管 PID 吗？**  
不管。这里是开环的“期望速度接口”。PID 在接收端用反馈去追这个期望；仿真里插件代劳，真机上多在驱动/`ros2_control` 里。

**Q：这两行代码在干什么？**

```python
msg.linear.x = float(self.get_parameter('linear_x').value)
msg.angular.z = float(self.get_parameter('angular_z').value)
```

| 片段 | 含义 |
|------|------|
| `msg.linear.x = ...` | 填前进速度 |
| `msg.angular.z = ...` | 填转向角速度 |
| `get_parameter(...).value` | 从 ROS 参数读数（可用 `-p` 改，不必改代码） |
| `float(...)` | 保证类型是浮点，匹配 `Twist` 字段 |

等于在填一张速度单；其它 `linear.y/z`、`angular.x/y` 对平面车一般保持 0。

**Q：为什么话题叫 cmd_vel？**  
社区约定：command velocity。导航、键盘遥控、很多仿真都认这个名字。

**Q：发了 cmd_vel 车为什么不动？**  
本课若没接仿真/真机，本来就不会动，只能 `topic echo` 看到数。第 11 课接上 Gazebo 后才会跑。

## 动手

```bash
source /opt/ros/humble/setup.bash
source $ROS_WS/install/setup.bash

ros2 run py_learning cmd_vel_publisher --ros-args \
  -p linear_x:=0.2 -p angular_z:=0.3
```

另开终端：

```bash
ros2 topic echo /cmd_vel
```

命令行发一次：

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}"
```

## 关键命令

```bash
ros2 topic info /cmd_vel
ros2 interface show geometry_msgs/msg/Twist
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.15}, angular: {z: 0.2}}"
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`cmd_vel_publisher.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/cmd_vel_publisher.py) | 周期发布 Twist |

关键两行（填速度单，不是 PID）：

```python
msg.linear.x = float(self.get_parameter('linear_x').value)   # 前进 m/s
msg.angular.z = float(self.get_parameter('angular_z').value) # 转向 rad/s
```

## 小练习

1. 只给 `angular.z`，想象原地转。
2. 第 11 课仿真起来后，用本节点或 `topic pub` 开车。

## 验证标准（给学员与 AI）

- [ ] `cmd_vel_publisher` 运行中 `ros2 topic echo /cmd_vel` 能看到 Twist
- [ ] 能说明：本课只发期望速度，不包含 PID
- [ ] 能解释 `linear.x` / `angular.z` 含义

失败时：先 source 工作区；本课无仿真时车不会动是正常的。

## 小结

- `/cmd_vel` 是移动机器人的“油门方向盘”话题。
- 下一课：[10 · URDF](/ros2/urdf/)
