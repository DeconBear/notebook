---
title: "08 TF2 坐标变换"
order: 90
---
# 08 · TF2 坐标变换

## 目标

发布静态外参 `base_link → sensor_link`，并用 listener 查询相对位姿。

## 概念

机器人上每个传感器、连杆都有坐标系（frame）。TF2 维护这些坐标系之间的变换树。

```text
base_link ──(固定外参)──► sensor_link
```

| 名词 | 含义 |
|------|------|
| `frame_id` | 父坐标系 |
| `child_frame_id` | 子坐标系 |
| Static TF | 相对位姿不变（如传感器安装位置） |
| Dynamic TF | 随时间变（如轮式里程计） |

本课：传感器在车体前方 0.2 m、上方 0.1 m。

## 通俗理解

TF 回答的是：**“这个零件相对那个零件，在空间里偏了多少、转了多少？”**

车上每个传感器、轮子、雷达都有自己的坐标系。若不知道激光相对车体装在哪，就无法把激光点变到地图上。TF2 维护这棵“坐标系树”。

本课是**静态外参**（螺丝拧死的安装位置）：`base_link → sensor_link` 固定为前方 0.2 m、上方 0.1 m。轮子滚动带来的位姿变化属于**动态 TF**，以后里程计/定位会发。

## 常见疑问

**Q：TF 是一种 Topic 吗？**  
底层会通过 `/tf`、`/tf_static` 等话题传，但你通常用 TF API（broadcast/lookup），而不是自己解析原始消息。

**Q：和 URDF 什么关系？**  
URDF 描述机器人结构；`robot_state_publisher` 会根据 URDF（和关节角）自动发很多 TF。本课是手写静态 TF，帮助先建立直觉。

**Q：lookup 报 frame 不存在？**  
多半是 broadcaster 还没起来，或父子坐标系名字写错。先让 broadcaster 稳定跑着再 lookup。

**Q：为什么要学这个？**  
几乎所有导航、感知、标定都依赖 TF。不懂 TF，后面仿真和 Nav2 会很懵。

## 动手

```bash
source /opt/ros/humble/setup.bash
source $ROS_WS/install/setup.bash
```

终端 A：

```bash
ros2 run py_learning tf_broadcaster
```

终端 B：

```bash
ros2 run py_learning tf_listener
```

应看到：`x=0.20 y=0.00 z=0.10`。

也可：

```bash
ros2 run tf2_ros tf2_echo base_link sensor_link
ros2 run tf2_tools view_frames   # 生成 frames.pdf（需 graphviz）
```

## 关键命令

```bash
ros2 run tf2_ros tf2_echo base_link sensor_link
ros2 topic echo /tf_static --once
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`tf_broadcaster.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/tf_broadcaster.py) | 静态 TF 发布 |
| [`tf_listener.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/tf_listener.py) | 查询变换 |

## 小练习

1. 把传感器改到左侧（`y=0.15`），重新 build 观察 listener。
2. 在 RViz 里加 TF 显示（Fixed Frame = `base_link`）。

## 验证标准（给学员与 AI）

- [ ] `tf_broadcaster` 显示已发布静态 TF
- [ ] `tf_listener` 打印约 `x=0.20 y=0.00 z=0.10`
- [ ] 或 `ros2 run tf2_ros tf2_echo base_link sensor_link` 有输出

失败时：broadcaster 需先于 listener 稳定运行。

## 小结

- TF = 机器人的“空间关系说明书”。
- 先搞懂静态外参，再学动态里程计 TF。
- 下一课：[09 · cmd_vel](/ros2/cmd-vel/)
