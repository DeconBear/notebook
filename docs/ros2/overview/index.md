---
title: "ROS 2 导读"
order: 1
---
# ROS 2 学习笔记

边做边记。环境：**Ubuntu 22.04 + ROS 2 Humble**。`$ROS_WS` 指 notebook 仓库里的 colcon 工作区 `workspaces/ros2-humble/`（源码在 `src/`，笔记在站点 `/ros2/`）。

本课来自 [ros2-humble-notes](https://github.com/DeconBear/ros2-humble-notes)，已并入本仓库。

## 怎么用这些笔记

先进入工作区（相对 notebook 仓库根目录）：

```bash
cd workspaces/ros2-humble
export ROS_WS="$PWD"
source /opt/ros/humble/setup.bash
source "$ROS_WS/install/setup.bash"   # 编译之后
```


1. **按课号顺序慢慢看**；每课都能在本机复现。
2. 笔记讲概念与步骤，**代码以 `workspaces/ros2-humble/src/` 为准**；每课文末有「对照代码」表（相对链接指向仓库内源文件）。
3. 每课都有 **「通俗理解」**、**「常见疑问」**、**「验证标准」**（供学员自检与 AI 判读是否过关）。
4. 编译（若 shell 里启用了 conda，建议先避开它的 Python）：

```bash
cd $ROS_WS
export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

只编某一阶段：

```bash
colcon build --packages-select lesson_interfaces py_learning simple_robot \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

## 课号 ↔ 代码对照

### 基础（00–05，包 `py_pubsub`）

| 课 | 笔记 | 代码 | 运行 |
|----|------|------|------|
| 00 | [00-环境与工作区.md](/ros2/env/) | 环境配置 | — |
| 01 | [01-Topic发布订阅.md](/ros2/topics/) | `py_pubsub` talker/listener | `ros2 run py_pubsub talker` |
| 02 | [02-Service请求应答.md](/ros2/services/) | add_two_ints_* | `ros2 run py_pubsub add_two_ints_server` |
| 03 | [03-Action长任务.md](/ros2/actions/) | fibonacci_action_* | `ros2 run py_pubsub fibonacci_action_server` |
| 04 | [04-Parameter参数.md](/ros2/parameters/) | param_talker | `ros2 run py_pubsub param_talker` |
| 05 | [05-Launch一键启动.md](/ros2/launch/) | `launch/*.launch.py` | `ros2 launch py_pubsub pubsub.launch.py` |

### 阶段 A–C（06–12）

| 课 | 笔记 | 包 / 代码 | 运行 |
|----|------|-----------|------|
| 06 | [06-自定义消息.md](/ros2/custom-msg/) | `lesson_interfaces` + `py_learning` sensor_* | `ros2 run py_learning sensor_status_publisher` |
| 07 | [07-自定义服务.md](/ros2/custom-srv/) | SetLed + set_led_* | `ros2 run py_learning set_led_server` |
| 08 | [08-TF2坐标变换.md](/ros2/tf2/) | tf_broadcaster / tf_listener | `ros2 run py_learning tf_broadcaster` |
| 09 | [09-速度控制cmd_vel.md](/ros2/cmd-vel/) | cmd_vel_publisher | `ros2 run py_learning cmd_vel_publisher` |
| 10 | [10-URDF与机器人描述.md](/ros2/urdf/) | `simple_robot` URDF | `ros2 launch simple_robot display.launch.py` |
| 11 | [11-Gazebo最小仿真.md](/ros2/gazebo/) | Gazebo world + bridge | `ros2 launch simple_robot gazebo.launch.py` |
| 12 | [12-RViz与组合Launch.md](/ros2/rviz-launch/) | sim_stack | `ros2 launch simple_robot sim_stack.launch.py` |

## 课程目录（状态）

| 课 | 主题 | 状态 |
|----|------|------|
| 00–05 | 环境 / Topic / Service / Action / Param / Launch | 已完成 |
| 06–07 | 自定义 msg / srv（阶段 A） | 已完成 |
| 08–10 | TF2 / cmd_vel / URDF（阶段 B） | 已完成 |
| 11–12 | Gazebo / RViz 组合栈（阶段 C） | 已完成 |

## 包一览

| 包 | 内容 |
|----|------|
| `py_pubsub` | 01–05 入门通信与 Launch |
| `lesson_interfaces` | 06–07 自定义接口 |
| `py_learning` | 06–09 示例节点 |
| `simple_robot` | 10–12 URDF / Gazebo / RViz |

## 形式约定

- 一课一个 Markdown；源码文件头可标 `Lesson 0x`。
- 结构：目标 → 概念 → **通俗理解** → **常见疑问** → 动手 → 命令 → 对照代码 → **验证标准** → 小结。
- 不整页复制源码；用对照表链接真实文件。
- 需要冻结某课版本时用 `git tag lesson-0x`。
