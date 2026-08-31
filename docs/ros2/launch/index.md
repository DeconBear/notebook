---
title: "05 Launch 一键启动"
order: 60
---
# 05 · Launch 一键启动

> 建议先看完 [04 · Parameter](/ros2/parameters/)，再看本课。

## 目标

用一个 launch 文件同时启动多个节点；再做一个能把参数传给 `param_talker` 的 launch。

## 概念

`ros2 run` 一次只起一个进程。真实机器人常要同时起：驱动、感知、导航……

**Launch** = 用 Python（或 XML）描述「要起哪些节点、叫什么名、带什么参数」，然后：

```bash
ros2 launch <包名> <launch文件>
```

本课两个文件：

| Launch 文件 | 做什么 |
|-------------|--------|
| `pubsub.launch.py` | 同时起 `talker` + `listener` |
| `param_pubsub.launch.py` | 起 `param_talker` + `listener`，并可从命令行覆盖参数 |

## 通俗理解

Launch 是机器人的**开机脚本 / 总装配单**：

- `ros2 run` = 只拧开一颗螺丝（起一个节点）。
- `ros2 launch` = 按图纸一次装好整机（多节点 + 参数 + 以后还有重映射、命名空间）。

**Parameter 管“跑成什么样”，Launch 管“怎么把系统拉起来”。** 两者常一起用，但职责不同。

## 常见疑问

**Q：Launch 会替代 Parameter 吗？**  
不会。Launch 经常**注入**参数；参数仍是节点上的配置机制。

**Q：Ctrl+C 会停掉所有子节点吗？**  
一般会：launch 拉起的进程由它统一管理，中断后一起收。

**Q：为什么 float 参数要用 `ParameterValue(..., value_type=float)`？**  
Launch 参数默认当字符串；不声明类型，可能把 `"1.0"` 错传成字符串，节点读参失败。

**Q：真实项目里 Launch 有多重要？**  
非常重要。导航/仿真栈往往一条 `ros2 launch ...` 起几十个节点；没有它几乎无法复现环境。

## 动手

```bash
cd $ROS_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 1. 最简：一对 pub/sub

```bash
ros2 launch py_pubsub pubsub.launch.py
```

一个终端里应同时看到 talker 的「发布」和 listener 的「收到」。`Ctrl+C` 会停掉 launch 拉起的全部子进程。

### 2. 带参数的 launch

```bash
ros2 launch py_pubsub param_pubsub.launch.py \
  message_prefix:=课五 \
  publish_period:=0.5
```

应看到 `已启动: prefix="课五", period=0.5s`，且 listener 收到 `课五 ROS 2: ...`。

## 关键命令

```bash
ros2 launch py_pubsub pubsub.launch.py
ros2 launch py_pubsub param_pubsub.launch.py message_prefix:=你好 publish_period:=1.0
# 查看包里安装了哪些 launch（路径因安装而异）
ros2 pkg prefix py_pubsub
ls $(ros2 pkg prefix py_pubsub)/share/py_pubsub/launch/
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`launch/pubsub.launch.py`](../../../workspaces/ros2-humble/src/py_pubsub/launch/pubsub.launch.py) | 双节点 |
| [`launch/param_pubsub.launch.py`](../../../workspaces/ros2-humble/src/py_pubsub/launch/param_pubsub.launch.py) | 参数 + 双节点 |
| [`setup.py`](../../../workspaces/ros2-humble/src/py_pubsub/setup.py) | 把 `launch/*.launch.py` 装进 share |

最小 launch 骨架：

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='py_pubsub', executable='talker', name='talker', output='screen'),
        Node(package='py_pubsub', executable='listener', name='listener', output='screen'),
    ])
```

传 float 参数时，launch 参数默认是字符串，需要 `ParameterValue(..., value_type=float)`，否则类型可能不对。

## 小练习（可选）

1. 改 `pubsub.launch.py`，只启动 `listener`，再用命令行 `ros2 topic pub` 发消息。
2. 给 `param_pubsub.launch.py` 换一组前缀/周期，观察差异。
3. 对比：`ros2 run` 开两个终端 vs `ros2 launch` 一个终端。

## 验证标准（给学员与 AI）

- [ ] `ros2 launch py_pubsub pubsub.launch.py` 同时出现 talker 与 listener 日志
- [ ] `ros2 launch py_pubsub param_pubsub.launch.py message_prefix:=课五 publish_period:=0.5` 能启动且前缀生效
- [ ] Ctrl+C 后相关进程退出

失败时：先 `colcon build` 确保 launch 已安装到 share；float 参数需类型正确。

## 小结

- Launch = 编排多个节点（以及参数、重映射等）的启动脚本。
- 与 Parameter 常一起用：launch 负责「怎么起」，参数负责「起成什么样」。
- 入门主线到此：环境 → Topic → Service → Action → Parameter → Launch。
- 下一阶段：[06 · 自定义消息](/ros2/custom-msg/)
- 上一课：[04 · Parameter](/ros2/parameters/) · 目录：[README](/ros2/overview/)
