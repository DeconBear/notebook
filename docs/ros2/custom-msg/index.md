---
title: "06 自定义消息"
order: 70
---
# 06 · 自定义消息（`.msg`）

## 目标

新建接口包 `lesson_interfaces`，定义 `SensorStatus.msg`，并用 Python 发布/订阅。

## 概念

官方消息（如 `std_msgs/String`）不够用时，要**自己定义数据结构**。

常见做法：单独一个 **接口包**（`ament_cmake` + `rosidl`），业务包只依赖它。

```text
lesson_interfaces/msg/SensorStatus.msg   ← 定义
py_learning/...publisher.py              ← 使用
```

本课消息模拟一块板子上报：设备 ID、温度、电压、是否正常。

## 通俗理解

自定义消息 = 给系统定一份**数据结构合同**（API 的“形状”）。

官方只有通用积木（`String`、`Twist`…）。你的板子要同时上报温度、电压、是否正常——硬拆成好几个话题会乱，也容易对不齐。自己写 `.msg`，发布方和订阅方就按同一张表填字段。

它**不是**控制逻辑，只规定“传什么、字段叫什么、类型是什么”。

```text
没有自定义消息：多个 Float32/String 拼凑，靠口头约定
有自定义消息：一个 SensorStatus，字段写死，改接口要重新编译对齐
```

## 常见疑问

**Q：自定义消息的作用到底是什么？**  
约定节点之间传什么数据。复杂/产品相关的状态、指令、标定结果，几乎都要自定义。

**Q：为什么单独做 `lesson_interfaces` 包？**  
接口和业务分离：多个包都能依赖同一份消息定义，避免复制粘贴、版本不一致。

**Q：改了 `.msg` 要做什么？**  
重新 `colcon build` 接口包和依赖它的包，再 `source install/setup.bash`。只改 py 不编接口，类型对不上。

**Q：和硬件什么关系？**  
驱动板/MCU 上报的状态，在 ROS 侧常常就落成自定义消息（本课的温度电压就是这个思路）。

## 动手

```bash
cd $ROS_WS
# 若终端里有 conda，建议先：export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
colcon build --packages-select lesson_interfaces py_learning \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

终端 A：

```bash
ros2 run py_learning sensor_status_publisher
```

终端 B：

```bash
ros2 run py_learning sensor_status_subscriber
```

查看接口：

```bash
ros2 interface show lesson_interfaces/msg/SensorStatus
ros2 topic echo /sensor_status
```

## 关键命令

```bash
ros2 interface list | grep lesson_interfaces
ros2 interface show lesson_interfaces/msg/SensorStatus
ros2 topic echo /sensor_status --once
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`msg/SensorStatus.msg`](../../../workspaces/ros2-humble/src/lesson_interfaces/msg/SensorStatus.msg) | 消息定义 |
| [`CMakeLists.txt`](../../../workspaces/ros2-humble/src/lesson_interfaces/CMakeLists.txt) | `rosidl_generate_interfaces` |
| [`sensor_status_publisher.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/sensor_status_publisher.py) | 发布 |
| [`sensor_status_subscriber.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/sensor_status_subscriber.py) | 订阅 |

`.msg` 核心字段：

```text
std_msgs/Header header
string device_id
float32 temperature_c
float32 voltage_v
bool ok
```

## 小练习

1. 给消息加一个 `float32 humidity`，改发布/订阅并重新 build。
2. 用参数改 `device_id`：`--ros-args -p device_id:=board_b`。

## 验证标准（给学员与 AI）

- [ ] `ros2 interface show lesson_interfaces/msg/SensorStatus` 成功
- [ ] publisher / subscriber 联调能打印温度电压
- [ ] `ros2 topic echo /sensor_status --once` 有自定义字段

失败时：先编 `lesson_interfaces` 再编 `py_learning`；避开 conda Python。

## 小结

- 自定义消息 = 自己的“数据结构合同”。
- 接口包与业务包分离，便于多包复用。
- 下一课：[07 · 自定义服务](/ros2/custom-srv/)
