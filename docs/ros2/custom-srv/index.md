---
title: "07 自定义服务"
order: 80
---
# 07 · 自定义服务（`.srv`）

## 目标

在 `lesson_interfaces` 中定义 `SetLed.srv`，实现点亮/关闭 LED 的服务端与客户端。

## 概念

`.srv` 用 `---` 分成两段：上面是 **Request**，下面是 **Response**。

本课模拟 MCU 上的 LED 控制（偏硬件向）：

```text
uint8 led_id
bool turn_on
---
bool success
string message
```

> 字段不要叫 `on`：命令行 YAML 里 `on:` 容易和布尔字面量冲突，故用 `turn_on`。

## 通俗理解

自定义 `.srv` = 自己定义的**远程函数签名**：对方要传什么参数进来，你返回什么结果出去。

本课模拟“点某一路 LED”：请求里是几号灯、开还是关；响应里是成功与否和说明文字。真机上服务端后面往往接 GPIO/串口；这里先用字典假装三路灯。

和自定义消息的关系：`.msg` 是“数据包长什么样”；`.srv` 是“一次调用的入参/出参长什么样”。

## 常见疑问

**Q：为什么字段不叫 `on`？**  
命令行 `ros2 service call ... "{on: true}"` 时，YAML 里 `on` 容易被当成特殊布尔写法，导致奇怪报错。用 `turn_on` 更安全。

**Q：自定义 srv 和官方 `AddTwoInts` 有何不同？**  
机制完全一样，只是字段换成你的业务。学会官方例子后，换自己的 `.srv` 即可。

**Q：服务端里的 `self.leds` 是真硬件吗？**  
不是，是内存里的演示状态。换成写串口/GPIO 就是硬件控制雏形。

**Q：什么时候用自定义 srv 而不是 Topic？**  
需要“做完并确认结果”的短操作：设参数、开关、触发一次动作。持续上报仍用 Topic + 自定义 msg。

## 动手

```bash
cd $ROS_WS
export PATH="/usr/bin:$PATH"   # 避开 conda 干扰接口生成时建议加上
source /opt/ros/humble/setup.bash
colcon build --packages-select lesson_interfaces py_learning \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

终端 A：

```bash
ros2 run py_learning set_led_server
```

终端 B：

```bash
ros2 run py_learning set_led_client 1 on
ros2 run py_learning set_led_client 1 off
```

或命令行：

```bash
ros2 service call /set_led lesson_interfaces/srv/SetLed "{led_id: 2, turn_on: true}"
```

## 关键命令

```bash
ros2 interface show lesson_interfaces/srv/SetLed
ros2 service list | grep set_led
ros2 service call /set_led lesson_interfaces/srv/SetLed "{led_id: 0, turn_on: false}"
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`srv/SetLed.srv`](../../../workspaces/ros2-humble/src/lesson_interfaces/srv/SetLed.srv) | 服务定义 |
| [`set_led_server.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/set_led_server.py) | 服务端 |
| [`set_led_client.py`](../../../workspaces/ros2-humble/src/py_learning/py_learning/set_led_client.py) | 客户端 |

## 小练习

1. 非法 `led_id=9`，观察 `success=False`。
2. 在服务端日志里打印当前三路 LED 状态字典。

## 验证标准（给学员与 AI）

- [ ] `set_led_server` 就绪
- [ ] `set_led_client 1 on` 返回 success
- [ ] `ros2 service call /set_led lesson_interfaces/srv/SetLed "{led_id: 2, turn_on: true}"` 成功

失败时：字段名是 `turn_on` 不是 `on`；先起 server。

## 小结

- 自定义 srv = 自己的“远程函数签名”。
- 与 Topic 自定义消息同一接口包管理。
- 下一课：[08 · TF2](/ros2/tf2/)
