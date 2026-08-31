---
title: "04 Parameter 参数"
order: 50
---
# 04 · Parameter 参数

> 建议单独看完本课再看 [05 · Launch](/ros2/launch/)。

## 目标

给节点加上可配置参数：启动时传入，运行中用命令行修改，观察发布内容与频率变化。

## 概念

**Parameter** 是挂在节点上的「旋钮」：名字 + 类型 + 值。

| 方式 | 何时生效 | 例子 |
|------|----------|------|
| 声明默认值 | 代码里 `declare_parameter` | `message_prefix='你好'` |
| 启动时覆盖 | `ros2 run ... --ros-args -p` | `-p message_prefix:=测试` |
| 运行中修改 | `ros2 param set` | 改前缀、改周期 |

和 Topic/Service/Action 不同：参数**不是节点之间的通信方式**，而是**配置节点行为**。

本课节点：`param_talker`，仍往 `/chatter` 发 `String`，但前缀和周期可变。

## 通俗理解

Parameter 是节点上的**旋钮/拨码开关**，不是“和别的节点聊天”的通道。

- Topic/Service/Action：节点 **之间** 传数据、下命令。
- Parameter：配置 **这一个节点自己** 怎么表现（频率、前缀、串口号、最大速度……）。

实际项目里：同一套程序，仿真用一套参数、真机用另一套；现场调试还能 `ros2 param set` 热改，而不用改代码重编。

## 常见疑问

**Q：参数和话题有什么本质区别？**  
话题是数据流；参数是配置项。传感器每帧读数不该做成参数；发布频率适合做成参数。

**Q：改参数一定立刻生效吗？**  
只有节点写了 `on_set_parameters` 一类回调（或自己定期读参）才会跟手。本课的 `param_talker` 支持动态改前缀和周期。

**Q：和 Launch 什么关系？**  
Launch 负责“把谁拉起来”；常在启动时把参数灌进节点。详见第 05 课。

**Q：参数会存盘吗？**  
默认改的是运行中内存里的值，进程退出就没了。要持久化需写 YAML，并由 launch/命令加载。

## 动手

```bash
cd $ROS_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 1. 用默认参数启动

```bash
ros2 run py_pubsub param_talker
```

应看到类似：`已启动: prefix="你好", period=0.5s`。

### 2. 启动时覆盖参数

另开终端（先停掉上一个）：

```bash
ros2 run py_pubsub param_talker --ros-args \
  -p message_prefix:=测试 \
  -p publish_period:=0.3
```

### 3. 运行中改参数

保持 `param_talker` 在跑，另开终端：

```bash
ros2 param list /param_talker
ros2 param get /param_talker message_prefix
ros2 param set /param_talker message_prefix 动态改
ros2 param set /param_talker publish_period 1.0
```

节点日志会出现 `参数更新: ...`，随后发布内容/频率跟着变。

## 关键命令

```bash
ros2 param list /param_talker
ros2 param get /param_talker message_prefix
ros2 param set /param_talker message_prefix 新前缀
ros2 param describe /param_talker publish_period
```

启动时传参模板：

```bash
ros2 run <包> <可执行文件> --ros-args -p 名字:=值 -p 名字2:=值2
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`param_talker.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/param_talker.py) | 声明参数、读参数、动态回调 |

关键几行：

```python
self.declare_parameter('message_prefix', '你好')
self.declare_parameter('publish_period', 0.5)
self.add_on_set_parameters_callback(self.on_params_changed)
```

在回调里改 `self.prefix`，或取消旧 timer、按新周期重建 timer。

## 小练习（可选）

1. 把 `publish_period` 设成 `0`，看节点是否拒绝（本课代码会拒绝 ≤0）。
2. 开一个 `listener`，确认改前缀后收到的字符串也变了。
3. 想一想：哪些适合做成参数？（频率、话题名、串口号……）哪些不适合？（每帧传感器数据）

## 验证标准（给学员与 AI）

- [ ] `ros2 run py_pubsub param_talker` 能启动
- [ ] `ros2 param list /param_talker` 能看到 `message_prefix`、`publish_period`
- [ ] `ros2 param set /param_talker message_prefix 测试` 后，日志/发布内容前缀变化

失败时：节点名是否为 `/param_talker`；是否写了参数回调才会热更新。

## 小结

- 参数 = 节点配置；可启动注入，可运行中改。
- 下一课用 Launch **一次启动多个节点**，并可在 launch 里注入参数：[05 · Launch](/ros2/launch/)
- 上一课：[03 · Action](/ros2/actions/) · 目录：[README](/ros2/overview/)
