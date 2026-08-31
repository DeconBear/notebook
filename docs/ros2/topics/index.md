---
title: "01 Topic 发布订阅"
order: 20
---
# 01 · Topic 发布/订阅

## 目标

写并跑通一对节点：一个不停发消息，一个接收打印。理解 **节点、话题、消息**。

## 概念

```text
  talker 节点  --publish-->  话题 /chatter  --subscribe-->  listener 节点
                 (String)                         (String)
```

| 名词 | 含义 | 本课例子 |
|------|------|----------|
| 节点 Node | 一个 ROS 程序角色 | `talker`、`listener` |
| 话题 Topic | 双方约定的频道名 | `chatter` |
| 消息 Message | 频道上的数据结构 | `std_msgs/msg/String` |
| 发布者 Publisher | 往话题发 | `create_publisher` |
| 订阅者 Subscriber | 从话题收 | `create_subscription` |

要点：

- 发布者不关心谁在听；订阅者不关心谁在发。
- **消息类型 + 话题名** 必须一致，否则收不到。
- Topic 适合：传感器、状态、持续数据流。

## 通俗理解

Topic 像**电台频道**：

- 主播（发布者）不停播报，不关心有没有人听。
- 听众（订阅者）调到同一频道就能收，不关心主播是谁。
- 频道名（话题名）和节目格式（消息类型）必须一致，否则对不上。

传感器数据、电池电量、速度指令这类“持续往外冒”的信息，最适合 Topic。

本课通信发生在本机进程之间；底层确实走 DDS/网络栈，但你写节点时通常感觉不到，先当“进程间广播”即可。

## 常见疑问

**Q：发布者必须先启动吗？**  
不必。后启动的一方连上后就能收/发。但若一直没人发布，订阅者就一直等。

**Q：`create_publisher(..., 10)` 里的 10 是什么？**  
队列深度：来不及处理时最多先攒几条。入门当“缓冲大小”即可，不必先学完整 QoS。

**Q：Topic 能保证对方一定收到吗？**  
默认偏“尽力而为”，适合高频传感器。要强可靠、要回执，更常考虑 Service/Action 或调 QoS（进阶）。

**Q：一个话题可以有多个订阅者吗？**  
可以，一对多很常见。

## 动手

```bash
cd $ROS_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端 A：

```bash
ros2 run py_pubsub talker
```

终端 B：

```bash
ros2 run py_pubsub listener
```

应看到类似：

```text
[talker]: 发布: "你好 ROS 2: 3"
[listener]: 收到: "你好 ROS 2: 3"
```

## 关键命令

```bash
ros2 node list                 # 有哪些节点
ros2 topic list                # 有哪些话题
ros2 topic echo /chatter       # 实时打印话题内容
ros2 topic info /chatter       # 类型、发布/订阅者数量
ros2 interface show std_msgs/msg/String
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`talker_node.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/talker_node.py) | 定时发布 |
| [`listener_node.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/listener_node.py) | 回调打印 |
| [`setup.py`](../../../workspaces/ros2-humble/src/py_pubsub/setup.py) | 注册 `talker` / `listener` 入口 |
| [`package.xml`](../../../workspaces/ros2-humble/src/py_pubsub/package.xml) | 声明依赖 `rclpy`、`std_msgs` |

发布关键几行：

```python
self.publisher_ = self.create_publisher(String, 'chatter', 10)
self.publisher_.publish(msg)
```

订阅关键几行：

```python
self.create_subscription(String, 'chatter', self.listener_callback, 10)
```

`10` 是队列长度（QoS 里的 depth），入门先当「缓冲几条消息」即可。

## 小练习（可选）

1. 把话题名改成 `hello`，两边一起改，重新 build 再跑。
2. 把定时器从 `0.5` 秒改成 `1.0` 秒。
3. 只开 listener，用命令行发一条：

```bash
ros2 topic pub --once /chatter std_msgs/msg/String "{data: '手动一条'}"
```

## 验证标准（给学员与 AI）

- [ ] `ros2 run py_pubsub talker` 持续打印「发布」
- [ ] 另一终端 `ros2 run py_pubsub listener` 能收到对应内容
- [ ] `ros2 topic echo /chatter` 能看到消息

失败时：确认两边都 `source "$ROS_WS/install/setup.bash"`；话题名与类型一致。

## 小结

- Topic = 广播频道；适合持续数据。
- 类型与话题名对齐是通信前提。
- 下一课：[02 · Service 请求/应答](/ros2/services/)
