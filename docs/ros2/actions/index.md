---
title: "03 Action 长任务"
order: 40
---
# 03 · Action 长任务（带进度）

## 目标

写并跑通一对 Action：客户端发 `order`，服务端边算斐波那契边发 **feedback**，最后返回完整 **result**。理解 Action 与 Topic / Service 的差别。

## 概念

```text
  client --goal(order)-->  Action /fibonacci
           <---feedback---   server 逐步计算
           <---result------   全部完成（或取消）
```

| | Topic | Service | Action |
|--|-------|---------|--------|
| 模式 | 持续广播 | 一问一答 | 目标 + 进度 + 结果 |
| 像什么 | 电台 | 函数调用 | 可取消的长任务 |
| 本课 | `chatter` | `add_two_ints` | `fibonacci` |
| 适合 | 传感器流 | 短查询/短命令 | 导航、抓取、长时间运算 |

Action 接口三部分（`---` 分隔）：

```text
# Goal（目标）
int32 order
---
# Result（最终结果）
int32[] sequence
---
# Feedback（过程进度）
int32[] sequence
```

类型名：`example_interfaces/action/Fibonacci`。

生命周期（入门版）：

1. 客户端发 **goal**
2. 服务端 **accept / reject**
3. 执行中反复发 **feedback**
4. 结束时给 **result**（成功 / 取消 / 中止）

## 通俗理解

Action 像**外卖订单**：

1. 你下单（goal）：要算长度为 8 的斐波那契。
2. 商家接单或拒单（accept/reject）。
3. 配送中不断推送进度（feedback）。
4. 送到后给你最终结果（result）；中途也可以取消。

Service 像“问一句立刻答完”；Action 适合**较久、要进度、可能取消**的事：导航到点、机械臂运动、长时间计算。

## 常见疑问

**Q：为什么不直接用 Service 做长任务？**  
Service 期间你很难优雅地拿中间进度，取消也不如 Action 标准。长任务用 Action 是社区习惯。

**Q：feedback 和 result 有什么区别？**  
feedback = 过程快照（可以很多条）；result = 最终答卷（通常一次）。

**Q：本课一定要会多线程 executor 吗？**  
入门先记住：Action 执行可能较久，服务端要用能同时处理取消/反馈的 executor。细节以后再挖。

**Q：和 Topic 怎么分工？**  
Topic 持续播报状态；Action 是“请完成这个目标”。导航里两者常一起出现。

## 动手

```bash
cd $ROS_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端 A：

```bash
ros2 run py_pubsub fibonacci_action_server
```

终端 B：

```bash
ros2 run py_pubsub fibonacci_action_client 8
```

应看到多行 `进度 feedback: [...]`，最后：

```text
最终结果: [0, 1, 1, 2, 3, 5, 8, 13]
```

也可用命令行发 goal（服务端需已启动）：

```bash
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 6}" --feedback
```

## 关键命令

```bash
ros2 action list
ros2 action info /fibonacci
ros2 interface show example_interfaces/action/Fibonacci
ros2 action send_goal /fibonacci example_interfaces/action/Fibonacci "{order: 5}" --feedback
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`fibonacci_action_server.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/fibonacci_action_server.py) | 接受 goal、发 feedback、返回 result |
| [`fibonacci_action_client.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/fibonacci_action_client.py) | 发 goal、收 feedback、打印 result |
| [`setup.py`](../../../workspaces/ros2-humble/src/py_pubsub/setup.py) | 注册两个可执行入口 |

服务端关键几行：

```python
ActionServer(self, Fibonacci, 'fibonacci', execute_callback=...)
goal_handle.publish_feedback(feedback_msg)
goal_handle.succeed()
```

客户端关键几行：

```python
ActionClient(self, Fibonacci, 'fibonacci')
send_goal_async(goal_msg, feedback_callback=...)
goal_handle.get_result_async()
```

说明：服务端用了 `MultiThreadedExecutor`，以便执行长任务时仍能处理取消等回调；入门先记住「Action 执行可能较久，executor 要能并发」即可。

## 小练习（可选）

1. 把 `order` 改成 `10`，观察 feedback 变长。
2. 用 `ros2 action send_goal ... --feedback` 代替自己的客户端。
3. 对比 Service：`add_two_ints` 没有中间进度；Action 有。

## 验证标准（给学员与 AI）

- [ ] `fibonacci_action_server` 显示 Action 就绪
- [ ] `fibonacci_action_client 8` 出现多条 feedback，最终结果含 `13`
- [ ] 或 `ros2 action send_goal /fibonacci ... --feedback` 能看到进度

失败时：确认已 source 工作区；server 需保持运行。

## 小结

- Action = **Goal + Feedback + Result**，适合长任务。
- 三种通信学完：Topic 流数据、Service 短请求、Action 长任务。
- 上一课：[02 · Service](/ros2/services/) · 下一课：[04 · Parameter](/ros2/parameters/) · 目录：[README](/ros2/overview/)
