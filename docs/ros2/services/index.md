---
title: "02 Service 请求应答"
order: 30
---
# 02 · Service 请求/应答

## 目标

写并跑通一对服务：客户端发 `a`、`b`，服务端返回 `sum`。理解 **Service 与 Topic 的差别**。

## 概念

```text
  client  --request(a,b)-->  服务 /add_two_ints  --response(sum)-->  client
              等待结果              server 计算
```

| | Topic | Service |
|--|-------|---------|
| 模式 | 持续广播，一对多 | 一问一答，一对一 |
| 像什么 | 电台 | 函数调用 |
| 本课 | `chatter` | `add_two_ints` |
| 适合 | 传感器、状态 | 「帮我做一件事并给结果」 |

服务接口（`---` 上为请求，下为响应）：

```text
int64 a
int64 b
---
int64 sum
```

类型名：`example_interfaces/srv/AddTwoInts`。

## 通俗理解

Service 像**打电话问一件事并等回答**：

- 你（客户端）：“3+5 等于几？”
- 对方（服务端）：算完回你 “8”，这次通话结束。

和 Topic 的电台不同：Service 是**一问一答**，适合“帮我做一件短事并给结果”，例如开关灯、查状态、触发一次标定。

不适合：导航走完整个路径（太久）、激光每秒几十帧（太频繁）——那些分别更像 Action / Topic。

## 常见疑问

**Q：服务端没启动就调用会怎样？**  
客户端通常会 `wait_for_service` 一直等；命令行调用也会提示服务不可用。

**Q：Service 是同步阻塞的吗？**  
体感上像同步（发了等结果）。代码里常用 `call_async` + `spin_until_future_complete`，实现上是异步 future。

**Q：能一对多吗？**  
一个服务名通常由一个服务端提供；多个客户端可以轮流调用它。

**Q：和 Topic 怎么选？**  
要持续流数据 → Topic；要“做完告诉我结果”的短请求 → Service。

## 动手

```bash
cd $ROS_WS
source /opt/ros/humble/setup.bash
source install/setup.bash
```

终端 A（服务端一直挂着）：

```bash
ros2 run py_pubsub add_two_ints_server
```

终端 B（客户端算完就退出）：

```bash
ros2 run py_pubsub add_two_ints_client 3 5
# 期望：结果: 3 + 5 = 8
```

不写客户端也可以命令行调用：

```bash
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 7, b: 8}"
```

## 关键命令

```bash
ros2 service list
ros2 service type /add_two_ints
ros2 interface show example_interfaces/srv/AddTwoInts
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"
```

## 对照代码

| 文件 | 作用 |
|------|------|
| [`add_two_ints_server.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/add_two_ints_server.py) | 创建服务、算加法 |
| [`add_two_ints_client.py`](../../../workspaces/ros2-humble/src/py_pubsub/py_pubsub/add_two_ints_client.py) | 发请求、打印结果 |
| [`setup.py`](../../../workspaces/ros2-humble/src/py_pubsub/setup.py) | 注册两个可执行入口 |
| [`package.xml`](../../../workspaces/ros2-humble/src/py_pubsub/package.xml) | 依赖含 `example_interfaces` |

服务端关键几行：

```python
self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)
# 回调里：response.sum = request.a + request.b; return response
```

客户端关键几行：

```python
self.create_client(AddTwoInts, 'add_two_ints')
future = self.cli.call_async(req)
rclpy.spin_until_future_complete(self, future)
```

两边 **服务类型 + 服务名** 必须一致。客户端通常先 `wait_for_service`，避免服务端还没起来就调用失败。

## 小练习（可选）

1. 用命令行再算一次 `100 + 200`。
2. 先开客户端、后开服务端，观察「等待服务…」日志。
3. 改服务名（两边一起改），体会对不上时会发生什么。

## 验证标准（给学员与 AI）

- [ ] `ros2 run py_pubsub add_two_ints_server` 显示服务就绪
- [ ] `ros2 run py_pubsub add_two_ints_client 3 5` 得到 `8`
- [ ] 或 `ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts "{a: 1, b: 2}"` 成功

失败时：先起 server 再 client；检查服务名 `/add_two_ints`。

## 小结

- Service = 同步感的请求/应答（实现上常用异步 future）。
- 适合短任务；长时间、要进度反馈的用 Action。
- 上一课：[01 · Topic](/ros2/topics/) · 下一课：[03 · Action](/ros2/actions/) · 目录：[README](/ros2/overview/)
