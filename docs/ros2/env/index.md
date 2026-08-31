---
title: "00 环境与工作区"
order: 10
---
# 00 · 环境与工作区

## 目标

搞清楚：ROS 装在哪、工作区是什么、为什么每次都要 `source`。

## 概念

### 发行版（distro）

本机是 **Humble**。不同 Ubuntu 对应不同发行版（22.04 → Humble）。

```bash
echo $ROS_DISTRO   # 应输出 humble
```

### 系统安装 vs 工作区

| 路径 | 是什么 |
|------|--------|
| `/opt/ros/humble/` | apt 装好的系统 ROS |
| `$ROS_WS/` | 你自己的工作区（写包、编译） |

### 工作区目录

```text
notebook/                          # 本仓库根
  docs/ros2/                       # 课程笔记（本站点）
  workspaces/ros2-humble/          # $ROS_WS
    src/          # 源码（包放这里）
    build/        # 编译中间文件（可删，会再生）
    install/      # 编译安装结果（运行前要 source）
    log/          # 编译日志
    scripts/      # check_env.sh
```

### 为什么要 source 两次

```bash
source /opt/ros/humble/setup.bash          # 系统 ROS：有 ros2 命令
source $ROS_WS/install/setup.bash    # 工作区：能找到你自己的包
```

顺序：先系统，后工作区。只 source 系统 → 跑不了 `py_pubsub`；只 source 工作区 → 可能缺底层依赖。

> 可选：把 `source /opt/ros/humble/setup.bash` 写入 `~/.bashrc`，新终端自动加载。工作区则在每次 `colcon build` 后执行 `source "$ROS_WS/install/setup.bash"`，或自行决定是否写入 bashrc。

## 通俗理解

把 ROS 想成两层：

- **系统层**（`/opt/ros/humble`）：别人装好的“操作系统插件”，提供 `ros2` 命令和官方库。
- **工作区**（`$ROS_WS`）：你自己的项目文件夹。源码在 `src/`，编译结果在 `install/`。

`source` 的作用很像“把某套工具的路径临时加进当前终端”。不 source，终端就找不到 `ros2` 或你自己的包。

**改代码后为什么要重新 build？**  
因为运行时用的是 `install/` 里的安装结果，不是直接读 `src/`。改完 → 编译 → 再 source，三步才对齐。

## 常见疑问

**Q：ROS 2 要不要先学很多网络原理？**  
入门阶段不用。先当“消息总线”用即可。多机连不上、丢包时再补 DDS/网络。

**Q：能不能一上来就学架构？**  
可以浏览，但别当主线。先会写节点、会 source/build，再回头看 rmw/DDS 会轻松很多。

**Q：bashrc 里已经 source 了，为什么有时还是找不到自己的包？**  
系统 Humble 已自动加载；工作区那行默认是注释的。新编译后要手动 `source $ROS_WS/install/setup.bash`，或取消 bashrc 里对应注释。

**Q：终端里有 conda 会怎样？**  
可能抢走系统 Python，导致自定义接口包编译失败。编译接口相关包时建议：`export PATH="/usr/bin:$PATH"`。

## 动手

```bash
# 确认环境
echo $ROS_DISTRO
ros2 --help | head

# 编译工作区里的包
cd $ROS_WS
colcon build --packages-select py_pubsub
source install/setup.bash

# 看包里有哪些可执行文件
ros2 pkg executables py_pubsub
```

## 关键命令

```bash
ros2 pkg list | grep py_pubsub
ros2 pkg executables py_pubsub
colcon build --packages-select py_pubsub
```

## 验证标准（给学员与 AI）

- [ ] `echo $ROS_DISTRO` 输出 `humble`（需已 source）
- [ ] `ros2 --help` 可用
- [ ] 在 `$ROS_WS` 下 `colcon build`（或至少能理解 src/build/install）成功或已有 install
- [ ] 知道：改代码后要 build，再 `source "$ROS_WS/install/setup.bash"`

失败时：先 `source /opt/ros/humble/setup.bash`；接口包编译若失败，避开 conda Python。

## 小结

- Humble 在 `/opt/ros/humble`，你的代码在 `$ROS_WS/src`。
- **改代码 → build → source install**，三步缺一不可。
- 下一课：[01 · Topic 发布/订阅](/ros2/topics/)
