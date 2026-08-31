# ROS 2 Humble 工作区

本目录是 notebook 里 ROS 2 课程的 **colcon 工作区**（源自 [ros2-humble-notes](https://github.com/DeconBear/ros2-humble-notes)）。笔记在站点 `/ros2/`，源码在本目录 `src/`。

`$ROS_WS` 就是本文件夹（相对仓库根：`workspaces/ros2-humble`）。

```bash
cd workspaces/ros2-humble
export ROS_WS="$PWD"
export PATH="/usr/bin:$PATH"
bash scripts/check_env.sh
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source "$ROS_WS/install/setup.bash"
```

许可证：Apache-2.0，见 [LICENSE](LICENSE)。
