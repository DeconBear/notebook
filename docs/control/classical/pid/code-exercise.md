---
title: "PID 控制 — exercise.py"
---

# PID 控制 — 练习

<a href="/notebook/code/control/classical/pid/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `pid_output(e, integ, de, kp, ki, kd) = kp*e + ki*integ + kd*de`。给定误差、积分累加、微分与三个增益，返回控制量；不要在函数里再做积分或差分。

```bash
cd docs/control/classical/pid/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/pid/code/exercise.py`
