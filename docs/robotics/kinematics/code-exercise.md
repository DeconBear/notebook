---
title: "机器人运动学 — exercise.py"
---

# 机器人运动学 — 练习

<a href="/notebook/code/robotics/kinematics/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `fk_x(th1, th2)`：返回平面 2R 末端的 $x$ 坐标 $\ell_1\cos\theta_1+\ell_2\cos(\theta_1+\theta_2)$。杆长与 demo 相同（`L1, L2 = 1.0, 0.7`）；$\theta=0$ 时应得到 $\ell_1+\ell_2$。

```bash
cd docs/robotics/kinematics/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/kinematics/code/exercise.py`
