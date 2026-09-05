---
title: "旋量代数 — exercise.py"
---

# 旋量代数 — 练习

<a href="/notebook/code/robotics/screw/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `planar_v(omega, qx, qy)`：纯转动线速度 $v=\omega(q_y,-q_x)$，与 demo 中 `v = omega * np.array([q[1], -q[0]])` 同一约定。自测 $\omega=2,q=(1,0)$ 应得 `[0, -2]`。

```bash
cd docs/robotics/screw/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/robotics/screw/code/exercise.py`
