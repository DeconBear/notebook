---
title: "LQR 与最优控制 — exercise.py"
---

# LQR 与最优控制 — 练习

<a href="/notebook/code/control/modern/lqr/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `lqr_u(K, x)`：返回标量 $-K x$。`K` 是 shape `(1, 2)` 的行向量，`x` 是长度为 2 的状态；自测用 $K=[2,0.5]$、$x=[1,-4]$，结果应为 $0$。

```bash
cd docs/control/modern/lqr/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/lqr/code/exercise.py`
