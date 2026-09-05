---
title: "观测器与卡尔曼 — exercise.py"
---

# 观测器与卡尔曼 — 练习

<a href="/notebook/code/control/modern/kalman/exercise.py" target="_blank" download>Download exercise.py</a>

实现一维卡尔曼增益 `kgain(P, R) = P / (P + R)`。$P=0$ 时增益为 $0$（完全信模型）；$R=0$ 时增益为 $1$（完全信无噪测量）。

```bash
cd docs/control/modern/kalman/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/modern/kalman/code/exercise.py`
