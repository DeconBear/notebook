---
title: "传递函数与时域响应 — exercise.py"
---

# 传递函数与时域响应 — 练习

<a href="/notebook/code/control/classical/transfer/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `damping_regime(zeta)`，按阻尼比返回字符串：

| 条件 | 返回值 |
|------|--------|
| $\zeta>1$ | `'over'` |
| $\zeta=1$（允许 $10^{-9}$ 误差） | `'critical'` |
| $0<\zeta<1$ | `'under'` |
| $\zeta=0$ | `'undamped'` |
| $\zeta<0$ | `'unstable'` |

```bash
cd docs/control/classical/transfer/code
python exercise.py
```

## 源码位置

clone 后打开（相对仓库根目录）：

`docs/control/classical/transfer/code/exercise.py`
