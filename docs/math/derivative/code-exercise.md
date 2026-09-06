---
title: "导数与微分 — exercise.py"
---

# 导数与微分 — 练习

<a href="/notebook/code/math/derivative/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `central_diff(f, x, h=1e-6)`：

$$
\frac{f(x+h)-f(x-h)}{2h}
$$

对 $t^3-2t$ 在 $2$ 处应得到 $10$（误差小于 $10^{-6}$）。不要写成单侧差分，否则过不了断言。

```bash
cd docs/math/derivative/code
python exercise.py
```

## 源码位置

`docs/math/derivative/code/exercise.py`
