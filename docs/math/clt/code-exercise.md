---
title: "大数定律与中心极限 — exercise.py"
---

# 大数定律与中心极限 — 练习

<a href="/notebook/code/math/clt/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `standardized_mean(x, mu, sigma)`：

$$
Z=\frac{\sqrt{n}\,(\bar x-\mu)}{\sigma}
$$

`n = x.size`。测例：均值正好等于 $\mu$ 时应为 $0$；四个 $2$、$\mu=0$、$\sigma=2$ 时应为 $2$。

```bash
cd docs/math/clt/code
python exercise.py
```

## 源码位置

`docs/math/clt/code/exercise.py`
