---
title: "特征值与二次型 — exercise.py"
---

# 特征值与二次型 — 练习

<a href="/notebook/code/math/eigen/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `power_iteration(A, n_iter=20, v0=None)`，返回 `(lam, v)`。`v` 必须是单位向量。与 `np.linalg.eigh` 的最大特征对比，允许 $v$ 差一个符号（用绝对值内积 $>0.999$ 检查）。

```bash
cd docs/math/eigen/code
python exercise.py
```

## 源码位置

`docs/math/eigen/code/exercise.py`
