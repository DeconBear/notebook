---
title: "线性方程组与秩 — exercise.py"
---

# 线性方程组与秩 — 练习

<a href="/notebook/code/math/systems/exercise.py" target="_blank" download>Download exercise.py</a>

实现 `numerical_rank(A, tol=1e-8)`：`np.linalg.svd(A, compute_uv=False)`，数大于 `tol` 的奇异值个数。第二行是第一行两倍的 $3\times 3$ 应变 `2`，`eye(3)` 变 `3`，全零变 `0`。

```bash
cd docs/math/systems/code
python exercise.py
```

## 源码位置

`docs/math/systems/code/exercise.py`
