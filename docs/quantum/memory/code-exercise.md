---
title: "量子存储 — exercise.py"
---

# 量子存储 — 练习

<a href="/notebook/code/quantum/memory/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/memory/code
python exercise.py
```

## 任务

初始 $|1\rangle$ 的 $T_1$ 布居：$\rho_{11}(t)=e^{-t/T_1}$。

```python
def population_1(t, t1=1.0):
    return float(np.exp(-t / t1))
```

## 完整代码

<<< @/quantum/memory/code/exercise.py
