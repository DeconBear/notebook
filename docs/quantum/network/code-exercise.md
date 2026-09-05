---
title: "量子网络 — exercise.py"
---

# 量子网络 — 练习

<a href="/notebook/code/quantum/network/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/network/code
python exercise.py
```

## 任务

实现纯态保真度 $F=|\langle a|b\rangle|^2$。

```python
def fidelity(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/network/code/exercise.py`
