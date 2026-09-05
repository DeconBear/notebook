---
title: "量子信息全景 — exercise.py"
---

# 量子信息全景 — 练习

<a href="/notebook/code/quantum/overview/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/overview/code
python exercise.py
```

## 任务

实现 `purity(rho) = Tr(ρ²)`。纯态为 1，单比特完全混合为 $1/2$。

参考实现：

```python
def purity(rho):
    return float(np.real(np.trace(rho @ rho)))
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/overview/code/exercise.py`
