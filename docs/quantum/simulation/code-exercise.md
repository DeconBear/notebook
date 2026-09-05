---
title: "量子模拟 — exercise.py"
---

# 量子模拟 — 练习

<a href="/notebook/code/quantum/simulation/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/simulation/code
python exercise.py
```

## 任务

把 $H=J Z\otimes Z + h(X\otimes I+I\otimes X)$ 拆成 `(A, B)`。

```python
def split_hamiltonian(j=1.0, h=0.7):
    a = j * np.kron(Z, Z)
    b = h * (np.kron(X, I2) + np.kron(I2, X))
    return a, b
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/simulation/code/exercise.py`
