---
title: "量子计算 — exercise.py"
---

# 量子计算 — 练习

<a href="/notebook/code/quantum/computing/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/computing/code
python exercise.py
```

## 任务

1. `apply_gate(gate, state)`：返回 `gate @ state`。
2. `bell_state()`：`CNOT @ kron(H, I) @ |00⟩`。

参考实现：

```python
def apply_gate(gate, state):
    return gate @ state

def bell_state():
    zero = np.array([1, 0, 0, 0], dtype=complex)
    return CNOT @ np.kron(H, I2) @ zero
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/computing/code/exercise.py`
