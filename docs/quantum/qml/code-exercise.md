---
title: "量子机器学习 — exercise.py"
---

# 量子机器学习 — 练习

<a href="/notebook/code/quantum/qml/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/quantum/qml/code
python exercise.py
```

## 任务

实现原仓角度头：`sigmoid(hidden @ W.T + b) * 2π`。不依赖 VQNet。

```python
def angle_from_hidden(hidden, weight, bias):
    linear = hidden @ weight.T + bias
    return (1.0 / (1.0 + np.exp(-linear))) * (2.0 * np.pi)
```


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/qml/code/exercise.py`
