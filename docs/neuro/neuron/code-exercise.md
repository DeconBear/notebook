---
title: "神经元与突触 — exercise.py"
---

# 神经元与突触 — 练习

<a href="/notebook/code/neuro/neuron/exercise.py" target="_blank" download>Download exercise.py</a>

## 运行方式

```bash
cd docs/neuro/neuron/code
python exercise.py
```

## 要做什么

实现指数 EPSP：`t >= t0` 时为 `amp * exp(-(t-t0)/tau)`，否则为 0。

## 完整代码

<<< @/neuro/neuron/code/exercise.py
