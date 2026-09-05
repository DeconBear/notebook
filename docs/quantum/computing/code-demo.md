---
title: "量子计算 — demo.py"
---

# 量子计算 — demo.py 代码详解

<a href="/notebook/code/quantum/computing/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/computing/code
python demo.py
```

## 在讲什么

NumPy 态矢量演示两件事：

1. $H|0\rangle$ 测量接近 50/50；
2. $H$ 再 CNOT 得到 Bell 态，直方图只剩 `00` 与 `11`。

门矩阵与正文 [量子计算](/quantum/computing/) 一致。没有用 Qiskit / PennyLane，方便默认环境。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/computing/code/demo.py`
