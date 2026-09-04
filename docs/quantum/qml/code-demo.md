---
title: "量子机器学习 — demo.py"
---

# 量子机器学习 — demo.py 代码详解

<a href="/notebook/code/quantum/qml/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/qml/code
python demo.py
```

未安装 `pyvqnet` 时：经典小 CNN 仍会训练并出图，量子部分打印安装说明后跳过（退出码 0）。

已安装时：额外对 VQNet 模型做一次前向冒烟。完整 24 epoch 训练：

```bash
python train.py
python eval.py
```

数据在 `dataset/*.npz`（16×16，数字 3 vs 6），迁自 [qml-mnist-classify](https://github.com/DeconBear/qml-mnist-classify)。

## 完整代码

<<< @/quantum/qml/code/demo.py
