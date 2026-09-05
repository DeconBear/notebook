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

未安装 `pyvqnet` 时：16×16 小 CNN 仍会训练并出图；若 torchvision 能下载 MNIST，还会跑 28×28 LeNet-5（0 vs 1）。量子部分打印安装说明后跳过（退出码 0）。

已安装时：额外对 VQNet 模型做一次前向冒烟。完整 24 epoch 训练：

```bash
python train.py
python eval.py
```

- 量子 / 同任务 CNN：`dataset/*.npz`（16×16，数字 3 vs 6）
- LeNet-5 对照：`lenet.py` 从 torchvision 筛 MNIST 的 0 与 1，**不使用 PNG 目录**


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/qml/code/demo.py`
