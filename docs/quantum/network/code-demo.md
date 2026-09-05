---
title: "量子网络 — demo.py"
---

# 量子网络 — demo.py 代码详解

<a href="/notebook/code/quantum/network/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/network/code
python demo.py
```

## 在讲什么

1. **传态**：随机纯态经 Bell 测量与 Pauli 修正，无噪声时保真度应贴近 1。
2. **BB84 玩具**：无窃听筛后误码接近 0；拦截-重发会明显抬高误码。

实现用显式 3 比特态矢量，便于对照正文线路，而不是调用黑盒库。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/network/code/demo.py`
