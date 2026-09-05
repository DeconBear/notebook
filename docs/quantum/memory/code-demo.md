---
title: "量子存储 — demo.py"
---

# 量子存储 — demo.py 代码详解

<a href="/notebook/code/quantum/memory/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/quantum/memory/code
python demo.py
```

## 在讲什么

单比特玩具信道：

- 振幅阻尼模拟 $T_1$（ $|1\rangle$ 布居指数掉）；
- 非对角元衰减模拟 $T_2$；
- 两者叠在「写-存-读」上，读出保真度随等待时间下降。

不是某个实验室的参数拟合，只对应正文的两条钟。


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/quantum/memory/code/demo.py`
