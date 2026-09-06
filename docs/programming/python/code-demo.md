---
title: "Python 基础 — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# Python 基础 — demo.py 代码详解

<a href="/notebook/code/programming/python/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/programming/python/code
python demo.py
```

会打印切片 / 推导，并用一个假的 `GRULike` 演示：**同一个 cell 对象调用三次，变的是 `h`，不是类。** 图 `prog-python-class.png` 写到本章 `images/`。

## 源码位置

`docs/programming/python/code/demo.py`

## 对照 RSSM

`GRULike.__call__(x, h_prev)` 对应 `h = self.gru(cat([s,a]), h)`。`__init__` 里的 `scale` 对应 GRU 的权重：造对象时定下来，前向不把它换成新的隐状态。
