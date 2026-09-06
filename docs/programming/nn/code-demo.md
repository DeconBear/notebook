---
title: "torch.nn 模块怎么用 — demo.py"
---

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# torch.nn — demo.py 代码详解

<a href="/notebook/code/programming/nn/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/programming/nn/code
python demo.py
```

打印 `TinyMLP` 的 `named_parameters`；造一颗 `GRUCell` 走两步（对象 `id` 不变）；比较 CrossEntropy 吃 logit vs 错误地先 Softmax。图 `prog-nn-three.png`。

## 源码位置

`docs/programming/nn/code/demo.py`

## 和 RSSM 对一行

```python
self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)  # 造机器
h = self.gru(torch.cat([s, prev_action], dim=-1), h)    # 得到 h_t
```

`cat` 的最后一维必须等于 Cell 的 `input_size`。
