---
title: "Python 基础"
order: 10
---
# Python 基础：脚本、容器，以及「类不是那台机器」

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

> 本笔记默认语言是 Python。读 [RSSM](/world-models/abstract/rssm/) 里 `self.gru = nn.GRUCell(...)` 之前，先分清：**左边是名字，等号右边是「用类造出来的对象」。** 下一章 [C++](/programming/cpp/) 给数学章的 `*.hpp`；张量见 [PyTorch](/programming/pytorch/)。

Python 不是「把英语句子缩进一下」。它管三件事：名字钉在哪个对象上、缩进决定谁属于谁、以及类只是图纸。本章把这三件事写到能对着 `nn.GRUCell` 那一行数清「谁是类、谁是实例、谁是返回值」。

## 一、怎么跑起来

本仓库每个演示都是**脚本**：一个 `.py` 文件从上到下执行。在章节的 `code/` 里：

```bash
cd docs/programming/python/code
python demo.py
```

文件头几乎总是：

```python
# -*- coding: utf-8 -*-
"""这一段是模块说明书，给人看的。"""
```

- `# -*- coding: utf-8 -*-`：告诉解释器源码是 UTF-8（中文注释才不会乱）。
- `if __name__ == '__main__':`：只有**直接** `python demo.py` 才跑主程序；被别人 `import` 时不跑。demo 里画图、打印都放这里。

本仓库约定：图写到章节自己的 `images/`，路径用脚本所在目录拼，避免你在别的文件夹启动时图丢了。

## 二、名字、对象、类型

Python 里一切都是对象。`a = 3` 是「名字 `a` 钉在整数对象 `3` 上」。

| 写法 | 类型 | 笔记里常见用途 |
|------|------|----------------|
| `3` / `0.1` | `int` / `float` | 超参、学习率 |
| `'hello'` | `str` | 路径、打印 |
| `True` | `bool` | 开关 |
| `None` | 空 | 「还没有值」 |
| `[1, 2]` | `list` | 变长序列，可改 |
| `(1, 2)` | `tuple` | 不可改，函数多返回值 |
| `{'lr': 1e-3}` | `dict` | 超参表、配置 |
| `{1, 2}` | `set` | 去重 |

**切片** `a[1:4]`：从下标 1 到 **不含** 4。`a[-1]` 是最后一个。`x[:, t]` 这种两维切片在 RSSM 里表示「所有 batch、第 t 步」。

**保姆级切片。** `a = [10, 20, 30, 40, 50]`：

| 写法 | 结果 | 为什么 |
|------|------|--------|
| `a[1:4]` | `[20, 30, 40]` | 含 1 不含 4 |
| `a[:2]` | `[10, 20]` | 从头到不含 2 |
| `a[-2:]` | `[40, 50]` | 倒数第二到末尾 |
| `a[::2]` | `[10, 30, 50]` | 步长 2 |

张量同理：`x[0]` 是第一个 batch；`x[:, -1, :]` 是每个序列的最后一步。

**列表推导**：`[f(x) for x in xs if cond]`，比手写 `for` + `append` 短，后面 demo 会用。

::: details 逐步说明：`a = 3` 之后再 `b = a` 钉的是谁（点击展开）

整数 `3` 是堆上的一个对象。`a = 3` 让名字 `a` 指向它。`b = a` 再钉一个名字到**同一个**对象，不是复制出另一个 3。`a = 4` 只是把 `a` 改钉到对象 `4`，`b` 仍指向 `3`。

可变对象就危险了：`xs = [1]`，`ys = xs`，`ys.append(2)` 之后 `xs` 也是 `[1, 2]`。函数默认参数 `def f(xs=[])` 的 `[]` 只造一次，多次调用会往同一个列表里塞——所以要用 `None` 再在函数里新建。

`if __name__ == '__main__'`：导入时 `__name__` 是模块名字符串，不是 `'__main__'`，主程序不跑。直接 `python demo.py` 时才是 `'__main__'`。这就是「别人 import 你的 `forward` 函数时，不会弹出 matplotlib 窗口」。

:::

## 三、函数

```python
def add(x: float, y: float = 0.0) -> float:
    return x + y
```

- `def` 定义；`return` 交出结果。没有 `return` 时得到 `None`。
- `y: float = 0.0`：类型标注（给人看的）+ 默认参数。
- `*args` / `**kwargs`：可变位置参数 / 关键字参数。`nn.Sequential(*layers)` 就是把列表拆开传进去。

## 四、类是图纸，实例才是机器 {#class-vs-instance}

这是读 PyTorch 最关键的一句。

```python
class Greeter:
    def __init__(self, name):   # 造对象时自动调用
        self.name = name        # 每个实例自己的数据

    def hello(self):
        return f'你好, {self.name}'
```

- `class Greeter:`：在**定义类**（图纸）。
- `g = Greeter('世界')`：在**构造实例**（按图纸造一台机器）。
- `g.hello()`：调用这台机器上的方法。`self` 就是这台机器自己。

`nn.Linear`、`nn.GRUCell` 都是**别人已经写好的类**。

```python
self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)
```

- `nn.GRUCell`：类（图纸），在 PyTorch 源码里。
- `nn.GRUCell(...)`：调用构造函数，**造一颗细胞对象**（分配门的权重）。
- `self.gru = ...`：把这颗对象存进 `RSSM` 这台更大的机器里。
- **`self.gru` 不是 \(h_t\)**。\(h_t\) 是你稍后 `self.gru(输入, h_prev)` **调用**时返回的张量。

一张图：左边图纸可以复印很多台；每台有自己的权重。五个时间步是**同一台** `self.gru` 被调用五次，不是五个类。

::: details 逐步说明：`self.gru = nn.GRUCell(...)` 里四个名字各是什么（点击展开）

1. `nn` 是模块（`import torch.nn as nn`），里面装着许多**类**。
2. `nn.GRUCell` 是类，还没有权重。你可以复印很多台：`g1 = nn.GRUCell(4, 8)` 和 `g2 = nn.GRUCell(4, 8)` 是两台机器，权重不共享。
3. `nn.GRUCell(stoch_dim + act_dim, deter_dim)` 调用 `__init__`，按图纸分配门的矩阵，得到**一个实例**。
4. `self.gru = ...` 把这台机器存进外层 `RSSM` 实例。以后 `self.gru(x, h)` 才是一次前向，返回新的隐藏张量。

时间维：`for t in range(T): h = self.gru(x_t, h)` 是**同一台**细胞吃 T 次。不要写成 `self.gru_t = nn.GRUCell(...)` 循环 T 次——那会造 T 套权重，参数量翻 T 倍，也学不到「同一种动力学」。

`self` 是当前实例。`RSSM.__init__` 里写 `self.gru`，等于往这台 RSSM 的抽屉里放子模块。`nn.Module` 会扫描赋给 `self` 的子模块，收进 `parameters()`。若写成局部变量 `gru = nn.GRUCell(...)` 而不赋给 `self`，优化器找不到这些权重。

:::

![类是图纸，实例是机器](./images/prog-python-class.png)

> **怎么读**：上：`class` 定义一次。中：`GRUCell(...)` 造出带权重的实例。下：`gru(x, h)` 才算出新的 \(h_t\)。

`self` 出现在方法第一个参数：谁调用，`self` 就是谁。`RSSM` 继承 `nn.Module` 之后，赋给 `self.xxx` 的模块会被自动登记进 `parameters()`，Adam 才能找到权重。

## 五、本仓库还会用到的几件事

- **缩进就是语法**：同一层必须对齐。没有 `{}`。
- **可变默认参数是坑**：不要写 `def f(xs=[])`，用 `None` 再在函数里 `xs = []`。
- **NumPy**：`import numpy as np`。`np.array`、`@` 矩阵乘、`axis`。数学章大量用。
- **中文图**：`matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']`，否则标签变方框。
- **随机种子**：`np.random.seed(42)` / `torch.manual_seed(42)`，改代码时图还对得上。

下一章若要编译数学章的 C++： [C++ 基础](/programming/cpp/)。只跑 Python / PyTorch 可直接去 [张量](/programming/pytorch/)。

## 📥 Code

| File | View | Download |
|------|------|----------|
| demo.py | [Open](./code-demo) | <a href="/notebook/code/programming/python/demo.py" target="_blank" download>Download</a> |
| exercise.py | [Open](./code-exercise) | <a href="/notebook/code/programming/python/exercise.py" target="_blank" download>Download</a> |
