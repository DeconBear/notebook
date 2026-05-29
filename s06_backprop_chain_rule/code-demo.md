---
title: "s06 反向传播与链式法则 — demo.py"
---

# s06 反向传播与链式法则 — demo.py

<a href="../code/s06_backprop_chain_rule/demo.py" target="_blank" download>Download demo.py</a>

## What This Code Does

从零实现一个微型自动微分引擎（mini autograd），模拟PyTorch的`.backward()`核心逻辑。每个`Value`节点记录data、grad、_backward和_prev，通过拓扑排序按逆序执行链式法则。包含5个演示：基本表达式求导、激活函数反向传播、Fan-out梯度累积、小神经网络训练、以及梯度下降求函数最小值。

## Run It

```bash
cd s06_backprop_chain_rule/code
python demo.py
```

## Key Output

- 演示1（基本表达式）：(a*b + c) * d，a=2,b=3,c=4,d=5，结果L=50；梯度dL/da=15, dL/db=10, dL/dc=5, dL/dd=10
- 演示2（激活函数）：ReLU正区导数=1.0，负区导数=0.0；Sigmoid/Tanh导数与公式一致
- 演示3（Fan-out梯度累积）：L=(2x)*(x+3)，x=2时dL/dx=14，展示多路径梯度自动累加
- 演示4和5：小神经网络训练和梯度下降求极值，验证框架正确性

<<< @/snippets/s06_backprop_chain_rule/demo.py
