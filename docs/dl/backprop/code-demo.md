---
title: "s06 反向传播与链式法则 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s06 反向传播与链式法则 — demo.py 代码详解

<a href="/notebook/code/dl/backprop/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd docs/dl/backprop/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import math
from typing import Set, List, Tuple
```

- **`os`**：操作系统接口库。此处用于创建 `images/` 输出目录，确保图片保存路径存在。
- **`math`**：数学函数库。提供 `math.exp()`（指数函数）、`math.tanh()`（双曲正切）等底层数学运算，用于实现 Sigmoid、Tanh 等激活函数。
- **`typing.Set, List, Tuple`**：类型注解。`Set` 用于记录计算图中已访问的节点，`List` 用于拓扑排序结果列表，`Tuple` 用于节点的前驱元组。

> **为什么不用 NumPy？** 因为这个 demo 的目标是展示自动微分的**底层原理**——每个数值都是标量 `Value`，而非 NumPy 数组。下一节 s07 才扩展到矩阵/向量级别的反向传播，届时才需要 NumPy。

---

### 第2步：Value 类 — 自动微分的核心节点

整个 demo 的基石是 `Value` 类。每个 `Value` 对象就是计算图中的一个节点，它存储四样东西：

| 属性 | 含义 | 用途 |
|------|------|------|
| `data` | 该节点的数值（标量） | 前向传播的结果值 |
| `grad` | 累积的梯度 $\frac{\partial L}{\partial \text{node}}$ | 反向传播时累加 |
| `_backward` | 局部反向传播函数（闭包） | 定义该操作对输入的梯度如何分配 |
| `_prev` | 前驱节点集合 | 用于拓扑排序，确定反向传播顺序 |

```python
class Value:
    def __init__(self, data: float, _children: Tuple = (), _op: str = ""):
        self.data = data                # 存储数值
        self.grad = 0.0                 # 梯度初始化为 0
        self._backward = lambda: None   # 默认无操作（叶子节点）
        self._prev = set(_children)     # 前驱节点集合
        self._op = _op                  # 操作名称（如 "+", "*", "ReLU"）
```

**设计要点**：
- `grad` 初始为 0，反传时通过 `+=` 累加而非 `=` 赋值——这是因为一个变量可能被多条路径使用（fan-out），梯度需要求和。
- `_backward` 是一个闭包（closure），它捕获了当前操作的局部上下文（如两个输入的 data 值），从而在反向时无需重新计算。
- `_children` 用 `set` 存储，去重一个节点被同一父节点多次引用的情况。

---

### 第3步：基本算术运算 — 局部梯度规则

每个运算 `__add__`、`__mul__` 等都实现了两个关键部分：**前向计算** 和 **局部反向传播闭包**。

#### 加法门 `__add__`

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += 1.0 * out.grad   # ∂(a+b)/∂a = 1
        other.grad += 1.0 * out.grad  # ∂(a+b)/∂b = 1

    out._backward = _backward
    return out
```

数学依据：加法门是**梯度分发器**。

$$
\frac{\partial (a + b)}{\partial a} = 1, \quad \frac{\partial (a + b)}{\partial b} = 1
$$

因此上游梯度 `out.grad`（即 $\frac{\partial L}{\partial out}$）原样传递给两个输入。链式法则：$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial out} \cdot \frac{\partial out}{\partial a} = out.grad \cdot 1$。

#### 乘法门 `__mul__`

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad   # ∂(a*b)/∂a = b
        other.grad += self.data * out.grad   # ∂(a*b)/∂b = a

    out._backward = _backward
    return out
```

数学依据：乘法门的梯度是**交换的**（gradient switcheroo）。

$$
\frac{\partial (a \cdot b)}{\partial a} = b, \quad \frac{\partial (a \cdot b)}{\partial b} = a
$$

注意：闭包中捕获的是 `other.data` 和 `self.data` 的值（前向时的值），而非反向时的值。这是自动微分的标准做法——前向传播时存储中间值，反向传播时使用。

#### 幂运算 `__pow__`

```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "仅支持数值指数"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data ** (other - 1)) * out.grad

    out._backward = _backward
    return out
```

数学公式：

$$
\frac{\partial (x^n)}{\partial x} = n \cdot x^{n-1}
$$

比如 $x^2$ 的导数是 $2x$。幂运算只有 `self` 一个输入参数（`_children` 中只有 `self`），所以反向传播时只有一个梯度接收者。

#### 除法 `__truediv__`：巧妙的复用法

```python
def __truediv__(self, other):
    return self * other ** -1
```

这里没有单独实现除法的反向传播，而是利用了**已有的乘法和幂运算的组合**：$a / b = a \times b^{-1}$。这是工程上非常聪明的做法——由于 `__mul__` 和 `__pow__` 已经分别定义了正确的 `_backward`，除法门的梯度会通过链式法则自动正确。

数学验证：

$$
\frac{\partial (a / b)}{\partial a} = \frac{1}{b}, \quad \frac{\partial (a / b)}{\partial b} = -\frac{a}{b^2}
$$

当 `b ** -1` 的反向传播与 `a * (b**-1)` 的反向传播组合时，链式法则会自动产出正确结果。

---

### 第4步：激活函数 — 非线性变换的反向传播

激活函数是神经网络非线性的来源。每个激活函数需要定义**前向计算**和**导数公式**。

#### ReLU：梯度门控

```python
def relu(self):
    out = Value(max(0.0, self.data), (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out
```

数学定义与导数：

$$
\text{ReLU}(x) = \max(0, x)
$$

$$
\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}
$$

**为什么此处用它？** ReLU 是隐藏层最常用的激活函数。它的导数非0即1，没有饱和区（不像 Sigmoid/Tanh 在两端导数趋近0），因此能有效缓解梯度消失问题。

**代码细节**：`(out.data > 0)` 在 Python 中产生布尔值 True/False，但在算术运算中 True=1、False=0。因此整个表达式相当于"如果前向值 > 0，梯度通过；否则截断为0"。这是一个优雅的**梯度门控**（gating）实现。

#### Sigmoid：数值稳定技巧

```python
def sigmoid(self):
    x = self.data
    if x >= 0:
        s = 1.0 / (1.0 + math.exp(-x))      # 标准公式
    else:
        exp_x = math.exp(x)
        s = exp_x / (1.0 + exp_x)            # 数值稳定版本

    out = Value(s, (self,), 'Sigmoid')

    def _backward():
        self.grad += out.data * (1 - out.data) * out.grad

    out._backward = _backward
    return out
```

数学定义与导数：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

Sigmoid 导数的优美性质是**可以用前向输出直接计算导数**，无需知道原始输入 $x$。这在工程上很方便——反向传播时直接用 `out.data` 即可。

**数值稳定技巧**：当 $x$ 是非常大的负数时，$e^{-x}$ 可能溢出（如 $x=-500$，$e^{500}$ 超出浮点数上限）。因此当 $x < 0$ 时，改用等效公式 $\sigma(x) = \frac{e^x}{1 + e^x}$，避免了 $e^{-x}$ 的计算。

#### Tanh：零中心输出

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), 'Tanh')

    def _backward():
        self.grad += (1 - out.data ** 2) * out.grad

    out._backward = _backward
    return out
```

数学定义与导数：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

与 Sigmoid 类似，Tanh 的导数也可以用前向输出直接计算。Tanh 的输出范围是 $(-1, 1)$，以零为中心（不像 Sigmoid 输出全正），因此在隐藏层有时优于 Sigmoid。

#### Exp：导数等于自身

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad

    out._backward = _backward
    return out
```

数学公式：$\frac{d}{dx}e^x = e^x$。Exp 函数的导数就是它自己的值——这是指数函数最独特的性质，也是 softmax 中常用的基本操作。

---

### 第5步：backward() — 拓扑排序 + 逆序执行

这是整个自动微分引擎的核心。`backward()` 方法的职责是：从当前节点（通常是损失 $L$）出发，按正确的顺序调用计算图中每个节点的 `_backward()`。

```python
def backward(self):
    # ---- 步骤 1: 拓扑排序（DFS 实现） ----
    topo = []
    visited = set()

    def build_topo(v: Value):
        if v not in visited:
            visited.add(v)
            for child in v._prev:       # 递归访问所有前驱
                build_topo(child)
            topo.append(v)              # 后序遍历：子节点在前

    build_topo(self)

    # ---- 步骤 2: 初始化梯度 ----
    self.grad = 1.0   # ∂L/∂L = 1

    # ---- 步骤 3: 按拓扑逆序调用 backward ----
    for node in reversed(topo):
        node._backward()
```

**为什么需要拓扑排序？**

反向传播要求**按计算图的逆序**执行：先计算离输出近的节点的梯度，再计算离输入近的。拓扑排序确保了：当调用节点 $v$ 的 `_backward()` 时，它的所有后继（路径上更靠近输出的节点）的梯度已经计算完毕。

具体来说，`build_topo` 使用 **DFS 后序遍历**：
1. 从根节点（损失 $L$）出发，沿 `_prev`（前驱关系）反向遍历
2. 后序遍历确保：子节点（离输入近的）先被 `topo.append()`，父节点后
3. 最终 `reversed(topo)` 得到的顺序就是：从 $L$ 出发，逐层向输入传播

**初始梯度为什么是 1.0？**

$$
\frac{\partial L}{\partial L} = 1
$$

损失对自身的导数恒为 1。这是整个反向传播中唯一"手动赋值"的梯度——其余所有梯度都通过链式法则和局部闭包自动计算。

---

### 第6步：演示1 — 基本表达式的反向传播

表达式：$L = (a \times b + c) \times d$

给定：$a=2, b=3, c=4, d=5$

```python
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
d = Value(5.0)

e = a * b        # e = 6
f = e + c        # f = 10
L = f * d        # L = 50
```

前向传播按顺序构建计算图：$a \xrightarrow{\times b} e \xrightarrow{+ c} f \xrightarrow{\times d} L$。

调用 `L.backward()` 后，代码按以下顺序自动计算梯度：
1. $\frac{\partial L}{\partial L} = 1$
2. $f$（乘法门 $L = f \times d$）：$\frac{\partial L}{\partial f} = d \cdot 1 = 5$，$\frac{\partial L}{\partial d} = f \cdot 1 = 10$
3. $e$（加法门 $f = e + c$）：$\frac{\partial L}{\partial e} = 1 \cdot \frac{\partial L}{\partial f} = 5$，$\frac{\partial L}{\partial c} = 5$
4. $a$（乘法门 $e = a \times b$）：$\frac{\partial L}{\partial a} = b \cdot \frac{\partial L}{\partial e} = 3 \times 5 = 15$，$\frac{\partial L}{\partial b} = a \cdot 5 = 2 \times 5 = 10$

验证：

$$
\frac{\partial L}{\partial a} = d \cdot b = 5 \times 3 = 15 \\
\frac{\partial L}{\partial b} = d \cdot a = 5 \times 2 = 10 \\
\frac{\partial L}{\partial c} = d \cdot 1 = 5 \\
\frac{\partial L}{\partial d} = a \cdot b + c = 10
$$

---

### 第7步：演示2 — 激活函数的反向传播验证

这段代码选了 $x = 1.5$ 作为测试点，依次计算三种激活函数的前向值和梯度：

- **ReLU**：$x=1.5 > 0$，所以 $\text{ReLU}(1.5) = 1.5$，梯度 $= 1.0$
- **Sigmoid**：$\sigma(1.5) \approx 0.8176$，梯度 $\sigma(1.5)(1 - \sigma(1.5)) \approx 0.1491$
- **Tanh**：$\tanh(1.5) \approx 0.9051$，梯度 $1 - 0.9051^2 \approx 0.1807$

另外测试了负输入 $x = -1.5$ 时的 ReLU：$\text{ReLU}(-1.5) = 0$，梯度 $= 0$（梯度被截断）。

**关键代码模式**：每次测试前调用 `x.zero_grad()` 清零梯度，否则前一次测试的梯度会残留并与本次累加，导致结果错误。这个模式与 PyTorch 的 `optimizer.zero_grad()` 完全对应。

---

### 第8步：演示3 — Fan-out 梯度累积

表达式：$L = (2x) \times (x + 3)$，取 $x = 2$

```python
x = Value(2.0)
u = x * 2      # 路径1: x → u
v = x + 3      # 路径2: x → v
L = u * v      # 两条路径汇合
```

$x$ 同时影响了 $u$（通过 $\times 2$）和 $v$（通过 $+ 3$），两个影响最终在 $L$ 处汇合。根据多元微积分中的全导数公式：

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial x}
$$

代入数值：
- $\frac{\partial L}{\partial u} = v = 5$，$\frac{\partial u}{\partial x} = 2$，路径1贡献 $= 5 \times 2 = 10$
- $\frac{\partial L}{\partial v} = u = 4$，$\frac{\partial v}{\partial x} = 1$，路径2贡献 $= 4 \times 1 = 4$

总梯度 $= 10 + 4 = 14$。

**这是 `grad += ...` 而非 `grad = ...` 的直接原因**——同一个变量出现在计算图的多条路径上时，每条路径的反向传播都会向该变量的 `grad` 中贡献一份，最终 `grad` 是所有路径贡献的**和**。

---

### 第9步：演示4 — 小神经网络完整训练

该演示构建了一个 2 输入 → 4 隐藏(ReLU) → 1 输出(线性) 的 MLP，在 4 个合成样本上训练 100 轮。

**训练数据**：目标函数 $y = 3x_1^2 - 2x_2 + 1$，用 4 个 $(x_1, x_2)$ 对生成标签。

```python
model = MLP(2, [4, 1], ["relu", "linear"])
```

**训练循环的三步曲**：

1. **前向传播** (`model(x)`)：输入经过 `Layer` → `Neuron` 的计算链，最终得到预测值。
2. **计算损失**：MSE 损失 $L = \sum (y_{pred} - y_{true})^2$（未除以 $N$，但不影响优化方向）。
3. **反向传播 + 更新**：
   ```python
   for p in model.parameters():
       p.grad = 0.0           # 清零梯度（避免跨 batch 累加）
   total_loss.backward()       # 反向传播，计算所有参数的梯度
   for p in model.parameters():
       p.data -= learning_rate * p.grad   # 梯度下降：θ := θ - α·∇L
   ```

**Neuron 类的权重初始化**使用了 He 初始化：

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$

其中 $n_{in}$ 是输入维度。He 初始化专门配合 ReLU 激活函数，能有效缓解训练初期的梯度消失。

**Layer 类**：将 `nin` 个输入连接到 `nout` 个 Neuron，前向传播时每个 Neuron 独立计算，输出组合成一个列表。

**MLP 类**：按顺序堆叠多个 Layer，最终取输出层第一个神经元的输出（对于回归任务）。

---

### 第10步：演示5 — 梯度下降求函数最小值

函数：$f(x) = x^2 + 3x$

解析解：$f'(x) = 2x + 3 = 0 \Rightarrow x = -1.5$，最小值 $f(-1.5) = -2.25$

从 $x = 5$ 开始，学习率 $\alpha = 0.1$，迭代 20 步：

```python
x = Value(5.0)
for step in range(20):
    loss = x * x + Value(3.0) * x   # 构造 f(x) = x² + 3x
    x.grad = 0.0                     # 清零梯度
    loss.backward()                  # 反向传播，x.grad = 2x + 3
    x.data -= 0.1 * x.grad          # 梯度下降更新
```

每次迭代中，`loss.backward()` 自动计算 $\frac{\partial f}{\partial x} = 2x + 3$，然后用梯度下降更新 $x$。

这个演示展示了自动微分在**优化问题**中的应用——不需要手动求导，只需用 Value 对象构造目标函数，`backward()` 自动给出梯度。

---

### 辅助组件：计算图可视化

`print_computation_graph(L)` 将计算图以文本表格形式打印：节点深度、操作类型、数据值、梯度、输入节点。这帮助直观地看到：
- 每个节点的前驱是谁（计算依赖关系）
- 前向值和梯度的大小
- 梯度从输出端向输入端递减的规律

---

### 关键概念速查表

| 概念 | 数学公式 | 代码实现 |
|------|---------|---------|
| 链式法则 | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$ | `self.grad += local_deriv * out.grad` |
| 加法门 | $\frac{\partial (a+b)}{\partial a} = 1$ | `self.grad += 1.0 * out.grad` |
| 乘法门 | $\frac{\partial (a \cdot b)}{\partial a} = b$ | `self.grad += other.data * out.grad`（梯度交换） |
| 幂运算 | $\frac{\partial x^n}{\partial x} = n x^{n-1}$ | `self.grad += (other * self.data ** (other-1)) * out.grad` |
| ReLU | $\text{ReLU}'(x) = \mathbb{1}[x > 0]$ | `self.grad += (out.data > 0) * out.grad` |
| Sigmoid | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ | `self.grad += out.data * (1 - out.data) * out.grad` |
| Tanh | $\tanh'(x) = 1 - \tanh^2(x)$ | `self.grad += (1 - out.data**2) * out.grad` |
| 拓扑排序（DFS） | 后序遍历计算图 | `build_topo(v)` 递归 + `reversed(topo)` 逆序 |
| 梯度累积（Fan-out） | $\frac{\partial L}{\partial h} = \sum_i \frac{\partial L}{\partial u_i} \cdot \frac{\partial u_i}{\partial h}$ | `self.grad += ...`（用 `+=` 而非 `=`） |
| 梯度清零 | — | `zero_grad()` / `p.grad = 0.0` |
| 梯度下降 | $\theta_{t+1} = \theta_t - \alpha \cdot \nabla_\theta L$ | `p.data -= lr * p.grad` |

## 完整代码

<<< @/dl/backprop/code/demo.py
