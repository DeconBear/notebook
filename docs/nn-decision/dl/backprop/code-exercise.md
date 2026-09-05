---
title: "s06 反向传播与链式法则 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s06 反向传播与链式法则 — exercise.py 练习指南

<a href="/notebook/code/nn-decision/dl/backprop/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

亲手补全一个微型自动微分引擎（mini autograd）的核心组件——通过实现 Tanh 的反向传播、除法操作的反向传播、以及拓扑排序驱动的 `backward()` 方法，深入理解 PyTorch/autograd 底层的工作原理。

## 预备知识

在开始练习前，确保你已理解以下概念（建议先阅读 index.md 并运行 demo.py）：
- **计算图**：前向传播时构建的有向无环图（DAG），每个节点代表一个操作
- **MSE**：$\ell=\frac{1}{2}(\hat{y}-y)^2$——把残差平方成可导的非负标量；$\partial\ell/\partial\hat{y}=\hat{y}-y$ 是反向传播的第一枪
- **为什么是 $\partial L/\partial w$**：训练时输入和标签不能改，只有权重是旋钮；梯度指向上坡，更新 $w\leftarrow w-\alpha\partial L/\partial w$ 走下坡
- **链式法则**：$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$——将间接依赖的梯度拆成局部导数连乘
- **局部梯度规则**：加法门梯度原样传递，乘法门梯度交换，ReLU 梯度门控
- **梯度累积（Fan-out）**：当一个变量被多条路径使用时，梯度需要求和（`+=` 而非 `=`）

## 任务清单

### 任务1：实现 Tanh 激活函数的反向传播

**描述**：补全 `Value.tanh()` 方法中的三个 TODO——前向计算、输出节点创建、反向传播闭包。

**数学公式**：

前向：$\tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$

反向（导数）：$\tanh'(x) = 1 - \tanh^2(x)$

**提示**：
- 使用 `math.tanh(self.data)` 计算前向值——Python 标准库已提供高效实现
- 导数公式的关键是：**直接用输出值 `out.data` 计算导数**，无需知道原始输入 $x$
- 反向传播闭包的写法与 `sigmoid` 完全一致，只是公式不同：`self.grad += (1 - out.data ** 2) * out.grad`

**期望输出**：
- $\tanh(0.5) \approx 0.4621$，$\partial \tanh / \partial x \approx 0.7864$
- $\tanh(-1.0) \approx -0.7616$，$\partial \tanh / \partial x \approx 0.4200$

---

### 任务2：实现除法的反向传播

**描述**：补全 `Value.__truediv__()` 和 `Value.__rtruediv__()` 方法。

**数学公式**：

$$
\frac{\partial (a/b)}{\partial a} = \frac{1}{b}, \quad \frac{\partial (a/b)}{\partial b} = -\frac{a}{b^2}
$$

**提示**：
- **不需要手动写反向传播闭包！** 利用 $a / b = a \times b^{-1}$，即 `self * (other ** -1)`
- `__pow__` 和 `__mul__` 已经分别实现了正确的 `_backward`，组合在一起会自动生成正确的梯度
- 这是**组合性（compositionality）**的绝佳体现——复杂操作可以由基本操作自由组合，梯度自动传播
- `__rtruediv__` 同理：`other / self = other * (self ** -1)`

**期望输出**：
- $6.0 / 2.0 = 3.0$，$\partial c / \partial a = 0.5$（因为 $1/b = 0.5$），$\partial c / \partial b = -1.5$（因为 $-a/b^2 = -1.5$）

---

### 任务3：实现 backward() 方法 + 梯度下降求最小值

**描述**：这是最重要的任务。补全 `Value.backward()` 方法的完整逻辑：拓扑排序、根节点梯度初始化、逆序遍历执行。

**算法步骤**：

1. **拓扑排序**（DFS 后序遍历）：
   ```python
   topo = []
   visited = set()

   def build_topo(v):
       if v not in visited:
           visited.add(v)
           for child in v._prev:   # 递归访问所有前驱
               build_topo(child)
           topo.append(v)          # 后序遍历：子节点先入列表
   ```

2. **设置根节点梯度**：`self.grad = 1.0`（因为 $\frac{\partial L}{\partial L} = 1$）

3. **逆序遍历执行**：`for node in reversed(topo): node._backward()`

**为什么后序遍历？** 因为我们要保证：当调用节点 $v$ 的 `_backward()` 时，$v$ 的所有后继（更靠近输出的节点）的梯度已经计算完毕。后序遍历 `topo` 中越靠近输出的节点越靠后，所以 `reversed(topo)` 正好是从输出到输入的合法顺序。

**任务3的续——梯度下降求函数最小值**：
补全 `find_minimum()` 函数，用自动微分 + 梯度下降找到 $f(x) = x^2 + 3x$ 的最小值。

- 初始化 `x = Value(5.0)`
- 循环 30 步，每次：构造 loss → 清零梯度 → `backward()` → 更新 `x.data`
- 解析解：$f'(x) = 2x + 3 = 0 \Rightarrow x = -1.5$，$f(-1.5) = -2.25$

**需要的函数/方法**：
- `Value(5.0)` 创建带梯度的参数
- `x * x` 或 `x ** 2` 构造 $x^2$ 项
- `x * Value(3.0)` 构造 $3x$ 项
- `x.zero_grad()` 清零梯度
- `loss.backward()` 自动计算梯度
- `x.data -= learning_rate * x.grad` 手动梯度下降

---

### 关键概念速查

| 任务 | 需要理解的概念 | 核心公式/操作 |
|------|--------------|-------------|
| TODO 1: Tanh | 激活函数的导数可用前向输出计算 | $\tanh'(x) = 1 - \tanh^2(x)$ |
| TODO 2: 除法 | 复杂操作 = 基本操作的组合 | $a/b = a \times b^{-1}$，梯度自动推导 |
| TODO 3: backward() | DFS 后序遍历 + 拓扑逆序 + 链式法则 | 先拓扑排序，逆序 `_backward()` |
| TODO 3(续): 梯度下降 | 自动微分用于优化 | $x \leftarrow x - \alpha \cdot \nabla f(x)$ |


## 源码位置

clone 后打开（相对仓库根目录）：

`docs/nn-decision/dl/backprop/code/exercise.py`
