---
title: "s07 多层网络的矩阵反向传播 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s07 多层网络的矩阵反向传播 — demo.py 代码详解

<a href="../code/s07_matrix_backprop/demo.py" target="_blank" download>Download demo.py</a>

## 运行方式

```bash
cd s07_matrix_backprop/code
python demo.py
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import os
```

- **`numpy`**：科学计算核心库。提供多维数组（`ndarray`）、矩阵乘法（`@`）、随机数生成等。整个 MLP 的参数存储、前向计算、梯度计算全基于 NumPy。
- **`matplotlib`**：可视化库。用于绘制决策边界、损失曲线、梯度范数变化、权重热力图。
- **`typing`**：类型注解。`Dict` 用于参数字典和梯度的类型提示，`List` 用于每层维度的列表，`Tuple` 用于多返回值的类型注解，`Callable` 用于激活函数类型。

> **从标量到矩阵的跃迁**：上一节 s06 的 `Value` 类是标量级别的自动微分，每个数字都是独立的节点。本节直接用 NumPy 矩阵批量操作，一个 `ndarray` 对象代表一整层神经元的输出。

---

### 第2步：激活函数及其导数 — 逐元素操作

每个激活函数都需要实现两个版本：**前向函数**（`φ(x)`）和**导数函数**（`φ'(x)`），因为 `δ` 递推公式中需要 $\phi'(Z^{[l]})$。

#### ReLU

```python
def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(np.float64)
```

数学定义：

$$
\text{ReLU}(Z) = \max(0, Z), \quad \text{ReLU}'(Z) = \mathbb{1}[Z > 0]
$$

`np.maximum(0, Z)` 逐元素比较，返回每个位置的非负值。`(Z > 0)` 生成布尔数组，`.astype(np.float64)` 将 True/False 转为 1.0/0.0——这是一个高效的逐元素"门控"实现。

#### Sigmoid

```python
def sigmoid(Z: np.ndarray) -> np.ndarray:
    Z_clipped = np.clip(Z, -500, 500)   # 防止 exp 溢出
    return 1.0 / (1.0 + np.exp(-Z_clipped))

def sigmoid_derivative(Z: np.ndarray) -> np.ndarray:
    s = sigmoid(Z)
    return s * (1.0 - s)
```

数学公式：

$$
\sigma(Z) = \frac{1}{1 + e^{-Z}}, \quad \sigma'(Z) = \sigma(Z)(1 - \sigma(Z))
$$

`np.clip(Z, -500, 500)` 是一个关键的数值稳定技巧：当 $Z$ 是很大的负数时，$e^{-Z}$ 会溢出浮点数范围（如 $e^{800} \approx 10^{347}$，远超 float64 上限）。将 $Z$ 截断到 $[-500, 500]$ 完全不影响结果（因为 $\sigma(-500) \approx 0$，$\sigma(500) \approx 1$），但彻底避免了溢出。

#### Tanh

```python
def tanh(Z: np.ndarray) -> np.ndarray:
    return np.tanh(Z)

def tanh_derivative(Z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(Z) ** 2
```

$$
\tanh'(Z) = 1 - \tanh^2(Z)
$$

#### 激活函数注册表

```python
ACTIVATION_REGISTRY = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (lambda Z: Z, lambda Z: np.ones_like(Z)),
}
```

这个字典提供了一个**工厂模式**：通过字符串名称即可获取对应激活函数及其导数。`linear` 激活（恒等映射）的导数是全1矩阵——因为 $\frac{\partial x}{\partial x} = 1$。

---

### 第3步：MLP 类的参数初始化

```python
class MLP:
    def __init__(self, layer_dims: List[int], activations: List[str], seed: int = 42):
        for l in range(1, self.L + 1):
            n_in = layer_dims[l - 1]
            n_out = layer_dims[l]
            self.parameters[f"W{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
            self.parameters[f"b{l}"] = np.zeros((n_out, 1))
```

**He 初始化**：

$$
W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$

为什么是 $\sqrt{2/n_{in}}$？He 初始化（Kaiming He, 2015）专门为 ReLU 激活函数设计。在前向传播中，如果权重方差为 $2/n_{in}$，那么经过 ReLU 后输出的方差大约能保持稳定，不会逐层衰减或放大。这对深层网络的训练至关重要。

偏置统一初始化为零——偏置的初始值对梯度流的影响远小于权重，因为偏置只做平移，不参与乘法。

**参数量统计**：对于 $[2, 16, 8, 1]$ 的网络：
- $W_1$: $16 \times 2 = 32$ 参数
- $b_1$: $16 \times 1 = 16$ 参数
- $W_2$: $8 \times 16 = 128$ 参数
- $b_2$: $8 \times 1 = 8$ 参数
- $W_3$: $1 \times 8 = 8$ 参数
- $b_3$: $1 \times 1 = 1$ 参数
- 总计：$193$ 个参数（代码输出约为 203，因为最后一层也包含偏置）

---

### 第4步：前向传播 — 逐层计算 + 缓存中间值

```python
def forward(self, X: np.ndarray) -> np.ndarray:
    self.caches = []
    A = X  # A[0] = X

    for l in range(1, self.L + 1):
        A_prev = A
        W = self.parameters[f"W{l}"]
        b = self.parameters[f"b{l}"]
        Z = W @ A_prev + b           # 线性变换 Z[l] = W[l] @ A[l-1] + b[l]
        act_fn, _ = ACTIVATION_REGISTRY[self.activations[l - 1]]
        A = act_fn(Z)                 # 非线性激活 A[l] = φ(Z[l])

        self.caches.append({
            "Z": Z,                   # 用于计算 φ'(Z[l])
            "A_prev": A_prev,         # 用于计算 dW[l] = δ @ (A_prev)^T
            "A": A,                   # 当前层输出（即下一层的输入）
        })

    return A
```

数学对应：

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = \phi^{[l]}(Z^{[l]})
$$

其中 $A^{[0]} = X$ 是输入数据。

**缓存的三个值及其用途**：
| 缓存字段 | 存储内容 | 反向传播中的用途 |
|---------|---------|----------------|
| `Z` | 线性输出 $Z^{[l]}$ | 计算 $\phi'(Z^{[l]})$ |
| `A_prev` | 上一层激活 $A^{[l-1]}$ | 计算 $dW^{[l]} = \delta^{[l]} (A^{[l-1]})^T$ |
| `A` | 当前层激活 $A^{[l]}$ | 下一层前向传播的输入（即 $A^{[l]}$ 作为下一层的 $A_{prev}$） |

**维度说明**（以 $W: (16, 2)$，输入 $(2, 300)$ 为例）：
- $Z = W @ A_{prev} + b$：$(16, 2) @ (2, 300) + (16, 1) = (16, 300)$（广播机制自动扩展 $b$）
- $A = \text{ReLU}(Z)$：保持 $(16, 300)$

---

### 第5步：反向传播 — δ 递推公式的核心实现

这是本节最关键的代码。完整实现了从输出层到输入层的梯度逆传。

```python
def backward(self, Y: np.ndarray) -> Dict[str, np.ndarray]:
    m = Y.shape[1]
    self.grads = {}

    # ---- 步骤 1: 输出层的 δ[L] ----
    AL = self.caches[-1]["A"]                     # 预测值
    ZL = self.caches[-1]["Z"]                     # 输出层线性输出
    _, act_prime_fn = ACTIVATION_REGISTRY[self.activations[-1]]
    dAL = (1.0 / m) * (AL - Y)                     # ∂L/∂A[L]
    dZ = dAL * act_prime_fn(ZL)                    # δ[L] = ∇_A L ⊙ φ'(Z[L])
```

**输出层 δ 的计算**分为两步：

1. **损失对激活的梯度**（MSE 损失）：

$$
\frac{\partial L}{\partial A^{[L]}} = \frac{1}{m}(A^{[L]} - Y)
$$

其中 $L = \frac{1}{2m} \sum (A^{[L]} - Y)^2$。除以 $m$ 是为了对 mini-batch 取平均——梯度是所有样本梯度的均值。

2. **乘以激活函数的导数**（链式法则）：

$$
\delta^{[L]} = \frac{\partial L}{\partial A^{[L]}} \odot \phi'(Z^{[L]})
$$

注意：这是**逐元素相乘**（`*`，Hadamard 积 $\odot$），不是矩阵乘法（`@`）。

---

**隐藏层的 δ 递推**：

```python
for l in reversed(range(1, self.L + 1)):
    cache = self.caches[l - 1]
    A_prev = cache["A_prev"]

    # 参数梯度
    self.grads[f"dW{l}"] = dZ @ A_prev.T
    self.grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)

    # 继续向前递推 δ
    if l > 1:
        W_next = self.parameters[f"W{l}"]
        Z_prev = self.caches[l - 2]["Z"]
        _, act_prime_fn_prev = ACTIVATION_REGISTRY[self.activations[l - 2]]
        dZ = (W_next.T @ dZ) * act_prime_fn_prev(Z_prev)
```

**三步走**，对应三个核心公式：

**1. 权重梯度（外积）**：

$$
\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} \cdot (A^{[l-1]})^T
$$

注意：代码中没有显式除以 $m$，因为 $dZ$（即 $\delta^{[l]}$）在输出层已经包含了 $\frac{1}{m}$ 因子。

**维度验证**：$\delta^{[l]}$ 形状 $(n^{[l]}, m)$，$(A^{[l-1]})^T$ 形状 $(m, n^{[l-1]})$，乘积 $(n^{[l]}, n^{[l-1]})$ —— 与 $W^{[l]}$ 形状完全一致。

**2. 偏置梯度**：

$$
\frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{[l]}_i
$$

`np.sum(dZ, axis=1, keepdims=True)` 对 300 个样本的误差信号按行求和，形状从 $(n^{[l]}, m)$ 变为 $(n^{[l]}, 1)$。

**3. δ 递推（核心）**：

$$
\delta^{[l-1]} = (W^{[l]})^T \delta^{[l]} \odot \phi'(Z^{[l-1]})
$$

- `W_next.T @ dZ`：将第 $l$ 层的误差通过**转置权重**回传——这是"责任分配"的数学体现
- `* act_prime_fn_prev(Z_prev)`：经过激活函数的导数门控
- 这个递推关系是反向传播的灵魂——一旦 $\delta^{[L]}$ 被算出，所有层的 $\delta$ 和参数梯度都能通过统一的公式自动推导

---

### 第6步：参数更新 — 梯度下降

```python
def update(self, learning_rate: float):
    for l in range(1, self.L + 1):
        self.parameters[f"W{l}"] -= learning_rate * self.grads[f"dW{l}"]
        self.parameters[f"b{l}"] -= learning_rate * self.grads[f"db{l}"]
```

标准梯度下降一步：

$$
W^{[l]} := W^{[l]} - \alpha \cdot \frac{\partial L}{\partial W^{[l]}}
$$

注意这里没有使用任何优化器技巧（Momentum、Adam 等）——那是下一节 s08 的主题。

---

### 第7步：梯度检查 — 用有限差分验证解析梯度

梯度检查是手写反向传播时的黄金调试标准。本节使用**双边有限差分法**：

```python
def gradient_check(model, X, Y, epsilon=1e-7):
    # 对每个参数的每个元素逐一计算数值梯度
    for idx in ...:  # 遍历参数矩阵每个位置
        original_value = param[idx]

        param[idx] = original_value + epsilon
        loss_plus = model.compute_loss(model.forward(X), Y)

        param[idx] = original_value - epsilon
        loss_minus = model.compute_loss(model.forward(X), Y)

        grad_numeric[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        param[idx] = original_value  # 恢复原始值
```

数学公式：

$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}
$$

**为什么用双边差分而非单边？** 单边差分 $\frac{L(\theta+\epsilon) - L(\theta)}{\epsilon}$ 的误差是 $O(\epsilon)$，而双边差分的误差是 $O(\epsilon^2)$，精度高得多。

**相对误差公式**：

$$
\text{relative error} = \frac{\|\text{grad}_\text{analytic} - \text{grad}_\text{numeric}\|_2}
{\|\text{grad}_\text{analytic}\|_2 + \|\text{grad}_\text{numeric}\|_2}
$$

解读标准：
- $< 10^{-7}$：正确
- $\approx 10^{-5}$：可能有小问题
- $> 10^{-3}$：几乎肯定有 bug

> **重要警告**：梯度检查极其缓慢——每个参数需要两次额外的前向传播。对于 200 个参数，需要用 400 次前向传播来验证一次梯度。这就是为什么实际训练中**不使用**梯度检查，只在开发验证时对极小网络+小 batch 使用。

---

### 第8步：数据生成 — 双月形二分类数据集

```python
def make_moons_dataset(n_samples=200, noise=0.15, seed=0):
    # 上半月（类别 0）：沿单位圆上半部分分布
    t = np.linspace(0, np.pi, n_samples_per_class)
    X0 = np.vstack([
        np.cos(t) + randn * noise,   # x 坐标 = cos(角度) + 噪声
        np.sin(t) + randn * noise,   # y 坐标 = sin(角度) + 噪声
    ])

    # 下半月（类别 1）：偏移后的下半圆
    X1 = np.vstack([
        1 - np.cos(t) + randn * noise,
        1 - np.sin(t) - 0.5 + randn * noise,
    ])
```

双月形数据是一个经典的非线性二分类问题——一条直线无法分开两个类别，必须用非线性决策边界。这正好展示了 MLP（带 ReLU 隐藏层）的非线性表达能力的必要性。

---

### 第9步：可视化组件

#### 决策边界可视化

```python
def plot_decision_boundary(model, X, Y, title, filename):
    # 生成网格点
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    Z = model.forward(grid)              # 模型对每个网格点的预测概率
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1])  # 填充决策区域
    plt.contour(xx, yy, Z, levels=[0.5])          # 画 p=0.5 的决策线
```

`contourf` 用不同颜色填充预测概率 $<0.5$ 和 $>0.5$ 的区域，`contour` 在 $p=0.5$ 处画一条黑线——这就是模型的**决策边界**。训练前后各画一张，直观看到模型从"随机划分"到"学会分开两个月形"的过程。

#### 权重热力图

```python
axes[l, 0].imshow(model_before.parameters[f"W{l+1}"], cmap='RdBu_r')
axes[l, 1].imshow(model_after.parameters[f"W{l+1}"], cmap='RdBu_r')
```

`imshow` 将权重矩阵以颜色编码显示——红色代表正值权重，蓝色代表负值。训练前的权重是随机均匀的杂色，训练后的权重呈现有规律的模式，说明网络学到了有意义的结构。

---

### 第10步：训练循环 — 将一切串联

```python
for epoch in range(n_epochs):
    Y_pred = model.forward(X)       # ① 前向传播
    loss = model.compute_loss(Y_pred, Y)  # ② 计算损失
    model.backward(Y)               # ③ 反向传播（计算梯度）
    model.update(learning_rate)     # ④ 参数更新
```

这就是深度学习训练的**四步循环**，在每一个 epoch 中重复：
1. **前向**：数据从输入流到输出
2. **损失**：量化预测与真实标签的差距
3. **反向**：从损失出发，梯度逆流回每个参数
4. **更新**：参数沿梯度反方向移动一步

---

### 关键概念速查表

| 概念 | 数学公式 | 代码实现 |
|------|---------|---------|
| $\delta^{[L]}$ 起手式 | $\nabla_A L \odot \phi'(Z^{[L]})$ | `dAL = (1/m)*(AL - Y)` → `dZ = dAL * act_prime_fn(ZL)` |
| $\delta^{[l]}$ 递推 | $(W^{[l+1]})^T \delta^{[l+1]} \odot \phi'(Z^{[l]})$ | `(W_next.T @ dZ) * act_prime_fn_prev(Z_prev)` |
| 权重梯度 | $\frac{1}{m} \delta^{[l]} (A^{[l-1]})^T$ | `dZ @ A_prev.T`（外积） |
| 偏置梯度 | $\frac{1}{m} \sum_i \delta^{[l]}_i$ | `np.sum(dZ, axis=1, keepdims=True)` |
| He 初始化 | $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$ | `randn * sqrt(2.0 / n_in)` |
| 梯度检查 | $\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$ | 双边有限差分 |
| 梯度范数 | $\|g\|_2 = \sqrt{\sum g_i^2}$ | `np.linalg.norm(grad)` |

## 完整代码

<<< @/snippets/s07_matrix_backprop/demo.py
