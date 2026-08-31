---
title: "s05 前向传播与计算图 — exercise.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s05 前向传播与计算图 — exercise.py 练习指南

<a href="/notebook/code/dl/forward-graph/exercise.py" target="_blank" download>Download exercise.py</a>

## 练习目标

通过实现三个核心模块——单层全连接前向传播、GELU 激活函数、计算图追踪，从代码层面深入理解神经网络前向传播的完整流程和计算图的概念。

## 预备知识

在开始练习前，确保你已经理解了以下概念（参见 [demo.py 代码详解](./code-demo) 中的详细解释）：

- 线性变换 $z = W a_{\text{prev}} + b$ 的矩阵维度规则
- ReLU、Sigmoid、Tanh 三种激活函数的定义和实现
- 中间值缓存（cache）的概念：存储 $z$、$a_{\text{prev}}$、$W$、$b$ 供反向传播使用
- GELU 的数学定义：$\text{GELU}(z) = z \cdot \Phi(z)$，其中 $\Phi(z)$ 是标准正态 CDF
- 计算图的概念：将复杂表达式分解为基本运算节点的有向无环图

## 任务清单

### 任务1：实现单层全连接层的前向传播 `dense_layer_forward(A_prev, W, b, activation)`

- **用到的公式**：
  - 线性变换：$Z = W \cdot A_{\text{prev}} + b$
  - ReLU：$\text{ReLU}(z) = \max(0, z)$
  - Sigmoid：$\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Tanh：$\tanh(z)$
- **实现步骤**：
  1. 计算 $Z = W @ A_{\text{prev}} + b$（矩阵乘法 + 广播加法）
  2. 根据 `activation` 参数选择激活函数：
     - `"relu"`：`np.maximum(0, Z)`
     - `"sigmoid"`：`1.0 / (1.0 + np.exp(-Z))`（注意数值稳定：可先用 `np.clip(Z, -500, 500)`）
     - `"tanh"`：`np.tanh(Z)`（NumPy 内置）
     - `"none"`：$A = Z$（恒等映射，无激活）
  3. 创建 cache 字典，包含 `{Z: 线性输出, A_prev: 上一层激活, W: 权重矩阵, b: 偏置向量}`
- **需要调用的函数**：`@` 运算符（矩阵乘法）、`np.maximum()`、`np.exp()`、`np.tanh()`、`np.clip()`
- **返回**：`(Z, A, cache)` 三元组
- **期望输出**：`Z.shape` 为 `(n_curr, m)`，`A.shape` 也为 `(n_curr, m)`，cache 包含四个键

### 任务2：实现 GELU 激活函数 `gelu_exact(z)` 与 `gelu_derivative(z)`

- **GELU 的精确数学定义**：
  $$\text{GELU}(z) = z \cdot \Phi(z)$$
  其中 $\Phi(z)$ 是标准正态分布的累积分布函数（CDF）：
  $$\Phi(z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$
  $\text{erf}(x)$ 是误差函数（error function）。

- **GELU 的近似实现**（如果需要避免引入 scipy）：
  $$\text{GELU}(z) \approx 0.5z \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715 z^3\right)\right)\right]$$

- **GELU 导数的精确形式**：
  $$\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)$$
  其中 $\phi(z)$ 是标准正态分布的概率密度函数（PDF）：
  $$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

- **实现提示**：
  - 如果使用 scipy：`from scipy.special import erf`，$\Phi(z) = 0.5 * (1 + \text{erf}(z / \sqrt{2}))$
  - 如果不想引入 scipy：使用 tanh 近似公式
  - 导数实现：需要同时用到 $\Phi(z)$ 和 $\phi(z)$
- **验证**：$\text{GELU}(0) = 0$，$\text{GELU}(2) \approx 1.95$（$z$ 大时行为接近 ReLU），$\text{GELU}(-2) \approx -0.05$

### 任务3：手动追踪计算图 `trace_computational_graph(X)`

给定表达式：

$$
f(x_1, x_2, x_3) = \sigma\left((x_1 \cdot w_1 + x_2 \cdot w_2 + b) \cdot w_3 + x_3\right)
$$

其中 $w_1=0.5$, $w_2=-0.3$, $w_3=2.0$, $b=0.1$，$\sigma$ 是 Sigmoid。

- **任务**：把这个复合函数分解为 7 个基本运算节点，每个节点执行一个简单操作（加减乘除、Sigmoid）

- **计算图的节点链**：

  | 节点 | 计算 | 操作类型 | 输入 |
  |------|------|---------|------|
  | `u1` | $x_1 \cdot w_1$ | multiply | $x_1$, $w_1$ |
  | `u2` | $x_2 \cdot w_2$ | multiply | $x_2$, $w_2$ |
  | `u3` | $u_1 + u_2$ | add | $u_1$, $u_2$ |
  | `u4` | $u_3 + b$ | add | $u_3$, $b$ |
  | `u5` | $u_4 \cdot w_3$ | multiply | $u_4$, $w_3$ |
  | `u6` | $u_5 + x_3$ | add | $u_5$, $x_3$ |
  | `u7` | $\sigma(u_6)$ | sigmoid | $u_6$ |

- **实现**：对每个节点，计算并记录到 `graph_nodes` 字典中
- **每个节点的记录格式**：
  ```python
  graph_nodes["u1"] = {
      "value": 计算结果,
      "inputs": ["x1", "w1"],
      "op": "multiply"
  }
  ```
- **Sigmoid 实现提示**：$\sigma(z) = 1 / (1 + e^{-z})$，可使用 `np.exp()`。为了防止数值溢出，建议先 clip 输入

- **核心理解**：这个练习让你手动体验 PyTorch/TensorFlow 底层在做什么——自动将复杂的数学表达式分解为计算图上的基本操作节点，每个节点只需要知道自己的局部操作和链式法则导数规则。

## 验证标准

运行 `python exercise.py`：

1. **TODO 1**：对 4 种激活函数（relu, sigmoid, tanh, none）都应输出正确的 `Z.shape` 和 $A$ 的范围
2. **TODO 2**：$\text{GELU}(0) = 0$，$\text{GELU}(z) \approx z$ for $z \gg 0$（如 $z=2$ 时约等于 $1.95$）
3. **TODO 3**：打印出 7 个节点的计算图，最终输出 `u7` 是一个在 $(0, 1)$ 之间的值（因为经过了 Sigmoid）

## 完整代码

<<< @/dl/forward-graph/code/exercise.py
