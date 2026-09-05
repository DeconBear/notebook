---
title: "s05 前向传播与计算图 — demo.py"
---


> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

# s05 前向传播与计算图 — demo.py 代码详解

<a href="/notebook/code/nn-decision/dl/forward-graph/demo.py" target="_blank" download>Download demo.py</a>
　
<a href="/notebook/code/nn-decision/dl/forward-graph/plot_demo.py" target="_blank" download>Download plot_demo.py</a>

## 运行方式

```bash
cd docs/nn-decision/dl/forward-graph/code
python demo.py          # 前向传播主线（形状打印，不画图）
python plot_demo.py     # 激活函数 / 网络结构 / 激活分布配图
```

## 代码逐段详解

### 第1步：导入库 — 每个库是做什么的

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Callable
```

- **`os`**：文件路径操作，创建 `images/` 目录
- **`numpy`**：数值计算核心。关键用法：`np.random.randn()` 生成随机数据，`np.maximum()` 实现 ReLU，`np.exp()` 实现 Sigmoid，`np.tanh()` 实现 Tanh，`@` 运算符矩阵乘法
- **`matplotlib`**：绘图，包括网络结构图、激活函数对比图、激活值分布直方图
- **`matplotlib.patches`**：提供绘图元素（如 `Circle` 用于绘制神经元节点，`Patch` 用于图例）
- **`typing`**：Python 类型提示（`Dict`, `List`, `Tuple`, `Callable`），让函数签名更清晰，便于理解参数和返回值的类型

### 第2步：激活函数及其导数 — 神经网络的非线性来源

这是前向传播中最关键的概念之一。激活函数在每个线性变换之后引入非线性，使得多层网络能够学习复杂函数。

#### 2.1 ReLU（Rectified Linear Unit）

```python
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)
```

数学定义：

$$
\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 0 & z < 0 \\ 1 & z > 0 \end{cases}
$$

- `np.maximum(0, z)` 是逐元素操作：对数组的每个元素，保留大于 0 的值，小于 0 的值替换为 0
- `(z > 0)` 先做比较，得到的是 **布尔数组**（`True` / `False`），dtype 是 `bool`。例如 `z = [-1, 2]` → `[False, True]`。布尔值不能直接拿去乘梯度，所以要转成数字。
- **`astype`**：NumPy 数组的「改类型」方法。`arr.astype(新类型)` 生成一份指定类型的新数组，原数组不变。
- **`np.float64`**：64 位浮点数，就是日常说的 `float`（双精度）。科学计算默认用它，和后面的权重、梯度类型一致。转完之后 `True → 1.0`，`False → 0.0`，正好对应 $\mathrm{ReLU}'(z)\in\{0,1\}$。

```python
z = np.array([-1.0, 0.5, 2.0])
mask = z > 0                  # array([False,  True,  True])  ← 还是 bool
grad = mask.astype(np.float64)  # array([0., 1., 1.])         ← 才能当导数用
```

**为什么 ReLU 是深度学习革命的英雄？** 正区间的导数恒为 1，这意味着在反向传播中梯度可以**无损传播**——20 层 ReLU 网络连乘梯度后仍是 1，而 Sigmoid 网络连乘 20 个最大 0.25 的导数后梯度只剩 $0.25^{20} \approx 9 \times 10^{-13}$（消失殆尽）。

**"死亡 ReLU"问题**：如果某个神经元的输出对所有输入都 $\leq 0$，则该神经元的梯度永远为 0，参数不再更新——这个神经元"死亡"了。这是 ReLU 的主要缺点，Leaky ReLU 设计了一个小斜率（0.01）在负区间来缓解这个问题。

#### 2.2 Sigmoid

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)
```

数学定义：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

`np.clip(z, -500, 500)` 将 $z$ 限制在 $[-500, 500]$ 内，防止 $e^{500}$ 上溢出（$e^{500} \approx 1.4 \times 10^{217}$，远超 float64 的最大值）。裁剪到 $\pm500$ 对 Sigmoid 值几乎没有影响——$\sigma(500)$ 和 $\sigma(\infty)$ 在浮点精度下不可区分。

Sigmoid 导数可以用自身表达（$\sigma(1-\sigma)$），这在代码上非常简洁。

**当前用途**：曾经是隐藏层的标准激活，现在因为梯度消失问题仅在二分类输出层使用（配合 BCE 损失）。

#### 2.3 Tanh

```python
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    t = np.tanh(z)
    return 1 - t ** 2
```

数学定义：

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)
$$

`np.tanh()` 是 NumPy 内置的数值稳定实现，直接调用即可。相比 Sigmoid，Tanh 的输出是**零中心**的（$(-1, 1)$ vs $(0, 1)$），这对优化有帮助——零中心的数据使得梯度更新方向更一致。

#### 2.4 GELU（Gaussian Error Linear Unit）

```python
def gelu_approximate(z: np.ndarray) -> np.ndarray:
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    return 0.5 * z * (1.0 + np.tanh(sqrt_2_over_pi * (z + 0.044715 * z ** 3)))
```

数学定义（精确版）：

$$
\text{GELU}(z) = z \cdot \Phi(z), \quad \Phi(z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]
$$

代码中使用的是 tanh 近似（高精度，广泛使用）：

$$
\text{GELU}(z) \approx 0.5z \left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715 z^3\right)\right)\right]
$$

GELU 的核心思想：不像 ReLU 那样"一刀切"地决定通过还是丢弃信息，而是根据 $z$ 的大小**概率性地**让信息通过。当 $z$ 很大时通过概率接近 1，$z$ 接近 0 时"是否通过"具有不确定性。这引入了类似 Dropout 的随机正则化效果，但是**确定性**的（不需要采样）。

GELU 是 Transformer 架构的标准激活函数——BERT、GPT、ViT 等全部使用它。

### 第3步：参数初始化 — He 初始化

```python
def initialize_parameters(layer_dims, seed=42):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        n_in = layer_dims[l - 1]
        n_out = layer_dims[l]
        parameters[f"W{l}"] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        parameters[f"b{l}"] = np.zeros((n_out, 1))
    return parameters
```

**为什么需要特殊的初始化策略？**

权重初始化对深层网络的训练至关重要。如果初始化不当：
- **太大**：前向传播时激活值爆炸，梯度也爆炸
- **太小**：前向传播时激活值消失，梯度也消失

**He 初始化**（Kaiming He, 2015）专为配合 ReLU 设计：

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

- $\sqrt{2/n_{\text{in}}}$ 是标准差：$n_{\text{in}}$ 是当前层的**输入个数**（前一层神经元数），不是权重取值区间
- 因子 2 是为了补偿 ReLU 将一半输入置零造成的方差减半效应
- 这个初始化使得每层输出的方差保持稳定，无论网络有多深

**不要把形状和取值混在一起。** `layer_dims = [3, 4, …]` 时，第一层权重是形状 `(4, 3)` 的矩阵：4 个神经元，每个接 3 个输入，一共 $4\times 3=12$ 个数。每个数独立抽自 $\mathcal{N}(0, \sqrt{2/3})$，标准差约 $0.82$，典型取值在 $-2$～$2$ 附近，**不是**「3 到 4 之间的随机数」。

同一层共用同一个标准差（因为每个神经元的扇入 $n_{\text{in}}$ 相同），但 **12 个权重彼此不同**。如果同一层所有神经元的权重完全一样，它们永远算同一个函数，堆再多神经元也等于 1 个——这叫对称性没被打破。`np.random.seed(42)` 只保证你下次再跑还是这组数，不是让同一层抄同一份权重。偏置才是整层都初始化成 0。

**Xavier 初始化**（配合 Tanh/Sigmoid）使用 $\sqrt{1/n_{\text{in}}}$ 作为标准差，因为 Sigmoid/Tanh 不会像 ReLU 那样丢弃负半轴。

**偏置初始化**：偏置向量通常初始化为全零。因为权重的随机初始化已经打破了对称性，偏置从零开始学习是合理且常见的做法。

### 第4步：前向传播 — 数据在网络中的旅程

```python
def forward_pass(X, parameters, activations, verbose=True):
    a = X  # a^{[0]} = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L + 1):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        z = W @ a + b                           # ① 线性变换
        a_new = activations[l - 1](z)            # ② 非线性激活
        cache = {"z": z, "a_prev": a, "a": a_new}  # ③ 缓存中间值
        caches.append(cache)
        a = a_new                               # ④ 更新当前激活，传给下一层
    return a, caches
```

最后一个参数 **`verbose`**（英语里就是「话多、啰嗦」）**不参与计算**，只决定要不要往终端打调试信息。默认 `True`：每层打印 $a,W,b,z$ 的形状，以及激活值的 min/max/mean/std。主程序里写的是 `forward_pass(..., verbose=True)`，所以你跑 `demo.py` 会看到「第 1 层 / 第 2 层 / …」那一大段。改成 `verbose=False`，前向结果完全一样，只是屏幕安静。真正改网络输出的是 `W @ a + b` 和激活函数；`verbose` 只是教学用的「打开形状字幕」。

这是前向传播的核心循环。每一层执行完全相同的两步操作：

**子步骤 1：线性变换** $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$

数学上，$W^{[l]}$ 是一个 $n^{[l]} \times n^{[l-1]}$ 的矩阵，它将 $n^{[l-1]}$ 维的输入映射到 $n^{[l]}$ 维的中间表示。代码里写成一行 NumPy：

```python
z = W @ a + b
```

**`@` 是矩阵乘法**（Python 3.5+ 运算符，NumPy 里等价于 `np.matmul(W, a)`），不是 `*`。`W * a` 是逐元素相乘，两边形状必须能对齐；`@` 才是「行乘列再求和」。

以第一层为例：`W` 形状 `(4, 3)`（4 个神经元 × 3 个输入），`a` 形状 `(3, 32)`（3 个特征 × 32 个样本），则

$$
W_{(4\times 3)}\,a_{(3\times 32)} = (W @ a)_{(4\times 32)}
$$

每个样本得到 4 个数。

**`+ b` 是广播加法。** `b` 形状 `(4, 1)`，NumPy 会把它加到全部 32 列上，等于每个样本共用同一份偏置，结果仍是 `(4, 32)`。整行的含义：先做 $Wa$，再给每个神经元加 $b$，此时还没有过 ReLU。

**手算小例子：batch 从 32 缩成 4。** 3 个特征、2 个隐藏神经元、4 个样本（整数，方便纸上乘）。每一列仍是一个样本，不是「4 维向量」。

$$
W=\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}_{2\times 3}
\qquad
X=\begin{bmatrix}
1&2&0&1\\
0&1&1&2\\
1&0&1&0
\end{bmatrix}_{3\times 4}
$$

| | 样本1 | 样本2 | 样本3 | 样本4 |
|--|------|------|------|------|
| $x_1$ | 1 | 2 | 0 | 1 |
| $x_2$ | 0 | 1 | 1 | 2 |
| $x_3$ | 1 | 0 | 1 | 0 |

$$
Z=WX
=
\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}
\begin{bmatrix}
1&2&0&1\\
0&1&1&2\\
1&0&1&0
\end{bmatrix}
=
\begin{bmatrix}3&2&2&1\\1&1&2&2\end{bmatrix}_{2\times 4}
$$

只看第 1 列，就是单个样本：

$$
\begin{bmatrix}1&0&2\\0&1&1\end{bmatrix}
\begin{bmatrix}1\\0\\1\end{bmatrix}
=
\begin{bmatrix}3\\1\end{bmatrix}
$$

正好是 $Z$ 的第 1 列。第 2、3、4 列同理：**同一张 $W$ 乘四次**，一次写成 $WX$。再加上 $b=\begin{bmatrix}1\\0\end{bmatrix}$（形状 $2\times 1$）会广播到 4 列：

$$
Z+b
=
\begin{bmatrix}3&2&2&1\\1&1&2&2\end{bmatrix}
+
\begin{bmatrix}1\\0\end{bmatrix}
=
\begin{bmatrix}4&3&3&2\\1&1&2&2\end{bmatrix}
$$

本课 `demo.py` 只是把上面的 4 换成 32：`W` 仍是「隐藏神经元数 × 特征数」，`X` 变成 `(3, 32)`，乘完第二维是 32。

**子步骤 2：非线性激活** $a^{[l]} = \phi^{[l]}(z^{[l]})$

$\phi^{[l]}$ 是激活函数（ReLU、Sigmoid 等），逐元素作用于 $z^{[l]}$ 的每个元素。这是神经网络非线性能力的来源——没有它，多层网络退化为单层线性模型。

**子步骤 3：存储中间值（cache）**

```python
cache = {
    "z": z,           # z^{[l]} — 反向传播中计算激活函数导数 φ'(z) 时需要
    "a_prev": a,      # a^{[l-1]} — 反向传播中计算 dW = δ · (a_prev)^T 时需要
    "a": a_new,       # a^{[l]} — 作为下一层的输入
}
```

这三个值是反向传播的"燃料"——没有它们，梯度无法从后往前传递。详细用途见下文的"为什么必须存储中间值"。

**张量形状追踪（由 `verbose` 打开）**：为 `True` 时，每层会打印 $a^{[l-1]}, W^{[l]}, b^{[l]}, z^{[l]}, a^{[l]}$ 的形状和激活值统计。形状对不上是神经网络里最常见的报错来源；教学时打开它，把这段代码嵌进更大训练循环时再关掉。

### 第5步：为什么必须存储中间值？

反向传播需要以下信息来计算每个参数的梯度：

| 存储的值 | 反向传播中的用途 | 对应的梯度公式 |
|---------|----------------|--------------|
| $z^{[l]}$ | 计算激活函数的导数 | $\delta^{[l]} = \delta^{[l+1]} W^{[l+1]T} \odot \phi'(z^{[l]})$ |
| $a^{[l-1]}$ | 计算权重梯度 | $\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$ |
| $\delta^{[l+1]}$ | 递推计算前一层误差 | 链式法则逐层传递 |

这就是为什么训练需要比推理更多的显存——前向传播的所有中间结果必须保留到反向传播完成。如果显存不够，有一个权衡技巧叫 **Checkpointing/Re-materialization**：不存储中间值，在反向传播时重新计算前向传播。这节省了显存但增加了计算量。

### 第6步：配图（`plot_demo.py`，与主线分开）

激活函数曲线、网络结构示意图、各层激活直方图**不参与** $Wa+b$ 的教学计算，因此单独放在 `plot_demo.py`。它会 `from demo import` 激活函数和 `forward_pass`，再调用 matplotlib。看图时运行：

```bash
python plot_demo.py
```

图保存到本章 `images/`：`activation_functions.png`、`network_structure.png`、`forward_data_flow.png`。

#### 6.1 网络结构图

`plot_network_structure()` 用 matplotlib 绘制神经网络的计算图视角：

- **蓝色圆点**：输入层神经元（$x_1, x_2, x_3$）
- **橙色圆点**：隐藏层神经元（$h_{1,1}, h_{1,2}, \dots$）
- **红色圆点**：输出层神经元（$\hat{y}_1$）
- **灰色连线**：连接权重，每对前后层神经元之间都有
- **绿色标注**：两柱之间的 $W[l]$ 矩阵形状

神经元在 y 轴上的位置通过 `np.linspace()` 均匀分布。网络使用 `plt.Circle()` 绘制圆形节点，`ax.plot()` 绘制连接线。

#### 6.2 激活函数对比图

`plot_activation_functions()` 绘制 4 种激活函数（ReLU、Sigmoid、Tanh、Leaky ReLU）的函数值曲线和导数曲线：

- 蓝色实线：$f(z)$（函数值）
- 红色虚线：$f'(z)$（导数值）
- 灰色虚线：$y=0$ 和 $y=1$ 参考线

从图上可以直观看到：
- Sigmoid 的导数值域是 $(0, 0.25]$，远小于 1——梯度消失的根源
- Tanh 的导数值域是 $(0, 1]$，最大值 1 但两端饱和
- ReLU 的导数在正区间恒为 1——这就是为什么它解决了梯度消失
- Leaky ReLU 在负区间有微小斜率 0.01——防止神经元"死亡"

#### 6.3 前向传播数据流

`plot_forward_data_flow()` 绘制每层激活值的分布直方图：

- 第一列：输入数据的分布（期望：标准正态分布 $N(0,1)$）
- 后续列：每层激活输出的分布
- 红色虚线标注 $x=0$ 参考线

**观察要点**：如果激活值的分布（均值和方差）在层间保持稳定，说明初始化参数设置合理。如果激活值越来越集中在 0（消失），或越来越发散（爆炸），说明初始化或网络结构有问题。

### 第7步：主程序 — 完整的 3 层 MLP 前向传播

```python
def main():
    # 1. 生成合成数据: 32 个样本，3 个特征 (3, 32)
    X = np.random.randn(3, 32)

    # 2. 定义网络结构: [3] → [4] → [4] → [1]
    layer_dims = [3, 4, 4, 1]

    # 3. He 初始化参数
    parameters = initialize_parameters(layer_dims)

    # 4. 选择激活函数: 隐藏层用 ReLU，输出层用 Sigmoid
    activations = [relu, relu, sigmoid]

    # 5. 执行前向传播
    y_pred, caches = forward_pass(X, parameters, activations)
```

这个 3 层 MLP 的网络结构为：
- **输入层**：3 个神经元（对应 3 个特征）
- **隐藏层 1**：4 个神经元，ReLU 激活
- **隐藏层 2**：4 个神经元，ReLU 激活
- **输出层**：1 个神经元，Sigmoid 激活（输出一个概率值，适合二分类）

总参数量：$3 \times 4 + 4 \ (\text{偏置}) + 4 \times 4 + 4 \ (\text{偏置}) + 4 \times 1 + 1 \ (\text{偏置}) = 41$ 个参数。

**为什么 `X` 长这样、又能对上这个网？** 这里的数据是**假的**：`np.random.randn(3, 32)` 抽 32 个样本、每个 3 维、服从标准正态。没有标签、没有真实任务，只为把形状走通。真正要对齐的只有两件事：

1. **特征数 = 输入层宽度。** `layer_dims[0] = 3`，所以 `X` 必须有 3 行（本教程约定是「特征 × 样本」）。第一层 $W^{[1]}$ 形状是 `(4, 3)`：列数 3 必须等于每个样本的特征数，否则 `W @ X` 乘不上。
2. **样本数是 batch，不改网络结构。** 32 只出现在 `X` 的第二维，权重形状里没有 32。换成 8 个或 128 个样本，`W` 还是 `(4,3)`，只是一次处理几列。

`randn` 而不是随便填 0～100，是因为 He 初始化假定输入尺度大约是方差 1；真实数据通常要先标准化，才和这种初始化匹配。

输出层 1 个神经元 + Sigmoid，在**真实任务**里表示「二分类概率」。本 demo **没有** `y`，所以还谈不上分类对不对，只是前向算出 32 个 $(0,1)$ 里的数。

**什么样的数据配什么样的网（记输入/输出两端）：**

| 数据 | 每个样本长什么样 | 常见网络 | 输入层 / 输出层 |
|------|------------------|----------|-----------------|
| 表格（房价、鸢尾花） | 一维特征向量 | MLP | 输入 = 特征数；输出 = 1（回归/二分类）或类别数 |
| 图像 | $H\times W\times C$ 网格 | CNN（或 ViT） | 不要先压成 MLP；输出 = 类别数 |
| 句子 / 时间序列 | 一串 token 或时间步 | RNN / Transformer | 输入是序列；输出可以是一类、一个词，或等长序列 |
| 图（分子、路网） | 节点 + 边 | GNN | 输入在节点上，输出可以是节点或整图 |

中间几层（这里的 4、4）是容量旋钮：宽一点更能拟合，也更容易过拟合。它们**不**由数据形状唯一决定；由数据形状钉死的是**第一层的扇入**和**最后一层的扇出**。

注意代码中的形状约定：输入 $X$ 是 `(n_features, n_samples)` 即 $(3, 32)$，而不是常见的 `(n_samples, n_features)`。这种约定在数学上等价，只是矩阵乘法的顺序不同。反向传播的推导通常使用这个约定。若你用 PyTorch 常见的 `(batch, features)`，对应乘法是 `X @ W.T`，不要和本课的 `W @ a` 混用。

### 第8步：张量形状总览

`print_tensor_shape_table()` 将前向传播中所有张量的形状以表格形式打印出来：

```
步骤       名称         形状                   说明
--------------------------------------------------------------------------------
输入       X (a[0])     (3, 32)               输入数据（特征数 × 样本数）
权重       W[1]         (4, 3)                第 1 层权重矩阵
偏置       b[1]         (4, 1)                第 1 层偏置向量
第 1 层    z[1]         (4, 32)               线性变换输出（W·a_prev + b）
第 1 层    a[1]         (4, 32)               激活函数输出（下一层输入）
...
```

这个表格是理解网络数据流的关键参考。每一层的输出维度由 $W^{[l]}$ 的行数决定，输入维度由 $W^{[l]}$ 的列数决定。循着形状追踪，可以验证整个网络结构的一致性。

## 关键概念速查表

| 概念 | 数学形式 | 代码位置 | 关键说明 |
|------|---------|---------|---------|
| 线性变换 | $z = Wa + b$ | `forward_pass()` | `W @ a + b`，广播加法 |
| ReLU | $\max(0, z)$ | `relu()` | 正区间导数=1，解决梯度消失 |
| Sigmoid | $1/(1+e^{-z})$ | `sigmoid()` | 输出范围 (0,1)，用于二分类输出层 |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | `tanh()` | 输出零中心 (-1,1)，用于 RNN |
| GELU | $z \cdot \Phi(z)$ | `gelu_approximate()` | Transformer 标配，概率性通过 |
| He 初始化 | $W \sim N(0, \sqrt{2/n_{\text{in}}})$ | `initialize_parameters()` | 配合 ReLU 使用，保持方差稳定 |
| 中间值缓存 | `{z, a_prev, a}` | `forward_pass()` cache | 反向传播的"燃料" |
| 计算图 | DAG 节点=操作，边=数据 | 概念层 | 前向/反向传播的基础抽象 |
| `verbose` | — | `forward_pass(..., verbose=)` | 只控制是否打印形状，不改变计算结果 |


## 源码位置

clone 后打开（相对仓库根目录）：

- `docs/nn-decision/dl/forward-graph/code/demo.py`（前向传播主线）
- `docs/nn-decision/dl/forward-graph/code/plot_demo.py`（同目录配图）
